from typing import Union
from copy import deepcopy
from obspy import UTCDateTime, read, Trace, Stream
import numpy as np
import tensorflow as tf
from datetime import datetime
from surfquakecore.project.surf_project import SurfProject
from surfquakecore.utils.obspy_utils import ObspyUtil


# def init_worker():
#     """Initialize TensorFlow settings for each worker process."""
#     tf.compat.v1.enable_eager_execution()

class Polarity:
    def __init__(self, project: Union[SurfProject, str], model_path, arrivals_path, threshold, output_path):

        self.model = tf.keras.models.load_model(model_path, compile=False)

        if isinstance(project, str):
            self.project = SurfProject.load_project(project)
        else:
            self.project = project

        self.model_path = model_path
        self.arrivals_path = arrivals_path
        self.threshold = threshold
        self.output_path = output_path
        self.import_picks()

    def __str__(self):
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        return model_summary

    def import_picks(self):
        self.pick_times_imported = ObspyUtil.get_NLL_phase_picks(input_file=self.arrivals_path)
        self.pick_times_imported_modify = deepcopy(self.pick_times_imported)

    def optimized_project_processing_pol(self):
        """
        Predict the polarity of the P picks of the whole project, one day at a time,
        and write the edited pick file.
        """
        info = self.project.get_project_basic_info()
        print(f"[INFO] Project from {info['Start']} to {info['End']}")

        start_time = UTCDateTime(datetime.strptime(info['Start'], '%Y-%m-%d %H:%M:%S'))
        end_time = UTCDateTime(datetime.strptime(info['End'], '%Y-%m-%d %H:%M:%S'))

        # One range per day; at least one, also for projects shorter than a day
        n_days = max(1, int(np.ceil((end_time - start_time) / 86400)))
        daily_ranges = [(start_time + i * 86400, start_time + (i + 1) * 86400) for i in range(n_days)]

        # Sequential on purpose: the TensorFlow model is loaded once in this process
        for start, end in daily_ranges:
            self.process_coincidence_pol((self.project, start, end))

        self.write_output()

    def _filter_trace_by_picks(self, tr: Trace):
        """
        P picks of the pick file that belong to this trace: same station and channel,
        vertical component, and pick time inside the trace.

        Returns:
        - filtered_picks (list): the matching picks
        - items (list): their positions in self.pick_times_imported[trace_id]
        """
        trace_id = f"{tr.stats.station}.{tr.stats.channel}"
        trace_start = tr.stats.starttime
        trace_end = tr.stats.endtime

        filtered_picks = []
        items = []

        # First-motion polarity is only defined on the vertical component
        if not tr.stats.channel.endswith("Z") or trace_id not in self.pick_times_imported:
            return filtered_picks, items

        for item, pick in enumerate(self.pick_times_imported[trace_id]):
            phase_name, pick_time = pick[0], pick[1]
            if phase_name == "P" and trace_start <= pick_time <= trace_end:
                filtered_picks.append(pick)
                items.append(item)

        return filtered_picks, items

    def process_coincidence_pol(self, args):
        """Process a single day's data and return events or 'empty'."""
        sp, start, end = args
        # Filter files for the given time range
        filtered_files = sp.filter_time(starttime=start, endtime=end, tol=3600, use_full=True)
        if filtered_files:
            self.run_predict_polarity_day(filtered_files)
        else:
            return "empty"

    def run_predict_polarity_day(self, filtered_files):
        """
        Predict the polarity of the P picks of the vertical traces in filtered_files.
        All the picks of one trace are sent to PolarCAP in a single batch.
        """
        half = 32  # PolarCAP input: 64 samples at 100 Hz (0.64 s) centred on the pick
        labels = ['Negative', 'Positive']

        for file in filtered_files:
            try:
                st = read(file)
            except Exception as e:
                print(f"[WARNING] Cannot read {file}: {e}")
                continue

            if len(st) == 0 or not st[0].stats.channel.endswith("Z"):
                continue  # polarities only on vertical channels

            tr = self.fill_gaps(st)
            if tr is None:
                print(f"[WARNING] Too many gaps, skipped: {file}")
                continue
            if tr.stats.sampling_rate != 100:
                tr.resample(100)

            filtered_picks, items = self._filter_trace_by_picks(tr)
            if not filtered_picks:
                continue
            trace_id = f"{tr.stats.station}.{tr.stats.channel}"
            data = tr.data

            windows, valid_items = [], []
            for pick, item in zip(filtered_picks, items):
                centre = int((pick[1] - tr.stats.starttime) * tr.stats.sampling_rate)
                start, end = centre - half, centre + half
                if start < 0 or end > len(data):
                    print(f"[WARNING] {trace_id}: pick at {pick[1]} too close to the trace edge, skipped")
                    continue
                window = data[start:end].astype(float)
                window -= window.mean()
                if not np.any(window):
                    continue  # flat window (e.g. filled gap): no polarity
                windows.append(window)
                valid_items.append(item)

            if not windows:
                continue

            X = self.norm(np.array(windows).reshape(len(windows), 2 * half, 1))
            try:
                y_pred = self.model.predict(X, verbose=0)
            except Exception as e:
                print(f"[WARNING] {trace_id}: polarity prediction failed: {e}")
                continue

            pol_pred = np.argmax(y_pred[1], axis=1)
            pred_prob = np.max(y_pred[1], axis=1)
            for item, pol, prob in zip(valid_items, pol_pred, pred_prob):
                self._edit_pick_input(trace_id, item, [(labels[pol], prob)])

    def fill_gaps(self, st, tol=5):
        """
        Merge the traces of a file into one. Gaps are filled by interpolation if they
        add up to at most tol % of the data; otherwise None is returned.
        """
        if isinstance(st, Trace):
            st = Stream(traces=[st])
        if len(st) == 1:
            return st[0]

        tol_seconds = (tol / 100) * sum(tr.stats.npts for tr in st) * st[0].stats.delta
        gaps = st.get_gaps()
        if gaps and not self._check_gaps(gaps, tol_seconds):
            st.print_gaps()
            return None
        st.merge(fill_value="interpolate", interpolation_samples=-1)
        return st[0]

    def _edit_pick_input(self, trace_id, item, predictions):

        if predictions[0][0] == "Positive" and predictions[0][1] >= self.threshold:
            self.pick_times_imported_modify[trace_id][item][3] = "U"
        elif predictions[0][0] == "Negative" and predictions[0][1] >= self.threshold:
            self.pick_times_imported_modify[trace_id][item][3] = "D"

    def _plot(self, data, data_full, label):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.plot(data_full, label=label)
        # ax.plot(data_full)
        ax.legend()
        plt.show()

    def norm(self, X):
        maxi = np.max(abs(X), axis=1)
        X_ret = X.copy()
        for i in range(X_ret.shape[0]):
            X_ret[i] = X_ret[i] / maxi[i]
        return X_ret

    def fill_gaps(self, tr, tol=5):

        tol_seconds_percentage = int((tol / 100) * len(tr.data)) * tr.stats.delta
        st = Stream(traces=tr)
        gaps = st.get_gaps()

        if len(gaps) > 0 and self._check_gaps(gaps, tol_seconds_percentage):
            st.print_gaps()
            st.merge(fill_value="interpolate", interpolation_samples=-1)
            return st[0]

        elif len(gaps) > 0 and not self._check_gaps(gaps, tol_seconds_percentage):
            st.print_gaps()
            return None
        elif len(gaps) == 0 and self._check_gaps(gaps, tol_seconds_percentage):
            return tr
        else:
            return tr

    def _check_gaps(self, gaps, tol):
        time_gaps = []
        for i in gaps:
            time_gaps.append(i[6])

        sum_total = sum(time_gaps)

        return sum_total <= tol

    def write_output(self):
        with open(self.output_path, 'w') as file:

            # Write the header line
            header = ("Station_name\tInstrument\tComponent\tP_phase_onset\tP_phase_descriptor\t"
                      "First_Motion\tDate\tHour_min\tSeconds\tErr\tErrMag\tCoda_duration\tAmplitude\tPeriod\n")

            file.write(header)

            for key in self.pick_times_imported_modify.keys():
                id = key.split(".")
                contents = self.pick_times_imported_modify[key]
                station = id[0].ljust(6)  # Station name, left-justified, 6 chars
                instrument = "?".ljust(4)  # Placeholder for Instrument
                component = id[1].ljust(4)  # Placeholder for Component

                for content in contents:
                    time = UTCDateTime(
                        content[1])  # Convert starttime and endtime to pandas-compatible datetime objects
                    date_concatanate = time.strftime("%Y%m%d")
                    p_phase_onset = "?"  # Placeholder for P phase onset
                    phase_descriptor = content[0].ljust(6)  # Phase descriptor (e.g., P, S)
                    first_motion = content[3]  # Placeholder for First Motion
                    date = f"{date_concatanate}"  # Date in yyyymmdd format
                    hour_min = f"{time.hour:02}{time.minute:02}"  # hhmm
                    seconds = f"{time.second + time.microsecond / 1e6:07.4f}"
                    err = "GAU"  # Error type (GAU)

                    err_mag = f"{content[5]:.2e}"  # Error magnitude in seconds
                    coda_duration = content[6]  # Placeholder for Coda duration
                    amplitude = f"{content[7]:.2e}"  # Amplitude
                    period = content[8]  # Placeholder for Period
                    # Construct the line
                    line = (
                        f"{station} {instrument} {component} {p_phase_onset} {phase_descriptor} {first_motion} "
                        f"{date} {hour_min} {seconds} {err} {err_mag} {coda_duration} {amplitude} {period}\n"
                    )
                    file.write(line)

                    # Add a blank line at the end for NLLoc format compliance
            file.write("\n")


if __name__ == "__main__":
    project_path = "/Users/roberto/Documents/ISP/earthquake_test"

    arrivals_path = ("/Users/roberto/Documents/ISP/isp/earthquakeAnalysis/loc_structure/"
                     "output_original_example.txt")
    model_path = '/Users/roberto/Documents/ISP/isp/PolarCap/PolarCAP.h5'
    output = "./"
    pol = Polarity(project=project_path, model_path=model_path, arrivals_path=arrivals_path, threshold=0.9,
                   output_path="/Users/roberto/Documents/ISP/isp/earthquakeAnalysis/loc_structure/test_focmec.txt")
    print(pol)
    pol.optimized_project_processing_pol()
