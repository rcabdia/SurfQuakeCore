#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
coincidence_trigger
"""

from __future__ import annotations
import os
from datetime import timedelta
from typing import Union
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from obspy import read
from obspy.signal.trigger import coincidence_trigger
from surfquakecore.coincidence_trigger.cf_SNR import CF_SNR
from surfquakecore.coincidence_trigger.cf_kurtosis import CFKurtosis
from surfquakecore.coincidence_trigger.coincidence_parse import load_concidence_configuration
from surfquakecore.coincidence_trigger.structures import CoincidenceConfig
from surfquakecore.utils.obspy_utils import MseedUtil
import pandas as pd
from obspy import Stream, UTCDateTime
import matplotlib.dates as mdt
import matplotlib.pyplot as plt

class CoincidenceTrigger:
    def __init__(self, projects: list, coincidence_config: Union[str, CoincidenceConfig], output_folder=None, plot=None):

        self.projects = projects
        self.plot = plot
        self.output_folder = output_folder
        self.__get_coincidence_config(coincidence_config)

        self.the_on: float = self.coincidence_config.cluster_configuration.threshold_on
        self.the_off: float = self.coincidence_config.cluster_configuration.threshold_off
        self.fmin: float = self.coincidence_config.cluster_configuration.fmin
        self.fmax: float = self.coincidence_config.cluster_configuration.fmax
        self.centroid_radio = self.coincidence_config.cluster_configuration.centroid_radio
        self.coincidence = self.coincidence_config.cluster_configuration.coincidence
        self.method_preferred = self.coincidence_config.cluster_configuration.method_preferred


        if self.method_preferred == "SNR":

            self.method_snr: str = self.coincidence_config.sta_lta_configuration.method
            self.sta = self.coincidence_config.sta_lta_configuration.sta_win
            self.lta = self.coincidence_config.sta_lta_configuration.lta_win

        elif self.method_preferred == "Kurtosis":

            self.CF_decay_win = self.coincidence_config.kurtosis_configuration.CF_decay_win
            self.hos_order = self.coincidence_config.kurtosis_configuration.hos_order
        else:
            print("Available preferred methods are: SNR or Kurtosis")

    def __get_coincidence_config(self, coincidence_config):
        if isinstance(coincidence_config, str) and os.path.isfile(coincidence_config):
            self.coincidence_config: CoincidenceConfig = load_concidence_configuration(coincidence_config)
        elif isinstance(coincidence_config, CoincidenceConfig):
            self.coincidence_config = coincidence_config
        else:
            raise ValueError(f"coincidence_config {coincidence_config} is not valid. It must be either a "
                             f" valid coincidence_config.ini file or a CoincidenceConfig instance.")

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

    def _extract_event_info(self, trigger):
        events_times = []
        for k in range(len(trigger)):
            detection = trigger[k]
            for key in detection:

                if key == 'time':
                    time = detection[key]
                    events_times.append(time)
        return events_times

    def _coincidence_info(self, events, events_times_cluster):
        header = "time\tnum_traces\tcoincidence_sum\tduration\n"

        output_file = os.path.join(self.output_folder, "coincidence_sum.txt")
        # Create file if not exists, and add header
        if not os.path.exists(output_file):
            with open(output_file, "w") as f:
                f.write(header)

        for item in events:
            if item["time"] in events_times_cluster:
                # Format datetime as "YYYY-MM-DD HH:MM:SS.s"
                t = item["time"].datetime
                time_str = t.strftime("%Y-%m-%d %H:%M:%S.") + str(int(t.microsecond / 100000))

                # Format the row
                row = f"{time_str}\t{len(item['trace_ids'])}\t{item['coincidence_sum']}\t{item['duration']}\n"

                # Append row to file
                with open(output_file, "a") as f:
                    f.write(row)

    def thresholding_sta_lta(self, files_list):

        traces = []
        events_times_cluster = None

        for file in files_list:
            try:
                tr = read(file)[0]
                tr = self.fill_gaps(tr)
                if tr is not None:
                    traces.append(tr)
            except:
                pass
        st = Stream(traces)
        st.merge()
        print("Ready to run coincidence Trigger at: ")
        print(st.__str__(extended=True))
        st.detrend(type='simple')
        st.taper(max_percentage=0.05)
        st.filter(type="bandpass", freqmin=self.fmin, freqmax=self.fmax)
        st.detrend(type='simple')
        st.taper(max_percentage=0.05)
        st_SNR, _ = CF_SNR.compute_sta_lta_cfs(st)

        events = coincidence_trigger(trigger_type=None, thr_on=self.the_on, thr_off=self.the_off,
                                     trigger_off_extension=self.centroid_radio,
                                     thr_coincidence_sum=self.coincidence, stream=st_SNR,
                                     similarity_threshold=0.8, details=True)

        triggers = self._extract_event_info(events)

        if len(triggers) > 0:
            events_times_cluster, _ = MseedUtil.cluster_events(triggers, eps=self.centroid_radio)
            print("Number of events detected", len(events_times_cluster))
            self._coincidence_info(events, events_times_cluster)

        if self.plot:
            PlotCoincidence.plot_stream_and_cf_simple(st, st_SNR, savepath=self.output_folder)

        return events, events_times_cluster

    def thresholding_cwt_kurt(self, files_list):

        traces = []
        events = []
        events_times_cluster = None

        for file in files_list:
            try:
                tr = read(file)[0]
                tr = self.fill_gaps(tr)
                if tr is not None:
                    traces.append(tr)
            except:
                print("File not included: ", file)

        st = Stream(traces)
        st.merge(method=1, fill_value='interpolate', interpolation_samples=-1)

        print("Ready to run coincidence Trigger at: ")
        print(st.__str__(extended=True))
        if st:
            cf_kurt = CFKurtosis(st, self.coincidence_config.kurtosis_configuration,
                                 self.coincidence_config.cluster_configuration.fmin,
                                 self.coincidence_config.cluster_configuration.fmax)

            st_kurt = cf_kurt.run_kurtosis()

            events = coincidence_trigger(trigger_type=None, thr_on=self.the_on, thr_off=self.the_off,
                                         trigger_off_extension=self.centroid_radio,
                                         thr_coincidence_sum=self.coincidence, stream=st_kurt,
                                         similarity_threshold=0.8, details=True)

            triggers = self._extract_event_info(events)

            if len(triggers) > 0:
                events_times_cluster, _ = MseedUtil.cluster_events(triggers, eps=self.centroid_radio)
                print("Number of events detected", len(events_times_cluster))
                self._coincidence_info(events, events_times_cluster)

            if self.plot:
                PlotCoincidence.plot_stream_and_cf_simple(st, st_kurt, savepath=self.output_folder)

        return events, events_times_cluster

    def _process_coincidence_trigger(self, filtered_files):

        """Process a single day's data and return events or 'empty'."""
        if self.method_preferred == "SNR":
            return self.thresholding_sta_lta(filtered_files)
        elif self.method_preferred == "Kurtosis":
            return self.thresholding_cwt_kurt(filtered_files)
        else:
            print("Method preferred available are: Kurtosis or SNR, please select one of this")

    def optimized_project_processing(self, input_file=None, output_file=None):

        final_filtered_results = []
        results = []
        details = []

        for project in self.projects:
            data_files = project.data_files
            results.append(self._process_coincidence_trigger(data_files))

        # Join the output of all days
        for item in results:
            if item[0] is not None and item[1] is not None:
                details.extend(item[0])
                final_filtered_results.extend(item[1])

        if len(final_filtered_results) > 0 and input_file is not None and output_file is not None:
            self.separate_picks_by_events(input_file, output_file, centroids=final_filtered_results)

        return final_filtered_results

    def separate_picks_by_events(self, input_file, output_file, centroids):
        """
        Separates picks by events based on centroids and their radius.

        :param input_file: Path to the input file with pick data.
        :param output_file: Path to the output file for separated picks.
        :param centroids: List of UTCDateTime objects representing centroids.
        :param radius: Radius in seconds for each centroid.
        """
        # Load the input data
        df = pd.read_csv(input_file, delimiter='\s+')
        # Ensure columns are properly typed
        # df['Date'] = df['Date'].astype(str)  # Date as string for slicing
        # df['Hourmin'] = df['Hourmin'].astype(int)  # Hourmin as integer
        # df['Seconds'] = df['Seconds'].astype(float)  # Seconds as float for fractional handling
        # Parse date and time columns into a single datetime object
        df['FullTime'] = df.apply(lambda row: UTCDateTime(
            f"{row['Date']}T{str(row['Hour_min']).zfill(4)}:{row['Seconds']}"
        ), axis=1)

        # Create a new column for event assignment
        df['Event'] = None

        # Assign picks to the closest centroid within the radius
        for i, centroid in enumerate(centroids):
            within_radius = df['FullTime'].apply(lambda t: abs((t - centroid)) <= self.centroid_radio / 2)
            df.loc[within_radius, 'Event'] = f"Event_{i + 1}"

        # Write grouped picks into the output file
        with open(output_file, 'w') as f:
            for event, group in df.groupby('Event'):
                if pd.isna(event):
                    continue  # Skip unassigned picks
                f.write(f"{event}\n")
                group.drop(columns=['Event', 'FullTime']).to_csv(f, sep='\t', index=False)
                f.write("\n")


class PlotCoincidence:
    import matplotlib as mplt
    mplt.rcParams["axes.xmargin"] = 0
    mplt.use("TkAgg")

    @staticmethod
    def plot_stream_and_cf_simple(stream: Stream, cf_stream: Stream, savepath: str | None = None):
        # --- one-decimal seconds formatter ---
        def _hms_tenths(x, pos=None):
            d = mdt.num2date(x) + timedelta(milliseconds=50)  # round to nearest 0.1 s
            return f"{d:%H:%M:%S}.{d.microsecond // 100000}"

        formatter_pow = ScalarFormatter(useMathText=True)
        formatter_pow.set_powerlimits((0, 0))  # Force scientific notation
        n = len(stream)
        max_traces = min(8, 2 * len(stream))
        fig, axes = plt.subplots(len(stream), 1, figsize=(12, max_traces), sharex=True,
                                 gridspec_kw={'hspace': 0.05})

        if n == 1:
            axes = [axes]
        ax_idx = 0
        for ax, tr_raw, tr_cf in zip(axes, stream[:n], cf_stream[:n]):
            starttime = tr_raw.stats.starttime
            t = tr_raw.times("matplotlib")

            ln1, = ax.plot(t, tr_raw.data, linewidth=1.0, label=f"{tr_raw.id}")
            ax.set_ylabel("Raw")

            ax2 = ax.twinx()
            ln2, = ax2.plot(t[:len(tr_cf.data)], tr_cf.data, color="orange", linestyle="-", linewidth=0.75,
                            alpha=0.75, label="CF")
            ax2.set_ylabel("CF")

            date_str = starttime.strftime("%Y-%m-%d")
            textstr = f"JD {starttime.julday} / {starttime.year}\n{date_str}"
            ax.text(0.01, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.5))

            ax.xaxis_date()
            ax.xaxis.set_major_locator(mdt.AutoDateLocator())
            ax.xaxis.set_major_formatter(FuncFormatter(_hms_tenths))  # <<< 0.1s precision here

            lines = [ln1, ln2]
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc="upper right")

            if ax_idx < len(axes) - 1:
                ax.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
                ax.spines['bottom'].set_visible(False)
            else:
                ax.tick_params(axis='x', which='both', labelbottom=True, bottom=True)
                ax.spines['bottom'].set_visible(True)

            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax.yaxis.set_major_formatter(formatter_pow)
            ax.yaxis.get_offset_text().set_visible(True)
            ax_idx += 1

        fig.tight_layout()
        if savepath:
            name = str(starttime)+"_"+"CF"+".pdf"
            plot_path = os.path.join(savepath, name)
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")