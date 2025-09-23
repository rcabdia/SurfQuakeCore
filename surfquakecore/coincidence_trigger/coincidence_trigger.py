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
import re

class CoincidenceTrigger:
    def __init__(self, projects: list, coincidence_config: Union[str, CoincidenceConfig], picking_file=None,
                 output_folder=None, plot=None):

        self.projects = projects
        self.picking_file = picking_file
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
            self.hos_order = 4 #kurtosis
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
        # New header with semicolons and split time columns
        header = "date;hour;num_traces;coincidence_sum;duration\n"

        output_file = os.path.join(self.output_folder, "coincidence_sum.txt")

        # Ensure header exists exactly once
        if not os.path.exists(output_file):
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(header)
        else:
            # If file exists but is empty, write header; otherwise don't duplicate it
            if os.path.getsize(output_file) == 0:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(header)
            else:
                # Optional: only write the new header if it's already the expected one
                # (avoid adding another header when appending to an old file format)
                with open(output_file, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                # If it's already our header, fine; if it's something else, we just append rows.
                # (No extra header inserted.)
                _ = first_line  # just to clarify intent

        for item in events:
            if item["time"] in events_times_cluster:
                # item["time"] is UTCDateTime (or similar); get python datetime
                t = item["time"].datetime

                date_str = t.strftime("%Y-%m-%d")
                # one decimal of second, no rounding (floor to 0.1 s)
                hour_str = t.strftime("%H:%M:%S.") + str(t.microsecond // 100000)

                # prefer explicit num_traces if present, else fallback to len(trace_ids)
                num_traces = item.get("num_traces", len(item.get("trace_ids", [])))
                duration_str = f"{item['duration']:.1f}"  # <<< one decimal

                row = (
                    f"{date_str};"
                    f"{hour_str};"
                    f"{num_traces};"
                    f"{item['coincidence_sum']};"
                    f"{duration_str}\n"
                )

                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(row)

    def thresholding_sta_lta(self, files_list):

        traces = []

        for file in files_list:
            try:
                tr = read(file)[0]
                tr = self.fill_gaps(tr)
                if tr is not None:
                    traces.append(tr)
            except:
                pass

        st = Stream(traces)
        st.merge(method=1, fill_value='interpolate', interpolation_samples=-1)
        print("Ready to run coincidence Trigger at: ")
        print(st.__str__(extended=True))
        st.detrend(type='simple')
        st.taper(max_percentage=0.05)
        st.filter(type="bandpass", freqmin=self.fmin, freqmax=self.fmax)
        st.detrend(type='simple')
        st.taper(max_percentage=0.05)

        print("Computing SNR")
        st_SNR, _ = CF_SNR.compute_sta_lta_cfs(st)

        events, events_times_cluster = self.run_coincidence(st_SNR)

        if self.plot and self.output_folder:
            PlotCoincidence.plot_stream_and_cf_simple(st, st_SNR, type="SNR", events=events_times_cluster,
                                                      savepath=self.output_folder, show=False)

        return events, events_times_cluster

    def thresholding_cwt_kurt(self, files_list):

        traces = []

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

        print("Computing Kurtosis")
        cf_kurt = CFKurtosis(st, self.CF_decay_win, self.hos_order,
                             self.coincidence_config.cluster_configuration.fmin,
                             self.coincidence_config.cluster_configuration.fmax)

        st_cf = cf_kurt.run_kurtosis()

        events, events_times_cluster = self.run_coincidence(st_cf)

        if self.plot:
            print("Plotting CFs")
            PlotCoincidence.plot_stream_and_cf_simple(st, st_cf, type="Kurtosis", events=events_times_cluster,
                                                      savepath=self.output_folder, show=False)

        return events, events_times_cluster


    def run_coincidence(self, st_cf):

        events = coincidence_trigger(trigger_type=None, thr_on=self.the_on, thr_off=self.the_off,
                                     trigger_off_extension=self.centroid_radio,
                                     thr_coincidence_sum=self.coincidence, stream=st_cf,
                                     similarity_threshold=0.8, details=True)

        triggers = self._extract_event_info(events)

        if len(triggers) > 0:
            events_times_cluster, _ = MseedUtil.cluster_events(triggers, eps=self.centroid_radio)
            print("Number of events detected", len(events_times_cluster))
            self._coincidence_info(events, events_times_cluster)

        return events, events_times_cluster

    def _process_coincidence_trigger(self, filtered_files):

        """Process a single day's data and return events or 'empty'."""
        if self.method_preferred == "SNR":
            return self.thresholding_sta_lta(filtered_files)
        elif self.method_preferred == "Kurtosis":
            return self.thresholding_cwt_kurt(filtered_files)
        else:
            print("Method preferred available are: Kurtosis or SNR, please select one of this")

    def optimized_project_processing(self):

        final_filtered_results = []
        results = []
        details = []

        for i, project in enumerate(self.projects):
            print("Working in project", i)
            try:
                results.append(self._process_coincidence_trigger(project.data_files))
            except:
                print("An exception raise at project ", i)

        # Join the output of all days
        if len(results) > 0:
            for item in results:
                if item[0] is not None and item[1] is not None:
                    details.extend(item[0])
                    final_filtered_results.extend(item[1])
            coincidence_file = os.path.join(self.output_folder, "coincidence_pick.txt")
            if len(final_filtered_results) > 0 and self.picking_file is not None and self.output_folder is not None:
                self.separate_picks_by_events(self.picking_file, coincidence_file, centroids=final_filtered_results,
                                              min_picks=self.coincidence)
        else:
            print("No detected events")

        return final_filtered_results

    def separate_picks_by_events(self, input_file, output_file, centroids, min_picks: int = 5):
        """
        Separates picks by events based on centroids and their radius (self.centroid_radio, seconds).
        Appends results and only writes events with at least `min_picks` picks.
        """
        # --- load input ---
        df = pd.read_csv(input_file, delimiter=r"\s+")

        # Build FullTime from Date, Hour_min (HHMM), Seconds (can be fractional)
        def to_utcdt(row):
            hhmm = int(row["Hour_min"])
            hh, mm = divmod(hhmm, 100)
            sec = float(row["Seconds"])
            return UTCDateTime(f"{row['Date']}T{hh:02d}:{mm:02d}:{sec:06.3f}")

        df["FullTime"] = df.apply(to_utcdt, axis=1)
        df["Event"] = None

        radius = float(self.centroid_radio)  # seconds

        # Assign picks to the nearest centroid within radius/2 (robust to overlaps)
        # (If you prefer the original overwrite-by-loop behavior, replace this block with your loop)
        import numpy as np
        if centroids:
            # compute time diffs matrix [n_picks x n_centroids]
            diffs = np.column_stack([np.abs(df["FullTime"].values - c) for c in centroids])
            nearest_idx = diffs.argmin(axis=1)
            nearest_dt = diffs[np.arange(len(df)), nearest_idx]
            # assign only if within radius/2
            mask = nearest_dt <= (radius / 2.0)
            df.loc[mask, "Event"] = ["Event_tmp_" + str(i + 1) for i in nearest_idx[mask]]

        # Collect only groups meeting the threshold
        qualifying = []
        for event, group in df.groupby("Event", dropna=True):
            if len(group) >= min_picks:
                # drop helper cols and (optionally) sort for readability
                g = group.drop(columns=["Event", "FullTime"]).copy()
                if {"Date", "Hour_min", "Seconds"}.issubset(g.columns):
                    g = g.sort_values(["Date", "Hour_min", "Seconds"])
                qualifying.append(g)

        if not qualifying:
            return 0  # nothing to write

        # Determine append mode and continuous numbering
        append = os.path.exists(output_file) and os.path.getsize(output_file) > 0
        start_idx = 1
        if append:
            with open(output_file, "r", encoding="utf-8") as f:
                text = f.read()
            found = re.findall(r"^Event_(\d+)\s*$", text, flags=re.MULTILINE)
            if found:
                start_idx = int(found[-1]) + 1

        # Write
        mode = "a" if append else "w"
        written = 0
        with open(output_file, mode, encoding="utf-8") as f:
            if append:
                f.write("\n")  # spacer only when we will actually write something

            for i, group in enumerate(qualifying, start=0):
                event_name = f"Event_{start_idx + i}"
                f.write(f"{event_name}\n")
                group.to_csv(f, sep="\t", index=False)
                f.write("\n")
                written += 1

        return written  # number of events written


class PlotCoincidence:
    import matplotlib as mplt
    mplt.use("Qt5Agg")

    @staticmethod
    def plot_stream_and_cf_simple(
        stream: Stream,
        cf_stream: Stream, type="SNR",
        events: list | None = None,          # list of UTCDateTime
        savepath: str | None = None,
        show: bool = True
    ):
        # --- one-decimal seconds formatter ---
        def _hms_tenths(x, pos=None):
            d = mdt.num2date(x) + timedelta(milliseconds=50)  # round to nearest 0.1 s
            return f"{d:%H:%M:%S}.{d.microsecond // 100000}"

        formatter_pow = ScalarFormatter(useMathText=True)
        formatter_pow.set_powerlimits((0, 0))

        n = len(stream)
        max_traces = min(8, 2 * len(stream))
        fig, axes = plt.subplots(
            len(stream), 1, figsize=(12, max_traces), sharex=True,
            gridspec_kw={'hspace': 0.05}
        )
        if n == 1:
            axes = [axes]

        # Pre-convert event times to matplotlib numbers (and sort)
        event_x = []
        if events:
            try:
                event_x = sorted(mdt.date2num(ev.datetime) for ev in events)
            except Exception:
                # If items are already datetimes or floats, try converting generically
                event_x = sorted(mdt.date2num(getattr(ev, "datetime", ev)) for ev in events)
        name = "Characteristic functions:"+" "+type

        ax_idx = 0
        for ax, tr_raw, tr_cf in zip(axes, stream[:n], cf_stream[:n]):
            if ax_idx==0:
                ax.set_title(name, fontsize=12, fontstyle='italic')
            starttime = tr_raw.stats.starttime
            t = tr_raw.times("matplotlib")

            # raw
            ln1, = ax.plot(t, tr_raw.data, linewidth=1.0, label=f"{tr_raw.id}")
            #ax.set_ylabel("Raw")

            # CF
            ax2 = ax.twinx()
            ln2, = ax2.plot(
                t[:len(tr_cf.data)], tr_cf.data,
                linestyle="-", linewidth=0.75, alpha=0.75, color="orange")
            #ax2.set_ylabel("CF")

            # event vertical lines (only those within this subplot's time window)
            if event_x:
                tmin, tmax = t[0], t[-1]
                first_label_done = False
                for x in event_x:
                    if tmin <= x <= tmax:
                        label = "Event" if (ax_idx == 0 and not first_label_done) else None
                        ax.axvline(x=x, linestyle=":", linewidth=1.2, alpha=0.9, color="tab:red", label=label)
                        first_label_done = True

            # annotation
            date_str = starttime.strftime("%Y-%m-%d")
            textstr = f"JD {starttime.julday} / {starttime.year}\n{date_str}"
            ax.text(0.01, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.5))

            # x-axis formatting
            ax.xaxis_date()
            ax.xaxis.set_major_locator(mdt.AutoDateLocator())
            ax.xaxis.set_major_formatter(FuncFormatter(_hms_tenths))

            # legend
            lines = [ln1, ln2]
            labels = [l.get_label() for l in lines if l.get_label() != "_nolegend_"]
            ax.legend(lines, labels, loc="upper right")

            # tick visibility for inner plots
            if ax_idx < len(axes) - 1:
                # raw axis
                ax.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
                ax.spines['bottom'].set_visible(False)
                # CF twin axis
                ax2.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
                ax2.spines['bottom'].set_visible(False)
            else:
                ax.tick_params(axis='x', which='both', labelbottom=True, bottom=True)
                ax.spines['bottom'].set_visible(True)
                ax2.tick_params(axis='x', which='both', labelbottom=True, bottom=True)
                ax2.spines['bottom'].set_visible(True)

            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax.yaxis.set_major_formatter(formatter_pow)
            ax.yaxis.get_offset_text().set_visible(True)
            ax_idx += 1


        fig.tight_layout()

        if savepath:
            name = f"{starttime}_CF.pdf"
            plot_path = os.path.join(savepath, name)
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")

        # Ensure the figure is shown when not saving
        if show:
            plt.show()
        else:
            plt.close(fig)