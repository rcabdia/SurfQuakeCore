#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
coincidence_trigger
"""

import os
from typing import Union
from obspy import read, Stream
from obspy.signal.trigger import coincidence_trigger
from surfquakecore.coincidence_trigger.cf_kurtosis import CFKurtosis
from surfquakecore.coincidence_trigger.coincidence_parse import load_concidence_configuration
from surfquakecore.coincidence_trigger.structures import CoincidenceConfig
from pprint import pprint
from surfquakecore.utils.obspy_utils import MseedUtil

class CoincidenceTrigger:
    def __init__(self, projects: list, coincidence_config: Union[str, CoincidenceConfig]):

        self.projects = projects
        self.__get_coincidence_config(coincidence_config)

        self.the_on: float = self.coincidence_config.cluster_configuration.threshold_on
        self.the_off: float = self.coincidence_config.cluster_configuration.threshold_off
        self.fmin: float =  self.coincidence_config.cluster_configuration.fmin
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

    @staticmethod
    def _coincidence_info(events, events_times_cluster):

        for item in events:
            if item['time'] in events_times_cluster:
                pprint(item)

    def thresholding_sta_lta(self, files_list, start, end):

        traces = []
        events_times_cluster = None
        events = None
        for file in files_list:
            try:
                tr = read(file)[0]
                tr = self.fill_gaps(tr)
                if tr is not None:
                    traces.append(tr)
            except:
                pass
        st = Stream(traces)
        st.trim(starttime=start, endtime=end)
        st.merge()
        print("Ready to run coincidence Trigger at: ")
        print(st.__str__(extended=True))
        st.detrend(type='simple')
        st.taper(max_percentage=0.05)
        st.filter(type="bandpass", freqmin=self.fmin, freqmax=self.fmax)
        st.detrend(type='simple')
        st.taper(max_percentage=0.05)

        try:
            events = coincidence_trigger('classicstalta', self.the_on, self.the_off, st,
                                         thr_coincidence_sum=self.coincidence,
                                         max_trigger_length=self.centroid_radio, sta=self.sta, lta=self.lta)
            triggers = self._extract_event_info(events)
            if len(triggers) > 0:
                events_times_cluster, _ = MseedUtil.cluster_events(triggers, eps=self.centroid_radio)
                self._coincidence_info(events, events_times_cluster)
                print("Number of events detected", len(events_times_cluster))
        except:
            print("An error raised, Coindicence Trigger no applied")

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
                       self.coincidence_config.cluster_configuration.fmin)

            st_kurt = cf_kurt.run_kurtosis()
            events = coincidence_trigger(trigger_type=None, thr_on=self.the_on, thr_off=self.the_off,
                                         trigger_off_extension=self.centroid_radio,
                                         thr_coincidence_sum=self.coincidence, stream=st_kurt,
                                         similarity_threshold=0.8, details=True)

            triggers = self._extract_event_info(events)

            if len(triggers) > 0:
                events_times_cluster, _ = MseedUtil.cluster_events(triggers, eps=self.centroid_radio)
                self._coincidence_info(events, events_times_cluster)
                print("Number of events detected", len(events_times_cluster))

        return events, events_times_cluster


    def _process_coincidence_trigger(self, filtered_files):

        """Process a single day's data and return events or 'empty'."""

        return self.thresholding_cwt_kurt(filtered_files)

    def optimized_project_processing(self):

        final_filtered_results = []

        for project in self.projects:
            data_files = project.data_files
            final_filtered_results.append(self._process_coincidence_trigger(data_files))

        return final_filtered_results

    # def separate_picks_by_events(self, input_file, output_file, centroids):
    #     """
    #     Separates picks by events based on centroids and their radius.
    #
    #     :param input_file: Path to the input file with pick data.
    #     :param output_file: Path to the output file for separated picks.
    #     :param centroids: List of UTCDateTime objects representing centroids.
    #     :param radius: Radius in seconds for each centroid.
    #     """
    #     # Load the input data
    #     df = pd.read_csv(input_file, delimiter='\s+')
    #     # Ensure columns are properly typed
    #     # df['Date'] = df['Date'].astype(str)  # Date as string for slicing
    #     # df['Hourmin'] = df['Hourmin'].astype(int)  # Hourmin as integer
    #     # df['Seconds'] = df['Seconds'].astype(float)  # Seconds as float for fractional handling
    #     # Parse date and time columns into a single datetime object
    #     df['FullTime'] = df.apply(lambda row: UTCDateTime(
    #         f"{row['Date']}T{str(row['Hour_min']).zfill(4)}:{row['Seconds']}"
    #     ), axis=1)
    #
    #     # Create a new column for event assignment
    #     df['Event'] = None
    #
    #     # Assign picks to the closest centroid within the radius
    #     for i, centroid in enumerate(centroids):
    #         within_radius = df['FullTime'].apply(lambda t: abs((t - centroid)) <= self.centroid_radio / 2)
    #         df.loc[within_radius, 'Event'] = f"Event_{i + 1}"
    #
    #     # Write grouped picks into the output file
    #     with open(output_file, 'w') as f:
    #         for event, group in df.groupby('Event'):
    #             if pd.isna(event):
    #                 continue  # Skip unassigned picks
    #             f.write(f"{event}\n")
    #             group.drop(columns=['Event', 'FullTime']).to_csv(f, sep='\t', index=False)
    #             f.write("\n")