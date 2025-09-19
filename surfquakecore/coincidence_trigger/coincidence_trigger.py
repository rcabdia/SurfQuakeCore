#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
coincidence_trigger
"""
import os
from typing import Union
from datetime import datetime
from obspy import UTCDateTime, read, Stream
from obspy.signal.trigger import coincidence_trigger
from multiprocessing import Pool
import pandas as pd
from surfquakecore.coincidence_trigger.cf_kurtosis import CFKurtosis
from surfquakecore.coincidence_trigger.coincidence_parse import load_concidence_configuration
from surfquakecore.coincidence_trigger.structures import CoincidenceConfig
from surfquakecore.project.surf_project import SurfProject
from pprint import pprint
from surfquakecore.utils.obspy_utils import MseedUtil


class CoincidenceTrigger:
    def __init__(self, project: Union[SurfProject, str], parameters: dict,
                 coincidence_config: Union[str, CoincidenceConfig]):

        if isinstance(project, str):
            self.project = SurfProject.load_project(project)
        else:
            self.project = project

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
            self.sta = parameters.pop("sta", 1)  # time_window_kurtosis
            self.lta = parameters.pop("lta", 40)

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
        st.merge()

        print("Ready to run coincidence Trigger at: ")
        print(st.__str__(extended=True))

        cf_kurt = CFKurtosis(files_list, self.coincidence_config.kurtosis_configuration,
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

    def process_coincidence_trigger(self, args):
        """Process a single day's data and return events or 'empty'."""
        sp, start, end = args
        # Filter files for the given time range
        filtered_files = sp.filter_time(starttime=start, endtime=end, tol=3600, use_full=True)
        if filtered_files:
            if self.coincidence_config.cluster_configuration.method_preferred == 'SNR':
                return self.thresholding_sta_lta(filtered_files, start, end)
            else:
                return self.thresholding_cwt_kurt(filtered_files)
        else:
            return "empty"

    def optimized_project_processing(self, **kwargs):

        final_filtered_results = []
        details = []

        input_file: str = kwargs.pop('input_file', None)
        output_file: str = kwargs.pop('output_file', None)

        info = self.project.get_project_basic_info()
        print(info['Start'], info['End'])

        # Parse start and end times
        start_time = UTCDateTime(datetime.strptime(info['Start'], '%Y-%m-%d %H:%M:%S'))
        end_time = UTCDateTime(datetime.strptime(info['End'], '%Y-%m-%d %H:%M:%S'))

        # Generate daily time ranges
        daily_ranges = [(start_time + i * 86400, start_time + (i + 1) * 86400)
                        for i in range(int((end_time - start_time) // 86400))]

        if len(daily_ranges) == 0 and (end_time - start_time) < 86400:
            daily_ranges = [(start_time, end_time)]

        # Prepare arguments for multiprocessing
        tasks = [(self.project, start, end) for start, end in daily_ranges]

        # Use multiprocessing to parallelize
        with Pool() as pool:
            results = pool.map(self.process_coincidence_trigger, tasks)

        # Join the output of all days
        for item in results:
            if item[0] is not None and item[1] is not None:
                details.extend(item[0])
                final_filtered_results.extend(item[1])

        if len(final_filtered_results) > 0 and input_file is not None and output_file is not None:
            self.separate_picks_by_events(input_file, output_file, centroids=final_filtered_results)

        return final_filtered_results, details

