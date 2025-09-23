# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: surf_project.py
# Program: surfQuake & ISP
# Date: January 2024
# Purpose: Project Manager
# Author: Roberto Cabieces & Thiago C. Junqueira
#  Email: rcabdia@roa.es
# --------------------------------------------------------------------

import glob
import pickle
from datetime import datetime
from multiprocessing import Pool, cpu_count
import os
from functools import partial
from typing import Tuple, List, Union
import re
from obspy import read, UTCDateTime
import copy
import csv

def _generate_subproject_for_time_window(args):
    (start, end), serialized_project, events, mode, overlap_threshold, margin = args
    project = pickle.loads(serialized_project)

    # Extract only traces within the window
    sub = project.extract_subproject_between(start, end, mode=mode, overlap_threshold=overlap_threshold,
                                             margin=margin)

    # Attach relevant events
    setattr(sub, "_events_metadata", events or [])

    # Assign only relevant files to data_files
    sub.data_files = []
    for trace_list in sub.project.values():
        for trace_path, _ in trace_list:
            sub.data_files.append(trace_path)

    return sub


class SurfProject:

    def __init__(self, root_path: Union[str, List[str]] = None):

        """

        SurfProject class is designed to be able to storage the path to seismograms
        files plus the file metadata information (i.e. sampling_rate, starttime...)

        Attributes:
        - root_path (str): The root path to the folder where the user have the data files or a list filled
        with paths to files.
        - project (dict): e.g. {'WM.ARNO.HHZ' : [[FILE1, STATS_FILE1], [[FILE2, STATS_FILE2] ....'WM.MELI.HHN': ... ]
        - data_files (list) [PATH_FILE1, PATH_FILE2, PATH_FILE3 ....]

        Methods:
        - __init__(root_path): Initialize a new instance of SurfProject.
        - load_project(path_to_project_file: str): Load a project from a file storage in hard-drive
        - save_project(path_file_to_storage: str): Saves a project as a pickle file in hard-drive
        - search_files(verbose=True, **kwargs): Create a project. It can be used filters by nets,
        stations, channels selection and/or filter by timestamp
        - filter_project_keys(**kwargs): Filter a project (once is crated) using regular expressions.
        - filter_time(**kwargs): Filter the project by span time and return a list with path of available files
        - split_by_time_spans(**kwargs): Splits the project into subprojects by time spans or event-based windows.
        """

        self.root_path = root_path
        self.project = {}
        self.data_files = []
        self.data_files_full = []

    def __add__(self, other):
        if isinstance(other, SurfProject):

            root_path = [self.root_path, other.root_path]
            data_files = self.data_files + other.data_files
            join_project = {**self.project, **other.project}
            sp = SurfProject(root_path)
            sp.project = join_project
            sp.data_files = data_files

            return sp
        else:
            raise TypeError("It couldn't be joined both projects")

    def __str__(self):
        if len(self.project) > 0:
            for key in self.project:
                print(f' {"Code"}      {"File"}     {"Sampling Rate"}     {"StartTime"}     {"EndTime"}')
                item = self.project[key]
                for value in item:
                    try:
                        print(key, os.path.basename(value[0]), value[1].sampling_rate, value[1].starttime,
                              value[1].endtime)
                    except:
                        print("exception arises at", key)

            info = self.get_project_basic_info()

            print('Networks: ', ','.join(info["Networks"][0]))
            print('Stations: ', ','.join(info["Stations"][0]))
            print('Channels: ', ','.join(info["Channels"][0]))
            print("Num Networks: ", info["Networks"][1], "Num Stations: ", info["Stations"][1], "Num Channels: ",
                  info["Channels"][1], "Num Total Files: ", info["num_files"])
            print("Start Project: ", info["Start"], ", End Project: ", info["End"])
        else:
            print("Empty Project")

        return ""

    def __copy__(self):
        # Create a new instance of the class with the same data
        new_instance = self.__class__(self.root_path)
        # Copy dynamically created attributes
        for key, value in vars(self).items():
            setattr(new_instance, key, value)
        return new_instance

    def copy(self):
        return copy.copy(self)

    def __deepcopy__(self, memo):
        # Deep copy: all referenced objects are copied as well
        new_instance = self.__class__(copy.deepcopy(self.root_path, memo))
        for key, value in vars(self).items():
            setattr(new_instance, key, copy.deepcopy(value, memo))
        return new_instance

    def deepcopy(self):
        return copy.deepcopy(self)

    @staticmethod
    def load_project(path_to_project_file: str):
        return pickle.load(open(path_to_project_file, "rb"))

    def save_project(self, path_file_to_storage: str) -> bool:

        if not self.project:
            print('Aqui')
            return False

        if os.path.isdir(os.path.dirname(path_file_to_storage)):
            pass
        else:
            try:
                os.makedirs(os.path.dirname(path_file_to_storage))
            except Exception as error:
                print("An exception occurred:", error)

        with open(path_file_to_storage, "wb") as file_to_store:
            pickle.dump(self, file_to_store, protocol=pickle.HIGHEST_PROTOCOL)

            print("Project successfully saved.")

        return os.path.isfile(path_file_to_storage)

    def extract_subproject_between(self, start, end, mode="tolerant", margin=5, overlap_threshold=0.05):
        """
        Extract a subproject containing only data files within a time window.

        Args:
            start (UTCDateTime): Time window start.
            end (UTCDateTime): Time window end.
            mode (str): 'tolerant', 'strict', 'max_overlap_per_day', or 'overlap_threshold'.
            margin (int): Extra buffer seconds for 'tolerant' mode.
            overlap_threshold (float): Minimum ratio of overlap for 'overlap_threshold' mode.

        Returns:
            SurfProject: New subproject with selected files.
        """
        selected = {}

        window_duration = (end - start)

        for key, trace_list in self.project.items():
            best_entry = None
            max_overlap = 0
            good_entries = []

            for trace_path, stats in trace_list:
                overlap = max(0, min(end, stats.endtime) - max(start, stats.starttime))

                if mode == "max_overlap_per_day":
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_entry = (trace_path, stats)

                elif mode == "overlap_threshold":
                    if overlap / window_duration >= overlap_threshold:
                        good_entries.append((trace_path, stats))

                elif mode == "strict":
                    if stats.starttime <= end and stats.endtime >= start:
                        if key not in selected:
                            selected[key] = []
                        selected[key].append((trace_path, stats))

                elif mode == "tolerant":
                    if stats.starttime <= end + margin and stats.endtime >= start - margin:
                        if key not in selected:
                            selected[key] = []
                        selected[key].append((trace_path, stats))

            if mode == "max_overlap_per_day" and best_entry:
                selected[key] = [best_entry]

            elif mode == "overlap_threshold" and good_entries:
                selected[key] = good_entries

        # Create and populate subproject
        new_project = self.clone_project_with_subset(selected)
        new_project.data_files = [trace_path for trace_list in selected.values() for trace_path, _ in trace_list]

        return new_project

    @staticmethod
    def collect_files(root_path) -> list:
        """
        Collect seismic files (e.g., MSEED, SAC) from a glob pattern or list.

        Parameters
        ----------
        root_path : str or list
            A glob string (e.g., './data/*.mseed') or a list of file paths.

        Returns
        -------
        list
            Filtered list of absolute paths to valid seismic files.
        """

        BLACKLIST_EXTENSIONS = {'.txt', '.log', '.png', '.jpg', '.csv', '.json', '.xml',
                                '.yaml', '.pdf', '.docx', '.pkl'}

        if isinstance(root_path, str):
            all_files = glob.glob(root_path, recursive=True)
            print(f"[INFO] Found {len(all_files)} files using glob pattern: {root_path}")
        elif isinstance(root_path, list):
            all_files = [os.path.abspath(p) for p in root_path if os.path.exists(p)]
            print(f"[INFO] Using {len(all_files)} explicitly provided files.")
        else:
            raise ValueError("[ERROR] root_path must be a string (glob) or a list of paths.")

        # Filter out blacklisted extensions
        valid_files = [
            f for f in all_files
            if os.path.splitext(f)[1].lower() not in BLACKLIST_EXTENSIONS
        ]

        print(f"[INFO] {len(valid_files)} files retained after filtering.")
        valid_files.sort()
        return valid_files

    def search_files(self, format="NONE", verbose=True, use_glob: bool = False, **kwargs):

        """
        Args:

        - verbose (bool): Description of arg1.
        - nets (str): String with the name of selected nets to be filtered (i.e., "WM,ES")
        - stations (str): String with the name of selected stations to be filtered (i.e., "ARNO,UCM,EMAL")
        - channels (str): String with the name of selected channels to be filtered (i.e., "HHN,HHZ,HHE")
        - starttime (str "%Y-%m-%d %H:%M:%S" ): String with the reference starttime, upper time spam threshold
        (i.e.,"2023-12-10 00:00:00")
        - endtime (str "%Y-%m-%d %H:%M:%S" ): String with the reference endtime, lower time spam threshold
        (i.e.,"2023-12-23 00:00:00")

        Returns:
        - type: Description of the return value.
        """

        date_format = "%Y-%m-%d %H:%M:%S"  # "2023-12-11 14:30:00"
        start: str = kwargs.pop('starttime', None)
        end: str = kwargs.pop('endtime', None)

        nets: str = kwargs.pop('nets', None)
        stations: str = kwargs.pop('stations', None)
        channels: str = kwargs.pop('channels', None)

        if start is not None and end is not None:
            start = datetime.strptime(start, date_format)
            end = datetime.strptime(end, date_format)
        else:
            start = None
            end = None

        if nets is not None:
            nets = nets.split(",")
        if stations is not None:
            stations = stations.split(",")
        if channels is not None:
            channels = channels.split(",")

        filter = {"start": start, "end": end, "nets": nets, "stations": stations, "channels": channels}
        data_files = []

        if isinstance(self.root_path, list):
            # Assume explicit file list
            data_files = [f for f in self.root_path if os.path.isfile(f)]
            if verbose:
                print(f"[INFO] Using {len(data_files)} explicitly provided waveform files.")

        elif isinstance(self.root_path, str):
            if use_glob:
                data_files = glob.glob(self.root_path, recursive=True)
                if verbose:
                    print(f"[INFO] Found {len(data_files)} files using glob pattern: {self.root_path}")
            else:
                for top_dir, sub_dir, files in os.walk(self.root_path):
                    for file in files:
                        data_files.append(os.path.join(top_dir, file))

        cpus = max(1, min(len(data_files), os.cpu_count() or 1))
        with Pool(processes=cpus) as pool:
            partial_task = partial(self._parse_data_file, format=format, filter=filter, verbose=verbose)
            returned_list = pool.map(partial_task, data_files)

        self._convert2dict(returned_list)
        self._fill_list()

        if verbose:
            info = self.get_project_basic_info()

            print('Networks: ', info["Networks"][0])
            print('Stations: ', info["Stations"][0])
            print('Channels: ', info["Channels"][0])
            print("Num Networks: ", info["Networks"][1], "Num Stations: ", info["Stations"][1], "Num Channels: ",
                  info["Channels"][1], "Num Total Files: ", info["num_files"])

    def get_data_files(self):
        self._fill_list()
        return self.data_files

    def _fill_list(self):
        self.data_files = []
        for item in self.project.items():
            list_channel = item[1]
            for file_path in list_channel:
                self.data_files.append(file_path[0])

    def _fill_list_full(self):
        self.data_files_full = []
        for item in self.project.items():
            list_channel = item[1]
            for file_path in list_channel:
                self.data_files_full.append([file_path[0], file_path[1].starttime, file_path[1].endtime])

    def _parse_data_file(self, file: str, format: str, filter: dict, verbose: bool):

        check_filter_time = True
        check_filter_selection = True

        try:
            if format == "NONE":
                header = read(file, headeronly=True)
            else:
                header = read(file, headeronly=True, format=format)
        except (TypeError, Exception) as e:
            print(f"Error occurred while parsing data file: {e}")
            return [None, None]

        try:
            net = header[0].stats.network
            sta = header[0].stats.station
            chn = header[0].stats.channel
            key = f"{net}.{sta}.{chn}"

            # filter time
            if filter["start"] is not None and filter["end"] is not None:
                if filter["start"] <= header[0].stats.starttime.datetime and header[0].stats.endtime.datetime <= filter[
                    "end"]:
                    pass
                else:
                    check_filter_time = False

            # filter selection
            if filter["nets"] is not None:
                if net in filter["nets"]:
                    pass
                else:
                    check_filter_selection = False

            if filter["stations"] is not None:
                if sta in filter["stations"]:
                    pass
                else:
                    check_filter_selection = False

            if filter["channels"] is not None:
                if chn in filter["channels"]:
                    pass
                else:
                    check_filter_selection = False

            if check_filter_time and check_filter_selection:
                data_map = [file, header[0].stats]
                if verbose:
                    print("included in the project file ", file)
            else:
                data_map = [None, None]

            return key, data_map
        except Exception as e:
            print(f"Error occurred during further processing: {e}")
            return [None, None]

    def _convert2dict(self, data: Tuple[str, List[str]]):
        self.project = {}
        for name in data:
            if name[0] in self.project.keys() and name[0] is not None:
                self.project[name[0]].append([name[1][0], name[1][1]])

            elif name[0] not in self.project.keys() and name[0] is not None:
                self.project[name[0]] = [[name[1][0], name[1][1]]]

    def filter_project_keys(self, **kwargs):

        """
        Args:
        - net (str): String with the name of selected nets to be filtered (i.e., ".")
        - station (str): String with the name of selected stations to be filtered (i.e., "ARNO|UCM|EMAL")
        - channel (str): String with the name of selected channels to be filtered (i.e., "HH.")
        """

        self.data_files = []

        # filter dict by python wilcards remind

        # * --> .+
        # ? --> .

        net = kwargs.pop('net', '.+')
        station = kwargs.pop('station', '.+')
        channel = kwargs.pop('channel', '.+')
        verbose = kwargs.pop('verbose', False)
        only_datafiles_list = kwargs.pop('only_datafiles_list', False)

        if net == '':
            net = '.+'
        if station == '':
            station = '.+'
        if channel == '':
            channel = '.+'

        # filter for regular expresions
        filter_list = [net, station, channel]
        project_filtered = self._search(self.project, filter_list)

        if not only_datafiles_list:
            self.project = project_filtered
            for key, value in self.project.items():
                if value[0][0] is not None:
                    for j in value:
                        self.data_files.append([j[0], j[1]['starttime'], j[1]['endtime']])

        else:
            for key, value in project_filtered.items():
                for j in value:
                    self.data_files.append([j[0], j[1]['starttime'], j[1]['endtime']])

        # clean possible empty lists
        # self.remove_empty_keys()

        # if verbose:
        #    info = self.get_project_basic_info()

        #    print('Networks: ', str(info["Networks"][0]))
        #    print('Stations: ', str(info["Stations"][0]))
        #    print('Channels: ', str(info["Channels"][0]))
        #    print("Num Networks: ", info["Networks"][1], "Num Stations: ", info["Stations"][1], "Num Channels: ",
        #          info["Channels"][1], "Num Total Files: ", info["num_files"])

    def _search(self, project: dict, event: list):
        res = {}
        for key in project.keys():
            name_list = key.split('.')
            net = name_list[0]
            sta = name_list[1]
            channel = name_list[2]
            if re.search(event[0], net) and re.search(event[1], sta) and re.search(event[2], channel):
                res[key] = project[key]

        return res

    def filter_project_time(self, starttime, endtime, tol=86400, verbose=False):

        """
        Filters project data based on time range.

        - starttime (str or UTCDateTime): Start time of the range.
          If str, it should follow the format "%Y-%m-%d %H:%M:%S".
          Example: "2023-12-10 00:00:00"
        - endtime (str or UTCDateTime): End time of the range.
          If str, it should follow the format "%Y-%m-%d %H:%M:%S".
          Example: "2023-12-23 00:00:00"
        """

        # Clean the data_files_list
        self.data_files = []

        # Convert starttime and endtime to datetime objects if they are strings
        date_format = "%Y-%m-%d %H:%M:%S"
        if isinstance(starttime, str):
            start = datetime.strptime(starttime, date_format)
            start = UTCDateTime(start)
        elif isinstance(starttime, datetime):
            start = UTCDateTime(starttime)
        elif isinstance(starttime, UTCDateTime):
            start = starttime
        else:
            raise TypeError("starttime must be a string or UTCDateTime object.")

        if isinstance(endtime, str):
            end = datetime.strptime(endtime, date_format)
            end = UTCDateTime(end)
        elif isinstance(endtime, datetime):
            end = UTCDateTime(endtime)
        elif isinstance(endtime, UTCDateTime):
            end = endtime
        else:
            raise TypeError("endtime must be a string or UTCDateTime object.")

        # Process project data
        if len(self.project) > 0:
            for key in self.project:
                item = self.project[key]
                indices_to_remove = []
                for index, value in enumerate(item):
                    start_data = value[1].starttime
                    end_data = value[1].endtime
                    if start_data >= start and end_data > end and (start_data - start) <= tol:
                        pass

                    elif start_data <= start and end_data >= end:
                        pass

                    elif start_data <= start and end_data <= end and (end - end_data) <= tol:
                        pass

                    elif start_data >= start and end_data <= end:
                        pass

                    else:
                        indices_to_remove.append(index)

                for index in reversed(indices_to_remove):
                    item.pop(index)
                self.project[key] = item
        # clean possible empty lists
        self.remove_empty_keys()
        # Fill the data_files list
        self._fill_list()

        if verbose:
            info = self.get_project_basic_info()

            print('Networks: ', info["Networks"][0])
            print('Stations: ', info["Stations"][0])
            print('Channels: ', info["Channels"][0])
            print("Num Networks: ", info["Networks"][1], "Num Stations: ", info["Stations"][1], "Num Channels: ",
                  info["Channels"][1], "Num Total Files: ", info["num_files"])

    def filter_time(self, **kwargs) -> list:

        result = []
        st1 = kwargs.pop('starttime', None)  # st1 is UTCDateTime
        et1 = kwargs.pop('endtime', None)  # st2 is UTCDateTime
        tol = kwargs.pop('tol', 86400)
        use_full = kwargs.pop('use_full', False)

        # filter the list output of filter_project_keys by trimed times
        if use_full:
            self._fill_list_full()
            data_files = self.data_files_full
        else:
            # continuing with old style, only accessible when self.filter_project_keys, otherwise the data_files
            # does not have info of span time
            data_files = self.data_files

        if st1 is None and et1 is None:
            for file in data_files:
                result.append(file[0])

        else:

            for file in data_files:

                pos_file = file[0]
                st0 = file[1]
                et0 = file[2]
                # check times as a filter

                if st1 >= st0 and et1 > et0 and (st1 - st0) <= tol:
                    result.append(pos_file)
                elif st1 <= st0 and et1 >= et0:
                    result.append(pos_file)
                elif st1 <= st0 and et1 <= et0 and (et0 - et1) <= tol:
                    result.append(pos_file)
                elif st1 >= st0 and et1 <= et0:
                    result.append(pos_file)
                else:
                    pass

        result.sort()

        return result

    def get_now_files(self, date, stations_list, channel_list, only_datafiles_list=False):
        date = UTCDateTime(date)

        selection = [".", stations_list, channel_list]

        self.filter_project_keys(net=selection[0], station=selection[1],
                                 channel=selection[2], only_datafiles_list=only_datafiles_list)
        start = date - 1300  # half and hour before
        end = date + 3 * 3600  # end 2 hours after
        files_path = self.filter_time(list_files=self.data_files, starttime=start, endtime=end)
        return files_path

    def get_project_basic_info(self):
        """
            Counts the number of unique stations in a dictionary where keys are in the format 'NET.STATION.CHANNEL'.

            Args:
                data_dict (dict): The input dictionary with keys in the format 'NET.STATION.CHANNEL'.

            Returns:
                int: The number of unique stations and channels
        """

        ## Take the stations names and its number

        info = {}

        networks = set()
        stations = set()  # Use a set to store unique stations
        channels = set()
        num_stations = 0
        num_channels = 0
        num_networks = 0
        start_time = []
        end_time = []
        total_components = 0

        for key in self.project.keys():
            parts = key.split('.')  # Split the key by '.'
            network = f"{parts[0]}"
            networks.add(network)  # Add to the set
            station = f"{parts[1]}"
            stations.add(station)
            channel = f"{parts[2]}"
            channels.add(channel)
            for item in self.project[key]:
                start_time.append(item[1].starttime)
                end_time.append(item[1].endtime)

        if len(stations) > 0:
            num_stations = len(stations)
            num_channels = len(channels)
            num_networks = len(networks)

        ## Take the number of files

        if len(stations) > 0:
            total_components = sum(len(value_list) for value_list in self.project.values())

        if len(stations) > 0:
            info["Networks"] = [networks, num_networks]
            info["Stations"] = [stations, num_stations]
            info["Channels"] = [channels, num_channels]
            info["num_files"] = total_components
            info["Start"] = min(start_time).strftime(format="%Y-%m-%d %H:%M:%S")
            info["End"] = max(end_time).strftime(format="%Y-%m-%d %H:%M:%S")

        return info

    def remove_empty_keys(self):
        """
        Removes keys from the dictionary where the values are empty lists.
        """
        # Use dictionary comprehension to filter out keys with empty lists
        self.project = {key: value for key, value in self.project.items() if value}

    def save_subprojects_list(self, path: str, subprojects: List["SurfProject"]) -> bool:
        """
        Save the entire list of subprojects to one pickle file.

        Args:
            path (str): Full path to output .pkl file.
            subprojects (list): List of SurfProject objects.

        Returns:
            bool: True if saved successfully.
        """

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(subprojects, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved {len(subprojects)} subprojects to {path}")
        return os.path.isfile(path)

    def clone_project_with_subset(self, selected_project_dict: dict):
        """
        Internal method to clone a SurfProject with a reduced set of files.

        Args:
            selected_project_dict (dict): Dict with same structure as self.project but
                                          filtered to only include desired keys and traces.

        Returns:
            SurfProject: A new SurfProject with the filtered data.
        """
        new_project = SurfProject()

        # Deepcopy necessary attributes
        new_project.metadata = copy.deepcopy(self.metadata) if hasattr(self, "metadata") else {}
        new_project.settings = copy.deepcopy(self.settings) if hasattr(self, "settings") else {}

        # Include only selected files
        new_project.project = selected_project_dict

        return new_project

    def extract_files_in_window(self, window_start, window_end, mode="tolerant", margin=5):
        """
        Select only files that intersect a time window.

        Args:
            window_start (UTCDateTime)
            window_end (UTCDateTime)
            mode (str): 'strict' or 'tolerant'
            margin (int): seconds of margin if mode='tolerant'

        Returns:
            Dict[str, List[Tuple[path, stats]]]
        """
        selected = {}
        for key, trace_list in self.project.items():
            for trace_path, stats in trace_list:
                if mode == "strict":
                    if stats.starttime <= window_end and stats.endtime >= window_start:
                        if key not in selected:
                            selected[key] = []
                        selected[key].append((trace_path, stats))
                else:  # tolerant mode
                    if stats.starttime <= window_end + margin and stats.endtime >= window_start - margin:
                        if key not in selected:
                            selected[key] = []
                        selected[key].append((trace_path, stats))
        return selected

    def split_by_time_spans(self,
                            span_seconds: int = 86400,
                            verbose: bool = False,
                            save_path: str = None,
                            min_date: str = None,
                            max_date: str = None,
                            event_file: str = None,
                            cut_start_time: float = 300,
                            cut_end_time: float = 300,
                            overlap_threshold=0.05,
                            file_selection_mode: str = "tolerant") -> List["SurfProject"]:
        """
        Splits the project into subprojects by time spans or event-based windows.

        Args:
            span_seconds (int): Duration for fixed-span split (ignored if event_file is given).
            verbose (bool): Print info about each subproject.
            save_path (str): Optional path to save list of subprojects.
            min_date (str): Filter start, format = "%Y-%m-%d %H:%M:%S.%f".
            max_date (str): Filter end.
            event_file (str): Optional CSV with events.
            cut_start_time (float): Pre-arrival time.
            cut_end_time (float): Post-arrival time.
            overlap_threshold (float): minimum percentage over span_seconds to be
            considered a file inside a span time window. Recommended small for very split daily files
            file_selection_mode (str): 'tolerant' or 'strict'.

        Returns:
            List[SurfProject]
        """


        def parse_event_file(path):
            events = []
            with open(path, "r") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    try:
                        dt = datetime.strptime(f"{row['date']} {row['hour']}", "%Y-%m-%d %H:%M:%S.%f")
                    except ValueError:
                        dt = datetime.strptime(f"{row['date']} {row['hour']}", "%Y-%m-%d %H:%M:%S")

                    event = {
                        "origin_time": UTCDateTime(dt)
                    }

                    # Add optional fields if they exist
                    for field in ["latitude", "longitude", "depth", "magnitude"]:
                        if field in row and row[field].strip():
                            event[field] = float(row[field])

                    events.append(event)

            return events

        # Get bounds
        info = self.get_project_basic_info()
        date_format = "%Y-%m-%d %H:%M:%S"
        global_start = UTCDateTime(datetime.strptime(info["Start"], date_format))
        global_end = UTCDateTime(datetime.strptime(info["End"], date_format))

        start = UTCDateTime(min_date) if min_date else global_start
        end = UTCDateTime(max_date) if max_date else global_end

        results = []
        serialized_self = pickle.dumps(self)

        if event_file:
            events = parse_event_file(event_file)
            task_args = []

            for event in events:
                origin = event["origin_time"]
                if not (start <= origin <= end):
                    continue

                window_start = origin - cut_start_time
                window_end = origin + cut_end_time
                task_args.append(((window_start, window_end), serialized_self, [event], file_selection_mode,
                                  overlap_threshold, 5))

            with Pool(processes=min(cpu_count(), len(task_args))) as pool:
                subprojects = pool.map(_generate_subproject_for_time_window, task_args)
                results.extend([sp for sp in subprojects if sp is not None])

        else:
            # Fixed time span mode
            task_args = []
            current = start
            while current < end:
                window_start = current
                window_end = current + span_seconds
                task_args.append(((window_start, window_end), serialized_self, [], file_selection_mode,
                                  overlap_threshold, 5))
                current += span_seconds

            with Pool(processes=min(cpu_count(), len(task_args))) as pool:
                subprojects = pool.map(_generate_subproject_for_time_window, task_args)
                results.extend([sp for sp in subprojects if sp is not None])

        # Group subprojects by identical content (based on data_files)
        deduped = {}
        for sp in results:
            key = tuple(sorted(sp.data_files))
            if key in deduped:
                # Merge events
                old_events = getattr(deduped[key], "_events_metadata", [])
                new_events = getattr(sp, "_events_metadata", [])
                merged_events = old_events + [e for e in new_events if e not in old_events]
                setattr(deduped[key], "_events_metadata", merged_events)
            else:
                deduped[key] = sp

        final_subprojects = list(deduped.values())

        if verbose:
            for i, sp in enumerate(final_subprojects):
                meta = getattr(sp, "_events_metadata", [])
                info = sp.get_project_basic_info()
                ev_label = meta[0]["origin_time"] if meta else "span"
                if "Start" in info and "End" in info:
                    print(f"[{i}] {info['Start']} → {info['End']} | {info['num_files']} files | Event: {ev_label}")
                else:
                    print(f"[{i}] No valid time range | {info.get('num_files', 0)} files | Event: {ev_label}")

        if save_path:
            self.save_subprojects_list(save_path, final_subprojects)

        return final_subprojects


class ProjectSaveFailed(Exception):
    def __init__(self, message="Error saving project"):
        self.message = message
        super().__init__(self.message)
