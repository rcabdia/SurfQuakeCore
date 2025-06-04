# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: surf_project.py
# Program: surfQuake & ISP
# Date: January 2024
# Purpose: Project Manager
# Author: Roberto Cabieces & Thiago C. Junqueira
#  Email: rcabdia@roa.es
# --------------------------------------------------------------------

import pickle
from datetime import datetime
from multiprocessing import Pool, cpu_count
import os
from functools import partial
from typing import Tuple, List, Union
import re
from obspy import read, UTCDateTime
import copy


def _generate_subproject_for_time_window(args):
    time_window, serialized_self, metadata = args
    start, end = time_window
    self = pickle.loads(serialized_self)

    self.filter_project_time(starttime=start, endtime=end)

    self.data_files = []
    for key, value in self.project.items():
        for item in value:
            self.data_files.append([item[0], item[1].starttime, item[1].endtime])

    if metadata:
        self._event_metadata = metadata

    return self if len(self.project) > 0 else None


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

    def search_files(self, format="NONE", verbose=True, **kwargs):

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

        if isinstance(self.root_path, list) and len(self.root_path) > 0:
            for file in self.root_path:
                data_files.append(file)
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

    def _fill_list(self):
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
        elif isinstance(starttime, UTCDateTime):
            start = starttime
        else:
            raise TypeError("starttime must be a string or UTCDateTime object.")

        if isinstance(endtime, str):
            end = datetime.strptime(endtime, date_format)
            end = UTCDateTime(end)
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

            # for file in data_files:

            for file in self.data_files:

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
        import os
        import pickle

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(subprojects, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved {len(subprojects)} subprojects to {path}")
        return os.path.isfile(path)

    def split_by_time_spans(self,
                            span_seconds: int = 86400,
                            verbose: bool = False,
                            save_path: str = None,
                            min_date: str = None,
                            max_date: str = None,
                            event_file: str = None,
                            event_window_seconds: int = 3600) -> List["SurfProject"]:
        """
        Splits the project by fixed spans or event times. Can attach event metadata.

        Args:
            span_seconds (int): Time window (ignored if event_file provided).
            verbose (bool): Print details per subproject.
            save_path (str): If set, saves the list of subprojects as a single pickle file.
            min_date (str): Optional min datetime string.
            max_date (str): Optional max datetime string.
            event_file (str): CSV with events.
            event_window_seconds (int): Window after event origin.

        Returns:
            List[SurfProject]
        """
        import csv


        def parse_event_file(path):
            events = []
            with open(path, "r") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    try:
                        dt = datetime.strptime(f"{row['date']} {row['hour']}", "%Y-%m-%d %H:%M:%S.%f")
                    except ValueError:
                        dt = datetime.strptime(f"{row['date']} {row['hour']}", "%Y-%m-%d %H:%M:%S")

                    events.append({
                        "origin_time": UTCDateTime(dt),
                        "latitude": float(row["latitude"]),
                        "longitude": float(row["longitude"]),
                        "depth": float(row["depth"]),
                        "magnitude": float(row["magnitude"])
                    })
            return events

        # Project time bounds
        info = self.get_project_basic_info()
        date_format = "%Y-%m-%d %H:%M:%S"
        global_start = UTCDateTime(datetime.strptime(info["Start"], date_format))
        global_end = UTCDateTime(datetime.strptime(info["End"], date_format))

        start = UTCDateTime(min_date) if min_date else global_start
        end = UTCDateTime(max_date) if max_date else global_end

        # Build time windows
        time_windows = []
        metadata_list = []

        if event_file:
            events = parse_event_file(event_file)
            for event in events:
                origin = event["origin_time"]
                if start <= origin <= end:
                    time_windows.append((origin, origin + event_window_seconds))
                    metadata_list.append(event)
        else:
            current = start
            while current < end:
                time_windows.append((current, current + span_seconds))
                metadata_list.append(None)
                current += span_seconds

        # Parallel build
        serialized_self = pickle.dumps(self)
        args = list(zip(time_windows, [serialized_self] * len(time_windows), metadata_list))

        with Pool(processes=min(cpu_count(), len(time_windows))) as pool:
            results = pool.map(_generate_subproject_for_time_window, args)

        subprojects = [sp for sp in results if sp is not None]

        if verbose:
            for i, sp in enumerate(subprojects):
                meta = getattr(sp, "_event_metadata", {})
                print(f"[{i}] {time_windows[i][0]} → {time_windows[i][1]} | "
                      f"{sp.get_project_basic_info()['num_files']} files | "
                      f"Event: {meta.get('origin_time', 'span')}")

        if save_path:
            self.save_subprojects_list(save_path, subprojects)

        return subprojects


class ProjectSaveFailed(Exception):
    def __init__(self, message="Error saving project"):
        self.message = message
        super().__init__(self.message)
