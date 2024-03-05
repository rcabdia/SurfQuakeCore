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
from multiprocessing import Pool
import os
from functools import partial
from typing import Tuple, List, Union
import re
from obspy import read, UTCDateTime
import copy


class SurfProject:

    def __init__(self, root_path: Union[str, List[str]]):

        """

        SurfProject class is designed to be able to storage the path to seismograms
        files plus the file metadata information (i.e. sampling_rate, starttime...)

        Attributes:
        - root_path (str): The root path to the folder where the user have the data files or a list filled
        with paths to files.
        - project (dict)
        - data_files (list)

        Methods:
        - __init__(root_path): Initialize a new instance of SurfProject.
        - load_project(path_to_project_file: str): Load a project from a file storage in hard-drive
        - save_project(path_file_to_storage: str): Saves a project as a pickle file in hard-drive
        - search_files(verbose=True, **kwargs): Create a project. It can be used filters by nets,
        stations, channels selection and/or filter by timestamp
        - filter_project_keys(**kwargs): Filter a project (once is crated) using regular expressions.
        """

        self.root_path = root_path
        self.project = {}
        self.data_files = []

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
                        pass
                        #print("exception arises at", key)
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

    @staticmethod
    def load_project(path_to_project_file: str):
        return pickle.load(open(path_to_project_file, "rb"))

    def save_project(self, path_file_to_storage: str)->bool:

        if not self.project:
            return False

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

        cpus = min(len(data_files), os.cpu_count())
        with Pool(processes=cpus) as pool:
            partial_task = partial(self._parse_data_file, format=format, filter=filter, verbose=verbose)
            returned_list = pool.map(partial_task, data_files)

        self._convert2dict(returned_list)

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
        project_filtered = {}

        # filter dict by python wilcards remind

        # * --> .+
        # ? --> .

        net = kwargs.pop('net', '.+')
        station = kwargs.pop('station', '.+')
        channel = kwargs.pop('channel', '.+')
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
                for j in value:
                    self.data_files.append([j[0], j[1]['starttime'], j[1]['endtime']])
        else:
            for key, value in project_filtered.items():
                for j in value:
                    self.data_files.append([j[0], j[1]['starttime'], j[1]['endtime']])



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

    def filter_project_time(self, starttime: str, endtime: str):

        """

        - starttime (str, "%Y-%m-%d %H:%M:%S"): String with the reference starttime, upper time spam threshold
        (i.e., "2023-12-10 00:00:00")

        - endtime (str, "%Y-%m-%d %H:%M:%S" ): String with the reference endtime, lower time spam threshold
        (i.e., "2023-12-23 00:00:00")

        """

        date_format = "%Y-%m-%d %H:%M:%S"
        start = datetime.strptime(starttime, date_format)
        end = datetime.strptime(endtime, date_format)
        if len(self.project) > 0:
            for key in self.project:
                item = self.project[key]
                indices_to_remove = []
                for index, value in enumerate(item):
                    start_data = value[1].starttime
                    end_data = value[1].endtime
                    if start <= start_data and end >= end_data:
                        pass
                    else:
                        indices_to_remove.append(index)
                for index in reversed(indices_to_remove):
                    item.pop(index)
                self.project[key] = item

    def filter_time(self, **kwargs) -> list:

        # filter the list output of filter_project_keys by trimed times

        result = []
        st1 = kwargs.pop('starttime', None)  # st1 is UTCDateTime
        et1 = kwargs.pop('endtime', None)  # st2 is UTCDateTime

        if st1 is None and et1 is None:
            for file in self.data_files:
                result.append(file[0])

        else:

            for file in self.data_files:
                pos_file = file[0]
                st0 = file[1]
                et0 = file[2]
                # check times as a filter

                if st1 >= st0 and et1 > et0 and (st1 - st0) <= 86400:
                    result.append(pos_file)
                elif st1 <= st0 and et1 >= et0:
                    result.append(pos_file)
                elif st1 <= st0 and et1 <= et0 and (et0 - et1) <= 86400:
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


class ProjectSaveFailed(Exception):
    def __init__(self, message="Error saving project"):
        self.message = message
        super().__init__(self.message)
