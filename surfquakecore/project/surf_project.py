import pickle
from datetime import datetime
from multiprocessing import Pool
import os
from functools import partial
from typing import Tuple, List
import re
from obspy import read, UTCDateTime
#from collections import ChainMap
import copy

class SurfProject:

    def __init__(self, root_path):

        self.root_path = root_path
        self.project = {}
        self.data_files = []

    def __add__(self, other):
        if isinstance(other, SurfProject):
            #join_project = ChainMap(self.project, other.project)

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
                    print(key, os.path.basename(value[0]), value[1].sampling_rate,  value[1].starttime,
                          value[1].endtime)
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

    def save_project(self, path):

        if len(self.project):
            try:
                file_to_store = open(path, "wb")
                pickle.dump(self.project, file_to_store)
                print("Succesfully saved project")
            except ProjectSaveFailed as e:
                print(f"Project couldn't be saved: {e}")

    def search_files(self, verbose=True, **kwargs):
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

        for top_dir, sub_dir, files in os.walk(self.root_path):
            for file in files:
                data_files.append(os.path.join(top_dir, file))

        cpus = min(len(data_files), os.cpu_count())
        with Pool(processes=cpus) as pool:
            partial_task = partial(self._parse_data_file, filter=filter, verbose=verbose)
            returned_list = pool.map(partial_task, data_files)

        self._convert2dict(returned_list)


    def _parse_data_file(self, file: str, filter: dict, verbose: bool):

        check_filter_time = True
        check_filter_selection = True

        try:
            header = read(file, headeronly=True)
        except TypeError:
            return [None, None]

        net = header[0].stats.network
        sta = header[0].stats.station
        chn = header[0].stats.channel
        key = f"{net}.{sta}.{chn}"

        # filter time
        if filter["start"] is not None and filter["end"] is not None:
            if filter["start"] <= header[0].stats.starttime.datetime and header[0].stats.endtime.datetime <= filter["end"]:
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

    def _convert2dict(self, data: Tuple[str, List[str]]):
        self.project = {}
        for name in data:
            if name[0] in self.project.keys() and name[0] is not None:
                self.project[name[0]].append([name[1][0], name[1][1]])

            elif name[0] not in self.project.keys() and name[0] is not None:
                self.project[name[0]] = [[name[1][0], name[1][1]]]


    def filter_project_keys(self, **kwargs):

        self.data_files = []

        # filter dict by python wilcards remind

        # * --> .+
        # ? --> .

        net = kwargs.pop('net', '.+')
        station = kwargs.pop('station', '.+')
        channel = kwargs.pop('channel', '.+')
        if net == '':
            net = '.+'
        if station == '':
            station = '.+'
        if channel == '':
            channel = '.+'

        # filter for regular expresions
        filter_list = [net, station, channel]
        project_filtered = self._search(self.project, filter_list)
        self.project = project_filtered
        for key, value in self.project.items():
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

    def filter_time(self, **kwargs):

        # filter the list output of filter_project_keys by trimed times

        result = []
        st1 = kwargs.pop('starttime', None) # st1 is UTCDateTime
        et1 = kwargs.pop('endtime', None) # st2 is UTCDateTime

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

    def get_now_files(self, date, stations_list, channel_list):
        date = UTCDateTime(date)

        selection = [".", stations_list, channel_list]

        self.filter_project_keys(net=selection[0], station=selection[1],
                                 channel=selection[2])
        start = date - 1300  # half and hour before
        end = date + 3 * 3600  # end 2 hours after
        files_path = self.filter_time(list_files=self.data_files, starttime=start, endtime=end)
        return files_path

class ProjectSaveFailed(Exception):
    pass

