import os
import pickle
import re
from multiprocessing import Pool
from typing import Tuple, List
import pandas as pd
from obspy import read, UTCDateTime, Inventory
from obspy.core.event import Origin
from surfquakecore.utils import read_nll_performance
from surfquakecore.utils.nll_org_errors import computeOriginErrors
from functools import partial

class MseedUtil:

    def __init__(self, robust=True, **kwargs):

        self.start = kwargs.pop('starttime', [])
        self.end = kwargs.pop('endtime', [])
        self.obsfiles = []
        self.pos_file = []
        self.robust = robust

        self._data_files = []

        self._project = {}

    @classmethod
    def load_project(cls, file: str):
        return pickle.load(open(file, "rb"))

    def get_files(self, folder: str):
        self._data_files = [os.path.join(top_dir, file)
                            for top_dir, sub_dir, files in os.walk(folder) for file in files]

        return self._data_files

    def _create_project(self, stations_dir: str):

        # TODO Improve this method and replace search_files()

        if not os.path.isdir(stations_dir):
            raise ValueError(f"The path {stations_dir} is not a valid directory")

        return self.get_files(stations_dir)

    def search_files(self, root_path: str, verbose=True):
        self._data_files = []
        for top_dir, sub_dir, files in os.walk(root_path):
            for file in files:
                self._data_files.append(os.path.join(top_dir, file))

        cpus = min(len(self._data_files), os.cpu_count())
        with Pool(processes=cpus) as pool:
            partial_task = partial(self._parse_data_file, verbose=verbose)
            returned_list = pool.map(partial_task, self._data_files)

        project = self._convert2dict(returned_list)

        return project

    @classmethod
    def filter_project_keys(cls, project, **kwargs):

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


        data = []

        # filter for regular expresions
        event = [net, station, channel]
        project = cls.search(project, event)

        for key, value in project.items():
            for j in value:
                data.append([j[0], j[1]['starttime'], j[1]['endtime']])

        return project, data

    @staticmethod
    def search(project, event):
        res = {}
        for key in project.keys():
            name_list = key.split('.')
            net = name_list[0]
            sta = name_list[1]
            channel = name_list[2]
            if re.search(event[0], net) and re.search(event[1], sta) and re.search(event[2], channel):
                res[key] = project[key]

        return res

    @classmethod
    def filter_time(cls, list_files, **kwargs):

        #filter the list output of filter_project_keys by trimed times

        result = []
        st1 = kwargs.pop('starttime', None)
        et1 = kwargs.pop('endtime', None)

        if st1 is None and et1 is None:
            for file in list_files:
                result.append(file[0])

        else:

            for file in list_files:
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

    def _parse_data_file(self, file: str, verbose: bool):

        try:
            header = read(file, headeronly=True)
        except TypeError:
            return [None, None]

        net = header[0].stats.network
        sta = header[0].stats.station
        chn = header[0].stats.channel
        key = f"{net}.{sta}.{chn}"
        data_map = [file, header[0].stats]
        if verbose:
            print("included in the project file ", file)

        return key, data_map

    def _convert2dict(self, data: Tuple[str, List[str]]):
        self._project = {}
        for name in data:
            if name[0] in self._project.keys() and name[0] is not None:
                self._project[name[0]].append([name[1][0], name[1][1]])

            elif name[0] not in self._project.keys() and name[0] is not None:
                self._project[name[0]] = [[name[1][0], name[1][1]]]

        return self._project

    def convert2dataframe(self, project):
        project_converted = []
        _names = project.keys()

        for name in _names:
            for i in range(len(project[name])):
                project_converted.append({
                    'id': name,
                    'fname': project[name][i][0],
                    'stats': project[name][i][1]
                })

        return pd.DataFrame.from_dict(project_converted)

    @staticmethod
    def get_now_files(project, date, stations_list, channel_list):
        date = UTCDateTime(date)

        selection = [".", stations_list, channel_list]

        _, files_path = MseedUtil.filter_project_keys(project, net=selection[0], station=selection[1],
                                                      channel=selection[2])
        start = date - 1300  # half and hour before
        end = date + 3 * 3600  # end 2 hours after
        files_path = MseedUtil.filter_time(list_files=files_path, starttime=start, endtime=end)
        return files_path

    @classmethod
    def get_metadata_files(cls, file):
        from obspy import read_inventory
        try:

            inv = read_inventory(file)

            return inv

        except IOError:

           return []

    @staticmethod
    def save_project(project, path):
        if isinstance(project, dict):
            try:
                file_to_store = open(path, "wb")
                pickle.dump(project, file_to_store)
                print("Succesfully saved project")
            except ProjectSaveFailed as e:
                print(f"Project couldn't be saved: {e}")

class ProjectSaveFailed(Exception):
    pass

class ObspyUtil:

    @staticmethod
    def real_write_station_file(Inventory: Inventory, working_directory):

        """
        Inventory: inventory object from Obspy
        working directory: destination folder where REAL is going to work
        """

        stations_file = os.path.join(working_directory, "station.dat")
        list_real_stations_coords = []
        list_real_stations_coords_done = []
        for network in Inventory:
            for station in network.stations:
                item_done = '\t'.join([network.code, station.code])
                if item_done not in list_real_stations_coords_done:
                    item = '\t'.join(
                        [str(station.longitude), str(station.latitude), network.code, station.code, "XXK",
                         str((station.elevation) / 1000)])
                    list_real_stations_coords_done.append(item_done)
                    list_real_stations_coords.append(item)

        with open(stations_file, 'w') as f:
            for item in list_real_stations_coords:
                f.write(f"{item}\n")
            f.close()

    @staticmethod
    def reads_hyp_to_origin(hyp_file_path: str) -> Origin:

        import warnings
        warnings.filterwarnings("ignore")

        """
        Reads an hyp file and returns the Obspy Origin.
        :param hyp_file_path: The file path to the .hyp file
        :return: An Obspy Origin
        """

        if os.path.isfile(hyp_file_path):
            cat = read_nll_performance.read_nlloc_hyp_ISP(hyp_file_path)
            event = cat[0]
            origin = event.origins[0]
            modified_origin_90, _, _ = computeOriginErrors(origin)
            origin.depth_errors["uncertainty"] = modified_origin_90['depth_errors'].uncertainty
            origin.origin_uncertainty.max_horizontal_uncertainty = modified_origin_90[
                'origin_uncertainty'].max_horizontal_uncertainty
            origin.origin_uncertainty.min_horizontal_uncertainty = modified_origin_90[
                'origin_uncertainty'].min_horizontal_uncertainty
            origin.origin_uncertainty.azimuth_max_horizontal_uncertainty = modified_origin_90[
                'origin_uncertainty'].azimuth_max_horizontal_uncertainty

        return origin