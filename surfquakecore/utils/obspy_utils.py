import os
import pickle
import re
from multiprocessing import Pool
import pandas as pd
from obspy import read


class MseedUtil:

    def __init__(self, robust=True, **kwargs):

        self.start = kwargs.pop('starttime', [])
        self.end = kwargs.pop('endtime', [])
        self.obsfiles = []
        self.pos_file = []
        self.robust = robust
        self.use_ind_files = False


    @classmethod
    def load_project(cls, file: str):
        project = {}
        try:
            project = pickle.load(open(file, "rb"))

        except:
            pass
        return project

    def search_indiv_files(self, list_files: list):

        self.use_ind_files = True
        self.list_files = list_files
        with Pool(processes=os.cpu_count()) as pool:
            returned_list = pool.map(self.create_dict, range(len(self.list_files)))

        project = self.convert2dict(returned_list)
        self.use_ind_files = False

        return project

    def search_files(self, rooth_path: str):
        project = {}
        self.search_file = []
        for top_dir, sub_dir, files in os.walk(rooth_path):
            for file in files:
                self.search_file.append(os.path.join(top_dir, file))

        with Pool(processes=os.cpu_count()) as pool:
            returned_list = pool.map(self.create_dict, range(len(self.search_file)))

        project = self.convert2dict(returned_list)

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

    def create_dict(self, i):
        key = None
        data_map = None

        try:
            if self.use_ind_files:
                header = read(self.list_files[i], headeronly=True)
                print(self.list_files[i])
                net = header[0].stats.network
                sta = header[0].stats.station
                chn = header[0].stats.channel
                key = net + "." + sta + "." + chn
                data_map = [self.list_files[i], header[0].stats]
            else:
                header = read(self.search_file[i], headeronly=True)
                print(self.search_file[i])
                net = header[0].stats.network
                sta = header[0].stats.station
                chn = header[0].stats.channel
                key = net + "." + sta + "." + chn
                data_map = [self.search_file[i], header[0].stats]

        except:
            pass

        return [key, data_map]

    def convert2dict(self, project):
        project_converted = {}
        for name in project:
            if name[0] in project_converted.keys() and name[0] is not None:
                project_converted[name[0]].append([name[1][0],name[1][1]])

            elif name[0] not in project_converted.keys() and name[0] is not None:
                project_converted[name[0]] = [[name[1][0],name[1][1]]]

        return project_converted

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