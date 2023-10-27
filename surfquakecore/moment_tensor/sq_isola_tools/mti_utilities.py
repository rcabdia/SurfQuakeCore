import os
import shutil
from obspy.taup import TauPyModel
import pandas as pd
from obspy.geodetics.base import gps2dist_azimuth, kilometer2degrees
from surfquake.seismogramInspector.signal_processing_advanced import get_rms_times


class MTIManager:

    def __init__(self, st, inv, lat0, lon0, depth, o_time, min_dist, max_dist, working_directory):
        """
        Manage MTI files for run isola class program.
        st: stream of seismograms
        in: inventory
        """
        self.__st = st
        self.__inv = inv
        self.lat = lat0
        self.lon = lon0
        self.depth = depth
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.working_directory = working_directory
        self.o_time = o_time
        self.model = TauPyModel(model="iasp91")
        self.check_rms = {}

    @staticmethod
    def __validate_file(file_path):
        if not os.path.isfile(file_path):
            raise FileNotFoundError("The file {} doesn't exist.".format(file_path))

    @staticmethod
    def __validate_dir(dir_path):
        if not os.path.isdir(dir_path):
            raise FileNotFoundError("The dir {} doesn't exist.".format(dir_path))

    @property
    def root_path(self):
        root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        self.__validate_dir(root_path)
        return root_path

    @property
    def get_stations_dir(self):
        stations_dir = os.path.join(self.root_path, "input")
        self.__validate_dir(stations_dir)
        return stations_dir

    def get_stations_index(self):

        ind = []
        file_list = []
        dist1 = []
        for tr in self.__st:
            net = tr.stats.network
            station = tr.stats.station
            channel = tr.stats.channel
            coords = self.__inv.get_coordinates(tr.id)
            lat = coords['latitude']
            lon = coords['longitude']
            if ind.count(station):
                pass
            else:
                [dist, _, _] = gps2dist_azimuth(self.lat, self.lon, lat, lon, a=6378137.0, f=0.0033528106647474805)

                item = '{net}:{station}::{channel}    {lat}    {lon}'.format(net=net,
                        station=station, channel=channel[0:2], lat=lat, lon=lon)

            # filter by distance
                if self.min_dist < self.max_dist and self.min_dist <= dist and dist <= self.max_dist:
                    # do the distance filter
                    ind.append(station)
                    file_list.append(item)
                    dist1.append(dist)
                    keydict = dict(zip(file_list, dist1))
                    file_list.sort(key=keydict.get)
                else:
                    # do not filter by distance
                    ind.append(station)
                    file_list.append(item)
                    dist1.append(dist)
                    keydict = dict(zip(file_list, dist1))
                    file_list.sort(key=keydict.get)

        self.stations_index = ind
        self.stream = self.sort_stream(dist1)

        deltas = self.get_deltas()
        data = {'item': file_list}

        df = pd.DataFrame(data, columns=['item'])
        outstations_path = os.path.join(self.working_directory, "stations.txt")
        df.to_csv(outstations_path, header=False, index=False)
        return self.stream, deltas

    def get_traces_participation(self, p_arrival_time, win_length, threshold=4, magnitude=None, distance=None):

        """
        Find which traces from self.stream are above RMS Threshold
        """

        for st in self.stream:
            for tr in st:

                try:
                    if p_arrival_time!=None:
                        pass
                    else:

                        coords = self.__inv.get_coordinates(tr.id)
                        lat = coords['latitude']
                        lon = coords['longitude']
                        [dist, _, _] = gps2dist_azimuth(self.lat, self.lon, lat, lon, a=6378137.0, f=0.0033528106647474805)
                        distance_degrees = kilometer2degrees(dist*1E-3)
                        arrivals = self.model.get_travel_times(source_depth_in_km=self.depth*1E-3,
                                                               distance_in_degree=distance_degrees)
                        p_arrival_time = self.o_time+arrivals[0].time

                    rms = get_rms_times(tr, p_arrival_time, win_length, freqmin=0.5, freqmax=8, win_threshold=30,
                                        magnitude=magnitude, distance=distance)
                    if rms >= threshold:
                        self.check_rms[tr.stats.network+"_" + tr.stats.station + "__" + tr.stats.channel] = True
                    else:
                        self.check_rms[tr.stats.network + "_" + tr.stats.station + "__" + tr.stats.channel] = False
                except:
                    self.check_rms[tr.stats.network + "_" + tr.stats.station + "__" + tr.stats.channel] = False


    def filter_mti_inputTraces(self, stations, stations_list):
        # guess that starts everything inside stations and stations_list to False
        # stations is a list
        # statiosn_list is a dictionary of dictionaries
        for key in self.check_rms.keys():

            if self.check_rms[key]:

                stations_list = self.__find_in_stations_list(stations_list, key)
                #stations = self.__find_in_stations(stations, key) #not necessary, it is doone automatically


        return stations, stations_list

    def __find_in_stations_list(self, stations_list, key_check):
        #stations_list is a dictionary with dictionaries inside
        key_search = 'use' + key_check[-1]
        key_check_find = key_check[0:-1]
        if key_check_find in stations_list.keys():
            stations_list[key_check_find][key_search] = self.check_rms[key_check]
        return stations_list

    def __find_in_stations(self, stations, key):
        # stations is a list of dictionaries
        network, code, loc, channelcode = key.split("_")
        for iter, index_dict in enumerate(stations):
            if index_dict["channelcode"] == channelcode[0:2] and index_dict["code"] == code and index_dict["network"] == network:
                if channelcode[-1] == "Z" or channelcode[-1] == 3:
                    stations[iter]["useZ"] = self.check_rms[key]
                elif channelcode[-1] == "N" or channelcode[-1] == "Y" or channelcode[-1] == 1:
                    stations[iter]["useN"] = self.check_rms[key]
                elif channelcode[-1] == "E" or channelcode[-1] == "X" or channelcode[-1] == 2:
                    stations[iter]["useN"] = self.check_rms[key]

        return stations

    def sort_stream(self, dist1):
        stream = []
        stream_sorted_order = []

        for station in self.stations_index:
            st2 = self.__st.select(station=station)
            stream.append(st2)

        # Sort by Distance
        stream_sorted = [x for _, x in sorted(zip(dist1, stream))]
        # Bayesian isola require ZNE order
        # reverse from E N Z --> Z N E
        # reverse from 1 2 Z --> Z 1 2

        # TODO REVERSE DEPENDS ON THE NAMING BUT ALWAYS OUTPUT MUST BE IN THE ORDER ZNE
        for stream_sort in stream_sorted:
            if "1" and "2" in stream_sort:
                stream_sorted_order.append(sorted(stream_sort, key=lambda x: (x.isnumeric(), int(x) if x.isnumeric() else x)))
            else:
                stream_sorted_order.append(stream_sort.reverse())

        return stream_sorted_order

    def get_deltas(self):
        deltas = []
        n = len(self.stream)
        for j in range(n):
            stream_unique = self.stream[j]
            delta_unique = stream_unique[0].stats.delta
            deltas.append(delta_unique)

        return deltas

    @staticmethod
    def move_files2workdir(src_dir, dest_dir):

        # getting all the files in the source directory
        files = os.listdir(src_dir)

        for fname in files:
            # copying the files to the
            # destination directory
            shutil.copy2(os.path.join(src_dir, fname), dest_dir)

    # def prepare_working_directory(self, green_path):
    #     # check that exists otherwise remove it
    #     if not os.path.exists(self.input_path):
    #         os.makedirs(self.input_path)
    #     # gather all files
    #     allfiles = os.listdir(green_path)
    #     # iterate on all files to move them to destination folder
    #     for f in allfiles:
    #         src_path = os.path.join(green_path, f)
    #         dst_path = os.path.join(self.input_path, f)
    #         os.rename(src_path, dst_path)

    def default_processing(self, paths, start, end):
        pass
        # start = UTCDateTime("2018-08-21T00:20:00.0")
        # end = UTCDateTime("2018-08-21T00:35:00.0")
        # o_time = "2018-08-21 00:28:57"  # '%Y-%m-%d %H:%M:%S'
        # st.trim(starttime=start, endtime=end)
        # f1 = 0.01
        # f2 = 0.02
        # f3 = 0.35 * st[0].stats.sampling_rate
        # f4 = 0.40 * st[0].stats.sampling_rate
        # pre_filt = (f1, f2, f3, f4)
        # st.detrend(type='constant')
        # # ...and the linear trend
        # st.detrend(type='linear')
        # st.taper(max_percentage=0.05)
        # st.remove_response(inventory=inv, pre_filt=pre_filt, output="VEL", water_level=60)
        # st.detrend(type='linear')
        # st.taper(max_percentage=0.05)