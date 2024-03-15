# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: run_nll.py
# Program: surfQuake & ISP
# Date: January 2024
# Purpose: Manage Event Locator
# Author: Roberto Cabieces & Thiago C. Junqueira
#  Email: rcabdia@roa.es
# --------------------------------------------------------------------


import math
import os
import shutil
import stat
from trace import Trace
from typing import Union, List, Optional
import numpy as np
import obspy
import pandas as pd
from obspy import UTCDateTime, read, Inventory
from obspy.core.stream import Stream
from obspy.geodetics.base import gps2dist_azimuth, kilometer2degrees
from obspy.signal.trigger import trigger_onset
from obspy.taup import TauPyModel

from surfquakecore.moment_tensor.structures import MomentTensorInversionConfig


class MTIManager:

    def __init__(self,   stream: Stream, inventory:  Inventory,
                 working_directory: str, mti_config:  MomentTensorInversionConfig):
        """
        Manage MTI files for run isola class program.

        Args:
            stream: an obspy Stream
            inventory: an obspy Inventory
            working_directory: A directory where some files will be saved temporary.
            mti_config: A dataclass instance of MomentTensorInversionConfig
        """

        self.__st = stream
        self.__inv = inventory

        self.lat = mti_config.latitude
        self.lon = mti_config.longitude
        self.depth = mti_config.depth_km
        self.o_time = UTCDateTime(mti_config.origin_date)
        self.min_dist = mti_config.inversion_parameters.min_dist * 1000.
        self.max_dist = mti_config.inversion_parameters.max_dist * 1000.
        self.magnitude = mti_config.magnitude
        self.threshold = mti_config.signal_processing_parameters.rms_thresh

        self.working_directory = working_directory
        self.model = TauPyModel(model="iasp91")
        self.check_rms = {}

        self.stations_index: List[str] = []
        self.streams: Optional[List[List[Trace]]] = None

    def get_stations_index(self):

        self.stations_index = []
        file_list = []
        dist1 = []
        for tr in self.__st:
            net = tr.stats.network
            station = tr.stats.station
            channel = tr.stats.channel
            coords = self.__inv.get_coordinates(tr.id)
            lat = coords['latitude']
            lon = coords['longitude']
            if station not in self.stations_index:

                dist, _, _ = gps2dist_azimuth(self.lat, self.lon, lat, lon, a=6378137.0, f=0.0033528106647474805)

                item = '{net}:{station}::{channel}    {lat}    {lon}'.format(net=net,
                                                                             station=station, channel=channel[0:2],
                                                                             lat=lat, lon=lon)

                # filter by distance
                if self.min_dist < self.max_dist and self.min_dist <= dist <= self.max_dist:
                    # do the distance filter
                    self.stations_index.append(station)
                    file_list.append(item)
                    dist1.append(dist)

        # sort list of streams according to distance
        self.streams = self.sort_stream(dist1)

        file_list.sort(key=dict(zip(file_list, dist1)).get)
        data = {'item': file_list}

        df = pd.DataFrame(data, columns=['item'])
        outstations_path = os.path.join(self.working_directory, "stations.txt")
        df.to_csv(outstations_path, header=False, index=False)

        return self.streams, self.get_deltas()

    def get_participation(self):

        """
        Find which traces from self.stream are above RMS Threshold
        """

        for st in self.streams:
            for tr in st:

                coords = self.__inv.get_coordinates(tr.id)
                lat = coords['latitude']
                lon = coords['longitude']
                [dist, _, _] = gps2dist_azimuth(self.lat, self.lon, lat, lon, a=6378137.0, f=0.0033528106647474805)
                distance_degrees = kilometer2degrees(dist * 1E-3)
                arrivals = self.model.get_travel_times(source_depth_in_km=self.depth * 1E-3,
                                                       distance_in_degree=distance_degrees)
                p_arrival_time = self.o_time + arrivals[0].time

                if distance_degrees <= 5:  # Go for waveform duration of strong motion.

                    rms = self.get_rms_times(tr, p_arrival_time, dist * 1E-3, self.magnitude, freqmin=0.5, freqmax=8)

                else:
                    # Go for estimate earthquake duration from expected Rayleigh Wave
                    rms = self.get_rms_times(tr, p_arrival_time, dist * 1E-3, self.magnitude, freqmin=0.05, freqmax=1.0)

                if rms >= self.threshold:
                    self.check_rms[tr.stats.network + "_" + tr.stats.station + "__" + tr.stats.channel] = True
                else:
                    self.check_rms[tr.stats.network + "_" + tr.stats.station + "__" + tr.stats.channel] = False

    def filter_mti_inputTraces(self, stations, stations_list):
        # guess that starts everything inside stations and stations_list to False
        # stations is a list
        # stations_list is a dictionary of dictionaries
        for key in self.check_rms.keys():

            if self.check_rms[key]:
                stations_list = self.__find_in_stations_list(stations_list, key)
                # stations = self.__find_in_stations(stations, key) #not necessary, it is done automatically

        return stations, stations_list

    def filter_stations_stream(self, st: Stream, stations_input: list):
        stations_check = []
        list_to_remove = []
        for item in stations_input:
            if item["code"] not in stations_check:
                stations_check.append(item["code"])

        for i, stream in enumerate(st):
            for tr in stream:
                if tr.stats.station not in stations_check:
                    list_to_remove.append(i)
                    stations_check.append(tr.stats.station)
        st_new = [value for index, value in enumerate(st) if index not in list_to_remove]

        # renew deltas just in case
        deltas = [st[0].stats.delta for st in st_new]

        return st_new, deltas

    def __find_in_stations_list(self, stations_list, key_check):
        # stations_list is a dictionary with dictionaries inside
        key_search = 'use' + key_check[-1]
        key_check_find = key_check[0:-1]
        if key_check_find in stations_list.keys():
            stations_list[key_check_find][key_search] = self.check_rms[key_check]
        return stations_list

    def __find_in_stations(self, stations, key):
        # stations is a list of dictionaries
        network, code, loc, channelcode = key.split("_")
        for _iter, index_dict in enumerate(stations):
            if (index_dict["channelcode"] == channelcode[0:2] and index_dict["code"] == code and
                    index_dict["network"] == network):
                if channelcode[-1] == "Z" or channelcode[-1] == 3:
                    stations[_iter]["useZ"] = self.check_rms[key]
                elif channelcode[-1] == "N" or channelcode[-1] == "Y" or channelcode[-1] == 1:
                    stations[_iter]["useN"] = self.check_rms[key]
                elif channelcode[-1] == "E" or channelcode[-1] == "X" or channelcode[-1] == 2:
                    stations[_iter]["useN"] = self.check_rms[key]

        return stations

    def sort_stream(self, dist1):

        stream = (self.__st.select(station=station) for station in self.stations_index)

        # Sort by Distance
        stream_sorted = (x for _, x in sorted(zip(dist1, stream)))
        # Bayesian isola require ZNE order
        # reverse from E N Z --> Z N E
        # reverse from X Y Z --> Z Y X
        # reverse from 1 2 Z --> Z 1 2

        def _inner_sort(value: Stream):
            # TODO How is it possible 1 or 2 be in a Stream ??
            if "1" in value and "2" in value:
                # noinspection PyTypeChecker
                return sorted(value, key=lambda x: (x.isnumeric(), int(x) if x.isnumeric() else x))

            else:
                return value.reverse()

        # TODO REVERSE DEPENDS ON THE NAMING BUT ALWAYS OUTPUT MUST BE IN THE ORDER ZNE
        return [_inner_sort(stream_sort) for stream_sort in stream_sorted]

    def get_deltas(self):
        return [st[0].stats.delta for st in self.streams]

    def copy_to_working_directory(self, src_dir: Union[str, os.PathLike[str]]):

        for file in os.listdir(src_dir):
            shutil.copy(os.path.join(src_dir, file), self.working_directory)
            if file == "gr_xyz" or file == "elemse":
                file_path = os.path.join(self.working_directory, file)
                st = os.stat(file_path)
                os.chmod(file_path, st.st_mode | stat.S_IEXEC)

    @classmethod
    def default_processing(cls, files_path, origin_time,
                           inventory, output_directory, regional=True, remove_response=True,
                           save_stream_plot=True):
        all_traces = []
        origin_time = UTCDateTime(origin_time)

        if regional:
            dt_noise = 10. * 60.
            dt_signal = 10. * 60.
        else:
            dt_noise = 10. * 60
            dt_signal = 60. * 60.

        start = origin_time - dt_noise
        end = origin_time + dt_signal
        for file in files_path:
            try:
                st = read(file)
                tr = st[0]
                tr.trim(starttime=start, endtime=end)

                # TODO: It is not still checked the fill_gaps functionality
                tr = cls.fill_gaps(tr)
                if tr is not None:
                    f1 = 0.01
                    f2 = 0.02
                    f3 = 0.35 * tr.stats.sampling_rate
                    f4 = 0.40 * tr.stats.sampling_rate
                    pre_filt = (f1, f2, f3, f4)
                    tr.detrend(type='constant')
                    # # ...and the linear trend
                    tr.detrend(type='linear')
                    tr.taper(max_percentage=0.05)
                    if remove_response:
                        tr.remove_response(inventory=inventory, pre_filt=pre_filt, output="VEL", water_level=60)
                        tr.detrend(type='linear')
                        tr.taper(max_percentage=0.05)
                    all_traces.append(tr)
            except:
                print("It cannot be processed file ", file)
        st = Stream(traces=all_traces)
        st.merge()
        if save_stream_plot:
            output_dir = os.path.join(output_directory, "stream_raw.png")
            st.plot(outfile=output_dir, size=(800, 600))

        return st

    def get_rms_times(self, tr: Trace, p_arrival_time: UTCDateTime, distance_km, magnitude,
                      freqmin=0.5, freqmax=8) -> float:

        """
        Get trace cut times between P arrival and end of envelope coda plus earthquake duration
        """

        tr_env = tr.copy()
        # remove the mean...
        tr_env.detrend(type='constant')
        # ...and the linear trend...
        tr_env.detrend(type='linear')
        # ...filter
        tr_env.taper(max_percentage=0.05)
        tr_env.filter(type='bandpass', freqmin=freqmin, freqmax=freqmax)
        tr_env.data = self._envelope(tr_env.data)
        # smooth
        tr_env.filter(type='lowpass', freq=0.15)
        tr_noise = tr_env.copy()
        tr_signal = tr_env.copy()

        if distance_km <= 150:

            t_duration = self._duration(magnitude, distance_km, component="vertical", S=1)
            t2 = p_arrival_time + t_duration
            win_length_noise = 60
            t1 = p_arrival_time - win_length_noise

        else:
            delta_t = distance_km / 4.0  # rough aproximation arrival of surface wave
            t2 = p_arrival_time + delta_t + 1.5 * delta_t
            win_length_noise = 5 * 60
            t1 = p_arrival_time - win_length_noise

        tr_noise.trim(starttime=t1, endtime=p_arrival_time, pad=False, fill_value=0)
        tr_signal.trim(starttime=p_arrival_time, endtime=t2, pad=False, fill_value=0)

        return self._compute_snr(tr_noise, tr_signal)

    @staticmethod
    def _compute_snr(trace_noise, trace_signal):

        rmsnoise = np.sqrt(np.power(trace_noise.data, 2).sum())
        rmsSignal = np.sqrt(np.power(trace_signal.data, 2).sum())
        sn_ratio = rmsSignal / rmsnoise

        return sn_ratio

    @staticmethod
    def _duration(magnitude, distance, component="vertical", S=1):
        # Trifunac and  Brady 1987
        # A new definition of strong motion duration and comparison with other definitions
        # N.A. THEOFANOPULOS AND M. WATABE 1989
        # S = 0, 1, 2 hard, intermediate and soft
        # distance in km
        if component == "vertical":
            a = -124.5
            b = 120.7
            c = 0.019
            d = 0.08641
            e = 4.352

        if component == "horizontal":
            a = 2.201
            b = 0.02489
            c = 0.860
            d = 0.05335
            e = 2.883

        D = a + b * np.exp(c * magnitude) + distance * d + e * S
        t = 0.05 * D + D  # seconds

        return t

    @staticmethod
    def _envelope(data):

        N = len(data)
        D = 2 ** math.ceil(math.log2(N))
        z = np.zeros(D - N)
        data = np.concatenate((data, z), axis=0)

        # Necesary padding with zeros
        data_envelope = obspy.signal.filter.envelope(data)
        data_envelope = data_envelope[0:N]

        return data_envelope

    @classmethod
    def fill_gaps(cls, tr, tol=5):

        tol_seconds_percentage = int((tol / 100) * len(tr.data)) * tr.stats.delta
        st = Stream(traces=tr)
        gaps = st.get_gaps()

        if len(gaps) > 0 and cls._check_gaps(gaps, tol_seconds_percentage):
            st.print_gaps()
            st.merge(fill_value="interpolate", interpolation_samples=-1)
            return st[0]

        elif len(gaps) > 0 and not cls._check_gaps(gaps, tol_seconds_percentage):
            st.print_gaps()
            return None
        elif len(gaps) == 0 and cls._check_gaps(gaps, tol_seconds_percentage):
            return tr
        else:
            return tr

    @staticmethod
    def _check_gaps(gaps, tol):
        time_gaps = []
        for i in gaps:
            time_gaps.append(i[6])

        sum_total = sum(time_gaps)

        return sum_total <= tol
