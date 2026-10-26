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
from obspy import Stream, Trace
from typing import Union, List, Optional
import numpy as np
import obspy
import pandas as pd
from obspy import UTCDateTime, read, Inventory
from obspy.signal.rotate import rotate_ne_rt
from obspy.geodetics.base import gps2dist_azimuth, kilometer2degrees
# from obspy.signal.trigger import trigger_onset
from obspy.taup import TauPyModel
from surfquakecore.moment_tensor.structures import MomentTensorInversionConfig


class MTIManager:

    # Role of each component letter (last character of the channel code) in the MTI pipeline.
    # BayesISOLA works with 3 components per station ordered Z, N, E. In a "Z,1,2" or "Z,Y,X" station the
    # second and third channels play the role of N and E (see sort_stream and
    # InversionDataManager.set_use_components), so 1/Y -> useN and 2/X -> useE.
    _COMPONENT_FLAG = {"Z": "useZ",
                       "N": "useN", "1": "useN", "Y": "useN",
                       "E": "useE", "2": "useE", "X": "useE"}

    # Possible (first horizontal, second horizontal) naming schemes
    _HORIZONTAL_PAIRS = (("N", "E"), ("1", "2"), ("Y", "X"))

    def __init__(self, stream: Stream, inventory: Inventory,
                 working_directory: str, mti_config: MomentTensorInversionConfig):
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
            if self._is_dummy(tr):
                # Dummy traces have no metadata: the coordinates of a station always come from a real trace
                continue
            net = tr.stats.network
            station = tr.stats.station
            channel = tr.stats.channel
            if station not in self.stations_index:
                coords = self.__inv.get_coordinates(tr.id)
                lat = coords['latitude']
                lon = coords['longitude']

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
        Find which traces from self.stream are above RMS Threshold.
        Dummy traces (see fill_stream_with_inferred_channels) never participate.
        """

        for st in self.streams:
            for tr in st:

                key = self._station_key(tr) + tr.stats.channel[-1]

                if self._is_dummy(tr):
                    self.check_rms[key] = False
                    continue

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

                self.check_rms[key] = bool(rms >= self.threshold)

    def filter_mti_inputTraces(self, stations, stations_list):
        # stations is a list
        # stations_list is a dictionary of dictionaries (InversionDataManager.stations_index)
        # The use{Z,N,E} flags start as False and are only switched on for the traces above the threshold.
        for key, participates in self.check_rms.items():

            if participates:
                stations_list = self.__find_in_stations_list(stations_list, key)

        return stations, stations_list

    def disable_dummy_components(self, stations_list):
        """
        Force use{Z,N,E} = False for every component that is a dummy trace, whatever the way the flags
        were set (RMS threshold or manual selection with map_stations).
        Must be called after get_stations_index() and after the flags have been set.

        :param stations_list: dictionary of dictionaries (InversionDataManager.stations_index)
        """
        for st in self.streams or []:
            for tr in st:
                if not self._is_dummy(tr):
                    continue
                flag = self._COMPONENT_FLAG.get(tr.stats.channel[-1])
                entry = stations_list.get(self._station_key(tr))
                if flag is not None and entry is not None:
                    entry[flag] = False

        return stations_list

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
        # key_check = "<net>_<station>__<band/instrument><component>", e.g. "XX_AAA__HH1"
        station_key, component = key_check[:-1], key_check[-1]
        flag = self._COMPONENT_FLAG.get(component)  # Z -> useZ, 1/Y/N -> useN, 2/X/E -> useE
        if flag is not None and station_key in stations_list:
            stations_list[station_key][flag] = self.check_rms[key_check]

        return stations_list

    def sort_stream(self, dist1):

        stream = (self.__st.select(station=station) for station in self.stations_index)

        # Sort by Distance
        stream_sorted = (x for _, x in sorted(zip(dist1, stream)))

        # Bayesian isola require ZNE order
        # reverse from E N Z --> Z N E
        # reverse from X Y Z --> Z Y X
        # reverse from 1 2 Z --> Z 1 2

        def _sort_stream_by_channels(stream):
            """
            Sorts a Stream object based on the channel configuration.

            Parameters:
                stream (Stream): The ObsPy Stream object to sort.

            Returns:
                Stream: The sorted Stream object.
            """
            # Extract channel suffixes to determine configuration
            channel_suffixes = [tr.stats.channel[-1] for tr in stream]

            # Determine the configuration based on channel suffixes
            if "1" in channel_suffixes and "2" in channel_suffixes:
                channel_order = ["Z", "1", "2"]  # Z, 1, 2 configuration
            elif "N" in channel_suffixes and "E" in channel_suffixes:
                channel_order = ["Z", "N", "E"]  # Z, N, E configuration
            elif "Y" in channel_suffixes and "X" in channel_suffixes:
                channel_order = ["Z", "Y", "X"]  # Z, Y, X configuration
            else:
                # Fallback: Leave the stream as is
                return stream

            # Sort the stream based on the determined channel order
            sorted_stream = Stream(
                sorted(stream, key=lambda tr: channel_order.index(tr.stats.channel[-1]))
            )

            return sorted_stream

        # TODO REVERSE DEPENDS ON THE NAMING BUT ALWAYS OUTPUT MUST BE IN THE ORDER ZNE
        return [_sort_stream_by_channels(stream_sort) for stream_sort in stream_sorted]

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

                # TODO: It is not TOTALLY still checked the fill_gaps functionality
                tr = cls.fill_gaps(tr)
                if tr is not None:
                    f1 = 0.010
                    f2 = 0.014
                    f3 = 0.35 * tr.stats.sampling_rate
                    f4 = 0.40 * tr.stats.sampling_rate
                    pre_filt = (f1, f2, f3, f4)
                    tr.detrend(type='constant')
                    # # ...and the linear trend
                    tr.detrend(type='linear')
                    tr.taper(max_percentage=0.05)
                    if remove_response:
                        tr.remove_response(inventory=inventory, pre_filt=pre_filt, output="VEL", water_level=80)
                        tr.detrend(type='linear')
                        tr.taper(max_percentage=0.05)
                    all_traces.append(tr)
            except:
                print("It cannot be processed file ", file)
        st = Stream(traces=all_traces)
        st.merge()
        try:
            st = cls.rotate_to_ne(st, inventory)
        except Exception as error:
            print('Rotating Stream Error: ' + repr(error))
            print(error)

        if save_stream_plot:
            output_dir = os.path.join(output_directory, "stream_raw.png")
            st.plot(outfile=output_dir, size=(800, 600))

        return st

    @classmethod
    def rotate_to_ne(cls, stream, inventory, min_angle=5.0):

        """
        Rotates the horizontal components of every station to the N/E directions if needed.

        The two horizontal channels of a station ("1","2" or "Y","X") are rotated using the azimuth of each
        channel in the inventory. The data are replaced IN PLACE and the channel names are NOT changed:
        afterwards the "1"/"Y" trace contains North and the "2"/"X" trace contains East (the metadata and
        BayesISOLA keep working with the original names). Stations that already have N/E channels, or whose
        horizontals are already oriented N/E within ``min_angle`` degrees, are left untouched.

        Rotated traces are flagged with ``stats.mti_rotated = True``, so calling this method again on the
        same traces (e.g. a preprocessed stream reused for several events) does not rotate them twice.

        Parameters:
            stream (Stream): ObsPy Stream containing multiple traces.
            inventory (Inventory): ObsPy Inventory containing metadata for traces.
            min_angle (float): tolerance in degrees to consider a channel already oriented N/E.

        Returns:
            Stream: Original (sorted) stream with the rotated data.
        """

        # group by (network, station, location, band/instrument code) -> {component letter: [traces]}
        groups = {}
        for trace in stream:
            s = trace.stats
            key = (s.network, s.station, s.location, s.channel[:2])
            groups.setdefault(key, {}).setdefault(s.channel[-1], []).append(trace)

        for (net, sta, loc, band), comps in groups.items():
            for first, second in (("1", "2"), ("Y", "X")):
                if first in comps and second in comps:
                    if len(comps[first]) > 1 or len(comps[second]) > 1:
                        print(f"Rotation skipped for {net}.{sta}.{loc}.{band}: more than one trace per component")
                    else:
                        cls._rotate_pair(comps[first][0], comps[second][0], inventory, min_angle)
                    break

        return stream.sort()

    @staticmethod
    def _angle_diff(a, b):
        """Smallest absolute difference between two angles, in degrees"""
        return abs((a - b + 180.0) % 360.0 - 180.0)

    @classmethod
    def _rotate_pair(cls, tr1, tr2, inventory, min_angle=5.0):
        """
        Replaces, in place, the data of two horizontal traces by the North (tr1) and East (tr2) components.
        tr1 is the "1"/"Y" channel and tr2 the "2"/"X" channel. Names are kept.

        Each channel records  c_i = N * cos(az_i) + E * sin(az_i)  (az = azimuth from North, clockwise), so
        the 2x2 system is solved for N and E. It works for any pair of non-parallel horizontal channels
        (not only for orthogonal right-handed ones).

        Returns:
            bool: True if the data were rotated.
        """
        pair_id = f"{tr1.id} / {tr2.id}"

        if tr1.stats.get("mti_rotated", False) or tr2.stats.get("mti_rotated", False):
            return False  # already rotated

        try:
            meta1 = inventory.get_channel_metadata(tr1.id, datetime=tr1.stats.starttime)
            meta2 = inventory.get_channel_metadata(tr2.id, datetime=tr2.stats.starttime)
        except Exception as error:  # ObsPy raises a plain Exception if the channel is not in the inventory
            print(f"Rotation skipped for {pair_id}: {error}")
            return False

        az1, az2 = meta1.get("azimuth"), meta2.get("azimuth")
        dip1, dip2 = meta1.get("dip"), meta2.get("dip")
        if az1 is None or az2 is None or dip1 is None or dip2 is None:
            print(f"Rotation skipped for {pair_id}: azimuth or dip missing in the inventory")
            return False

        if abs(dip1) > 5 or abs(dip2) > 5:
            return False  # not horizontal channels

        if cls._angle_diff(az1, 0.0) < min_angle and cls._angle_diff(az2, 90.0) < min_angle:
            return False  # already oriented N/E

        s1, s2 = tr1.stats, tr2.stats
        if (s1.npts != s2.npts or not np.isclose(s1.sampling_rate, s2.sampling_rate)
                or abs(s1.starttime - s2.starttime) > 0.5 * s1.delta):
            print(f"Rotation skipped for {pair_id}: traces differ in npts, sampling rate or start time")
            return False

        if np.ma.is_masked(tr1.data) or np.ma.is_masked(tr2.data):
            print(f"Rotation skipped for {pair_id}: traces have gaps (masked data)")
            return False

        a1, a2 = np.radians(az1), np.radians(az2)
        det = np.sin(a2 - a1)
        if abs(det) < 1E-3:
            print(f"Rotation skipped for {pair_id}: both channels have the same orientation")
            return False

        d1 = tr1.data.astype(np.float64)
        d2 = tr2.data.astype(np.float64)
        north = (d1 * np.sin(a2) - d2 * np.sin(a1)) / det
        east = (d2 * np.cos(a1) - d1 * np.cos(a2)) / det

        print("Rotating Traces: ", tr1.id, tr2.id)
        tr1.data = north
        tr2.data = east
        tr1.stats.mti_rotated = True
        tr2.stats.mti_rotated = True

        return True

    @classmethod
    def rotate_to_ne_change_name(cls, stream, inventory):
        """
        Rotates horizontal components in a stream to NE orientation if needed.

        Parameters:
            stream (Stream): ObsPy Stream containing multiple traces.
            inventory (Inventory): ObsPy Inventory containing metadata for traces.
            save_path (str): Optional path to save the rotated stream.

        Returns:
            Stream: Original stream with rotated traces substituted.
        """
        rotated_stream = obspy.Stream()  # Store rotated traces
        rotated_ids = set()  # Track rotated trace IDs for removal

        for trace in stream:
            trace_id = trace.id

            # Skip if already processed
            if trace_id in rotated_ids:
                continue

            # Retrieve station and channel information
            station = trace.stats.station
            network = trace.stats.network
            channel = trace.stats.channel
            location = trace.stats.location

            # Find corresponding station metadata in the inventory
            try:
                station_metadata = inventory.get_channel_metadata(trace.id)
            except KeyError:
                continue  # Skip if no metadata is available

            # Extract azimuth and dip
            azimuth = station_metadata['azimuth']
            dip = station_metadata['dip']

            # Skip non-horizontal traces
            if not (-5 <= dip <= 5):
                continue

            # Skip if already in NE orientation
            if channel[-1] in ["N", "E"]:
                continue

            # Find paired horizontal channel
            paired_channel = None
            for tr in stream:
                if (
                    tr.stats.station == station and
                    tr.stats.network == network and
                    tr.stats.location == location and
                    tr.stats.channel != channel and
                    tr.stats.channel[-1] in ["1", "2", "R", "T"]
                ):
                    paired_channel = tr
                    break

            if not paired_channel:
                continue  # Skip if no pair found

            # Retrieve metadata for paired channel
            try:
                paired_metadata = inventory.get_channel_metadata(paired_channel.id)
                paired_azimuth = paired_metadata["azimuth"]
            except KeyError:
                continue

            # Rotate if azimuth difference is significant
            if np.abs(azimuth) >= 5:
                # Extract data
                tr1_data = trace.data
                tr2_data = paired_channel.data

                # Perform RT -> NE rotation
                n_data, e_data = rotate_ne_rt(tr1_data, tr2_data, azimuth)

                # Create rotated traces
                n_trace = trace.copy()
                n_trace.data = n_data
                n_trace.stats.channel = channel[:-1] + "N"

                e_trace = paired_channel.copy()
                e_trace.data = e_data
                e_trace.stats.channel = channel[:-1] + "E"

                # Add rotated traces to the rotated stream
                rotated_stream.append(n_trace)
                rotated_stream.append(e_trace)

                # Mark as processed
                rotated_ids.add(trace_id)
                rotated_ids.add(paired_channel.id)

        # Remove original traces that were rotated
        stream.traces = [tr for tr in stream if tr.id not in rotated_ids]

        # Add rotated traces to the original stream
        stream += rotated_stream

        stream = stream.sort()
        return stream


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

    @staticmethod
    def _is_dummy(trace) -> bool:
        """True for the traces created by fill_stream_with_inferred_channels"""
        return bool(trace.stats.get("mti_dummy", False))

    @staticmethod
    def _station_key(trace) -> str:
        """
        Key of the station in InversionDataManager.stations_index: '<net>_<station>_<location>_<band/instrument>'.
        The location is always empty there because stations.txt (see get_stations_index) does not write it.
        """
        return f"{trace.stats.network}_{trace.stats.station}__{trace.stats.channel[:2]}"

    def fill_stream_with_inferred_channels(self, amplitude_ratio: float = 1E-3, seed: int = 0) -> None:
        """
        Completes every station of the stream to three components (Z + two horizontals), the 3-trace
        structure BayesISOLA expects. Missing components are filled with low-amplitude white-noise dummy traces.

        - Dummy traces are flagged with ``stats.mti_dummy = True``. They never participate in the inversion
          (see get_participation and disable_dummy_components) and they are never looked up in the inventory.
        - The horizontal naming (N/E, 1/2 or Y/X) follows the traces the station already has; a station with
          only the vertical follows the naming that is most frequent in the stream.
        - Amplitude of a dummy: ``amplitude_ratio`` x the maximum of a real trace of the station (the vertical
          when available). The same ``seed`` always gives the same noise.
        - Only one (location, band/instrument) set per station is completed: the one of the first trace of the
          station, which is the one get_stations_index() uses.

        The result replaces ``self.__st``; nothing is returned.
        """
        rng = np.random.default_rng(seed)

        # groups: (net, sta, loc, band) -> traces.   primary: (net, sta) -> group of the first trace
        groups = {}
        primary = {}
        for tr in self.__st:
            s = tr.stats
            group_key = (s.network, s.station, s.location, s.channel[:2])
            groups.setdefault(group_key, []).append(tr)
            primary.setdefault((s.network, s.station), group_key)

        default_pair = self._dominant_horizontal_pair(groups.values())
        filled = Stream(traces=list(self.__st))

        for (net, sta), group_key in primary.items():
            location, band = group_key[2], group_key[3]

            others = [k for k in groups if k[:2] == (net, sta) and k != group_key]
            if others:
                print(f"Station {net}.{sta}: more than one location/channel set found "
                      f"({', '.join(f'{k[2]}.{k[3]}' for k in [group_key] + others)}). "
                      f"Only {location}.{band} is completed")

            traces = groups[group_key]
            existing = {tr.stats.channel[-1] for tr in traces}
            missing = sorted(self._expected_components(existing, default_pair) - existing)
            if not missing:
                continue

            # reference trace: the vertical if available
            ref = next((tr for tr in traces if tr.stats.channel[-1] == "Z"), traces[0])
            data = np.nan_to_num(np.ma.filled(ref.data.astype(np.float64), 0.0))
            peak = float(np.max(np.abs(data))) if data.size else 0.0
            max_amplitude = amplitude_ratio * peak if peak > 0 else 1E-12

            for component in missing:
                header = {"network": net, "station": sta, "location": location, "channel": band + component,
                          "sampling_rate": ref.stats.sampling_rate, "starttime": ref.stats.starttime}
                dummy = Trace(data=self._generate_white_noise(ref.stats.npts, max_amplitude, rng), header=header)
                dummy.stats.mti_dummy = True
                filled.append(dummy)
                print(f"Station {net}.{sta}: component {band + component} missing, dummy trace added")

        self.__st = filled

    @classmethod
    def _dominant_horizontal_pair(cls, groups_traces):
        """
        Horizontal naming scheme used by most of the stations that have both horizontals.
        N/E is returned if there is a tie or if no station has both horizontals.
        """
        suffix_sets = [{tr.stats.channel[-1] for tr in traces} for traces in groups_traces]
        votes = [sum(set(pair) <= suffixes for suffixes in suffix_sets) for pair in cls._HORIZONTAL_PAIRS]
        return cls._HORIZONTAL_PAIRS[int(np.argmax(votes))]

    @classmethod
    def _expected_components(cls, existing, default_pair):
        """
        Components a station should have, given the ones it has.

        Parameters:
            existing (set): last letters of the channels the station has, e.g. {"Z", "1"}.
            default_pair (tuple): horizontal naming used when the station has no horizontal at all.

        Returns:
            set: expected components, e.g. {"Z", "1", "2"}. If the existing channels do not fit any known
            naming scheme (e.g. "N" together with "1") the existing ones are returned: nothing is guessed.
        """
        for pair in cls._HORIZONTAL_PAIRS:
            if existing & set(pair):
                expected = {"Z", *pair}
                break
        else:
            expected = {"Z", *default_pair}

        if not existing <= expected:
            print(f"Unrecognised channel naming {sorted(existing)}: no dummy traces added")
            return set(existing)

        return expected

    def _generate_white_noise(self, length, max_amplitude, rng=None):
        """
        Generates a vector of white noise.

        Parameters:
            length (int): Length of the vector (number of samples).
            max_amplitude (float): Maximum amplitude of the noise (absolute value).
            rng (np.random.Generator): optional, for reproducible noise.

        Returns:
            np.ndarray: A vector of white noise.
        """
        rng = np.random.default_rng() if rng is None else rng

        # Generate random values from a normal distribution (mean=0, std=1)
        noise = rng.standard_normal(length)

        # Scale the noise to have the desired maximum amplitude
        return max_amplitude * noise / np.max(np.abs(noise))

