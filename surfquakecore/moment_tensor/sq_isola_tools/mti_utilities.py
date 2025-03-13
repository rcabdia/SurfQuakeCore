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
    def rotate_to_ne(cls, stream, inventory):

        """
        Rotates horizontal components in a stream to NE orientation if needed.

        Parameters:
            stream (Stream): ObsPy Stream containing multiple traces.
            inventory (Inventory): ObsPy Inventory containing metadata for traces.
            save_path (str): Optional path to save the rotated stream.

        Returns:
            Stream: Original stream with rotated traces substituted.
        """

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
                        tr.stats.channel[-1] in ["1", "2", "R", "T", "Y", "X"]
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


                # Extract correct azimuth from north:
                if trace.stats.channel in ["1", "R",  "Y"]:
                    pass
                elif trace.stats.channel in ["2", "T",  "X"]:
                    azimuth = paired_azimuth


                # Extract data
                tr1_data = trace.data
                tr2_data = paired_channel.data

                # Perform RT -> NE rotation
                print("Rotating Traces: ", trace.id, paired_channel.id)
                baz = azimuth+180
                if baz >= 360:
                    baz = baz-360
                if trace.stats.channel[-1] in ["1", "R",  "Y"]:
                    n_data, e_data = rotate_ne_rt(tr1_data, tr2_data, baz)
                elif trace.stats.channel in ["2", "T",  "X"]:
                    n_data, e_data = rotate_ne_rt(tr2_data, tr1_data, baz)

                # Replace original traces with rotated data
                trace.data = n_data
                paired_channel.data = e_data

                # Mark as processed
                rotated_ids.add(trace_id)
                rotated_ids.add(paired_channel.id)

        stream = stream.sort()
        return stream

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

    from obspy import Stream, Trace
    import numpy as np

    def fill_stream_with_inferred_channels(self):
        """
        Automatically deduce the channel configuration for each station and fill missing channels with dummy traces.

        Parameters:
            stream (Stream): The input ObsPy Stream object.

        Returns:
            Stream: A Stream object where missing channels are filled with dummy traces.
        """
        # Group traces by station (network, station)
        stations = {}
        for trace in self.__st:
            station_id = (trace.stats.network, trace.stats.station)
            if station_id not in stations:
                stations[station_id] = []
            stations[station_id].append(trace)

        # Create a new Stream to hold the filled data
        filled_stream = Stream()

        for (network, station), traces in stations.items():
            # Get existing channels for this station
            existing_channels = {trace.stats.channel[-1] for trace in traces}

            # Automatically infer the channel configuration
            inferred_config = self.__infer_channel_configuration(existing_channels)

            # Identify missing channels
            missing_channels = set(inferred_config) - existing_channels

            # Add existing traces to the new stream
            for trace in traces:
                filled_stream.append(trace)

            # Fill missing channels with dummy traces
            for channel in missing_channels:
                # Create dummy data based on the first trace's properties, or default values
                if traces:
                    npts = traces[0].stats.npts
                    sampling_rate = traces[0].stats.sampling_rate
                    starttime = traces[0].stats.starttime
                    channel = traces[0].stats.channel[0:2] + channel
                    header = {"network": traces[0].stats.network, "station": traces[0].stats.station,
                              "sampling_rate": sampling_rate, "channel": channel, "location": traces[0].stats.location,
                              "starttime": starttime, "npts": npts}

                    # dummy_data = np.zeros(npts)
                    dummy_data = self._generate_white_noise(npts, 1E-3*np.max(np.abs(traces[0].data)))
                    dummy_trace = Trace(data=dummy_data, header=header)
                    filled_stream.append(dummy_trace)

        self.__st = filled_stream

    def __infer_channel_configuration(self, existing_channels):
        """
        Infers the most likely channel configuration based on available channels.

        Parameters:
            existing_channels (set): Set of existing channel names for a given station.

        Returns:
            list: The inferred channel configuration.
        """
        # Define possible channel naming schemes and their corresponding configurations
        configurations = [
            ["Z", "1", "2"],  # Vertical, Horizontal 1, Horizontal 2
            ["Z", "N", "E"],  # Vertical, North, East
            ["Z", "Y", "X"]  # Vertical, North, East (with different naming)
        ]

        # Check which configuration the existing channels most closely match
        for config in configurations:
            if all(channel in config for channel in existing_channels):
                return config

        # If no configuration is fully matched, return the most common channel naming scheme
        # (We assume "Z" is always present if the station has a vertical component)
        if "Z" in existing_channels:
            return ["Z", "1", "2"]  # Default configuration: Vertical, Horizontal 1, Horizontal 2
        else:
            return list(existing_channels)  # Fallback, return whatever channels exist

    def _generate_white_noise(self, length, max_amplitude):
        """
        Generates a vector of white noise.

        Parameters:
            length (int): Length of the vector (number of samples).
            max_amplitude (float): Maximum amplitude of the noise (absolute value).

        Returns:
            np.ndarray: A vector of white noise.
        """
        # Generate random values from a normal distribution (mean=0, std=1)
        noise = np.random.randn(length)

        # Scale the noise to have the desired maximum amplitude
        noise = max_amplitude * noise / np.max(np.abs(noise))

        return noise
