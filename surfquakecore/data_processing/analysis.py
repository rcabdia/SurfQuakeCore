from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml
from obspy import read_inventory
from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth, kilometer2degrees
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from multiprocessing import freeze_support
from surfquakecore.project.surf_project import SurfProject
from surfquakecore.data_processing.seismogram_analysis import SeismogramData
from surfquakecore.seismoplot.plot import PlotProj
from obspy import Stream, read, UTCDateTime
from surfquakecore.data_processing.parser.config_parser import parse_configuration_file
from typing import Union, List, Optional
from obspy.core.trace import Trace

class Analysis:

    def __init__(self, output: Optional[str] = None,
                 inventory_file: Optional[str] = None, config_file: Optional[str] = None,
                 event_file: Optional[str] = None):

        self.model = TauPyModel("iasp91")
        self.output = output
        self._exist_folder = False
        self.inventory = None
        self.all_traces = []

        if inventory_file:
            try:
                self.inventory = read_inventory(inventory_file)
                print(self.inventory)
            except:
                print("Warning NO VALID inventory file: ", inventory_file)

        self.config = None
        self.df_events = None

        if config_file is not None:
            self.config = self.load_analysis_configuration(config_file)

        if self.output is not None:
            if os.path.isdir(self.output):
                print("Output Directory: ", self.output)
            else:
                print("Traces will be process and might be plot: ", self.output)
                self.output = None

        if event_file is not None:
            try:
                self.df_events = pd.read_csv(event_file, sep=';')
                self.df_events['datetime'] = pd.to_datetime((self.df_events['date'] + ' ' + self.df_events['hour']),
                                                            utc=True)
            except:
                print("Cannot import event csv files")

    def run_processing_loop(self, start, end, project: Union[SurfProject, List[SurfProject]], plot=False):
        if isinstance(project, List):
            for proj in project:
                self._run_processing(start, end, proj, plot=plot)

        else:
            self._run_processing(start, end, project, plot=plot)
    def _run_processing(self, start, end, project: Union[SurfProject, List[SurfProject]], plot=False):

        traces = []

        # 1. Check self.event_file is not None
        if self.df_events is not None and self.inventory is not None:

            # 2. Cut event files & Check if Rotate
            rotate = next((item for item in self.config if item.get('name') == 'rotate'), None)
            self.cut_files(start, end, project, rotate)

            if len(self.all_traces) > 0:
                traces = self.all_traces

        elif self.config is not None:
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        self.process_trace,
                        project.data_files[i][0],
                        self.config,
                        self.inventory,
                        self.output
                    )
                    for i in range(len(project.data_files))
                ]

                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        traces.append(result)

        # Check if shifts are requested
        shift_dict = next((item for item in self.config if item.get('name') == 'shift'), None)
        if shift_dict:
            try:
                for i in range(len(traces) - 1):
                    original = traces[i].stats.starttime
                    traces[i].stats.starttime = original + shift_dict['time_shifts'][i]
                    print(original, traces[i].stats.starttime, shift_dict['time_shifts'][i])
                # Now `traces` is updated in-place
            except Exception as e:
                print(f"Error applying time shifts: {e}")

        if plot and len(traces) > 0:
            plotter = PlotProj(traces, metadata=self.inventory)
            plotter.plot(traces_per_fig=5, sort_by='distance')

    @staticmethod
    def filter_files(project, net=None, station=None, channel=None, starttime=None, endtime=None):
        filter = {}
        time = {}
        date_format = "%Y-%m-%d %H:%M:%S"

        # Filter net
        if net is not None:
            filter['net'] = net

        # Filter station
        if station is not None:
            filter['station'] = station

        # Filter channel
        if channel is not None:
            filter['channel'] = channel

        # Filter start time
        if starttime is not None:
            try:
                time['startime'] = UTCDateTime(datetime.strptime(starttime, date_format))
            except ValueError:
                raise ValueError(f"Error: start time '{starttime}' does not hace the correct format ({date_format}).")

                # Filter end time
        if endtime is not None:
            try:
                time['endtime'] = UTCDateTime(datetime.strptime(endtime, date_format))
            except ValueError:
                raise ValueError(f"Error: end time '{endtime}' does not hace the correct format ({date_format}).")

                # Apply filter
        if len(filter) > 0:
            project.filter_project_keys(**filter)
        else:
            project.filter_project_keys()

        # Apply filter time
        if len(time) > 0:
            return project.filter_time(**time)
        else:
            return project.filter_time()

    def load_analysis_configuration(self, config_file: str):
        with open(config_file, 'r') as file:
            try:
                yaml_data = yaml.safe_load(file)
                print(f"Loaded: {os.path.basename(file.name)}")
            except Exception as e:
                print(f"Conflictive file at {os.path.basename(file.name)}: {e}")
                return None

        return parse_configuration_file(yaml_data)

    def get_project_files(self, project_path):
        freeze_support()
        sp = SurfProject(project_path)
        sp.search_files()
        return sp.project

    def process_trace(self, file_path, config_file, inventory, output_dir):

        try:
            st = read(file_path)
            sd = SeismogramData(st, inventory)
            tr = sd.run_analysis(config_file)

            # Optional: Write to disk if output directory exists
            if output_dir is not None:
                tr.write(os.path.join(output_dir, tr.id), 'mseed')
            else:
                pass

            return tr

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None



    def create_folder(self, name):
        if not os.path.exists(name):
            print("The output folder does not exists")
            self._exist_folder = False
            # os.makedirs(name)

    def process_station(self, _station, df_files, df_inventory, event, lat, lon, depth, model, start, end, config_file,
                        inventory):

        st_trimed_local = []
        channels = df_files['channel'].unique()
        #df_station_filtered = df_files[df_files['station'] == _station]
        inventory_filtered = df_inventory[(df_inventory['station'] == _station)]

        for _channel in channels:
            files_filtered = df_files[(df_files['station'] == _station) & (df_files['channel'] == _channel)]
            st = Stream()

            if not inventory_filtered.empty:
                for _, _file in files_filtered.iterrows():
                    _st = read(_file['file'])
                    gaps = _st.get_gaps()
                    if len(gaps) > 0:
                        _st.print_gaps()
                    st += _st

                st.merge(fill_value="interpolate")

                distance, BAZ, AZ = gps2dist_azimuth(lat, lon,
                                                     inventory_filtered['latitude'].tolist()[0],
                                                     inventory_filtered['longitude'].tolist()[0])
                distance_deg = kilometer2degrees(distance / 1000)

                arrivals = model.get_travel_times(source_depth_in_km=depth, distance_in_degree=distance_deg)
                p_time = arrivals[0].time

                start_cut = event["datetime"] + timedelta(seconds=p_time) - timedelta(seconds=start)
                end_cut = event["datetime"] + timedelta(seconds=p_time) + timedelta(seconds=end)

                if not files_filtered.empty:
                    if end_cut is not None and start_cut is not None:
                        st.trim(UTCDateTime(start_cut), UTCDateTime(end_cut))

                    if config_file:
                        sd = SeismogramData(st, inventory)
                        tr = sd.run_analysis(config_file)
                        tr = self._set_header(tr, distance/1000, BAZ, AZ, UTCDateTime(event["datetime"]), lat, lon, depth)
                        st_trimed_local.append([files_filtered["file"].iloc[0], tr])
                    else:
                        st_trimed_local.append([files_filtered["file"].iloc[0], st.traces[0]])
        return st_trimed_local

    def cut_files(self, start, end, project, rotate=False):

        # 1. Project dataframe
        df_project = self.project_table(project)

        # 2. Invetory dataframe
        df_inventory = self.inventory_table()

        # 3. Loop over events
        for index, event in self.df_events.iterrows():
            df_files = pd.DataFrame(columns=df_project.columns)
            id = event['datetime'].strftime("%Y-%m-%d %H:%M:%S")
            lat = event['latitude']
            lon = event['longitude']
            depth = event['depth']
            event_time = event['datetime']
            start_event = event['datetime'] - timedelta(seconds=start)
            end_event = event['datetime'] + timedelta(seconds=end)

            # Crear carpeta
            _id = id.split(" ")
            _id[1] = _id[1].split(":")
            _id[1] = ''.join(_id[1])

            if self.output is not None:
                event_folder = os.path.join(self.output, _id[0] + "_" + _id[1])
                self.create_folder(event_folder)
            else:
                event_folder = None

            for index, file in df_project.iterrows():
                if (file['start'] <= start_event and file['end'] >= start_event) or \
                        (file['start'] <= end_event and file['end'] >= end_event) or \
                        (file['start'] <= start_event and file['end'] >= end_event):
                    df_files = pd.concat([df_files, file.to_frame().T], ignore_index=True)

            stations = df_files['station'].unique()
            channels = df_files['channel'].unique()
            df_files['file'] = df_files['file'].astype(str)
            st_trimed = []

            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(self.process_station, _station, df_files, df_inventory, event, lat, lon, depth,
                                    self.model, start, end, self.config, self.inventory)
                    for _station in stations
                ]

                for future in as_completed(futures):
                    try:
                        st_trimed.extend(future.result())
                    except Exception as e:
                        print(f"Error processing station: {e}")

            df_files['trace'] = ''
            df_files['component1'] = ''
            df_files['component2'] = ''

            if rotate and len(st_trimed) > 0:
                for i in range(len(st_trimed)):
                    filtro = df_files['file'] == st_trimed[i][0]
                    _index = df_files.index[filtro][0]
                    df_files.at[_index, 'trace'] = st_trimed[i][1]

                    _component = list(st_trimed[i][1].stats['channel'])

                    df_files.at[_index, 'component1'] = _component[0] + _component[1]
                    df_files.at[_index, 'component2'] = _component[2]

                # Rotate and saved once rotated
                st_trimed = self.rotate(df_files, df_inventory)
            if len(st_trimed) > 0:
                self._save_trimmed_traces(st_trimed, event_folder)

    def rotate(self, df_rotate, inventory):

        # 1. Project dataframe
        stations = df_rotate['station'].unique()
        component = df_rotate['component1'].unique()
        ZNE = ['Z', 'N', 'E']
        Z12 = ['Z', '1', '2']
        ZXY = ['Z', 'X', 'Y']
        # channels = df_files['channel'].unique()
        st_rotated_all = []
        for _station in stations:
            st_rotated = []
            df_station_filtered = df_rotate[df_rotate['station'] == _station]
            inventory_filtered = inventory[(inventory['station'] == _station)]
            merged = Stream()
            if len(df_station_filtered) > 1:
                for _component in component:
                    df_component_filtered = df_station_filtered[df_station_filtered['component1'] == _component]

                    if df_component_filtered['component2'].isin(ZNE).sum() == 3 or df_component_filtered[
                        'component2'].isin(Z12).sum() == 3 or df_component_filtered['component2'].isin(ZXY).sum() == 3:
                        for index, _trace in df_component_filtered.iterrows():
                            if _trace['trace'].stats.channel[-1] in ['1', 'Y']:
                                _trace['trace'].stats.channel = _trace['trace'].stats.channel[0:2] + 'N'
                                # _trace['trace'].stats.channel.replace(_trace['trace'].stats.channel[-1], "N")
                            elif _trace['trace'].stats.channel[-1] in ['2', 'X']:
                                _trace['trace'].stats.channel = _trace['trace'].stats.channel[0:2] + 'E'

                                # _trace['trace'].stats.channel.replace(_trace['trace'].stats.channel[-1], "E")
                            merged += _trace['trace']

                        st_rotated = self.rotate_stream_to_GEAC(merged, self.inventory,
                                                                inventory_filtered['latitude'].iloc[0],
                                                                inventory_filtered['longitude'].iloc[0])
                        st_rotated_all.append(st_rotated)

        try:
            if len(st_rotated_all) > 0:
                return st_rotated_all
        except:
            return None



    def project_table(self, project):
        df_project = pd.DataFrame(columns=['file', 'start', 'end', 'net', 'station', 'channel', 'day'])
        _file = []
        _start = []
        _end = []
        _net = []
        _station = []
        _channel = []

        for project_files in project.data_files:
            _file.append(project_files[0])
            _start.append(project_files[1])
            _end.append(project_files[2])
            _net.append(self.find_stats(project.project, project_files[0], 'network'))
            _station.append(self.find_stats(project.project, project_files[0], 'station'))
            _channel.append(self.find_stats(project.project, project_files[0], 'channel'))

        df_project['file'] = _file
        df_project['start'] = _start
        df_project['end'] = _end
        df_project['net'] = _net
        df_project['station'] = _station
        df_project['channel'] = _channel

        return df_project

    @staticmethod
    def find_stats(lists, name, stat):
        _value = [key for key, list in lists.items() if any(name in sublist for sublist in list)]

        if len(_value) > 0:
            for i, sublist in enumerate(lists[_value[0]]):
                if name in sublist:
                    _i = sublist.index(name)
                    return lists[_value[0]][_i][1][stat]

        return None

    def inventory_table(self):
        df_inventory = pd.DataFrame(columns=['net', 'station', 'latitude', 'longitude'])
        net_inventory = []
        station_inventory = []
        latitude_inventory = []
        longitude_inventory = []

        for network in self.inventory.networks:
            for station in network.stations:
                net_inventory.append(network.code)
                station_inventory.append(station.code)
                latitude_inventory.append(station.latitude)
                longitude_inventory.append(station.longitude)

        df_inventory['net'] = net_inventory
        df_inventory['station'] = station_inventory
        df_inventory['latitude'] = latitude_inventory
        df_inventory['longitude'] = longitude_inventory

        return df_inventory

    def rotate_stream_to_GEAC(self, stream, inventory, epicenter_lat, epicenter_lon):
        """
        Rotates an ObsPy Stream to Great Circle Arc Coordinates (GEAC) using an inventory.
        Includes the vertical component ("Z") in the output.

        Args:
            stream (Stream): The ObsPy Stream object containing the traces.
            inventory (Inventory): ObsPy Inventory containing station metadata.
            epicenter_lat (float): Latitude of the epicenter.
            epicenter_lon (float): Longitude of the epicenter.

        Returns:
            list: A list of ObsPy Stream objects, each corresponding to a station with rotated components.
        """
        # inventory = read_inventory(inventory)

        # Step 1: Group traces by station
        station_dict = {}
        for trace in stream:
            station_id = trace.stats.network + "." + trace.stats.station
            if station_id not in station_dict:
                station_dict[station_id] = []
            station_dict[station_id].append(trace)

        rotated_streams = []

        # Step 2: Process each station
        for station_id, traces in station_dict.items():
            # Extract components
            components = {tr.stats.channel[-1]: tr for tr in traces}  # {'E': Trace, 'N': Trace, 'Z': Trace}

            if 'N' not in components or 'E' not in components:
                print(f"Skipping station {station_id}: Missing N or E component.")
                continue

            tr_n = components['N']
            tr_e = components['E']
            tr_z = components.get('Z', None)  # Z may not exist, handle it safely

            # Step 3: Check sampling rate and sample count
            if tr_n.stats.sampling_rate != tr_e.stats.sampling_rate:
                print(f"Skipping station {station_id}: Sampling rates do not match.")
                continue
            if len(tr_n.data) != len(tr_e.data):
                print(f"Skipping station {station_id}: Number of samples do not match.")
                continue
            if tr_z and (tr_n.stats.sampling_rate != tr_z.stats.sampling_rate or len(tr_n.data) != len(tr_z.data)):
                print(f"Skipping station {station_id}: Z component sampling rate or samples do not match.")
                continue

            # Step 4: Get station coordinates from inventory
            try:
                network_code, station_code = station_id.split(".")
                station = inventory.select(network=network_code, station=station_code)[0][0]
                station_lat, station_lon = station.latitude, station.longitude
            except Exception as e:
                print(f"Skipping station {station_id}: Station not found in inventory. ({e})")
                continue

            # Step 5: Compute back azimuth
            _, baz, _ = gps2dist_azimuth(epicenter_lat, epicenter_lon, station_lat, station_lon)

            # Step 6: Rotate to GEAC (Radial & Transverse)
            theta = np.deg2rad(baz)
            data_r = tr_n.data * np.cos(theta) + tr_e.data * np.sin(theta)
            data_t = -tr_n.data * np.sin(theta) + tr_e.data * np.cos(theta)

            # Step 7: Create new rotated traces
            tr_r = tr_n.copy()
            tr_r.data = data_r
            tr_r.stats.channel = tr_r.stats.channel[:-1] + 'R'  # Rename to Radial

            tr_t = tr_e.copy()
            tr_t.data = data_t
            tr_t.stats.channel = tr_t.stats.channel[:-1] + 'T'  # Rename to Transverse

            # Step 8: Store in new stream
            rotated_traces = [tr_r, tr_t]

            # Include Z component if available
            if tr_z:
                rotated_traces.append(tr_z.copy())  # Copy Z component as is

            rotated_streams.append(Stream(traces=rotated_traces))

        if len(rotated_streams) > 0:
            return rotated_streams
        else:
            return stream

    def _set_header(self, tr, distance_km, BAZ, AZ, otime, lat, lon, depth):
        tr.stats['geodetic'] = {'otime': otime, 'geodetic': [distance_km, AZ, BAZ], 'event': [lat, lon, depth]}
        return tr

    def _save_trimmed_traces(self, st_trimmed, event_folder):
        """
        Save trimmed traces to disk and store them in self.all_traces.

        Args:
            st_trimmed (list): List of Trace, (key, Trace) tuples, or nested lists of Trace.
            event_folder (str): Directory where files will be saved.
        """

        def extract_traces(obj):
            # Return a flat list of Trace objects
            traces = []
            if isinstance(obj, Trace):
                traces.append(obj)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    if isinstance(item, Trace):
                        traces.append(item)
                    elif isinstance(item, (list, tuple)) and isinstance(item[1], Trace):
                        traces.append(item[1])
                    elif isinstance(item, Stream):
                        for tr in item:
                            traces.append(tr)
            return traces


        for entry in st_trimmed:
            traces = extract_traces(entry)
            for tr in traces:
                try:
                    if event_folder is not None:
                        t1 = tr.stats.starttime
                        base_name = f"{tr.id}.D.{t1.year}.{t1.julday}"
                        path_output = os.path.join(event_folder, base_name)

                        # Ensure uniqueness
                        counter = 1
                        while os.path.exists(path_output):
                            path_output = os.path.join(event_folder, f"{base_name}_{counter}")
                            counter += 1

                        # Save if self.output is defined
                        if self.output is not None:
                            print(f"{tr.id} - Writing processed data to {path_output}")
                            tr.write(path_output, format="H5")

                    if len(tr.data) > 0:
                        self.all_traces.append(tr)

                except Exception as e:
                    print(f"Failed to process trace {tr.id if hasattr(tr, 'id') else ''}: {e}")

