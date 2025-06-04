# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: analysis_events.py
# Program: surfQuake & ISP
# Date: June 2025
# Purpose: Project Manager
# Author: Roberto Cabieces, Thiago C. Junqueira & C. Palacios
#  Email: rcabdia@roa.es
# --------------------------------------------------------------------

import yaml
from obspy import read_inventory
from obspy.taup import TauPyModel
import os
from surfquakecore.data_processing.parser.config_parser import parse_configuration_file
from typing import Optional, List
from surfquakecore.data_processing.seismogram_analysis import SeismogramData
from surfquakecore.project.surf_project import SurfProject
from multiprocessing import Pool, cpu_count
from obspy.geodetics import gps2dist_azimuth
from obspy import read
from obspy.core.stream import Stream, Trace
from surfquakecore.seismoplot.plot import PlotProj
from collections import defaultdict
import importlib.util

class AnalysisEvents:

    def __init__(self, output: Optional[str] = None,
                 inventory_file: Optional[str] = None, config_file: Optional[str] = None,
                 surf_projects: List[SurfProject] = None, plot_config_file: Optional[str] = None,
                 post_script: Optional[str] = None):

        self.model = TauPyModel("iasp91")
        self.output = output
        self._exist_folder = False
        self.inventory = None
        self.all_traces = []
        self.surf_projects = surf_projects

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

        self.plot_config = None
        if plot_config_file is not None and os.path.exists(plot_config_file):
            with open(plot_config_file, 'r') as f:
                try:
                    self.plot_config = yaml.safe_load(f).get("plotting", {})
                except Exception as e:
                    print(f"[WARNING] Plot config not loaded: {e}")

        self.post_script_func = None
        if post_script and os.path.exists(post_script):
            spec = importlib.util.spec_from_file_location("post_module", post_script)
            post_module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(post_module)
                if hasattr(post_module, "run") and callable(post_module.run):
                    self.post_script_func = post_module.run
                    print(f"[INFO] Post-script loaded: {post_script}")
                else:
                    print(f"[WARNING] Script {post_script} must define a `run(stream, event)` function")
            except Exception as e:
                print(f"[ERROR] Failed to import script {post_script}: {e}")

    def load_analysis_configuration(self, config_file: str):
        with open(config_file, 'r') as file:
            try:
                yaml_data = yaml.safe_load(file)
                print(f"Loaded: {os.path.basename(file.name)}")
            except Exception as e:
                print(f"Conflictive file at {os.path.basename(file.name)}: {e}")
                return None

        return parse_configuration_file(yaml_data)

    def _process_station_traces(self, args):
        file_group, event, model, cut_start, cut_end, inventory, set_header_func, additional_processing = args
        try:
            # Read & merge all files for this station
            st = Stream()
            for trace_path, _ in file_group:
                st += read(trace_path)
            st.merge(method=1, fill_value='interpolate')

            if len(st) == 0:
                return []

            # Extract station info from first trace
            stats = file_group[0][1]
            net, sta = stats.network, stats.station
            origin = event["origin_time"]
            lat, lon, depth = event["latitude"], event["longitude"], event["depth"]

            inv_sta = inventory.select(network=net, station=sta)
            sta_coords = inv_sta[0][0].latitude, inv_sta[0][0].longitude, inv_sta[0][0].elevation
            distance_m, az, baz = gps2dist_azimuth(sta_coords[0], sta_coords[1], lat, lon)
            distance_deg = distance_m / 1000 / 111.19

            arrivals = model.get_travel_times(source_depth_in_km=depth,
                                              distance_in_degree=distance_deg)
            if not arrivals:
                raise ValueError("No valid arrivals returned by TauPyModel.")

            first_arrival = origin + arrivals[0].time
            t1 = first_arrival - cut_start
            t2 = first_arrival + cut_end
            st.trim(starttime=t1, endtime=t2)

            if len(st) == 0:
                return []

            traces = []
            for tr in st:
                if self.config:
                    sd = SeismogramData(Stream(tr), inventory, fill_gaps=True)
                    tr = sd.run_analysis(self.config)
                tr = set_header_func(tr, distance_km=distance_m / 1000, BAZ=baz, AZ=az,
                                     otime=origin, lat=lat, lon=lon, depth=depth)
                traces.append(tr)

            # In this point we can go ahead with rotate  if  needed
            if len(additional_processing["rotate"]) > 0:
                # TODO, WE CAN INCLUDE LATER MORE METHOD LIKE LQT
                st_rotate = Stream(traces)
                st_rotate.rotate(method='NE->RT', back_azimuth=baz)

                # Back to list of traces
                traces = []
                for tr in st:
                    traces.append(tr)

            return traces

        except Exception as e:
            print(f"[WARNING] Station group failed: {e}")
            return []

    # def run_waveform_cutting(self, cut_start: float, cut_end: float, plot=True):
    #     additional_processing = {"rotate": {}, "shift": {}}
    #     rotate = {}
    #     shift = {}
    #     # Check if Rotate & Shift
    #     if self.config:
    #         rotate = next((item for item in self.config if item.get('name') == 'rotate'), None)
    #         shift = next((item for item in self.config if item.get('name') == 'shift'), None)
    #
    #     if rotate:
    #         additional_processing["rotate"] = rotate
    #     if shift:
    #         additional_processing["shift"] = shift
    #
    #     if self.surf_projects is None:
    #         print("No projects to process.")
    #         return
    #
    #     for i, project in enumerate(self.surf_projects):
    #         event = getattr(project, "_event_metadata", None)
    #         if not event:
    #             print(f"[INFO] Skipping subproject {i}: no event metadata")
    #             continue
    #
    #         print(f"[INFO] Cutting traces for event {i} at {event['origin_time']}")
    #
    #         # ---- Group trace files by station code ----
    #         station_files = defaultdict(list)
    #         for trace_list in project.project.values():
    #             for trace_path, stats in trace_list:
    #                 station_key = f"{stats.network}.{stats.station}"
    #                 station_files[station_key].append((trace_path, stats))
    #
    #         # ---- Create parallel tasks ----
    #         tasks = []
    #         for station_key, file_group in station_files.items():
    #             tasks.append((
    #                 file_group,  # all files for one station
    #                 event,
    #                 self.model,
    #                 cut_start,
    #                 cut_end,
    #                 self.inventory,
    #                 self._set_header,
    #                 additional_processing
    #             ))
    #
    #         # ---- Multiprocessing ----
    #         with Pool(processes=min(cpu_count(), len(tasks))) as pool:
    #             results = pool.map(self._process_station_traces, tasks)
    #
    #         # Flatten lists of list of traces
    #         all_traces = [tr for sublist in results for tr in sublist if tr is not None]
    #         full_stream = self._clean_traces(all_traces)
    #
    #         # let's apply shift
    #         if shift:
    #             full_stream = self._shift(additional_processing, full_stream)
    #
    #         print(f"[INFO] Subproject {i}: {len(full_stream)} traces kept")
    #
    #         # let's apply custom script.py
    #         if self.post_script_func:
    #             try:
    #                 print(f"[INFO] Running user post-script for event {i}")
    #                 full_stream = self.post_script_func(full_stream, event)
    #             except Exception as e:
    #                 print(f"[WARNING] Post-script for event {i} failed: {e}")
    #
    #         # ---- Optional Plot ----
    #         if full_stream:
    #             if plot and len(full_stream) > 0:
    #                 PlotProj(full_stream, plot_config=self.plot_config).plot()
    #
    #         # ---- Optional Save ----
    #         if self.output:
    #             self._write_files(full_stream)

    def _write_files(self, full_stream):
        errors = False
        for j, tr in enumerate(full_stream):
            try:
                t1 = tr.stats.starttime
                base_name = f"{tr.id}.D.{t1.year}.{t1.julday}"
                path_output = os.path.join(self.output, base_name)

                # Check if file exists and append a number if necessary
                counter = 1
                while os.path.exists(path_output):
                    path_output = os.path.join(self.output, f"{base_name}_{counter}")
                    counter += 1

                print(f"{tr.id} - Writing processed data to {path_output}")
                tr.write(path_output, format="H5")

            except Exception as e:
                errors = True
                print(f"File cannot be written: {self.output}, Error: {e}")

            if errors:
                print("Writting Complete with Errors, check output", self.output)
            else:
                print("Writting Complete, check output", self.output)

    def _set_header(self, tr, distance_km, BAZ, AZ, otime, lat, lon, depth):
        tr.stats['geodetic'] = {'otime': otime, 'geodetic': [distance_km, AZ, BAZ], 'event': [lat, lon, depth]}
        return tr

    def _clean_traces(self, traces: List[Trace]) -> Stream:
        """
        Filter out empty or invalid Trace objects and return an ObsPy Stream.
        """
        stream = Stream()
        for tr in traces:
            if isinstance(tr, Trace) and len(tr.data) > 0:
                stream.append(tr)
        return stream

    # def run_waveform_analysis(self, plot: bool = False):
    #     if self.surf_projects is None:
    #         print("No subprojects to process.")
    #         return
    #
    #     for i, project in enumerate(self.surf_projects):
    #         print(f"[INFO] Processing subproject {i} (daily stream)")
    #
    #         # Group by station
    #         station_files = defaultdict(list)
    #         for trace_list in project.project.values():
    #             for trace_path, stats in trace_list:
    #                 key = f"{stats.network}.{stats.station}"
    #                 station_files[key].append((trace_path, stats))
    #
    #         tasks = []
    #         for _, file_group in station_files.items():
    #             tasks.append((
    #                 file_group,
    #                 None,  # No event metadata used
    #                 None,  # No TauPyModel needed
    #                 0, 0,  # No cut window
    #                 self.inventory,
    #                 self._set_header  # Still tags geodetic but without origin time
    #             ))
    #
    #         with Pool(processes=min(cpu_count(), len(tasks))) as pool:
    #             results = pool.map(self._process_station_analysis, tasks)
    #
    #         all_traces = [tr for group in results for tr in group if tr is not None]
    #         full_stream = self._clean_traces(all_traces)
    #         print(f"[INFO] Subproject {i}: {len(full_stream)} traces processed")
    #
    #         if plot and len(full_stream) > 0:
    #             PlotProj(full_stream, plot_config=self.plot_config).plot()
    #
    #         if self.output:
    #             self._write_files(full_stream)

    def run_waveform_analysis(self, plot: bool = False):
        if self.surf_projects is None:
            print("No subprojects to process.")
            return

        for i, project in enumerate(self.surf_projects):
            print(f"[INFO] Processing subproject {i} (daily stream)")

            station_files = defaultdict(list)
            for trace_list in project.project.values():
                for trace_path, stats in trace_list:
                    key = f"{stats.network}.{stats.station}"
                    station_files[key].append((trace_path, stats))

            tasks = []
            for file_group in station_files.values():
                tasks.append((
                    file_group,
                    None,  # No event
                    None,  # No model
                    0, 0,  # No cut
                    self.inventory,
                    self._set_header
                ))

            with Pool(processes=min(cpu_count(), len(tasks))) as pool:
                results = pool.map(self._process_station_analysis, tasks)

            all_traces = [tr for group in results for tr in group if tr is not None]
            full_stream = self._clean_traces(all_traces)

            print(f"[INFO] Subproject {i}: {len(full_stream)} traces processed")

            if plot and len(full_stream) > 0:
                PlotProj(full_stream, plot_config=self.plot_config).plot()

            if self.output:
                self._write_files(full_stream)

    def run_waveform_cutting(self, cut_start: float, cut_end: float, plot=True):
        additional_processing = {"rotate": {}, "shift": {}}
        rotate = next((item for item in self.config if item.get('name') == 'rotate'), None) if self.config else None
        shift = next((item for item in self.config if item.get('name') == 'shift'), None) if self.config else None

        if rotate:
            additional_processing["rotate"] = rotate
        if shift:
            additional_processing["shift"] = shift

        if self.surf_projects is None:
            print("No projects to process.")
            return

        for i, project in enumerate(self.surf_projects):
            events = getattr(project, "_events_metadata", [])
            if not events:
                print(f"[INFO] Skipping subproject {i}: no associated events")
                continue

            for j, event in enumerate(events):
                print(f"[INFO] Cutting traces for subproject {i}, event {j} at {event['origin_time']}")

                # ---- Group files by station ----
                station_files = defaultdict(list)
                for trace_list in project.project.values():
                    for trace_path, stats in trace_list:
                        station_key = f"{stats.network}.{stats.station}"
                        station_files[station_key].append((trace_path, stats))

                # ---- Parallel tasks ----
                tasks = []
                for file_group in station_files.values():
                    tasks.append((
                        file_group,
                        event,
                        self.model,
                        cut_start,
                        cut_end,
                        self.inventory,
                        self._set_header,
                        additional_processing
                    ))

                with Pool(processes=min(cpu_count(), len(tasks))) as pool:
                    results = pool.map(self._process_station_traces, tasks)

                all_traces = [tr for sublist in results for tr in sublist if tr is not None]
                full_stream = self._clean_traces(all_traces)

                if shift:
                    full_stream = self._shift(additional_processing, full_stream)

                print(f"[INFO] Subproject {i}, event {j}: {len(full_stream)} traces kept")

                # ---- Post-user script if defined ----
                if self.post_script_func:
                    try:
                        print(f"[INFO] Running post-script for subproject {i}, event {j}")
                        full_stream = self.post_script_func(full_stream, event)
                    except Exception as e:
                        print(f"[WARNING] Post-script failed: {e}")

                if plot and len(full_stream) > 0:
                    PlotProj(full_stream, plot_config=self.plot_config).plot()

                if self.output:
                    self._write_files(full_stream)
    def _process_station_analysis(self, args):
        file_group, _, _, _, _, inventory, set_header_func = args
        try:
            st = Stream()
            for trace_path, _ in file_group:
                st += read(trace_path)
            st.merge(method=1, fill_value='interpolate')

            if len(st) == 0:
                return []

            # Metadata from first trace
            stats = file_group[0][1]
            net, sta = stats.network, stats.station

            traces = []
            for tr in st:
                if self.config:
                    sd = SeismogramData(Stream(tr), inventory, fill_gaps=True)
                    tr = sd.run_analysis(self.config)

                tr = set_header_func(tr, distance_km=-1, BAZ=-1, AZ=-1,
                                     otime=None, lat=None, lon=None, depth=None)
                traces.append(tr)

            return traces

        except Exception as e:
            print(f"[WARNING] Failed station processing: {e}")
            return []

    def _shift(self, additional_processing, full_stream):
        try:
            for i, tr in enumerate(full_stream):
                tr[i].stats.starttime = tr[i].stats.starttime + additional_processing["shift"][i]
                # print(original, additional_processing["shift"][i].stats.starttime,
                # additional_processing["shift"]['time_shifts'][i])

            # Now `traces` is updated in-place
        except Exception as e:
            print(f"Error applying time shifts: {e}")
        return full_stream