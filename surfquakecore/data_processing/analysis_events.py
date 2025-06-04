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

class AnalysisEvents:

    def __init__(self, output: Optional[str] = None,
                 inventory_file: Optional[str] = None, config_file: Optional[str] = None,
                 surf_projects: List[SurfProject] = None, plot_config_file: Optional[str]= None):

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
        file_group, event, model, cut_start, cut_end, inventory, set_header_func = args
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

            return traces

        except Exception as e:
            print(f"[WARNING] Station group failed: {e}")
            return []

    def run_waveform_cutting(self, cut_start: float, cut_end: float, plot=True):
        if self.surf_projects is None:
            print("No projects to process.")
            return

        for i, project in enumerate(self.surf_projects):
            event = getattr(project, "_event_metadata", None)
            if not event:
                print(f"[INFO] Skipping subproject {i}: no event metadata")
                continue

            print(f"[INFO] Cutting traces for event {i} at {event['origin_time']}")

            # ---- Group trace files by station code ----
            station_files = defaultdict(list)
            for trace_list in project.project.values():
                for trace_path, stats in trace_list:
                    station_key = f"{stats.network}.{stats.station}"
                    station_files[station_key].append((trace_path, stats))

            # ---- Create parallel tasks ----
            tasks = []
            for station_key, file_group in station_files.items():
                tasks.append((
                    file_group,  # all files for one station
                    event,
                    self.model,
                    cut_start,
                    cut_end,
                    self.inventory,
                    self._set_header
                ))

            # ---- Multiprocessing ----
            with Pool(processes=min(cpu_count(), len(tasks))) as pool:
                results = pool.map(self._process_station_traces, tasks)

            # Flatten list of list of traces
            all_traces = [tr for sublist in results for tr in sublist if tr is not None]
            full_stream = self._clean_traces(all_traces)
            print(f"[INFO] Subproject {i}: {len(full_stream)} traces kept")

            # ---- Optional Plot ----
            if plot and len(full_stream) > 0:
                PlotProj(full_stream, plot_config=self.plot_config).plot()

            # ---- Optional Save ----
            if self.output:
                self._write_files(full_stream)

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