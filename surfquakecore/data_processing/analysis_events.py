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


class AnalysisEvents:

    def __init__(self, output: Optional[str] = None,
                 inventory_file: Optional[str] = None, config_file: Optional[str] = None,
                 surf_projects: List[SurfProject] = None):

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

    def load_analysis_configuration(self, config_file: str):
        with open(config_file, 'r') as file:
            try:
                yaml_data = yaml.safe_load(file)
                print(f"Loaded: {os.path.basename(file.name)}")
            except Exception as e:
                print(f"Conflictive file at {os.path.basename(file.name)}: {e}")
                return None

        return parse_configuration_file(yaml_data)

    def _process_trace_cut(self, args):
        trace_path, stats, event, model, cut_start, cut_end, inventory, set_header_func = args
        try:
            net, sta, loc, cha = stats.network, stats.station, stats.location, stats.channel
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

            st = read(trace_path)
            st.trim(starttime=t1, endtime=t2)

            if len(st) > 0:
                sd = SeismogramData(st, inventory, fill_gaps=True)
                if self.config is not None:
                    tr = sd.run_analysis(self.config)
                else:
                    tr = st[0]

                tr = set_header_func(tr, distance_km=distance_m / 1000, BAZ=baz, AZ=az,
                                         otime=origin, lat=lat, lon=lon, depth=depth)

                return tr
            else:
                return None

        except Exception as e:
            print(f"[WARNING] Failed to process trace {trace_path}: {e}")
            return None

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

            # Build args for parallel trace processing
            tasks = []
            for key, trace_list in project.project.items():
                for trace_entry in trace_list:
                    tasks.append((
                        trace_entry[0],  # path
                        trace_entry[1],  # stats
                        event,
                        self.model,
                        cut_start,
                        cut_end,
                        self.inventory,
                        self._set_header
                    ))

            with Pool(processes=min(cpu_count(), len(tasks))) as pool:
                results = pool.map(self._process_trace_cut, tasks)

            # Combine valid streams
            full_stream = self._clean_traces(results)
            #self.all_traces.extend(full_stream) #TOO_MUCH MEMORY

            print(f"[INFO] Subproject {i}: {len(full_stream)} traces kept")

            # Optional: save
            if self.output:
                folder = os.path.join(self.output, f"event_{i}")
                os.makedirs(folder, exist_ok=True)
                for tr in full_stream:
                    t1 = tr.stats.starttime
                    base = f"{tr.id}.D.{t1.year}.{t1.julday}"
                    path = os.path.join(folder, base)
                    tr.write(path, format="MSEED")

            if plot and len(full_stream) > 0:
                plotter = PlotProj(full_stream, metadata=self.inventory)
                plotter.plot(traces_per_fig=3, sort_by=None)

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