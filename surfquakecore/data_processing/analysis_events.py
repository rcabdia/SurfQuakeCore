# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: analysis_events.py
# Program: surfQuake & ISP
# Date: June 2025
# Purpose: Project Manager
# Author: Roberto Cabieces, Thiago C. Junqueira & C. Palacios
#  Email: rcabdia@roa.es
# --------------------------------------------------------------------

import traceback
import yaml
from obspy import read_inventory, UTCDateTime
from obspy.taup import TauPyModel
import os
from surfquakecore.data_processing.parser.config_parser import parse_configuration_file
from typing import Optional, List
from surfquakecore.data_processing.seismogram_analysis import SeismogramData, StreamProcessing
from surfquakecore.project.surf_project import SurfProject
from multiprocessing import Pool, cpu_count
from obspy.geodetics import gps2dist_azimuth
from obspy import read
from obspy.core.stream import Stream, Trace
from surfquakecore.seismoplot.plot import PlotProj
from collections import defaultdict
import importlib.util
import gc


class AnalysisEvents:

    def __init__(self, output: Optional[str] = None,
                 inventory_file: Optional[str] = None, config_file: Optional[str] = None,
                 surf_projects: List[SurfProject] = None, plot_config_file: Optional[str] = None,
                 post_script: Optional[str] = None,
                 post_script_stage: Optional[str] = "before", time_segment_start: Optional[str] = None,
                 time_segment_end: Optional[str] = None, reference: Optional[str] = None,
                 phase_list: Optional[list] = None, vel: Optional[float] = None,
                 time_segment: Optional[bool] = False):

        self.model = TauPyModel("iasp91")
        self.output = output
        self._exist_folder = False
        self.inventory = None
        self.post_script_func = None
        self.vel = vel
        self.reference = reference
        self.all_traces = []
        self.surf_projects = surf_projects
        self.config_file = config_file
        # Store user-specified time segment (as string or UTCDateTime)
        self.time_segment_start = time_segment_start
        self.time_segment_end = time_segment_end
        self.post_script_stage = post_script_stage
        self.phase_list = phase_list

        if inventory_file:
            try:
                self.inventory = read_inventory(inventory_file)
                print(self.inventory)
            except:
                print("Warning NO VALID inventory file: ", inventory_file)



        self.config = None
        self.df_events = None
        self.time_segment=time_segment

        if self.reference is None:
            self.reference = "event_time"

        if config_file is not None:
            self.config = self.load_analysis_configuration(config_file)

        if self.output is not None:
            if os.path.isdir(self.output):
                print("Output Directory: ", self.output)
            else:
                print("Traces will be process and might be plot: ", self.output)
                self.output = None

        # Load plotting config
        self.plot_config = None
        if plot_config_file and os.path.exists(plot_config_file):
            try:
                with open(plot_config_file, "r") as f:
                    self.plot_config = yaml.safe_load(f).get("plotting", {})
            except Exception as e:
                print(f"[WARNING] Plot config not loaded: {e}")

        # Load post-processing script
        if post_script:
            self._load_post_script(post_script)

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
            distance_m = baz = az = incidence_angle = -1
            lat = lon = depth = None
            arrivals_info = []

            if self.reference == "event_time":

                lat, lon, depth = event["latitude"], event["longitude"], event["depth"]
                inv_sta = inventory.select(network=net, station=sta)
                sta_coords = inv_sta[0][0].latitude, inv_sta[0][0].longitude, inv_sta[0][0].elevation
                distance_m, az, baz = gps2dist_azimuth(sta_coords[0], sta_coords[1], lat, lon)
                distance_deg = distance_m / 1000 / 111.19

                if self.vel:  # Use velocity-based travel time estimate

                    travel_time = distance_m*1E-3 / float(self.vel)  # seconds
                    arrival_time = origin + travel_time
                    t1 = arrival_time - cut_start
                    t2 = arrival_time + cut_end
                    st.trim(starttime=t1, endtime=t2)

                else:

                    if self.phase_list:
                        arrivals = model.get_travel_times(source_depth_in_km=depth, distance_in_degree=distance_deg,
                                                      phase_list=self.phase_list)
                    else:
                        arrivals = model.get_travel_times(source_depth_in_km=depth, distance_in_degree=distance_deg)
                    if not arrivals:
                        raise ValueError("No valid arrivals returned by TauPyModel.")


                    for arr in arrivals:
                        arrivals_info.append({
                            "phase": arr.name,
                            "time": (origin + arr.time).timestamp,
                            "ray_param": arr.ray_param_sec_degree,
                            "incident_angle": arr.incident_angle
                        })

                    incidence_angle = arrivals[0].incident_angle
                    first_arrival = origin + arrivals[0].time
                    t1 = first_arrival - cut_start
                    t2 = first_arrival + cut_end
                    st.trim(starttime=t1, endtime=t2)

            else:
                t1 = origin - cut_start
                t2 = origin + cut_end
                st.trim(starttime=t1, endtime=t2)

            if len(st) == 0:
                return []

            traces = []
            for tr in st:
                if self.config:
                    tr = SeismogramData.run_analysis(tr, self.config, inventory=self.inventory)
                if tr is not None and self.reference == "event_time":
                    tr = set_header_func(tr, distance_km=distance_m / 1000, BAZ=baz, AZ=az,
                                         incidence_angle=incidence_angle, otime=origin.timestamp, lat=lat, lon=lon,
                                         depth=depth, arrivals=arrivals_info)
                traces.append(tr)

            return traces

        except Exception as e:
            print(f"[WARNING] Station group failed: {e}")
            return []

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

    def _set_header(self, tr, distance_km, BAZ, AZ, incidence_angle, otime, lat, lon, depth, arrivals=None):
        tr.stats['geodetic'] = {
            'otime': otime,
            'geodetic': [distance_km, AZ, BAZ, incidence_angle],
            'event': [lat, lon, depth],
            'arrivals': arrivals or []  # ← store arrivals here
        }
        tr.stats.back_azimuth = BAZ
        tr.stats.inclination = incidence_angle
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

    def run_waveform_analysis(self, auto: bool = False):

        plot = not auto
        interactive = False

        if self.surf_projects is None:
            print("No subprojects to process.")
            return

        # Convert string-based time segment (if set) to UTCDateTime
        if hasattr(self, "time_segment_start") and isinstance(self.time_segment_start, str):
            self.time_segment_start = UTCDateTime(self.time_segment_start)
        if hasattr(self, "time_segment_end") and isinstance(self.time_segment_end, str):
            self.time_segment_end = UTCDateTime(self.time_segment_end)

        if isinstance(self.surf_projects, SurfProject):
            self.surf_projects = [self.surf_projects]


        for i, project in enumerate(self.surf_projects):
            while True:
                try:
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

                    if self.post_script_func:
                        results = [self._process_station_analysis(task) for task in tasks]
                    else:
                        with Pool(processes=min(cpu_count(), len(tasks))) as pool:
                            results = pool.map(self._process_station_analysis, tasks)

                    all_traces = [tr for group in results for tr in group if tr is not None]
                    full_stream = self._clean_traces(all_traces)

                    # save memory for further usage
                    del all_traces
                    del results
                    gc.collect()

                    sp = StreamProcessing(full_stream, self.config, self.inventory)
                    full_stream = sp.run_stream_processing()

                    print(f"[INFO] Subproject {i}: {len(full_stream)} traces processed")

                    # If post-script is to run BEFORE plotting
                    if self.post_script_func and self.post_script_stage == "before":
                        try:
                            full_stream = self.post_script_func(full_stream, inventory=self.inventory)
                        except Exception as e:
                            print(f"[WARNING] Post-script (before plotting) failed: {e}")

                    # --- Plot if requested ---
                    if full_stream is not None and plot and len(full_stream) > 0:
                        plotter = PlotProj(full_stream, plot_config=self.plot_config, interactive=interactive)
                        full_stream = plotter.plot()

                        for tr in full_stream:
                            if hasattr(tr.stats, "picks"):
                                print(f"Picks found for {tr.id}: {tr.stats.picks}")

                    if self.post_script_func and self.post_script_stage == "after":
                        try:
                            full_stream = self.post_script_func(full_stream, inventory=self.inventory)
                        except Exception as e:
                            print(f"[WARNING] Post-script (after plotting) failed: {e}")

                    # Save output only in auto mode
                    if auto and self.output:
                        self._write_files(full_stream)

                    # Prompt only in interactive mode
                    if plot:
                        user_choice = input(
                            f"\n[Prompt] Finished subproject {i}. Type 'n' to continue, "
                            f"'r' to reprocess this event, or 'exit': "
                        ).strip().lower()

                        if user_choice == "c":
                            break
                        elif user_choice == "r":
                            if self.config_file:
                                print(f"[INFO] Reloading config and reprocessing subproject {i}...")
                                self.config = self.load_analysis_configuration(self.config_file)
                            continue
                        elif user_choice == "exit":
                            print("[INFO] Exiting waveform analysis by user request.")
                            return
                        else:
                            print("[WARN] Unknown command. Assuming 'next'.")
                            break
                    else:
                        break  # automatic mode just continues

                except Exception as e:
                    print(f"[ERROR] Exception in subproject {i}, {e}")
                    traceback.print_exc()
                    print("[INFO] Skipping to next event...")
                    break  # go to next event despite the error

    def run_waveform_cutting(self, cut_start: float, cut_end: float, auto=False):

        plot = not auto
        interactive = False

        if self.surf_projects is None:
            print("No projects to process.")
            return

        for i, project in enumerate(self.surf_projects):
            events = getattr(project, "_events_metadata", [])
            if isinstance(events, dict):
                events = [events]
            if not events:
                print(f"[INFO] Skipping subproject {i}: no associated events")
                continue

            for j, event in enumerate(events):
                while True:
                    try:
                        print(f"[INFO] Cutting traces for subproject {i}, event {j} at {event['origin_time']}")

                        # --- Prepare station-wise file groups ---
                        station_files = defaultdict(list)
                        for trace_list in project.project.values():
                            for trace_path, stats in trace_list:
                                station_key = f"{stats.network}.{stats.station}"
                                station_files[station_key].append((trace_path, stats))

                        # --- Create tasks for each station ---
                        tasks = []
                        for file_group in station_files.values():
                            tasks.append((
                                file_group, event, self.model, cut_start, cut_end,
                                self.inventory, self._set_header
                            ))

                        # --- Process stations (parallel if no post-script) ---
                        if self.post_script_func:
                            #print("[INFO] Using sequential mode due to post-script")
                            results = [self._process_station_traces(task) for task in tasks]
                        else:
                            with Pool(processes=min(cpu_count(), len(tasks))) as pool:
                                results = pool.map(self._process_station_traces, tasks)

                        # --- Gather and clean traces ---
                        all_traces = [tr for sub in results for tr in sub if tr is not None]
                        full_stream = self._clean_traces(all_traces)

                        del all_traces
                        del results
                        gc.collect()

                        sp = StreamProcessing(full_stream, self.config, self.inventory)
                        full_stream = sp.run_stream_processing()

                        print(f"[INFO] Subproject {i}, event {j}: {len(full_stream)} traces kept")

                        # If post-script is to run BEFORE plotting
                        if self.post_script_func and self.post_script_stage == "before":
                            try:
                                full_stream = self.post_script_func(full_stream, inventory=self.inventory,
                                                                    event=event)
                            except Exception as e:
                                print(f"[WARNING] Post-script (before plotting) failed: {e}")

                        # --- Plot if requested ---
                        if full_stream is not None and plot and len(full_stream) > 0:
                            plotter = PlotProj(full_stream, plot_config=self.plot_config, interactive=interactive,
                                               inventory=self.inventory)
                            try:
                                full_stream = plotter.plot()
                            except Exception as e:
                                print(f"[ERROR] Plotting failed: {e}")

                            for tr in full_stream:
                                if hasattr(tr.stats, "picks"):
                                    print(f"Picks found for {tr.id}: {tr.stats.picks}")

                        # --- Post-script (optional) --- # Might be user has edited the header of full_stream traces
                        if self.post_script_func and self.post_script_stage == "after":
                            try:
                                full_stream = self.post_script_func(full_stream, inventory=self.inventory,
                                                                    event=event)
                            except Exception as e:
                                print(f"[WARNING] Post-script (after plotting) failed: {e}")

                        # --- Save output if requested ---
                        if auto and self.output:
                            self._write_files(full_stream)

                        # --- User prompt for next action ---
                        if plot:
                            user_choice = input(
                                f"\n[Prompt] Finished subproject {i}, event {j}. Type 'c' to continue, "
                                f"'r' to reprocess this event, or 'exit': "
                            ).strip().lower()

                            if user_choice == "c":
                                break  # Exit the while-loop → go to next event

                            elif user_choice == "r":
                                if self.config:
                                    print(f"[INFO] Loading parametrization and Reprocessing subproject {i}, event {j}...")
                                    self.config = self.load_analysis_configuration(self.config_file)
                                continue  # Rerun same event

                            elif user_choice == "exit":
                                print("[INFO] Exiting waveform cutting by user request.")
                                return  # Exit entire `run_waveform_cutting`

                            else:
                                print("[WARN] Unknown command. Assuming 'next'.")
                                break
                        else:
                            break  # No prompt: go to next event

                    except Exception as e:
                        print(f"[ERROR] Exception in subproject {i}, event {j}: {e}")
                        traceback.print_exc()
                        print("[INFO] Skipping to next event...")
                        break  # go to next event despite the error

    def run_fast_waveform_analysis(self, data_files, auto: bool = True):

        """
        Lightweight waveform analysis for raw file-based input using self.data_files.
        Skips event/model logic, cuts, and segment filters.
        """

        plot = not auto
        interactive = False

        station_files = defaultdict(list)

        for file_path in data_files:
            try:
                st = read(file_path)
                for tr in st:
                    key = f"{tr.stats.network}.{tr.stats.station}"
                    station_files[key].append((file_path, tr.stats))
            except Exception as e:
                print(f"[WARNING] Failed to read file {file_path}: {e}")
                continue

        tasks = [
            (file_group, None, None, 0, 0, self.inventory, self._set_header)
            for file_group in station_files.values()
        ]

        try:
            while True:
                if self.post_script_func:

                    results = [self._process_station_analysis(task) for task in tasks]
                else:
                    with Pool(processes=min(cpu_count(), len(tasks))) as pool:
                        results = pool.map(self._process_station_analysis, tasks)

                all_traces = [tr for group in results for tr in group if tr is not None]
                full_stream = self._clean_traces(all_traces)

                del all_traces, results
                gc.collect()

                sp = StreamProcessing(full_stream, self.config, self.inventory)
                full_stream = sp.run_stream_processing()

                print(f"[INFO] Fast mode: {len(full_stream)} traces processed")

                if self.post_script_func and self.post_script_stage == "before":
                    try:
                        full_stream = self.post_script_func(full_stream, inventory=self.inventory)
                    except Exception as e:
                        print(f"[WARNING] Post-script (before plotting) failed: {e}")


                if full_stream and plot:
                    plotter = PlotProj(full_stream, plot_config=self.plot_config, inventory=self.inventory,
                                       interactive=interactive, data_files=data_files)
                    full_stream = plotter.plot()

                    for tr in full_stream:
                        if hasattr(tr.stats, "picks"):
                            print(f"Picks found for {tr.id}: {tr.stats.picks}")

                if self.post_script_func and self.post_script_stage == "after":
                    try:
                        full_stream = self.post_script_func(full_stream, inventory=self.inventory)
                    except Exception as e:
                        print(f"[WARNING] Post-script (after plotting) failed: {e}")

                if auto and self.output:
                    self._write_files(full_stream)

                # --- User prompt for next action ---
                if plot:
                    user_choice = input(
                        f"\n[Prompt] Finished processing. Type 'c' to continue, "
                        f"'r' to reprocess this event, or 'exit': "
                    ).strip().lower()

                    if user_choice == "c":
                        break  # Exit the while-loop → go to next event

                    elif user_choice == "r":
                        if self.config_file:
                            print(f"[INFO] Loading parametrization and Reprocessing...")
                            self.config = self.load_analysis_configuration(self.config_file)
                        continue  # Rerun same event

                    elif user_choice == "exit":
                        print("[INFO] Exiting waveform cutting by user request.")
                        return  # Exit entire `run_waveform_cutting`

                    else:
                        print("[WARN] Unknown command. Assuming 'next'.")
                        break
                else:
                    break  # No prompt: go to next event

        except Exception as e:
            print(f"[ERROR] Exception during fast waveform analysis: {e}")
            import traceback
            traceback.print_exc()

    def _process_station_analysis(self, args):

        file_group, _, _, _, _, inventory, set_header_func = args
        try:
            st = Stream()
            for trace_path, _ in file_group:
                st += read(trace_path)

            # TODO: This line is very important, default behaviour is not merging traces /
            #  in processing_daily when time_segment automatically merge the stream

            if self.time_segment:
                st.merge(method=1, fill_value='interpolate')

            # Trim to time segment if defined
            if hasattr(self, "time_segment_start") and hasattr(self, "time_segment_end"):
                st.trim(starttime=self.time_segment_start, endtime=self.time_segment_end)

            if len(st) == 0:
                return []

            stats = file_group[0][1]
            # net, sta = stats.network, stats.station

            traces = []
            for tr in st:
                if self.config:
                    tr = SeismogramData.run_analysis(tr, self.config, inventory=self.inventory)

                tr = set_header_func(tr, distance_km=-1, BAZ=-1, AZ=-1, incidence_angle=-1,
                                     otime=None, lat=None, lon=None, depth=None)
                traces.append(tr)

            return traces

        except Exception as e:
            print(f"[WARNING] Failed station processing: {e}")
            return []

    def _load_post_script(self, script_path: str):
        if not os.path.exists(script_path):
            print(f"[WARNING] Post-script file not found: {script_path}")
            return

        try:
            spec = importlib.util.spec_from_file_location("post_module", script_path)
            post_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(post_module)

            if hasattr(post_module, "run") and callable(post_module.run):
                self.post_script_func = post_module.run
                print(f"[INFO] Post-script loaded successfully from: {script_path}")
            else:
                print(f"[WARNING] Script {script_path} must define a callable `run(stream, inventory, event=None)`")
        except Exception as e:
            print(f"[ERROR] Failed to import post-script {script_path}: {e}")

# def _safe_plot_worker(stream, plot_config, inventory, interactive, queue):
#     try:
#         from surfquakecore.seismoplot.plot import PlotProj  # import locally to isolate
#         plotter = PlotProj(stream, plot_config=plot_config, interactive=interactive, inventory=inventory)
#         modified = plotter.plot()
#         queue.put(modified)
#     except Exception as e:
#         import traceback
#         print("[ERROR] Plotting subprocess crashed:", e)
#         traceback.print_exc()
#         queue.put(None)
#
#     def compute_theoretical_arrivals(self, phases=None, model_name='ak135', npoints=25):
#         """
#         Compute theoretical arrival times for selected phases and distances.
#
#         Returns
#         -------
#         dict[str, list[tuple[float, float]]]
#             phase name → list of (matplotlib_time, distance_km)
#         """
#         from obspy.taup import TauPyModel
#
#         model = TauPyModel(model_name)
#         distances_km = [self._get_geodetic_info(tr)[0] for tr in self.trace_list]
#         otime = self.trace_list[0].stats['geodetic']['otime']
#         depth = self.trace_list[0].stats['geodetic']['event'][2]
#
#         min_dist = min(distances_km)
#         max_dist = max(distances_km)
#         dist_km_array = np.linspace(min_dist, max_dist, npoints)
#         dist_deg_array = dist_km_array / 111.19
#
#         phase_curves = {}
#
#         for dist_km, dist_deg in zip(dist_km_array, dist_deg_array):
#             try:
#                 if phases is not None:
#                     arrivals = model.get_travel_times(
#                         source_depth_in_km=depth,
#                         distance_in_degree=dist_deg,
#                         phase_list=phases
#                     )
#                 else:
#                     arrivals = model.get_travel_times(
#                         source_depth_in_km=depth,
#                         distance_in_degree=dist_deg
#                     )
#                 if arrivals is None:
#                     print(f"[DEBUG] No arrivals at dist={dist_km:.1f} km for model={model_name}")
#                 for arr in arrivals:
#                     arr_time = otime + arr.time
#                     if arr.name not in phase_curves:
#                         phase_curves[arr.name] = []
#                     phase_curves[arr.name].append((mdt.date2num(arr_time.datetime), dist_km))
#             except Exception as e:
#                 print(f"[WARNING] TauPy error at dist={dist_km:.1f} km: {e}")
#         del model
#         gc.collect()
#         return phase_curves
