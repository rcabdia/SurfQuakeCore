# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: run_magnitudes.py
# Program: surfQuake & ISP
# Date: January 2024
# Purpose: Manage Event Locator
# Author: Roberto Cabieces, Thiago C. Junqueira & Claudio Satriano
#  Email: rcabdia@roa.es
# --------------------------------------------------------------------

import gc
import os
from obspy import read, read_events, UTCDateTime, Stream
from sourcespec.ssp_setup import configure
from sourcespec.ssp_read_traces import read_traces
from surfquakecore.magnitudes.ssp_process_traces_mod import process_traces
from sourcespec.ssp_build_spectra import build_spectra
from sourcespec.ssp_plot_traces import plot_traces
from sourcespec.ssp_inversion import spectral_inversion
from sourcespec.ssp_radiated_energy import radiated_energy
from sourcespec.ssp_local_magnitude import local_magnitude
from sourcespec.ssp_summary_statistics import compute_summary_statistics
from sourcespec.ssp_output import write_output
from sourcespec.ssp_residuals import spectral_residuals
from sourcespec.ssp_plot_spectra import plot_spectra
from sourcespec.ssp_plot_stacked_spectra import plot_stacked_spectra
from sourcespec.ssp_plot_params_stats import box_plots
from sourcespec.ssp_plot_stations import plot_stations
from sourcespec.ssp_html_report import html_report
from surfquakecore.magnitudes.ssp_setup_mod import setup_logging
from surfquakecore.project.surf_project import SurfProject
import pandas as pd
from obspy.core.event import Catalog, Event, Origin, ResourceIdentifier
from surfquakecore.magnitudes.structures import SoursceSpecOptions, EventRecord

class Automag:

    def __init__(self, project: SurfProject, locations_directory: str, inventory_path, source_config: str,
             output_directory: str, scale: str, gui_mod=None, events_file: str = None):

        """
        Manage SourceSpec program to estimate source parameters.
        :param project: SurfProject object.
        :param inventory_path: Path to metadata file.
        :param source_config: Path to source config file.
        :param output_directory: Path to output folder.
        :param scale: if regional waveforms will cut with small adapted time windows, else will be cut with a
        long time window
        """

        self.project = project
        self.locations_directory = locations_directory
        self.inventory_path = inventory_path
        self.output_directory = output_directory
        self.scale = scale
        self.gui_mod = gui_mod
        self.rename_list = []
        self._check_folders()
        self.events_file = events_file
        self.events = []
        self.files_path = None

        if isinstance(source_config, str) and os.path.isfile(source_config):
            self.config_path = source_config
        else:
            raise ValueError(f"source_config {source_config} is not valid. It must be either a "
                             f" valid real_config.ini file or a SourceConfig instance.")

    # Old method
    # def _get_config(self, event, id_name):
    #     id_list = id_name.split(".")
    #     id = id_list[0]+"/"+id_list[1]+"_"+id_list[2]
    #
    #     options = SoursceSpecOptions(config_file=self.config_path, evid=None, evname=id_name, hypo_file=None,
    #         outdir=self.output_directory, pick_file=None, qml_file=event, run_id="", sampleconf=False, station=None,
    #         station_metadata=self.inventory_path, trace_path=self.files_path, updateconf=None)
    #
    #     return options

    # New method under review
    def _get_config(self, ev: EventRecord):

        return SoursceSpecOptions(config_file=self.config_path, evid=None, evname=ev.evname, hypo_file=None,
                                  outdir=self.output_directory, pick_file=None, qml_file=ev.qml_file, run_id="",
                                  sampleconf=False, station=None, station_metadata=self.inventory_path,
                                  trace_path=self.files_path, updateconf=None)

    def _check_folders(self):

        if bool(self.locations_directory) == bool(self.events_file):
            raise ValueError("Provide exactly one of locations_directory (NLLoc .hyp) or events_file")
        if self.locations_directory and not os.path.isdir(self.locations_directory):
            raise Exception("Loc files directory does not exist")
        if self.events_file and not os.path.isfile(self.events_file):
            raise Exception("Events file does not exist")


    def get_now_files(self, date):

        start = date.split(".")
        start = UTCDateTime(year=int(start[1]), julday=int(start[0]), hour=00, minute=00, second=00) + 1
        end = start + (24 * 3600 - 2)
        sp = self.project.copy()
        sp.filter_project_keys()
        self.files_path = sp.filter_time(starttime=start, endtime=end)

        print(self.files_path)

    def filter_station(self, station):

        filtered_list = []

        for file in self.files_path:
            header = read(file, headlonly=True)
            sta = header[0].stats.station
            if station == sta:
                filtered_list.append(file)

        return filtered_list

    # def scan_folder(self):
    #     obsfiles1 = []
    #     dates = {}
    #     for top_dir, _, files in os.walk(self.locations_directory):
    #
    #         for file in files:
    #             try:
    #                 file_hyp = os.path.join(top_dir, file)
    #                 cat = read_events(file_hyp, format="NLLOC_HYP")
    #                 ev = cat[0]
    #                 date = ev.origins[0]["time"]
    #                 date = str(date.julday) + "." + str(date.year)
    #
    #                 obsfiles1.append(file_hyp)
    #                 if date not in dates:
    #                     dates[date] = [file_hyp]
    #                 else:
    #                     dates[date].append(file_hyp)
    #             except:
    #                 pass
    #
    #     self.dates = dates
    #
    # def scan_from_origin(self, origin):
    #
    #     self.date = origin["time"]

    def scan_folder(self):
        """NLLoc .hyp files -> list[EventRecord] (same naming as before)"""
        events = []
        for top_dir, _, files in os.walk(self.locations_directory):
            for file in files:
                file_hyp = os.path.join(top_dir, file)
                try:
                    o = read_events(file_hyp, format="NLLOC_HYP")[0].origins[0]
                except Exception:
                    continue
                parts = file.split(".")
                event_id = parts[1] + "_" + parts[2] if len(parts) >= 3 else o.time.strftime("%Y%m%d_%H%M%S")
                events.append(EventRecord(event_id=event_id, evname=file, origin_time=o.time,
                                          latitude=o.latitude, longitude=o.longitude,
                                          depth_km=o.depth * 1E-3, qml_file=file_hyp))
        return events

    def scan_events_file(self):
        """Simple csv/txt (origin_time, lat, lon, depth[km], optional event_id) -> list[EventRecord]"""
        df = pd.read_csv(self.events_file, sep=None, engine="python", comment="#")
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={"latitude": "lat", "longitude": "lon", "depth_km": "depth",
                                "time": "origin_time", "origin": "origin_time"})
        missing = {"origin_time", "lat", "lon", "depth"} - set(df.columns)
        if missing:
            raise ValueError(f"Events file lacks columns: {sorted(missing)}")

        qml_dir = os.path.join(self.output_directory, "_events_qml")
        os.makedirs(qml_dir, exist_ok=True)

        events, used_ids = [], set()
        for _, row in df.iterrows():
            t = UTCDateTime(str(row["origin_time"]).strip())
            event_id = str(row["event_id"]).strip() if "event_id" in df.columns else t.strftime("%Y%m%d_%H%M%S")
            base, k = event_id, 1
            while event_id in used_ids:  # two events in the same second, etc.
                event_id, k = f"{base}_{k}", k + 1
            used_ids.add(event_id)

            origin = Origin(time=t, latitude=float(row["lat"]), longitude=float(row["lon"]),
                            depth=float(row["depth"]) * 1000.0)  # QuakeML depth is in meters
            ev = Event(origins=[origin], preferred_origin_id=origin.resource_id,
                       resource_id=ResourceIdentifier(f"smi:local/{event_id}"))
            qml_path = os.path.join(qml_dir, f"{event_id}.xml")
            Catalog(events=[ev]).write(qml_path, format="QUAKEML")

            events.append(EventRecord(event_id=event_id, evname=event_id, origin_time=t,
                                      latitude=float(row["lat"]), longitude=float(row["lon"]),
                                      depth_km=float(row["depth"]), qml_file=qml_path))
        return events
    def _get_stations(self, arrivals):
        stations = []
        for pick in arrivals:
            if pick.station not in stations:
                stations.append(pick.station)

        return stations

    def __cut_signal_wise(self, st,  origin_time):

        all_traces = []
        st.merge()

        if self.scale == "regional":
            dt_noise = 10
            dt_signal = 10 * 60
        else:
            dt_noise = 10
            dt_signal = 2700

        start = origin_time - dt_noise
        end = origin_time + dt_signal

        for tr in st:

            tr.trim(starttime=start, endtime=end)
            all_traces.append(tr)
            # TODO: It is not still checked the fill_gaps functionality
            #tr = cls.fill_gaps(tr)
        st = Stream(traces=all_traces)
        return st

    def modify_config(self, config):
        for key in config:
            for gui_key in self.gui_mod:
                if key == gui_key:
                    config[key] = self.gui_mod[key]
        return config

    # def __run_core_source(self, event, id_name, focal_parameters):
    def __run_core_source(self, ev: EventRecord):
        options = self._get_config(ev)
        config = configure(options, progname='source_spec')
        if self.gui_mod:
            config = self.modify_config(config)
        setup_logging(config)
        self.rename_list.append([options.outdir, ev.event_id])
        st = read_traces(config)
        st_trim = self.__cut_signal_wise(st, origin_time=ev.origin_time)

        ######### Old way ###########
        # options = self._get_config(event, id_name)
        #
        # # Setup stage
        # config = configure(options, progname='source_spec')
        #
        # if self.gui_mod:
        #     config = self.modify_config(config)
        #
        # setup_logging(config)
        # name_event = id_name.split(".")
        # name_event_id = name_event[1]+"_"+name_event[2]
        # self.rename_list.append([options.outdir, name_event_id])
        # st = read_traces(config)
        # st_trim = self.__cut_signal_wise(st, origin_time=focal_parameters[0])
        ######### Old way ###########

        # Deconvolve, filter, cut traces:
        proc_st = process_traces(config, st_trim)

        # Build spectra (amplitude in magnitude units)
        spec_st, specnoise_st, weight_st = build_spectra(config, proc_st)
        plot_traces(config, proc_st)

        # Spectral inversion
        sspec_output = spectral_inversion(config, spec_st, weight_st)

        # Radiated energy
        radiated_energy(config, spec_st, specnoise_st, sspec_output)

        # Local magnitude
        if config.compute_local_magnitude:
            local_magnitude(config, st, proc_st, sspec_output)

        # Compute summary statistics from station spectral parameters
        try:
            compute_summary_statistics(config, sspec_output)
        except:
            print("No able to compute statistics")


        # Save output
        write_output(config, sspec_output)

        # Save residuals
        spectral_residuals(config, spec_st, sspec_output)

        # Plotting
        plot_spectra(config, spec_st, specnoise_st, plot_type='regular')
        plot_spectra(config, weight_st, plot_type='weight')
        plot_stacked_spectra(config, spec_st, sspec_output)
        box_plots(config, sspec_output)
        if config.plot_station_map:
            plot_stations(config, sspec_output)

        if config.html_report:
            html_report(config, sspec_output)


        del options
        del config
        del st
        del st_trim
        gc.collect()

    #### old method ###
    # def estimate_source_parameters(self):
    #     # Loop over loc folder files and run source parameters estimation
    #     self.scan_folder()
    #     events_names = []
    #     for date in self.dates.keys():
    #
    #         events = self.dates[date]
    #         self.get_now_files(date)
    #
    #         for event in events:
    #
    #             print(event)
    #             cat = read_events(event)
    #             focal_parameters = [cat[0].origins[0]["time"], cat[0].origins[0]["latitude"],
    #                                 cat[0].origins[0]["longitude"],
    #                                 cat[0].origins[0]["depth"] * 1E-3]
    #             name_event = (str(cat[0].origins[0]["time"]))
    #
    #             events_names.append(name_event)
    #
    #             try:
    #                 run_id_name = os.path.basename(event)
    #                 self.__run_core_source(event, run_id_name, focal_parameters)
    #             except:
    #                 print(f"Error occurred trying to estimate source parameters at event, please review log file: "
    #                       f"{run_id_name}")
    #     try:
    #         self.rename_folders()
    #     except:
    #         print("Coudn't rename folders")

    #### new method ###
    def estimate_source_parameters(self):
        self.events = self.scan_folder() if self.locations_directory else self.scan_events_file()

        by_date = {}
        for ev in self.events:
            key = f"{ev.origin_time.julday}.{ev.origin_time.year}"  # same "julday.year" key get_now_files expects
            by_date.setdefault(key, []).append(ev)

        for date, evs in by_date.items():
            self.get_now_files(date)
            for ev in evs:
                print(ev.event_id, ev.origin_time, ev.latitude, ev.longitude, ev.depth_km)
                try:
                    self.__run_core_source(ev)
                except Exception as error:
                    print(f"Error estimating source parameters for event {ev.event_id}: {error}")
        try:
            self.rename_folders()
        except Exception:
            print("Couldn't rename folders")

    def rename_folders(self):
        for item in self.rename_list:
            old_name = item[0]
            new_name = os.path.join(self.output_directory,  item[1])
            os.rename(old_name, new_name)