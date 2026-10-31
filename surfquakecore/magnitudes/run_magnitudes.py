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

import pandas as pd
from obspy import read_events, UTCDateTime, Stream
from obspy.core.event import Catalog, Event, Origin, ResourceIdentifier

from sourcespec.ssp_read_traces import read_traces
from sourcespec.ssp_build_spectra import build_spectra
from sourcespec.ssp_plot_traces import plot_traces
from sourcespec.ssp_inversion import spectral_inversion
from sourcespec.ssp_radiated_energy import radiated_energy_and_apparent_stress
from sourcespec.ssp_local_magnitude import local_magnitude
from sourcespec.ssp_summary_statistics import compute_summary_statistics
from sourcespec.ssp_output import write_output, save_spectra
from sourcespec.ssp_residuals import spectral_residuals
from sourcespec.ssp_plot_spectra import plot_spectra
from sourcespec.ssp_plot_stacked_spectra import plot_stacked_spectra
from sourcespec.ssp_plot_params_stats import box_plots
from sourcespec.ssp_plot_stations import plot_stations
from sourcespec.ssp_html_report import html_report

from surfquakecore.magnitudes.structures import SoursceSpecOptions, EventRecord
from surfquakecore.magnitudes.ssp_setup_mod import configure, setup_logging
from surfquakecore.magnitudes.ssp_process_traces_mod import process_traces
from surfquakecore.project.surf_project import SurfProject


class Automag:

    def __init__(self, project: SurfProject, locations_directory: str, inventory_path, source_config: str,
                 output_directory: str, scale: str, gui_mod=None, events_file: str = None):

        """
        Manage SourceSpec program to estimate source parameters.
        :param project: SurfProject object.
        :param locations_directory: folder with NonLinLoc *.hyp files (use None if events_file is given).
        :param inventory_path: Path to metadata file.
        :param source_config: Path to source config file.
        :param output_directory: Path to output folder.
        :param scale: if regional waveforms will cut with small adapted time windows, else will be cut with a
        long time window
        :param gui_mod: optional dict of config parameters overriding the config file.
        :param events_file: csv/txt with columns origin_time, lat, lon, depth (km) and optionally event_id
        (use None if locations_directory is given).
        """

        self.project = project
        self.locations_directory = locations_directory
        self.events_file = events_file
        self.inventory_path = inventory_path
        self.output_directory = output_directory
        self.scale = scale
        self.gui_mod = gui_mod
        self.events = []
        self.files_path = None
        self.rename_list = []
        self._check_folders()

        if isinstance(source_config, str) and os.path.isfile(source_config):
            self.config_path = source_config
        else:
            raise ValueError(f"source_config {source_config} is not valid. It must be either a "
                             f" valid real_config.ini file or a SourceConfig instance.")

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    def _check_folders(self):

        if bool(self.locations_directory) == bool(self.events_file):
            raise ValueError("Provide exactly one of locations_directory (NLLoc .hyp) or events_file")

        if self.locations_directory and not os.path.isdir(self.locations_directory):
            raise Exception("Loc files directory does not exist")

        if self.events_file and not os.path.isfile(self.events_file):
            raise Exception("Events file does not exist")

        if not os.path.isdir(self.output_directory):
            try:
                os.makedirs(self.output_directory)
            except Exception as error:
                print("An exception occurred:", error)

    def scan_folder(self):
        """NonLinLoc .hyp files -> list[EventRecord] (same naming as before)."""
        events = []
        for top_dir, _, files in os.walk(self.locations_directory):
            for file in files:
                file_hyp = os.path.join(top_dir, file)
                try:
                    o = read_events(file_hyp, format="NLLOC_HYP")[0].origins[0]
                except Exception:
                    continue
                parts = file.split(".")
                if len(parts) >= 3:
                    event_id = parts[1] + "_" + parts[2]
                else:
                    event_id = o.time.strftime("%Y%m%d_%H%M%S")
                events.append(EventRecord(event_id=event_id, evname=file, origin_time=o.time,
                                          latitude=o.latitude, longitude=o.longitude,
                                          depth_km=o.depth * 1E-3, qml_file=file_hyp))
        return events

    def scan_events_file(self):
        """Simple csv/txt (origin_time, lat, lon, depth[km], optional event_id) -> list[EventRecord]."""
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
            if "event_id" in df.columns:
                event_id = str(row["event_id"]).strip()
            else:
                event_id = t.strftime("%Y%m%d_%H%M%S")
            base, k = event_id, 1
            while event_id in used_ids:  # two events in the same second, repeated ids...
                event_id, k = f"{base}_{k}", k + 1
            used_ids.add(event_id)

            lat, lon, depth_km = float(row["lat"]), float(row["lon"]), float(row["depth"])
            origin = Origin(time=t, latitude=lat, longitude=lon, depth=depth_km * 1000.0)  # QuakeML: meters
            ev = Event(origins=[origin], preferred_origin_id=origin.resource_id,
                       resource_id=ResourceIdentifier(f"smi:local/{event_id}"))
            qml_path = os.path.join(qml_dir, f"{event_id}.xml")
            Catalog(events=[ev]).write(qml_path, format="QUAKEML")

            events.append(EventRecord(event_id=event_id, evname=event_id, origin_time=t,
                                      latitude=lat, longitude=lon, depth_km=depth_km,
                                      qml_file=qml_path))
        return events

    def get_now_files(self, date):

        start = date.split(".")
        start = UTCDateTime(year=int(start[1]), julday=int(start[0]), hour=00, minute=00, second=00) + 1
        end = start + (24 * 3600 - 2)
        sp = self.project.copy()
        sp.filter_project_keys()
        self.files_path = sp.filter_time(starttime=start, endtime=end)

        print(self.files_path)

    # ------------------------------------------------------------------
    # SourceSpec call chain
    # ------------------------------------------------------------------
    def _get_config(self, ev: EventRecord):
        return SoursceSpecOptions(config_file=self.config_path, evid=None, evname=ev.evname, hypo_file=None,
                                  outdir=self.output_directory, pick_file=None, qml_file=ev.qml_file,
                                  run_id="", sampleconf=False, station=None,
                                  station_metadata=self.inventory_path, trace_path=self.files_path,
                                  updateconf=None)

    def __cut_signal_wise(self, st, origin_time):

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

    def __run_core_source(self, ev: EventRecord):
        options = self._get_config(ev)

        # Setup stage
        config = configure(options, progname='source_spec')

        if self.gui_mod:
            config = self.modify_config(config)

        setup_logging(config)
        self.rename_list.append([options.outdir, ev.event_id])
        st = read_traces(config)
        st_trim = self.__cut_signal_wise(st, origin_time=ev.origin_time)

        # Deconvolve, filter, cut traces:
        proc_st = process_traces(config, st_trim)

        # Build spectra (amplitude in magnitude units)
        spec_st, specnoise_st, weight_st = build_spectra(config, proc_st)
        plot_traces(config, proc_st)

        # Spectral inversion
        sspec_output = spectral_inversion(config, spec_st, weight_st)

        # Radiated energy and apparent stress
        radiated_energy_and_apparent_stress(config, spec_st, specnoise_st, sspec_output)

        # Local magnitude
        if config.compute_local_magnitude:
            local_magnitude(config, st, proc_st, sspec_output)

        # Compute summary statistics from station spectral parameters
        try:
            compute_summary_statistics(config, sspec_output)
        except Exception:
            print("No able to compute statistics")

        # Save output
        write_output(config, sspec_output)
        save_spectra(config, spec_st)

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

    def estimate_source_parameters(self):
        self.events = self.scan_folder() if self.locations_directory else self.scan_events_file()

        # group by day: same "julday.year" key that get_now_files expects
        by_date = {}
        for ev in self.events:
            key = f"{ev.origin_time.julday}.{ev.origin_time.year}"
            by_date.setdefault(key, []).append(ev)

        for date, evs in by_date.items():
            self.get_now_files(date)

            for ev in evs:
                print(ev.event_id, ev.origin_time, ev.latitude, ev.longitude, ev.depth_km)
                try:
                    self.__run_core_source(ev)
                # SystemExit: sourcespec modules (unmodified) call ssp_exit() -> sys.exit()
                except (Exception, SystemExit) as error:
                    print(f"Error estimating source parameters for event {ev.event_id}: "
                          f"{type(error).__name__}: {error}. Please review the log file.")

        self.rename_folders()

    def rename_folders(self):
        for old_name, event_id in self.rename_list:
            new_name = os.path.join(self.output_directory, event_id)
            try:
                os.rename(old_name, new_name)
            except OSError as error:
                print(f"Couldn't rename {old_name} -> {new_name}: {error}")
