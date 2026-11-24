#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Filename: plot.py
# Program: surfQuake & ISP
# Date: May 2025
# Purpose: Plotting tools for processing
# Author: Roberto Cabieces, Thiago C. Junqueira & Cristina Palacios
# Email: rcabdia@roa.es
"""

import math
import os
import time
from collections import defaultdict
from datetime import timedelta, datetime
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter, FuncFormatter, Formatter
from obspy import UTCDateTime, Stream
from obspy.core.trace import Trace
import matplotlib as mplt
import matplotlib.dates as mdt
import numpy as np
from obspy.signal.polarization import flinn
from surfquakecore.arrayanalysis import array_analysis
from surfquakecore.data_processing.spectral_tools import SpectrumTool
from surfquakecore.data_processing.wavelet import ConvolveWaveletScipy
from surfquakecore.seismoplot.crosshair import MultiAxesCrosshair, CROSSHAIR_MODES
from surfquakecore.seismoplot.fast_plot import FastLine
from surfquakecore.seismoplot.plot_command_prompt import PlotCommandPrompt
from surfquakecore.seismoplot.spanselector import ExtendSpanSelector
from surfquakecore.utils.obspy_utils import MseedUtil
from matplotlib.backend_bases import MouseButton
PSEUDO_MODES = ("music", "ev")


class DayTimeFormatter(Formatter):
    """
    Time-axis formatter for matplotlib dates.

    Tick labels show HH:MM:SS (or HH:MM:SS.t). The date (YYYY-MM-DD) is added
    under the first tick and under every tick where the day changes, so plots
    spanning several days stay readable. Works with zoom and pan.
    The cursor read-out (status bar) shows the full date and time.
    """

    def __init__(self, tenths=True):
        self.tenths = tenths

    def _round(self, x):
        # round to the nearest shown unit (0.1 s or 1 s)
        return mdt.num2date(x) + timedelta(milliseconds=50 if self.tenths else 500)

    def _time(self, d):
        return f"{d:%H:%M:%S}.{d.microsecond // 100000}" if self.tenths else f"{d:%H:%M:%S}"

    def __call__(self, x, pos=None):
        d = self._round(x)
        return f"{d:%Y-%m-%d} {self._time(d)}"

    def format_ticks(self, values):
        labels = []
        prev_day = None
        for x in values:
            d = self._round(x)
            label = self._time(d)
            if d.date() != prev_day:
                label += f"\n{d:%Y-%m-%d}"
            prev_day = d.date()
            labels.append(label)
        return labels
# Choose the best backend before importing pyplot,
# more Options: TkAgg, MacOSX, Qt5Agg, QtAgg, WebAgg, Agg

class PlotProj:
    def __init__(self, stream,
                 plot_config: Optional[dict] = None, **kwargs):

        """
        Parameters
        ----------
        stream : Stream or list of Traces
        plot_config : dict, optional
            Dictionary of plotting preferences.
        """
        self.available_modes = ["standard", "overlay", "record"]
        self.trace_list = list(stream)
        self.inventory = kwargs.pop("inventory", None)
        self.__selector = {}  # Use single underscore to avoid name mangling
        self.__selected_ax_index = 0
        self.current_page = 0  # to track which subplot page is currently shown
        # Default plotting configuration
        self.plot_config = {
            "traces_per_fig": 6,
            "sort_by": False,  # options: distance, backazimuth, None
            "vspace": 0.05,
            "title_fontsize": 9,
            "show_legend": True,
            "plot_type": "standard",  #'record' for record section and overlay for all traces at the same plot
            "record_option": "dates",
            "scale_factor": 1,
            "sharey": False,
            "show_arrivals": False,
            "show_info_picks": False,
            "pick_output_file": "./picks.csv",
            "auto_load_pick_file": False,
            "show_crosshair": False,
            "show_help": True,
            "plot_output_path": None,  # <-- NEW: folder to auto-save figures; None = disabled
            "show_plot": True,  # <-- NEW: set False for headless/batch saving
            "format_save": "png",
            "fast": False,  # QDP-style min/max decimated plotting
            "fast_points": 1000,
            "back2future": False,
            "backend": "TkAgg"}


        self.available_types = ['standard', 'record', 'overlay']
        self.enable_command_prompt = kwargs.pop("interactive", False)

        # Override defaults with user config if provided
        if plot_config:
            self.plot_config.update(plot_config)

        # NEW: pristine backup only if requested (copy can be slow for large datasets)
        if self.plot_config.get("back2future", False):
            self.original_traces = [tr.copy() for tr in self.trace_list]
        else:
            self.original_traces = None

        if self.plot_config["show_help"]:
            self._print_help_table()

        self.picks = {}
        self.current_pick = None
        self.pick_lines = {}
        self.events = []  # events imported from a multi-event pick file (dicts)
        self.event_artists = []  # event lines + labels currently drawn
        self._pick_event = {}  # (trace_id, round(timestamp, 3)) -> index in self.events
        self.fig = None
        self.axs = None
        self.pick_type = "P"
        self.prompt_active = False
        self.utc_start = None
        self.utc_end = None
        self._last_click_info = None  # stores (ax, xdata) after 'e' press
        self._hover = None  # (trace_id, line, label, pick_time) when hovering a pick
        self._hover_tol_px = 8  # pixel tolerance for hover hit testing

        self.trace_list = list(stream)

        # Keys that must not be changed from the plot config
        self._RC_BLACKLIST = {"backend", "interactive"}
        self._rc = {}

    def _get_rcparams(self) -> dict:
        """Validated Matplotlib rcParams taken from plot_config['rcparams']."""
        raw = self.plot_config.get("rcparams") or {}
        if not isinstance(raw, dict):
            print("[WARNING] 'rcparams' must be a dictionary, ignored")
            return {}

        valid = {}
        for key, value in raw.items():
            if key in self._RC_BLACKLIST:
                print(f"[WARNING] rcParam '{key}' cannot be set from the plot config, ignored")
                continue
            try:
                valid[key] = mplt.rcParams.validate[key](value)
            except KeyError:
                print(f"[WARNING] Unknown Matplotlib rcParam '{key}', ignored")
            except (ValueError, TypeError) as e:
                print(f"[WARNING] Invalid value for rcParam '{key}': {value!r} ({e}), ignored")
        return valid

    def _add_date_box(self, ax, starttime, endtime=None):
        """
        Date box in the upper-left corner of ax (same style as the spectrogram):
        JD / year and the start date-time. If endtime falls on another day,
        the end JD and date-time are added below.
        Disabled with plot_config['show_date'] = False.
        """
        if not self.plot_config.get("show_date", True):
            return
        lines = [f"JD {starttime.julday} / {starttime.year}",
                 starttime.strftime("%Y-%m-%d %H:%M:%S")]
        if endtime is not None and (endtime.year, endtime.julday) != (starttime.year, starttime.julday):
            lines += [f"to JD {endtime.julday} / {endtime.year}",
                      endtime.strftime("%Y-%m-%d %H:%M:%S")]
        ax.text(0.01, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=8,
                va="top", ha="left", zorder=5,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.5))

    def make_abs(self, path: Optional[str]) -> Optional[str]:
        return os.path.abspath(os.path.expanduser(path)) if path else None

    def _unique_figure_path(self, out_dir, kind, page, format_save, traces):
        """
        Path of an auto-saved figure, named after the start time of the plotted data:
            sqProject[_<kind>]_page<NNN>_<YYYYmmdd-HHMMSS>.<format>
        so each event or time span gets its own name. If the file already exists,
        _1, _2, ... is appended, so figures are never overwritten.
        """
        try:
            t0 = min(tr.stats.starttime for tr in traces)
            stamp = t0.strftime("%Y%m%d-%H%M%S")
        except (ValueError, AttributeError):  # no traces: fall back to the clock
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        prefix = f"sqProject_{kind}" if kind else "sqProject"
        base_name = f"{prefix}_page{page:03d}_{stamp}"
        out_file = os.path.join(out_dir, f"{base_name}.{format_save}")
        counter = 1
        while os.path.exists(out_file):
            out_file = os.path.join(out_dir, f"{base_name}_{counter}.{format_save}")
            counter += 1
        return out_file

    def _crosshair_mode(self):
        """Crosshair mode from plot_config['show_crosshair'], or None if disabled.
        Accepts false/true (true = single) or single|vertical|horizontal|all."""
        value = self.plot_config.get("show_crosshair", False)
        if value is True:
            return "single"
        if not value:
            return None
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("false", "no", "off", "none"):
                return None
            if v in ("true", "yes", "on"):
                return "single"
            if v in CROSSHAIR_MODES:
                return v
        print(f"[WARNING] show_crosshair '{value}' not recognized "
              f"(use false or {', '.join(CROSSHAIR_MODES)}), using 'single'")
        return "single"

    def _print_help_table(self):
        """Print available keyboard commands in a table format."""

        commands = [
            ("KEY", "ACTION"),
            ("---", "------"),
            ("n", "Next page"),
            ("b", "Previous page"),
            ("w", "Add reference time"),
            ("p", "Remove last reference"),
            ("c", "Clear all picks"),
            ("e", "Manual pick (interactive input)"),
            ("1–6", "Quick picks (P/S + polarity)"),
            ("d / Del", "Delete pick under cursor"),
            ("v", "Open command prompt"),
            ("u", "Set start time at cursor"),
            ("i", "Set end time at cursor"),
            ("Right-click drag", "Select time window"),
            ("Double left-click", "Add pick"),
            ("ESC / Enter", "Exit current mode"),
        ]

        print("\n" + "=" * 55)
        print("Plot Interaction Help")
        print("=" * 55)

        for key, desc in commands:
            print(f"{key:<15} | {desc}")

        print("=" * 55 + "\n")

    def _print_fk_help(self):

        commands = [
            ("KEY", "ACTION"),
            ("---", "------"),
            ("1", "Conventional FK"),
            ("2", "Capon"),
            ("3", "Multitaper Coherence"),
            ("4", "Music for 1 signal"),
            ("5", "Music for 2 signals"),
            ("6", "Music for 3 signals"),
        ]

        print("\n" + "=" * 55)
        print("Plot Interaction Help, hold mouse over power pannel at any time.\n Press key '1-6' to select a method, "
              "then press the key 'e' to run the process")
        print("=" * 55)

        for key, desc in commands:
            print(f"{key:<15} | {desc}")

        print("=" * 55 + "\n")
    def _set_output_target(self, path: str, fmt: str):
        self.plot_config["pick_output_file"] = path
        self.plot_config["pick_output_format"] = fmt  # "NLLOC_OBS" or "ISP_TABLE"
        print(f"[INFO] Output target set → {fmt}: {path}")

    def _get_geodetic_info(self, trace: Trace) -> Tuple[float, float]:
        """
        Returns distance (km) and back-azimuth from trace header.
        """
        try:
            return trace.stats.geodetic['geodetic'][0], trace.stats.geodetic['geodetic'][2]
        except Exception:
            return float('inf'), float('inf')

    def plot(self, page=0):
        backend = "Agg" if not self.plot_config.get("show_plot", True) else self.plot_config["backend"]
        mplt.use(backend)

        self._rc = self._get_rcparams()
        plot_type = self.plot_config.get("plot_type", "standard")

        with mplt.rc_context(self._rc):
            if plot_type == "standard" and not self.plot_config.get("show_plot", True):
                # Batch mode: save every page automatically, no interaction needed
                traces_per_fig = self.plot_config["traces_per_fig"]
                n_pages = -(-len(self.trace_list) // traces_per_fig)  # ceil division
                for p in range(n_pages):
                    self.current_page = p
                    self._plot_standard_traces(page=p)
                return self.trace_list

            self.current_page = page
            if plot_type == "record":
                self._plot_record_section()
            elif plot_type == "overlay":
                self._plot_overlay_traces()
            elif plot_type == "standard":
                self._plot_standard_traces(page=page)

        return self.trace_list

    def _plot_standard_traces(self, page):
        plt.close(self.fig)

        formatter_pow = ScalarFormatter(useMathText=True)
        formatter_pow.set_powerlimits((0, 0))  # Force scientific notation

        if not self.trace_list:
            print("No traces to plot.")
            return

        traces = self.trace_list.copy()
        sort_mode = self.plot_config.get("sort_by")
        if sort_mode == 'distance':
            traces.sort(key=lambda tr: self._get_geodetic_info(tr)[0])
        elif sort_mode == 'backazimuth':
            traces.sort(key=lambda tr: self._get_geodetic_info(tr)[1])

        traces_per_fig = self.plot_config["traces_per_fig"]
        n_traces = len(traces)
        i = page * traces_per_fig

        if i >= n_traces:
            print("[INFO] No more traces to display.")
            return

        sub_traces = traces[i:i + traces_per_fig]
        self.displayed_traces = sub_traces

        max_traces = min(8, 2 * len(sub_traces))
        self.fig, self.axs = plt.subplots(
            len(sub_traces), 1, figsize=(12, max_traces),
            sharex=True, sharey=self.plot_config["sharey"],
            gridspec_kw={'hspace': self.plot_config["vspace"]})

        # Ensure axs is always a list
        if len(sub_traces) == 1:
            self.axs = [self.axs]

        # Connect enter event once
        if not hasattr(self, "_entered_hook"):
            self.fig.canvas.mpl_connect("axes_enter_event", self.__on_enter_axes)
            self._entered_hook = True
        self.__selected_ax_index = 0

        # Crosshair cursor
        # Crosshair cursor
        self.crosshair = None
        mode = self._crosshair_mode()
        if mode and self.plot_config.get("show_plot", True):  # pointless in batch mode
            self.crosshair = MultiAxesCrosshair(self.axs, mode=mode)

        self._setup_pick_interaction()
        self._restore_state()

        use_fast = self.plot_config.get("fast", False)
        #fast_points = self.plot_config.get("fast_points")
        if use_fast:
            fast_points = self.plot_config.get("fast_points", 1000)


        # Plot each trace
        for ax_idx, (ax, tr) in enumerate(zip(self.axs, sub_traces)):

            if sort_mode:
                dist, baz = self._get_geodetic_info(tr)
                label = f"{tr.id} | Dist: {dist:.1f} km | Baz: {baz:.1f}°"
            else:
                label = tr.id

            # One single plotting call per trace
            lw = self._rc.get("lines.linewidth", 0.75)
            if use_fast:
                FastLine(ax, tr, n_points=fast_points, linewidth=lw, label=label)
            else:
                t = tr.times("matplotlib")
                ax.plot(t, tr.data, linewidth=lw, label=label)

            # --- Plot theoretical arrivals and origin time if present ---
            arrivals = tr.stats.get("geodetic", {}).get("arrivals", [])
            origin_time = tr.stats.get("geodetic", {}).get("otime", None)

            # Plot origin time
            if self.plot_config["show_arrivals"]:

                if origin_time:
                    ot = UTCDateTime(origin_time)
                    ax.axvline(ot.matplotlib_date, color="blue", linestyle="-", linewidth=1.0, alpha=0.7)
                    ax.text(ot.matplotlib_date, ax.get_ylim()[1] * 0.95, "Origin Time", color="blue",
                            fontsize=8, ha="center", va="bottom")

                # Plot arrivals
                for arr in arrivals:
                    phase = arr.get("phase")
                    arr_time = arr.get("time")  # float timestamp
                    if phase and arr_time:
                        arr_dt = UTCDateTime(arr_time)
                        ax.axvline(arr_dt.matplotlib_date, color="green", linestyle="--", linewidth=0.8)
                        ax.text(arr_dt.matplotlib_date, ax.get_ylim()[1] * 0.85, phase, color="green",
                                fontsize=8, rotation=0, ha="center", va="top")

            if self.plot_config["show_legend"] and len(self.axs) <= 9:
                ax.legend(loc="upper right")  # the date box is in the upper-left corner

            # Annotate with date (start, and end if the trace reaches another day)
            self._add_date_box(ax, tr.stats.starttime, tr.stats.endtime)

            # Format x-axis
            ax.xaxis_date()
            ax.xaxis.set_major_locator(mdt.AutoDateLocator())
            ax.xaxis.set_major_formatter(DayTimeFormatter(tenths=True))

            # Hide bottom ticks except for last subplot
            if ax_idx < len(self.axs) - 1:
                ax.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
                ax.spines['bottom'].set_visible(False)
            else:
                ax.tick_params(axis='x', which='both', labelbottom=True, bottom=True)
                ax.spines['bottom'].set_visible(True)

            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax.yaxis.set_major_formatter(formatter_pow)
            ax.yaxis.get_offset_text().set_visible(True)

        # Set x-limits across all axes (no need to build the full time arrays)
        auto_start = min(tr.stats.starttime.matplotlib_date for tr in sub_traces)
        auto_end = max(tr.stats.endtime.matplotlib_date for tr in sub_traces)

        for ax in self.axs:
            ax.set_xlim(auto_start, auto_end)

        plt.ion()
        plt.tight_layout()

        # --- Auto-save figure---
        out_dir = self.plot_config.get("plot_output_path")
        if out_dir:
            out_dir = self.make_abs(out_dir)

        format_save = self.plot_config.get("format_save")
        if out_dir and os.path.isdir(out_dir):
            out_file = self._unique_figure_path(out_dir, None, page, format_save, sub_traces)
            try:
                self.fig.savefig(out_file, dpi=self._rc.get("savefig.dpi", 150))
                print(f"[INFO] Figure saved to {out_file}")
            except Exception as e:
                print(f"[ERROR] Failed to save figure: {e}")

        ####

        if self.plot_config.get("auto_load_pick_file", False):
            pick_file = self.plot_config.get("pick_output_file", "./picks.csv")
            self.import_nlloc_obs(pick_file)

        ### Automatic saving plot
        if not self.plot_config.get("show_plot", True):
            plt.close(self.fig)
            return

        if self.enable_command_prompt:
            plt.show(block=False)
            self.fig.canvas.draw_idle()
            plt.pause(0.25)
            print("\n" + "=" * 55)
            print(" Command Line Interface Activated")
            print("=" * 55)
            print("[INFO] Type 'help' to see command line options or 'p' return to picking mode")

            while True:
                prompt = PlotCommandPrompt(self)
                result = prompt.run()
                if result == "p":
                    self.enable_command_prompt = False
                    self._register_span_selector()
                    plt.show(block=True)
                    break
                elif result == "exit":
                    break
        else:
            self._register_span_selector()
            plt.show(block=True)

    def _plot_record_section(self):

        plt.close(self.fig)

        cfg = self.plot_config
        traces = self.trace_list
        phase_curves = defaultdict(list)
        N_traces = len(traces)

        # record_option: "dates" (default) | "seconds" | "egf"
        record_option = cfg.get("record_option", "dates")

        # Collect arrival times grouped by phase — solo para modo "dates"
        if record_option == "dates":
            for tr in traces:
                dist = tr.stats.get("geodetic", {}).get("geodetic", [None])[0]
                arrivals = tr.stats.get("geodetic", {}).get("arrivals", [])
                seen_phases = set()
                for arr in arrivals:
                    phase = arr.get("phase")
                    arr_time = arr.get("time")
                    if (
                            phase and phase not in seen_phases
                            and dist is not None
                            and isinstance(arr_time, float)
                    ):
                        phase_curves[phase].append((mdt.date2num(UTCDateTime(arr_time).datetime), dist))
                        seen_phases.add(phase)

        # Sort by distance
        traces.sort(key=lambda tr: self._get_geodetic_info(tr)[0])
        distances = [self._get_geodetic_info(tr)[0] for tr in traces]
        scale = cfg.get("scale_factor", 1.0)

        fig, ax = plt.subplots(figsize=(12, 8))
        self.fig = fig
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

        for tr, dist in zip(traces, distances):
            # Normalize trace to its max amplitude
            norm_data = tr.data / np.max(np.abs(tr.data)) if np.max(np.abs(tr.data)) != 0 else tr.data

            if record_option == "dates":
                t = tr.times("matplotlib")

            elif record_option == "seconds":
                # t=0 en el starttime de la traza
                t = tr.times()  # segundos desde starttime, ya es lo que devuelve obspy por defecto

            elif record_option == "egf":
                # t=0 en la muestra central de la traza
                npts = tr.stats.npts
                dt = tr.stats.delta
                half = (npts // 2) * dt  # segundos desde inicio hasta la muestra central
                t = tr.times() - half  # desplaza para que el centro sea 0

            lw = self._rc.get("lines.linewidth", 0.6)
            if N_traces >= 12:
                ax.plot(t, norm_data * scale + dist, color="black", alpha=0.75, linewidth=lw)
            else:
                ax.plot(t, norm_data * scale + dist, alpha=0.75, linewidth=lw, label=tr.id)

        # Fases teóricas — solo en modo "dates"
        if record_option == "dates":
            for phase, points in phase_curves.items():
                if len(points) < 2:
                    continue
                points.sort()
                times, dists = zip(*points)
                ax.plot(times, dists, linestyle='--', linewidth=1.0, alpha=0.6, label=f"{phase}")

        # Línea vertical en t=0 para modo "egf"
        if record_option == "egf":
            ax.axvline(x=0, color="red", linestyle="--", linewidth=1.0, alpha=0.8)

        # Configuración del eje X según modo
        if record_option == "dates":
            ax.set_xlabel("Time (UTC)")
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(DayTimeFormatter(tenths=False))
            min_time = min(tr.stats.starttime for tr in traces)
            max_time = max(tr.stats.endtime for tr in traces)
            ax.set_xlim(mdt.date2num(min_time.datetime), mdt.date2num(max_time.datetime))

        elif record_option == "seconds":
            ax.set_xlabel("Time (s)")
            max_dur = max(tr.stats.npts * tr.stats.delta for tr in traces)
            ax.set_xlim(0, max_dur)

        elif record_option == "egf":
            ax.set_xlabel("Lag time (s)")
            max_half = max((tr.stats.npts // 2) * tr.stats.delta for tr in traces)
            ax.set_xlim(-max_half, max_half)

        ax.set_ylabel("Distance (km)")
        ax.set_title("Record Section")

        # Date box (not for egf: lag times have no absolute date)
        if record_option in ("dates", "seconds"):
            self._add_date_box(ax, min(tr.stats.starttime for tr in traces),
                               max(tr.stats.endtime for tr in traces))

        if cfg["show_legend"]:
            ax.legend(loc="upper right", fontsize=self._rc.get("legend.fontsize", 6))

        # --- Auto-save figure ---
        out_dir = self.plot_config.get("plot_output_path")
        if out_dir:
            out_dir = self.make_abs(out_dir)
        format_save = self.plot_config.get("format_save")
        if out_dir and os.path.isdir(out_dir):
            out_file = self._unique_figure_path(out_dir, "record", self.current_page, format_save,
                                                self.trace_list)
            try:
                self.fig.savefig(out_file, dpi=self._rc.get("savefig.dpi", 150))
                print(f"[INFO] Figure saved to {out_file}")
            except Exception as e:
                print(f"[ERROR] Failed to save figure: {e}")

        if not self.plot_config.get("show_plot", True):
            plt.close(self.fig)
            return

        if self.enable_command_prompt:
            plt.show(block=False)
            self.fig.canvas.draw_idle()
            plt.pause(0.5)
            print("\n" + "=" * 55)
            print(" Command Line Interface Activated")
            print("=" * 55)
            print("[INFO] Type 'help' to see command line options or 'p' return to picking mode")
            prompt = PlotCommandPrompt(self)
            result = prompt.run()
            if result == "p":
                self.enable_command_prompt = False
                plt.show(block=True)
        else:
            plt.show(block=True)

    def _plot_overlay_traces(self):
        traces = self.trace_list

        if len(traces) == 0:
            print("No traces to plot.")
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        n = len(traces)
        lw = self._rc.get("lines.linewidth", 0.8)
        for tr in traces:
            t = tr.times("matplotlib")
            if n > 12:
                ax.plot(t, tr.data, color="black", linewidth=lw, alpha=0.7)
            else:
                ax.plot(t, tr.data, linewidth=lw, alpha=0.7, label=tr.id)

        # X axis from the earliest start to the latest end of all traces
        x_min = min(tr.stats.starttime.matplotlib_date for tr in traces)
        x_max = max(tr.stats.endtime.matplotlib_date for tr in traces)
        ax.set_xlim(x_min, x_max)

        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Amplitude")
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(DayTimeFormatter(tenths=False))
        ax.set_title("Overlay of All Traces")
        self._add_date_box(ax, min(tr.stats.starttime for tr in traces),
                           max(tr.stats.endtime for tr in traces))

        if self.plot_config["show_legend"] or n <= 12:
            ax.legend(loc="upper right", fontsize=self._rc.get("legend.fontsize", 6))

        plt.tight_layout()

        # --- Auto-save figure ---
        out_dir = self.plot_config.get("plot_output_path")
        if out_dir:
            out_dir = self.make_abs(out_dir)
        format_save = self.plot_config.get("format_save")
        if out_dir and os.path.isdir(out_dir):
            out_file = self._unique_figure_path(out_dir, "overlay", self.current_page, format_save,
                                                self.trace_list)
            try:
                fig.savefig(out_file, dpi=150)
                print(f"[INFO] Figure saved to {out_file}")
            except Exception as e:
                print(f"[ERROR] Failed to save figure: {e}")

        if not self.plot_config.get("show_plot", True):
            plt.close(fig)
            return

        plt.show(block=True)

    def _setup_pick_interaction(self):
        def filtered_doubleclick(event):
            if event.button == MouseButton.RIGHT:
                return  # allow SpanSelector to use right-click
            if event.dblclick and event.button == MouseButton.LEFT:
                self._on_doubleclick(event)

        self.fig.canvas.mpl_connect('button_press_event', filtered_doubleclick)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.info_box = self.fig.text(0.75, 0.8, "", ha='left', va='top', fontsize=9)
        if self.plot_config["show_info_picks"]:
            self._update_info_box()
        self._restore_state()


    def _on_doubleclick(self, event):

        """Handle double-clicks to place picks."""

        # Ignore invalid clicks
        #if not event.inaxes not in self.axs:
        #    return

        # Check which subplot was clicked

        ax = event.inaxes
        try:
            tr_idx = next(i for i, a in enumerate(self.axs) if a == ax)
        except StopIteration:
            print("[WARNING] Clicked axis not found.")
            return
        trace = self.trace_list[tr_idx]
        pick_time = mdt.num2date(event.xdata).replace(tzinfo=None)

        # Ask for pick type (optional input)
        plt.pause(0.01)
        try:
            raw_input = input(
                f"Enter pick type and polarity for {trace.id} at {pick_time.strftime('%H:%M:%S')} (e.g., 'P,U' or 'Sg'): ").strip()
            if ',' in raw_input:
                phase, polarity = [s.strip() or '?' for s in raw_input.split(",", maxsplit=1)]
            else:
                phase = raw_input if raw_input else "?"
                polarity = "?"
        except Exception:
            phase, polarity = "?", "?"

        if polarity not in ['U', 'D', '?']:
            print("[WARNING] Polarity not recognized. Defaulting to '?'.")
            polarity = '?'

        # Calculate relative time and amplitude
        rel_time = (pick_time - trace.stats.starttime.datetime).total_seconds()
        amplitude = np.interp(rel_time, trace.times(), trace.data)

        # Build and store pick dictionary in trace
        pick_entry = {
            "time": UTCDateTime(pick_time).timestamp,
            "phase": phase,
            "amplitude": amplitude,
            "polarity": polarity
        }

        if not hasattr(trace.stats, "picks"):
            trace.stats.picks = []
        trace.stats.picks.append(pick_entry)

        # Save to CSV file if configured
        csv_path = self.plot_config.get("pick_output_file")
        if csv_path:
            self.write_isp_table_manual(csv_path)
            # header_needed = not os.path.exists(csv_path)
            # with open(csv_path, "a") as f:
            #     if header_needed:
            #         f.write("id,time,phase,amplitude,polarity\n")
            #     f.write(f"{trace.id},{pick_time.isoformat()},{phase},{amplitude:.4f},{polarity}\n")

        # Add to in-memory pick tracking
        if trace.id not in self.picks:
            self.picks[trace.id] = []
        self.picks[trace.id].append((pick_time, phase, amplitude, polarity))

        # Draw pick line and update display
        self.current_pick = (trace.id, pick_time)
        self._draw_pick_lines(trace.id, ax, pick_time)
        if self.plot_config["show_info_picks"]:
            self._update_info_box()
        self.fig.canvas.draw()

    def _draw_pick_lines(self, trace_id, ax, pick_time):
        """Draw vertical line and annotate pick info."""
        if trace_id not in self.pick_lines:
            self.pick_lines[trace_id] = []

        x = mdt.date2num(pick_time)
        color = self._pick_color(trace_id, pick_time)
        line = ax.axvline(x=x, color=color, linestyle='--', alpha=0.8)

        # Tag artists for hover/delete
        line._surfq_meta = {"trace_id": trace_id, "pick_time": pick_time, "color": color}

        # Find matching pick
        pick_info = None
        for pick in self.picks.get(trace_id, []):
            if pick[0] == pick_time:
                if isinstance(pick[0], (int, float)):
                    pick[0] = UTCDateTime(pick[0])
                if len(pick) == 4:
                    _, phase, _, polarity = pick
                    pick_info = f"{phase}, {polarity}"
                break

        # Default position for the label
        y_pos = ax.get_ylim()[1] * 0.25
        label = ax.text(
            x, y_pos, pick_info or "", fontsize=8,
            color='black', ha='left', va='bottom', rotation=0,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=0.6)
        )

        self.pick_lines[trace_id].append((line, label))


    def _draw_pick_all(self):
        """Redraw all pick lines."""
        for line in self.pick_lines:
            line.remove()
        self.pick_lines = []

        for tr_id, pick_list in self.picks.items():
            for pick_time, _ in pick_list:
                x = mdt.date2num(pick_time)
                for ax in self.axs:
                    line = ax.axvline(x=x, color='r', linestyle='--', alpha=0.7)
                    self.pick_lines.append(line)

    def _on_key_press(self, event):
        """Handle key presses for pick management and navigation."""
        key = event.key.lower() if event.key else ''

        # Generic exit flags
        if key in ("escape", "enter"):
            self._exit = True
            return

        # Add a global reference time (green dashed) at cursor across all traces
        if key == 'w' and event.inaxes in getattr(self, "axs", []):
            ref_time = mdt.num2date(event.xdata).replace(tzinfo=None)
            utc_ref_time = UTCDateTime(ref_time)
            self.last_reference = utc_ref_time
            for tr in self.trace_list:
                if not hasattr(tr.stats, "references"):
                    tr.stats.references = []
                tr.stats.references.append(utc_ref_time.timestamp)
            for ax in self.axs:
                ax.axvline(x=mdt.date2num(ref_time), color='g', linestyle='--', alpha=0.5)
            self.fig.canvas.draw()
            print(f"[INFO] Reference time added at {utc_ref_time.isoformat()}")
            return

        # Pagination: next page
        if key == 'n':
            if (self.current_page + 1) * self.plot_config["traces_per_fig"] < len(self.trace_list):
                self.current_page += 1
                self._plot_standard_traces(self.current_page)
            else:
                print("[INFO] Already at last page.")
            return

        # Pagination: previous page
        if key == 'b':
            if self.current_page > 0:
                self.current_page -= 1
                self._plot_standard_traces(self.current_page)
            else:
                print("[INFO] Already at first page.")
            self._redraw_picks()
            if self.plot_config["show_info_picks"]:
                self._update_info_box()
            self.current_pick = None
            return

        # Clear all picks
        if key == 'c':
            self._clear_picks()
            return

        # Remove the last reference time (from all traces)
        if key == 'p':
            removed_any = False
            for tr in self.trace_list:
                if hasattr(tr.stats, "references") and tr.stats.references:
                    tr.stats.references.pop()
                    removed_any = True
                    if not tr.stats.references:
                        del tr.stats.references
            if removed_any:
                print("[INFO] Last reference time removed from all traces.")
            else:
                print("[INFO] No reference times to remove.")
            self._redraw_all_reference_lines()
            self.fig.canvas.draw()
            return

        # Alias to remove last reference (kept for backward compatibility)
        if key == 'm':
            self._remove_last_reference()
            return

        # Toggle command prompt mode
        # if key == 'v':
        #     self.enable_command_prompt = True
        #     self.plot(page=self.current_page)
        #     return

        if key == 'v':
            self.enable_command_prompt = False  # ← empieza en False
            plt.show(block=False)
            self.fig.canvas.draw_idle()
            plt.pause(0.1)
            print("\n" + "=" * 55)
            print(" Command Line Interface Activated")
            print("=" * 55)
            print("[INFO] Type 'help' to see command line options or 'p' return to picking mode")
            prompt = PlotCommandPrompt(self)
            result = prompt.run()
            if result == "p":
                self._register_span_selector()
                plt.show(block=True)
            elif result == "replot":
                # if we want bak to picking style set to False
                print("Backing to picking mode, press 'v' to reopen a prompt")
                self.enable_command_prompt = False
                self.plot(page=self.current_page)
            elif result == "exit":
                plt.close(self.fig)
            return

        # --- Set start time with key 'u' ---
        if key == 'u':
            if event.inaxes not in getattr(self, "axs", []) or event.xdata is None:
                print("[WARNING] Move the mouse over a trace before pressing 'i'.")
                return

            start_time = UTCDateTime(mdt.num2date(event.xdata))
            self.utc_start = start_time

            print(f"[INFO] Start time set to: {start_time.isoformat()}")

            self._draw_time_window()
            return

        # --- Set end time with key 'i' ---
        if key == 'i':
            if event.inaxes not in getattr(self, "axs", []) or event.xdata is None:
                print("[WARNING] Move the mouse over a trace before pressing 'o'.")
                return

            end_time = UTCDateTime(mdt.num2date(event.xdata))
            self.utc_end = end_time

            print(f"[INFO] End time set to: {end_time.isoformat()}")

            self._draw_time_window()
            return

        # Interactive pick with prompt at cursor time
        if key == 'e' and self.plot_config["backend"] != "Qt5Agg":
            if event.inaxes not in getattr(self, "axs", []) or event.xdata is None:
                print("[WARNING] Move the mouse over a trace before pressing 'e'.")
                return

            ax = event.inaxes
            try:
                tr_idx = next(i for i, a in enumerate(self.axs) if a == ax)
            except StopIteration:
                print("[WARNING] Clicked axis not found.")
                return

            # Use the trace actually shown in this subplot page/sort order
            try:
                trace = self.displayed_traces[tr_idx]
            except Exception:
                print("[WARNING] No displayed trace found for this subplot.")
                return

            pick_time = mdt.num2date(event.xdata).replace(tzinfo=None)

            # Ask for pick type and polarity
            plt.pause(0.01)
            try:
                raw_input_str = input(
                    f"Enter pick type and polarity for {trace.id} at "
                    f"{pick_time.strftime('%H:%M:%S')} (e.g., 'P,U' or 'Sg'): "
                ).strip()
                if ',' in raw_input_str:
                    phase, polarity = [s.strip() or '?' for s in raw_input_str.split(",", maxsplit=1)]
                else:
                    phase = raw_input_str if raw_input_str else "?"
                    polarity = "?"
            except Exception:
                phase, polarity = "?", "?"

            if polarity not in ['U', 'D', '?']:
                print("[WARNING] Polarity not recognized. Defaulting to '?'.")
                polarity = '?'

            # Calculate relative time and amplitude at the cursor
            rel_time = (pick_time - trace.stats.starttime.datetime).total_seconds()
            amplitude = np.interp(rel_time, trace.times(), trace.data)

            # Persist into trace header
            pick_entry = {
                "time": UTCDateTime(pick_time).timestamp,
                "phase": phase,
                "amplitude": amplitude,
                "polarity": polarity
            }
            if not hasattr(trace.stats, "picks"):
                trace.stats.picks = []
            trace.stats.picks.append(pick_entry)

            # Mirror into in-memory structure
            if trace.id not in self.picks:
                self.picks[trace.id] = []
            self.picks[trace.id].append((pick_time, phase, amplitude, polarity))

            # Save to ISP/CSV (overwrite with current picks)
            csv_path = self.plot_config.get("pick_output_file")
            if csv_path:
                self.write_isp_table_manual(csv_path)

            # Draw pick line + update side box
            self.current_pick = (trace.id, pick_time)
            self._draw_pick_lines(trace.id, ax, pick_time)
            if self.plot_config["show_info_picks"]:
                self._update_info_box()
            self.fig.canvas.draw()
            print(f"[INFO] Pick added: {trace.id} at {pick_time.strftime('%H:%M:%S')}, phase ({phase},{polarity}), "
                  f" amplitude {amplitude}")
            return

        # Delete a pick under the cursor (hover highlight)
        if key in ('d', 'delete', 'backspace'):
            if self._hover:
                trace_id, line, label, pick_time = self._hover
                if trace_id is not None and pick_time is not None:
                    self._delete_pick(trace_id, pick_time)
                    print(f"[INFO] Deleted pick on {trace_id} at {pick_time.strftime('%H:%M:%S.%f')[:-3]}")
                else:
                    print("[INFO] No pick under cursor to delete.")
            else:
                print("[INFO] Hover a pick (red dashed line) and press D/Delete/Backspace to remove it.")
            return

        # Quick-pick hotkeys at cursor (1–6)
        if key in ['1', '2', '3', '4', '5', '6']:
            if event.inaxes not in getattr(self, "axs", []) or event.xdata is None:
                print("[WARNING] Move your mouse over a trace before pressing 1–6 to pick.")
                return

            key_map = {
                '1': ("P", "?"),
                '2': ("P", "U"),
                '3': ("P", "D"),
                '4': ("S", "?"),
                '5': ("S", "U"),
                '6': ("S", "D"),
            }
            phase, polarity = key_map[key]

            ax = event.inaxes
            try:
                tr_idx = next(i for i, a in enumerate(self.axs) if a == ax)
            except StopIteration:
                print("[WARNING] Could not find axis index.")
                return

            try:
                trace = self.displayed_traces[tr_idx]
            except Exception:
                print("[WARNING] No displayed trace found for this subplot.")
                return

            pick_time = mdt.num2date(event.xdata).replace(tzinfo=None)
            rel_time = (pick_time - trace.stats.starttime.datetime).total_seconds()
            amplitude = np.interp(rel_time, trace.times(), trace.data)

            pick_entry = {
                "time": UTCDateTime(pick_time).timestamp,
                "phase": phase,
                "amplitude": amplitude,
                "polarity": polarity
            }

            if not hasattr(trace.stats, "picks"):
                trace.stats.picks = []
            trace.stats.picks.append(pick_entry)

            if trace.id not in self.picks:
                self.picks[trace.id] = []
            self.picks[trace.id].append((pick_time, phase, amplitude, polarity))

            csv_path = self.plot_config.get("pick_output_file")
            if csv_path:
                self.write_isp_table_manual(csv_path)

            self.current_pick = (trace.id, pick_time)
            self._draw_pick_lines(trace.id, ax, pick_time)
            if self.plot_config["show_info_picks"]:
                self._update_info_box()
            self.fig.canvas.draw()
            print(f"[INFO] Pick added: {trace.id} at {pick_time.strftime('%H:%M:%S')}, phase ({phase},{polarity}), "
                  f" amplitude {amplitude}")
            return

    def _draw_time_window(self):
        """Draw start/end time window on all axes."""

        if self.axs is None:
            return

        # Remove previous blue lines + labels
        for ax in self.axs:
            lines_to_remove = [
                line for line in ax.lines
                if line.get_linestyle() == '--' and line.get_color() == 'blue'
            ]
            for line in lines_to_remove:
                line.remove()

            texts_to_remove = [
                txt for txt in ax.texts
                if txt.get_text() in ('Start', 'End')
            ]
            for txt in texts_to_remove:
                txt.remove()

        # Draw new ones if defined
        for ax in self.axs:
            if self.utc_start:
                xmin = mdt.date2num(self.utc_start.datetime)
                ax.axvline(x=xmin, color='blue', linestyle='--', alpha=0.5)

            if self.utc_end:
                xmax = mdt.date2num(self.utc_end.datetime)
                ax.axvline(x=xmax, color='blue', linestyle='--', alpha=0.5)

        # Labels only on first axis
        if self.axs is not None and len(self.axs) > 0:
            ax0 = self.axs[0]
            ylim = ax0.get_ylim()

            if self.utc_start:
                xmin = mdt.date2num(self.utc_start.datetime)
                ax0.text(xmin, ylim[1] * 0.95, 'Start',
                         color='blue', fontsize=8,
                         ha='left', va='top',
                         bbox=dict(fc='white', alpha=0.6))

            if self.utc_end:
                xmax = mdt.date2num(self.utc_end.datetime)
                ax0.text(xmax, ylim[1] * 0.95, 'End',
                         color='blue', fontsize=8,
                         ha='right', va='top',
                         bbox=dict(fc='white', alpha=0.6))

    def _redraw_picks(self):
        """Redraw all pick lines only on their corresponding axes."""

        # Clear previous lines
        for lines in self.pick_lines.values():
            for line, label in lines:
                line.remove()
                label.remove()
        self.pick_lines = {}

        # Sanity check
        if not hasattr(self, "displayed_traces") or self.axs is None:
            return

        # Build map from trace ID → axis
        trace_id_to_ax = {
            tr.id: ax for tr, ax in zip(self.displayed_traces, self.axs)
        }

        # Redraw picks for visible traces
        for trace_id, pick_list in self.picks.items():
            ax = trace_id_to_ax.get(trace_id)
            if ax is None:
                continue  # Not shown in current figure
            for pick in pick_list:
                pt = pick[0]
                self._draw_pick_lines(trace_id, ax, pt)

        self._draw_events()
        self.fig.canvas.draw()

    def _clear_picks(self, event=None):
        # Clear internal pick records
        self.picks.clear()

        # Clear from trace.stats if exists
        for trace in self.trace_list:
            if hasattr(trace.stats, "picks"):
                del trace.stats.picks

        # Remove pick lines from plot
        for lines in self.pick_lines.values():
            for line, label in lines:
                line.remove()
                label.remove()
        self.pick_lines = {}

        # Imported events go with the picks
        self.events = []
        self._pick_event = {}
        self._remove_event_artists()

        self.current_pick = None
        if self.plot_config["show_info_picks"]:
            self._update_info_box()
        self.fig.canvas.draw()

    def get_picks(self):
        """Return picks as {trace_id: [UTCDateTime]}."""
        return {k: [UTCDateTime(p) for p in v]
                for k, v in self.picks.items() if v}

    def _update_info_box(self):

        """Update the side info box with picks."""

        lines = []
        for tr_id, pick_list in self.picks.items():
            for pick in pick_list:
                if len(pick) == 4:
                    pt, ptype, amp, pol = pick
                else:
                    pt, ptype, amp = pick
                    pol = "?"
                time_str = pt.strftime('%H:%M:%S.%f')[:-3]
                lines.append(f"{tr_id}: {time_str} ({ptype}, {pol}) amp={amp:.2f}")

        self.info_box.set_text('\n'.join(lines))

    def set_pick_type(self, pick_type: str):
        """Set current pick type (e.g., 'P', 'S', etc.)"""
        self.pick_type = pick_type.upper()
        print(f"[INFO] Pick type set to: {self.pick_type}")

    def _redraw_all_reference_lines(self):
        """Remove old reference lines and redraw all from trace stats."""
        # Remove previous green reference lines
        for ax in self.axs:
            lines_to_remove = [line for line in ax.lines if line.get_color() == 'g' and line.get_linestyle() == '--']
            for line in lines_to_remove:
                line.remove()

        # Redraw reference lines from stats
        for tr in self.trace_list:
            if hasattr(tr.stats, "references"):
                for ref_time in tr.stats.references:
                    if isinstance(ref_time, (float, int)):
                        x = mdt.date2num(UTCDateTime(ref_time).datetime)
                    else:
                        x = mdt.date2num(ref_time.datetime)
                    for ax in self.axs:
                        ax.axvline(x=x, color='g', linestyle='--', alpha=0.5)

        self.fig.canvas.draw()

    def _remove_last_reference(self):
        """Remove the last reference time from all traces and update the plot."""
        removed_any = False

        for tr in self.trace_list:
            if hasattr(tr.stats, "references") and tr.stats.references:
                tr.stats.references.pop()
                removed_any = True

        if removed_any:
            print("[INFO] Last reference time removed from all traces.")
            self._redraw_all_reference_lines()
        else:
            print("[INFO] No reference times found to remove.")

    def _window_copy(self, trace):
        """Copy of the trace trimmed to the plot window (utc_start / utc_end). Never modifies the original."""
        tr = trace.copy()
        if self.utc_start is not None or self.utc_end is not None:
            tr.trim(starttime=self.utc_start, endtime=self.utc_end)
        return tr

    def _plot_single_spectrum(self, idx, axis_type, mode="fft", order=20, nsig=None, scaling="fourier"):

        self._exit = False
        trace = self._window_copy(self.displayed_traces[idx])
        spectrum, freqs = SpectrumTool.compute_spectrum(trace.data, trace.stats.delta, mode=mode,
                                                        order=order, nsig=nsig, scaling=scaling)

        fig, ax = plt.subplots()

        if axis_type == "loglog":
            ax.loglog(freqs[1:], spectrum[1:], label=trace.id, linewidth=0.75)
        elif axis_type == "xlog":
            ax.semilogx(freqs[1:], spectrum[1:], label=trace.id, linewidth=0.75)
        elif axis_type == "ylog":
            ax.semilogy(freqs[1:], spectrum[1:], label=trace.id, linewidth=0.75)
        elif axis_type == "linlin":
            ax.plot(freqs[1:], spectrum[1:], label=trace.id, linewidth=0.75)

        if axis_type != "linlin":
            ax.set_ylim(spectrum[1:].min() / 10.0, spectrum[1:].max() * 100.0)

        ax.set_xlim(freqs[1], freqs[-1])

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(SpectrumTool.spectrum_ylabel(scaling, mode))
        ax.set_title(f"Spectrum for {trace.id}  [{self._spectrum_label(mode, order, nsig)}]")
        ax.legend()
        plt.grid(True, which="both", ls="-", color='grey', alpha=0.4)
        plt.tight_layout()
        cid = fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        def on_close(event):
            pass  # nothing needed, just let it close

        fig.canvas.mpl_connect("close_event", on_close)
        fig.canvas.mpl_disconnect(cid)
        plt.show(block=False)  # non-blocking, return immediately to run()

    def _plot_all_spectra(self, axis_type, mode="fft", order=20, nsig=None, scaling="fourier"):

        self._exit = False
        self.fig_spec, self.ax_spec = plt.subplots()

        traces = [self._window_copy(tr) for tr in self.displayed_traces]

        ymin, ymax = np.inf, -np.inf
        for trace in traces:
            try:
                spectrum, freqs = SpectrumTool.compute_spectrum(trace.data, trace.stats.delta, mode=mode,
                                                                order=order, nsig=nsig, scaling=scaling)
            except Exception as e:
                print(f"[WARN] {trace.id}: spectrum failed ({e}), skipped.")
                continue

            if axis_type == "loglog":
                self.ax_spec.loglog(freqs[1:], spectrum[1:], label=trace.id, linewidth=0.75)
            elif axis_type == "xlog":
                self.ax_spec.semilogx(freqs[1:], spectrum[1:], label=trace.id, linewidth=0.75)
            elif axis_type == "ylog":
                self.ax_spec.semilogy(freqs[1:], spectrum[1:], label=trace.id, linewidth=0.75)
            elif axis_type == "linlin":
                self.ax_spec.plot(freqs[1:], spectrum[1:], label=trace.id, linewidth=0.75)

            ymin = min(ymin, spectrum[1:].min())
            ymax = max(ymax, spectrum[1:].max())

        if axis_type != "linlin" and np.isfinite(ymin) and ymin > 0:
            self.ax_spec.set_ylim(ymin / 10.0, ymax * 100.0)

        self.ax_spec.set_xlabel("Frequency (Hz)")
        self.ax_spec.set_ylabel(SpectrumTool.spectrum_ylabel(scaling, mode))
        self.ax_spec.set_title(f"Spectra [{self._spectrum_label(mode, order, nsig)}]")
        self.ax_spec.legend(fontsize=6)
        plt.grid(True, which="both", ls="-", color='grey')
        plt.tight_layout()
        cid = self.fig_spec.canvas.mpl_connect("key_press_event", self._on_key_press)

        def on_close(event):
            pass  # nothing needed, just let it close

        self.fig_spec.canvas.mpl_connect("close_event", on_close)
        self.fig_spec.canvas.mpl_disconnect(cid)
        plt.show(block=False)  # non-blocking, return immediately to run()

    @staticmethod
    def _spectrum_label(mode, order, nsig):
        """Text for the title: method plus the parameters that apply to it."""
        if mode in PSEUDO_MODES:
            return f"{mode}, order {order}, nsig {nsig}"
        if mode in ("burg", "yule", "covar", "modcovar", "minvar"):
            return f"{mode}, order {order}"
        return mode

    def _plot_spectrogram(self, idx, win_sec=5.0, overlap_percent=50.0, clip=None,
                          method="multitaper", nw=None, dates=False):

        self._exit = False
        trace = self._window_copy(self.displayed_traces[idx])

        try:
            stime = self.utc_start
        except:
            stime = trace.stats.starttime

        try:
            etime = self.utc_end
        except:
            etime = trace.stats.endtime

        print("Spectrogram window: ", stime, etime)

        print(idx, int(win_sec * trace.stats.sampling_rate), (100 - overlap_percent) * 1E-2,
            method, nw)


        spectrum, num_steps, t, f = SpectrumTool.compute_spectrogram(trace.data, win=int(win_sec * trace.stats.sampling_rate),
            dt=trace.stats.delta, linf=0, lsup=int(trace.stats.sampling_rate // 2),
            step_percentage=(100 - overlap_percent) * 1E-2, method=method, nw=nw)

        # --- Set up GridSpec with reserved space for colorbar ---
        self.fig_spec = plt.figure(figsize=(10, 5))

        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.03], height_ratios=[1, 1],
            hspace=0.02, wspace=0.02)

        ax_waveform = self.fig_spec.add_subplot(gs[0, 0])
        ax_spec = self.fig_spec.add_subplot(gs[1, 0], sharex=ax_waveform)
        ax_cbar = self.fig_spec.add_subplot(gs[1, 1])

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        ax_waveform.yaxis.set_major_formatter(formatter)

        # ------------------------------------------------------
        # X coordinates
        # ------------------------------------------------------
        if dates:

            import matplotlib.dates as mdates

            start_num = mdates.date2num(trace.stats.starttime.datetime)
            waveform_x = (start_num + trace.times() / 86400.0)
            spec_x = (start_num + np.asarray(t) / 86400.0)
            locator = mdates.AutoDateLocator()
            ax_spec.xaxis.set_major_locator(locator)
            ax_spec.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

            ax_spec.set_xlabel("Date / Time [UTC]")

        else:

            waveform_x = trace.times()
            spec_x = t
            ax_spec.set_xlabel("Time [s]")

        ax_waveform.set_xlim(spec_x[0], spec_x[-1])
        # --- Plot waveform ---
        ax_waveform.plot(waveform_x, trace.data, linewidth=0.75)
        ax_waveform.set_title(f"Spectrogram for {trace.id}")
        ax_waveform.tick_params(labelbottom=False)

        # ------------------------------------------------------
        # Convert spectrum to relative dB
        # ------------------------------------------------------
        spectrum = 10 * np.log10(spectrum / np.max(spectrum))

        # Compute water level
        if clip is not None:
            spectrum = np.clip(spectrum,a_min=clip, a_max=0)

        # --- Plot spectrogram ---
        pcm = ax_spec.pcolormesh(spec_x, f, spectrum,
            shading="auto", cmap="rainbow", vmin=np.min(spectrum), vmax=0)

        ax_waveform.set_ylabel("Amplitude")
        ax_spec.set_ylabel("Frequency [Hz]")

        # --- Add colorbar without shifting axes ---
        cbar = self.fig_spec.colorbar(pcm, cax=ax_cbar, orientation="vertical")
        cbar.set_label("Power [dB]")
        cid = self.fig_spec.canvas.mpl_connect("key_press_event", self._on_key_press)

        def on_close(event):
            pass

        self.fig_spec.canvas.mpl_connect("close_event", on_close)
        self.fig_spec.canvas.mpl_disconnect(cid)

        plt.show(block=False)

    def _plot_wavelet(self, idx, wavelet_type, param, **kwargs):
        self._exit = False

        if wavelet_type == "cm":
            wavelet_type = "Complex Morlet"
        elif wavelet_type == "mh":
            wavelet_type = "Mexican Hat"
        elif wavelet_type == "pa":
            wavelet_type = "Paul"

        param = float(param)
        tr = self._window_copy(self.displayed_traces[idx])

        f_min = kwargs.pop("fmin", 0.5)
        f_max = kwargs.pop("fmax", tr.stats.sampling_rate // 2)
        clip = kwargs.pop("clip", None)
        dates = kwargs.pop("dates", False)

        try:
            stime = self.utc_start
        except:
            stime = tr.stats.starttime

        try:
            etime = self.utc_end
        except:
            etime = tr.stats.endtime

        print("CWT window: ", stime, etime)

        tr.trim(starttime=stime, endtime=etime)

        cw = ConvolveWaveletScipy(tr, m=int(param))

        tt = int(tr.stats.sampling_rate / f_min)

        cw.setup_wavelet(wmin=param, wmax=param, tt=tt, fmin=f_min, fmax=f_max,
            voices_per_octave=16, use_wavelet=wavelet_type, m=param,
                         decimate=False)

        scalogram2 = cw.scalogram_in_dbs()

        # Compute water level
        if clip is not None:
            scalogram2 = np.clip(scalogram2, a_min=clip, a_max=0)

        t = np.linspace(0, tr.stats.delta * scalogram2.shape[1], scalogram2.shape[1])

        f = np.logspace(np.log10(f_min), np.log10(f_max), scalogram2.shape[0])

        c_f = param / (2 * math.pi)

        ff = f      # same log-spaced axis as the scalogram

        pred = ((math.sqrt(2) * c_f / ff) - (math.sqrt(2) * c_f / f_max))

        pred_comp = t[len(t) - 1] - pred

        # ------------------------------------------------------
        # X coordinates
        #
        # Matplotlib dates are expressed in days, therefore
        # seconds are converted using / 86400.
        # ------------------------------------------------------
        if dates:

            import matplotlib.dates as mdates

            start_num = mdates.date2num(tr.stats.starttime.datetime)
            waveform_x = (start_num + tr.times() / 86400.0)
            spec_x = (start_num + t / 86400.0)

            # pred and pred_comp are also time coordinates
            pred_x = (start_num + pred / 86400.0)
            pred_comp_x = (start_num + pred_comp / 86400.0)

        else:

            waveform_x = tr.times()
            spec_x = t

            pred_x = pred
            pred_comp_x = pred_comp

        # Mesh for the scalogram
        x, y = np.meshgrid(spec_x, f)

        # --- Set up GridSpec with reserved space for colorbar ---
        self.fig_spec = plt.figure(figsize=(10, 5))

        gs = gridspec.GridSpec(2,2, width_ratios=[1, 0.03], height_ratios=[1, 1], hspace=0.02,
            wspace=0.02)

        ax_waveform = self.fig_spec.add_subplot(gs[0, 0])
        ax_spec = self.fig_spec.add_subplot(gs[1, 0], sharex=ax_waveform)
        ax_cbar = self.fig_spec.add_subplot(gs[1, 1])
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        ax_waveform.yaxis.set_major_formatter(formatter)

        # --- Plot waveform ---
        ax_waveform.plot(waveform_x, tr.data, linewidth=0.75)

        ax_waveform.set_title(f"CWT Scalogram for {tr.id}")

        ax_waveform.tick_params(labelbottom=False)

        # --- Plot scalogram ---
        pcm = ax_spec.pcolormesh(x, y, scalogram2, shading="auto", cmap="rainbow",
                                 vmin=np.min(scalogram2), vmax=0)

        ax_spec.fill_between(pred_x, ff, f_min, color="black", edgecolor="red", alpha=0.3)
        ax_spec.fill_between(pred_comp_x, ff, f_min, color="black", edgecolor="red", alpha=0.3)

        ax_waveform.set_ylabel("Amplitude")
        ax_spec.set_ylim([np.min(f), np.max(f)])
        ax_spec.set_yscale("log")
        ax_spec.set_ylabel("Frequency [Hz]")

        # ------------------------------------------------------
        # X axis
        # ------------------------------------------------------
        if dates:

            locator = mdates.AutoDateLocator()
            ax_spec.xaxis.set_major_locator(locator)
            ax_spec.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax_spec.set_xlabel("Date / Time [UTC]")

        else:

            ax_spec.set_xlabel("Time [s]")

        # --- Add colorbar without shifting axes ---
        cbar = self.fig_spec.colorbar(pcm, cax=ax_cbar, orientation="vertical")

        cbar.set_label("Power [dB]")

        cid = self.fig_spec.canvas.mpl_connect("key_press_event", self._on_key_press)

        def on_close(event):
            pass  # nothing needed, just let it close

        self.fig_spec.canvas.mpl_connect("close_event", on_close)
        self.fig_spec.canvas.mpl_disconnect(cid)

        plt.show(block=False)


    def _plot_stack(self, tr:Trace):
        self._exit = False
        def _hms_tenths(x, pos=None):
            d = mdt.num2date(x) + timedelta(milliseconds=50)  # round to nearest 0.1 s
            return f"{d:%H:%M:%S}.{d.microsecond // 100000}"

        self.fig_stack, ax = plt.subplots(figsize=(8, 5))
        starttime = tr.stats.starttime
        t = tr.times("matplotlib")
        ax.plot(t, tr.data, linewidth=1.0, label=f"{tr.id}")
        # annotation
        date_str = starttime.strftime("%Y-%m-%d")
        textstr = f"JD {starttime.julday} / {starttime.year}\n{date_str}"
        ax.text(0.01, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.5))

        # x-axis formatting
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdt.AutoDateLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(_hms_tenths))
        self.fig_stack.tight_layout()

        cid = self.fig_stack.canvas.mpl_connect("key_press_event", self._on_key_press)
        # Poll until the figure is closed
        def on_close(event):
            pass  # nothing needed, just let it close
        self.fig_stack.canvas.mpl_connect("close_event", on_close)
        self.fig_stack.canvas.mpl_disconnect(cid)
        plt.show(block=False)  # non-blocking, return immediately to run()

    def _slowness_map(self, **kwargs):

        self._exit = False
        try:
            plt.close(self.fig_slow_map)
        except:
            pass

        self.fmin = kwargs.pop("fmin", 0.8)
        self.fmax = kwargs.pop("fmax", 2.2)
        self.smax = kwargs.pop("smax", 0.3)
        self.slow_grid = kwargs.pop("slow_grid", 0.05)
        self.method_beam = kwargs.pop("method", "FK")
        self.nsignals = kwargs.pop("nsignals", 1)

        traces = Stream(self.trace_list)

        try:
            stime = self.utc_start
        except:
            stime = traces.stats.starttime

        try:
            etime = self.utc_end
        except:
            etime = traces.stats.endtime

        print("Slowness Map time window: ", stime, etime)

        traces_fk = traces.copy()
        traces_fk.trim(starttime=stime, endtime=etime)

        try:
            timewindow = etime - stime

            selection = MseedUtil.filter_inventory_by_stream(traces_fk, self.inventory)
            wavenumber = array_analysis.array()

            if self.method_beam in ["FK", "CAPON", "MTP.COHERENCE"]:
                Z, Sxpow, Sypow, coord = wavenumber.FKCoherence(
                    traces, selection, stime, self.fmin, self.fmax, self.smax, timewindow,
                    self.slow_grid, self.method_beam)
            elif self.method_beam == "MUSIC":
                Z, Sxpow, Sypow, coord = wavenumber.run_music(
                    traces, selection, stime, self.fmin, self.fmax,
                    self.smax, timewindow, self.slow_grid, self.nsignals)

            backacimuth = wavenumber.azimuth2mathangle(np.arctan2(Sypow, Sxpow) * 180 / np.pi)
            slowness = np.hypot(Sxpow, Sypow)

            Sx = np.arange(-1 * self.smax, self.smax, self.slow_grid)[np.newaxis]
            nx = len(Sx[0])
            x = y = np.linspace(-1 * self.smax, self.smax, nx)
            X, Y = np.meshgrid(x, y)

            info_text = (
                f"Slowness: {slowness[0]:.2f} s/km\n"
                f"Azimuth: {backacimuth[0]:.2f}°\n"
                f"Power: {np.max(Z):.2f}")
            print(info_text)

            if self.method_beam == "FK":
                clabel = "FK Normalized Power"
            elif self.method_beam == "CAPON":
                clabel = "CAPON Power"
            elif self.method_beam == "MTP.COHERENCE":
                clabel = "Multitaper Magnitude Coherence"
            elif self.method_beam == "MUSIC":
                clabel = "MUSIC Pseudospectrum"

            self.fig_slow_map, ax_slow = plt.subplots(figsize=(8, 5))
            contour = ax_slow.contourf(X, Y, Z, cmap="rainbow", levels=50)
            ax_slow.set_xlabel("Sx (s/km)")
            ax_slow.set_ylabel("Sy (s/km)")
            ax_slow.text(
                0.02, 0.98, info_text,
                transform=ax_slow.transAxes,
                ha="left", va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8)
            )

            cbar = plt.colorbar(contour, ax=ax_slow)
            cbar.set_label(clabel)
            plt.tight_layout()
            cid = self.fig_slow_map.canvas.mpl_connect("key_press_event", self._on_key_press)
            plt.show(block=False)

            while plt.fignum_exists(self.fig_slow_map.number) and not self._exit:
                plt.pause(0.2)
                time.sleep(0.1)
            self.fig_slow_map.canvas.mpl_disconnect(cid)
        except Exception as e:
            print(f"[ERROR] Could not compute or plot FK coherence: {e}")

    def _run_fk(self, **kwargs):

        self._exit = False
        try:
            plt.close(self.fig_fk)
            plt.close(self.fig_slow_map)
        except:
            pass

        self.timewindow = kwargs.pop("timewindow", 3)
        self.overlap = kwargs.pop("overlap", 0.05)
        self.fmin = kwargs.pop("fmin", 0.8)
        self.fmax = kwargs.pop("fmax", 2.2)
        self.smax = kwargs.pop("smax", 0.3)
        self.slow_grid = kwargs.pop("slow_grid", 0.05)
        self.method_beam = kwargs.pop("method", "FK")

        traces = Stream(self.trace_list)
        common_start = max(tr.stats.starttime for tr in traces)
        common_end = min(tr.stats.endtime for tr in traces)
        traces.trim(starttime=common_start, endtime=common_end, pad=True, fill_value=0)

        if self.utc_start is not None:

            stime = self.utc_start
        else:
            stime = traces[0].stats.starttime

        if self.utc_end is not None:

            etime = self.utc_end
        else:
            etime = traces[0].stats.endtime

        try:

            print("FK window: ", stime, etime)

            traces_fk = traces.copy()
            traces_fk.trim(starttime=stime, endtime=etime)
            selection = MseedUtil.filter_inventory_by_stream(traces, self.inventory)
            wavenumber = array_analysis.array()
            self.relpower, self.abspower, self.AZ, self.Slowness, self.T = wavenumber.FK(traces_fk, selection, stime, etime,
                           self.fmin, self.fmax, self.smax, self.slow_grid, self.timewindow, self.overlap)

            # --- Create grid layout with reserved space for colorbar ---
            self.fig_fk = plt.figure(figsize=(9, 6))
            #self.fig_fk.canvas.mpl_connect("button_press_event", self._on_fk_doubleclick)
            self.fig_fk.canvas.mpl_connect('key_press_event', self._on_fk_key_press)
            gs = gridspec.GridSpec(3, 2, width_ratios=[35, 1], height_ratios=[1, 1, 1], hspace=0.0,
                                   wspace=0.02)

            # Create subplots
            ax0 = self.fig_fk.add_subplot(gs[0, 0])
            ax1 = self.fig_fk.add_subplot(gs[1, 0], sharex=ax0)
            ax2 = self.fig_fk.add_subplot(gs[2, 0], sharex=ax0)
            ax_cbar = self.fig_fk.add_subplot(gs[:, 1])  # colorbar takes all rows

            # Scatter plots
            sc0 = ax0.scatter(self.T, self.relpower, c=self.relpower, cmap='rainbow', s=20)
            sc1 = ax1.scatter(self.T, self.Slowness, c=self.relpower, cmap='rainbow', s=20)
            sc2 = ax2.scatter(self.T, self.AZ, c=self.relpower, cmap='rainbow', s=20)

            # Labels and formatting
            ax0.set_ylabel("Rel. Power")
            ax1.set_ylabel("Slowness\n(s/km)")
            ax2.set_ylabel("BackAzimuth")
            ax2.set_xlabel("Time (UTC)")
            ax0.set_title("FK Analysis - Relative Power")

            formatter = mdt.DateFormatter('%H:%M:%S')
            ax2.xaxis.set_major_formatter(formatter)
            ax2.tick_params(axis='x', rotation=0)

            # Add colorbar
            cbar = self.fig_fk.colorbar(sc2, cax=ax_cbar)
            cbar.set_label("Relative Power")

            plt.tight_layout()
            cid = self.fig_fk.canvas.mpl_connect("key_press_event", self._on_key_press)

            if self.plot_config["show_help"]:
                self._print_fk_help()

            # Poll until the figure is closed
            def on_close(event):
                pass  # nothing needed, just let it close

            self.fig_fk.canvas.mpl_connect("close_event", on_close)
            self.fig_fk.canvas.mpl_disconnect(cid)
            plt.show(block=False)  # non-blocking, return immediately to run()
        except:
            print("Cannot compute fk analysis, please review the inventory and parameter values")


    def _on_fk_key_press(self, event):

        method_map = {
            '1': "FK",
            '2': "CAPON",
            '3': "MTP.COHERENCE",
            '4': "MUSIC",
            '5': "MUSIC_2_signals",
            '6': "MUSIC_3_signals"
        }

        """Handle key presses for pick management."""
        key = event.key.lower() if event.key else ''

        if key in method_map:
            self.method_beam = method_map[event.key]
            print(f"[INFO] Method selected: {self.method_beam}")
            return

        if key == 'e':
            self._exit = False
            xdata = event.xdata
            if xdata is None:
                return

            try:
                time_clicked = mdt.num2date(xdata).replace(tzinfo=None)
                print(f"[INFO] Double-clicked time: {time_clicked.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
            except Exception as e:
                print(f"[ERROR] Failed to convert time: {e}")
                return

            try:
                traces = Stream(self.trace_list)
                traces_slow = traces.copy()
                selection = MseedUtil.filter_inventory_by_stream(traces_slow, self.inventory)
                wavenumber = array_analysis.array()

                if self.method_beam in ["FK", "CAPON", "MTP.COHERENCE"]:
                    Z, Sxpow, Sypow, coord = wavenumber.FKCoherence(
                        traces, selection, xdata, self.fmin, self.fmax, self.smax, self.timewindow,
                        self.slow_grid, self.method_beam)

                elif self.method_beam == "MUSIC":
                    n_signals = 1
                    Z, Sxpow, Sypow, coord = wavenumber.run_music(
                        traces, selection, xdata, self.fmin, self.fmax,
                        self.smax, self.timewindow, self.slow_grid, n_signals)

                elif self.method_beam == "MUSIC_2_signals":
                    n_signals = 2
                    Z, Sxpow, Sypow, coord = wavenumber.run_music(
                        traces, selection, xdata, self.fmin, self.fmax,
                        self.smax, self.timewindow, self.slow_grid, n_signals)

                elif self.method_beam == "MUSIC_3_signals":
                    n_signals = 3
                    Z, Sxpow, Sypow, coord = wavenumber.run_music(
                        traces, selection, xdata, self.fmin, self.fmax,
                        self.smax, self.timewindow, self.slow_grid, n_signals)

                backacimuth = wavenumber.azimuth2mathangle(np.arctan2(Sypow, Sxpow) * 180 / np.pi)
                slowness = np.hypot(Sxpow, Sypow)

                Sx = np.arange(-1 * self.smax, self.smax, self.slow_grid)[np.newaxis]
                nx = len(Sx[0])
                x = y = np.linspace(-1 * self.smax, self.smax, nx)
                X, Y = np.meshgrid(x, y)

                info_text = (
                    f"Time: {time_clicked.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Slowness: {slowness[0]:.2f} s/km\n"
                    f"Azimuth: {backacimuth[0]:.2f}°\n"
                    f"Power: {np.max(Z):.2f}")
                print(info_text)

                if self.method_beam == "FK":
                    clabel = "FK Normalized Power"
                elif self.method_beam == "CAPON":
                    clabel = "CAPON Power"
                elif self.method_beam == "MTP.COHERENCE":
                    clabel = "Multitaper Magnitude Coherence"
                elif (self.method_beam == "MUSIC" or self.method_beam == "MUSIC_2_signals" or
                          self.method_beam == "MUSIC_3_signals"):
                    clabel = "MUSIC Pseudospectrum"

                self.fig_slow_map, ax_slow = plt.subplots(figsize=(8, 5))
                contour = ax_slow.contourf(X, Y, Z, cmap="rainbow", levels=50)
                ax_slow.set_xlabel("Sx (s/km)")
                ax_slow.set_ylabel("Sy (s/km)")
                ax_slow.text(
                    0.02, 0.98, info_text,
                    transform=ax_slow.transAxes,
                    ha="left", va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8)
                )

                cbar = plt.colorbar(contour, ax=ax_slow)
                cbar.set_label(clabel)
                plt.tight_layout()
                #cid = self.fig_slow_map.canvas.mpl_connect("key_press_event", self._on_key_press)
                plt.show(block=False)
            except Exception as e:
                print(f"[ERROR] Could not compute or plot FK coherence: {e}")

    def _restore_state(self):
        self._load_picks_from_headers()
        self._redraw_picks()
        self._redraw_all_reference_lines()

    def _load_picks_from_headers(self):
        """Load picks from tr.stats.picks inside self.picks."""
        for tr in self.trace_list:
            if not hasattr(tr.stats, "picks"):
                continue
            for p in tr.stats.picks:
                ts = p.get("time")
                phase = p.get("phase", "?")
                amp = p.get("amplitude", float("nan"))
                pol = p.get("polarity", "?")
                if ts is None:
                    continue
                pick_time = UTCDateTime(ts).datetime
                if tr.id not in self.picks:
                    self.picks[tr.id] = []
                # Evitar duplicados
                already = any(abs((pt - pick_time).total_seconds()) < 1e-4
                              for pt, *_ in self.picks[tr.id])
                if not already:
                    self.picks[tr.id].append((pick_time, phase, amp, pol))

    def __on_enter_axes(self, event):
        ax = event.inaxes
        try:
            self.__selected_ax_index = list(self.axs).index(ax)
        except:
            self.__selected_ax_index = -1

        for selector in self.__selector.values():
            selector.set_visible(False)
            selector.clear()
            selector.clear_subplot()
            selector.new_axes(ax)
            selector.update_background(event)
            selector.set_sub_axes(self.axs)

    def _register_span_selector(self):
        """
        Use ExtendSpanSelector and route selections through __on_span_select.
        """

        def on_select(xmin, xmax):
            print(f"[DEBUG] on_select triggered — current ax idx: {self.__selected_ax_index}")
            print("Selection occurred:", xmin, xmax)
            utc_start = UTCDateTime(mdt.num2date(xmin))
            utc_end = UTCDateTime(mdt.num2date(xmax))
            print(f"[INFO] Selected time window: {utc_start} to {utc_end}")

            # Clear previous vertical lines and labels
            for ax in self.axs:
                lines_to_remove = [
                    line for line in ax.lines
                    if line.get_linestyle() == '--' and line.get_color() == 'blue'
                ]
                for line in lines_to_remove:
                    line.remove()

                texts_to_remove = [
                    txt for txt in ax.texts
                    if txt.get_text() in ('Start', 'End')
                ]
                for txt in texts_to_remove:
                    txt.remove()

            # Store in trace headers
            # if hasattr(self, "displayed_traces"):
            #     for tr in self.displayed_traces:
            #         tr.stats.references = [utc_start, utc_end]
            self.utc_start = utc_start
            self.utc_end = utc_end
            # Visual feedback
            for ax in self.axs:
                ax.axvline(x=xmin, color='blue', linestyle='--', alpha=0.5)
                ax.axvline(x=xmax, color='blue', linestyle='--', alpha=0.5)
            self.axs[0].text(xmin, self.axs[0].get_ylim()[1] * 0.95, 'Start', color='blue', fontsize=8,
                        ha='left', va='top', bbox=dict(fc='white', alpha=0.6))
            self.axs[0].text(xmax, self.axs[0].get_ylim()[1] * 0.95, 'End', color='blue', fontsize=8,
                        ha='right', va='top', bbox=dict(fc='white', alpha=0.6))

            self.fig.canvas.draw_idle()

        # Make sure self.axs is a flat list
        axs = list(self.axs) if isinstance(self.axs, (list, np.ndarray)) else [self.axs]

        self._span_selector = ExtendSpanSelector(
            axs[0],
            onselect=on_select,
            direction="horizontal",
            useblit=False,
            minspan=0.1,
            props=dict(alpha=0.2, facecolor='blue'),
            button=MouseButton.RIGHT,
            sharex=True
        )
        self._span_selector.set_sub_axes(axs)

    def plot_particle_motion(self, z, n, e):

        z_data = z.data - np.mean(z.data)
        n_data = n.data - np.mean(n.data)
        e_data = e.data - np.mean(e.data)

        st = Stream(traces=[z.copy(), n.copy(), e.copy()])
        azimuth, incidence, rect, plan = flinn(st)

        max_val = max(np.max(np.abs(z_data)), np.max(np.abs(n_data)), np.max(np.abs(e_data)))
        lim = 1.05 * max_val
        #incidence = 90 - incidence
        # Plot
        self.fig_part, axs = plt.subplots(2, 2, figsize=(8, 6))
        self.fig_part.suptitle(f"Particle Motion: {z.stats.station}", fontsize=12)

        # --- Z vs N ---
        axs[0, 0].plot(n_data, z_data, linewidth=0.5)
        axs[0, 0].set_xlabel("Radial / North")
        axs[0, 0].set_ylabel("Vertical")
        axs[0, 0].set_xlim(-lim, lim)
        axs[0, 0].set_ylim(-lim, lim)
        axs[0, 0].grid(True, which="both", ls="-", color='grey', alpha=0.4)

        # Incidence angle line
        inc_rad = np.radians(incidence)
        axs[0, 0].plot([0, np.cos(inc_rad) * lim], [0, np.sin(inc_rad) * lim], 'k--', linewidth=0.8)
        axs[0, 0].text(0.05 * lim, 0.9 * lim, f"Inc: {incidence:.1f}°", color='blue', fontsize=9, weight='bold')

        # --- Z vs E ---
        axs[0, 1].plot(e_data, z_data, linewidth=0.5)
        axs[0, 1].set_xlabel("Transversal / East")
        axs[0, 1].set_ylabel("Vertical")
        axs[0, 1].set_xlim(-lim, lim)
        axs[0, 1].set_ylim(-lim, lim)
        axs[0, 1].grid(True, which="both", ls="-", color='grey', alpha=0.4)
        # Add in Z–E panel
        axs[0, 1].plot([0, np.cos(inc_rad) * lim], [0, np.sin(inc_rad) * lim], 'k--', linewidth=0.8)
        axs[0, 1].text(0.05 * lim, 0.9 * lim,
                       f"Inc: {incidence:.1f}°", color='blue', fontsize=9, weight='bold')

        # --- N vs E ---
        axs[1, 0].plot(e_data, n_data, linewidth=0.5)
        axs[1, 0].set_xlabel("Transversal / East")
        axs[1, 0].set_ylabel("Radial / North")
        axs[1, 0].set_xlim(-lim, lim)
        axs[1, 0].set_ylim(-lim, lim)
        axs[1, 0].grid(True, which="both", ls="-", color='grey', alpha=0.4)

        # Add azimuth arrow
        az_rad = np.radians(azimuth)
        arrow_len = lim * 0.8
        axs[1, 0].arrow(0, 0,
                        arrow_len * np.sin(az_rad),
                        arrow_len * np.cos(az_rad),
                        width=0.01 * lim, head_width=0.05 * lim,
                        color='red', edgecolor='black', length_includes_head=True)
        axs[1, 0].text(0.05 * lim, 0.9 * lim, f"Az: {azimuth:.1f}°", color='red', fontsize=9, weight='bold')

        # --- Info box ---
        axs[1, 1].axis("off")
        summary = (f"Azimuth:         {azimuth:.2f}°\n" f"Incidence:       {incidence:.2f}°\n"
                   f"Rectilinearity:  {rect:.2f}\n" f"Planarity:       {plan:.2f}")

        axs[1, 1].text(0.05, 0.6, summary, fontsize=10, va="center", ha="left")

        plt.tight_layout()
        def on_close(event):
            pass  # nothing needed, just let it close

        self.fig_part.canvas.mpl_connect("close_event", on_close)
        plt.show(block=False)  # non-blocking, return immediately to run()

    def clear_plot(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None

    def restore_original(self):
        """Replace trace_list with a fresh copy of the traces as originally loaded,
        keeping the references placed so far (also those set on rotated channels)."""

        current_ids = {tr.id for tr in self.trace_list}
        orig_ids = {tr.id for tr in self.original_traces}

        def station_key(tr):
            return (tr.stats.network, tr.stats.station, tr.stats.location)

        # 1) Collect current references
        refs_by_id = {}  # channels that still exist in the originals
        refs_by_station = {}  # renamed channels (e.g. after rotate)
        for tr in self.trace_list:
            refs = getattr(tr.stats, "references", None)
            if not refs:
                continue
            if tr.id in orig_ids:
                refs_by_id.setdefault(tr.id, []).extend(refs)
            else:
                refs_by_station.setdefault(station_key(tr), []).extend(refs)

        # 2) Fresh copies of the pristine traces
        restored = [tr.copy() for tr in self.original_traces]

        # 3) Re-attach references (without duplicates)
        for tr in restored:
            new_refs = list(refs_by_id.get(tr.id, []))
            if tr.id not in current_ids:
                # channel renamed away by rotate: inherit the station-level references
                new_refs += refs_by_station.get(station_key(tr), [])
            if not new_refs:
                continue
            current = list(getattr(tr.stats, "references", []))
            for ref in new_refs:
                if ref not in current:
                    current.append(ref)
            tr.stats.references = current

        self.trace_list = restored
        self.current_page = 0
        self.clear_plot()

    def _clear_hover_style(self):
        """Remove hover style from previously hovered pick, if any."""
        if self._hover:
            _, line, label, _ = self._hover
            # restore default style (event colour if the pick belongs to an event)
            try:
                line.set_color(getattr(line, "_surfq_meta", {}).get("color", 'r'))
                line.set_linewidth(1.0)
            except Exception:
                pass
            try:
                # restore original label box
                label.set_bbox(dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=0.6))
            except Exception:
                pass
            self._hover = None

    def _on_mouse_move(self, event):
        """Highlight the pick under the cursor (within pixel tolerance)."""
        if event.inaxes not in getattr(self, "axs", []) or event.x is None:
            # moved outside axes—clear any hover
            if self._hover:
                self._clear_hover_style()
                self.fig.canvas.draw_idle()
            return

        ax = event.inaxes
        hovered = None
        min_dist = self._hover_tol_px + 1  # strict less-than

        # Check only picks in this axes
        for trace_id, items in self.pick_lines.items():
            for line, label in items:
                if not line.get_visible() or line.axes is not ax:
                    continue
                try:
                    # vertical line: its xdata is constant
                    xdata = line.get_xdata()
                    if not xdata:
                        continue
                    x_val = xdata[0]
                    # transform to display (pixel) coords
                    xpix = ax.transData.transform((x_val, 0))[0]
                    dist = abs(event.x - xpix)
                    if dist < min_dist:
                        meta = getattr(line, "_surfq_meta", {})
                        pick_time = meta.get("pick_time", None)
                        hovered = (trace_id, line, label, pick_time)
                        min_dist = dist
                except Exception:
                    continue

        # Update hover styling if changed
        if hovered != self._hover:
            self._clear_hover_style()
            self._hover = hovered
            if hovered:
                _, line, label, _ = hovered
                try:
                    line.set_color('orange')
                    line.set_linewidth(1.8)
                except Exception:
                    pass
                try:
                    label.set_bbox(dict(boxstyle='round,pad=0.2', fc='yellow', ec='black', alpha=0.8))
                except Exception:
                    pass
            self.fig.canvas.draw_idle()

    def _delete_pick(self, trace_id, pick_time):
        """Delete a pick (by trace_id and exact pick_time) from UI and state."""
        # 1) Remove from self.picks (in-memory)
        plist = self.picks.get(trace_id, [])
        new_plist = [p for p in plist if p[0] != pick_time]
        if new_plist:
            self.picks[trace_id] = new_plist
        else:
            self.picks.pop(trace_id, None)

        # 2) Remove from trace.stats.picks (persistent in headers)
        for tr in self.trace_list:
            if tr.id == trace_id and hasattr(tr.stats, "picks"):
                # stored as timestamp in your code
                ts = UTCDateTime(pick_time).timestamp
                tr.stats.picks = [p for p in tr.stats.picks if p.get("time") != ts]
                if not tr.stats.picks:
                    del tr.stats.picks

        # 3) Remove artists and clean pick_lines
        items = self.pick_lines.get(trace_id, [])
        keep = []
        for line, label in items:
            meta = getattr(line, "_surfq_meta", {})
            if meta.get("pick_time") == pick_time:
                try:
                    line.remove()
                except Exception:
                    pass
                try:
                    label.remove()
                except Exception:
                    pass
            else:
                keep.append((line, label))
        if keep:
            self.pick_lines[trace_id] = keep
        else:
            self.pick_lines.pop(trace_id, None)

        # 4) Housekeeping
        if self.current_pick and self.current_pick == (trace_id, pick_time):
            self.current_pick = None
        csv_path = self.plot_config.get("pick_output_file")

        if csv_path:
            self.write_isp_table_manual(csv_path)

        self._clear_hover_style()
        if self.plot_config["show_info_picks"]:
            self._update_info_box()
        self.fig.canvas.draw_idle()

    # ==================================================================
    #  PICK FILE IMPORT (single or multi-event) AND EVENT DISPLAY
    # ==================================================================
    # no blue (traces) and no red (picks without event)
    EVENT_COLORS = ("tab:purple", "tab:green", "tab:brown", "tab:pink",
                    "tab:olive", "tab:cyan", "black", "tab:gray")

    _PICK_COLUMNS = {
        "station": ["Station_name", "Station", "STA"],
        "component": ["Component", "CHA", "Channel"],
        "phase": ["P_phase_descriptor", "Phase", "PHASE"],
        "first_motion": ["First_Motion", "FM", "Polarity"],
        "date": ["Date", "YYYYMMDD"],
        "hour_min": ["Hour_min", "Hourmin", "HHMM"],
        "seconds": ["Seconds", "SS", "Sec", "SEC"],
        "amplitude": ["Amplitude", "AMP"],
    }

    @staticmethod
    def _split_pick_blocks(text):
        """
        Split a pick file into blocks. A block ends at a blank line or at a line
        starting with 'Event' (Event_1, Event 1, EVENT...).

        Returns (blocks, explicit): blocks is a list of (event_label or None, [lines]);
        explicit is True when the file contains 'Event' lines.
        A block without header line reuses the header of the previous block.
        """
        import re
        event_re = re.compile(r"^\s*event", re.IGNORECASE)
        blocks, cur, label, explicit = [], [], None, False

        def flush():
            nonlocal cur
            if [ln for ln in cur if ln.strip()]:
                blocks.append((label, cur))
            cur = []

        for ln in text.splitlines():
            if event_re.match(ln):
                flush()
                explicit = True
                tok = ln.split()
                label = " ".join(tok[:2]) if tok[0].lower() == "event" and len(tok) > 1 else tok[0]
            elif not ln.strip():
                if [x for x in cur if x.strip()]:  # end of a block (blank right after 'Event' is ignored)
                    flush()
                    label = None if explicit else label
            else:
                cur.append(ln)
        flush()
        return blocks, explicit

    @staticmethod
    def _fix_gau_header(header, first_row):
        """Old ISP format: header without the GAU column but data rows with a GAU
        token before Err (one token more than the header). Add the column name."""
        h, r = header.split(), first_row.split()
        if "GAU" in h or "Err" not in h or len(r) != len(h) + 1:
            return header
        i = h.index("Err")
        if r[i].upper() != "GAU":
            return header
        return " ".join(h[:i] + ["GAU"] + h[i:])

    @staticmethod
    def _row_time(datestr, hhmm, sec):
        """Date as YYYYMMDD or YYYY-MM-DD, Hour_min as HHMM (or HMM), Seconds as float."""
        import re
        d = re.sub(r"\D", "", str(datestr).strip())
        hm = re.sub(r"\D", "", str(hhmm).split(".")[0]).zfill(4)
        sec = float(sec)
        s_int = int(sec)
        us = int(round((sec - s_int) * 1_000_000))
        if us >= 1_000_000:
            s_int, us = s_int + 1, us - 1_000_000
        t = UTCDateTime(int(d[:4]), int(d[4:6]), int(d[6:8]), int(hm[:-2]), int(hm[-2:]))
        return t + s_int + us * 1e-6

    def _match_trace(self, sta, comp):
        """Trace with this station and channel; fallback: same station and same
        component letter; last fallback: same station."""
        same_sta = [tr for tr in self.trace_list if getattr(tr.stats, "station", "") == sta]
        for tr in same_sta:
            if tr.stats.channel == comp:
                return tr
        for tr in same_sta:
            if comp and tr.stats.channel[-1:] == comp[-1:]:
                return tr
        return same_sta[0] if same_sta else None

    def import_nlloc_obs(self, input_file: str, delimiter=r"\s+"):
        """
        Load picks from an ISP/CSV-like phase table with headers:
          Station_name Instrument Component P_phase_onset P_phase_descriptor First_Motion
          Date Hour_min Seconds Err ErrMag Coda_duration Amplitude Period

        Single or MULTI-EVENT files. Events are separated by blank lines and/or by
        lines starting with 'Event' (e.g. 'Event_1'); each block may repeat the header.
        In a multi-event file every event is drawn as a vertical line with a label,
        and its picks get the same colour.

        Imports into trace.stats.picks (with key 'event' for event picks), self.picks,
        and draws labels/lines. Picks already loaded are not duplicated.
        S-wave polarity on import: '+'->'U', '-'->'D'  (P keeps U/D/?)
        """
        import io
        import pandas as pd

        if not os.path.isfile(input_file):
            print(f"[ERROR] File not found: {input_file}")
            return
        self._set_output_target(input_file, "NLLOC_OBS")

        try:
            with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            print(f"[ERROR] Could not read pick file: {e}")
            return

        blocks, explicit = self._split_pick_blocks(text)
        if not blocks:
            print(f"[INFO] No picks in '{input_file}'.")
            return
        # Blocks are events when the file has 'Event' lines (only those blocks),
        # or when it has several blank-line separated blocks (ISP multi-event).
        multi = explicit or len(blocks) > 1

        new_events = []
        added = duplicated = unmatched = skipped = 0
        header = None

        for n_block, (label, lines) in enumerate(blocks, start=1):
            first = lines[0].split()
            if any(name in first for name in self._PICK_COLUMNS["station"]):
                header = lines[0]
                body = lines
            elif header is not None:
                body = [header] + lines
            else:
                print(f"[WARN] Block {n_block} has no header line, skipped.")
                continue
            if len(body) > 1:
                fixed = self._fix_gau_header(body[0], body[1])
                if fixed != body[0]:
                    body = [fixed] + body[1:]

            try:
                df = pd.read_csv(io.StringIO("\n".join(body)), delimiter=delimiter, engine="python")
            except Exception as e:
                print(f"[WARN] Block {n_block} could not be read: {e}")
                continue

            colmap = {}
            for key, names in self._PICK_COLUMNS.items():
                colmap[key] = next((nm for nm in names if nm in df.columns), None)
            missing = [k for k, v in colmap.items() if v is None and k != "amplitude"]
            if missing:
                print(f"[WARN] Block {n_block}: missing required columns {missing}, skipped.")
                continue

            is_event = multi and (label is not None or not explicit)
            if is_event:
                ev = {"label": label or f"Event_{len(new_events) + 1}", "picks": [],
                      "stations": set(), "matched": 0, "entries": [], "keys": []}
                new_events.append(ev)

            for _, row in df.iterrows():
                try:
                    sta = str(row[colmap["station"]]).strip()
                    comp = str(row[colmap["component"]]).strip()
                    phase = str(row[colmap["phase"]]).strip().upper()
                    fm_up = str(row[colmap["first_motion"]]).strip().upper()
                    t = self._row_time(row[colmap["date"]], row[colmap["hour_min"]],
                                       row[colmap["seconds"]])
                    amp = float(row[colmap["amplitude"]]) if colmap["amplitude"] else float("nan")
                except Exception as e:
                    skipped += 1
                    print(f"[WARN] Row skipped: {e}")
                    continue

                if phase.startswith("S"):
                    pol_ui = {"+": "U", "-": "D"}.get(fm_up, fm_up if fm_up in ("U", "D", "?") else "?")
                else:
                    pol_ui = fm_up if fm_up in ("U", "D", "?") else "?"

                if is_event:
                    ev["picks"].append((t, phase))
                    ev["stations"].add(sta)

                match = self._match_trace(sta, comp)
                if match is None:
                    unmatched += 1
                    continue

                if not hasattr(match.stats, "picks"):
                    match.stats.picks = []
                if any(abs(p.get("time", -1e12) - t.timestamp) < 1e-3 and p.get("phase") == phase
                       for p in match.stats.picks):
                    duplicated += 1
                    if is_event:  # an already loaded pick can still belong to this event
                        ev["keys"].append((match.id, round(t.timestamp, 3)))
                        ev["matched"] += 1
                    continue

                entry = {"time": t.timestamp, "phase": phase, "amplitude": amp, "polarity": pol_ui}
                if is_event:
                    ev["entries"].append(entry)  # 'event' label set once the event is final
                    ev["keys"].append((match.id, round(t.timestamp, 3)))
                    ev["matched"] += 1
                match.stats.picks.append(entry)
                self.picks.setdefault(match.id, []).append((t.datetime, phase, amp, pol_ui))
                added += 1

        # Event time = centroid of the P picks (all picks if there is no P)
        for ev in new_events:
            if not ev["picks"]:
                continue
            times = [t for t, ph in ev["picks"] if ph.startswith("P")] or [t for t, _ in ev["picks"]]
            ev["time"] = UTCDateTime(sum(t.timestamp for t in times) / len(times))
            ev["first"] = min(t for t, _ in ev["picks"])
            ev["last"] = max(t for t, _ in ev["picks"])
            ev["n_picks"] = len(ev["picks"])
            ev["n_stations"] = len(ev["stations"])

            # Same event already loaded (file imported twice): reuse it
            same = next((e for e in self.events if abs(e["time"] - ev["time"]) < 1e-3
                         and e["label"].split("#")[0] == ev["label"]), None)
            if same is None:
                # unique label (two files can both have 'Event_1')
                base, used, k = ev["label"], {e["label"] for e in self.events}, 2
                while ev["label"] in used:
                    ev["label"] = f"{base}#{k}"
                    k += 1
                self.events.append(ev)
                same = ev
            for entry in ev.pop("entries"):
                entry["event"] = same["label"]
            same.setdefault("keys", [])
            same["keys"] = list(dict.fromkeys(same["keys"] + ev.pop("keys")))
        self.events.sort(key=lambda e: e["time"])
        self._reindex_pick_events()

        if self.fig and getattr(self, "displayed_traces", None) and self.axs is not None:
            self._redraw_picks()
            if self.plot_config["show_info_picks"]:
                self._update_info_box()

        msg = f"[INFO] Imported {added} picks from '{input_file}'"
        extra = []
        if duplicated:
            extra.append(f"{duplicated} already loaded")
        if unmatched:
            extra.append(f"{unmatched} without trace")
        if skipped:
            extra.append(f"{skipped} unreadable")
        print(msg + (f" ({', '.join(extra)})." if extra else "."))
        if new_events:
            self.print_event_summary()

    def _reindex_pick_events(self):
        """(trace_id, timestamp) -> event index, rebuilt after sorting self.events."""
        self._pick_event = {key: i for i, e in enumerate(self.events) for key in e.get("keys", [])}

    def print_event_summary(self):
        """Print the imported events: time, picks, stations, time span."""
        if not self.events:
            print("[INFO] No events loaded.")
            return
        total = sum(e.get("n_picks", 0) for e in self.events)
        print(f"[INFO] {len(self.events)} event(s), {total} picks:")
        print(f"   {'#':>3}  {'Event':<14} {'Time (P centroid)':<23} {'Picks':>5} "
              f"{'Shown':>5} {'Stations':>8} {'Span (s)':>9}")
        for i, e in enumerate(self.events, start=1):
            span = e["last"] - e["first"]
            print(f"   {i:>3}  {e['label'][:14]:<14} {e['time'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]:<23} "
                  f"{e['n_picks']:>5} {e['matched']:>5} {e['n_stations']:>8} {span:>9.2f}")

    def _pick_color(self, trace_id, pick_time):
        """Colour of the event the pick belongs to, red if it belongs to none."""
        try:
            ts = UTCDateTime(pick_time).timestamp
        except Exception:
            return 'r'
        idx = self._pick_event.get((trace_id, round(ts, 3)))
        if idx is None:
            return 'r'
        return self.EVENT_COLORS[idx % len(self.EVENT_COLORS)]

    def _remove_event_artists(self):
        for artist in self.event_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.event_artists = []

    def _draw_events(self):
        """
        Draw every imported event inside the displayed time range: a solid vertical
        line on all subplots (event colour) and a label on the top subplot.
        Disabled with plot_config['show_events'] = False.
        """
        self._remove_event_artists()
        if not self.events or not self.plot_config.get("show_events", True):
            return
        if self.axs is None or not getattr(self, "displayed_traces", None):
            return

        t_min = min(tr.stats.starttime for tr in self.displayed_traces)
        t_max = max(tr.stats.endtime for tr in self.displayed_traces)
        ax0 = self.axs[0]

        for i, ev in enumerate(self.events):
            t = ev.get("time")
            if t is None or not (t_min <= t <= t_max):
                continue  # drawing it would stretch the time axis
            color = self.EVENT_COLORS[i % len(self.EVENT_COLORS)]
            x = mdt.date2num(t.datetime)
            for ax in self.axs:
                line = ax.axvline(x=x, color=color, linestyle='-', linewidth=1.2, alpha=0.8, zorder=4)
                line._surfq_event = ev["label"]
                self.event_artists.append(line)
            # label at the top of the first subplot (x in data units, y in axes fraction)
            label = ax0.text(x, 0.98, f"{ev['label']}\n{t.strftime('%H:%M:%S.%f')[:-4]}",
                             transform=ax0.get_xaxis_transform(), fontsize=7, color="white",
                             ha="center", va="top", zorder=6, clip_on=True,
                             bbox=dict(boxstyle="round,pad=0.2", fc=color, ec=color, alpha=0.85))
            self.event_artists.append(label)

    def write_isp_table_manual(self, filepath: str, delimiter: str = " ", write_header_when_empty: bool = True):
        """
        Overwrite an ISP/CSV-like table with current picks.
        If there are no picks, still overwrite the file:
          - If write_header_when_empty=True (default): write only the header row.
          - If False: create an empty file.
        """
        import os, tempfile
        from obspy import UTCDateTime

        header = ("Station_name Instrument Component P_phase_onset P_phase_descriptor First_Motion "
                  "Date Hour_min Seconds Err ErrMag Coda_duration Amplitude Period")

        rows = []
        event_rows = {}  # event label -> rows (picks imported from a multi-event file)
        for tr in self.trace_list:
            if not hasattr(tr.stats, "picks"):
                continue
            sta = getattr(tr.stats, "station", "") or "UNK"
            inst = "?"
            comp = getattr(tr.stats, "channel", "") or "???"
            onset = "?"

            for p in tr.stats.picks:
                ts = p.get("time")
                if ts is None:
                    continue
                phase = (p.get("phase") or "?").upper()
                pol = (p.get("polarity") or "?").upper()

                # ISP/NLLoc polarity mapping:
                fm = "+" if (phase.startswith("S") and pol == "U") else \
                    "-" if (phase.startswith("S") and pol == "D") else \
                        (pol if pol in ("U", "D", "?") else "?")

                amp = p.get("amplitude", 0.0)
                period = p.get("period", 0.0)
                coda = p.get("coda", 0.0)
                err_model = "GAU"
                err_mag = 0.0

                t = UTCDateTime(ts)
                date = t.strftime("%Y%m%d")
                hhmm = t.strftime("%H%M")

                # Seconds must be SS.sss with leading zero, e.g., 02.560
                sec_val = t.second + t.microsecond * 1e-6
                if sec_val >= 60:
                    sec_val = 59.999  # guard rare round-up
                seconds_str = f"{sec_val:06.3f}"

                row = [
                    sta, inst, comp, onset, phase, fm,
                    date, hhmm, seconds_str, err_model, f"{err_mag:.2E}",
                    f"{float(coda):.1f}", f"{float(amp):.2f}", f"{float(period):.2f}"
                ]
                if p.get("event"):
                    event_rows.setdefault(p["event"], []).append(row)
                else:
                    rows.append(row)

        # Always overwrite (even if rows is empty)
        dirpath = os.path.dirname(os.path.abspath(filepath)) or "."
        with tempfile.NamedTemporaryFile("w", delete=False, dir=dirpath) as tf:
            tmp = tf.name
            if rows or (write_header_when_empty and not event_rows):
                tf.write(header + "\n")
            for r in rows:
                tf.write(delimiter.join(map(str, r)) + "\n")
            # picks of imported events: one block per event, so the file can be re-imported
            order = [e["label"] for e in getattr(self, "events", [])]
            order += [lab for lab in event_rows if lab not in order]
            for lab in order:
                if lab not in event_rows:
                    continue
                line = lab if lab.lower().startswith("event") else f"Event {lab}"
                tf.write(f"\n{line}\n{header}\n")
                for r in event_rows[lab]:
                    tf.write(delimiter.join(map(str, r)) + "\n")
        os.replace(tmp, filepath)

        n_rows = len(rows) + sum(len(v) for v in event_rows.values())
        if n_rows:
            print(f"[INFO] Wrote {n_rows} picks to ISP table (overwritten): {filepath}")
        else:
            msg = "[INFO] No picks remain — CSV overwritten "
            msg += "(header only)." if write_header_when_empty else "(empty file)."
            print(msg, filepath)



