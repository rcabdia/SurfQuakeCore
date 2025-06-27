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
import platform
import time
from collections import defaultdict
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter
from obspy import UTCDateTime, Stream
from obspy.core.trace import Trace
import matplotlib as mplt
import matplotlib.dates as mdt
import numpy as np
from obspy.signal.polarization import flinn
from surfquakecore.arrayanalysis import array_analysis
from surfquakecore.data_processing.spectral_tools import SpectrumTool
from surfquakecore.data_processing.wavelet import ConvolveWaveletScipy
from surfquakecore.seismoplot.crosshair import BlittedCursor
from surfquakecore.seismoplot.plot_command_prompt import PlotCommandPrompt
from surfquakecore.seismoplot.spanselector import ExtendSpanSelector
from surfquakecore.utils.obspy_utils import MseedUtil
from matplotlib.backend_bases import MouseButton
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
            "plot_type": "standard",  # ← NEW: 'record' for record section and overlay for all traces at the same plot
            "pick_output_file": "./picks.csv",
            "sharey": False,
            "show_crosshair": False,
            "show_arrivals": False}

        self.available_types = ['standard', 'record', 'overlay']
        self.enable_command_prompt = kwargs.pop("interactive", False)
        # Override defaults with user config if provided
        if plot_config:
            self.plot_config.update(plot_config)

        self.picks = {}
        self.current_pick = None
        self.pick_lines = {}
        self.fig = None
        self.axs = None
        self.pick_type = "P"
        self.prompt_active = False
        self.utc_start = None
        self.utc_end = None

    def _get_geodetic_info(self, trace: Trace) -> Tuple[float, float]:
        """
        Returns distance (km) and back-azimuth from trace header.
        """
        try:
            dist, az, baz, incidence_ang = trace.stats.geodetic['geodetic']
            return dist, baz
        except Exception:
            return float('inf'), float('inf')

    def plot(self, page=0):

        if platform.system() == 'Darwin':
            mplt.use("MacOSX")
        else:
            mplt.use("TkAgg")

        self.current_page = page
        plot_type = self.plot_config.get("plot_type", "standard")

        if plot_type == "record":
            self._plot_record_section()

        elif plot_type == "overlay":
            self._plot_overlay_traces()

        elif plot_type == "standard":
            self._plot_standard_traces(page=page)

        return self.trace_list  # Return modified trace list

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
            gridspec_kw={'hspace': self.plot_config["vspace"]}
        )

        # Ensure axs is always a list
        if len(sub_traces) == 1:
            self.axs = [self.axs]

        # Connect enter event once
        if not hasattr(self, "_entered_hook"):
            self.fig.canvas.mpl_connect("axes_enter_event", self.__on_enter_axes)
            self._entered_hook = True
        self.__selected_ax_index = 0

        # Crosshair cursor
        if self.plot_config.get("show_crosshair", True):
            self.cursors = []
            for ax in self.axs:
                cursor = BlittedCursor(ax, self.axs)
                ax.figure.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
                self.cursors.append(cursor)

        self._setup_pick_interaction()
        self._restore_state()

        # Plot each trace
        for ax_idx, (ax, tr) in enumerate(zip(self.axs, sub_traces)):
            t = tr.times("matplotlib")

            if sort_mode:
                dist, baz = self._get_geodetic_info(tr)
                label = f"{tr.id} | Dist: {dist:.1f} km | Baz: {baz:.1f}°"
            else:
                label = tr.id

            ax.plot(t, tr.data, linewidth=0.75, label=label)

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
                ax.legend()

            # Annotate with date
            starttime = tr.stats.starttime
            date_str = starttime.strftime("%Y-%m-%d")
            textstr = f"JD {starttime.julday} / {starttime.year}\n{date_str}"
            ax.text(0.01, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.5))

            # Format x-axis
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdt.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdt.AutoDateLocator())

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

        # Set x-limits across all axes
        auto_start = min(tr.times("matplotlib")[0] for tr in sub_traces)
        auto_end = max(tr.times("matplotlib")[-1] for tr in sub_traces)
        for ax in self.axs:
            ax.set_xlim(auto_start, auto_end)

        plt.ion()
        plt.tight_layout()

        if self.enable_command_prompt:
            plt.show(block=False)
            self.fig.canvas.draw_idle()
            plt.pause(0.5)
            print("[INFO] Type 'command parameter', 'help', 'n' to next, 'b' to previous, 'p' for picking mode")
            prompt = PlotCommandPrompt(self)
            result = prompt.run()
            if result == "p":
                self.enable_command_prompt = False
                self._register_span_selector()
                plt.show(block=True)
        else:
            self._register_span_selector()
            plt.show(block=True)

    def _plot_record_section(self):

        plt.close(self.fig)

        cfg = self.plot_config
        traces = self.trace_list
        phase_curves = defaultdict(list)
        N_traces = len(traces)
        # Collect arrival times grouped by phase
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
                    seen_phases.add(phase)  # only first arrival for this phase
        # Sort by distance
        traces.sort(key=lambda tr: self._get_geodetic_info(tr)[0])
        distances = [self._get_geodetic_info(tr)[0] for tr in traces]
        scale = cfg.get("scale_factor", 1.0)

        # Compute global start time for alignment
        fig, ax = plt.subplots(figsize=(12, 8))
        self.fig = fig  # ensure fig is accessible throughout
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

        for tr, dist in zip(traces, distances):
            # Normalize trace to its max amplitude
            norm_data = tr.data / np.max(np.abs(tr.data)) if np.max(np.abs(tr.data)) != 0 else tr.data
            # Align time axis to earliest start
            t = tr.times("matplotlib")

            if N_traces >= 12:
                ax.plot(t, norm_data * scale + dist, color="black", alpha=0.75, linewidth=0.6)
            else:
                ax.plot(t, norm_data * scale + dist, alpha=0.75, linewidth=0.6, label=tr.id)

        # Plot arrival time curves for each phase
        for phase, points in phase_curves.items():
            if len(points) < 2:
                continue  # skip too short
            points.sort()  # ensure time ordering
            times, dists = zip(*points)
            ax.plot(times, dists, linestyle='--', linewidth=1.0, alpha=0.6, label=f"{phase}")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Distance (km)")
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdt.DateFormatter('%H:%M:%S'))
        ax.set_title("Record Section")

        min_time = min(tr.stats.starttime for tr in traces)
        max_time = max(tr.stats.endtime for tr in traces)
        ax.set_xlim(mdt.date2num(min_time.datetime), mdt.date2num(max_time.datetime))

        if cfg["show_legend"]:
            ax.legend(fontsize=6)

        if self.enable_command_prompt:
            plt.show(block=False)
            self.fig.canvas.draw_idle()
            plt.pause(0.5)

            print("[INFO] Type 'command parameter', 'help', 'exit' or 'p' to return to picking mode")
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
        for tr in traces:
            t = tr.times("matplotlib")

            if n > 12:
                ax.plot(t, tr.data, color="black", linewidth=0.8, alpha=0.7)
            else:
                ax.plot(t, tr.data, linewidth=0.8, alpha=0.7, label=tr.id)
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Normalized Amplitude")
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdt.DateFormatter('%H:%M:%S'))
        ax.set_title("Overlay of All Traces")

        if self.plot_config["show_legend"] or n <= 12:
            ax.legend(fontsize=6)

        plt.tight_layout()
        plt.show(block=True)

    def _setup_pick_interaction(self):
        def filtered_doubleclick(event):
            if event.button == MouseButton.RIGHT:
                return  # allow SpanSelector to use right-click
            if event.dblclick and event.button == MouseButton.LEFT:
                self._on_doubleclick(event)

        self.fig.canvas.mpl_connect('button_press_event', filtered_doubleclick)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.info_box = self.fig.text(0.75, 0.8, "", ha='left', va='top', fontsize=9)
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
            header_needed = not os.path.exists(csv_path)
            with open(csv_path, "a") as f:
                if header_needed:
                    f.write("id,time,phase,amplitude,polarity\n")
                f.write(f"{trace.id},{pick_time.isoformat()},{phase},{amplitude:.4f},{polarity}\n")

        # Add to in-memory pick tracking
        if trace.id not in self.picks:
            self.picks[trace.id] = []
        self.picks[trace.id].append((pick_time, phase, amplitude, polarity))

        # Draw pick line and update display
        self.current_pick = (trace.id, pick_time)
        self._draw_pick_lines(trace.id, ax, pick_time)
        self._update_info_box()
        self.fig.canvas.draw()

    def _draw_pick_lines(self, trace_id, ax, pick_time):
        """Draw vertical line and annotate pick info."""
        if trace_id not in self.pick_lines:
            self.pick_lines[trace_id] = []

        x = mdt.date2num(pick_time)
        line = ax.axvline(x=x, color='r', linestyle='--', alpha=0.8)

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
        """Handle key presses for pick management."""

        # Reference marker key
        if event.key == 'w' and event.inaxes in self.axs:
            ref_time = mdt.num2date(event.xdata).replace(tzinfo=None)
            utc_ref_time = UTCDateTime(ref_time)
            self.last_reference = utc_ref_time
            # Store the reference in all traces
            for tr in self.trace_list:
                if not hasattr(tr.stats, "references"):
                    tr.stats.references = []
                tr.stats.references.append(utc_ref_time.timestamp)

            # Draw the reference line on all axes
            for ax in self.axs:
                ax.axvline(x=mdt.date2num(ref_time), color='g', linestyle='--', alpha=0.5)

            self.fig.canvas.draw()
            print(f"[INFO] Reference time added at {utc_ref_time.isoformat()}")

        elif event.key == 'd' and self.current_pick:
            trace_id, pick_time = self.current_pick
            # Remove from in-memory picks
            if trace_id in self.picks:
                self.picks[trace_id] = [p for p in self.picks[trace_id] if p[0] != pick_time]
                if not self.picks[trace_id]:
                    del self.picks[trace_id]

            # Remove from trace.stats
            for tr in self.trace_list:
                if tr.id == trace_id and hasattr(tr.stats, "picks"):
                    tr.stats.picks = [p for p in tr.stats.picks if p["time"] != UTCDateTime(pick_time)]
                    if not tr.stats.picks:
                        del tr.stats.picks

        elif event.key == 'n':
            # Go to next page
            if (self.current_page + 1) * self.plot_config["traces_per_fig"] < len(self.trace_list):
                self.current_page += 1
                self._plot_standard_traces(self.current_page)
            else:
                print("[INFO] Already at last page.")

        elif event.key == 'b':
            # Go back to previous page
            if self.current_page > 0:
                self.current_page -= 1
                self._plot_standard_traces(self.current_page)
            else:
                print("[INFO] Already at first page.")

            self._redraw_picks()
            self._update_info_box()
            self.current_pick = None

        elif event.key == 'c':
            self._clear_picks()

        elif event.key == 'p':
            removed_any = False
            for tr in self.trace_list:
                if hasattr(tr.stats, "references") and tr.stats.references:
                    last_ref = tr.stats.references.pop()
                    removed_any = True
                    if not tr.stats.references:
                        del tr.stats.references

            if removed_any:
                print("[INFO] Last reference time removed from all traces.")
            else:
                print("[INFO] No reference times to remove.")

            # Redraw to remove last green line (could be refined for exact removal)
            self._redraw_all_reference_lines()
            self.fig.canvas.draw()

        elif event.key == 'm':
            self._remove_last_reference()

        elif event.key == 'v':

            self.enable_command_prompt = True
            self.plot(page=self.current_page)

        elif event.key == 'e':
            # Ignore invalid clicks
            if event.inaxes not in self.axs:
                return

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
                header_needed = not os.path.exists(csv_path)
                with open(csv_path, "a") as f:
                    if header_needed:
                        f.write("id,time,phase,amplitude,polarity\n")
                    f.write(f"{trace.id},{pick_time.isoformat()},{phase},{amplitude:.4f},{polarity}\n")

            # Add to in-memory pick tracking
            if trace.id not in self.picks:
                self.picks[trace.id] = []
            self.picks[trace.id].append((pick_time, phase, amplitude, polarity))

            # Draw pick line and update display
            self.current_pick = (trace.id, pick_time)
            self._draw_pick_lines(trace.id, ax, pick_time)
            self._update_info_box()
            self.fig.canvas.draw()

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

        self.current_pick = None
        self._update_info_box()
        self.fig.canvas.draw()

    def get_picks(self):
        """Return picks as {trace_id: [UTCDateTime]}."""
        return {k: [UTCDateTime(p) for p in v]
                for k, v in self.picks.items() if v}

    def _update_info_box(self):

        """Update the side info box with picks."""

        lines = ["Picks:"]
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

    def _plot_single_spectrum(self, idx, axis_type):

        trace = self.displayed_traces[idx]
        spectrum, freqs = SpectrumTool.compute_spectrum(trace, trace.stats.delta)
        trace = self.displayed_traces[idx]

        try:
            stime = self.utc_start
        except:
            stime = trace.stats.starttime

        try:
            etime = self.utc_end
        except:
            etime = trace.stats.endtime

        print("Spectrum window: ", stime, etime)

        trace.trim(starttime=stime, endtime=etime)
        self.fig_spec, self.ax_spec = plt.subplots()

        if axis_type == "loglog":
            self.ax_spec.loglog(freqs, spectrum, label=trace.id, linewidth=0.75)
        elif axis_type == "xlog":
            self.ax_spec.semilogx(freqs, spectrum, label=trace.id, linewidth=0.75)
        elif axis_type == "ylog":
            self.ax_spec.semilogy(freqs, spectrum, label=trace.id, linewidth=0.75)

        self.ax_spec.set_ylim(spectrum.min() / 10.0, spectrum.max() * 100.0)
        self.ax_spec.set_xlabel("Frequency (Hz)")
        self.ax_spec.set_ylabel("Amplitude")
        self.ax_spec.set_title(f"Spectrum for {trace.id}")
        self.ax_spec.legend()
        plt.grid(True, which="both", ls="-", color='grey', alpha=0.4)
        plt.tight_layout()
        plt.show(block=False)

        # Poll until the figure is closed
        while plt.fignum_exists(self.fig_spec.number):
            plt.pause(0.2)
            time.sleep(0.1)

    def _plot_all_spectra(self, axis_type):

        stime_trim = etime_trim = None
        self.fig_spec, self.ax_spec = plt.subplots()

        st = Stream(self.displayed_traces)

        try:
            stime_trim = self.utc_start
        except:
            print("Not starttime set")

        try:
            etime_trim = self.utc_end
        except:
            print("Not endtime set")

        if stime_trim is not None and etime_trim is not None:

            print("Spectrum window: ", stime_trim, etime_trim)
            st.trim(starttime=stime_trim, endtime=etime_trim)


            for trace in st:
                spectrum, freqs = SpectrumTool.compute_spectrum(trace, trace.stats.delta)
                if axis_type == "loglog":
                    self.ax_spec.loglog(freqs, spectrum, label=trace.id, linewidth=0.75)
                elif axis_type == "xlog":
                    self.ax_spec.semilogx(freqs, spectrum, label=trace.id, linewidth=0.75)
                elif axis_type == "ylog":
                    self.ax_spec.semilogy(freqs, spectrum, label=trace.id, linewidth=0.75)

            self.ax_spec.set_xlabel("Frequency (Hz)")
            self.ax_spec.set_ylabel("Amplitude")
            self.ax_spec.legend(fontsize=6)
            plt.grid(True, which="both", ls="-", color='grey')
            plt.tight_layout()
            plt.show(block=False)

            # Poll until the figure is closed
            while plt.fignum_exists(self.fig_spec.number):
                plt.pause(0.2)
                time.sleep(0.1)
        else:
            print("Please select set starttime and endtime using span selector, "
                  "click right mouse and dragging it to the right")

    def _plot_spectrogram(self, idx, win_sec=5.0, overlap_percent=50.0):

        trace = self.displayed_traces[idx]
        try:
            stime = self.utc_start
        except:
            stime = trace.stats.starttime

        try:
            etime = self.utc_end
        except:
            etime = trace.stats.endtime

        print("Spectrogram window: ", stime, etime)

        trace.trim(starttime=stime, endtime=etime)

        spectrum, num_steps, t, f = SpectrumTool.compute_spectrogram(
            trace.data,
            win=int(win_sec * trace.stats.sampling_rate),
            dt=trace.stats.delta,
            linf=0,
            lsup=int(trace.stats.sampling_rate // 2),
            step_percentage=(100-overlap_percent)*1E-2
        )

        # --- Set up GridSpec with reserved space for colorbar ---
        self.fig_spec = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.03], height_ratios=[1, 1],
                               hspace=0.02, wspace=0.02)

        ax_waveform = self.fig_spec.add_subplot(gs[0, 0])
        ax_spec = self.fig_spec.add_subplot(gs[1, 0], sharex=ax_waveform)
        ax_cbar = self.fig_spec.add_subplot(gs[1, 1])
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))  # Forces scientific notation always
        ax_waveform.yaxis.set_major_formatter(formatter)

        # --- Plot waveform ---
        ax_waveform.plot(trace.times(), trace.data, linewidth=0.75)
        ax_waveform.set_title(f"Spectrogram for {trace.id}")
        ax_waveform.tick_params(labelbottom=False)

        # --- Plot spectrogram ---
        pcm = ax_spec.pcolormesh(
            t, f, 10 * np.log10(spectrum / np.max(spectrum)),
            shading='auto', cmap='rainbow'
        )
        ax_waveform.set_ylabel('Amplitude')
        ax_spec.set_ylabel('Frequency [Hz]')
        ax_spec.set_xlabel('Time [s]')

        # --- Add colorbar without shifting axes ---
        cbar = self.fig_spec.colorbar(pcm, cax=ax_cbar, orientation='vertical')
        cbar.set_label("Power [dB]")

        plt.show(block=False)

        while plt.fignum_exists(self.fig_spec.number):
            plt.pause(0.2)
            time.sleep(0.1)

    def _plot_wavelet(self, idx, wavelet_type, param, **kwargs):

        if wavelet_type == "cm":
            wavelet_type = "Complex Morlet"
        elif wavelet_type == "mh":
            wavelet_type = "Mexican Hat"
        elif wavelet_type == "pa":
            wavelet_type = "Paul"

        param = float(param)
        tr = self.displayed_traces[idx]
        f_min = kwargs.pop("fmin", 0.5)
        f_max = kwargs.pop("fmax", tr.stats.sampling_rate//2)

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

        cw = ConvolveWaveletScipy(tr)
        tt = int(tr.stats.sampling_rate/f_min)
        cw.setup_wavelet(wmin=param, wmax=param, tt=tt, fmin=f_min, fmax=f_max, nf=80,
                         use_wavelet=wavelet_type, m=param, decimate=False)
        scalogram2 = cw.scalogram_in_dbs()

        t = np.linspace(0, tr.stats.delta * scalogram2.shape[1], scalogram2.shape[1])
        f = np.logspace(np.log10(f_min), np.log10(f_max), scalogram2.shape[0])
        x, y = np.meshgrid(t, f)

        c_f = param / 2 * math.pi
        ff = np.linspace(f_min, f_max, scalogram2.shape[0])
        pred = (math.sqrt(2) * c_f / ff) - (math.sqrt(2) * c_f / f_max)

        pred_comp = t[len(t) - 1] - pred
        # --- Set up GridSpec with reserved space for colorbar ---
        self.fig_spec = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.03], height_ratios=[1, 1],
                               hspace=0.02, wspace=0.02)

        ax_waveform = self.fig_spec.add_subplot(gs[0, 0])
        ax_spec = self.fig_spec.add_subplot(gs[1, 0], sharex=ax_waveform)
        ax_cbar = self.fig_spec.add_subplot(gs[1, 1])

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))  # Forces scientific notation always
        ax_waveform.yaxis.set_major_formatter(formatter)

        # --- Plot waveform ---
        ax_waveform.plot(tr.times(), tr.data, linewidth=0.75)
        ax_waveform.set_title(f"CWT Scalogram for {tr.id}")
        ax_waveform.tick_params(labelbottom=False)

        # --- Plot scalogram ---
        pcm = ax_spec.pcolormesh(x, y, scalogram2, shading='auto', cmap='rainbow')
        ax_spec.fill_between(pred, ff, 0, color="black", edgecolor="red", alpha=0.3)
        ax_spec.fill_between(pred_comp, ff, 0, color="black", edgecolor="red", alpha=0.3)
        ax_waveform.set_ylabel('Amplitude')
        ax_spec.set_ylim([np.min(f), np.max(f)])
        ax_spec.set_ylabel('Frequency [Hz]')
        ax_spec.set_xlabel('Time [s]')

        # --- Add colorbar without shifting axes ---
        cbar = self.fig_spec.colorbar(pcm, cax=ax_cbar, orientation='vertical')
        cbar.set_label("Power [dB]")

        plt.show(block=False)

        while plt.fignum_exists(self.fig_spec.number):
            plt.pause(0.2)
            time.sleep(0.1)


    def _run_fk(self, **kwargs):
        try:
            plt.close(self.fig_fk)
            plt.close(self.fig_slow_map)
        except:
            pass

        # TODO IMPLEMENT THE METHOD "CAPON or MUSIC"
        self.timewindow = kwargs.pop("timewindow", 3)
        self.overlap = kwargs.pop("overlap", 0.05)
        self.fmin = kwargs.pop("fmin", 0.8)
        self.fmax = kwargs.pop("fmax", 2.2)
        self.smax = kwargs.pop("smax", 0.3)
        self.slow_grid = kwargs.pop("slow_grid", 0.05)
        self.method_beam = kwargs.pop("method", "FK")


        if self.utc_start is not None and self.utc_end is not None:
            try:
                stime = self.utc_start
                etime = self.utc_end

                print("FK window: ", stime, etime)
                traces = Stream(self.trace_list)
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
                plt.show(block=False)

                # Keep figure open
                while plt.fignum_exists(self.fig_fk.number):
                    plt.pause(0.2)
                    time.sleep(0.1)
            except:
                print("annot compute fk analysis, please review the inventory and parameter values")
        else:
            print("A time window span needs to be selected")

    def _on_fk_key_press(self, event):

        if event.key == 'e':

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
                # Get data
                traces = Stream(self.trace_list)
                traces_slow = traces.copy()
                selection = MseedUtil.filter_inventory_by_stream(traces_slow, self.inventory)
                wavenumber = array_analysis.array()

                if self.method_beam == "FK" or self.method_beam == "CAPON":
                    Z, Sxpow, Sypow, coord = wavenumber.FKCoherence(
                        traces, selection, xdata, self.fmin, self.fmax, self.smax, self.timewindow,
                        self.slow_grid, self.method_beam)

                elif self.method_beam == "MUSIC":
                    Z, Sxpow, Sypow, coord = wavenumber.run_music(traces, selection, xdata, self.fmin, self.fmax,
                                                              self.smax, self.timewindow, self.slow_grid, "MUSIC")

                backacimuth = wavenumber.azimuth2mathangle(np.arctan2(Sypow, Sxpow) * 180 / np.pi)
                slowness = np.abs(Sxpow, Sypow)

                # Build slowness grid
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

                if self.method_beam == "FK" or self.method_beam == "CAPON":
                    clabel = "Power"
                elif self.method_beam == "MTP.COHERENCE":
                    clabel = "Magnitude Coherence"
                elif self.method_beam == "MUSIC":
                    clabel = "MUSIC Pseudospectrum"
                #

                # Plot
                self.fig_slow_map, ax_slow = plt.subplots(figsize=(8, 5))
                contour = ax_slow.contourf(X, Y, Z, cmap="rainbow", levels=50)
                ax_slow.set_xlabel("Sx (s/km)")
                ax_slow.set_ylabel("Sy (s/km)")
                # Add top-left annotation text (instead of title)
                ax_slow.text(
                    0.02, 0.98, info_text,
                    transform=ax_slow.transAxes,
                    ha="left", va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8)
                )

                #ax_slow.set_title(f"Beam Power at {time_clicked.strftime('%H:%M:%S')}")
                cbar = plt.colorbar(contour, ax=ax_slow)
                cbar.set_label(clabel)
                plt.tight_layout()
                plt.show(block=False)

                while plt.fignum_exists(self.fig_slow_map.number):
                    plt.pause(0.2)
                    time.sleep(0.1)

            except Exception as e:
                print(f"[ERROR] Could not compute or plot FK coherence: {e}")

    def _restore_state(self):
        self._redraw_picks()
        self._redraw_all_reference_lines()

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
        plt.show(block=False)

        # Hold open as long as figure exists
        while plt.fignum_exists(self.fig_part.number):
            plt.pause(0.2)
            time.sleep(0.1)

    def clear_plot(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None

