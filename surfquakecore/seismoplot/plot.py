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
import os
import platform
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from obspy.core.trace import Trace
import matplotlib as mplt
import matplotlib.dates as mdt
import numpy as np
import matplotlib.dates as mdates
# Choose best backend before importing pyplot,
# more Options: TkAgg, MacOSX, Qt5Agg, QtAgg, WebAgg, Agg

class PlotProj:
    def __init__(self, stream,
                 plot_config: Optional[dict] = None):
        """
        Parameters
        ----------
        stream : Stream or list of Traces
        plot_config : dict, optional
            Dictionary of plotting preferences.
        """
        self.trace_list = list(stream)

        # Default plotting configuration
        self.plot_config = {
            "traces_per_fig": 6,
            "sort_by": None,  # options: distance, backazimuth, None
            "vspace": 0.0,
            "title_fontsize": 9,
            "show_legend": True,
            "autosave": False,
            "save_folder": "./plots"
        }

        # Override defaults with user config if provided
        if plot_config:
            self.plot_config.update(plot_config)

        self.picks = {}
        self.current_pick = None
        self.pick_lines = {}
        self.fig = None
        self.axs = None
        self.pick_type = "P"


    def _get_geodetic_info(self, trace: Trace) -> Tuple[float, float]:
        """
        Returns distance (km) and back-azimuth from trace header.
        """
        try:
            dist, az, baz, incidence_ang = trace.stats.geodetic['geodetic']
            return dist, baz
        except Exception:
            return float('inf'), float('inf')

    def plot(self):
        if platform.system() == 'Darwin':
            mplt.use("MacOSX")
        elif platform.system() == 'Linux':
            mplt.use("TkAgg")
        cfg = self.plot_config
        traces = self.trace_list

        if cfg["sort_by"] == 'distance':
            traces.sort(key=lambda tr: self._get_geodetic_info(tr)[0])
        elif cfg["sort_by"] == 'backazimuth':
            traces.sort(key=lambda tr: self._get_geodetic_info(tr)[1])

        n_traces = len(traces)
        if n_traces == 0:
            print("No traces to plot.")
            return

        traces_per_fig = cfg["traces_per_fig"]

        for i in range(0, n_traces, traces_per_fig):
            sub_traces = traces[i:i + traces_per_fig]
            self.fig, self.axs = plt.subplots(
                len(sub_traces), 1,
                figsize=(10, 2 * len(sub_traces)),
                sharex=True,
                gridspec_kw={'hspace': cfg["vspace"]}
            )
            if len(sub_traces) == 1:
                self.axs = [self.axs]

            self._setup_pick_interaction()

            # Inside your plot() method, after ax.plot(...)
            for ax, tr in zip(self.axs, sub_traces):
                t = tr.times("matplotlib")
                dist, baz = self._get_geodetic_info(tr)
                ax.plot(t, tr.data, linewidth=0.75,
                        label=f"{tr.id} | Dist: {dist:.1f} km | Baz: {baz:.1f}°")
                ax.set_title(str(tr.stats.starttime), fontsize=cfg["title_fontsize"])
                if cfg["show_legend"]:
                    ax.legend()

                # ✅ Set x-axis to datetime format
                ax.xaxis_date()
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.tick_params(axis='x')

            auto_start = min(tr.times("matplotlib")[0] for tr in sub_traces)
            auto_end = max(tr.times("matplotlib")[-1] for tr in sub_traces)
            for ax in self.axs:
                ax.set_xlim(auto_start, auto_end)

            if cfg["autosave"]:
                os.makedirs(cfg["save_folder"], exist_ok=True)
                fig_path = os.path.join(cfg["save_folder"], f"waveform_plot_{i // traces_per_fig + 1}.png")
                plt.savefig(fig_path, dpi=150)
                print(f"[INFO] Plot saved to {fig_path}")
            else:
                plt.ion()
                plt.tight_layout()
                plt.show(block=True)

    def _setup_pick_interaction(self):
        """Set up double-click mouse event and key press events."""
        self.fig.canvas.mpl_connect('button_press_event', self._on_doubleclick)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        plt.gcf().text(0.1, 0.95,
                       "Click to place picks | 'd' to delete last | 'c' to clear",
                       fontsize=9)
        # Reserve space for displaying pick info
        self.info_box = self.fig.add_axes([0.8, 0.1, 0.18, 0.8], frameon=False)
        self.info_box.axis('off')
        self._update_info_box()

    def _on_doubleclick(self, event):
        """Handle double-clicks to place picks."""
        # if event.dblclick is not True or event.inaxes not in self.axs:
        #     return
        if event.dblclick == 3 or event.inaxes not in self.axs:
            return  # Only left-clicks inside trace axes
        # Check if click was in any of our axes
        if not any(ax == event.inaxes for ax in self.axs.flat):
            return

        if event.dblclick:
            # Find which trace was clicked
            ax = event.inaxes
            tr_idx = np.where(self.axs.flat == event.inaxes)[0][0]
            trace = self.trace_list[tr_idx]
            pick_time = mdt.num2date(event.xdata).replace(tzinfo=None)

            # Ask for pick type on each pick
            try:
                print("Input prompt")
                pick_type = input(f"Enter pick type for {trace.id} at {pick_time.strftime('%H:%M:%S')}: ").strip()
                if not pick_type:
                    pick_type = "?"
            except Exception:
                pick_type = "?"

                # Convert pick_time to trace-relative seconds
            rel_time = (pick_time - trace.stats.starttime.datetime).total_seconds()

            # Interpolate amplitude at that time
            times = trace.times()
            amplitude = np.interp(rel_time, times, trace.data)

            # Store pick
            if trace.id in self.picks:
                self.picks[trace.id].append((pick_time, pick_type, amplitude))
            else:
                self.picks[trace.id] = [(pick_time, pick_type, amplitude)]

            self.current_pick = (trace.id, pick_time)
            self._draw_pick_lines(trace.id, ax, pick_time)
            self._update_info_box()
            self.fig.canvas.draw()

    def _draw_pick_lines(self, trace_id, ax, pick_time):
        """Draw vertical line only on the selected axis."""
        if trace_id not in self.pick_lines:
            self.pick_lines[trace_id] = []

        x = mdt.date2num(pick_time)
        line = ax.axvline(x=x, color='r', linestyle='--', alpha=0.8)
        self.pick_lines[trace_id].append(line)

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
        if event.key == 'd' and self.current_pick:
            trace_id, pick_time = self.current_pick
            if trace_id in self.picks:
                self.picks[trace_id] = [p for p in self.picks[trace_id] if p[0] != pick_time]
            self._draw_pick_lines()
            self._update_info_box()

        elif event.key == 'c':
            self._clear_picks()

    def _redraw_picks(self):
        """Redraw all pick lines."""
        for line in self.pick_lines:
            line.remove()
        self.pick_lines = []

        # Redraw remaining picks
        for picks in self.picks.values():
            for pick in picks:
                x = mdt.date2num(pick)
                for ax in self.axs:
                    line = ax.axvline(x=x, color='r', linestyle='--', alpha=0.7)
                    self.pick_lines.append(line)

        self.fig.canvas.draw()

    def _clear_picks(self, event=None):
        for trace_id in self.picks:
            self.picks[trace_id] = []

        for lines in self.pick_lines.values():
            for line in lines:
                line.remove()
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
        self.info_box.clear()
        self.info_box.axis('off')

        lines = ["Picks:"]
        for tr_id, pick_list in self.picks.items():
            for pt, ptype, amp in pick_list:
                time_str = pt.strftime('%H:%M:%S.%f')[:-3]
                lines.append(f"{tr_id}: {time_str} ({ptype}) amp={amp:.2f}")

        self.info_box.text(0, 1, '\n'.join(lines), va='top', fontsize=9)

    def set_pick_type(self, pick_type: str):
        """Set current pick type (e.g., 'P', 'S', etc.)"""
        self.pick_type = pick_type.upper()
        print(f"[INFO] Pick type set to: {self.pick_type}")

if __name__ == "__main__":
    print("READY TO TEST")
    # def _get_station_coords(self, trace: Trace) -> Optional[Tuple[float, float]]:
    #     """
    #     Retrieve station coordinates from metadata.
    #     """
    #     if self.metadata is None:
    #         return None
    #
    #     network = trace.stats.network
    #     station = trace.stats.station
    #
    #     if isinstance(self.metadata, Inventory):
    #         try:
    #             coords = self.metadata.get_coordinates(f"{network}.{station}")
    #             return coords['latitude'], coords['longitude']
    #         except Exception:
    #             return None
    #     elif isinstance(self.metadata, dict):
    #         return self.metadata.get(f"{network}.{station}")
    #
    #     return None

    # def _compute_distance_azimuth(self, trace: Trace) -> Tuple[float, float]:
    #     """
    #     Prefer distance and backazimuth from trace header if available.
    #     """
    #     if "geodetic" in trace.stats and isinstance(trace.stats.geodetic, dict):
    #         try:
    #             dist, az, baz = trace.stats.geodetic['geodetic']
    #             return dist, baz
    #         except Exception:
    #             pass  # fallback below
    #
    #     # Fallback: compute from station coordinates and epicenter
    #     if trace.id in self._dist_az_cache:
    #         return self._dist_az_cache[trace.id]
    #
    #     coords = self._get_station_coords(trace)
    #     if coords is None or self.epicenter is None:
    #         return float('inf'), float('inf')
    #
    #     epi_lat, epi_lon = self.epicenter
    #     sta_lat, sta_lon = coords
    #     dist_m, az, baz = gps2dist_azimuth(epi_lat, epi_lon, sta_lat, sta_lon)
    #     dist_km = dist_m / 1000.0
    #
    #     self._dist_az_cache[trace.id] = (dist_km, baz)
    #     return dist_km, baz
