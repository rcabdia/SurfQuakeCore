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
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter
from obspy import UTCDateTime, Stream
from obspy.core.trace import Trace
import matplotlib as mplt
import matplotlib.dates as mdt
import numpy as np
import matplotlib.dates as mdates
from surfquakecore.arrayanalysis import array_analysis
from surfquakecore.data_processing.spectral_tools import SpectrumTool
from surfquakecore.data_processing.wavelet import ConvolveWaveletScipy
from surfquakecore.seismoplot.crosshair import BlittedCursor
from surfquakecore.seismoplot.plot_command_prompt import PlotCommandPrompt
from surfquakecore.utils.obspy_utils import MseedUtil


# Choose the best backend before importing pyplot, more Options: TkAgg, MacOSX, Qt5Agg, QtAgg, WebAgg, Agg

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
        self.trace_list = list(stream)
        self.inventory = kwargs.pop("inventory", None)

        # Default plotting configuration
        self.plot_config = {
            "traces_per_fig": 6,
            "sort_by": False,  # options: distance, backazimuth, None
            "vspace": 0.05,
            "title_fontsize": 9,
            "show_legend": True,
            "autosave": False,
            "plot_type": "standard",  # ← NEW: 'record' for record section and overlay for all traces at the same plot
            "save_folder": "./plots",
            "pick_output_file": "./picks.csv",
            "sharey": False,
            "show_crosshair": True}

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

        plot_type = self.plot_config.get("plot_type", "standard")

        if plot_type == "record":
            self._plot_record_section()
        elif plot_type == "overlay":
            self._plot_overlay_traces()
        else:
            self._plot_standard_traces()

        return self.trace_list  # Return modified trace list

    def _plot_standard_traces(self):

        plt.close(self.fig)
        formatter_pow = ScalarFormatter(useMathText=True)
        formatter_pow.set_powerlimits((0, 0))  # Forces scientific notation always

        traces = self.trace_list
        if self.plot_config["sort_by"] == 'distance':
            traces.sort(key=lambda tr: self._get_geodetic_info(tr)[0])
        elif self.plot_config["sort_by"] == 'backazimuth':
            traces.sort(key=lambda tr: self._get_geodetic_info(tr)[1])

        n_traces = len(traces)
        if n_traces == 0:
            print("No traces to plot.")
            return

        traces_per_fig = self.plot_config["traces_per_fig"]

        for i in range(0, n_traces, traces_per_fig):
            sub_traces = traces[i:i + traces_per_fig]
            max_traces = min(8, 2 * len(sub_traces))
            self.fig, self.axs = plt.subplots(len(sub_traces), 1, figsize=(12, max_traces),
                sharex=True, sharey=self.plot_config["sharey"],
                gridspec_kw={'hspace': self.plot_config["vspace"]}
            )
            if len(sub_traces) == 1:
                self.axs = [self.axs]

            if self.plot_config.get("show_crosshair", True):
                self.cursors = []
                for ax in self.axs:
                    cursor = BlittedCursor(ax, self.axs)
                    ax.figure.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
                    self.cursors.append(cursor)

            self._setup_pick_interaction()
            self._restore_state()

            # Inside your plot() method, after ax.plot(...)
            for i, (ax, tr) in enumerate(zip(self.axs, sub_traces)):
                t = tr.times("matplotlib")

                if self.plot_config["sort_by"]:

                    dist, baz = self._get_geodetic_info(tr)
                    ax.plot(t, tr.data, linewidth=0.75,
                            label=f"{tr.id} | Dist: {dist:.1f} km | Baz: {baz:.1f}°")
                else:
                    ax.plot(t, tr.data, linewidth=0.75, label = tr.id)


                if self.plot_config["show_legend"] and len(self.axs) <= 9:
                    ax.legend()

                    # Add starttime info box in top-left of each subplot
                    starttime = tr.stats.starttime
                    julday = starttime.julday
                    year = starttime.year
                    date_str = starttime.strftime("%Y-%m-%d")

                    textstr = f"JD {julday} / {year}\n{date_str}"
                    ax.text(0.01, 0.95, textstr,
                            transform=ax.transAxes,
                            fontsize=8, va='top', ha='left',
                            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.5))

                # Set x-axis to datetime format
                ax.xaxis_date()
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())

                # Hide x-axis ticks and bottom spine if not last
                if i < len(self.axs) - 1:
                    ax.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
                    ax.spines['bottom'].set_visible(False)
                else:
                    ax.tick_params(axis='x', which='both', labelbottom=True, bottom=True)
                    ax.spines['bottom'].set_visible(True)

                for spine in ['top', 'right']:
                    ax.spines[spine].set_visible(False)

                ax.tick_params(axis='x')
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                ax.yaxis.set_major_formatter(formatter_pow)
                ax.yaxis.get_offset_text().set_visible(True)

            auto_start = min(tr.times("matplotlib")[0] for tr in sub_traces)
            auto_end = max(tr.times("matplotlib")[-1] for tr in sub_traces)


            for ax in self.axs:
                ax.set_xlim(auto_start, auto_end)

            plt.ion()
            plt.tight_layout()

            if self.plot_config["autosave"]:
                os.makedirs(self.plot_config["save_folder"], exist_ok=True)
                fig_path = os.path.join(self.plot_config["save_folder"], f"waveform_plot_{i // traces_per_fig + 1}.png")
                plt.savefig(fig_path, dpi=150)
                print(f"[INFO] Plot saved to {fig_path}")

            else:

                if self.enable_command_prompt:

                    # plt.show(block=False)
                    # # Simulate a non-blocking wait for GUI responsiveness
                    # print("[INFO] Waiting for GUI to be responsive...")
                    # start_time = time.time()
                    # while not plt.fignum_exists(self.fig.number):
                    #     plt.pause(0.10)
                    #     if time.time() - start_time > 25:  # Timeout failsafe
                    #         print("[WARNING] GUI did not become active.")
                    #         break
                    #
                    # # Give it one last short pause to fully draw
                    # plt.pause(0.5)

                    plt.show(block=False)
                    self.fig.canvas.draw_idle()  # schedule a draw
                    plt.pause(0.5)  # give time for GUI to process at least one frame

                    # Then launch prompt
                    print("[INFO] Type 'command parameter' or 'q' to next set of traces'")
                    prompt = PlotCommandPrompt(self)
                    result = prompt.run()

                    if result == "pick":
                        # Go back to interactive picking for this figure only
                        plt.show(block=True)
                else:
                    plt.show(block=True)


    def _plot_record_section(self):

        cfg = self.plot_config
        traces = self.trace_list

        # Sort by distance
        traces.sort(key=lambda tr: self._get_geodetic_info(tr)[0])
        distances = [self._get_geodetic_info(tr)[0] for tr in traces]
        scale = cfg.get("scale_factor", 1.0)

        # Compute global start time for alignment
        #t0 = min(tr.stats.starttime for tr in traces)
        fig, ax = plt.subplots(figsize=(12, 8))

        for tr, dist in zip(traces, distances):
            # Normalize trace to its max amplitude
            norm_data = tr.data / np.max(np.abs(tr.data)) if np.max(np.abs(tr.data)) != 0 else tr.data

            # Align time axis to earliest start
            t = tr.times("matplotlib")
            ax.plot(t, norm_data * scale + dist, linewidth=0.6, label=tr.id)

        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Distance (km)")
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.set_title("Record Section")

        if cfg["show_legend"]:
            ax.legend(fontsize=6)

        if cfg["autosave"]:
            os.makedirs(cfg["save_folder"], exist_ok=True)
            fig_path = os.path.join(cfg["save_folder"], f"record_section.png")
            plt.savefig(fig_path, dpi=150)
            print(f"[INFO] Record section saved to {fig_path}")
        else:
            plt.tight_layout()
            plt.show(block=True)

    def _plot_overlay_traces(self):
        traces = self.trace_list

        if len(traces) == 0:
            print("No traces to plot.")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        for tr in traces:
            t = tr.times("matplotlib")
            ax.plot(t, tr.data, linewidth=0.8, alpha=0.7, label=tr.id)

        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Normalized Amplitude")
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.set_title("Overlay of All Traces")

        if self.plot_config["show_legend"]:
            ax.legend(fontsize=6)

        if self.plot_config["autosave"]:
            os.makedirs(self.plot_config["save_folder"], exist_ok=True)
            fig_path = os.path.join(self.plot_config["save_folder"], "overlay_plot.png")
            plt.savefig(fig_path, dpi=150)
            print(f"[INFO] Overlay plot saved to {fig_path}")
        else:
            plt.tight_layout()
            plt.show(block=True)

    def _setup_pick_interaction(self):
        """Set up double-click mouse event and key press events."""
        self.fig.canvas.mpl_connect('button_press_event', self._on_doubleclick)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        #plt.gcf().text(0.1, 0.95,
        #               "Click to place picks | 'd' to delete last | 'c' to clear",
        #               fontsize=9)
        # Reserve space for displaying pick info
        self.info_box = self.fig.text(0.75, 0.8, "", ha='left', va='top', fontsize=9)
        self._update_info_box()
        self._restore_state()


    def _on_doubleclick(self, event):
        """Handle double-clicks to place picks."""

        # Ignore invalid clicks
        if not event.dblclick or event.inaxes not in self.axs:
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
            "time": UTCDateTime(pick_time),
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

            # Store the reference in all traces
            for tr in self.trace_list:
                if not hasattr(tr.stats, "references"):
                    tr.stats.references = []
                tr.stats.references.append(utc_ref_time)

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
            self.plot()

    def _redraw_picks(self):
        """Redraw all pick lines only on their corresponding axes."""
        # Remove all existing pick lines
        for lines in self.pick_lines.values():
            for line, label in lines:
                line.remove()
                label.remove()
        self.pick_lines = {}

        for tr_idx, trace in enumerate(self.trace_list):
            trace_id = trace.id
            if trace_id in self.picks:
                ax = self.axs[tr_idx]
                for pick in self.picks[trace_id]:
                    if len(pick) == 4:
                        pt, _, _, _ = pick
                    else:
                        pt, _, _ = pick
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

    def _plot_single_spectrum(self, idx):

        trace = self.trace_list[idx]
        spectrum, freqs = SpectrumTool.compute_spectrum(trace, trace.stats.delta)

        self.fig_spec, self.ax_spec = plt.subplots()
        self.ax_spec.loglog(freqs, spectrum, label=trace.id, linewidth=0.75)
        self.ax_spec.set_ylim(spectrum.min() / 10.0, spectrum.max() * 100.0)
        self.ax_spec.set_xlabel("Frequency (Hz)")
        self.ax_spec.set_ylabel("Amplitude")
        self.ax_spec.set_title(f"Spectrum for {trace.id}")
        self.ax_spec.legend()
        plt.tight_layout()
        plt.show(block=False)

        # Poll until the figure is closed
        while plt.fignum_exists(self.fig_spec.number):
            plt.pause(0.2)
            time.sleep(0.1)

    def _plot_all_spectra(self):

        self.fig_spec, self.ax_spec = plt.subplots()

        for trace in self.trace_list:
            spectrum, freqs = SpectrumTool.compute_spectrum(trace, trace.stats.delta)
            self.ax_spec.loglog(freqs, spectrum, label=trace.id, linewidth=0.75)

        self.ax_spec.set_xlabel("Frequency (Hz)")
        self.ax_spec.set_ylabel("Amplitude")
        self.ax_spec.legend(fontsize=6)
        plt.tight_layout()
        plt.show(block=False)

        # Poll until the figure is closed
        while plt.fignum_exists(self.fig_spec.number):
            plt.pause(0.2)
            time.sleep(0.1)

    def _plot_spectrogram(self, idx, win_sec=5.0, overlap_percent=50.0):

        trace = self.trace_list[idx]
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
        tr = self.trace_list[idx]
        f_min = kwargs.pop("fmin", 0.5)
        f_max = kwargs.pop("fmax", tr.stats.sampling_rate//2)

        cw = ConvolveWaveletScipy(tr)
        tt = int(tr.stats.sampling_rate/ f_min)
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
        ax_waveform.set_title(f"Spectrogram for {tr.id}")
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

        # TODO IMPLEMENT THE METHOD
        self.timewindow = kwargs.pop("timewindow", 3)
        self.overlap = kwargs.pop("overlap", 0.05)
        self.fmin = kwargs.pop("fmin", 0.8)
        self.fmax = kwargs.pop("fmax", 2.2)
        self.smax = kwargs.pop("smax", 0.3)
        self.slow_grid = kwargs.pop("slow_grid", 0.05)

        try:
            stime = self.trace_list[0].stats.references[0]
        except:
            stime = self.trace_list[0].stats.starttime
        try:
            etime = self.trace_list[0].stats.references[1]
        except:
            etime = self.trace_list[0].stats.stats.endtime

        print("FK window: ", stime, etime)
        traces = Stream(self.trace_list)

        traces_fk = traces.copy()
        selection = MseedUtil.filter_inventory_by_stream(traces, self.inventory)
        wavenumber = array_analysis.array()
        self.relpower, self.abspower, self.AZ, self.Slowness, self.T = wavenumber.FK(traces_fk, selection, stime, etime,
                                                                                     self.fmin, self.fmax, self.smax,
                                                                                     self.slow_grid,
                                                                                     self.timewindow, self.overlap)

        # --- Create grid layout with reserved space for colorbar ---
        self.fig_fk = plt.figure(figsize=(9, 6))
        self.fig_fk.canvas.mpl_connect("button_press_event", self._on_fk_doubleclick)
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

        formatter = mdates.DateFormatter('%H:%M:%S')
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

    def _on_fk_doubleclick(self, event):
        """On double-click, show time and plot slowness map at that moment."""
        if not event.dblclick or event.inaxes is None:
            return

        ax = event.inaxes
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

            Z, Sxpow, Sypow, coord = wavenumber.FKCoherence(
                traces, selection, xdata, self.fmin, self.fmax, self.smax, self.timewindow, self.slow_grid, "FK"
            )

            # Build slowness grid
            Sx = np.arange(-1 * self.smax, self.smax, self.slow_grid)[np.newaxis]
            nx = len(Sx[0])
            x = y = np.linspace(-1 * self.smax, self.smax, nx)
            X, Y = np.meshgrid(x, y)

            # Plot
            self.fig_slow_map, ax_slow = plt.subplots(figsize=(8, 5))
            contour = ax_slow.contourf(X, Y, Z, cmap="rainbow", levels=50)
            ax_slow.set_xlabel("Sx (s/km)")
            ax_slow.set_ylabel("Sy (s/km)")
            ax_slow.set_title(f"FK Coherence at {time_clicked.strftime('%H:%M:%S')}")
            cbar = plt.colorbar(contour, ax=ax_slow)
            cbar.set_label("Normalized Power")
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








