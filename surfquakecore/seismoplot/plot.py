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
from obspy import UTCDateTime
from obspy.core.trace import Trace
import matplotlib as mplt
import matplotlib.dates as mdt
import numpy as np
import matplotlib.dates as mdates
from surfquakecore.data_processing.spectral_tools import SpectrumTool
from surfquakecore.data_processing.wavelet import ConvolveWaveletScipy


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

        # Default plotting configuration
        self.plot_config = {
            "traces_per_fig": 6,
            "sort_by": None,  # options: distance, backazimuth, None
            "vspace": 0.0,
            "title_fontsize": 9,
            "show_legend": True,
            "autosave": False,
            "plot_type": "standard",  # ← NEW: 'record' for record section and overlay for all traces at the same plot
            "save_folder": "./plots",
            "pick_output_file": "./picks.csv"
        }

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
                sharex=True,
                gridspec_kw={'hspace': self.plot_config["vspace"]}
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
                ax.set_title(str(tr.stats.starttime), fontsize=self.plot_config["title_fontsize"])
                if self.plot_config["show_legend"]:
                    ax.legend()

                # Set x-axis to datetime format
                ax.xaxis_date()
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
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
                    plt.show(block=False)
                    # Simulate a non-blocking wait for GUI responsiveness
                    print("[INFO] Waiting for GUI to be responsive...")
                    start_time = time.time()
                    while not plt.fignum_exists(self.fig.number):
                        plt.pause(0.10)
                        if time.time() - start_time > 25:  # Timeout failsafe
                            print("[WARNING] GUI did not become active.")
                            break

                    # Give it one last short pause to fully draw
                    plt.pause(0.5)

                    # Then launch prompt
                    print("[INFO] Type 'command parameter' or 'n' to next set of traces'")
                    self.command_prompt()
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
        plt.gcf().text(0.1, 0.95,
                       "Click to place picks | 'd' to delete last | 'c' to clear",
                       fontsize=9)
        # Reserve space for displaying pick info
        self.info_box = self.fig.add_axes([0.8, 0.1, 0.18, 0.8], frameon=False)
        self.info_box.axis('off')
        self._update_info_box()

    def _on_doubleclick(self, event):
        """Handle double-clicks to place picks."""

        # Ignore invalid clicks
        if not event.dblclick or event.inaxes not in self.axs:
            return

        # Check which subplot was clicked
        ax = event.inaxes
        tr_idx = np.where(self.axs == ax)[0][0]
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

        # Reference marker key
        if event.key == 'r' and event.inaxes in self.axs:
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

        elif event.key == 'x':
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

        elif event.key == 'k':
            self._remove_last_reference()

    def _redraw_picks(self):
        """Redraw all pick lines only on their corresponding axes."""
        # Remove all existing pick lines
        for lines in self.pick_lines.values():
            for line in lines:
                line.remove()
        self.pick_lines = {}

        # Redraw picks on correct axes
        for tr_idx, trace in enumerate(self.trace_list):
            trace_id = trace.id
            if trace_id in self.picks:
                ax = self.axs[tr_idx]  # Match trace to its axis
                for pick in self.picks[trace_id]:
                    if len(pick) == 4:
                        pt, _, _, _ = pick
                    else:
                        pt, _, _ = pick
                    x = mdt.date2num(pt)
                    line = ax.axvline(x=x, color='r', linestyle='--', alpha=0.7)
                    if trace_id not in self.pick_lines:
                        self.pick_lines[trace_id] = []
                    self.pick_lines[trace_id].append(line)

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
            for pick in pick_list:
                if len(pick) == 4:
                    pt, ptype, amp, pol = pick
                else:
                    pt, ptype, amp = pick
                    pol = "?"
                time_str = pt.strftime('%H:%M:%S.%f')[:-3]
                lines.append(f"{tr_id}: {time_str} ({ptype}, {pol}) amp={amp:.2f}")

        self.info_box.text(0, 1, '\n'.join(lines), va='top', fontsize=9)

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

    def command_prompt(self):
        if self.prompt_active:
            return  # Prevent re-entry if already running
        self.prompt_active = True

        while True:
            cmd = input(">> ").strip().lower()
            if cmd == "n":
                self.prompt_active = False
                break

            elif cmd.startswith("spectrogram") or cmd.startswith("spec"):
                parts = cmd.split()
                if len(parts) == 4:
                    _, index_str, win_str, overlap_str = parts
                    try:
                        idx = int(index_str)
                        win_sec = float(win_str)
                        overlap = float(overlap_str)
                        if 0 <= idx < len(self.trace_list):
                            self._plot_spectrogram(idx, win_sec, overlap)
                        else:
                            print(f"[ERROR] Index {idx} out of range.")
                    except ValueError:
                        print("[ERROR] Usage: spectrogram <index> <win_sec> <overlap%>")

                elif len(parts) == 2:
                    _, index_str = parts
                    try:
                        idx = int(index_str)
                        if 0 <= idx < len(self.trace_list):
                            # Use default values if not provided
                            self._plot_spectrogram(idx)
                        else:
                            print(f"[ERROR] Index {idx} out of range.")
                    except ValueError:
                        print("[ERROR] Invalid index for spectrogram")
                else:
                    print("[ERROR] Use: spectrogram <index> [<win_sec> <overlap%>]")

            elif cmd.startswith("cwt"):
                parts = cmd.split()
                if len(parts) == 6:
                    _, index_str, wavelet_type, param, fmin, fmax = parts
                    try:
                        idx = int(index_str)
                        wavelet_type = str(wavelet_type)
                        param = float(param)
                        fmin = float(fmin)
                        fmax = float(fmax)
                        if 0 <= idx < len(self.trace_list):
                            self._plot_wavelet(idx, wavelet_type, param, fmin=fmin, fmax=fmax)
                        else:
                            print(f"[ERROR] Index {idx} out of range.")
                    except ValueError:
                        print("[ERROR] Usage: cwt <index> <wavelet_type> <parameter>")
                elif len(parts) == 4:
                    _, index_str, wavelet_type, param = parts
                    try:
                        idx = int(index_str)
                        if 0 <= idx < len(self.trace_list):
                            # Use default values if not provided
                            self._plot_wavelet(idx, wavelet_type, param)
                        else:
                            print(f"[ERROR] Index {idx} out of range.")
                    except ValueError:
                        print("[ERROR] Invalid index for spectrogram")
                else:
                    print("[ERROR] Use: spectrogram <index> [<win_sec> <overlap%>]")

            elif cmd.startswith("spectrum") or cmd.startswith("sp"):
                parts = cmd.split()
                if len(parts) == 2:
                    _, index = parts
                    if index == "all":
                        self._plot_all_spectra()
                    elif index.isdigit():
                        idx = int(index)
                        if 0 <= idx < len(self.trace_list):
                            self._plot_single_spectrum(idx)
                        else:
                            print(f"[ERROR] Index {idx} out of range.")
                    else:
                        print("[ERROR] Invalid spectrum command.")
                else:
                    print("[ERROR] Use 'spectrum all' or 'spectrum <index>'")

            else:
                print(f"[WARN] Unknown command: {cmd}")

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
            wavelet_type  = "Complex Morlet"
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

        # --- Plot scaloogram ---
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
