#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
beamforming obj
"""

from matplotlib import gridspec
from obspy import Inventory, UTCDateTime
from surfquakecore.arrayanalysis import array_analysis
from surfquakecore.utils.obspy_utils import MseedUtil
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import gzip
import platform
import matplotlib.dates as mdt

class TraceBeamResult:
    def __init__(self, **kwargs):

        self.timewindow = kwargs.pop("timewindow", 3)
        self.overlap = kwargs.pop("overlap", 0.05)
        self.fmin = kwargs.pop("fmin", 0.8)
        self.fmax = kwargs.pop("fmax", 2.2)
        self.smax = kwargs.pop("smax", 0.3)
        self.slow_grid = kwargs.pop("slow_grid", 0.05)
        self.method_beam = kwargs.pop("method", "FK")
        self.inventory = kwargs.pop("inventory", None)
        self.st = kwargs.pop("stream", None)

    def compute_beam(self):
        if self.st is None or len(self.st) == 0:
            raise ValueError("[ERROR] Stream is empty or not set.")

        if self.inventory is None or not isinstance(self.inventory, Inventory):
            raise TypeError("[ERROR] 'inventory' must be a valid ObsPy Inventory object.")

        # Determine common time window across all traces
        start_times = [tr.stats.starttime for tr in self.st]
        end_times = [tr.stats.endtime for tr in self.st]

        common_start = max(start_times)
        common_end = min(end_times)

        if common_start >= common_end:
            raise ValueError("[ERROR] Invalid trimming window — traces do not overlap in time.")

        # Trim all traces to the common interval
        self.st.trim(starttime=common_start, endtime=common_end, pad=True, fill_value=0)

        # Filter inventory stations to match stream
        selection = MseedUtil.filter_inventory_by_stream(self.st, self.inventory)

        # Run beamforming
        wavenumber = array_analysis.array()
        self.relpower, self.abspower, self.AZ, self.Slowness, self.T = wavenumber.FK(
            self.st, selection,
            common_start, common_end,
            self.fmin, self.fmax,
            self.smax, self.slow_grid,
            self.timewindow, self.overlap
        )

    def plot_beam(self, save_path: str = None):

        if not hasattr(self, "relpower") or self.relpower is None:
            raise RuntimeError("[ERROR] Beam not computed. Call compute_beam() first.")

        import matplotlib.pyplot as plt
        import matplotlib as mplt

        if platform.system() == 'Darwin':
            mplt.use("MacOSX")
        else:
            mplt.use("Qt5Agg")

        # --- Create grid layout with reserved space for colorbar ---
        self.fig_fk = plt.figure(figsize=(9, 6))
        # self.fig_fk.canvas.mpl_connect("button_press_event", self._on_fk_doubleclick)
        self.fig_fk.canvas.mpl_connect('key_press_event', self._on_beam_key_press)
        gs = gridspec.GridSpec(3, 2, width_ratios=[35, 1], height_ratios=[1, 1, 1], hspace=0.0,
                               wspace=0.02)

        # Create subplots
        ax0 = self.fig_fk.add_subplot(gs[0, 0])
        ax1 = self.fig_fk.add_subplot(gs[1, 0], sharex=ax0)
        ax2 = self.fig_fk.add_subplot(gs[2, 0], sharex=ax0)
        ax_cbar = self.fig_fk.add_subplot(gs[:, 1])  # colorbar takes all rows
        t1 = UTCDateTime(mdt.num2date(self.T[0]))
        date_str = t1.strftime("%Y-%m-%d")
        textstr = f"JD {t1.julday} / {t1.year}\n{date_str}"

        # Scatter plots
        sc0 = ax0.scatter(self.T, self.relpower, c=self.relpower, cmap='rainbow', s=20)
        sc1 = ax1.scatter(self.T, self.Slowness, c=self.relpower, cmap='rainbow', s=20)
        sc2 = ax2.scatter(self.T, self.AZ, c=self.relpower, cmap='rainbow', s=20)
        ax0.text(0.01, 0.95, textstr, transform=ax0.transAxes, fontsize=8,
                va='top', ha='left', bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.5))
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
        if save_path:
            self.fig_fk.savefig(save_path, dpi=300)
            plt.close(self.fig_fk)
        else:
            plt.show()

    def _on_beam_key_press(self, event):
        method_map = {
            '1': "FK",
            '2': "CAPON",
            '3': "MTP.COHERENCE",
            '4': "MUSIC",
            '5': "MUSIC_2_signals",
            '6': "MUSIC_3_signals"
        }

        if event.key in method_map:
            self.method_beam = method_map[event.key]
            print(f"[INFO] Method selected: {self.method_beam}")
            return

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
                traces = self.st
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
                slowness = np.abs(Sxpow, Sypow)

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
                plt.show()

            except Exception as e:
                print(f"[ERROR] Could not compute or plot FK coherence: {e}")


    def to_pickle(self, folder_path: str, compress: bool = True):
        """
        Serialize the full object to a pickle file.
        """

        # Parse argument for folder path

        if not folder_path:
            print("[ERROR] --folder_path must be specified")
            return

        # Ensure the output directory exists
        if not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path)
                print(f"[INFO] Created folder: {folder_path}")
            except Exception as e:
                print(f"[ERROR] Failed to create folder '{folder_path}': {e}")
                return

        t1 = self.st[0].stats.starttime
        base_name = f"{t1.year}.{t1.julday}"
        path_output = os.path.join(folder_path, base_name)

        counter = 1
        while os.path.exists(path_output + ".beam"):
            path_output = os.path.join(folder_path, f"{base_name}_{counter}")
            counter += 1

        path_output += ".sp"

        open_func = gzip.open if compress else open
        mode = 'wb'

        with open_func(path_output, mode) as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[INFO] Writing beam result to {path_output}")

    @staticmethod
    def from_pickle(filepath: str, compress: bool = True):
        """
        Load the full object from a pickle file.
        """
        open_func = gzip.open if compress else open
        mode = 'rb'

        with open_func(filepath, mode) as f:
            obj = pickle.load(f)

        if not isinstance(obj, TraceBeamResult):
            raise TypeError("Pickle file does not contain a TraceSpectrumResult object.")

        return obj

    def detect_beam_peaks(self, phase_dict="regional", min_power=0.75, peak_kwargs=None,
                          bazimuth_range=None, output_file=None, min_time_separation=10.0):
        """
        Detect relative maxima (peaks) in self.relpower using optional slowness and backazimuth constraints.

        Parameters:
            phase_dict (dict or str): Phase slowness dictionary or shortcut ("regional", "teleseismic").
            min_power (float): Minimum relative power to accept a peak.
            peak_kwargs (dict): Arguments passed to scipy.signal.find_peaks().
            bazimuth_range (tuple): Optional (min, max) azimuth range in degrees.
            output_file (str): Optional file path to append detected results.
            min_time_separation (float): Minimum time (in seconds) between peaks across all phases.

        Returns:
            dict: {phase_name: list of (datetime, azimuth, slowness, power)}
        """
        from scipy.signal import find_peaks
        from matplotlib.dates import num2date
        from datetime import datetime

        regional_phases = {
            "Pg": (0.20, 0.35),
            "Pn": (0.06, 0.10),
            "Lg": (0.15, 0.25),
            "Rg": (0.25, 0.40),
            "Sn": (0.08, 0.14),
        }

        teleseismic_phases = {
            "P": (0.04, 0.07),
            "PP": (0.06, 0.08),
            "PKP": (0.05, 0.07),
            "S": (0.07, 0.12),
            "SS": (0.09, 0.13),
            "SKS": (0.07, 0.10),
            "ScS": (0.09, 0.13),
            "PcP": (0.05, 0.07),
            "Rayleigh": (0.20, 0.40),
            "Love": (0.20, 0.35),
        }

        if isinstance(phase_dict, str):
            key = phase_dict.lower()
            if key == "regional":
                phase_dict = regional_phases
            elif key == "teleseismic":
                phase_dict = teleseismic_phases
            else:
                raise ValueError(f"[ERROR] Unknown phase_dict shortcut: '{phase_dict}'")

        if peak_kwargs is None:
            peak_kwargs = {"prominence": 0.1, "distance": 10}

        results = {}
        seen_phase_times = set()
        accepted_times = []

        for phase, (smin, smax) in phase_dict.items():
            mask = (self.Slowness >= smin) & (self.Slowness <= smax)
            if not np.any(mask):
                continue

            masked_power = np.where(mask, self.relpower, 0.0)
            peaks, _ = find_peaks(masked_power, **peak_kwargs)

            phase_results = []
            for idx in peaks:
                power = self.relpower[idx]
                if power < min_power:
                    continue

                az = self.AZ[idx]
                if bazimuth_range:
                    baz_min, baz_max = bazimuth_range
                    if not (baz_min <= az <= baz_max):
                        continue

                t = num2date(self.T[idx]).replace(tzinfo=None)
                slow = self.Slowness[idx]
                key = (phase, t.isoformat())

                # Avoid exact duplicates
                if key in seen_phase_times:
                    continue

                # Enforce global time separation across phases
                if any(abs((t - tprev).total_seconds()) < min_time_separation for tprev in accepted_times):
                    continue

                seen_phase_times.add(key)
                accepted_times.append(t)
                phase_results.append((t, az, slow, power))

            if phase_results:
                # Sort by power descending and keep the strongest
                phase_results.sort(key=lambda x: x[3], reverse=True)
                results[phase] = [phase_results[0]]

        # Optional file write
        if output_file:
            try:
                with open(output_file, "a") as f:
                    for phase, picks in results.items():
                        for t, az, slow, pwr in picks:
                            f.write(f"{phase:5s},{t.isoformat()},{az:6.2f} "
                                    f"{slow:6.3f},{pwr:5.3f}\n")
                    f.write("\n")
                print(f"[INFO] Beam peak results appended to {output_file}")
            except Exception as e:
                print(f"[ERROR] Could not write to file: {e}")

        return results
    def __repr__(self):
        return (f"<TraceBeamResult method={self.method_beam} "
                f"window={self.timewindow}s fmin={self.fmin}Hz fmax={self.fmax}Hz>")


# example
"""

# Regional_events
phase_slowness_ranges = {
    "Pg":  (0.16, 0.22),  # Crustal P-wave (v ~ 5–6.2 km/s)
    "Pn":  (0.12, 0.16),  # Upper mantle P-wave (v ~ 6.3–8.3 km/s)
    "Sg":  (0.27, 0.33),  # Crustal S-wave (v ~ 3–3.7 km/s)
    "Sn":  (0.18, 0.27),  # Upper mantle S-wave (v ~ 3.7–5.5 km/s)
    "Lg":  (0.25, 0.35),  # Multi-reflected S/L surface phases
    "Rayleigh": (0.28, 0.45),  # Fundamental mode Rayleigh
    "Love":     (0.28, 0.42),  # Fundamental mode Love wave
}

phase_slowness_ranges_teleseismic = {
    "P":      (0.04, 0.07),   # Direct teleseismic P-waves (v ~ 13–20 km/s)
    "PP":     (0.06, 0.08),   # Surface-reflected P-wave
    "PKP":    (0.05, 0.07),   # Core-transiting P-wave
    "S":      (0.07, 0.12),   # Direct S-wave (slower than P)
    "SS":     (0.09, 0.13),   # Surface-reflected S-wave
    "SKS":    (0.07, 0.10),   # S→P→S phase through the core
    "ScS":    (0.09, 0.13),   # S reflected off the core-mantle boundary
    "PcP":    (0.05, 0.07),   # P reflected off the core-mantle boundary
    "Rayleigh": (0.20, 0.40), # Surface wave (frequency-dependent)
    "Love":     (0.20, 0.35), # Surface wave (frequency-dependent)
}

peaks = beam.get_relative_maxima_by_phase(
    power_min=0.7,
    baz_range=(90, 180),
    phase_dict=phase_ranges,
    peak_kwargs={"prominence": 0.1, "distance": 10}
)

for p in peaks:
    print(f"{p['time']} | Phase: {p['phase']} | Power: {p['relpower']:.2f} | "
          f"Azimuth: {p['azimuth']:.1f}° | Slowness: {p['slowness']:.3f}")

"""