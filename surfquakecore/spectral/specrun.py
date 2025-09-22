#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
spectools
"""

import os
import pickle
import gzip
import platform
import numpy as np
from surfquakecore.data_processing.spectral_tools import SpectrumTool

class TraceSpectrumResult:
    def __init__(self, trace, spectrum=None):

        self.freq = None
        self.method = None
        self.trace = trace  # original trace (or trimmed version)
        self.stats = trace.stats
        self.spectrum = spectrum  # tuple: (freqs, amplitudes)

    def compute_spectrum(self, method="multitaper"):

        self.spectrum, self.freq = SpectrumTool.compute_spectrum(self.trace.data, self.trace.stats.delta,
                                                                 mode=method)
        self.method = method

    def plot_spectrum(self, axis_type="loglog", save_path: str = None):

        import matplotlib.pyplot as plt
        import matplotlib as mplt

        if platform.system() == 'Darwin':
            mplt.use("MacOSX")
        else:
            mplt.use("Qt5Agg")

        fig, ax = plt.subplots()

        if axis_type == "loglog":
            ax.loglog(self.freq, self.spectrum, linewidth=0.75)
        elif axis_type == "xlog":
            ax.semilogx(self.freq, self.spectrum, linewidth=0.75)
        elif axis_type == "ylog":
            ax.semilogy(self.freq, self.spectrum, linewidth=0.75)
        else:
            print("No accepted axis_type: available loglog, xlog and ylog")

        ax.set_ylim(self.spectrum.min() / 10.0, self.spectrum.max() * 100.0)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Spectrum for {self.trace.id}")
        plt.grid(True, which='both', linestyle='--', alpha=0.4)
        plt.tight_layout()

        if save_path:
            self.fig_spec.savefig(save_path, dpi=300)
            plt.close(self.fig_spec)
        else:
            plt.show()

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

        t1 = self.trace.stats.starttime
        base_name = f"{self.trace.id}.D.{t1.year}.{t1.julday}"
        path_output = os.path.join(folder_path, base_name)

        counter = 1
        while os.path.exists(path_output + ".sp"):
            path_output = os.path.join(folder_path, f"{base_name}_{counter}")
            counter += 1

        path_output += ".sp"

        open_func = gzip.open if compress else open
        mode = 'wb'

        with open_func(path_output, mode) as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[INFO] {self.trace.id} - Writing spectrum to {path_output}")

    @staticmethod
    def from_pickle(filepath: str, compress: bool = True):
        """
        Load the full object from a pickle file.
        """
        open_func = gzip.open if compress else open
        mode = 'rb'

        with open_func(filepath, mode) as f:
            obj = pickle.load(f)

        if not isinstance(obj, TraceSpectrumResult):
            raise TypeError("Pickle file does not contain a TraceSpectrumResult object.")

        return obj


class TraceSpectrogramResult:
    def __init__(self, trace, spectrogram=None):
        self.trace = trace
        self.stats = trace.stats
        self.spectrogram = spectrogram  # tuple: (times, freqs, power_matrix)

    def compute_spectrogram(self, win=5.0, overlap_percent=50.0, linf=0, lsup=None):

        if lsup is None:
            lsup = int(self.trace.stats.sampling_rate // 2)

        step_percentage = (100 - overlap_percent) * 1E-2
        self.spectrogram, self.num_steps, self.time, self.freq = \
            SpectrumTool.compute_spectrogram(self.trace.data, int(win * self.trace.stats.sampling_rate),
                                             self.trace.stats.delta, linf, lsup, step_percentage)

    def plot_spectrogram(self, save_path: str = None, clip: float = None):

        import matplotlib.pyplot as plt
        import matplotlib as mplt
        from matplotlib import gridspec
        from matplotlib.ticker import ScalarFormatter

        if platform.system() == 'Darwin':
            mplt.use("MacOSX")
        else:
            mplt.use("Qt5Agg")


        if clip is not None:
            spectrogram = np.clip(10 * np.log10(self.spectrogram / np.max(self.spectrogram)), a_min=clip, a_max=0)
        else:
            spectrogram = 10 * np.log10(self.spectrogram / np.max(self.spectrogram))

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
        ax_waveform.plot(self.trace.times(), self.trace.data, linewidth=0.75)
        ax_waveform.set_title(f"Spectrogram for {self.trace.id}")
        ax_waveform.tick_params(labelbottom=False)

        # Annotate with date
        starttime = self.trace.stats.starttime
        date_str = starttime.strftime("%Y-%m-%d %H:%M:%S")
        textstr = f"JD {starttime.julday} / {starttime.year}\n{date_str}"
        ax_waveform.text(0.01, 0.95, textstr, transform=ax_waveform.transAxes, fontsize=8,
                         va='top', ha='left',
                         bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.5))

        # --- Plot spectrogram ---
        if clip is not None:
            clip = float(clip)
            spectrogram = np.clip(spectrogram, a_min=clip, a_max=0)

        pcm = ax_spec.pcolormesh(self.time, self.freq, spectrogram, shading='auto', cmap='rainbow'
                                 , vmin=np.min(spectrogram), vmax=0)

        ax_waveform.set_ylabel('Amplitude')
        ax_spec.set_ylabel('Frequency [Hz]')
        ax_spec.set_xlabel('Time [s]')

        # --- Add colorbar without shifting axes ---
        cbar = self.fig_spec.colorbar(pcm, cax=ax_cbar, orientation='vertical')
        cbar.set_label("Power [dB]")
        plt.tight_layout()

        if save_path:
            self.fig_spec.savefig(save_path, dpi=300)
            plt.close(self.fig_spec)
        else:
            plt.show()

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

        t1 = self.trace.stats.starttime
        base_name = f"{self.trace.id}.D.{t1.year}.{t1.julday}"
        path_output = os.path.join(folder_path, base_name)

        counter = 1
        while os.path.exists(path_output + ".spec"):
            path_output = os.path.join(folder_path, f"{base_name}_{counter}")
            counter += 1

        path_output += ".spec"

        open_func = gzip.open if compress else open
        mode = 'wb'

        with open_func(path_output, mode) as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[INFO] {self.trace.id} - Writing spectrum to {path_output}")

    @staticmethod
    def from_pickle(filepath: str, compress: bool = True):
        """
        Load the full object from a pickle file.
        """
        open_func = gzip.open if compress else open
        mode = 'rb'

        with open_func(filepath, mode) as f:
            obj = pickle.load(f)

        if not isinstance(obj, TraceSpectrogramResult):
            raise TypeError("Pickle file does not contain a TraceSpectrogramResult object.")

        return obj
