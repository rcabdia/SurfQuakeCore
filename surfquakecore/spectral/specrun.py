#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
spectools
"""
import os
import pickle
import gzip
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
        self.method=method

    def plot_spectrum(self, axis_type="loglog"):
        import matplotlib.pyplot as plt


        fig, ax = plt.subplots()
        if axis_type == "loglog":
            ax.loglog(self.freq, self.spectrum)
        elif axis_type == "xlog":
            ax.semilogx(self.freq, self.spectrum)
        elif axis_type == "ylog":
            ax.semilogy(self.freq, self.spectrum)
        else:
            ax.plot(self.freq, self.spectrum)

        ax.set_ylim(self.spectrum.min() / 10.0, self.spectrum.max() * 100.0)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Spectrum for {self.trace.id}")
        plt.grid(True, which='both', linestyle='--', alpha=0.4)
        plt.tight_layout()
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
            lsup = int(self.trace.sampling_rate//2)

        step_percentage = (100 - overlap_percent) * 1E-2
        self.spectrogram = SpectrumTool.compute_spectrogram(self.trace, win, self.trace.delta, linf,
                                                            lsup, step_percentage)

    def plot_spectrogram(self, db=True):
        import matplotlib.pyplot as plt
        if not self.spectrogram:
            raise ValueError("Spectrogram not computed yet.")

        times, freqs, power = self.spectrogram
        Z = 10 * np.log10(power) if db else power

        fig, ax = plt.subplots(figsize=(10, 5))
        mesh = ax.pcolormesh(times, freqs, Z, shading='auto', cmap='viridis')
        ax.set_title(f"Spectrogram for {self.trace.id}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        fig.colorbar(mesh, ax=ax, label="Power [dB]" if db else "Amplitude")
        plt.tight_layout()
        plt.show()

    def summary(self):
        return {
            "id": self.trace.id,
            "starttime": str(self.stats.starttime),
            "endtime": str(self.stats.endtime),
            "sampling_rate": self.stats.sampling_rate,
            "npts": self.stats.npts,
            "computed_spectrogram": self.spectrogram is not None,
        }

    def to_pickle(self, filepath: str, compress: bool = True):
        """
        Serialize the analysis object to a pickle file.
        """
        data = {
            "version": "1.0",
            "trace_id": self.trace.id,
            "trace": self.trace,
            "spectrogram": self.spectrogram,
        }

        open_func = gzip.open if compress else open
        with open_func(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[INFO] Serialized to: {filepath}")

    @staticmethod
    def from_pickle(filepath: str, compress: bool = True):
        """
        Load a TraceSpectrogramResult from a pickle file.
        """
        open_func = gzip.open if compress else open
        with open_func(filepath, 'rb') as f:
            data = pickle.load(f)

        obj = TraceSpectrogramResult(trace=data["trace"], spectrogram=data.get("spectrogram"))
        return obj