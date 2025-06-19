#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
spectools
"""
import pickle
import gzip

import numpy as np

from surfquakecore.data_processing.spectral_tools import SpectrumTool

class TraceSpectrumResult:
    def __init__(self, trace, spectrum=None):
        self.trace = trace  # original trace (or trimmed version)
        self.stats = trace.stats
        self.spectrum = spectrum  # tuple: (freqs, amplitudes)

    def compute_spectrum(self, method="multitaper"):

        self.spectrum = SpectrumTool.compute_spectrum(self.trace, self.trace.stats.delta, mode=method)

    def plot_spectrum(self, axis_type="loglog"):
        import matplotlib.pyplot as plt

        if not self.spectrum:
            raise ValueError("Spectrum not computed yet.")

        freqs, amps = self.spectrum

        fig, ax = plt.subplots()
        if axis_type == "loglog":
            ax.loglog(freqs, amps)
        elif axis_type == "xlog":
            ax.semilogx(freqs, amps)
        elif axis_type == "ylog":
            ax.semilogy(freqs, amps)
        else:
            ax.plot(freqs, amps)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Spectrum for {self.trace.id}")
        plt.grid(True, which='both', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()

    def summary(self):
        return {
            "id": self.trace.id,
            "starttime": str(self.stats.starttime),
            "endtime": str(self.stats.endtime),
            "sampling_rate": self.stats.sampling_rate,
            "npts": self.stats.npts,
            "computed_spectrum": self.spectrum is not None,
        }



    def to_pickle(self, filepath: str, compress: bool = True):
        """
        Serialize the analysis object to a pickle file.

        Parameters
        ----------
        filepath : str
            Path to output file (e.g., 'trace_result.pkl' or '.pkl.gz')
        compress : bool
            Whether to gzip the pickle file.
        """
        data = {
            "version": "1.0",
            "trace_id": self.trace.id,
            "trace": self.trace,
            "spectrum": self.spectrum,
        }

        open_func = gzip.open if compress else open
        mode = 'wb'

        with open_func(filepath, mode) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[INFO] Serialized to: {filepath}")

    @staticmethod
    def from_pickle(filepath: str, compress: bool = True):
        """
        Load a TraceAnalysisResult from a pickle file.

        Parameters
        ----------
        filepath : str
            Path to .pkl or .pkl.gz file.
        compress : bool
            Whether the file is compressed.

        Returns
        -------
        TraceAnalysisResult
        """

        open_func = gzip.open if compress else open
        mode = 'rb'

        with open_func(filepath, mode) as f:
            data = pickle.load(f)

        obj = TraceSpectrumResult(trace=data["trace"], spectrum=data.get("spectrum"))
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