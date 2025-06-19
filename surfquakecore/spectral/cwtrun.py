#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cwtrun
"""

import pickle
import gzip
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
import time
from surfquakecore.data_processing.wavelet import ConvolveWaveletScipy

class TraceCWTResult:
    def __init__(self, trace, cwt_data=None):
        self.trace = trace
        self.stats = trace.stats
        self.cwt_data = cwt_data  # tuple: (times, freqs, scalogram, pred_mask, pred_mask_comp)

    def compute_cwt(self, wavelet_type="cm", param=6.0, fmin=0.5, fmax=None, nf=80):

        if wavelet_type == "cm":
            wavelet_type = "Complex Morlet"
        elif wavelet_type == "mh":
            wavelet_type = "Mexican Hat"
        elif wavelet_type == "pa":
            wavelet_type = "Paul"

        if fmax is None:
            fmax = self.trace.stats.sampling_rate / 2

        tr = self.trace.copy()

        # Trim window
        stime = getattr(self, "utc_start", tr.stats.starttime)
        etime = getattr(self, "utc_end", tr.stats.endtime)
        tr.trim(starttime=stime, endtime=etime)

        cw = ConvolveWaveletScipy(tr)
        tt = int(tr.stats.sampling_rate / fmin)

        cw.setup_wavelet(wmin=param, wmax=param, tt=tt, fmin=fmin, fmax=fmax,
                         nf=nf, use_wavelet=wavelet_type, m=param, decimate=False)

        scalogram2 = cw.scalogram_in_dbs()
        t = np.linspace(0, tr.stats.delta * scalogram2.shape[1], scalogram2.shape[1])
        f = np.logspace(np.log10(fmin), np.log10(fmax), scalogram2.shape[0])

        # Prediction masks
        c_f = param / (2 * math.pi)
        ff = np.linspace(fmin, fmax, scalogram2.shape[0])
        pred = (math.sqrt(2) * c_f / ff) - (math.sqrt(2) * c_f / fmax)
        pred_comp = t[-1] - pred

        self.cwt_data = (t, f, scalogram2, pred, pred_comp)

    def plot_cwt(self):
        if not self.cwt_data:
            raise ValueError("CWT data not computed yet.")

        t, f, scalogram2, pred, pred_comp = self.cwt_data
        tr = self.trace

        self.fig_spec = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.03], height_ratios=[1, 1],
                               hspace=0.02, wspace=0.02)

        ax_waveform = self.fig_spec.add_subplot(gs[0, 0])
        ax_spec = self.fig_spec.add_subplot(gs[1, 0], sharex=ax_waveform)
        ax_cbar = self.fig_spec.add_subplot(gs[1, 1])

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        ax_waveform.yaxis.set_major_formatter(formatter)

        # Plot waveform
        ax_waveform.plot(tr.times(), tr.data, linewidth=0.75)
        ax_waveform.set_title(f"CWT Scalogram for {tr.id}")
        ax_waveform.tick_params(labelbottom=False)

        # Plot scalogram
        x, y = np.meshgrid(t, f)
        pcm = ax_spec.pcolormesh(x, y, scalogram2, shading='auto', cmap='rainbow')
        ax_spec.fill_between(pred, ff, 0, color="black", edgecolor="red", alpha=0.3)
        ax_spec.fill_between(pred_comp, ff, 0, color="black", edgecolor="red", alpha=0.3)
        ax_waveform.set_ylabel('Amplitude')
        ax_spec.set_ylim([np.min(f), np.max(f)])
        ax_spec.set_ylabel('Frequency [Hz]')
        ax_spec.set_xlabel('Time [s]')

        cbar = self.fig_spec.colorbar(pcm, cax=ax_cbar, orientation='vertical')
        cbar.set_label("Power [dB]")

        plt.tight_layout()
        plt.show(block=False)

        while plt.fignum_exists(self.fig_spec.number):
            plt.pause(0.2)
            time.sleep(0.1)

    def summary(self):
        return {
            "id": self.trace.id,
            "starttime": str(self.stats.starttime),
            "endtime": str(self.stats.endtime),
            "sampling_rate": self.stats.sampling_rate,
            "npts": self.stats.npts,
            "computed_cwt": self.cwt_data is not None,
        }

    def to_pickle(self, filepath: str, compress: bool = True):
        data = {
            "version": "1.0",
            "trace_id": self.trace.id,
            "trace": self.trace,
            "cwt_data": self.cwt_data,
        }

        open_func = gzip.open if compress else open
        with open_func(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[INFO] Serialized to: {filepath}")

    @staticmethod
    def from_pickle(filepath: str, compress: bool = True):
        open_func = gzip.open if compress else open
        with open_func(filepath, 'rb') as f:
            data = pickle.load(f)

        return TraceCWTResult(trace=data["trace"], cwt_data=data.get("cwt_data"))