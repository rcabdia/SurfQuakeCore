#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cwtrun
"""

import os
import pickle
import gzip
import math
import numpy as np
from surfquakecore.data_processing.wavelet import ConvolveWaveletScipy

class TraceCWTResult:
    def __init__(self, trace, cwt_data=None):
        self.trace = trace
        self.stats = trace.stats
        self.cwt_data = cwt_data  # tuple: (times, freqs, scalogram, pred_mask, pred_mask_comp)

    def compute_cwt(self, wavelet_type="cm", param=6.0, fmin=None, fmax=None, nf=80):

        if wavelet_type == "cm":
            wavelet_type = "Complex Morlet"
        elif wavelet_type == "mh":
            wavelet_type = "Mexican Hat"
        elif wavelet_type == "pa":
            wavelet_type = "Paul"

        if fmax is None:
            fmax = self.trace.stats.sampling_rate // 2

        if fmin is None:
            fmin = 4//len(self.trace.data)

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

    def plot_cwt(self, save_path: str = None, clip: float = None):

        import matplotlib.pyplot as plt
        import matplotlib as mplt
        import platform
        import matplotlib.gridspec as gridspec
        from matplotlib.ticker import ScalarFormatter

        if platform.system() == 'Darwin':
            mplt.use("MacOSX")
        else:
            mplt.use("Qt5Agg")

        t, f, scalogram, pred, pred_comp = self.cwt_data
        tr = self.trace

        if clip is not None:
            clip = float(clip)
            scalogram = np.clip(scalogram, a_min=clip, a_max=0)

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
        # Annotate with date
        starttime = tr.stats.starttime
        date_str = starttime.strftime("%Y-%m-%d %H:%M:%S")
        textstr = f"JD {starttime.julday} / {starttime.year}\n{date_str}"
        ax_waveform.text(0.01, 0.95, textstr, transform=ax_waveform.transAxes, fontsize=8,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.5))

        # Plot scalogram
        x, y = np.meshgrid(t, f)
        pcm = ax_spec.pcolormesh(x, y, scalogram, shading='auto', cmap='rainbow',  vmin=np.min(scalogram), vmax=0)


        ax_spec.fill_between(pred, f, 0, color="black", edgecolor="red", alpha=0.3)
        ax_spec.fill_between(pred_comp, f, 0, color="black", edgecolor="red", alpha=0.3)
        ax_waveform.set_ylabel('Amplitude')
        ax_spec.set_ylim([np.min(f), np.max(f)])
        ax_spec.set_ylabel('Frequency [Hz]')
        ax_spec.set_xlabel('Time [s]')

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
        while os.path.exists(path_output + ".cwt"):
            path_output = os.path.join(folder_path, f"{base_name}_{counter}")
            counter += 1

        path_output += ".cwt"

        open_func = gzip.open if compress else open
        mode = 'wb'

        with open_func(path_output, mode) as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[INFO] {self.trace.id} - Writing scalogram to {path_output}")

    @staticmethod
    def from_pickle(filepath: str, compress: bool = True):
        """
        Load the full object from a pickle file.
        """
        open_func = gzip.open if compress else open
        mode = 'rb'

        with open_func(filepath, mode) as f:
            obj = pickle.load(f)

        if not isinstance(obj, TraceCWTResult):
            raise TypeError("Pickle file does not contain a TraceSpectrogramResult object.")

        return obj