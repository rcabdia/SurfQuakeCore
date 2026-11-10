#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ceprun: cepstrum and cepstrogram results for ObsPy traces (SurfQuakeCore).

Same pattern as specrun.py:
    TraceCepstrumResult     -> single cepstrum,   pickled as *.cep
    TraceCepstrogramResult  -> sliding cepstrum,  pickled as *.ceps
"""

import os
import sys
import gzip
import pickle
import platform

import numpy as np

from surfquakecore.spectral.cepstrum import cepstrum, cepstrogram, find_cepstral_peaks


# ---------------------------------------------------------------------- helpers
def _select_backend(save_path=None):
    """
    Choose a backend only if pyplot has not been imported yet:
    Agg when the figure is only saved (works on headless servers),
    otherwise the interactive one used in specrun.py.
    """
    import matplotlib as mplt
    if "matplotlib.pyplot" not in sys.modules:
        if save_path:
            mplt.use("Agg")
        else:
            mplt.use("MacOSX" if platform.system() == "Darwin" else "QtAgg")


def _write_pickle(obj, folder_path, ext, label, compress=True):
    if not folder_path:
        print("[ERROR] --folder_path must be specified")
        return None

    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            print(f"[INFO] Created folder: {folder_path}")
        except Exception as e:
            print(f"[ERROR] Failed to create folder '{folder_path}': {e}")
            return None

    t1 = obj.trace.stats.starttime
    base_name = f"{obj.trace.id}.D.{t1.year}.{t1.julday}"
    path_output = os.path.join(folder_path, base_name)

    counter = 1
    while os.path.exists(path_output + ext):
        path_output = os.path.join(folder_path, f"{base_name}_{counter}")
        counter += 1
    path_output += ext

    open_func = gzip.open if compress else open
    with open_func(path_output, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[INFO] {obj.trace.id} - Writing {label} to {path_output}")
    return path_output


def _read_pickle(filepath, cls, compress=True):
    open_func = gzip.open if compress else open
    with open_func(filepath, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, cls):
        raise TypeError(f"Pickle file does not contain a {cls.__name__} object.")
    return obj


def _save_or_show(fig, trace, save_path, type_spec, extension):
    import matplotlib.pyplot as plt

    if save_path:
        if os.path.isdir(save_path):
            st = trace.stats.starttime
            textstr = str(st.julday) + "_" + str(st.year) + "_" + st.strftime("%Y_%m_%d_%H_%M_%S")
            file_name = trace.id + "." + textstr + "." + type_spec + "." + extension
            save_path = os.path.join(save_path, file_name)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def _date_box(ax, starttime):
    date_str = starttime.strftime("%Y-%m-%d %H:%M:%S")
    textstr = f"JD {starttime.julday} / {starttime.year}\n{date_str}"
    ax.text(0.01, 0.95, textstr, transform=ax.transAxes, fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.5))


# ==============================================================================
# Single cepstrum
# ==============================================================================
class TraceCepstrumResult:
    """
    Cepstrum of one trace (or of a window of it).

    Stores the CepstrumResult from cepstrum.py, so quefrency, cepstrum, kind,
    nfft... are available as properties.
    """

    def __init__(self, trace):
        self.trace = trace
        self.stats = trace.stats
        self.result = None          # CepstrumResult
        self.window = None          # (t0, t1) seconds from trace start, or None

    # ------------------------------------------------------------------ compute
    def compute_cepstrum(self, kind="real", window=None, taper=0.1, detrend=True,
                         nfft=None, water_level=1e-10):
        """Arguments as cepstrum.cepstrum()."""
        self.result = cepstrum(self.trace, kind=kind, window=window, taper=taper, detrend=detrend,
                               nfft=nfft, water_level=water_level)
        self.window = window
        return self.result

    @property
    def quefrency(self):
        return self.result.quefrency

    @property
    def cepstrum(self):
        return self.result.cepstrum

    @property
    def kind(self):
        return self.result.kind

    def find_peaks(self, qmin=None, qmax=None, n_peaks=5, min_snr=4.0):
        """Repetition periods, see cepstrum.find_cepstral_peaks()."""
        return find_cepstral_peaks(self.result, qmin=qmin, qmax=qmax, n_peaks=n_peaks, min_snr=min_snr)

    # --------------------------------------------------------------------- plot
    def plot_cepstrum(self, qmin=None, qmax=None, show_peaks=True, n_peaks=5, min_snr=4.0,
                      save_path: str = None, type_spec: str = "cep", extension: str = "png"):
        """
        Waveform (analysis window shaded) + cepstrum, with the detected
        repetition periods marked.

        qmin, qmax : quefrency range shown (s). Default qmin = 5 samples, which
                     hides the low-quefrency spectral envelope that otherwise
                     dominates the y scale.
        """
        _select_backend(save_path)
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter

        if self.result is None:
            raise ValueError("Cepstrum not computed: call compute_cepstrum() first.")

        res = self.result
        q, c = res.quefrency, res.cepstrum
        if res.kind == "complex":
            keep = q >= 0
            q, c = q[keep], c[keep]

        lo = 5 * res.delta if qmin is None else float(qmin)
        hi = q[-1] if qmax is None else float(qmax)
        sel = (q >= lo) & (q <= hi)
        if not np.any(sel):
            raise ValueError(f"No quefrencies in [{lo:g}, {hi:g}] s.")

        fig, (ax_w, ax_c) = plt.subplots(2, 1, figsize=(10, 5.5), gridspec_kw={"height_ratios": [1, 1.4]})

        # ---- waveform
        ax_w.plot(self.trace.times(), self.trace.data, linewidth=0.75)
        if self.window is not None:
            t0, t1 = self.window
            st = self.trace.stats.starttime
            t0 = t0 - st if not isinstance(t0, (int, float)) else t0
            t1 = t1 - st if not isinstance(t1, (int, float)) else t1
            ax_w.axvspan(t0, t1, color="orange", alpha=0.2, label="analysis window")
            ax_w.legend(loc="upper right", fontsize=8)
        ax_w.set_xlim(self.trace.times()[0], self.trace.times()[-1])
        ax_w.set_xlabel("Time [s]")
        ax_w.set_ylabel("Amplitude")
        ax_w.set_title(f"{res.kind.capitalize()} cepstrum for {self.trace.id}")
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax_w.yaxis.set_major_formatter(fmt)
        _date_box(ax_w, self.trace.stats.starttime)

        # ---- cepstrum
        ax_c.plot(q[sel], c[sel], linewidth=0.8)
        ax_c.set_xlim(q[sel][0], q[sel][-1])
        ax_c.set_xlabel("Quefrency [s]")
        ax_c.set_ylabel(f"{res.kind} cepstrum")
        ax_c.grid(True, linestyle="--", alpha=0.4)

        if show_peaks:
            peaks = find_cepstral_peaks(res, qmin=lo, qmax=hi, n_peaks=n_peaks, min_snr=min_snr)
            for p in peaks:
                ax_c.plot(p["quefrency"], p["value"], "rv", markersize=6)
                ax_c.annotate(f"{p['quefrency']:.3g} s\n{p['frequency']:.3g} Hz",
                              (p["quefrency"], p["value"]), textcoords="offset points",
                              xytext=(4, 6), fontsize=7, color="r")

        fig.tight_layout()
        _save_or_show(fig, self.trace, save_path, type_spec, extension)

    # ------------------------------------------------------------------- pickle
    def to_pickle(self, folder_path: str, compress: bool = True):
        """Serialize the full object to a pickle file (.cep)."""
        return _write_pickle(self, folder_path, ".cep", "cepstrum", compress)

    @staticmethod
    def from_pickle(filepath: str, compress: bool = True):
        return _read_pickle(filepath, TraceCepstrumResult, compress)


# ==============================================================================
# Cepstrogram
# ==============================================================================
class TraceCepstrogramResult:
    """
    Sliding-window cepstrum of one trace. Same windowing and time convention
    (window centre) as TraceSpectrogramResult.
    """

    def __init__(self, trace, cepstrogram=None):
        self.trace = trace
        self.stats = trace.stats
        self.cepstrogram = cepstrogram   # 2D array (quefrency x time)
        self.time = None
        self.quefrency = None
        self.num_steps = None
        self.kind = None
        self.win = None                  # seconds

    # ------------------------------------------------------------------ compute
    def compute_cepstrogram(self, win=5.0, overlap_percent=50.0, kind="real", taper=0.1, nfft=None,
                            water_level=1e-10, qmin=0.0, qmax=None):
        """
        win   : window length in SECONDS (as compute_spectrogram).
        qmax  : default = win (longest resolvable repetition period).
        """
        step_percentage = (100 - overlap_percent) * 1E-2
        win_samples = round(win * self.trace.stats.sampling_rate)

        self.cepstrogram, self.num_steps, self.time, self.quefrency = cepstrogram(
            self.trace.data, self.trace.stats.delta, win_samples, step_percentage=step_percentage,
            kind=kind, taper=taper, nfft=nfft, water_level=water_level, qmin=qmin, qmax=qmax)

        self.kind = kind
        self.win = win

    # ------------------------------------------------------- derived quantities
    def _crop(self, qmin=None, qmax=None):
        q = np.asarray(self.quefrency)
        lo = max(5 * self.trace.stats.delta, q[0]) if qmin is None else float(qmin)
        hi = q[-1] if qmax is None else float(qmax)
        sel = (q >= lo) & (q <= hi)
        if not np.any(sel):
            raise ValueError(f"No quefrencies in [{lo:g}, {hi:g}] s.")
        return q[sel], self.cepstrogram[sel, :]

    def track_peak(self, qmin=None, qmax=None):
        """
        Dominant repetition period in every window.

        Returns
        -------
        t      : time (s, window centre)
        q_peak : quefrency of the maximum in [qmin, qmax] (NaN for empty windows)
        value  : cepstrum value at that quefrency
        """
        q, c = self._crop(qmin, qmax)
        q_peak = np.full(c.shape[1], np.nan)
        value = np.full(c.shape[1], np.nan)
        valid = np.all(np.isfinite(c), axis=0)
        if np.any(valid):
            i = np.argmax(c[:, valid], axis=0)
            q_peak[valid] = q[i]
            value[valid] = c[i, np.where(valid)[0]]
        return np.asarray(self.time), q_peak, value

    # --------------------------------------------------------------------- plot
    def plot_cepstrogram(self, save_path: str = None, qmin: float = None, qmax: float = None,
                         clip: float = None, vmax: float = None, plot_date: bool = False,
                         cmap: str = "rainbow", smooth: bool = False, track: bool = False,
                         extension="png", type_spec: str = "ceps"):
        """
        qmin, qmax : quefrency range shown (s). Default qmin = 5 samples (hides the
                     low-quefrency spectral envelope).
        clip, vmax : colour limits. "power" -> dB relative to the maximum of the
                     displayed range (default vmax 0 dB, clip = 1st percentile).
                     "real"  -> cepstrum units (default 5th / 99.5th percentiles).
        track      : overlay the dominant quefrency of every window.
        """
        _select_backend(save_path)
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib import gridspec
        from matplotlib.ticker import ScalarFormatter

        if self.cepstrogram is None:
            raise ValueError("Cepstrogram not computed: call compute_cepstrogram() first.")

        q, C = self._crop(qmin, qmax)
        time = np.asarray(self.time)

        finite = C[np.isfinite(C)]
        if finite.size == 0:
            raise ValueError("Cepstrogram contains no finite values.")

        # ------------------------------------------------ values to display
        if self.kind == "power":
            ref = np.nanmax(C)
            if ref <= 0:
                raise ValueError("Power cepstrogram maximum must be greater than zero.")
            with np.errstate(divide="ignore", invalid="ignore"):
                Z = 10.0 * np.log10(C / ref)
            finite = Z[np.isfinite(Z)]
            v_hi = 0.0 if vmax is None else float(vmax)
            v_lo = float(np.percentile(finite, 1)) if clip is None else float(clip)
            cbar_label = "Power cepstrum [dB]"
        else:
            Z = C
            v_hi = float(np.percentile(finite, 99.5)) if vmax is None else float(vmax)
            v_lo = float(np.percentile(finite, 5)) if clip is None else float(clip)
            cbar_label = "Real cepstrum"

        if v_lo >= v_hi:
            raise ValueError(f"clip ({v_lo:g}) must be lower than vmax ({v_hi:g}).")

        Z = np.ma.masked_invalid(Z)
        levels = np.linspace(v_lo, v_hi, 100)

        # ------------------------------------------------ x coordinates
        starttime = self.trace.stats.starttime
        if plot_date:
            start_num = mdates.date2num(starttime.datetime)
            waveform_x = start_num + self.trace.times() / 86400.0
            cep_x = start_num + time / 86400.0
        else:
            waveform_x = self.trace.times()
            cep_x = time

        # ------------------------------------------------ figure
        fig = plt.figure(figsize=(10, 5))
        self.fig_ceps = fig
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.03], height_ratios=[1, 1], hspace=0.02, wspace=0.02)
        ax_w = fig.add_subplot(gs[0, 0])
        ax_c = fig.add_subplot(gs[1, 0], sharex=ax_w)
        ax_cbar = fig.add_subplot(gs[1, 1])

        ax_w.plot(waveform_x, self.trace.data, linewidth=0.75)
        ax_w.set_title(f"{self.kind.capitalize()} cepstrogram for {self.trace.id}  (window {self.win:g} s)")
        ax_w.set_ylabel("Amplitude")
        ax_w.tick_params(labelbottom=False)
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax_w.yaxis.set_major_formatter(fmt)

        if smooth:
            pcm = ax_c.contourf(cep_x, q, Z, levels=levels, cmap=cmap, vmin=v_lo, vmax=v_hi, extend="both")
        else:
            pcm = ax_c.pcolormesh(cep_x, q, Z, shading="auto", cmap=cmap, vmin=v_lo, vmax=v_hi)

        if track:
            _, q_peak, _ = self.track_peak(qmin=q[0], qmax=q[-1])
            ax_c.plot(cep_x, q_peak, "w.", markersize=3, label="dominant quefrency")
            ax_c.legend(loc="upper right", fontsize=8, framealpha=0.6)

        ax_c.set_ylim(q[0], q[-1])
        ax_c.set_ylabel("Quefrency [s]")
        ax_w.set_xlim(cep_x[0], cep_x[-1])

        if plot_date:
            locator = mdates.AutoDateLocator()
            ax_c.xaxis.set_major_locator(locator)
            ax_c.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax_c.set_xlabel("Date / Time [UTC]")
        else:
            ax_c.set_xlabel("Time [s]")

        cbar = fig.colorbar(pcm, cax=ax_cbar, orientation="vertical")
        cbar.set_label(cbar_label)

        _date_box(ax_w, starttime)
        plt.tight_layout()
        _save_or_show(fig, self.trace, save_path, type_spec, extension)

    # ------------------------------------------------------------------- pickle
    def to_pickle(self, folder_path: str, compress: bool = True):
        """Serialize the full object to a pickle file (.ceps)."""
        return _write_pickle(self, folder_path, ".ceps", "cepstrogram", compress)

    @staticmethod
    def from_pickle(filepath: str, compress: bool = True):
        return _read_pickle(filepath, TraceCepstrogramResult, compress)
