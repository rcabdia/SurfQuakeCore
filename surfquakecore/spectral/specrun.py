#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
spectools
"""

import os
import pickle
import gzip
import platform
import sys

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

    def plot_spectrum(self, axis_type="loglog", save_path: str = None, type_spec: str = "sp", extension: str = "png"):

        import matplotlib.pyplot as plt
        import matplotlib as mplt

        if platform.system() == 'Darwin':
            mplt.use("MacOSX")
        else:
            mplt.use("QtAgg")

        fig, ax = plt.subplots()

        if axis_type == "loglog":
            ax.loglog(self.freq[1:], self.spectrum[1:], linewidth=0.75)
        elif axis_type == "xlog":
            ax.semilogx(self.freq[1:], self.spectrum[1:], linewidth=0.75)
        elif axis_type == "ylog":
            ax.semilogy(self.freq[1:], self.spectrum[1:], linewidth=0.75)
        elif axis_type == "linlin":
            ax.plot(self.freq[1:], self.spectrum[1:], linewidth=0.75)
        else:
            print("No accepted axis_type: available loglog, xlog and ylog")

        if axis_type != "linlin":
            ax.set_ylim(self.spectrum.min() / 10.0, self.spectrum.max() * 100.0)
        
        ax.set_xlim(self.freq[1], self.freq[-1])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Spectrum for {self.trace.id}")
        plt.grid(True, which='both', linestyle='--', alpha=0.4)
        plt.tight_layout()

        if save_path:
            starttime = self.trace.stats.starttime
            textstr = str(starttime.julday) + "_" + str(starttime.year) + "_" + starttime.strftime("%Y_%m_%d_%H_%M_%S")
            file_name = self.trace.id + "." + textstr + "." + type_spec + "." + extension
            save_path = os.path.join(save_path, file_name)
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
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

    def compute_spectrogram(self, win=5.0, overlap_percent=50.0, linf=0, lsup=None, method="multitaper", nw=None):

        if lsup is None:
            lsup = int(self.trace.stats.sampling_rate // 2)

        step_percentage = (100 - overlap_percent) * 1E-2
        self.spectrogram, self.num_steps, self.time, self.freq = \
            SpectrumTool.compute_spectrogram(self.trace.data, round(win * self.trace.stats.sampling_rate),
                                             self.trace.stats.delta, linf, lsup, step_percentage,
                                             method, nw)

    def plot_spectrogram(self, save_path: str = None, clip: float = None, plot_date: bool = False, split=None,
            vmax_db: float = None, cmap: str = "rainbow",
            smooth: bool = False, extension="png", type_spec: str = ""):

        import platform
        import numpy as np
        import matplotlib as mplt

        # Select backend before importing pyplot
        if platform.system() == "Darwin":
            mplt.use("MacOSX")
        else:
            mplt.use("QtAgg")

        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib import gridspec
        from matplotlib.ticker import ScalarFormatter, FixedLocator, FuncFormatter

        # ======================================================
        # Prepare spectrogram in relative dB
        # ======================================================
        freq = np.asarray(self.freq)
        time = np.asarray(self.time)

        max_power = np.nanmax(self.spectrogram)

        if not np.isfinite(max_power) or max_power <= 0:
            raise ValueError(
                "Spectrogram maximum must be greater than zero.")

        with np.errstate(divide="ignore", invalid="ignore"):

            spectrogram = 10.0 * np.log10(self.spectrogram / max_power)


        if vmax_db is None:

            #vmax_db = np.percentile(spectrogram, q=99.5, axis=None)
            vmax_db = 0.0

        else:
            vmax_db = float(vmax_db)

        # ------------------------------------------------------
        # Lower display limit
        #
        # If clip is given -> use it directly.
        # Otherwise -> estimate a robust lower limit
        # from the 1st percentile.
        # ------------------------------------------------------
        finite_values = spectrogram[np.isfinite(spectrogram)]

        if finite_values.size == 0:
            raise ValueError(
                "Spectrogram contains no finite values.")

        if clip is not None:

            vmin_db = float(clip)

        else:

            vmin_db = float(np.percentile(finite_values, q=1))

        # ------------------------------------------------------
        # Validate limits
        # ------------------------------------------------------
        if vmin_db >= vmax_db:
            raise ValueError(f"vmin_db ({vmin_db:g} dB) must be lower than "
                f"vmax_db ({vmax_db:g} dB).")

        # Replace -inf produced by log10(0)
        spectrogram = np.where(np.isfinite(spectrogram), spectrogram, vmin_db)

        levels = np.linspace(vmin_db, vmax_db, 100)
        # ======================================================
        # Interpret split
        #
        # None / False -> normal representation
        # True         -> split at 1 Hz
        # float        -> user-selected cutoff
        # ======================================================
        if isinstance(split, bool):

            split_freq = 1.0 if split else None

        elif split is None:

            split_freq = None

        else:

            split_freq = float(split)

        if split_freq is not None and split_freq <= 0:
            raise ValueError("split frequency must be greater than 0 Hz.")

        starttime = self.trace.stats.starttime

        # ======================================================
        # X coordinates
        # ======================================================
        if plot_date:

            start_num = mdates.date2num(starttime.datetime)

            waveform_x = (start_num + self.trace.times() / 86400.0)

            spec_x = (start_num + time / 86400.0)

        else:

            waveform_x = self.trace.times()
            spec_x = time

        x_lim = (spec_x[0], spec_x[-1])

        # ======================================================
        # NORMAL MODE
        # ======================================================
        if split_freq is None:

            self.fig_spec = plt.figure(figsize=(10, 5))

            gs = gridspec.GridSpec(2,2, width_ratios=[1, 0.03], height_ratios=[1, 1], hspace=0.02,
                                   wspace=0.02)

            ax_waveform = self.fig_spec.add_subplot(gs[0, 0])

            ax_spec = self.fig_spec.add_subplot(gs[1, 0], sharex=ax_waveform)

            ax_cbar = self.fig_spec.add_subplot(gs[1, 1])

            # --------------------------------------------------
            # Original waveform
            # --------------------------------------------------
            ax_waveform.plot(waveform_x, self.trace.data, linewidth=0.75)
            ax_waveform.set_title(f"Spectrogram for {self.trace.id}")
            ax_waveform.set_ylabel("Amplitude")
            ax_waveform.tick_params(labelbottom=False)
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))
            ax_waveform.yaxis.set_major_formatter(formatter)

            # --------------------------------------------------
            # Full spectrogram
            # --------------------------------------------------
            #pcm = ax_spec.pcolormesh(spec_x, freq, spectrogram, shading="auto",
            #    cmap=cmap, vmin=vmin_db, vmax=vmax_db)


            if smooth:
                # Levels used only by contourf

                pcm = ax_spec.contourf(spec_x, freq, spectrogram, levels=levels,
                    cmap=cmap, vmin=vmin_db, vmax=vmax_db, extend="both")

            else:

                pcm = ax_spec.pcolormesh(spec_x, freq, spectrogram, shading="auto", cmap=cmap,
                    vmin=vmin_db, vmax=vmax_db)

            ax_spec.set_ylabel("Frequency [Hz]")

            # --------------------------------------------------
            # X axis
            # --------------------------------------------------

            ax_waveform.set_xlim(*x_lim)

            if plot_date:

                locator = mdates.AutoDateLocator()
                ax_spec.xaxis.set_major_locator(locator)
                ax_spec.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
                ax_spec.set_xlabel("Date / Time [UTC]")

            else:

                ax_spec.set_xlabel("Time [s]")

            # --------------------------------------------------
            # Colorbar
            # --------------------------------------------------
            cbar = self.fig_spec.colorbar(pcm, cax=ax_cbar, orientation="vertical")

            cbar.set_label("Power [dB]")

            annotation_axis = ax_waveform

        # ======================================================
        # SPLIT MODE
        # ======================================================
        else:

            nyquist = (self.trace.stats.sampling_rate / 2.0)

            if split_freq >= nyquist:
                raise ValueError(
                    f"split frequency ({split_freq:g} Hz) " f"must be below Nyquist " f"({nyquist:g} Hz).")

            if split_freq >= np.max(freq):
                raise ValueError(
                    f"split frequency ({split_freq:g} Hz) " f"must be below the maximum plotted "
                    f"frequency ({np.max(freq):g} Hz).")

            # --------------------------------------------------
            # Divide the EXISTING spectrogram.
            #
            # 0 Hz is excluded from the lower part because
            # period = 1/f cannot be defined at f = 0.
            # --------------------------------------------------
            low_mask = ((freq > 0) & (freq < split_freq))

            high_mask = (freq >= split_freq)

            if not np.any(low_mask):
                raise ValueError(
                    f"No positive frequency bins below "
                    f"{split_freq:g} Hz."
                )

            if not np.any(high_mask):
                raise ValueError(
                    f"No frequency bins at or above "
                    f"{split_freq:g} Hz."
                )

            # --------------------------------------------------
            # High-frequency representation remains in Hz
            # --------------------------------------------------
            freq_high = freq[high_mask]
            spec_high = spectrogram[high_mask, :]

            # --------------------------------------------------
            # Low-frequency representation becomes period
            # --------------------------------------------------
            freq_low = freq[low_mask]
            spec_low = spectrogram[low_mask, :]

            period_low = 1.0 / freq_low

            # pcolormesh behaves best with monotonically
            # increasing coordinates, so sort the periods.
            order = np.argsort(period_low)

            period_low = period_low[order]

            spec_low = spec_low[order, :]

            # ==================================================
            # Figure layout
            #
            # waveform     -> height 1
            # high freq    -> height 1
            # period panel -> height 2
            # ==================================================
            self.fig_spec = plt.figure(figsize=(10, 8))

            gs = gridspec.GridSpec(3, 2, width_ratios=[1, 0.03],
                height_ratios=[1, 1, 2], hspace=0.04, wspace=0.02)

            # Low-frequency waveform uses LEFT amplitude axis
            ax_waveform_low = self.fig_spec.add_subplot(gs[0, 0])

            # High-frequency waveform uses RIGHT amplitude axis
            ax_waveform_high = (ax_waveform_low.twinx())

            ax_high = self.fig_spec.add_subplot(gs[1, 0], sharex=ax_waveform_low)

            ax_low = self.fig_spec.add_subplot(gs[2, 0], sharex=ax_waveform_low)

            # One colorbar for both TF panels
            ax_cbar = self.fig_spec.add_subplot(gs[1:, 1])

            # ==================================================
            # FILTERED TIME SERIES
            # ==================================================
            tr_low = self.trace.copy()
            tr_high = self.trace.copy()

            tr_low.detrend(type="linear")
            tr_low.detrend(type="simple")

            tr_high.detrend(type="linear")
            tr_high.detrend(type="simple")

            tr_low.taper(max_percentage=0.05)
            tr_high.taper(max_percentage=0.05)

            # Zero-phase filtering prevents time shifts
            tr_low.filter("lowpass", freq=split_freq, corners=4, zerophase=True)
            tr_high.filter("highpass", freq=split_freq, corners=4, zerophase=True)
            tr_low.detrend(type="simple")
            tr_high.detrend(type="simple")

            # Low-frequency signal -> LEFT y axis
            line_low, = ax_waveform_low.plot(waveform_x, tr_low.data, linewidth=0.8, alpha=0.75,
                                             label=f"Low-pass < {split_freq:g} Hz")

            ax_waveform_low.set_ylabel("Low-freq amplitude")

            # High-frequency signal -> RIGHT y axis
            line_high, = ax_waveform_high.plot(waveform_x, tr_high.data, linewidth=0.65, alpha=0.55, color="black",
                                               label=f"High-pass > {split_freq:g} Hz")

            ax_waveform_high.set_ylabel("High-freq amplitude")

            # Scientific notation independently on both axes
            formatter_left = ScalarFormatter(useMathText=True)

            formatter_left.set_powerlimits((0, 0))

            ax_waveform_low.yaxis.set_major_formatter(formatter_left)

            formatter_right = ScalarFormatter(useMathText=True)

            formatter_right.set_powerlimits((0, 0))

            ax_waveform_high.yaxis.set_major_formatter(formatter_right)

            ax_waveform_low.set_title(f"Spectrogram for {self.trace.id}")

            ax_waveform_low.tick_params(labelbottom=False)

            # Combined legend for both waveform axes
            ax_waveform_low.legend([line_low, line_high], [
                    line_low.get_label(), line_high.get_label()], loc="upper right", fontsize=8, framealpha=0.7)

            # ==================================================
            # HIGH-FREQUENCY SPECTROGRAM
            # ==================================================
            if smooth:

                pcm = ax_high.contourf(spec_x, freq_high, spec_high, levels=levels, cmap=cmap, vmin=vmin_db,
                    vmax=vmax_db, extend="both")

            else:

                pcm = ax_high.pcolormesh(spec_x, freq_high, spec_high, shading="auto", cmap=cmap,
                    vmin=vmin_db, vmax=vmax_db)

            ax_high.set_ylabel("Frequency [Hz]")

            ax_high.set_ylim(split_freq, np.max(freq_high))

            ax_high.tick_params(labelbottom=False)

            # ==================================================
            # LOW-FREQUENCY SPECTROGRAM AS PERIOD
            # ==================================================
            if smooth:

                ax_low.contourf( spec_x, period_low, spec_low, levels=levels, cmap=cmap,
                    vmin=vmin_db, vmax=vmax_db, extend="both")

            else:

                ax_low.pcolormesh(spec_x, period_low, spec_low, shading="auto", cmap=cmap,
                                  vmin=vmin_db, vmax=vmax_db)

            ax_low.set_yscale("log")

            ax_low.set_ylim(np.min(period_low), np.max(period_low))

            # Short periods/high frequencies at the top.
            # Long periods/low frequencies at the bottom.
            ax_low.invert_yaxis()

            ax_low.set_ylabel("Period [s]")

            # ==================================================
            # Useful period annotations:
            #
            # 0.1, 0.2, 0.5,
            # 1, 2, 5,
            # 10, 20, 50, ...
            # ==================================================
            period_min = np.min(period_low)
            period_max = np.max(period_low)

            exponent_min = int(np.floor(np.log10(period_min)))

            exponent_max = int(
                np.ceil(np.log10(period_max)))

            period_ticks = []

            for exponent in range(
                    exponent_min,
                    exponent_max + 1):

                for multiplier in (1, 2, 5):

                    value = (multiplier * 10.0 ** exponent)

                    if (period_min <= value <= period_max):
                        period_ticks.append(value)

            if period_ticks:
                ax_low.yaxis.set_major_locator(
                    FixedLocator(period_ticks))

                ax_low.yaxis.set_major_formatter(
                    FuncFormatter(lambda value, _: f"{value:g}"))

            ax_waveform_low.set_xlim(*x_lim)

            # ==================================================
            # X axis
            # ==================================================
            if plot_date:

                locator = mdates.AutoDateLocator()

                ax_low.xaxis.set_major_locator(locator)

                ax_low.xaxis.set_major_formatter(
                    mdates.ConciseDateFormatter(locator))

                ax_low.set_xlabel(
                    "Date / Time [UTC]")

            else:

                ax_low.set_xlabel("Time [s]")

            # --------------------------------------------------
            # Shared colorbar
            # --------------------------------------------------
            cbar = self.fig_spec.colorbar(
                pcm, cax=ax_cbar, orientation="vertical")

            cbar.set_label("Power [dB]")

            annotation_axis = ax_waveform_low

        # ======================================================
        # Trace start-time annotation
        # ======================================================
        date_str = starttime.strftime("%Y-%m-%d %H:%M:%S")

        textstr = f"JD {starttime.julday} / "f"{starttime.year}\n" f"{date_str}"

        annotation_axis.text(0.01, 0.95, textstr, transform=annotation_axis.transAxes,
            fontsize=8, va="top", ha="left", bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                                                       ec="gray", alpha=0.5))

        plt.tight_layout()

        # ======================================================
        # Save or display
        # ======================================================
        if save_path:

            if os.path.isdir(save_path):
                textstr = str(starttime.julday)+"_"+str(starttime.year)+"_"+starttime.strftime("%Y_%m_%d_%H_%M_%S")
                file_name = self.trace.id+"."+textstr+"."+type_spec+"."+extension
                save_path = os.path.join(save_path, file_name)

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

        print(f"[INFO] {self.trace.id} - Writing spectrogram to {path_output}")

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


class TraceCrossSpectralResult:
    """
    Cross-spectral analysis between a reference trace (input) and a target
    trace (output): coherence and transfer function.

    What is stored are the primitives (freq, gxx, gyy, gxy, n_avg). Coherence,
    transfer function, amplitude and phase are derived properties, so nothing
    has to be recomputed to plot or export a different view.

    Convention: gxy = E[conj(X) * Y], H = gxy / gxx (reference -> target).
    """

    def __init__(self, ref_trace, trace):
        self.ref_trace = ref_trace
        self.trace = trace
        self.stats = trace.stats
        self.ref_stats = ref_trace.stats
        self.method = None
        self.freq = None
        self.gxx = None
        self.gyy = None
        self.gxy = None
        self.n_avg = None

    # ------------------------------------------------------------------ compute
    def compute(self, method="welch", win=None, overlap_percent=50.0, nw=4.0, linf=0.0, lsup=None):
        """
        win : window length in SECONDS (welch only). None -> N/8 samples.
        """
        n = min(self.ref_trace.stats.npts, self.trace.stats.npts)
        delta = self.trace.stats.delta
        win_samples = round(win / delta) if win else None
        res = SpectrumTool.compute_cross_spectra(self.ref_trace.data[:n], self.trace.data[:n], delta,
                                                 method=method, win=win_samples,
                                                 overlap_percent=overlap_percent, nw=nw,
                                                 linf=linf, lsup=lsup)
        self.method = method
        self.freq = res["freq"]
        self.gxx = res["gxx"]
        self.gyy = res["gyy"]
        self.gxy = res["gxy"]
        self.n_avg = res["n_avg"]

    @classmethod
    def from_stream(cls, stream, reference=0, include_reference=False, trim=True, **kwargs):
        """
        Same logic as apply_cross_correlation: one trace of the stream is the
        reference and every other trace is analysed with respect to it.

        kwargs are passed to compute() (method, win, overlap_percent, nw, linf, lsup).

        Returns
        -------
        list of TraceCrossSpectralResult (one per target trace)
        """
        st = stream.copy()
        if len(st) < 2:
            print("[WARNING] Need at least two traces for coherence / transfer function")
            return []

        sr = st[0].stats.sampling_rate
        for tr in st:
            if not np.isclose(tr.stats.sampling_rate, sr):
                raise ValueError("Inconsistent sampling rates in stream.")

        if trim:
            # Common window only, no zero padding (padding would bias the coherence)
            common_start = max(tr.stats.starttime for tr in st)
            common_end = min(tr.stats.endtime for tr in st)
            if common_end <= common_start:
                raise ValueError("Traces have no common time window.")
            st.trim(starttime=common_start, endtime=common_end)

        ref_trace = st[reference]
        results = []
        for i, tr in enumerate(st):
            if i == reference and not include_reference:
                continue
            try:
                r = cls(ref_trace, tr)
                r.compute(**kwargs)
            except Exception as e:
                print(f"[WARNING] Failed cross-spectral analysis {ref_trace.id} -> {tr.id}: {e}")
                continue
            results.append(r)
        return results

    # ------------------------------------------------------- derived quantities
    @property
    def coherence(self):
        """Magnitude-squared coherence, 0..1."""
        with np.errstate(divide="ignore", invalid="ignore"):
            coh = np.abs(self.gxy) ** 2 / (self.gxx * self.gyy)
        return np.clip(np.nan_to_num(coh), 0.0, 1.0)

    @property
    def transfer(self):
        """Complex transfer function H1 = Gxy / Gxx (reference -> target)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.gxy / self.gxx

    @property
    def amplitude(self):
        return np.abs(self.transfer)

    @property
    def phase_deg(self):
        return np.degrees(np.angle(self.transfer))

    def coherence_threshold(self, alpha=0.05):
        return SpectrumTool.coherence_threshold(self.n_avg, alpha)

    # --------------------------------------------------------------------- plot
    def plot(self, kind="both", axis_type="loglog", alpha=0.05, save_path: str = None,
             type_spec: str = "coh", extension: str = "png"):
        """
        kind      : 'both' (amplitude + phase + coherence), 'transfer' or 'coherence'
        axis_type : 'loglog' (log f, log |H|), 'xlog' (log f) or 'linlin'
        alpha     : significance level of the dashed coherence threshold
        """
        import matplotlib as mplt
        import matplotlib.pyplot as plt

        if "matplotlib.pyplot" not in sys.modules:  # only choose a backend if none is active yet
            mplt.use("MacOSX" if platform.system() == "Darwin" else "QtAgg")

        if kind not in ("both", "transfer", "coherence"):
            raise ValueError("kind must be 'both', 'transfer' or 'coherence'")

        f = self.freq[1:]  # skip 0 Hz, as in TraceSpectrumResult
        panels = {"both": ["amp", "phase", "coh"], "transfer": ["amp", "phase"], "coherence": ["coh"]}[kind]
        height = {1: 3.5, 2: 5.5, 3: 7.5}[len(panels)]
        fig, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(9, height))
        axes = np.atleast_1d(axes)

        for ax, p in zip(axes, panels):
            if p == "amp":
                ax.plot(f, self.amplitude[1:], linewidth=0.75)
                if axis_type == "loglog":
                    ax.set_yscale("log")
                ax.set_ylabel("|H|")
                ax.set_title(f"Transfer function / coherence  {self.ref_trace.id} -> {self.trace.id}")
            elif p == "phase":
                ax.plot(f, self.phase_deg[1:], linewidth=0.75)
                ax.set_ylim(-180, 180)
                ax.set_yticks([-180, -90, 0, 90, 180])
                ax.set_ylabel("Phase [deg]")
            else:
                ax.plot(f, self.coherence[1:], linewidth=0.75)
                ax.axhline(self.coherence_threshold(alpha), color="r", linestyle="--", linewidth=0.8,
                           label=f"{100 * (1 - alpha):g}% significance")
                ax.set_ylim(0, 1.02)
                ax.set_ylabel("Coherence")
                ax.legend(loc="lower right", fontsize=8)
            if axis_type in ("loglog", "xlog"):
                ax.set_xscale("log")
            ax.grid(True, which="both", linestyle="--", alpha=0.4)

        axes[-1].set_xlabel("Frequency (Hz)")
        axes[-1].set_xlim(f[0], f[-1])
        fig.tight_layout()

        if save_path:
            starttime = self.trace.stats.starttime
            textstr = (str(starttime.julday) + "_" + str(starttime.year) + "_"
                       + starttime.strftime("%Y_%m_%d_%H_%M_%S"))
            file_name = (self.ref_trace.id + "__" + self.trace.id + "." + textstr + "."
                         + type_spec + "." + extension)
            fig.savefig(os.path.join(save_path, file_name), dpi=300)
            plt.close(fig)
        else:
            plt.show()

    # ------------------------------------------------------------------- pickle
    def to_pickle(self, folder_path: str, compress: bool = True):
        """Serialize the full object to a pickle file (.coh)."""
        if not folder_path:
            print("[ERROR] --folder_path must be specified")
            return
        if not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path)
                print(f"[INFO] Created folder: {folder_path}")
            except Exception as e:
                print(f"[ERROR] Failed to create folder '{folder_path}': {e}")
                return

        t1 = self.trace.stats.starttime
        base_name = f"{self.ref_trace.id}__{self.trace.id}.D.{t1.year}.{t1.julday}"
        path_output = os.path.join(folder_path, base_name)
        counter = 1
        while os.path.exists(path_output + ".coh"):
            path_output = os.path.join(folder_path, f"{base_name}_{counter}")
            counter += 1
        path_output += ".coh"

        open_func = gzip.open if compress else open
        with open_func(path_output, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[INFO] {self.ref_trace.id} -> {self.trace.id} - Writing cross-spectral result to {path_output}")

    @staticmethod
    def from_pickle(filepath: str, compress: bool = True):
        """Load the full object from a pickle file."""
        open_func = gzip.open if compress else open
        with open_func(filepath, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, TraceCrossSpectralResult):
            raise TypeError("Pickle file does not contain a TraceCrossSpectralResult object.")
        return obj
