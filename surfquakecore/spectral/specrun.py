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
            mplt.use("QtAgg")

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

    def compute_spectrogram(self, win=5.0, overlap_percent=50.0, linf=0, lsup=None, method="multitaper", nw=None):

        if lsup is None:
            lsup = int(self.trace.stats.sampling_rate // 2)

        step_percentage = (100 - overlap_percent) * 1E-2
        self.spectrogram, self.num_steps, self.time, self.freq = \
            SpectrumTool.compute_spectrogram(self.trace.data, round(win * self.trace.stats.sampling_rate),
                                             self.trace.stats.delta, linf, lsup, step_percentage,
                                             method, nw)

    def plot_spectrogram(self, save_path: str = None, clip: float = None, plot_date: bool = False, split=None,
            vmax_db: float = None, vmin_db: float = None, cmap: str = "rainbow"):

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
            vmax_db = 0
            print(vmax_db)

        vmax_db = float(vmax_db)

        # ------------------------------------------------------
        # Lower display limit
        # ------------------------------------------------------
        if clip is not None:

            clip = float(clip)

            if clip >= vmax_db:
                raise ValueError(f"clip ({clip:g} dB) must be lower than "
                    f"vmax_db ({vmax_db:g} dB).")

            spectrogram = np.maximum(spectrogram, clip)

            vmin_db = clip

        else:

            if vmin_db is None:
                finite_values = spectrogram[np.isfinite(spectrogram)]

                if finite_values.size == 0:
                    raise ValueError("Spectrogram contains no finite values.")
                vmin_db = np.percentile(finite_values, q=1, axis=None)
                print(vmin_db)

        # Avoid invalid normalization in pathological cases
        if vmin_db >= vmax_db:
            vmin_db = vmax_db - 1.0

        # Replace -inf produced by log10(0)
        spectrogram = np.where(np.isfinite(spectrogram), spectrogram, vmin_db)

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
            pcm = ax_spec.pcolormesh(spec_x, freq, spectrogram, shading="auto",
                cmap=cmap, vmin=vmin_db, vmax=vmax_db)

            ax_spec.set_ylabel("Frequency [Hz]")

            # --------------------------------------------------
            # X axis
            # --------------------------------------------------
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
            spec_high = spectrogram[
                        high_mask, :]

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
            line_high, = ax_waveform_high.plot(
                waveform_x,
                tr_high.data,
                linewidth=0.65,
                alpha=0.55,
                color="black", label=f"High-pass > {split_freq:g} Hz")

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
            pcm = ax_high.pcolormesh(spec_x, freq_high, spec_high,
                shading="auto", cmap=cmap, vmin=vmin_db, vmax=vmax_db)

            ax_high.set_ylabel("Frequency [Hz]")

            ax_high.set_ylim(split_freq, np.max(freq_high))

            ax_high.tick_params(labelbottom=False)

            # ==================================================
            # LOW-FREQUENCY SPECTROGRAM AS PERIOD
            # ==================================================
            ax_low.pcolormesh(spec_x, period_low, spec_low, shading="auto",
                cmap=cmap, vmin=vmin_db, vmax=vmax_db)

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
        date_str = starttime.strftime(
            "%Y-%m-%d %H:%M:%S")

        textstr = (
            f"JD {starttime.julday} / "
            f"{starttime.year}\n"
            f"{date_str}"
        )

        annotation_axis.text(
            0.01,
            0.95,
            textstr,
            transform=annotation_axis.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc="lightyellow",
                ec="gray",
                alpha=0.5
            )
        )

        plt.tight_layout()

        # ======================================================
        # Save or display
        # ======================================================
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
