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

        scalogram_amplitude = cw.scalogram()
        t = np.linspace(0, tr.stats.delta * scalogram_amplitude.shape[1], scalogram_amplitude.shape[1])
        f = np.logspace(np.log10(fmin), np.log10(fmax), scalogram_amplitude.shape[0])

        # Prediction masks
        c_f = param / (2 * math.pi)
        ff = np.linspace(fmin, fmax, scalogram_amplitude.shape[0])
        pred = (math.sqrt(2) * c_f / ff) - (math.sqrt(2) * c_f / fmax)
        pred_comp = t[-1] - pred

        self.cwt_data = (t, f, scalogram_amplitude, pred, pred_comp)

    def plot_cwt(
            self,
            save_path: str = None,
            clip: float = None,
            plot_date: bool = False,
            split=None,
            vmax_db: float = None,
            cmap: str = "rainbow",
            smooth: bool = False):

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
        import matplotlib.gridspec as gridspec
        from matplotlib.ticker import ScalarFormatter, FixedLocator, FuncFormatter

        # ======================================================
        # CWT data
        # ======================================================
        t, f, scalogram, pred, pred_comp = self.cwt_data

        tr = self.trace

        t = np.asarray(t)
        f = np.asarray(f)

        # Copy so plotting modifications do not affect
        # the original CWT stored in self.cwt_data
        scalogram = np.asarray(scalogram).copy()

        pred = np.asarray(pred)
        pred_comp = np.asarray(pred_comp)

        # ======================================================
        # Convert CWT power to relative dB
        # ======================================================

        max_power = np.nanmax(scalogram)

        if not np.isfinite(max_power) or max_power <= 0:
            raise ValueError(
                "Scalogram maximum must be greater than zero."
            )

        with np.errstate(divide="ignore", invalid="ignore"):

            scalogram = 10.0 * np.log10(
                scalogram / max_power
            )

        # ======================================================
        # Display limits
        # ======================================================
        if vmax_db is None:

            # vmax_db = np.percentile(scalogram, q=99.5, axis=None)
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
        finite_values = scalogram[np.isfinite(scalogram)]

        if finite_values.size == 0:
            raise ValueError(
                "Scalogram contains no finite values."
            )

        if clip is not None:

            vmin_db = float(clip)

        else:

            vmin_db = float(
                np.percentile(
                    finite_values,
                    q=1
                )
            )

        # ------------------------------------------------------
        # Validate limits
        # ------------------------------------------------------
        if vmin_db >= vmax_db:
            raise ValueError(
                f"vmin_db ({vmin_db:g} dB) must be lower than "
                f"vmax_db ({vmax_db:g} dB)."
            )

        # Replace invalid / infinite values for plotting
        scalogram = np.where(
            np.isfinite(scalogram),
            scalogram,
            vmin_db
        )

        # Levels used only by contourf
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
            raise ValueError(
                "split frequency must be greater than 0 Hz."
            )

        starttime = tr.stats.starttime

        # ======================================================
        # X coordinates
        # ======================================================
        if plot_date:

            start_num = mdates.date2num(
                starttime.datetime
            )

            waveform_x = (
                    start_num
                    + tr.times() / 86400.0
            )

            cwt_x = (
                    start_num
                    + t / 86400.0
            )

            # pred and pred_comp are also time coordinates
            pred_x = (
                    start_num
                    + pred / 86400.0
            )

            pred_comp_x = (
                    start_num
                    + pred_comp / 86400.0
            )

        else:

            waveform_x = tr.times()
            cwt_x = t

            pred_x = pred
            pred_comp_x = pred_comp

        # ======================================================
        # NORMAL MODE
        # ======================================================
        if split_freq is None:

            self.fig_spec = plt.figure(
                figsize=(10, 5)
            )

            gs = gridspec.GridSpec(
                2,
                2,
                width_ratios=[1, 0.03],
                height_ratios=[1, 1],
                hspace=0.02,
                wspace=0.02
            )

            ax_waveform = self.fig_spec.add_subplot(
                gs[0, 0]
            )

            ax_spec = self.fig_spec.add_subplot(
                gs[1, 0],
                sharex=ax_waveform
            )

            ax_cbar = self.fig_spec.add_subplot(
                gs[1, 1]
            )

            # --------------------------------------------------
            # Original waveform
            # --------------------------------------------------
            ax_waveform.plot(
                waveform_x,
                tr.data,
                linewidth=0.75
            )

            ax_waveform.set_title(
                f"CWT Scalogram for {tr.id}"
            )

            ax_waveform.set_ylabel(
                "Amplitude"
            )

            ax_waveform.tick_params(
                labelbottom=False
            )

            formatter = ScalarFormatter(
                useMathText=True
            )

            formatter.set_powerlimits(
                (0, 0)
            )

            ax_waveform.yaxis.set_major_formatter(
                formatter
            )

            # --------------------------------------------------
            # Full CWT
            # --------------------------------------------------


            if smooth:

                pcm = ax_spec.contourf(
                    cwt_x,
                    f,
                    scalogram,
                    levels=levels,
                    cmap=cmap,
                    vmin=vmin_db,
                    vmax=vmax_db,
                    extend="both"
                )

            else:

                pcm = ax_spec.pcolormesh(
                    cwt_x,
                    f,
                    scalogram,
                    shading="auto",
                    cmap=cmap,
                    vmin=vmin_db,
                    vmax=vmax_db
                )

            # Prediction / mask regions
            ax_spec.fill_between(
                pred_x,
                f,
                0,
                color="black",
                edgecolor="red",
                alpha=0.3
            )

            ax_spec.fill_between(
                pred_comp_x,
                f,
                0,
                color="black",
                edgecolor="red",
                alpha=0.3
            )

            ax_spec.set_ylim(
                np.min(f),
                np.max(f)
            )

            ax_spec.set_ylabel(
                "Frequency [Hz]"
            )

            # --------------------------------------------------
            # X axis
            # --------------------------------------------------
            if plot_date:

                locator = mdates.AutoDateLocator()

                ax_spec.xaxis.set_major_locator(
                    locator
                )

                ax_spec.xaxis.set_major_formatter(
                    mdates.ConciseDateFormatter(
                        locator
                    )
                )

                ax_spec.set_xlabel(
                    "Date / Time [UTC]"
                )

            else:

                ax_spec.set_xlabel(
                    "Time [s]"
                )

            # --------------------------------------------------
            # Colorbar
            # --------------------------------------------------
            cbar = self.fig_spec.colorbar(
                pcm,
                cax=ax_cbar,
                orientation="vertical"
            )

            cbar.set_label(
                "Power [dB]"
            )

            annotation_axis = ax_waveform

        # ======================================================
        # SPLIT MODE
        # ======================================================
        else:

            nyquist = (
                    tr.stats.sampling_rate / 2.0
            )

            if split_freq >= nyquist:
                raise ValueError(
                    f"split frequency ({split_freq:g} Hz) "
                    f"must be below Nyquist "
                    f"({nyquist:g} Hz)."
                )

            if split_freq >= np.max(f):
                raise ValueError(
                    f"split frequency ({split_freq:g} Hz) "
                    f"must be below the maximum CWT frequency "
                    f"({np.max(f):g} Hz)."
                )

            # --------------------------------------------------
            # Divide the EXISTING CWT.
            #
            # 0 Hz is excluded from the lower part because
            # period = 1/f cannot be defined at f = 0.
            # --------------------------------------------------
            low_mask = (
                    (f > 0)
                    & (f < split_freq)
            )

            high_mask = (
                    f >= split_freq
            )

            if not np.any(low_mask):
                raise ValueError(
                    f"No positive CWT frequencies below "
                    f"{split_freq:g} Hz."
                )

            if not np.any(high_mask):
                raise ValueError(
                    f"No CWT frequencies at or above "
                    f"{split_freq:g} Hz."
                )

            # --------------------------------------------------
            # High-frequency representation remains in Hz
            # --------------------------------------------------
            freq_high = f[high_mask]

            scalogram_high = scalogram[
                             high_mask, :
                             ]

            pred_high_x = pred_x[
                high_mask
            ]

            pred_comp_high_x = pred_comp_x[
                high_mask
            ]

            # --------------------------------------------------
            # Low-frequency representation becomes period
            # --------------------------------------------------
            freq_low = f[low_mask]

            scalogram_low = scalogram[
                            low_mask, :
                            ]

            pred_low_x = pred_x[
                low_mask
            ]

            pred_comp_low_x = pred_comp_x[
                low_mask
            ]

            period_low = (
                    1.0 / freq_low
            )

            # pcolormesh behaves best with monotonically
            # increasing coordinates, so sort periods.
            order = np.argsort(
                period_low
            )

            period_low = period_low[
                order
            ]

            scalogram_low = scalogram_low[
                            order, :
                            ]

            # The prediction curves must follow exactly
            # the same scale reordering.
            pred_low_x = pred_low_x[
                order
            ]

            pred_comp_low_x = pred_comp_low_x[
                order
            ]

            # ==================================================
            # Figure layout
            #
            # waveform     -> height 1
            # high freq    -> height 1
            # period panel -> height 2
            # ==================================================
            self.fig_spec = plt.figure(
                figsize=(10, 8)
            )

            gs = gridspec.GridSpec(
                3,
                2,
                width_ratios=[1, 0.03],
                height_ratios=[1, 1, 2],
                hspace=0.04,
                wspace=0.02
            )

            # Low-frequency waveform uses LEFT amplitude axis
            ax_waveform_low = self.fig_spec.add_subplot(
                gs[0, 0]
            )

            # High-frequency waveform uses RIGHT amplitude axis
            ax_waveform_high = (
                ax_waveform_low.twinx()
            )

            ax_high = self.fig_spec.add_subplot(
                gs[1, 0],
                sharex=ax_waveform_low
            )

            ax_low = self.fig_spec.add_subplot(
                gs[2, 0],
                sharex=ax_waveform_low
            )

            # One colorbar for both CWT panels
            ax_cbar = self.fig_spec.add_subplot(
                gs[1:, 1]
            )

            # ==================================================
            # FILTERED TIME SERIES
            # ==================================================
            tr_low = tr.copy()
            tr_high = tr.copy()

            # Keep the same preprocessing as your
            # final spectrogram implementation
            tr_low.detrend(
                type="linear"
            )

            tr_low.detrend(
                type="simple"
            )

            tr_high.detrend(
                type="linear"
            )

            tr_high.detrend(
                type="simple"
            )

            tr_low.taper(
                max_percentage=0.05
            )

            tr_high.taper(
                max_percentage=0.05
            )

            # Zero-phase filtering prevents time shifts
            tr_low.filter(
                "lowpass",
                freq=split_freq,
                corners=4,
                zerophase=True
            )

            tr_high.filter(
                "highpass",
                freq=split_freq,
                corners=4,
                zerophase=True
            )

            tr_low.detrend(
                type="simple"
            )

            tr_high.detrend(
                type="simple"
            )

            # --------------------------------------------------
            # Low-frequency signal -> LEFT y axis
            # --------------------------------------------------
            line_low, = ax_waveform_low.plot(
                waveform_x,
                tr_low.data,
                linewidth=0.8,
                alpha=0.75,
                label=f"Low-pass < {split_freq:g} Hz"
            )

            ax_waveform_low.set_ylabel(
                "Low-freq amplitude"
            )

            # --------------------------------------------------
            # High-frequency signal -> RIGHT y axis
            # --------------------------------------------------
            line_high, = ax_waveform_high.plot(
                waveform_x,
                tr_high.data,
                linewidth=0.65,
                alpha=0.55,
                color="black",
                label=f"High-pass > {split_freq:g} Hz"
            )

            ax_waveform_high.set_ylabel(
                "High-freq amplitude"
            )

            # Scientific notation independently on both axes
            formatter_left = ScalarFormatter(
                useMathText=True
            )

            formatter_left.set_powerlimits(
                (0, 0)
            )

            ax_waveform_low.yaxis.set_major_formatter(
                formatter_left
            )

            formatter_right = ScalarFormatter(
                useMathText=True
            )

            formatter_right.set_powerlimits(
                (0, 0)
            )

            ax_waveform_high.yaxis.set_major_formatter(
                formatter_right
            )

            ax_waveform_low.set_title(
                f"CWT Scalogram for {tr.id}"
            )

            ax_waveform_low.tick_params(
                labelbottom=False
            )

            # Combined legend for both waveform axes
            ax_waveform_low.legend(
                [line_low, line_high],
                [
                    line_low.get_label(),
                    line_high.get_label()
                ],
                loc="upper right",
                fontsize=8,
                framealpha=0.7
            )

            # ==================================================
            # HIGH-FREQUENCY CWT
            # ==================================================

            if smooth:

                pcm = ax_high.contourf(
                    cwt_x,
                    freq_high,
                    scalogram_high,
                    levels=levels,
                    cmap=cmap,
                    vmin=vmin_db,
                    vmax=vmax_db,
                    extend="both"
                )

            else:

                pcm = ax_high.pcolormesh(
                    cwt_x,
                    freq_high,
                    scalogram_high,
                    shading="auto",
                    cmap=cmap,
                    vmin=vmin_db,
                    vmax=vmax_db
                )

            # Preserve the prediction / mask regions
            ax_high.fill_between(
                pred_high_x,
                freq_high,
                split_freq,
                color="black",
                edgecolor="red",
                alpha=0.3
            )

            ax_high.fill_between(
                pred_comp_high_x,
                freq_high,
                split_freq,
                color="black",
                edgecolor="red",
                alpha=0.3
            )

            ax_high.set_ylabel(
                "Frequency [Hz]"
            )

            ax_high.set_ylim(
                split_freq,
                np.max(freq_high)
            )

            ax_high.tick_params(
                labelbottom=False
            )

            # ==================================================
            # LOW-FREQUENCY CWT AS PERIOD
            # ==================================================
            ax_low.pcolormesh(
                cwt_x,
                period_low,
                scalogram_low,
                shading="auto",
                cmap=cmap,
                vmin=vmin_db,
                vmax=vmax_db
            )

            if smooth:

                ax_low.contourf(
                    cwt_x,
                    period_low,
                    scalogram_low,
                    levels=levels,
                    cmap=cmap,
                    vmin=vmin_db,
                    vmax=vmax_db,
                    extend="both"
                )

            else:

                ax_low.pcolormesh(
                    cwt_x,
                    period_low,
                    scalogram_low,
                    shading="auto",
                    cmap=cmap,
                    vmin=vmin_db,
                    vmax=vmax_db
                )
            # --------------------------------------------------
            # The original frequency-domain mask extends
            # toward f = 0.
            #
            # In period representation:
            #
            # f -> 0  means  T -> infinity
            #
            # therefore extend the shaded region toward the
            # largest displayed period.
            # --------------------------------------------------
            max_period = np.max(
                period_low
            )

            ax_low.fill_between(
                pred_low_x,
                period_low,
                max_period,
                color="black",
                edgecolor="red",
                alpha=0.3
            )

            ax_low.fill_between(
                pred_comp_low_x,
                period_low,
                max_period,
                color="black",
                edgecolor="red",
                alpha=0.3
            )

            ax_low.set_yscale(
                "log"
            )

            ax_low.set_ylim(
                np.min(period_low),
                np.max(period_low)
            )

            # Short periods / high frequencies at the top.
            # Long periods / low frequencies at the bottom.
            ax_low.invert_yaxis()

            ax_low.set_ylabel(
                "Period [s]"
            )

            # ==================================================
            # Useful period annotations
            #
            # 0.1, 0.2, 0.5,
            # 1, 2, 5,
            # 10, 20, 50, ...
            # ==================================================
            period_min = np.min(
                period_low
            )

            period_max = np.max(
                period_low
            )

            exponent_min = int(
                np.floor(
                    np.log10(period_min)
                )
            )

            exponent_max = int(
                np.ceil(
                    np.log10(period_max)
                )
            )

            period_ticks = []

            for exponent in range(
                    exponent_min,
                    exponent_max + 1):

                for multiplier in (
                        1,
                        2,
                        5):

                    value = (
                            multiplier
                            * 10.0 ** exponent
                    )

                    if (
                            period_min
                            <= value
                            <= period_max
                    ):
                        period_ticks.append(
                            value
                        )

            if period_ticks:
                ax_low.yaxis.set_major_locator(
                    FixedLocator(
                        period_ticks
                    )
                )

                ax_low.yaxis.set_major_formatter(
                    FuncFormatter(
                        lambda value, _:
                        f"{value:g}"
                    )
                )

            # ==================================================
            # X axis
            # ==================================================
            if plot_date:

                locator = mdates.AutoDateLocator()

                ax_low.xaxis.set_major_locator(
                    locator
                )

                ax_low.xaxis.set_major_formatter(
                    mdates.ConciseDateFormatter(
                        locator
                    )
                )

                ax_low.set_xlabel(
                    "Date / Time [UTC]"
                )

            else:

                ax_low.set_xlabel(
                    "Time [s]"
                )

            # --------------------------------------------------
            # Shared colorbar
            # --------------------------------------------------
            cbar = self.fig_spec.colorbar(
                pcm,
                cax=ax_cbar,
                orientation="vertical"
            )

            cbar.set_label(
                "Power [dB]"
            )

            annotation_axis = ax_waveform_low

        # ======================================================
        # Trace start-time annotation
        # ======================================================
        date_str = starttime.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

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

            self.fig_spec.savefig(
                save_path,
                dpi=300
            )

            plt.close(
                self.fig_spec
            )

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