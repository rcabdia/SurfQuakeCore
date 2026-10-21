#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
spectral_tools
"""

import math
import numpy as np
import nitime.algorithms as tsa  # awesome !!!
from scipy.signal import welch, csd
# from spectrum import pmtm # might be for the future

class SpectrumTool:

    @staticmethod
    def compute_spectrum(data, delta, mode="fft"):
        """
        Return the amplitude spectrum using multitaper and compare with FFT.

        Parameters:
        - data: array-like, time-domain signal
        - delta: float, sample spacing (1 / sampling rate)
        - sta: unused (can be removed or kept for future use)

        Returns:
        - amplitude: amplitude spectrum from multitaper method (same units as input)
        - freq: frequencies corresponding to the spectrum
        - fft_vals: amplitude spectrum from FFT (same units as input)
        """
        # PSD (amplitude²/Hz)	Amplitude (unit)	amplitude = sqrt(psd * df)

        # Remove mean
        data = data - np.mean(data)

        # Pad data to next power of 2
        N = len(data)
        nfft = 2 ** math.ceil(math.log2(N))
        #data = np.pad(data, (0, D - N_orig), mode='constant')

        if mode == "multitaper":

            # Compute multitaper PSD
            freq, psd, _ = tsa.multi_taper_psd(data, 1 / delta, adaptive=True, jackknife=False, low_bias=True, NFFT=nfft)
            df = freq[1] - freq[0]  # Frequency bin width

            # Convert PSD to amplitude spectrum
            amplitude = np.sqrt(psd * df)

        else:
            # Compute FFT amplitude spectrum for comparison

            # Taper length must also be the REAL window
            # ----------------------------------------------------
            # Taper for conventional FFT
            # 5% cosine taper on EACH side
            # ----------------------------------------------------

            taper = np.ones(N)
            edge = int(round(0.05 * N))

            if edge > 1:

                ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, edge)))

                # 0 -> 1
                taper[:edge] = ramp

                # 1 -> 0
                taper[-edge:] = ramp[::-1]

            elif edge == 1:

                taper[0] = 0.0
                taper[-1] = 0.0

            data_tapered = data * taper

            # FFT using zero-padding to nfft
            fft_vals = np.fft.rfft(data_tapered, n=nfft)

            # One-sided amplitude spectrum
            amplitude = (2.0 / N) * np.abs(fft_vals)

            # DC must not be doubled
            amplitude[0] /= 2.0

            # Nyquist must not be doubled
            if nfft % 2 == 0:
                amplitude[-1] /= 2.0

            freq = np.fft.rfftfreq(nfft, d=delta)

        return amplitude, freq

    @staticmethod
    def compute_spectrogram(data, win, dt, linf, lsup, step_percentage=0.5, method="multitaper", nw=None):
        """
        Compute spectrogram using either multitaper or simple rFFT.

        Parameters
        ----------
        data : 1D array
            Time series.
        win : int
            Window length in samples.
        dt : float
            Sampling interval (seconds).
        linf, lsup : float
            Lower and upper frequency limits (Hz) to keep.
        step_percentage : float, optional
            Step size as a fraction of window length (0<step<=1).
        method : str, optional
            'multitaper' (using nitime) or 'fft' (simple FFT-based PSD).
        nw: time-bandwidth, when set to None it will be optimized (recommended: nw < 4.0)

        Returns
        -------
        spectrum : 2D array (freq x time)
        num_steps : int
        t : 1D array
            Time vector (center of each window).
        f : 1D array
            Frequency vector (Hz) within [linf, lsup].
        """

        data = np.asarray(data)
        # win -- samples
        win = int(win)
        # Ensure nfft is a power of 2
        nfft = 2 ** math.ceil(math.log2(win))  # Next power to 2

        # ----------------------------------------------------
        # Step MUST be relative to the original window
        # ----------------------------------------------------

        # Step size as a percentage of window size
        step_size = max(1,int(round(win * step_percentage)))  # Ensure step size is at least 1

        # Last possible REAL window
        lim = len(data) - win

        if lim < 0:
            raise ValueError("Window length is longer than data length.")

        # Exact window starting positions
        starts = np.arange(0, lim + 1, step_size)

        num_steps = len(starts) # Total number of steps

        S = np.zeros((nfft // 2 + 1, num_steps)) # Adjust output size for reduced steps

        # Precompute sampling frequency
        fs = 1.0 / dt  # Sampling frequency

        # Precompute taper for conventional method: 5% cosine taper on each side
        taper = None
        if method.lower() == "fft":
            # Taper length must also be the REAL window
            # ----------------------------------------------------
            # Taper for conventional FFT
            # 5% cosine taper on EACH side
            # ----------------------------------------------------

            taper = np.ones(win)
            edge = int(round(0.05 * win))

            if edge > 1:

                ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, edge)))

                # 0 -> 1
                taper[:edge] = ramp

                # 1 -> 0
                taper[-edge:] = ramp[::-1]

            elif edge == 1:

                taper[0] = 0.0
                taper[-1] = 0.0

        # Main loop
        # ----------------------------------------------------
        # Sliding windows
        # ----------------------------------------------------
        for idx, n in enumerate(range(0, lim + 1, step_size)):
            # Extract windowed data
            # IMPORTANT: extract 'win' samples, NOT 'nfft'
            data1 = data[n:n + win]
            # Remove mean
            data1 = data1 - np.mean(data1)

            if method.lower() == "multitaper":
                if nw:
                    freq, spec, _ = tsa.multi_taper_psd(data1, fs, NW=nw, NFFT=nfft,
                        jackknife=False, low_bias=False)
                else:
                    freq, spec, _ = tsa.multi_taper_psd(data1, fs, NFFT=nfft, adaptive=True,
                                                        jackknife=False, low_bias=True)

            elif method.lower() == "fft":
                # Apply 5% taper
                data1 = data1 * taper
                # Conventional rFFT-based PSD, Zero-padding happens HERE
                spec = np.fft.rfft(data1, n=nfft)
                spec = (np.abs(spec) ** 2) / (fs * win)

            else:
                raise ValueError("method must be 'multitaper' or 'fft'")

            S[:, idx] = spec

        # Frequency axis from nfft (for both methods)
        freq = np.fft.rfftfreq(nfft, d=dt)
        freq_mask = ((freq >= linf) & (freq <= lsup))

        # chop for the desire frequencies
        f = freq[freq_mask]
        spectrum = S[freq_mask, :]

        # Time axis: keep your original style for now
        #t = np.linspace(0, len(data) * dt, spectrum.shape[1])

        # ----------------------------------------------------
        # REAL time axis
        #
        # Give time at CENTER of every analysis window
        # ----------------------------------------------------
        t = (starts + (win - 1) / 2.0) * dt

        # we prefer starts at beginning
        #t = starts * dt

        return spectrum, num_steps, t, f

    @staticmethod
    def find_nearest(array, value):
        idx, val = min(enumerate(array), key=lambda x: abs(x[1] - value))
        return idx, val

    @staticmethod
    def compute_cross_spectra(x, y, delta, method="welch", win=None, overlap_percent=50.0, nw=4.0,
                              linf=0.0, lsup=None):
        """
        Auto- and cross-spectral densities between a reference signal x (input)
        and a target signal y (output).

        Convention: Gxy = E[ conj(X) * Y ], so the transfer function x -> y is
        H = Gxy / Gxx, and a pure delay tau of y with respect to x gives
        phase(H) = -2*pi*f*tau.

        Coherence / transfer function are NOT computed here on purpose: they are
        cheap to derive from (Gxx, Gyy, Gxy) and keeping the three spectra allows
        H1 = Gxy/Gxx, H2 = Gyy/conj(Gxy), phase, etc. later without recomputing.

        Parameters
        ----------
        x, y : 1D arrays, same length
            Reference (input) and target (output) time series.
        delta : float
            Sample spacing in seconds.
        method : str
            'welch'      -> segment averaging (Hann window, demeaned segments).
            'multitaper' -> DPSS tapers via nitime (averaging over tapers).
            Coherence is identically 1 without averaging, so a single plain FFT
            is deliberately not offered.
        win : int or None
            Segment length in samples (welch only). None -> max(16, N // 8).
        overlap_percent : float
            Segment overlap in percent (welch only).
        nw : float
            Time-bandwidth product (multitaper only).
        linf, lsup : float
            Frequency band (Hz) to keep. lsup=None -> Nyquist.

        Returns
        -------
        dict with keys
            freq : 1D array (Hz)
            gxx, gyy : 1D real arrays (auto spectral densities)
            gxy : 1D complex array (cross spectral density)
            n_avg : int, number of averaged segments / tapers (needed for the
                    coherence significance level)
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 1 or x.shape != y.shape:
            raise ValueError("x and y must be 1D arrays of the same length.")
        n = len(x)
        fs = 1.0 / delta
        x = x - np.mean(x)
        y = y - np.mean(y)

        if method.lower() == "welch":
            nperseg = int(win) if win else max(16, n // 8)
            nperseg = min(nperseg, n)
            noverlap = int(round(nperseg * overlap_percent / 100.0))
            noverlap = min(noverlap, nperseg - 1)
            nfft = 2 ** math.ceil(math.log2(nperseg))
            kw = dict(fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                      detrend="constant", scaling="density", return_onesided=True)
            freq, gxx = welch(x, **kw)
            _, gyy = welch(y, **kw)
            _, gxy = csd(x, y, **kw)  # scipy: conj(X) * Y
            step = nperseg - noverlap
            n_avg = 1 + (n - nperseg) // step

        elif method.lower() == "multitaper":
            nfft = 2 ** math.ceil(math.log2(n))
            freq, c = tsa.multi_taper_csd(np.vstack([x, y]), Fs=fs, NW=nw, adaptive=False,
                                          low_bias=True, NFFT=nfft)
            gxx = np.real(c[0, 0])
            gyy = np.real(c[1, 1])
            gxy = c[1, 0]  # nitime stores conj(Y)*X in [0,1]; [1,0] = conj(X)*Y
            _, eigs = tsa.dpss_windows(n, nw, int(2 * nw))
            n_avg = max(1, int(np.sum(eigs > 0.9)))
        else:
            raise ValueError("method must be 'welch' or 'multitaper'")

        if lsup is None:
            lsup = fs / 2.0
        mask = (freq >= linf) & (freq <= lsup)
        return {"freq": freq[mask], "gxx": gxx[mask], "gyy": gyy[mask], "gxy": gxy[mask],
                "n_avg": int(n_avg)}

    @staticmethod
    def coherence_threshold(n_avg, alpha=0.05):
        """
        Coherence value above which the estimate is significant at level alpha
        (null hypothesis: true coherence = 0), for n_avg independent averages.
        """
        if n_avg < 2:
            return 1.0
        return float(1.0 - alpha ** (1.0 / (n_avg - 1)))
