#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
spectral_tools
"""

import math
import numpy as np
import nitime.algorithms as tsa  # awesome !!!
# from spectrum import pmtm # might be for the future

class SpectrumTool:

    @staticmethod
    def compute_spectrum(data, delta, mode="multitaper"):
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
        N_orig = len(data)
        D = 2 ** math.ceil(math.log2(N_orig))
        data = np.pad(data, (0, D - N_orig), mode='constant')
        N = len(data)

        # Apply Hann taper (window)
        taper = np.hanning(N)
        data_tapered = data * taper

        if mode == "multitaper":
            # Compute multitaper PSD
            freq, psd, _ = tsa.multi_taper_psd(data, 1 / delta, adaptive=True, jackknife=False, low_bias=True)
            df = freq[1] - freq[0]  # Frequency bin width

            # Convert PSD to amplitude spectrum
            amplitude = np.sqrt(psd * df)
        else:

            # Compute FFT amplitude spectrum for comparison
            amplitude = (2.0 / N) * np.abs(np.fft.rfft(data_tapered))
            amplitude[0] = amplitude[0] / 2
            if N % 2 == 0:
                amplitude[-1] = amplitude[-1] / 2
            freq = np.fft.rfftfreq(N, d=delta)

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
            'multitaper' (using nitime) or 'rfft' (simple FFT-based PSD).
        nw: time-bandwidth

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
        # Ensure nfft is a power of 2
        nfft = 2 ** math.ceil(math.log2(win))  # Next power to 2

        # Step size as a percentage of window size
        step_size = max(1, int(nfft * step_percentage))  # Ensure step size is at least 1
        lim = len(data) - nfft  # Define sliding window limit
        if lim < 0:
            raise ValueError("Window length (nfft) is longer than data length.")
        num_steps = (lim // step_size) + 1  # Total number of steps

        S = np.zeros([nfft // 2 + 1, num_steps])  # Adjust output size for reduced steps

        # Precompute sampling frequency
        fs = 1.0 / dt  # Sampling frequency

        # Precompute taper for conventional method: 5% cosine taper on each side
        taper = None
        if method.lower() == "fft":
            taper = np.ones(nfft)
            edge = int(0.05 * nfft)  # 5% of the window at each edge
            if edge > 0:
                # cosine ramp up at start
                ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 2 * edge)))
                taper[:edge] = ramp[:edge]
                # cosine ramp down at end
                taper[-edge:] = ramp[-edge:]

        # Main loop
        for idx, n in enumerate(range(0, lim + 1, step_size)):
            # Extract windowed data
            data1 = data[n:nfft + n]
            # Remove mean
            data1 = data1 - np.mean(data1)

            if method.lower() == "multitaper":
                if nw:
                    freq, spec, _ = tsa.multi_taper_psd(
                        data1,
                        fs,
                        NW=nw,
                        jackknife=False,
                        low_bias=False)
                else:
                    freq, spec, _ = tsa.multi_taper_psd(
                        data1,
                        fs,
                        adaptive=True,
                        jackknife=False,
                        low_bias=True)

            elif method.lower() == "fft":
                # Apply 5% taper
                data1 = data1 * taper
                # Conventional rFFT-based PSD
                spec = np.fft.rfft(data1)
                spec = (np.abs(spec) ** 2) / (fs * nfft)

            else:
                raise ValueError("method must be 'multitaper' or 'fft'")

            S[:, idx] = spec

        # Frequency axis from nfft (for both methods)
        freq = np.fft.rfftfreq(nfft, d=dt)
        value1, freq1 = SpectrumTool.find_nearest(freq, linf)
        value2, freq2 = SpectrumTool.find_nearest(freq, lsup)

        spectrum = S[value1:value2, :]

        # Time axis: keep your original style for now
        t = np.linspace(0, len(data) * dt, spectrum.shape[1])
        f = np.linspace(linf, lsup, spectrum.shape[0])

        return spectrum, num_steps, t, f

    @staticmethod
    def find_nearest(array, value):
        idx, val = min(enumerate(array), key=lambda x: abs(x[1] - value))
        return idx, val
