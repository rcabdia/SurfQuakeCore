#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
spectral_tools
"""

import math
import numpy as np
# from spectrum import pmtm # might be for the future
import nitime.algorithms as tsa  # awesome !!!


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
    def compute_spectrogram(data, win, dt, linf, lsup, step_percentage=0.5):
        # Sxx = spectrogram(data, fs=1/delta, nperseg=time_window/delta)

        # win -- samples
        # Ensure nfft is a power of 2
        nfft = 2 ** math.ceil(math.log2(win))  # Next power to 2

        # Step size as a percentage of window size
        step_size = max(1, int(nfft * step_percentage))  # Ensure step size is at least 1
        lim = len(data) - nfft  # Define sliding window limit
        num_steps = (lim // step_size) + 1  # Total number of steps
        S = np.zeros([nfft // 2 + 1, num_steps])  # Adjust output size for reduced steps

        # Precompute frequency indices for slicing spectrum
        fs = 1 / dt  # Sampling frequency
        # freq, _, _ = tsa.multi_taper_psd(np.zeros(nfft), fs, adaptive=True, jackknife=False, low_bias=False)
        for idx, n in enumerate(range(0, lim, step_size)):
            # print(f"{(n + 1) * 100 / lim:.2f}% done")
            data1 = data[n:nfft + n]
            data1 = data1 - np.mean(data1)
            freq, spec, _ = tsa.multi_taper_psd(data1, fs, adaptive=True, jackknife=False, low_bias=True)
            # spec, weights, eigenvalues = pmtm(data1, NW=2.5, k=4, show=False)
            # spec = np.mean(spec * np.transpose(weights), axis=0)
            S[:, idx] = spec

        freq = np.fft.rfftfreq(nfft, d=dt)
        value1, freq1 = SpectrumTool.find_nearest(freq, linf)
        value2, freq2 = SpectrumTool.find_nearest(freq, lsup)

        spectrum = S[value1:value2, :]

        t = np.linspace(0, len(data) * dt, spectrum.shape[1])
        f = np.linspace(linf, lsup, spectrum.shape[0])

        return spectrum, num_steps, t, f

    @staticmethod
    def find_nearest(array, value):
        idx, val = min(enumerate(array), key=lambda x: abs(x[1] - value))
        return idx, val
