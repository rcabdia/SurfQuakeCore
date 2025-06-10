#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
spectral_tools
"""

import math
import numpy as np

class SpectrumTool:

    @staticmethod
    def compute_spectrum(data, delta):
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

        # Compute FFT amplitude spectrum for comparison
        amplitude = (2.0 / N) * np.abs(np.fft.rfft(data_tapered))
        amplitude[0] = amplitude[0] / 2
        if N % 2 == 0:
            amplitude[-1] = amplitude[-1] / 2
        freq = np.fft.rfftfreq(N, d=delta)
        return amplitude, freq