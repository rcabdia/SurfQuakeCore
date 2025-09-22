#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
constants
"""

ANALYSIS_KEYS = ['rmean', 'taper', 'normalize', 'differentiate', 'integrate', 'filter', 'wiener_filter',
                 'shift', 'remove_response', 'add_noise', 'whitening', 'remove_spikes',
                 'time_normalization', 'wavelet_denoise', 'resample', 'fill_gaps', 'smoothing', 'rotate',
                 'cross_correlate', 'stack', 'synch', 'envelope', 'cut', 'concat', 'spectrum', 'spectrogram', 'cwt',
                 'entropy', 'snr', 'raw', 'beam', 'particle_motion', 'rename', 'kurtosis']

RMEAN_METHODS = ['simple', 'linear', 'constant', 'demean', 'polynomial', 'spline']

ROTATE_METHODS = ['GAC', "FREE"]

ROTATE_TYPES = ['NE->RT', 'RT->NE', 'ZNE->LQT', 'LQT->ZNE']

TAPER_METHODS = ['cosine', 'barthann', 'bartlett', 'blackman', 'blackmanharris', 'bohman', 'boxcar', 'chebwin',
                 'flattop', 'gaussian', 'general_gaussian', 'hamming', 'hann', 'kaiser', 'nuttall', 'parzen', 'slepian',
                 'triang']

WAVELET_METHODS = ['db2','db4','db6','db8','db10','db12','db14','db16','db18','db20', 'sym2', 'sym4', 'sym6', 'sym8',
                   'sym10', 'sym12', 'sym14', 'sym16', 'sym18', 'sym20', 'coif2', 'coif3', 'coif4', 'coif6', 'coif8',
                   'coif10', 'coif12', 'coif14', 'coif16', 'bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4',
                   'bior2.6', 'bior2.8', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8']

INTEGRATE_METHODS = ['cumtrapz', 'spline', 'spectral']

CROSS_CORRELATE_MODE = ['full', 'valid', 'same']

FILTER_METHODS = ['bandpass', 'bandstop', 'lowpass', 'highpass', 'cheby1', 'cheby2', 'elliptic', 'bessel']

TIME_METHODS = ['1bit', 'clipping', 'clipping iteration','time normalization']

FILL_GAP_METHODS = ['latest', 'interpolate']

SMOOTHING_METHODS = ['mean', 'gaussian', 'adaptive', 'tkeo']

TAPER_SIDE = ['both', 'left', 'right']

STACK_METHODS = ['linear', 'pw', 'root']

SYNCH_METHODS = ['starttime', 'MCCC']

ENVELOPE_MODE = ['FULL', 'SMOOTH']

SPECTRUM_METHODS = ["multitaper", "fft"]

CWT_WAVELETS = ["cm", "mh", "pa"]