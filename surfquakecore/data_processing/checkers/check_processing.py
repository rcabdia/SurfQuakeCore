#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_processing
"""

ANALYSIS_KEYS = ['rmean', 'taper', 'normalize', 'differentiate', 'integrate', 'filter', 'wiener_filter',
                 'shift', 'remove_response', 'add_white_noise', 'whitening', 'remove_spikes',
                 'time_normalization', 'wavelet_denoise', 'resample', 'fill_gaps', 'smoothing']

RMEAN_METHODS = ['simple', 'linear', 'constant', 'demean', 'polynomial', 'spline']

TAPER_METHODS = ['cosine', 'barthann', 'bartlett', 'blackman', 'blackmanharris', 'bohman', 'boxcar', 'chebwin',
                 'flattop', 'gaussian', 'general_gaussian', 'hamming', 'hann', 'kaiser', 'nuttall', 'parzen', 'slepian',
                 'triang']

WAVELET_METHODS = ['db2', 'db4', 'db6', 'db8', 'db10', 'db12', 'db14', 'db16', 'db18', 'db20', 'sym2', 'sym4', 'sym6',
                   'sym8', 'sym10', 'sym12', 'sym14', 'sym16', 'sym18', 'sym20', 'coif2', 'coif3', 'coif4', 'coif6',
                   'coif8', 'coif10', 'coif12', 'coif14', 'coif16', 'bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4',
                   'bior2.6', 'bior2.8', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8']

INTEGRATE_METHODS = ['cumtrapz', 'spline']

FILTER_METHODS = ['bandpass', 'bandstop', 'lowpass', 'highpass', 'cheby1', 'cheby2', 'elliptic', 'bessel']

TIME_METHODS = ['1bit', 'clipping', 'clipping_iteration', 'time_normalization']

FILL_GAP_METHODS = ['latest', 'interpolate']

SMOOTHING_METHODS = ['mean', 'gaussian', 'adaptive', 'tkeo']

TAPER_KEYS = ['method', 'max_percentage', 'max_length', 'side']

TAPER_SIDE = ['both', 'left', 'right']

def check_rmean(config):
    _keys = []

    if 'method' in config:
        if config['method'] in RMEAN_METHODS:
            if config['method'] in ['polynomial', 'spline']:
                _keys = config.keys()
            else:
                return True

            if len(_keys) > 0:
                if config['method'] == 'polynomial':
                    if 'order' in _keys:
                        if isinstance(config['order'], int):
                            return True
                        else:
                            raise ValueError(
                                f"RMEAN: config {config} is not valid. It must be a valid .yaml file for "
                                f"AnalysisConfig. \n"
                                f"Wrong type for order value. Must be int")
                    else:
                        raise ValueError(f"RMEAN: config {config} is not valid. It must be a valid .yaml file for "
                                         f"AnalysisConfig. \n"
                                         f"order parameter is required for polynomial method")
                elif config['method'] == 'spline':
                    if 'order' in _keys and 'dspline' in _keys:
                        if isinstance(config['order'], int) and isinstance(config['dspline'], int):
                            if 1 <= config['order'] <= 5:
                                return True
                            else:
                                raise ValueError(
                                    f"RMEAN: config {config} is not valid. It must be a valid .yaml file for "
                                    f"AnalysisConfig.\n"
                                    f"Wrong order value. 1 <= order <= 5")
                        else:
                            raise ValueError(
                                f"RMEAN: config {config} is not valid. It must be a valid .yaml file for "
                                f"AnalysisConfig.\n"
                                f"Wrong type for order and/or dspline values. Must be int")
                    else:
                        raise ValueError(f"RMEAN: config {config} is not valid. It must be a valid .yaml file for "
                                         f"AnalysisConfig.\n"
                                         f"order and dspline parameters are required for spline method")
            else:
                raise ValueError(f"RMEAN: config {config} is not valid. It must be a valid .yaml file for "
                                 f"AnalysisConfig. \n"
                                 f"missing order and/or dspline parameters")
        else:
            raise ValueError(f"RMEAN: config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig\n. "
                             f"Method not supported.\n"
                             f"Supported methods: simple, linear, constant, demean, polynomial, spline.")
    else:
        raise ValueError(f"RMEAN: config {config} is not valid. It must be a valid .yaml file for "
                         f"AnalysisConfig.\n"
                         f"method parameter is required.")


def check_taper(config):
    _keys = config.keys()
    _method = True
    _max_percentage = True

    if 'method' in _keys:
        if config['method'] not in TAPER_METHODS:
            raise ValueError(f"TAPER: config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig\n. "
                             f"Method not supported.\n"
                             f"Supported methods: 'cosine', 'barthann', 'bartlett', 'blackman', 'blackmanharris', "
                             f"'bohman', 'boxcar', 'chebwin', 'flattop', 'gaussian', 'general_gaussian', 'hamming', "
                             f"'hann', 'kaiser', 'nuttall', 'parzen', 'slepian', 'triang'")
    else:
        raise ValueError(f"TAPER: config {config} is not valid. It must be a valid .yaml file for "
                         f"AnalysisConfig.\n"
                         f"method parameter is required.")

    if 'max_percentage' in _keys:
        if isinstance(config['max_percentage'], float):
            if config['max_percentage'] < 0.0 or config['max_percentage'] > 0.5:
                raise ValueError(
                    f"TAPER: config {config} is not valid. It must be a valid .yaml file for "
                    f"AnalysisConfig.\n"
                    f"Wrong max_percentage value. 0.0 <= max_percentage <= 0.5")
        else:
            raise ValueError(
                f"TAPER: config {config} is not valid. It must be a valid .yaml file for "
                f"AnalysisConfig.\n"
                f"Wrong type for max_percentage value. Must be float")
    else:
        raise ValueError(f"TAPER: config {config} is not valid. It must be a valid .yaml file for "
                         f"AnalysisConfig.\n"
                         f"max_percentage parameter is required.")

    if 'max_lenght' in _keys:
        if not isinstance(config['max_length'], float):
            raise ValueError(
                f"TAPER: config {config} is not valid. It must be a valid .yaml file for "
                f"AnalysisConfig.\n"
                f"Wrong type for max_length value. Must be float")

    if 'side' in _keys:
        if config['side'] not in TAPER_SIDE:
            raise ValueError(f"Taper: config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig.\n"
                             f"Wrong value for side.\n"
                             f"Valid values for side: 'both', 'left', 'right'")

    return True


def check_normalize(config):
    _config_keys = config.keys()

    if 'norm' in _config_keys:
        if config['norm'] is None or isinstance(config['norm'], float):
            return True
        else:
            raise ValueError(f"NORMALIZE: config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig.\n"
                             f"Wrong type for norm value. Must be None or float")
    else:
        raise ValueError(f"NORMALIZE: config {config} is not valid. It must be a valid .yaml file for "
                         f"AnalysisConfig.\n"
                         f"norm parameter is required.")


def check_shift(config):
    i = 0
    if len(config) > 0:
        for i in range(0, len(config)):
            _config_keys = config[i].keys()

            if 'name' in _config_keys and 'time' in _config_keys:
                return True
            else:
                raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                                 f"AnalysisConfig")


def check_integrate(config):
    _config_keys = config.keys()

    if 'method' in _config_keys:
        if config['method'] in INTEGRATE_METHODS:
            return True
        else:
            raise ValueError(f"INTEGRATE:config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig."
                             f"Method not supported.\n"
                             f"Supported methods: 'cumtrapz', 'spline'")
    else:
        raise ValueError(f"INTEGRATE:config {config} is not valid. It must be a valid .yaml file for "
                         f"AnalysisConfig."
                         f"method parameter is required.")


def check_filter(config):
    _config_keys = config.keys()

    if 'zerophase' in _config_keys:
        if not isinstance(config['zerophase'], bool):
            raise ValueError(f"FILTER: config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig."
                             f"Wrong type for zerophase value. Must be boolean")

    if 'method' in _config_keys:
        if config['method'] == 'bandpass' or config['method'] == 'bandstop' or config['method'] == 'remez_fir':
            if 'freqmin' in _config_keys and 'freqmax' in _config_keys:
                if isinstance(config['freqmin'], float) or isinstance(config['freqmax'], float):
                    return True
                else:
                    raise ValueError(f"FILTER: config {config} is not valid. It must be a valid .yaml file for "
                                     f"AnalysisConfig."
                                     f"Wrong type for freqmin and freqmax values. Must be float")
            else:
                raise ValueError(f"FILTER: config {config} is not valid. It must be a valid .yaml file for "
                                 f"AnalysisConfig."
                                 f"freqmin adn freqmax parameter are required for {config['method']} method.")
        elif (config['method'] == 'lowpass' or config['method'] == 'highpass' or config['method'] == 'lowpass_fir'
              or config['method'] == 'lowpass_cheby_2'):
            if 'freq' in _config_keys:
                return True
            else:
                raise ValueError(f"FILTER: config {config} is not valid. It must be a valid .yaml file for "
                                 f"AnalysisConfig."
                                 f"freq parameter are required for {config['method']} method.")
        else:
            raise ValueError(f"FILTER: config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig."
                             f"Method not supported.\n"
                             f"Supported methods: 'bandpass', 'bandstop', 'lowpass', 'highpass', 'lowpass_cheby_2', "
                             f"'lowpass_fir', 'remez_fir'")
    else:
        raise ValueError(f"FILTER: config {config} is not valid. It must be a valid .yaml file for "
                         f"AnalysisConfig."
                         f"method parameter is required.")


def check_remove_response(config):
    _config_keys = config.keys()

    if ('inventory' in _config_keys and 'water_level' in _config_keys and 'units' in _config_keys
            and 'pre_filt' in _config_keys):

        return True
    else:
        raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                         f"AnalysisConfig")


def check_differentiate(config):
    if 'method' in config.keys():
        return True

    else:
        raise ValueError(f"DIFFERENTIATE:config {config} is not valid. It must be a valid .yaml file for "
                         f"AnalysisConfig."
                         f"diff parameter is required.")


def check_wiener_filter(config):
    _config_keys = config.keys()

    if 'time_window' in _config_keys and 'noise_power' in _config_keys:
        return True
    else:
        raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                         f"AnalysisConfig")

def check_add_white_noise(config):
    _config_keys = config.keys()

    if 'SNR_dB' in _config_keys:
        return True
    else:
        raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                         f"AnalysisConfig")


def check_whiten(config):
    _config_keys = config.keys()

    if 'taper_edge' in _config_keys:
        if isinstance(config['taper_edge'], bool):
            return True
        else:
            raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig")
    else:
        return True


def check_remove_spikes(config):
    _config_keys = config.keys()

    if 'window_size' in _config_keys:
        return True
    else:
        raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                         f"AnalysisConfig")


def check_time_normalization(config):
    _config_keys = config.keys()

    if 'method' in _config_keys and 'norm_win' in _config_keys:
        if config['method'] in TIME_METHODS:
            return True
        else:
            raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig")
    else:
        raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                         f"AnalysisConfig")


def check_wavelet_denoise(config):
    _config_keys = config.keys()

    if 'dwt' in _config_keys and 'threshold' in _config_keys:
        if config['dwt'] in WAVELET_METHODS:
            return True
        else:
            raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig")
    else:
        raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                         f"AnalysisConfig")


def check_resample(config):
    _config_keys = config.keys()

    if 'sampling_rate' in _config_keys and 'pre_filter' in _config_keys:
        if isinstance(config['pre_filter'], bool):
            return True
        else:
            raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig")
    else:
        raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                         f"AnalysisConfig")


def check_fill_gaps(config):
    _config_keys = config.keys()

    if 'method' in _config_keys:
        if config['method'] in FILL_GAP_METHODS:
            return True
        else:
            raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig")
    else:
        raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                         f"AnalysisConfig")


def check_smoothing(config):
    _config_keys = config.keys()

    if 'method' in _config_keys and 'time_window' in _config_keys and 'FWHM' in _config_keys:
        if config['method'] in SMOOTHING_METHODS:
            return True
        else:
            raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig")
    else:
        raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                         f"AnalysisConfig")
