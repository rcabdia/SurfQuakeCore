#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
config_parser
"""

from surfquakecore.data_processing.validators.constants import ANALYSIS_KEYS
from surfquakecore.data_processing.validators import validate_step


def parse_configuration_file(config):
    """
    Main parser for analysis configuration. Validates and parses each step.

    Args:
        config (dict): Full YAML configuration dictionary.

    Returns:
        list: List of validated and parsed analysis step configurations.

    Raises:
        ValueError: If required sections or keys are missing.
    """
    if 'Analysis' not in config:
        raise ValueError("Missing 'Analysis' section in config.")

    steps = config['Analysis']
    ordered_keys = sorted(steps.keys(), key=lambda k: int(k.split('_')[-1]))
    parsed_config = []

    for step_key in ordered_keys:
        step_config = steps[step_key]
        name = step_config.get('name')
        if not name:
            raise ValueError(f"Step '{step_key}' missing 'name' field.")

        if name not in ANALYSIS_KEYS:
            raise ValueError(f"Unsupported step: {name}")

        validate_step(name, step_config)
        if name == 'rmean':
            parsed_config.append(parse_rmean(step_config))
        elif name == 'taper':
            parsed_config.append(parse_taper(step_config))
        elif name == 'normalize':
            parsed_config.append(parse_normalize(step_config))
        elif name == 'filter':
            parsed_config.append(parse_filter(step_config))
        elif name == 'whitening':
            parsed_config.append(parse_whitening(step_config))
        elif name == 'time_normalization':
            parsed_config.append(parse_time_normalization(step_config))
        elif name == 'dwt':
            parsed_config.append(parse_dwt(step_config))
        elif name == 'remove_spikes':
            parsed_config.append(parse_remove_spikes(step_config))
        elif name == 'add_noise':
            parsed_config.append(parse_add_noise(step_config))
        elif name == 'differentiate':
            parsed_config.append(parse_differentiate(step_config))
        elif name == 'integrate':
            parsed_config.append(parse_integrate(step_config))
        elif name == 'smoothing':
            parsed_config.append(parse_smoothing(step_config))
        elif name == 'fill_gaps':
            parsed_config.append(parse_fill_gaps(step_config))
        elif name == 'resample':
            parsed_config.append(parse_resample(step_config))
        elif name == 'wiener_filter':
            parsed_config.append(parse_wiener_filter(step_config))
        elif name == 'remove_response':
            parsed_config.append(parse_remove_response(step_config))


        else:
            parsed_config.append(step_config)

    return parsed_config

def parse_rmean(config):
    """
    Parse detrend configuration and fill in defaults where necessary.

    Args:
        config (dict): User-provided taper configuration.

    Returns:
        dict: Completed detrend configuration.
    """
    template = {
        'name': 'rmean',
        'type': 'simple'}
    template.update(config)
    return template

def parse_taper(config):
    """
    Parse taper configuration and fill in defaults where necessary.

    Args:
        config (dict): User-provided taper configuration.

    Returns:
        dict: Completed taper configuration.
    """
    template = {
        'name': 'taper',
        'method': 'hann',
        'max_percentage': 0.05,
        'max_length': None,
        'side': 'both'
    }
    template.update(config)
    return template

def parse_normalize(config):
    """
    Parse normalize configuration and fill in defaults where necessary.

    Args:
        config (dict): User-provided taper configuration.

    Returns:
        dict: Completed normalize configuration.
    """
    template = {
        'name': 'normalize',
        'norm': True}
    template.update(config)
    return template

def parse_smoothing(config):
    """
        Parse smoothing configuration and fill in defaults where necessary.

        Args:
            config (dict): User-provided smoothing configuration.

        Returns:
            dict: Completed smoothing configuration.
        """

    template = {
        'name': 'smoothing',
        'method': 'gaussian',
        'time_window': 5,
        'FWHM': 0.05}

    template.update(config)

    return template

def parse_fill_gaps(config):
    """
            Parse fill_gaps configuration and fill in defaults where necessary.

            Args:
                config (dict): User-provided fill_gaps configuration.

            Returns:
                dict: Completed fill_gaps configuration.
            """

    template = {
        'name': 'fill_gaps',
        'method': 'latest'}

    template.update(config)

    return template

def parse_resample(config):
    """
            Parse resample configuration and fill in defaults where necessary.

            Args:
                config (dict): User-provided resample configuration.

            Returns:
                dict: Completed resample configuration.
            """

    template = {
        'name': 'resample',
        'sampling_rate': 10,
        'pre_filter': True}

    template.update(config)

    return template

def parse_filter(config):
    """
    Parse and normalize filter configuration into standard template.

    Args:
        config (dict): User-provided filter configuration.

    Returns:
        dict: Normalized filter configuration suitable for processing.
    """
    template_band = {
        'name': 'filter',
        'type': 'bandpass',
        'corners': 4,
        'zerophase': False}

    type = config.get('type')
    if type in ['bandpass', 'bandstop', 'lowpass', 'highpass', 'cheby1', 'cheby2', 'elliptic', 'bessel']:
        target = template_band

    else:
        raise ValueError(f"Unknown filter method: {type}")

    for key, val in config.items():
        target[key] = val

    return target

def parse_remove_response(config):
    """
    Parse and remove_response configuration into standard template.

    Args:
        config (dict): User-provided remove_response configuration.

    Returns:
        dict: remove_response configuration suitable for processing.
    """
    template = {
        'name': 'remove_response',
        'pre_filt': [0.01, 0.02, 40, 42],
        'water_level': 90,
        'units': 'VEL'}

    template.update(config)
    return template

def parse_dwt(config):
    """
    Parse whitening configuration and set defaults if not provided.

    Args:
        config (dict): User-provided whitening configuration.

    Returns:
        dict: Completed whitening configuration.
    """
    template = {
            'name': 'wavelet_denoise',
            'dwt': "sym4",
            'threshold': 0.05}
    template.update(config)
    return template


def parse_whitening(config):
    """
    Parse whitening configuration and set defaults if not provided.

    Args:
        config (dict): User-provided whitening configuration.

    Returns:
        dict: Completed whitening configuration.
    """
    template = {
        'name': 'whitening',
        'taper_edge': True,
        'freq_width': 0.05
    }
    template.update(config)
    return template

def parse_differentiate(config):
    """
        Parse differentiate configuration and set defaults if not provided.

        Args:
            config (dict): User-provided differentiate configuration.

        Returns:
            dict: Completed differentiate configuration.
        """
    template = {
        'name': 'differentiate',
        'method': 'spectral'
    }
    template.update(config)
    return template

def parse_integrate(config):
    """
        Parse integrate configuration and set defaults if not provided.

        Args:
            config (dict): User-provided integrate configuration.

        Returns:
            dict: Completed integrate configuration.
        """
    template = {
        'name': 'integrate',
        'method': 'spectral'
    }
    template.update(config)
    return template


def parse_remove_spikes(config):
    """
    Parse remove_spikes configuration, filling default for missing fields.

    Args:
        config (dict): User-provided remove_spikes configuration.

    Returns:
        dict: Normalized remove_spikes configuration.
    """
    template = {
        'name': 'remove_spikes',
        'window_size': 5,
        'sigma': 3
    }
    template.update(config)
    return template

def parse_add_noise(config):
    """
        Parse add_noise configuration, filling default for missing fields.

        Args:
            config (dict): User-provided remove_spikes configuration.

        Returns:
            dict: add_noise configuration.
    """

    template = {
        'name': 'add_noise',
        'noise_type': 'white',
        'SNR_dB': 1}

    template.update(config)
    return template

def parse_time_normalization(config):
    """
        Parse time_normalization configuration, filling default for missing fields.

        Args:
            config (dict): User-provided time_normalization configuration.

        Returns:
            dict: time_normalization configuration.
    """

    template = {
        'name': 'time_normalization',
        'method': '1bit'}

    template.update(config)
    return template

def parse_wiener_filter(config):
    """
        Parse wiener_filter configuration, filling default for missing fields.

        Args:
            config (dict): User-provided wiener_filter configuration.

        Returns:
            dict: wiener_filter configuration.
    """

    template = {
        'name': 'wiener_filter',
        'time_window': 1.0,
        'noise_power': 0}

    template.update(config)
    return template




