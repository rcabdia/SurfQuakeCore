#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
config_parser
"""

from surfquakecore.data_processing.validators.constants import ANALYSIS_KEYS
from surfquakecore.data_processing.validators import validate_step


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
        'corners': 4,
        'zerophase': False
    }
    template_fir = {
        'name': 'filter',
        'winlen': 2048
    }
    template_cheby = {
        'name': 'filter',
        'maxorder': 12,
        'ba': False,
        'freq_passband': False
    }

    method = config.get('method')
    if method in ['bandpass', 'bandstop', 'lowpass', 'highpass', 'remez_fir']:
        target = template_band
    elif method == 'lowpass_fir':
        target = template_fir
    elif method == 'lowpass_cheby_2':
        target = template_cheby
    else:
        raise ValueError(f"Unknown filter method: {method}")

    for key, val in config.items():
        target[key] = val

    return target


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
        'n': 3
    }
    template.update(config)
    return template


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

        if name == 'taper':
            parsed_config.append(parse_taper(step_config))
        elif name == 'filter':
            parsed_config.append(parse_filter(step_config))
        elif name == 'whitening':
            parsed_config.append(parse_whitening(step_config))
        elif name == 'remove_spikes':
            parsed_config.append(parse_remove_spikes(step_config))
        else:
            parsed_config.append(step_config)

    return parsed_config
