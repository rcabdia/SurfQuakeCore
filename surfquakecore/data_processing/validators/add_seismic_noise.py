#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
add_seismic_noise
"""
from surfquakecore.data_processing.validators.utils import optional_type

SEISMIC_NOISE_UNITS = ['counts', 'ACC', 'VEL', 'DISP']


def validate_add_seismic_noise(config):
    optional_type(config, 'noise_level', (float, int))
    optional_type(config, 'seed', (int, type(None)))
    optional_type(config, 'units', str)
    if 'units' in config and config['units'].upper() not in [u.upper() for u in SEISMIC_NOISE_UNITS]:
        raise ValueError(f"'units' must be one of {SEISMIC_NOISE_UNITS}, got '{config['units']}'")
    return True