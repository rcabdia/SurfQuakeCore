#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
cross_spectrum (power, mscoherence and transfer function)
"""

from surfquakecore.data_processing.validators.constants import CROSS_SPECTRUM_METHODS
from surfquakecore.data_processing.validators.utils import require_keys, require_type

def validate_cross_spectrum(config):

    if config['method'] not in CROSS_SPECTRUM_METHODS:
        raise ValueError(f"Unsupported integrate method: {config['method']}")

    require_keys(config, ['method'])
    require_keys(config, ['win'])
    require_keys(config, ['overlap_percent'])
    require_keys(config,['reference_idx'])
    require_keys(config, ['output_path'])

    require_type(config, "reference_idx", int)
    require_type(config, "method", str)
    require_type(config, "win", float)
    require_type(config, "overlap_percent", float)
    require_type(config, "output_path", str)