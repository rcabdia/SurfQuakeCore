#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
spectrum
"""

from surfquakecore.data_processing.validators.constants import SPECTRUM_METHODS
from surfquakecore.data_processing.validators.utils import require_keys

def validate_spectrum(config):

    require_keys(config, ['method'])
    require_keys(config, ['output_path'])

    if config['method'] not in SPECTRUM_METHODS:
        raise ValueError(f"Unsupported integrate method: {config['method']}")

    return True