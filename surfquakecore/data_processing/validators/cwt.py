#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cwt
"""
from surfquakecore.data_processing.validators.constants import CWT_WAVELETS
from surfquakecore.data_processing.validators.utils import require_keys

def validate_cwt(config):
    require_keys(config, ['wavelet'])
    if config['wavelet'] not in CWT_WAVELETS:
        raise ValueError(f"Unsupported integrate method: {config['mode']}")
    return True
