#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kurtosis
"""

from surfquakecore.data_processing.validators.utils import require_keys

def validate_kurtosis(config):
    require_keys(config, ['CF_decay_win'])
    require_keys(config, ['fmin'])
    require_keys(config, ['fmax'])
    return True