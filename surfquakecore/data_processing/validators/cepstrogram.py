#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cepstrogram
"""

from surfquakecore.data_processing.validators.utils import require_keys

def validate_cepstrogram(config):

    require_keys(config, ['win'])
    require_keys(config, ['overlap_percent'])
    require_keys(config, ['kind'])
    require_keys(config, ['output_path'])
    # optional qmin=0.0, qmax=None

    return True