#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
snr
"""

from surfquakecore.data_processing.validators.utils import require_keys

def validate_snr(config):
    require_keys(config, ['method'])
    require_keys(config, ['sign_win'])
    require_keys(config, ['noise_win'])
    return True

