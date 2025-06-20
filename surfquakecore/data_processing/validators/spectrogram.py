#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
spectrogram
"""

from surfquakecore.data_processing.validators.utils import require_keys

def validate_spectrogram(config):
    require_keys(config, ['win'])
    require_keys(config, ['output_path'])
    # optional overlap, fmin, fmax
    return True