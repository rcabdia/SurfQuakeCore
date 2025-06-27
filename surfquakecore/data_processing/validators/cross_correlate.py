#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cross_correlate
"""
from surfquakecore.data_processing.validators.constants import CROSS_CORRELATE_MODE
from surfquakecore.data_processing.validators.utils import require_keys, require_type


def validate_cross_correlate(config):
    require_keys(config, ['mode'])
    if config['mode'] not in CROSS_CORRELATE_MODE:
        raise ValueError(f"Unsupported integrate method: {config['mode']}")

    require_type(config, "normalize", str)
    require_type(config, "reference_idx", int)
    require_type(config, "trim", bool)

