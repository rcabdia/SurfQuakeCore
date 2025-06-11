#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
envelope
"""
from surfquakecore.data_processing.validators.constants import ENVELOPE_MODE
from surfquakecore.data_processing.validators.utils import require_keys, require_type


def validate_envelope(config):
    require_keys(config, ['method'])
    method = config['method']
    if config['method'] not in ENVELOPE_MODE:
        raise ValueError(f"Unsupported integrate method: {config['method']}")

    if method == "SMOOTH":
        require_type(config, 'corner_freq', float)