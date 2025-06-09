#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synch
"""

from surfquakecore.data_processing.validators.constants import SYNCH_METHODS
from surfquakecore.data_processing.validators.utils import require_keys, require_type

def validate_synch(config):

    require_keys(config, ['method'])
    if config['method'] not in SYNCH_METHODS:
        raise ValueError(f"Unsupported integrate method: {config['method']}")
