#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rotate components
"""

from surfquakecore.data_processing.validators.utils import require_keys, require_type
from surfquakecore.data_processing.validators.constants import ROTATE_METHODS, ROTATE_TYPES

def validate_rotate(config):
    LQT_rotate = ['ZNE->LQT', 'LQT->ZNE']

    require_keys(config, ['method']) # not required rotate angle
    method = config['method']
    if method not in ROTATE_METHODS:
        raise ValueError(f"Unsupported rotate method: '{method}'")

    require_keys(config, ['type'])
    type = config['type']
    if type not in ROTATE_TYPES:
        raise ValueError(f"Unsupported rotate method: '{type}'")


    if method == 'FREE':
        require_keys(config, ['angle'])
        require_type(config, 'angle', float)
        if method == "FREE" and type in LQT_rotate:
            require_keys(config, ['inclination'])
            require_type(config, 'inclination', float)

    return True