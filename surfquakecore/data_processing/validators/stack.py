#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stack
"""

from surfquakecore.data_processing.validators.constants import STACK_METHODS
from surfquakecore.data_processing.validators.utils import require_keys, require_type


def validate_stack(config):
    require_keys(config, ['method'])
    if config['method'] not in STACK_METHODS:
        raise ValueError(f"Unsupported integrate method: {config['method']}")

    if config["method"] == "pw" or config["method"] == "root":
        require_type(config, "order", int)
