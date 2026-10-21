#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cross_correlate
"""
from datetime import datetime
from surfquakecore.data_processing.validators.utils import require_keys, require_type

def validate_wiener(config):
    require_keys(config, ['window_start'])
    require_keys(config, ['window_end'])
    require_type(config, "window_start", (float, int, datetime))
    require_type(config, "window_end", (float, int, datetime))

    if "ncoeff" in config:
        require_type(config, "ncoeff", int)
    if "mu" in config:
        require_type(config, "mu", float)
    if "epsilon" in config:
        require_type(config, "epsilon", float)