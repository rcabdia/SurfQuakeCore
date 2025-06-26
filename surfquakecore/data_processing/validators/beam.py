#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
beam
"""

from surfquakecore.data_processing.validators.utils import require_type

def validate_beam(config):

    require_type(config, "timewindow", float)
    require_type(config, "overlap", float)
    require_type(config, "fmin", float)
    require_type(config, "fmax", float)
    require_type(config, "slow_grid", float)
    require_type(config, "smax", float)
    require_type(config, "output_folder", str)