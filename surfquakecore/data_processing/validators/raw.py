#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
raw
"""
from surfquakecore.data_processing.validators.utils import require_keys

def validate_raw(config):
    require_keys(config, ['factor'])
    require_keys(config, ['integers'])
    return True