#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cut
"""


from surfquakecore.data_processing.validators.utils import require_keys

def validate_cut(config):
    require_keys(config, ['mode'])
    return True
