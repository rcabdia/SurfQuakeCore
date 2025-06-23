#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
entropy
"""
from surfquakecore.data_processing.validators.utils import require_keys

def validate_entropy(config):

    require_keys(config, ['win'])

    return True


