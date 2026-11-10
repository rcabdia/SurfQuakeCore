#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
cepstrum
"""

from surfquakecore.data_processing.validators.utils import require_keys


def validate_cepstrum(config):
    require_keys(config, ['kind'])
    require_keys(config, ['output_path'])

    return True