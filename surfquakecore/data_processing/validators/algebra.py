#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
algebra
"""

from surfquakecore.data_processing.validators.utils import require_keys, _validate_equation, require_type

def validate_algebra(config: dict) -> bool:
    """
    Validate the algebra processing config.

    Expected keys:
      - equation  (str)   : math expression referencing traces and ALGEBRA_FUNCTIONS/CONSTANTS
      - trim      (bool)  : whether to trim traces to common time window before applying algebra
      - fill_gaps (bool)  : whether to fill gaps before applying algebra
      - resample  (int|float) : target sampling rate; must be positive
    """
    require_keys(config, ['expression', 'trim', 'fill_gaps', 'resample'])

    _validate_equation(config['expression'])

    require_type(config, 'trim', bool)
    require_type(config, 'fill_gaps', bool)

    require_type(config, 'resample', (int, float))
    if config['resample'] <= 0:
        raise ValueError("'resample' must be a positive number.")

    return True
