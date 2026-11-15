#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
spectrum
"""

from surfquakecore.data_processing.validators.constants import SPECTRUM_METHODS
from surfquakecore.data_processing.validators.utils import require_keys

def validate_spectrum(config):

    require_keys(config, ['method'])
    require_keys(config, ['output_path'])

    if config['method'] not in SPECTRUM_METHODS:
        raise ValueError(f"Unsupported spectrum method: {config['method']}")

    # optional: 'fourier' (|X|·dt, unit·s, default) or 'asd' (unit/√Hz)
    if config.get('scaling', 'fourier') not in ('fourier', 'asd'):
        raise ValueError(f"Unsupported spectrum scaling: {config['scaling']} (use fourier or asd)")

    # music / ev need nsig (2 per pure tone); order is optional (default 20)
    if config['method'] in ('music', 'ev') and config.get('nsig') is None:
        raise ValueError(f"spectrum method '{config['method']}' needs nsig (2 per pure tone)")

    return True