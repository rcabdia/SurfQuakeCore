#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cwt
"""
from surfquakecore.data_processing.validators.constants import CWT_WAVELETS
from surfquakecore.data_processing.validators.utils import require_keys


def _is_number(value):
    """True for int or float, but not bool (YAML 'True' is a bool, and bool is a subclass of int)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_cwt(config):
    """
    Validate the 'cwt' (Continuous Wavelet Transform) step.

    Required keys
      wavelet     : 'cm' (Complex Morlet), 'mh' (Mexican Hat) or 'pa' (Paul)
      param       : main wavelet parameter, > 0
                      cm -> number of cycles (e.g. 6.0)
                      pa -> order m, a positive whole number (e.g. 4 or 6)
                      mh -> not used by the wavelet shape, any positive value
      output_path : folder where the .cwt files are written

    Optional keys
      fmin, fmax        : analysis band in Hz, 0 < fmin < fmax.
                          fmax must also be below the Nyquist frequency of the traces,
                          which is only known when the data are read.
      voices_per_octave : frequencies per octave, positive integer (default 16)
    """
    require_keys(config, ['wavelet', 'param', 'output_path'])

    # --- wavelet
    wavelet = config['wavelet']
    if wavelet not in CWT_WAVELETS:
        raise ValueError(f"cwt: unsupported wavelet '{wavelet}'. "
                         f"Options: {', '.join(CWT_WAVELETS)} (cm = Complex Morlet, mh = Mexican Hat, pa = Paul)")

    # --- param
    param = config['param']
    if not _is_number(param) or param <= 0:
        raise ValueError(f"cwt: 'param' must be a positive number, got {param!r}")
    if wavelet == 'pa' and float(param) != int(param):
        raise ValueError(f"cwt: with the Paul wavelet 'param' is the order m and must be a whole "
                         f"number (e.g. 4 or 6), got {param!r}")

    # --- output_path
    if not isinstance(config['output_path'], str) or not config['output_path'].strip():
        raise ValueError("cwt: 'output_path' must be a non-empty folder path")

    # --- optional frequency band
    for key in ('fmin', 'fmax'):
        if config.get(key) is not None:
            if not _is_number(config[key]) or config[key] <= 0:
                raise ValueError(f"cwt: '{key}' must be a positive number in Hz, got {config[key]!r}")

    fmin, fmax = config.get('fmin'), config.get('fmax')
    if fmin is not None and fmax is not None and fmin >= fmax:
        raise ValueError(f"cwt: 'fmin' ({fmin}) must be smaller than 'fmax' ({fmax})")

    # --- optional voices_per_octave
    if config.get('voices_per_octave') is not None:
        vpo = config['voices_per_octave']
        if not _is_number(vpo) or vpo <= 0 or float(vpo) != int(vpo):
            raise ValueError(f"cwt: 'voices_per_octave' must be a positive whole number "
                             f"(8 coarse, 16 standard, 32 fine), got {vpo!r}")

    return True
