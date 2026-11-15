#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
structures
"""

# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: earthquake_location/structures.py
# Program: surfQuake & ISP
# Date: October 2025
# Purpose: Dataclass structures for Coincidence Trigger configuration.
# Author: Roberto Cabieces & Thiago C. Junqueira
#  Email: rcabdia@roa.es
# --------------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional

from surfquakecore.utils import BaseDataClass

@dataclass
class Kurtosis(BaseDataClass):
    CF_decay_win: float

@dataclass
class STA_LTA(BaseDataClass):
    method: str
    sta_win: float
    lta_win: float

@dataclass
class Cluster(BaseDataClass):
    method_preferred: str
    centroid_radio: float
    coincidence: int
    threshold_on: float
    threshold_off: float
    fmin: float
    fmax: float

@dataclass
class CWT(BaseDataClass):
    cycles: float
    voices_per_octave: int
    wavelet_type: str                 # cm | mh | pa
    # CF types (one, or several separated by commas -> combined per station)
    cf_type: str = "power"            # power | power_win | power_mod | entropy | centroid | kurtosis
    lowpass_freq: float = 0.15
    # window CFs
    win_before: float = 10.0
    win_after: float = 2.0
    gap: float = 0.5
    direction: str = "up"             # up | down | both
    # kurtosis
    win_kurt: float = 3.0
    # normalisation / combination
    combine: str = "max"              # max | mean | stouffer | gmean
    norm: str = "mad"                 # mad | tail
    tail_quantile: float = 0.99
    tolerance: float = 0.0
    p: float = 3.0
    final_norm: bool = False

@dataclass
class CoincidenceConfig(BaseDataClass):
    kurtosis_configuration: Kurtosis
    sta_lta_configuration: STA_LTA
    cluster_configuration: Cluster
    cwt_configuration: Optional[CWT] = None