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
class CoincidenceConfig(BaseDataClass):

    kurtosis_configuration: Kurtosis
    sta_lta_configuration: STA_LTA
    cluster_configuration: Cluster
