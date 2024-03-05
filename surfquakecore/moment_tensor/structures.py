# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: moment_tensor/structures.py
# Program: surfQuake & ISP
# Date: January 2024
# Purpose: Dataclass structures for MTI configuration.
# Author: Roberto Cabieces & Thiago C. Junqueira
#  Email: rcabdia@roa.es
# --------------------------------------------------------------------


from dataclasses import dataclass, field
from datetime import datetime
from surfquakecore.utils import BaseDataClass


@dataclass
class SignalProcessingParameters(BaseDataClass):
    remove_response: bool = True
    max_freq: float = 0.15
    min_freq: float = 0.02
    rms_thresh: float = 5.0

    def __post_init__(self):

        if self.min_freq >= self.max_freq:
            raise ValueError(f"freq_min >= freq_max. Minimum frequency cannot be bigger than maximum.")


@dataclass
class InversionParameters(BaseDataClass):
    earth_model_file: str
    location_unc: float
    time_unc: float
    depth_unc: float
    rupture_velocity: float
    min_dist: float
    max_dist: float
    max_number_stations: int
    source_duration: float
    covariance: bool = True
    deviatoric: bool = False
    source_type: str = "Heaviside"


@dataclass
class StationConfig(BaseDataClass):
    name: str
    channels: list[str]

    def validate_channels(self):
        if not self.channels or len(self.channels) > 3:
            raise AttributeError(f"Channels must contain a maximum of 3 values")

        for ch in self.channels:
            if len(ch) > 3:
                raise AttributeError(f"Channel name: {ch} must contain a maximum of 3 characters")

    def __post_init__(self):
        self.validate_channels()


@dataclass
class MomentTensorInversionConfig(BaseDataClass):

    origin_date: datetime
    latitude: float
    longitude: float
    depth_km: float  # km
    magnitude: float
    stations: list[StationConfig]
    inversion_parameters: InversionParameters
    signal_processing_parameters: SignalProcessingParameters = field(default_factory=SignalProcessingParameters)


@dataclass
class MomentTensorCentroid(BaseDataClass):
    time: datetime = field(default_factory=datetime.now)
    origin_shift: float = 0.
    latitude: float = 0.
    longitude: float = 0.
    depth: float = 0.
    vr: float = 0.
    cn: float = 0.
    mrr: float = 0.
    mtt: float = 0.
    mpp: float = 0.
    mrt: float = 0.
    mrp: float = 0.
    mtp: float = 0.
    rupture_length: float = 0.


@dataclass
class MomentTensorScalar(BaseDataClass):
    mo: float = 0.
    mw: float = 0.
    dc: float = 0.
    clvd: float = 0.
    isotropic_component: float = 0.
    plane_1_strike: float = 0.
    plane_1_dip: float = 0.
    plane_1_slip_rake: float = 0.
    plane_2_strike: float = 0.
    plane_2_dip: float = 0.
    plane_2_slip_rake: float = 0.


@dataclass
class MomentTensorResult(BaseDataClass):
    centroid: MomentTensorCentroid = field(default_factory=MomentTensorCentroid)
    scalar: MomentTensorScalar = field(default_factory=MomentTensorScalar)
