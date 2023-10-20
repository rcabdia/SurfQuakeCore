from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass
class SignalProcessingParameters:
    remove_response: bool
    freq_max: float
    freq_min: float


@dataclass
class InversionParameters:
    earth_model: str
    location_unc: float
    time_unc: float
    deviatoric: bool  # rename this?? What deviatoric means?
    depth_unc: float
    covariance: bool
    rupture_velocity: float
    source_type: str
    min_dist: float
    max_dist: float


@dataclass
class Station:
    name: str
    channels: List[str]

    def __post_init__(self):
        if not self.channels and len(self.channels) > 3:
            raise AttributeError(f"channels must contain a maximum of 3 values")


@dataclass
class MomentTensorInversionConfig:

    origin_date: datetime
    latitude: float
    longitude: float
    depth: float  # km
    magnitude: float
    stations: List[Station]
    inversion_parameters: InversionParameters
    signal_processing_pams: SignalProcessingParameters

