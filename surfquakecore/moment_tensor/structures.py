import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List

from surfquakecore.utils import Cast


@dataclass
class BaseDataClass:

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dto: dict):
        return Cast(dto, cls)


@dataclass
class SignalProcessingParameters(BaseDataClass):
    remove_response: bool = True
    freq_max: float = 0.15
    freq_min: float = 0.02
    rms_thresh: float = 5.0

    def __post_init__(self):

        if self.freq_min >= self.freq_max:
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
    covariance: bool = True
    deviatoric: bool = False
    source_type: str = "PointSource"


@dataclass
class StationConfig(BaseDataClass):
    name: str
    channels: List[str]

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
    depth: float  # km
    magnitude: float
    stations: List[StationConfig]
    inversion_parameters: InversionParameters
    signal_processing_pams: SignalProcessingParameters = field(default_factory=SignalProcessingParameters)

