from dataclasses import dataclass, field
from datetime import datetime
from typing import List

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
