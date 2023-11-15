from configparser import ConfigParser
from dataclasses import dataclass, asdict
from surfquakecore.utils import Cast

def _read_config_file(file_path: str):
    config = ConfigParser()
    config.read(file_path)
    return config

@dataclass
class BaseDataClass:

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dto: dict):
        return Cast(dto, cls)


@dataclass
class GridConfiguration(BaseDataClass):
    latitude: float
    longitude: float
    depth: float
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    geo_transformation: str
    grid_type: str
    path_to_1d_model: str
    path_to_3d_model: str

@dataclass
class TravelTimesConfiguration(BaseDataClass):
    distance_limit: float
    grid1d: bool
    grid3d: bool
@dataclass
class LocationParameters(BaseDataClass):
    search: str
    method: str

@dataclass
class NLLConfig(BaseDataClass):

    grid_configuration: GridConfiguration
    travel_times_configuration: TravelTimesConfiguration
    location_parameters: LocationParameters



