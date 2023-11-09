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
class GeographicFrame(BaseDataClass):
    lat_ref_max: float
    lat_ref_min: float
    lon_ref_max: float
    lon_ref_min: float
    depth: float

@dataclass
class GridSearch(BaseDataClass):
    horizontal_search_range: float
    depth_search_range: float
    event_time_window: float
    horizontal_search_grid_size: float
    depth_search_grid_size: float

@dataclass
class TravelTimeGridSearch(BaseDataClass):
    horizontal_range: float
    depth_range: float
    depth_grid_resolution_size: float
    horizontal_grid_resolution_size: float

@dataclass
class ThresholdPicks(BaseDataClass):
    min_num_p_wave_picks: float
    min_num_s_wave_picks: float
    num_stations_recorded: float

@dataclass
class RealConfig(BaseDataClass):

    geographic_frame: GeographicFrame
    grid_search_parameters: GridSearch
    travel_time_grid_search: TravelTimeGridSearch
    threshold_picks: ThresholdPicks
