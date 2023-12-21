import os
import unittest
from surfquakecore.real.real_parse import load_real_configuration
from surfquakecore.real.structures import RealConfig, GeographicFrame, GridSearch, ThresholdPicks, TravelTimeGridSearch
from tests.test_resources.real.__ini__ import test_resources_real_path

class TestRealConfig(unittest.TestCase):
    def test_load_real_config_ini(self):
        config_path = os.path.join(test_resources_real_path, "real_config.ini")
        real_config = load_real_configuration(config_path)
        #print(real_config.to_dict())

    def test_load_real_configuration(self):
        # loading the class
       real_config = RealConfig(
            geographic_frame=GeographicFrame(
                lat_ref_max=43.0000,
                lon_ref_max=2.2000,
                lat_ref_min=42.0000,
                lon_ref_min=0.8000,
                depth=20.00
            ),
            grid_search_parameters=GridSearch(
                horizontal_search_range= 4.80,
                depth_search_range=50.00,
                event_time_window=120.00,
                horizontal_search_grid_size= 0.60,
                depth_search_grid_size=10.00),
            travel_time_grid_search=TravelTimeGridSearch(
                horizontal_range=5.00,
                depth_range=50.00,
                depth_grid_resolution_size=2.00,
                horizontal_grid_resolution_size=0.01),
            threshold_picks=ThresholdPicks(
                min_num_p_wave_picks=3,
                min_num_s_wave_picks=1,
                num_stations_recorded=1)
        )

       return  real_config

if __name__ == '__main__':
    unittest.main()
    #rc = test_load_real_configuration()
    #print(rc)