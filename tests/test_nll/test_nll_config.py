import os
import unittest
from surfquakecore.earthquake_location.nll_parse import load_nll_configuration
from surfquakecore.earthquake_location.structures import NLLConfig, GridConfiguration, TravelTimesConfiguration, \
    LocationParameters
from tests.test_resources.nll import test_resources_nll_path


class TestNLLConfig(unittest.TestCase):
    def test_config(self):
        nllconfig = NLLConfig(
            grid_configuration=GridConfiguration(
                latitude=42.0,
                longitude=0.5,
                depth=-3,
                x=350,
                y=350,
                z=100,
                dx=1,
                dy=1,
                dz=1,
                geo_transformation="SIMPLE",
                grid_type="SLOW_LEN",
                path_to_1d_model="./SurfQuakeCore/examples/earthquake_locate/nll_picks/nll_input.txt",
                path_to_3d_model="NONE",
                path_to_picks="./SurfQuakeCore/examples/earthquake_locate/nll_picks/nll_input.txt",
                p_wave_type=True,
                s_wave_type=True,
                model_1D=True,
                model_3D=False),
            travel_times_configuration=TravelTimesConfiguration(
                distance_limit=400,
                grid1d=True,
                grid3d=False),
            location_parameters=LocationParameters(
                search="OCT-TREE",
                method="GAU_ANALYTIC"))

        print("NLLConfig ", "grid_configuration: ", nllconfig.grid_configuration, nllconfig.travel_times_configuration,
              nllconfig.location_parameters)

    def test_load_nll_configuration(self):
         config_path = os.path.join(test_resources_nll_path, "nll_config.ini")
         nll_config = load_nll_configuration(config_path)

         print(nll_config.to_dict())

if __name__ == '__main__':
    unittest.main()
    #test_load_nll_configuration()
