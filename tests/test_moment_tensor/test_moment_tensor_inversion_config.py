import os.path
import unittest
from datetime import datetime

from surfquakecore.moment_tensor.mti_parse import load_mti_configuration
from surfquakecore.moment_tensor.structures import MomentTensorInversionConfig, StationConfig, InversionParameters

from tests.test_resources.mti import test_resources_mti_path


class TestMomentTensorInversionConfig(unittest.TestCase):

    def setUp(self):
        self.expect_mti_dto = {
            'origin_date': datetime(2022, 2, 28, 2, 7, 59, 433000),
            'latitude': 42.5414, 'longitude': 1.4505, 'depth': 5.75, 'magnitude': 3.0,
            'stations': [{'name': 'TEST', 'channels': ['NNH', 'NNZ', 'NNE']}],
            'inversion_parameters': {'earth_model_file': "earthmodel/Iberia.txt", 'location_unc': 0.7,
                                     'time_unc': .2, 'depth_unc': 3.,
                                     'rupture_velocity': 2500., 'min_dist': 10.,
                                     'max_dist': 300., 'covariance': True,
                                     'deviatoric': False,
                                     'source_type': 'PointSource'
                                     },
            'signal_processing_pams': {'remove_response': True, 'freq_max': 0.15, 'freq_min': 0.02, 'rms_thresh': 5.0}
        }

    def test_config(self):
        date_str = "28/02/2022 02:07:59.433"
        origin_date = datetime.strptime(date_str, '%d/%m/%Y %H:%M:%S.%f')
        # still implementing test
        mti_config = MomentTensorInversionConfig(
            origin_date=origin_date,
            latitude=42.5414,
            longitude=1.4505,
            depth=5.75,
            magnitude=3.0,
            stations=[StationConfig(name="TEST", channels=["NNH", "NNZ", "NNE"])],
            inversion_parameters=InversionParameters(
                earth_model_file="earthmodel/Iberia.txt",
                location_unc=0.7,
                time_unc=.2,
                depth_unc=3.,
                rupture_velocity=2500.,
                min_dist=10.,
                max_dist=300.,
            ),
        )
        self.assertEqual(mti_config.to_dict(), self.expect_mti_dto)

    def test_load_mti_configuration(self):
        config_path = os.path.join(test_resources_mti_path, "mti_config.ini")
        self.assertTrue(os.path.isfile(config_path))
        mti_config = load_mti_configuration(config_path)

        self.expect_mti_dto["stations"] = [
            {'name': "MAHO", 'channels': ["HHZ", "HHN", "HHE"]},
            {'name': "WAIT", 'channels': ["BH*"]},
            {'name': "EVO", 'channels': ["*"]},
        ]
        self.expect_mti_dto["inversion_parameters"]["source_type"] = "Triangle"
        self.expect_mti_dto["signal_processing_pams"]["rms_thresh"] = 10.

        self.assertEqual(mti_config.to_dict(), self.expect_mti_dto)


if __name__ == '__main__':
    unittest.main()
