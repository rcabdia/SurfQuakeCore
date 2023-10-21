import unittest
from datetime import datetime

from surfquakecore.moment_tensor.structures import MomentTensorInversionConfig


class TestMomentTensorInversionConfig(unittest.TestCase):

    def test_config(self):
        date_str = "28/02/2022 02:07:59.433"
        origin_date = datetime.strptime(date_str, '%d/%m/%Y %H:%M:%S.%f')
        # still implementing test
        MomentTensorInversionConfig(
            origin_date=origin_date,
            latitude=42.5414,
            longitude=1.4505,
            depth=5.75,
            magnitude=3.0,
            stations=[],
            inversion_parameters=None,
            signal_processing_pams=None,
        )


if __name__ == '__main__':
    unittest.main()
