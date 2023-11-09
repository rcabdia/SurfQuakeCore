import os
#import unittest
from surfquakecore.real.real_parse import load_real_configuration
from tests.test_resources.real.__ini__ import test_resources_real_path


def test_load_mti_configuration():
    config_path = os.path.join(test_resources_real_path, "real_config.ini")
    real_config = load_real_configuration(config_path)
    print(real_config.to_dict())

if __name__ == '__main__':
    #unittest.main()
    test_load_mti_configuration()