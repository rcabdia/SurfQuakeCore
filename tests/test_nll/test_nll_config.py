import os
from surfquakecore.earthquake_location.nll_parse import load_nll_configuration
from tests.test_resources.nll import test_resources_nll_path


def test_load_nll_configuration():
    config_path = os.path.join(test_resources_nll_path, "nll_config.ini")
    nll_config = load_nll_configuration(config_path)
    print(nll_config.to_dict())

if __name__ == '__main__':
    #unittest.main()
    test_load_nll_configuration()