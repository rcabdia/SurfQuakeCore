import os.path
import unittest

from surfquakecore.moment_tensor.mti_parse import read_isola_log
from tests.test_resources.mti import test_resources_mti_path


class TestMtiParse(unittest.TestCase):

    def setUp(self):
        self.isola_log_file = os.path.join(test_resources_mti_path, "log.txt")

    def test_read_isola_log(self):
        self.assertTrue(os.path.isfile(self.isola_log_file))
        r = read_isola_log(self.isola_log_file)




if __name__ == '__main__':
    unittest.main()
