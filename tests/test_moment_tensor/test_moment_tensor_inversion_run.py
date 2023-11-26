import os
import shutil
import unittest

from surfquakecore.moment_tensor.sq_isola_tools.sq_bayesian_isola import BayesianIsolaCore
from surfquakecore.utils.obspy_utils import MseedUtil
from tests.test_resources.mti.mti_run_inversion_resources import test_inversion_resource_path


class TestBayesianIsolaCore(unittest.TestCase):

    def setUp(self):
        root_resource = test_inversion_resource_path

        self.inventory_path = os.path.join(root_resource, "inv_surfquakecore.xml")
        self.data_dir_path = os.path.join(root_resource, "waveforms")
        self.path_to_configfiles = os.path.join(root_resource, "configs")
        self.working_directory = os.path.join(root_resource, "working_directory")
        self.output_directory = os.path.join(root_resource, "output_directory")

    def tearDown(self):
        try:
            shutil.rmtree(self.output_directory)
            shutil.rmtree(self.working_directory)
        except FileNotFoundError:
            pass

    def test_create_project(self):
        project = MseedUtil().search_files(self.data_dir_path)
        print(project)

        project = MseedUtil()._create_project(self.data_dir_path)
        print(project)

    def test_run_mti_inversion(self):

        self.assertTrue(os.path.isdir(self.data_dir_path))

        project = MseedUtil().search_files(self.data_dir_path)
        self.assertIsInstance(project, dict)

        bic = BayesianIsolaCore(project, self.inventory_path, self.path_to_configfiles,
                                self.working_directory, self.output_directory,
                                save_plots=False)
        bic.run_mti_inversion()

        log_file = os.path.join(self.output_directory, "0", "log.txt")
        self.assertTrue(os.path.isfile(log_file))


if __name__ == '__main__':
    unittest.main()
