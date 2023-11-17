import os
import unittest

from surfquakecore.utils.obspy_utils import MseedUtil
from tests.test_resources.mti.mti_run_inversion_resources import test_inversion_resource_path


class MyTestCase(unittest.TestCase):

    def setUp(self):
        root_resource = test_inversion_resource_path

        self.inventory_path = os.path.join(root_resource, "inv_surfquakecore.xml")
        self.data_dir_path = os.path.join(root_resource, "waveforms")
        self.path_to_configfiles = os.path.join(root_resource, "configs")
        self.working_directory = os.path.join(root_resource, "working_directory")
        self.output_directory = os.path.join(root_resource, "output_directory")

        # project_tobe_saved = os.path.join(path_to_project, project_name)
        # ms = MseedUtil()
        # project = ms.search_files(data_dir_path)
        # print("End of project creation, number of files ", len(project))
        #
        # # it is possible to save the project for later use
        # # project = ms.save_project(project, project_tobe_saved)
        #
        # # alternatively one can load the project
        # # project = MseedUtil.load_project(file=project_tobe_saved)
        #
        # # build the class
        # bic = BayesianIsolaCore(project, inventory_path, path_to_configfiles, working_directory, output_directory,
        #                         save_plots=True)
        # bic.run_mti_inversion()

    def test_run_mti_inversion(self):

        self.assertTrue(os.path.isdir(self.data_dir_path))

        project = MseedUtil().search_files(self.data_dir_path)
        self.assertIsInstance(project, dict)



if __name__ == '__main__':
    unittest.main()
