import os
import shutil
import unittest
import warnings
from datetime import datetime
from surfquakecore.moment_tensor.mti_parse import load_mti_configuration
from surfquakecore.moment_tensor.sq_isola_tools import BayesianIsolaCore, generate_mti_id_output
from surfquakecore.moment_tensor.structures import MomentTensorResult, MomentTensorCentroid, MomentTensorScalar
from surfquakecore.project.surf_project import SurfProject
from tests.test_resources.mti.mti_run_inversion_resources import test_inversion_resource_path


class TestBayesianIsolaCore(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        self.root_resource = test_inversion_resource_path

        self.inventory_path = os.path.join(self.root_resource, "inv_surfquakecore.xml")
        self.data_dir_path = os.path.join(self.root_resource, "waveforms")
        self.path_to_configfiles = os.path.join(self.root_resource, "configs")
        self.working_directory = os.path.join(self.root_resource, "working_directory")
        self.output_directory = os.path.join(self.root_resource, "output_directory")

        self.expect_mti_result = MomentTensorResult(
            centroid=MomentTensorCentroid(
                time=datetime(2018, 8, 21, 0, 28, 56),
                origin_shift=-0.48,
                latitude=42.72091508976003,
                longitude=-7.707582232785574,
                depth=7.663932265728514,
                vr=80.61974783417627, cn=3.253688865228685,
                mrr=-42014947574324.055,
                mtt=-69496958267000.62, mpp=111511905841324.67,
                mrt=32067686266240.53, mrp=-35174962412835.227, mtp=-155850038466389.66),
            scalar=MomentTensorScalar(
                mo=209819900677709.94,
                mw=3.5145644514075203,
                dc=52.65547866382941,
                clvd=47.34452133617057, isotropic_component=2.4822875792575555e-14,
                plane_1_strike=195.62989511207923, plane_1_dip=78.88430433605211,
                plane_1_slip_rake=3.75047357676921, plane_2_strike=104.90584246536562,
                plane_2_dip=86.31998357096703, plane_2_slip_rake=168.86104365728653)
        )

    def tearDown(self):
        try:
            shutil.rmtree(self.output_directory)
            shutil.rmtree(self.working_directory)
        except FileNotFoundError:
            pass

    def test_create_project(self):

        sp = SurfProject(self.data_dir_path)
        sp.search_files(verbose=True)
        print(sp)

    def test_run_mti_inversion(self):

        self.assertTrue(os.path.isdir(self.data_dir_path))
        sp = SurfProject(self.data_dir_path)
        sp.search_files(verbose=True)
        print(sp)
        mti_config = load_mti_configuration(os.path.join(self.path_to_configfiles, "mti_config_test.ini"))

        self.assertTrue(os.path.isfile(os.path.join(self.path_to_configfiles, "mti_config_test.ini")))

        mti_config.inversion_parameters.earth_model_file = os.path.join(self.root_resource, "Iberia_test.dat")

        bic = BayesianIsolaCore(
             project=sp,
             inventory_file=self.inventory_path,
             output_directory=self.output_directory,
             save_plots=False,
        )

        bic.run_inversion(mti_config=mti_config)

        self.assertFalse(os.path.isdir(self.working_directory))

        log_file = os.path.join(self.output_directory, generate_mti_id_output(mti_config=mti_config), "log.txt")
        self.assertTrue(os.path.isfile(log_file))

        for result_file in bic.results:
            self.assertTrue(os.path.isfile(result_file))

        for result in bic.inversion_results:
            self.assertEqual(self.expect_mti_result, result)


if __name__ == '__main__':
    unittest.main()

