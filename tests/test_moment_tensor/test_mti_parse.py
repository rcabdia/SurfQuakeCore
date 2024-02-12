import os.path
import unittest
from datetime import datetime

from surfquakecore.moment_tensor.mti_parse import read_isola_result
from surfquakecore.moment_tensor.structures import MomentTensorResult, MomentTensorCentroid, MomentTensorScalar
from tests.test_resources.mti import test_resources_mti_path


class TestMtiParse(unittest.TestCase):

    def setUp(self):
        self.isola_result_file = os.path.join(test_resources_mti_path, "inversion.json")

        self.expect = MomentTensorResult(
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

    # def test_read_isola_log(self):
    #     self.assertTrue(os.path.isfile(self.isola_result_file))
    #     r = read_isola_result(self.isola_result_file)
    #     self.assertEqual(self.expect, r)


if __name__ == '__main__':
    unittest.main()
