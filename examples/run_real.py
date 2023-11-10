import os

from surfquakecore.real.real_core import RealCore
from tests.test_resources.real.__ini__ import test_resources_real_path

inventory_path = "/Volumes/LaCie/Andorra/Downloads/meta/inv_all.xml"
picks_path = '/Volumes/LaCie/test_surfquake_core/test_picking'
working_directory = '/Volumes/LaCie/test_surfquake_core/test_real/working_directory'
output_directory = '/Volumes/LaCie/test_surfquake_core/test_real/output_directory'
config_path = os.path.join(test_resources_real_path, "real_config.ini")

rc = RealCore(inventory_path, config_path, working_directory, output_directory)
rc.run_real()
print("End_Process")