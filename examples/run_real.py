import os
from surfquakecore.real.real_core import RealCore
from tests.test_resources.real.__ini__ import test_resources_real_path

# Inventory Information
inventory_path = "/Volumes/LaCie/Andorra/Downloads/meta/inv_all.xml"

# picking Output of PhaseNet
picks_path = '/Volumes/LaCie/test_surfquake_core/test_picking_new'

# Set working_directory and output
working_directory = '/Volumes/LaCie/test_surfquake_core/test_real/working_directory'
output_directory = '/Volumes/LaCie/test_surfquake_core/test_real/output_directory'

# Set path to REAL configuration
config_path = os.path.join(test_resources_real_path, "real_config.ini")
# Run association
rc = RealCore(inventory_path, config_path, picks_path, working_directory, output_directory)
rc.run_real()
print("End of Events AssociationProcess, please see for results: ", output_directory)