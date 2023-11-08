from surfquakecore.utils.obspy_utils import ObspyUtil
from obspy import read_inventory
inventory_path = "/Users/roberto/Documents/SurfQuakeCore/tests/test_resources/mti/inventories/inv_surfquakecore.xml"
working_directory = ""

inventory = read_inventory(inventory_path)
ObspyUtil.realStation(inventory, working_directory)