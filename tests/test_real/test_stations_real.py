from obspy import read_inventory
from surfquakecore.utils import obspy_utils

inventory_path = "/Volumes/LaCie/Andorra/Downloads/meta/inv_all.xml"
working_directory = ""
inventory = read_inventory(inventory_path)
obspy_utils.ObspyUtil.realStation(inventory, working_directory)