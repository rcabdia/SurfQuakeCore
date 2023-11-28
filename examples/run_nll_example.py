import os

from surfquakecore.earthquake_location.run_nll import NllManager

if __name__ == "__main__":
    cwd = os.path.dirname(__file__)
    ## Basic input: working_directory, invenoty file path and config_file input
    working_directory = os.path.join(cwd, "earthquake_locate")
    inventory_path = os.path.join(working_directory, "inventories", "inv_surfquakecore.xml")
    path_to_configfiles = os.path.join(working_directory, "config/nll_config.ini")
    nll_manager = NllManager(path_to_configfiles, inventory_path, working_directory)
    nll_manager.vel_to_grid()
    nll_manager.grid_to_time()
    nll_manager.run_nlloc()