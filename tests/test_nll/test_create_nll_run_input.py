import os
from surfquakecore.earthquake_location.run_nll import Nllcatalog, NllManager

if __name__ == "__main__":
    cwd = os.path.dirname(__file__)
    # Basic input: working_directory, inventory file path and config_file input
    working_directory = '/Users/roberto/Documents/SurfQuakeCore/examples/earthquake_locate'
    inventory_path = '/Users/roberto/Documents/SurfQuakeCore/examples/earthquake_locate/inventories/inv_surfquakecore.xml'
    path_to_configfiles = '/Users/roberto/Documents/SurfQuakeCore/examples/earthquake_locate/config/nll_config.ini'
    nll_manager = NllManager(path_to_configfiles, inventory_path, working_directory)
    #nll_manager.vel_to_grid()
    #nll_manager.grid_to_time()
    nll_manager.run_nlloc()
    #nll_catalog = Nllcatalog(working_directory)
    #nll_catalog.run_catalog(working_directory)