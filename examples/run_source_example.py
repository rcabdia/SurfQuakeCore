from surfquakecore.magnitudes.run_magnitudes import Automag
from surfquakecore.utils.obspy_utils import MseedUtil
import os

if __name__ == "__main__":

    cwd = os.path.dirname(__file__)
    ## Project definition ##
    path_to_project = "source_estimations/data"
    project_tobe_saved = os.path.join(path_to_project, "surfquake_project_test.pkl")
    print("project:", project_tobe_saved)
    ms = MseedUtil()
    #project = ms.search_files(path_to_project)
    #print("End of project creation, number of files ", len(project))

    # it is possible to save the project for later use
    #project = ms.save_project(project, project_tobe_saved)

    # alternatively one can load the project
    project = MseedUtil.load_project(file=project_tobe_saved)
    print(project)
    ## Basic input: working_directory, invenoty file path and config_file input
    working_directory = os.path.join(cwd, "source_estimations")

    # inventory path must be placed inside config_file
    inventory_path = os.path.join(working_directory, "inventories", "inv_surfquakecore.xml")
    path_to_configfiles = os.path.join(working_directory, "config/source_spec.conf")
    locations_directory = os.path.join(working_directory, "locations")
    output_directory = os.path.join(working_directory, "output")
    mg = Automag(project, locations_directory, inventory_path, path_to_configfiles, output_directory, "regional")
    mg.estimate_source_parameters()