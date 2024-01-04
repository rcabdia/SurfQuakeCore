from surfquakecore.magnitudes.run_magnitudes import Automag
from surfquakecore.magnitudes.source_tools import ReadSource
from surfquakecore.project.surf_project import SurfProject
import os

if __name__ == "__main__":

    cwd = os.path.dirname(__file__)
    ## Project definition ##
    path_to_project = "/Volumes/LaCie/test_surfquake_core/testing_data"
    project_path_file = os.path.join(path_to_project, "surfquake_project_new.pkl")
    print("project:", project_path_file)
    #project = ms.search_files(path_to_project)
    #print("End of project creation, number of files ", len(project))

    # it is possible to save the project for later use
    #project = ms.save_project(project, project_tobe_saved)

    # alternatively one can load the project
    sp_loaded = SurfProject.load_project(path_to_project_file=project_path_file)
    print(sp_loaded)

    ## Basic input: working_directory, invenoty file path and config_file input
    working_directory = os.path.join(cwd, "source_estimations")

    # inventory path must be placed inside config_file
    inventory_path = os.path.join(working_directory, "inventories", "inv_surfquakecore.xml")
    path_to_configfiles = os.path.join(working_directory, "config/source_spec.conf")
    locations_directory = os.path.join(working_directory, "locations")
    output_directory = os.path.join(working_directory, "output")
    summary_path = '/Users/roberto/Documents/SurfQuakeCore/examples/source_estimations/source_summary'

    # Running stage
    mg = Automag(sp_loaded.project, locations_directory, inventory_path, path_to_configfiles, output_directory, "regional")
    mg.estimate_source_parameters()

    # Now we can read the output and even write a txt summarizing the results
    rs = ReadSource(output_directory)
    summary = rs.generate_source_summary()
    rs.write_summary(summary, summary_path)