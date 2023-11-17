import os

from surfquakecore.moment_tensor.sq_isola_tools.sq_bayesian_isola import BayesianIsolaCore
from surfquakecore.utils.obspy_utils import MseedUtil

if __name__ == "__main__":
    cwd = os.path.dirname(__file__)
    resource_root = os.path.join(cwd, "mti")
    inventory_path = os.path.join(resource_root, "inventories", "inv_surfquakecore.xml")
    data_dir_path = os.path.join(resource_root, "waveforms")
    path_to_project = os.path.join(resource_root, "project")
    path_to_configfiles = os.path.join(resource_root, "list_earthquakes")
    working_directory = os.path.join(resource_root, "working_directory")
    output_directory = os.path.join(resource_root, "output_directory")

    project_tobe_saved = os.path.join(path_to_project, "surfquake_project_test.pkl")
    print("project:", project_tobe_saved)
    ms = MseedUtil()
    project = ms.search_files(data_dir_path)
    print("End of project creation, number of files ", len(project))

    # it is possible to save the project for later use
    #project = ms.save_project(project, project_tobe_saved)

    # alternatively one can load the project
    #project = MseedUtil.load_project(file=project_tobe_saved)

    # build the class
    bic = BayesianIsolaCore(project, inventory_path, path_to_configfiles, working_directory, output_directory,
                            save_plots=True)
    bic.run_mti_inversion()
    # example of reading output file
    #results = read_log("/Volumes/LaCie/mti_surfquakecore/output_directory/1/log.txt")
    #print(results.keys())