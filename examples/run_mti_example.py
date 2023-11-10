import os
from surfquakecore.moment_tensor.read_log import read_log
from surfquakecore.moment_tensor.sq_isola_tools.sq_bayesian_isola import BayesianIsolaCore
from surfquakecore.utils.obspy_utils import MseedUtil
import warnings
warnings.filterwarnings("ignore")
if __name__ == "__main__":

    inventory_path = "/Users/roberto/Documents/SurfQuakeCore/tests/test_resources/mti/inventories/inv_surfquakecore.xml"
    data_dir_path = "/Users/roberto/Desktop/surfquakecore_test_data"
    path_to_project = "/Users/roberto/Documents/SurfQuakeCore/tests/test_resources/mti/project"
    project_name = 'surfquake_project_test.pkl'
    path_to_configfiles = '/Users/roberto/Documents/SurfQuakeCore/tests/test_resources/mti/list_earthquakes'
    working_directory = "/Volumes/LaCie/mti_surfquakecore/working_directory"
    output_directory = "/Volumes/LaCie/mti_surfquakecore/output_directory"

    project_tobe_saved = os.path.join(path_to_project, project_name)
    #ms = MseedUtil()
    #project = ms.search_files(data_dir_path)
    #print("End of project creation, number of files ", len(project))

    # it is possible to save the project for later use
    #project = ms.save_project(project, project_tobe_saved)

    # alternatively one can load the project
    project = MseedUtil.load_project(file=project_tobe_saved)

    # build the class
    bic = BayesianIsolaCore(project, inventory_path, path_to_configfiles, working_directory, output_directory,
                            save_plots=True)
    bic.run_mti_inversion()
    # example of reading output file
    #results = read_log("/Volumes/LaCie/mti_surfquakecore/output_directory/1/log.txt")
    #print(results.keys())