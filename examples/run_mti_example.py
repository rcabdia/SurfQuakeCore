import os

from surfquakecore.moment_tensor.mti_parse import read_isola_log, read_isola_result
from surfquakecore.moment_tensor.sq_isola_tools.sq_bayesian_isola import BayesianIsolaCore
from surfquakecore.utils.obspy_utils import MseedUtil

def list_files_with_iversion_json(root_folder):
    iversion_json_files = []

    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename == "iversion.json":
                iversion_json_files.append(os.path.join(foldername, filename))

    return iversion_json_files

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
    # project = MseedUtil.load_project(file=project_tobe_saved)

    # build the class
    bic = BayesianIsolaCore(project=project, inventory_file=inventory_path, output_directory=output_directory,
                            save_plots=True)
    bic.run_inversion(mti_config=path_to_configfiles)
    print("Finished Inversion")
    iversion_json_files = list_files_with_iversion_json(output_directory)

    for result_file in iversion_json_files:
        result = read_isola_result(result_file)
        print(result)
        
    # example of reading log_output file
    # for r in bic.results:
    #     read_isola_log(r)
    #print(results.keys())