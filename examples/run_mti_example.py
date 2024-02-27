import os
from surfquakecore.moment_tensor.mti_parse import read_isola_log, read_isola_result
from surfquakecore.moment_tensor.sq_isola_tools.sq_bayesian_isola import BayesianIsolaCore
from surfquakecore.project.surf_project import SurfProject

def list_files_with_iversion_json(root_folder):
    iversion_json_files = []

    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename == "inversion.json":
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

    # Load the Project
    project_name = "mti_project.pkl"
    path_to_project = os.path.join(path_to_project, project_name)
    sp = SurfProject(path_to_project)
    sp.search_files(verbose=True)
    print(sp)


    # Build the class
    bic = BayesianIsolaCore(project=sp, inventory_file=inventory_path, output_directory=output_directory,
                            save_plots=True)

    # Run Inversion
    bic.run_inversion(mti_config=path_to_configfiles)
    print("Finished Inversion")
    iversion_json_files = list_files_with_iversion_json(output_directory)

    for result_file in iversion_json_files:
        result = read_isola_result(result_file)
        print(result)
        

