import os
from multiprocessing import freeze_support
from surfquakecore.utils.obspy_utils import MseedUtil

data_dir_path = "/Volumes/LaCie/test_surfquake_core/minimal_data"
path_to_project = "/Volumes/LaCie/test_surfquake_core/project"
project_name = 'surfquake_project_test2.pkl'
project_file_path = os.path.join(path_to_project, project_name)
if __name__ == '__main__':
    freeze_support()
    ms = MseedUtil()
    project = ms.search_files(data_dir_path, verbose=True)
    print("End of project creation, number of files ", len(project))
    # Filter project

    #project_filtered, data = ms.filter_project_keys(project, net="FR", station="CARF")
    #print(project_filtered)

    # it is possible to save the project for later use
    ms.save_project(project, project_file_path)
    # alternatively one can load the project
    #project = MseedUtil.load_project(file=project_file_path)
    #print(project)