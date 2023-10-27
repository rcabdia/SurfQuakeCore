import os
from surfquakecore.utils.obspy_utils import MseedUtil





if __name__ == "__main__":

    data_dir_path = "/Users/roberto/Desktop/surfquakecore_test_data"
    path_to_project = "/Users/roberto/Desktop/surfquakecore_test_data/surfquake_project_test.pkl"
    project_name = 'surfquake_project_test.pkl'
    project_tobe_saved = os.path.join(data_dir_path, project_name)
    ms = MseedUtil()
    project = ms.search_files(data_dir_path)
    print("End of project creation, number of files ", len(project))

    # it is possible to save the project for later use
    #project = ms.save_project(project, project_tobe_saved)

    # alternatively one can load the project
    #project = MseedUtil.load_project(file=path_to_project)