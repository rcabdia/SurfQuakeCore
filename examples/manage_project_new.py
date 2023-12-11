from multiprocessing import freeze_support
from surfquakecore.project.surf_project import SurfProject
import time
#path_to_data = "/Users/admin/Documents/iMacROA/SurfQuakeCore/tests/test_resources/mti/mti_run_inversion_resources/waveforms"
path_to_data ="/Volumes/LaCie/test_surfquake_core/testing_data"
path_to_project ="/Volumes/LaCie/test_surfquake_core/testing_data/surfquake_project_new.pkl"

if __name__ == '__main__':

    freeze_support()
    sp = SurfProject(path_to_data)
    #sp.search_files(starttime="2022-01-30 23:55:00", endtime="2022-02-01 00:30:00", stations="SALF,VALC", channels="HHZ")
    sp.search_files()
    #sp_original_project = copy.copy()
    sp.filter_project_keys(station="SALF|VALC|CEST")
    sp_original1 = sp.copy()
    sp_original1.filter_project_keys(station="SALF")
    sp_original2 = sp.copy()
    sp_original2.filter_project_keys(station="VALC")

    sp_join = sp_original1 + sp_original2
    print("With no filter")
    print(sp_join)
    sp_join.filter_project_time(starttime="2022-01-30 23:55:00", endtime="2022-02-01 00:30:00")
    print("With filter")
    print(sp_join)
    sp_join.save_project(path_file_to_storage=path_to_project)
    time.sleep(5)
    sp_loaded = SurfProject.load_project(path_to_project_file=path_to_project)
    print(sp_loaded)


