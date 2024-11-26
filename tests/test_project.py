from multiprocessing import freeze_support
from surfquakecore.project.surf_project import SurfProject


path_to_data = "C:\\Users\\cripalalo\\Documents\\GitHub\\test_surfquake\\test_surfquake-main\\test_inputs" #"/home/sysop/Escritorio/geofisica/test_surfquake/inputs/waveforms_cut"
path_to_project = "C:\\Users\\cripalalo\\Documents\\GitHub\\test_surfquake\\test_surfquake-main\\outputs\\test2_project.pkl"#"/home/sysop/Escritorio/geofisica/test_surfquake/outputs/project/project.pkl"

if __name__ == '__main__':
    attributes = {
        'nets': 'CA, PM'
    }
    filter_time = {
        'starttime': '2022-2-1 01:30:00',
        'endtime': '2022-2-1 02:30:00'
    }
    freeze_support()
    sp = SurfProject(path_to_data)
    sp.search_files()

    sp.filter_project_keys(net="CA|PM")
    #sp.filter_time(**filter_time)
    #r = sp.load_project(path_to_project)
    #sp.search_files()
    print(sp)
    sp.save_project(path_file_to_storage=path_to_project)