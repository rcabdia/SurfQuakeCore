import surfquakecore.cli
from surfquakecore.utils.configuration_utils import parse_configuration_file
from surfquakecore.magnitudes.source_tools import ReadSource
from obspy import read
from multiprocessing import freeze_support
from surfquakecore.project.surf_project import SurfProject


if __name__ == "__main__":
    action = "project"

    #cfg =
    #rs = ReadSource("/home/sysop/Escritorio/geofisica/SurfQuakeCore/")
    #summary = rs.read_file('/home/sysop/Escritorio/geofisica/SurfQuakeCore/analysis_config.yaml')
    #rs.write_summary(summary, summary_path_file)
    #if summary['Analysis']:
    #    print('config')
    #    r = summary['Analysis']
    #    print(r)
    #cfg = parse_configuration_file("/home/sysop/Escritorio/geofisica/SurfQuakeCore/analysis_config.ini")


    s = surfquakecore.cli.main()
    print('fin')
