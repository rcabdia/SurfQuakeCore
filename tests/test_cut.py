import surfquakecore.cli
from surfquakecore.utils.configuration_utils import parse_configuration_file
from surfquakecore.magnitudes.source_tools import ReadSource
from obspy import read
from multiprocessing import freeze_support
from surfquakecore.project.surf_project import SurfProject


if __name__ == "__main__":
    s = surfquakecore.cli.main()
    print('fin')