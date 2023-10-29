import os
from obspy import UTCDateTime, Stream, read_inventory

from surfquakecore.moment_tensor.mti_parse import load_mti_configuration
from surfquakecore.moment_tensor.sq_isola_tools import BayesISOLA
from surfquakecore.moment_tensor.sq_isola_tools.mti_utilities import MTIManager
from surfquakecore.utils.obspy_utils import MseedUtil


class bayesian_isola_core:
    def __init__(self, project: dict, metadata_file: str, parameters_folder: str, working_directory: str,
                 ouput_directory: str):

        """
        ----------
        Parameters
        ----------
        project dict: Information of seismogram data files available
        metadata str: Path to the inventory information of stations coordinates and instrument description
        parameters: dictionary generated from the dataclass
        """

        self.metadata_file = metadata_file
        self.project = project
        self.parameters = parameters_folder
        self.cpuCount = os.cpu_count() - 1
        self.working_directory = working_directory
        self.ouput_directory = ouput_directory

    def _get_path_files(self, project, date, stations_list, channel_list):
        files_path = []
        ms = MseedUtil()
        try:
            files_path = ms.get_now_files(project, date, stations_list, channel_list)
        except:
            print("No files to be readed in the project")
        return files_path

    def _return_inventory(self):
        inv = {}
        inv = read_inventory(self.metadata_file)
        return inv

    def _load_parameters(self, config_path):
        mti_config = load_mti_configuration(config_path)
        return mti_config

    def run_mti_inversion(self):
        """
        This method should be to loop over config files and run the inversion.
        Previously it is needed to load the project and metadata.
        """
        pass

    def _run_inversion(self, i):

        self.st = []
        if not os.path.exists(self.ouput_directory):
            os.makedirs(self.ouput_directory)

        output_directory_local = os.path.join(self.ouput_directory, str(i))

    def test_running(self):
        ms = MseedUtil()
        files_list = []
        list_of_earthquakes = ms.list_folder_files(self.parameters)
        for earthquake in list_of_earthquakes:
            try:
                mti_config = self._load_parameters(earthquake)
            except:
                mti_config = None
            if mti_config is not None:
                parameters = mti_config.to_dict()
                print(parameters)
                for station in parameters['stations']:
                    channels = '|'.join(map(str, station['channels']))
                    files = self._get_path_files(self.project, parameters['origin_date'], station['name'], channels)
                    files_list += files

