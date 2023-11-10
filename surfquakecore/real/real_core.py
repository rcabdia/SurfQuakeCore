import glob
import os
from obspy import read_inventory
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
from surfquakecore.real.real_parse import load_real_configuration
from surfquakecore.real.tt_db.taup_tt import create_tt_db
from surfquakecore.utils import obspy_utils

class RealCore:
    def __init__(self, metadata_file: str, parameters_file: str, working_directory: str,
                 output_directory: str):

        """
        ----------
        Parameters
        ----------
        metadata str: Path to the inventory information of stations coordinates and instrument description
        parameters_file: .init file with the full configuration to run real
        """

        self.metadata_file = metadata_file
        self.parameters_file = parameters_file
        self.working_directory = working_directory
        self.output_directory = output_directory
        self._start_folder_tree()


    def _start_folder_tree(self):
        if not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
        else:
            files = glob.glob(self.working_directory + "/*")
            for f in files:
                os.remove(f)
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        else:
            files = glob.glob(self.output_directory + "/*")
            for f in files:
                os.remove(f)

    def _return_inventory(self):
        inv = {}
        inv = read_inventory(self.metadata_file)
        return inv

    def _get_files_from_config(self, real_config_path):
        parameters = None
        real_config = load_real_configuration(real_config_path)
        self.parameters = real_config.to_dict()

    def _get_real_grid(self):

        lon_min = self.parameters['geographic_frame']['lon_ref_min']
        lon_max = self.parameters['geographic_frame']['lon_ref_max']
        lat_max = self.parameters['geographic_frame']['lat_ref_max']
        lat_min = self.parameters['geographic_frame']['lat_ref_min']
        self.latitude_center = (lat_min + lat_max) / 2
        self.longitude_center = (lon_min + lon_max) / 2
        distance, az1, az2 = gps2dist_azimuth(self.latitude_center, self.longitude_center, lat_max, lon_max)
        self.h_range = kilometers2degrees(distance*0.001)

    def run_real(self):
        inventory = self._return_inventory()
        self._get_files_from_config(self.parameters_file)

        obspy_utils.ObspyUtil.realStation(inventory, self.working_directory)

        # create travel times
        self._get_real_grid()
        tt_db = create_tt_db()
        tt_db.run_tt_db(ttime=self.working_directory, dist=self.h_range,
                        depth=self.parameters['geographic_frame']['depth'], ddist=0.01, ddep=1)