import glob
import os
from obspy import read_inventory
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
from surfquakecore.real.real_manager import RealManager
from surfquakecore.real.real_parse import load_real_configuration
from surfquakecore.real.tt_db.taup_tt import create_tt_db
from surfquakecore.utils import obspy_utils
from surfquakecore.utils.conversion_utils import ConversionUtils


class RealCore:
    def __init__(self, metadata_file: str, parameters_file: str, picking_directory: str, working_directory: str,
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
        self.picking_directory = picking_directory
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

    def __get_lat_mean(self):

        lat_max = self.parameters['geographic_frame']['lat_ref_max']
        lat_min = self.parameters['geographic_frame']['lat_ref_min']
        latitude_center = (lat_min + lat_max) / 2
        return latitude_center

    def run_real(self):
        inventory = self._return_inventory()
        self._get_files_from_config(self.parameters_file)

        obspy_utils.ObspyUtil.realStation(inventory, self.working_directory)
        stationfile = os.path.join(self.working_directory, "station.dat")
        ttime_file = os.path.join(self.working_directory, 'ttdb.txt')
        nllinput = os.path.join(self.output_directory, "nll_input.txt")
        realout = os.path.join(self.output_directory, "phase_sel_total.txt")

        # create travel times
        self._get_real_grid()
        tt_db = create_tt_db()
        tt_db.run_tt_db(ttime=self.working_directory, dist=self.h_range,
                        depth=self.parameters['geographic_frame']['depth'], ddist=0.01, ddep=1)

        # real_handler = RealManager(pick_dir=self.picking_directory, station_file=stationfile,
        # out_data_dir=self.output_directory,
        # time_travel_table_file=ttime_file,
        # gridSearchParamHorizontalRange=self.parameters['grid_search_parameters']['horizontal_search_range'],
        # HorizontalGridSize=self.parameters['grid_search_parameters']['horizontal_search_grid_size'],
        # DepthSearchParamHorizontalRange=self.parameters['grid_search_parameters']['depth_search_range'],
        # DepthGridSize=self.parameters['grid_search_parameters']['depth_search_grid_size'],
        # EventTimeW=self.parameters['grid_search_parameters']['event_time_window'],
        # TTHorizontalRange=self.parameters['travel_time_grid_search']['horizontal_range'],
        # TTHorizontalGridSize=self.parameters['travel_time_grid_search']['horizontal_grid_resolution_size'],
        # TTDepthGridSize=self.parameters['travel_time_grid_search']['depth_grid_resolution_size'],
        # TTDepthRange=self.parameters['travel_time_grid_search']['depth_range'],
        # ThresholdPwave=self.parameters['threshold_picks']['min_num_p_wave_picks'],
        # ThresholdSwave=self.parameters['threshold_picks']['min_num_s_wave_picks'],
        # number_stations_picks=self.parameters['threshold_picks']['num_stations_recorded'])


        real_handler = RealManager(pick_dir="/Users/admin/Documents/iMacROA/SurfQuake/surfquake/data/picks",
                                   station_file="/Users/admin/Documents/iMacROA/SurfQuake/surfquake/data/station/station.dat",
        out_data_dir=self.output_directory,
        time_travel_table_file=ttime_file,
        gridSearchParamHorizontalRange=4.8,
        HorizontalGridSize=0.6,
        DepthSearchParamHorizontalRange=50,
        DepthGridSize=10,
        EventTimeW=120,
        TTHorizontalRange=5,
        TTHorizontalGridSize=0.01,
        TTDepthGridSize=10,
        TTDepthRange=50,
        ThresholdPwave=3,
        ThresholdSwave=1,
        number_stations_picks=1)


        real_handler.latitude_center = self.__get_lat_mean()
        for events_info in real_handler:
            print(events_info)
            print(events_info.events_date)

        real_handler.save()
        real_handler.compute_t_dist()
        ConversionUtils.real2nll(realout, nllinput)