# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: real_core.py
# Program: surfQuake & ISP
# Date: January 2024
# Purpose: Associator Core
# Author: Roberto Cabieces & Thiago C. Junqueira
#  Email: rcabdia@roa.es
# --------------------------------------------------------------------

import os
from typing import Union
from obspy import read_inventory
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
from surfquakecore.real.real_manager import RealManager
from surfquakecore.real.real_parse import load_real_configuration
from surfquakecore.real.structures import RealConfig
from surfquakecore.real.tt_db.taup_tt import create_tt_db
from surfquakecore.utils import obspy_utils
from surfquakecore.utils.conversion_utils import ConversionUtils


class RealCore:
    def __init__(self, metadata_file: str, real_config: Union[str, RealConfig], picking_directory: str, working_directory: str,
                 output_directory: str):

        """
        ----------
        Parameters
        ----------
        metadata_file str: Path to the inventory information of stations coordinates and instrument description
        real_config: Either the path to a real_config.ini or a RealConfig object.
        picking_directory str: Root path to the folder wher picks P and S wave arrival time picks are storage
        working_directory str: Root path to the folder that the associator uses to save intermediate files sucha as travel-times.
        """

        if isinstance(real_config, str) and os.path.isfile(real_config):
            self.__get_real_config(real_config)
        elif isinstance(real_config, RealConfig):
            self.real_config = real_config
        else:
            raise ValueError(f"mti_config {real_config} is not valid. It must be either a "
                             f" valid real_config.ini file or a RealConfig instance.")

        self.metadata_file = metadata_file
        self.picking_directory = picking_directory
        self.working_directory = working_directory
        self.output_directory = output_directory
        self._start_folder_tree()

    def __get_real_config(self, config_file_path):

        self.real_config: RealConfig = load_real_configuration(config_file_path)

    def _start_folder_tree(self):

        if os.path.isdir(self.working_directory) and os.path.isdir(self.output_directory):
            pass
        else:
            try:
                os.makedirs(self.working_directory)
                os.makedirs(self.output_directory)
            except Exception as error:
                print("An exception occurred:", error)


    def _return_inventory(self):
        inv = {}
        inv = read_inventory(self.metadata_file)
        return inv

    def _get_real_grid(self):

        lon_min = self.real_config.geographic_frame.lon_ref_min
        lon_max = self.real_config.geographic_frame.lon_ref_max
        lat_max = self.real_config.geographic_frame.lat_ref_max
        lat_min = self.real_config.geographic_frame.lat_ref_min
        self.latitude_center = (lat_min + lat_max) / 2
        self.longitude_center = (lon_min + lon_max) / 2
        distance, az1, az2 = gps2dist_azimuth(self.latitude_center, self.longitude_center, lat_max, lon_max)
        self.h_range = kilometers2degrees(distance * 0.001)

    def __get_lat_mean(self):

        lat_max = self.real_config.geographic_frame.lat_ref_max
        lat_min = self.real_config.geographic_frame.lat_ref_min
        latitude_center = (lat_min + lat_max) / 2
        return latitude_center

    def run_real(self):
        inventory = self._return_inventory()
        obspy_utils.ObspyUtil.real_write_station_file(inventory, self.working_directory)
        stationfile = os.path.join(self.working_directory, "station.dat")
        ttime_file = os.path.join(self.working_directory, 'ttdb.txt')
        nllinput = os.path.join(self.output_directory, "nll_input.txt")
        realout = os.path.join(self.output_directory, "phase_sel_total.txt")

        # create travel times
        self._get_real_grid()
        tt_db = create_tt_db()

        tt_db.run_tt_db(ttime=self.working_directory, dist=self.h_range,
                        depth=self.real_config.geographic_frame.depth, ddist=0.01, ddep=1)

        ThresholdPwave = self.real_config.threshold_picks.min_num_p_wave_picks
        ThresholdSwave = self.real_config.threshold_picks.min_num_s_wave_picks
        number_stations_picks = self.real_config.threshold_picks.min_num_s_wave_picks
        real_handler = RealManager(pick_dir=self.picking_directory, station_file=stationfile,
                                   out_data_dir=self.output_directory,
                                   time_travel_table_file=ttime_file,
                                   gridSearchParamHorizontalRange=self.real_config.grid_search_parameters.horizontal_search_range,
                                   HorizontalGridSize=self.real_config.grid_search_parameters.horizontal_search_grid_size,
                                   DepthSearchParamHorizontalRange=self.real_config.grid_search_parameters.depth_search_range,
                                   DepthGridSize=self.real_config.grid_search_parameters.depth_search_grid_size,
                                   EventTimeW=self.real_config.grid_search_parameters.event_time_window,
                                   TTHorizontalRange=self.real_config.travel_time_grid_search.horizontal_range,
                                   TTHorizontalGridSize=self.real_config.travel_time_grid_search.horizontal_grid_resolution_size,
                                   TTDepthGridSize=self.real_config.travel_time_grid_search.depth_grid_resolution_size,
                                   TTDepthRange=self.real_config.travel_time_grid_search.depth_range,
                                   ThresholdPwave=ThresholdPwave,
                                   ThresholdSwave=ThresholdSwave,
                                   number_stations_picks=number_stations_picks)

        real_handler.latitude_center = self.__get_lat_mean()
        for events_info in real_handler:
            print(events_info)
            print(events_info.events_date)

        real_handler.save()
        real_handler.compute_t_dist()
        ConversionUtils.real2nll(realout, nllinput)
