# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: real_parse.py
# Program: surfQuake & ISP
# Date: January 2024
# Purpose: Load Associator
# Author: Roberto Cabieces & Thiago C. Junqueira
#  Email: rcabdia@roa.es
# --------------------------------------------------------------------


from surfquakecore.real.structures import RealConfig, GeographicFrame, GridSearch, TravelTimeGridSearch, ThresholdPicks
from surfquakecore.utils import Cast
from surfquakecore.utils.configuration_utils import parse_configuration_file


def load_real_configuration(config_file: str) -> RealConfig:
    """
    Load earthquake associator configuration from a .ini file.

    .ini example:
        >>> f"
            [GEGRAPHIC_FRAME]
            LAT_REF_MAX = 43.0000
            LAT_REF_MIN = 41.0000
            LON_REF_MIN = -2.0000
            LON_REF_MAX = 0.0000
            DEPTH = 5.75
            #
            [GRID_SEARCH_PARAMETERS]
            HORIZONTAL_SEARCH_RANGE = 4.80
            DEPTH_SEARCH_RANGE = 50.00
            EVENT_TIME_WINDOW = 120.00
            HORIZONTAL_SEARCH_GRID_SIZE = 0.60
            DEPTH_SEARCH_GRID_SIZE = 10.00
            EVENT_TIME_WINDOW = 120
            #
            [TRAVEL_TIME_GRID_SEARCH]
            HORIZONTAL_RANGE = 5.00
            DEPTH_RANGE = 50.00
            DEPTH_GRID_RESOLUTION_SIZE = 2.00
            HORIZONTAL_GRID_RESOLUTION_SIZE = 0.01
            #
            [THRESHOLD_PICKS]
            MIN_NUM_P_WAVE_PICKS = 3
            MIN_NUM_S_WAVE_PICKS = 1
            NUM_STATIONS_RECORDED = 1
            "


    :param config_file: The full path to the .ini configuration file.
    :return: An instance of MomentTensorInversionConfig
    """

    real_config_ini = parse_configuration_file(config_file)
    return RealConfig(
        geographic_frame=GeographicFrame(
            lat_ref_max=Cast(real_config_ini['GEOGRAPHIC_FRAME']['LAT_REF_MAX'], float),
            lon_ref_max=Cast(real_config_ini['GEOGRAPHIC_FRAME']['LON_REF_MAX'], float),
            lat_ref_min=Cast(real_config_ini['GEOGRAPHIC_FRAME']['LAT_REF_MIN'], float),
            lon_ref_min=Cast(real_config_ini['GEOGRAPHIC_FRAME']['LON_REF_MIN'], float),
            depth=Cast(real_config_ini['GEOGRAPHIC_FRAME']['DEPTH'], float)
        ),
        grid_search_parameters=GridSearch(
            horizontal_search_range=Cast(real_config_ini['GRID_SEARCH_PARAMETERS']['HORIZONTAL_SEARCH_RANGE'], float),
            depth_search_range=Cast(real_config_ini['GRID_SEARCH_PARAMETERS']['DEPTH_SEARCH_RANGE'], float),
            event_time_window=Cast(real_config_ini['GRID_SEARCH_PARAMETERS']['EVENT_TIME_WINDOW'], float),
            horizontal_search_grid_size=Cast(
                real_config_ini['GRID_SEARCH_PARAMETERS']['HORIZONTAL_SEARCH_GRID_SIZE'], float),
            depth_search_grid_size=Cast(real_config_ini['GRID_SEARCH_PARAMETERS']['DEPTH_SEARCH_GRID_SIZE'], float)),
        travel_time_grid_search=TravelTimeGridSearch(
                horizontal_range=Cast(real_config_ini['TRAVEL_TIME_GRID_SEARCH']['HORIZONTAL_RANGE'], float),
                depth_range=Cast(real_config_ini['TRAVEL_TIME_GRID_SEARCH']['DEPTH_RANGE'], float),
                depth_grid_resolution_size=Cast(
                    real_config_ini['TRAVEL_TIME_GRID_SEARCH']['DEPTH_GRID_RESOLUTION_SIZE'], float),
                horizontal_grid_resolution_size=Cast(
                    real_config_ini['TRAVEL_TIME_GRID_SEARCH']['HORIZONTAL_GRID_RESOLUTION_SIZE'], float)),
        threshold_picks=ThresholdPicks(
            min_num_p_wave_picks=Cast(real_config_ini['THRESHOLD_PICKS']['MIN_NUM_P_WAVE_PICKS'], int),
            min_num_s_wave_picks=Cast(real_config_ini['THRESHOLD_PICKS']['MIN_NUM_S_WAVE_PICKS'], int),
            num_stations_recorded=Cast(real_config_ini['THRESHOLD_PICKS']['NUM_STATIONS_RECORDED'], int))
    )
