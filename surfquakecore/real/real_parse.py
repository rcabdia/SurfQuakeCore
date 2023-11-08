# from configparser import ConfigParser
# from surfquakecore.real.structures import RealConfig, GeographicFrame
# from surfquakecore.utils import Cast
#
#
# def _read_config_file(file_path: str):
#     config = ConfigParser()
#     config.read(file_path)
#     return config
#
#
# def load_real_configuration(config_file: str) -> RealConfig:
#     """
#     Load earthquake associator configuration from a .ini file.
#
#     .ini example:
#         >>> f"
#             [GEGRAPHIC_FRAME]
#             LAT_REF_MAX = 43.0000
#             LAT_REF_MIN = 41.0000
#             LON_REF_MIN = -2.0000
#             LON_REF_MAX = 0.0000
#             DEPTH = 5.75
#             #
#             [GRID_SEARCH_PARAMETERS]
#             HORIZONTAL_SEARCH_RANGE = 4.80
#             DEPTH_SEARCH_RANGE = 50.00
#             EVENT_TIME_WINDOW = 120.00
#             HORIZONTAL_SEARCH_GRID_SIZE = 0.60
#             DEPTH_SEARCH_GRID_SIZE = 10.00
#             EVENT_TIME_WINDOW = 120
#             #
#             [TRAVEL_TIME_GRID_SEARCH]
#             HORIZONTAL_RANGE = 5.00
#             DEPTH_RANGE = 50.00
#             DEPTH_GRID_RESOLUTION_SIZE = 2.00
#             HORIZONTAL_GRID_RESOLUTION_SIZE = 0.01
#             #
#             [THRESHOLD_PICKS]
#             MIN_NUM_P_WAVE_PICKS = 3
#             MIN_NUM_S_WAVE_PICKS = 1
#             NUM_STATIONS_RECORDED = 1
#             "
#
#
#     :param config_file: The full path to the .ini configuration file.
#     :return: An instance of MomentTensorInversionConfig
#     """
#
#     real_config_ini = _read_config_file(config_file)
#
#     return RealConfig(geographic_frame=GeographicFrame(geographic_frame =real_config_ini['GEOGRAPHIC_FRAME']['LAT_REF_MAX']))
