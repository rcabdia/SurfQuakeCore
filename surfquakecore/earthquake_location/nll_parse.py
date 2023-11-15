from configparser import ConfigParser
from surfquakecore.earthquake_location.structures import NLLConfig, GridConfiguration, TravelTimesConfiguration, \
    LocationParameters
from surfquakecore.utils import Cast


def _read_config_file(file_path: str):
    config = ConfigParser()
    config.read(file_path)
    return config


def load_nll_configuration(config_file: str) -> NLLConfig:
    """
    Load earthquake location configuration from a .ini file.

    .ini example:
        >>> f"
            [GRID_CONFIGURATION]
            LATITUDE = 42.5414
            LONGITUDE = 1.4505
            DEPTH_KM = 5.75
            X = 400
            Y = 400
            Z = -2
            DX = 1
            DY = 1
            DZ = 1
            GEO_TRANSFORMATION = SIMPLE
            GRID_TYPE = SLOW_LEN
            PATH_TO_1D_MODEL = /HOME/ROBERTO/1DMODELS
            PATH_TO_3D_MODEL = NONE

            #
            [TRAVEL_TIMES_CONGIGURATION]
            DISTANCE_LIMIT = 500
            GRID1D = TRUE
            GRID3D = FALSE

            #
            [LOCATION_PARAMETERS]
            SEARCH = OCT-TREE
            METHOD = GAU_ANALYTIC
            "


    :param config_file: The full path to the .ini configuration file.
    :return: An instance of EarthquakeLocationConfig
    """

    nll_config_ini = _read_config_file(config_file)

    return NLLConfig(grid_configuration=GridConfiguration(latitude=Cast(nll_config_ini['GRID_CONFIGURATION']['LATITUDE'], float),
            longitude=Cast(nll_config_ini['GRID_CONFIGURATION']['LONGITUDE'], float),
            depth=Cast(nll_config_ini['GRID_CONFIGURATION']['DEPTH_KM'], float),
            x=Cast(nll_config_ini['GRID_CONFIGURATION']['X'], float), y=Cast(nll_config_ini['GRID_CONFIGURATION']['Y'], float),
            z=Cast(nll_config_ini['GRID_CONFIGURATION']['Z'], float), dx=Cast(nll_config_ini['GRID_CONFIGURATION']['DX'], float),
            dy=Cast(nll_config_ini['GRID_CONFIGURATION']['DY'], float), dz=Cast(nll_config_ini['GRID_CONFIGURATION']['DZ'], float),
            geo_transformation=Cast(nll_config_ini['GRID_CONFIGURATION']['GEO_TRANSFORMATION'], str),
            grid_type=Cast(nll_config_ini['GRID_CONFIGURATION']['GRID_TYPE'], str),
            path_to_1d_model=Cast(nll_config_ini['GRID_CONFIGURATION']['PATH_TO_1D_MODEL'], str),
            path_to_3d_model=Cast(nll_config_ini['GRID_CONFIGURATION']['PATH_TO_3D_MODEL'], str)),
            travel_times_configuration=TravelTimesConfiguration(
            distance_limit=Cast(nll_config_ini['TRAVEL_TIMES_CONFIGURATION']['DISTANCE_LIMIT'], float),
            grid1d=Cast(nll_config_ini['TRAVEL_TIMES_CONFIGURATION']['GRID1D'], bool),
            grid3d=Cast(nll_config_ini['TRAVEL_TIMES_CONFIGURATION']['GRID3D'], bool)),
            location_parameters=LocationParameters(search=Cast(nll_config_ini['LOCATION_PARAMETERS']['SEARCH'], str),
            method = Cast(nll_config_ini['LOCATION_PARAMETERS']['METHOD'], str)))
