from configparser import ConfigParser
from datetime import datetime

from surfquakecore.moment_tensor.structures import MomentTensorInversionConfig, StationConfig, InversionParameters, \
    SignalProcessingParameters
from surfquakecore.utils import Cast


def _read_config_file(file_path: str):
    config = ConfigParser()
    config.read(file_path)
    return config


def load_mti_configuration(config_file: str) -> MomentTensorInversionConfig:
    """
    Load moment tensor inversion configuration from a .ini file.

    .ini example:
        >>> f"
            [ORIGIN]
            ORIGIN_DATE = 28/02/2022 02:07:59.433
            LATITUDE = 42.5414
            LONGITUDE= 1.4505
            DEPTH_KM = 5.75
            MAGNITUDE = 3.0
            #
            [STATIONS_AND_CHANNELS]
            MAHO = HHZ, HHN, HHE
            WAIT = BH*
            EVO = *
            #
            [MTI_PARAMETERS]
            EARTH_MODEL_FILE = earthmodel/Iberia.txt
            LOCATION_UNC = 0.7
            TIME_UNC = .2
            DEVIATORIC = True
            DEPTH_UNC = 3
            COVARIANCE = True
            RUPTURE_VELOCITY = 2500
            SOURCE_TYPE = Triangle
            MIN_DIST = 10
            MAX_DIST = 300
            #
            [SIGNAL_PROCESSING]
            REMOVE_RESPONSE = True
            MAX_FREQ = 0.15
            MIN_FREQ = 0.02
            "


    :param config_file: The full path to the .ini configuration file.
    :return: An instance of MomentTensorInversionConfig
    """

    mti_config_ini = _read_config_file(config_file)

    # load stations and channels
    stations = \
        [
            StationConfig(name=name.strip().upper(), channels=[ch.strip() for ch in channels.split(',')])
            for name, channels in mti_config_ini.items("STATIONS_AND_CHANNELS")
        ]

    return MomentTensorInversionConfig(
        origin_date=Cast(mti_config_ini["ORIGIN"]["ORIGIN_DATE"], datetime),
        latitude=Cast(mti_config_ini["ORIGIN"]["LATITUDE"], float),
        longitude=Cast(mti_config_ini["ORIGIN"]["LONGITUDE"], float),
        depth=Cast(mti_config_ini["ORIGIN"]["DEPTH_KM"], float),
        magnitude=Cast(mti_config_ini["ORIGIN"]["MAGNITUDE"], float),
        stations=stations,
        inversion_parameters=InversionParameters(
            earth_model_file=mti_config_ini["MTI_PARAMETERS"]["EARTH_MODEL_FILE"].strip(),
            location_unc=Cast(mti_config_ini["MTI_PARAMETERS"]["LOCATION_UNC"], float),
            time_unc=Cast(mti_config_ini["MTI_PARAMETERS"]["TIME_UNC"], float),
            depth_unc=Cast(mti_config_ini["MTI_PARAMETERS"]["DEPTH_UNC"], float),
            rupture_velocity=Cast(mti_config_ini["MTI_PARAMETERS"]["RUPTURE_VELOCITY"], float),
            min_dist=Cast(mti_config_ini["MTI_PARAMETERS"]["MIN_DIST"], float),
            max_dist=Cast(mti_config_ini["MTI_PARAMETERS"]["MAX_DIST"], float),
        ),
        signal_processing_pams=SignalProcessingParameters(
            remove_response=Cast(mti_config_ini["SIGNAL_PROCESSING"]["REMOVE_RESPONSE"], bool),
            freq_min=Cast(mti_config_ini["SIGNAL_PROCESSING"]["MIN_FREQ"], float),
            freq_max=Cast(mti_config_ini["SIGNAL_PROCESSING"]["MAX_FREQ"], float),
        )
    )
