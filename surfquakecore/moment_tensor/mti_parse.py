import re
from configparser import ConfigParser
from datetime import datetime

import numpy as np

from surfquakecore.moment_tensor.structures import MomentTensorInversionConfig, StationConfig, InversionParameters, \
    SignalProcessingParameters
from surfquakecore.utils import Cast
from surfquakecore.utils.string_utils import is_float


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
            SOURCE_DURATION = 2.0
            MIN_DIST = 10
            MAX_DIST = 300
            #
            [SIGNAL_PROCESSING]
            REMOVE_RESPONSE = True
            MAX_FREQ = 0.15
            MIN_FREQ = 0.02
            RMS_THRESH = 5.0
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
            source_type=Cast(mti_config_ini["MTI_PARAMETERS"]["SOURCE_TYPE"], str),
            covariance=Cast(mti_config_ini["MTI_PARAMETERS"]["COVARIANCE"], bool),
            deviatoric=Cast(mti_config_ini["MTI_PARAMETERS"]["DEVIATORIC"], bool),
            source_duration=Cast(mti_config_ini["MTI_PARAMETERS"]["SOURCE_DURATION"], float),
        ),
        signal_processing_parameters=SignalProcessingParameters(
            remove_response=Cast(mti_config_ini["SIGNAL_PROCESSING"]["REMOVE_RESPONSE"], bool),
            freq_min=Cast(mti_config_ini["SIGNAL_PROCESSING"]["MIN_FREQ"], float),
            freq_max=Cast(mti_config_ini["SIGNAL_PROCESSING"]["MAX_FREQ"], float),
            rms_thresh=Cast(mti_config_ini["SIGNAL_PROCESSING"]["RMS_THRESH"], float),
        )
    )


def read_isola_log(file: str):
    """
    Reads the ISOLA-ObsPy output log file.

    Parameters
    ----------
    file : string
        The full path to the log.txt file.

    Returns
    -------
    log_dict : dict
        A dictionary containing the results from the moment tensor inversion.

    """

    with open(file, 'r') as f:
        lines = f.readlines()

    ri = 0
    for i, line in enumerate(lines):
        if line == "Centroid location:\n":
            ri = i
            break

    lat_log_dep_line = tuple(v for v in lines[ri + 2].split(' ') if is_float(v))
    log_dict = {
        "latitude": float(lat_log_dep_line[0]),
        "longitude": float(lat_log_dep_line[1]),
        "depth": float(lat_log_dep_line[2]),
        "VR": float(lines[ri + 7].strip('\n').split(':')[1].strip(' %')),
        "CN": float(lines[ri + 8].strip('\n').split(':')[1].strip(' '))
    }

    # log_dict["Time"] = obspy.UTCDateTime('T'.join(lines[ri+1].strip('\n').split(' ')[4:]))

    MT_exp = float(lines[ri + 10].split('*')[1].strip(' \n'))
    MT = np.array([float(x) for x in lines[ri + 10].split('*')[0].strip('[ ]').split(' ') if x != ''])
    MT *= MT_exp

    mrr, mtt, mpp, mrt, mrp, mtp = MT
    log_dict["mrr"] = mrr
    log_dict["mtt"] = mtt
    log_dict["mpp"] = mpp
    log_dict["mrt"] = mrt
    log_dict["mrp"] = mrp
    log_dict["mtp"] = mtp

    log_dict["mo"] = float(lines[ri + 12].split('M0')[1].split('Nm')[0].strip(' ='))
    log_dict["mw_mt"] = float(lines[ri + 12].split('M0')[1].split('Nm')[1].strip('\n ( ) Mw = '))

    regex = re.compile('[a-zA-Z =:%,\n]')

    dc, clvd, iso = [float(x) for x in re.sub(regex, ' ', lines[ri + 13]).split(' ') if x != '']
    log_dict["dc"] = dc
    log_dict["clvd"] = clvd
    log_dict["iso"] = iso

    fp1_strike, fp1_dip, fp1_rake = [float(x) for x in re.sub(regex, ' ', lines[ri + 14].split(':')[1]).split(' ') if
                                     x != '' and x != '-']
    fp2_strike, fp2_dip, fp2_rake = [float(x) for x in re.sub(regex, ' ', lines[ri + 15].split(':')[1]).split(' ') if
                                     x != '' and x != '-']
    log_dict["strike_mt"] = fp1_strike
    log_dict["dip_mt"] = fp1_dip
    log_dict["rake_mt"] = fp1_rake
    log_dict["fp2_strike"] = fp2_strike
    log_dict["fp2_dip"] = fp2_dip
    log_dict["fp2_rake"] = fp2_rake

    return log_dict
