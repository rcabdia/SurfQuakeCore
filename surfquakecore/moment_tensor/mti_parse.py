import os
import re
from datetime import datetime

import numpy as np

from surfquakecore.moment_tensor.structures import MomentTensorInversionConfig, StationConfig, InversionParameters, \
    SignalProcessingParameters
from surfquakecore.utils import Cast
from surfquakecore.utils.configuration_utils import parse_configuration_file
from surfquakecore.utils.string_utils import is_float


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

    mti_config_ini = parse_configuration_file(config_file)

    # load stations and channels
    stations = \
        [
            {'name': name.strip().upper(), 'channels': [ch.strip() for ch in channels.split(',')]}
            for name, channels in mti_config_ini.items("STATIONS_AND_CHANNELS")
        ]
    _params = {k.strip().lower(): v.strip() for k, v in mti_config_ini.items("ORIGIN")}
    _params['inversion_parameters'] = \
        {k.strip().lower(): v.strip() for k, v in mti_config_ini.items("MTI_PARAMETERS")}
    _params['signal_processing_parameters'] = \
        {k.strip().lower(): v.strip() for k, v in mti_config_ini.items("SIGNAL_PROCESSING")}
    _params['stations'] = stations

    return MomentTensorInversionConfig.from_dict(_params)


def load_mti_configurations(dir_path: str):
    """
    Load moment tensor inversion from a directory with configuration from a .ini file.
    Args:
        dir_path:

    Returns:

    """

    return (
        load_mti_configuration(os.path.join(dir_path, file))
        for file in os.listdir(dir_path) if file.endswith(".ini")
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
