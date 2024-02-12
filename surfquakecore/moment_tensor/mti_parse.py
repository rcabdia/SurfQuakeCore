import json
import os
import re
import pandas as pd
import numpy as np
from surfquakecore.moment_tensor.structures import MomentTensorInversionConfig, MomentTensorResult
from surfquakecore.utils.configuration_utils import parse_configuration_file
from surfquakecore.utils.string_utils import is_float
from surfquakecore.utils.system_utils import deprecated


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


def read_isola_result(file: str) -> dict:
    """
    Reads the ISOLA-ObsPy output inversion.json file.

    :param file: The location of inversion.json from isola.
    :return:
    """
    with open(file, 'r') as f:
        return json.load(f)
        #return MomentTensorResult.from_dict(json.load(f))


@deprecated("Use read_isola_result instead.")
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

class WriteMTI:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    def list_files_with_iversion_json(self):
        iversion_json_files = []

        for foldername, subfolders, filenames in os.walk(self.root_folder):
            for filename in filenames:
                if filename == "inversion.json":
                    iversion_json_files.append(os.path.join(foldername, filename))

        return iversion_json_files

    def mti_summary(self, output):
        dates = []
        lats = []
        longs = []
        depths = []
        c_dates = []
        c_longs = []
        c_lats = []
        c_depths = []
        vr = []
        cn = []
        mrr = []
        mtt = []
        mpp = []
        mrt = []
        mrp = []
        mtp = []
        rupture_length = []
        mo = []
        mw = []
        dc = []
        clvd = []
        isotropic_component = []
        plane_2_strike = []
        plane_2_dip = []
        plane_2_slip_rake = []
        plane_1_strike = []
        plane_1_dip = []
        plane_1_slip_rake = []
        mti_dict = {'date_id': dates, 'lat': lats, 'long': longs, 'depth': depths,
                    'c_date': c_dates, 'c_lat': c_lats, 'c_longs': c_longs, 'c_depths': c_depths, "mw": mw, "mo": mo,
                    "rupture_legth": rupture_length, "dc": dc, "clvd": clvd, "vr": vr, "cn": cn,
                    "isotropic_component": isotropic_component,
                    "mrr": mrr, "mtt": mtt, "mpp": mpp,
                    "mrt": mrt, "mrp": mrp, "mtp": mtp, "plane_1_strike": plane_1_strike, "plane_1_dip": plane_1_dip,
                    "plane_1_slip_rake": plane_1_slip_rake, "plane_2_strike": plane_2_strike, "plane_2_dip": plane_2_dip,
                    "plane_2_slip_rake": plane_2_slip_rake }

        mti_files = self.list_files_with_iversion_json()
        for file in mti_files:
            mti_result = read_isola_result(file)
            dates.append(mti_result["origin_date"])
            lats.append(mti_result["latitude"])
            longs.append(mti_result["longitude"])
            depths.append(mti_result["depth_km"])
            c_lats.append(mti_result["centroid"]["latitude"])
            c_longs.append(mti_result["centroid"]["longitude"])
            c_depths.append(mti_result["centroid"]["depth"])
            c_dates.append(mti_result["centroid"]["time"])
            vr.append(mti_result["centroid"]["vr"])
            cn.append(mti_result["centroid"]["cn"])
            mrr.append(mti_result["centroid"]["mrr"])
            mtt.append(mti_result["centroid"]["mtt"])
            mpp.append(mti_result["centroid"]["mpp"])
            mrt.append(mti_result["centroid"]["mrt"])
            mrp.append(mti_result["centroid"]["mrp"])
            mtp.append(mti_result["centroid"]["mtp"])
            rupture_length.append(mti_result["centroid"]["rupture_length"])
            mo.append(mti_result["scalar"]["mo"])
            mw.append(mti_result["scalar"]["mw"])
            dc.append(mti_result["scalar"]["dc"])
            clvd.append(mti_result["scalar"]["clvd"])
            isotropic_component.append(mti_result["scalar"]["isotropic_component"])
            plane_1_strike.append(mti_result["scalar"]["plane_1_strike"])
            plane_1_dip.append(mti_result["scalar"]["plane_1_dip"])
            plane_1_slip_rake.append(mti_result["scalar"]["plane_1_slip_rake"])
            plane_2_strike.append(mti_result["scalar"]["plane_2_strike"])
            plane_2_dip.append(mti_result["scalar"]["plane_2_dip"])
            plane_2_slip_rake.append(mti_result["scalar"]["plane_2_slip_rake"])

        df_mti = pd.DataFrame.from_dict(mti_dict)
        print(df_mti)
        df_mti.to_csv(output, sep=";", index=False)
        print("Saved MTI summary at ", output)
