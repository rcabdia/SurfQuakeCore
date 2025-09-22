#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
coincidence_parse
"""


# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: nll_parse.py
# Program: surfQuake & ISP
# Date: September 2025
# Purpose: Load Event Location Config file
# Author: Roberto Cabieces & Thiago C. Junqueira
#  Email: rcabdia@roa.es
# --------------------------------------------------------------------

from surfquakecore.coincidence_trigger.structures import CoincidenceConfig, Kurtosis, STA_LTA, Cluster
from surfquakecore.utils import Cast
from surfquakecore.utils.configuration_utils import parse_configuration_file


def load_concidence_configuration(config_file: str) -> CoincidenceConfig:
    """
    Load earthquake location configuration from a .ini file.

    .ini example:
        >>> f"
            [Kurtosis]
            fmin = 0.5
            fmax = 8.0
            CF_decay_win = 4.0
            hos_order = 4

            [STA_LTA]
            sta_win = 1
            lta_win = 40
            "


    :param config_file: The full path to the .ini configuration file.
    :return: An instance of EarthquakeLocationConfig
    """

    coincidence_config_ini = parse_configuration_file(config_file)

    return CoincidenceConfig(
            kurtosis_configuration=Kurtosis(
                    CF_decay_win=Cast(coincidence_config_ini['Kurtosis']['CF_decay_win'], float)),
            sta_lta_configuration=STA_LTA(
                    method=Cast(coincidence_config_ini['STA_LTA']['method'], str),
                    sta_win=Cast(coincidence_config_ini['STA_LTA']['sta_win'], float),
                    lta_win=Cast(coincidence_config_ini['STA_LTA']['lta_win'], float)),
            cluster_configuration=Cluster(
                method_preferred=Cast(coincidence_config_ini['Cluster']['method_preferred'], str),
                centroid_radio=Cast(coincidence_config_ini['Cluster']['centroid_radio'], float),
                coincidence=Cast(coincidence_config_ini['Cluster']['coincidence'], int),
                threshold_off=Cast(coincidence_config_ini['Cluster']['threshold_off'], float),
                threshold_on=Cast(coincidence_config_ini['Cluster']['threshold_on'], float),
                fmin=Cast(coincidence_config_ini['Cluster']['fmin'], float),
                fmax=Cast(coincidence_config_ini['Cluster']['fmax'], float)))

if __name__ == '__main__':
    config_file = "/Users/robertocabiecesdiaz/Desktop/surf_test/coincidence.ini"
    CoincidenceConfig = load_concidence_configuration(config_file)
    print(CoincidenceConfig)