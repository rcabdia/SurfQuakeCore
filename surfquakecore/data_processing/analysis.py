from operator import truediv
from datetime import datetime
from obspy import read, Stream, read_inventory, UTCDateTime
from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth, kilometer2degrees
from tensorflow.python.ops.random_ops import parameterized_truncated_normal
import os
import copy
import scipy, numpy as np
import math
import pywt
import pandas as pd

from surfquakecore.Structures.structures import TracerStatsAnalysis, TracerStats
from surfquakecore.utils.obspy_utils import ObspyUtil, Filters
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from surfquakecore.utils import BaseDataClass
from surfquakecore.data_processing.source_tools import ReadSource
from multiprocessing import freeze_support
from surfquakecore.project.surf_project import SurfProject
from surfquakecore.data_processing.seismogram_analysis import SeismogramData

ANALYSIS_KEYS = ['rmean', 'taper', 'normalize', 'differentiate', 'integrate', 'filter', 'wiener_filter',
                 'shift', 'remove_response', 'add_white_noise', 'whitening', 'remove_spikes',
                 'time_normalization', 'wavelet_denoise', 'resample', 'fill_gaps', 'smoothing']

RMEAN_METHODS = ['simple', 'linear', 'constant', 'demean', 'polynomial', 'spline']

TAPER_METHODS = ['cosine', 'barthann', 'bartlett', 'blackman', 'blackmanharris', 'bohman', 'boxcar', 'chebwin',
                 'flattop', 'gaussian', 'general_gaussian', 'hamming', 'hann', 'kaiser', 'nuttall', 'parzen', 'slepian',
                 'triang']

WAVELET_METHODS = ['db2','db4','db6','db8','db10','db12','db14','db16','db18','db20', 'sym2', 'sym4', 'sym6', 'sym8',
                   'sym10', 'sym12', 'sym14', 'sym16', 'sym18', 'sym20', 'coif2', 'coif3', 'coif4', 'coif6', 'coif8',
                   'coif10', 'coif12', 'coif14', 'coif16', 'bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4',
                   'bior2.6', 'bior2.8', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8']

INTEGRATE_METHODS = ['cumtrapz', 'spline']

FILTER_METHODS = ['bandpass', 'bandstop', 'lowpass', 'highpass', 'lowpass_cheby_2', 'lowpass_fir', 'remez_fir']

TIME_METHODS = ['1bit', 'clipping', 'clipping_iteration','time_normalization']

FILL_GAP_METHODS = ['latest', 'interpolate']

SMOOTHING_METHODS = ['mean', 'gaussian', 'tkeo']

TAPER_KEYS = ['method', 'max_percentage', 'max_length', 'side']

TAPER_SIDE = ['both', 'left', 'right']


class Check:
    def check_rmean(self, config):
        _keys = []

        if 'method' in config:
            if config['method'] in RMEAN_METHODS:
                if config['method'] in ['polynomial', 'spline']:
                    _keys = config.keys()
                else:
                    return True

                if len(_keys) > 0:
                    if config['method'] == 'polynomial':
                        if 'order' in _keys:
                            if isinstance(config['order'], int):
                                return True
                            else:
                                raise ValueError(
                                    f"RMEAN: config {config} is not valid. It must be a valid .yaml file for "
                                    f"AnalysisConfig. \n"
                                    f"Wrong type for order value. Must be int")
                        else:
                            raise ValueError(f"RMEAN: config {config} is not valid. It must be a valid .yaml file for "
                                             f"AnalysisConfig. \n"
                                             f"order parameter is required for polynomial method")
                    elif config['method'] == 'spline':
                        if 'order' in _keys and 'dspline' in _keys:
                            if isinstance(config['order'], int) and isinstance(config['dspline'], int):
                                if 1 <= config['order'] <= 5:
                                    return True
                                else:
                                    raise ValueError(
                                        f"RMEAN: config {config} is not valid. It must be a valid .yaml file for "
                                        f"AnalysisConfig.\n"
                                        f"Wrong order value. 1 <= order <= 5")
                            else:
                                raise ValueError(
                                    f"RMEAN: config {config} is not valid. It must be a valid .yaml file for "
                                    f"AnalysisConfig.\n"
                                    f"Wrong type for order and/or dspline values. Must be int")
                        else:
                            raise ValueError(f"RMEAN: config {config} is not valid. It must be a valid .yaml file for "
                                             f"AnalysisConfig.\n"
                                             f"order and dspline parameters are required for spline method")
                else:
                    raise ValueError(f"RMEAN: config {config} is not valid. It must be a valid .yaml file for "
                                     f"AnalysisConfig. \n"
                                     f"missing order and/or dspline parameters")
            else:
                raise ValueError(f"RMEAN: config {config} is not valid. It must be a valid .yaml file for "
                                 f"AnalysisConfig\n. "
                                 f"Method not supported.\n"
                                 f"Supported methods: simple, linear, constant, demean, polynomial, spline.")
        else:
            raise ValueError(f"RMEAN: config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig.\n"
                             f"method parameter is required.")

    def check_taper(self, config):
        _keys = config.keys()
        _method = True
        _max_percentage = True

        if 'method' in _keys:
            if config['method'] not in TAPER_METHODS:
                raise ValueError(f"TAPER: config {config} is not valid. It must be a valid .yaml file for "
                                 f"AnalysisConfig\n. "
                                 f"Method not supported.\n"
                                 f"Supported methods: 'cosine', 'barthann', 'bartlett', 'blackman', 'blackmanharris', "
                                 f"'bohman', 'boxcar', 'chebwin', 'flattop', 'gaussian', 'general_gaussian', 'hamming', "
                                 f"'hann', 'kaiser', 'nuttall', 'parzen', 'slepian', 'triang'")
        else:
            raise ValueError(f"TAPER: config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig.\n"
                             f"method parameter is required.")

        if 'max_percentage' in _keys:
            if isinstance(config['max_percentage'], float):
                if config['max_percentage'] < 0.0 or config['max_percentage'] > 0.5:
                    raise ValueError(
                        f"TAPER: config {config} is not valid. It must be a valid .yaml file for "
                        f"AnalysisConfig.\n"
                        f"Wrong max_percentage value. 0.0 <= max_percentage <= 0.5")
            else:
                raise ValueError(
                    f"TAPER: config {config} is not valid. It must be a valid .yaml file for "
                    f"AnalysisConfig.\n"
                    f"Wrong type for max_percentage value. Must be float")
        else:
            raise ValueError(f"TAPER: config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig.\n"
                             f"max_percentage parameter is required.")

        if 'max_lenght' in _keys:
            if not isinstance(config['max_length'], float):
                raise ValueError(
                    f"TAPER: config {config} is not valid. It must be a valid .yaml file for "
                    f"AnalysisConfig.\n"
                    f"Wrong type for max_length value. Must be float")

        if 'side' in _keys:
            if config['side'] not in TAPER_SIDE:
                raise ValueError(f"Taper: config {config} is not valid. It must be a valid .yaml file for "
                                 f"AnalysisConfig.\n"
                                 f"Wrong value for side.\n"
                                 f"Valid values for side: 'both', 'left', 'right'")

        return True

    def check_normalize(self, config):
        _config_keys = config.keys()

        if 'norm' in _config_keys:
            if config['norm'] is None or isinstance(config['norm'], float):
                return True
            else:
                raise ValueError(f"NORMALIZE: config {config} is not valid. It must be a valid .yaml file for "
                                 f"AnalysisConfig.\n"
                                 f"Wrong type for norm value. Must be None or float")
        else:
            raise ValueError(f"NORMALIZE: config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig.\n"
                             f"norm parameter is required.")

    def check_shift(self, config):
        i = 0
        if len(config) > 0:
            for i in range(0, len(config)):
                _config_keys = config[i].keys()

                if 'name' in _config_keys and 'time' in _config_keys:
                    return True
                else:
                    raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                                     f"AnalysisConfig")

    def check_integrate(self, config):
        _config_keys = config.keys()

        if 'method' in _config_keys:
            if config['method'] in INTEGRATE_METHODS:
                return True
            else:
                raise ValueError(f"INTEGRATE:config {config} is not valid. It must be a valid .yaml file for "
                                 f"AnalysisConfig."
                                 f"Method not supported.\n"
                                 f"Supported methods: 'cumtrapz', 'spline'")
        else:
            raise ValueError(f"INTEGRATE:config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig."
                             f"method parameter is required.")

    def check_filter(self, config):
        _config_keys = config.keys()

        if 'zerophase' in _config_keys:
            if not isinstance(config['zerophase'], bool):
                raise ValueError(f"FILTER: config {config} is not valid. It must be a valid .yaml file for "
                                 f"AnalysisConfig."
                                 f"Wrong type for zerophase value. Must be boolean")


        if 'method' in _config_keys:
            if config['method'] == 'bandpass' or config['method'] == 'bandstop' or config['method'] == 'remez_fir':
                if 'freqmin' in _config_keys and 'freqmax' in _config_keys:
                    if isinstance(config['freqmin'], float) or isinstance(config['freqmax'], float):
                        return True
                    else:
                        raise ValueError(f"FILTER: config {config} is not valid. It must be a valid .yaml file for "
                                         f"AnalysisConfig."
                                         f"Wrong type for freqmin and freqmax values. Must be float")
                else:
                    raise ValueError(f"FILTER: config {config} is not valid. It must be a valid .yaml file for "
                                     f"AnalysisConfig."
                                     f"freqmin adn freqmax parameter are required for {config['method']} method.")
            elif (config['method'] == 'lowpass' or config['method'] == 'highpass' or config['method'] == 'lowpass_fir'
                  or config['method'] == 'lowpass_cheby_2'):
                if 'freq' in _config_keys:
                    return True
                else:
                    raise ValueError(f"FILTER: config {config} is not valid. It must be a valid .yaml file for "
                                     f"AnalysisConfig."
                                     f"freq parameter are required for {config['method']} method.")
            else:
                raise ValueError(f"FILTER: config {config} is not valid. It must be a valid .yaml file for "
                                 f"AnalysisConfig."
                                 f"Method not supported.\n"
                                 f"Supported methods: 'bandpass', 'bandstop', 'lowpass', 'highpass', 'lowpass_cheby_2', "
                                 f"'lowpass_fir', 'remez_fir'")
        else:
            raise ValueError(f"FILTER: config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig."
                             f"method parameter is required.")

    def check_remove_response(self, config):
        _config_keys = config.keys()

        if ('inventory' in _config_keys and 'water_level' in _config_keys and 'units' in _config_keys
            and 'pre_filt' in _config_keys):

            return True
        else:
            raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig")

    def check_differentiate(self, config):
        _config_keys = config.keys()

        if 'diff' in _config_keys:
            if isinstance(config['diff'], bool):
                return True
            else:
                raise ValueError(f"DIFFERENTIATE: config {config} is not valid. It must be a valid .yaml file for "
                                 f"AnalysisConfig."
                                 f"Wrong type for diff. Must be bool")

        else:
            raise ValueError(f"DIFFERENTIATE:config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig."
                             f"diff parameter is required.")

    def check_wiener_filter(self, config):
        _config_keys = config.keys()

        if 'time_window' in _config_keys and 'noise_power' in _config_keys:
            return True
        else:
            raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig")

    def check_add_white_noise(self, config):
        _config_keys = config.keys()

        if 'SNR_dB' in _config_keys:
            return True
        else:
            raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig")

    def check_whiten(self, config):
        _config_keys = config.keys()

        if 'taper_edge' in _config_keys:
            if isinstance(config['taper_edge'], bool):
                return True
            else:
                raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                                 f"AnalysisConfig")
        else:
            return True

    def check_remove_spikes(self, config):
        _config_keys = config.keys()

        if 'window_size' in _config_keys:
            return True
        else:
            raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig")

    def check_time_normalization(self, config):
        _config_keys = config.keys()

        if 'method' in _config_keys and 'norm_win' in _config_keys:
            if config['method'] in TIME_METHODS:
                return True
            else:
                raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                                 f"AnalysisConfig")
        else:
            raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig")

    def check_wavelet_denoise(self, config):
        _config_keys = config.keys()

        if 'dwt' in _config_keys and 'threshold' in _config_keys:
            if config['dwt'] in WAVELET_METHODS:
                return True
            else:
                raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                                 f"AnalysisConfig")
        else:
            raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig")

    def check_resample(self, config):
        _config_keys = config.keys()

        if 'sampling_rate' in _config_keys and 'pre_filter' in _config_keys:
            if isinstance(config['pre_filter'], bool):
                return True
            else:
                raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                                 f"AnalysisConfig")
        else:
            raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig")

    def check_fill_gaps(self, config):
        _config_keys = config.keys()

        if 'method' in _config_keys:
            if config['method'] in FILL_GAP_METHODS:
                return True
            else:
                raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                                 f"AnalysisConfig")
        else:
            raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig")

    def check_smoothing(self, config):
        _config_keys = config.keys()

        if 'method' in _config_keys and 'time_window' in _config_keys and 'FWHM' in _config_keys:
            if config['method'] in SMOOTHING_METHODS:
                return True
            else:
                raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                                 f"AnalysisConfig")
        else:
            raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig")

    def parse_filter(self, config):
        _config = {}

        # bandpass bandstop lowpass highpass remez_fir
        _filter_template_1 = {'name': 'filter',
                              'corners': 4,
                              'zerophase': False}
        # lowpass_fir
        _filter_template_2 = {'name': 'filter',
                              'winlen': 2048}

        # lowpass_cheby_2
        _filter_template_3 = {'name': 'filter',
                              'maxorder': 12,
                              'ba': False,
                              'freq_passband': False}

        _config_keys = []

        if config is not None:
            _config = config
            _config_keys = config.keys()

        for _key in _config_keys:
            if _key == 'method':
                if _config[_key] in ['bandpass', 'bandstop', 'lowpass', 'highpass', 'remez_fir']:
                    _filter_template_1[_key] = _config[_key]
                elif _config[_key] in ['lowpass_fir']:
                    _filter_template_2[_key] = _config[_key]
                elif _config[_key] in ['lowpass_cheby_2']:
                    _filter_template_3[_key] = _config[_key]

            if _key == 'freqmin' or _key == 'freqmax':
                _filter_template_1[_key] = _config[_key]

            if _key == 'freq':
                if _config['method'] in ['lowpass', 'highpass']:
                    _filter_template_1[_key] = _config[_key]
                elif _config['method'] in ['lowpass_fir']:
                    _filter_template_2[_key] = _config[_key]
                elif _config['method'] in ['lowpass_cheby_2']:
                    _filter_template_3[_key] = _config[_key]

            if _key == 'poles':
                _filter_template_1['corners'] = _config[_key]

            if _key == 'zerophase':
                _filter_template_1[_key] = _config[_key]

            if _key == 'winlen':
                _filter_template_2[_key] = _config[_key]

            if _key == 'maxorder':
                _filter_template_3[_key] = _config[_key]

            if _key == 'ba':
                _filter_template_3[_key] = _config[_key]

            if _key == 'freq_passband':
                _filter_template_3[_key] = _config[_key]

        if _config['method'] in ['bandpass', 'bandstop', 'lowpass', 'highpass', 'remez_fir']:
            return _filter_template_1
        elif _config['method'] in ['lowpass_fir']:
            return _filter_template_2
        elif _config['method'] in ['lowpass_cheby_2']:
            return _filter_template_3

    def parse_taper(self, config):
        _config = {}
        _taper_template = {'name': 'taper',
                           'method': 'hann',
                           'max_percentage': 0.05,
                           'max_length': None,
                           'side': 'both'}
        _config_keys = []

        if config is not None:
            _config = config
            _config_keys = config.keys()

        for _key in _config_keys:
            if _key == 'method':
                _taper_template[_key] = _config[_key]

            if _key == 'max_percentage':
                _taper_template[_key] = _config[_key]

            if _key == 'max_length':
                _taper_template[_key] = _config[_key]

            if _key == 'side':
                _taper_template[_key] = _config[_key]

        return _taper_template

    def parse_whiten(self, config):
        _config = {}
        _whiten_template = {'name': 'whitening',
                           'taper_edge': True,
                           'freq_width': 0.05}
        _config_keys = []

        if config is not None:
            _config = config
            _config_keys = config.keys()

        for _key in _config_keys:
            if _key == 'taper_edge':
                _whiten_template[_key] = _config[_key]

            if _key == 'freq_width':
                _whiten_template[_key] = _config[_key]

        return _whiten_template

    def parse_remove_spikes(self, config):
        _config = {}
        _remove_spikes_template = {'name': 'remove_spikes',
                           'n': 3}
        _config_keys = []

        if config is not None:
            _config = config
            _config_keys = config.keys()

        for _key in _config_keys:
            if _key == 'window_size':
                _remove_spikes_template[_key] = _config[_key]

            if _key == 'n':
                _remove_spikes_template[_key] = _config[_key]

        return _remove_spikes_template

    def parse_configuration_file(self, config: dict):
        _config = {}
        _config_template = []
        _config_keys = []

        print(config)

        if 'Analysis' in config:
            _config = config['Analysis']
            _config_keys = _config.keys()
            _rmean_keys = []
            i = 0
            _process = []

            for i in range(len(_config_keys)):
                _name = 'process_' + str(i + 1)
                _process.append(_config[_name])

            for i in range(len(_process)):
                _keys = _process[i].keys()
                _process_config = _process[i]

                if 'name' in _keys:
                    for _key in _keys:
                        if _process_config[_key] in ANALYSIS_KEYS:
                            if _process_config[_key] == 'rmean':
                                if self.check_rmean(_process[i]):
                                    _config_template.append(_process[i])

                            if _process_config[_key] == 'taper':
                                if self.check_taper(_process[i]):
                                    _config_template.append(self.parse_taper(_process[i]))

                            if _process_config[_key] == 'normalize':
                                if self.check_normalize(_process[i]):
                                    _config_template.append(_process[i])

                            if _process_config[_key] == 'differentiate':
                                if self.check_differentiate(_process[i]):
                                    _config_template.append(_process[i])

                            if _process_config[_key] == 'integrate':
                                if self.check_integrate(_process[i]):
                                    _config_template.append(_process[i])

                            if _process_config[_key] == 'filter':
                                if self.check_filter(_process[i]):
                                    _config_template.append(self.parse_filter(_process[i]))

                            if _process_config[_key] == 'wiener_filter':
                                if self.check_wiener_filter(_process[i]):
                                    _config_template.append(_process[i])

                            if _process_config[_key] == 'shift':
                                if self.check_shift(_process[i]):
                                    _config_template.append(_process[i])

                            if _process_config[_key] == 'remove_response':
                                if self.check_remove_response(_process[i]):
                                    _config_template.append(_process[i])

                            if _process_config[_key] == 'add_white_noise':
                                if self.check_add_white_noise(_process[i]):
                                    _config_template.append(_process[i])
                            if _process_config[_key] == 'whitening':
                                if self.check_whiten(_process[i]):
                                    _config_template.append(self.parse_whiten(_process[i]))

                            if _process_config[_key] == 'remove_spikes':
                                if self.check_remove_spikes(_process[i]):
                                    _config_template.append(self.parse_remove_spikes(_process[i]))

                            if _process_config[_key] == 'time_normalization':
                                if self.check_time_normalization(_process[i]):
                                    _config_template.append(_process[i])

                            if _process_config[_key] == 'wavelet_denoise':
                                if self.check_wavelet_denoise(_process[i]):
                                    _config_template.append(_process[i])

                            if _process_config[_key] == 'resample':
                                if self.check_resample(_process[i]):
                                    _config_template.append(_process[i])

                            if _process_config[_key] == 'fill_gaps':
                                if self.check_fill_gaps(_process[i]):
                                    _config_template.append(_process[i])

                            if _process_config[_key] == 'smoothing':
                                if self.check_smoothing(_process[i]):
                                    _config_template.append(_process[i])
                else:
                    raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                                     f"AnalysisConfig")
                    # Check type

            return _config_template
        else:
            raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig. 'Analysis' header is missing")


class Analysis:

    def __init__(self, files, output, inventory, config_file=None, event_file=None):
        print('CLASE ANALISIS')
        self.output = output
        self.files = files
        self.inventory = read_inventory(inventory)

        self.config_file = None
        self.event_file = None

        if config_file is not None:
            self.config_file = self.load_analysis_configuration(config_file)
        

        if event_file is not None:
            self.event_file = event_file        

    @staticmethod
    def filter_files(project, net=None, station=None, channel=None, starttime=None, endtime=None):
        filter = {}
        time = {}
        date_format = "%Y-%m-%d %H:%M:%S"

        # Filter net
        if net is not None:
            filter['net'] = net
        
        # Filter station
        if station is not None:
            filter['station'] = station

        # Filter channel
        if channel is not None:
            filter['channel'] = channel

        # Filter start time
        if starttime is not None:
            try:
                time['startime'] = UTCDateTime(datetime.strptime(starttime, date_format))
            except ValueError:
                raise ValueError(f"Error: start time '{starttime}' does not hace the correct format ({date_format}).")    

        # Filter end time
        if endtime is not None:
            try:
                time['endtime'] = UTCDateTime(datetime.strptime(endtime, date_format))
            except ValueError:
                raise ValueError(f"Error: end time '{endtime}' does not hace the correct format ({date_format}).")  

        # Apply filter
        if len(filter) > 0:
            project.filter_project_keys(**filter)
        else:
            project.filter_project_keys()

        # Apply filter time
        if len(time) > 0:
            return project.filter_time(**time)
        else:
            return project.filter_time()

    def load_analysis_configuration(self, config_file: str):
        rs = ReadSource(config_file)
        config = rs.read_file(config_file)
        chk = Check()
        return chk.parse_configuration_file(config)

    def get_project_files(self, project_path):
        freeze_support()
        sp = SurfProject(project_path)
        sp.search_files()
        return sp.project

    def run_analysis(self, start, end):
        # 1.- Check self.event_file is not None
        if self.event_file is not None:
            # 2.- Cut event files
            print('Cut Event Files')
            self.cut_files(start, end)

        #for i in range(len(self.files)):
        #    sd = SeismogramData(self.files[i])
        #    tr = sd.run_analysis(self.config_file)
        #    tr.write(os.path.join(self.output, tr.id), 'mseed')

    def run_cut_waveforms(self, project, events, inventories, deltastart, deltaend):
        model = TauPyModel("iasp91")
        distance_event = []

        for index, event in events.iterrows():
            id = event['datetime'].strftime("%Y-%m-%d %H:%M:%S")
            lat = event['latitude']
            lon = event['longitude']
            depth = event['depth']
            event_time = event['datetime']
            
            # Crear carpeta
            event_folder = os.path.join(self.output, id)

            self.create_folder(event_folder)

            for index, inventory in inventories.iterrows():
                distance_km, _, _ = gps2dist_azimuth(lat, lon, inventory['latitude'], inventory['longitude'])
                distance_km = kilometer2degrees(distance_km/1000)

                # Calcular los tiempos de arribo para cada onda (P y S) para cada distancia
                arrivals = model.get_travel_times(source_depth_in_km=depth, distance_in_degree=distance_km, phase_list=["P", "S"])
                
                if len(arrivals) == 0:
                   arrivals = model.get_travel_times(source_depth_in_km=depth, distance_in_degree=distance_km, phase_list=["p", "s"])
                 
                
                p_time = None
                s_time = None

                if len(arrivals) > 0:
                # Guardamos los tiempos de las ondas P y S
                    for arrival in arrivals:
                        if (arrival.name == 'P' or arrival.name == 'p') and p_time is None:
                            p_time = arrival.time
                        elif (arrival.name == 'S' or arrival.name == 's') and s_time is None:
                            s_time = arrival.time
            
                    start = event["datetime"] + timedelta(seconds=p_time) - timedelta(seconds=deltastart)
                    end = event["datetime"] + timedelta(seconds=s_time) + timedelta(seconds=deltaend)

                    file = inventory["file"].split("/")

                    if end is not None and start is not None:
                        st = read(inventory["file"])
                        st.trim(UTCDateTime(start), UTCDateTime(end))

                        if self.config_file:
                            sd = SeismogramData(st)
                            tr = sd.run_analysis(self.config_file)
                        else:
                            tr = st
                      
                        try:
                            #if config -> tratar config
                            #st.write(os.path.join(event_folder, file[len(file)-1]), 'mseed')
                            tr.write(os.path.join(event_folder, file[len(file)-1]), 'mseed')
                        except:
                            print('Empty file')                        
    
    def create_folder(self, name):
        if not os.path.exists(name):
            os.makedirs(name)

    def cut_files(self, start, end):
        # 1. Panda dataframe with events
        df_events = pd.read_csv(self.event_file, sep=';')
        df_events['datetime'] = pd.to_datetime((df_events['date'] + ' ' + df_events['hour']), utc=True)
        
        # 2. Project dataframe
        df_project = self.project_table()
        
        # 3. Invetory dataframe
        df_inventory = self.inventory_table()

        print('Three dataframes')
        # 4. Ver cada evento. Devolv
        for index, event in df_events.iterrows():
            df_files = pd.DataFrame(columns=df_project.columns)
            id = event['datetime'].strftime("%Y-%m-%d %H:%M:%S")
            lat = event['latitude']
            lon = event['longitude']
            depth = event['depth']
            event_time = event['datetime']
            start_event = event['datetime'] - timedelta(seconds=start)
            end_event = event['datetime'] + timedelta(seconds=end)
            
            # Crear carpeta
            event_folder = os.path.join(self.output, id)

            self.create_folder(event_folder)
    
            for index, file in df_project.iterrows():
                print('hola')
                if (file['start'] <= start_event and file['end'] >= start_event) or (file['start'] <= end_event and file['end'] >= end_event) or (file['start'] <= start_event and file['end'] >= end_event):
                    df_files = pd.concat([df_files, file.to_frame().T], ignore_index=True)

            stations = df_files['station'].unique()
            channels = df_files['channel'].unique()

            for _station in stations:
                for _channel in channels:
                    files_filtered = df_files[(df_files['station'] == _station) & (df_files['channel'] == _channel)]
                    inventory_filtered = df_inventory[(df_inventory['station'] == _station)]
                    print(files_filtered)

                    st = Stream()

                    for index, _file in files_filtered.iterrows():
                        _st = read(_file['file'])
                        gaps = _st.get_gaps()

                        if len(gaps) > 0:
                            _st.print_gaps()
                        
                        st += _st

                    st.merge(fill_value="interpolate")

                    model = TauPyModel("iasp91")

                    distance_km, _, _ = gps2dist_azimuth(lat, lon, inventory_filtered['latitude'].tolist()[0], inventory_filtered['longitude'].tolist()[0])
                    distance_km = kilometer2degrees(distance_km/1000)

                # Calcular los tiempos de arribo para cada onda (P y S) para cada distancia
                    arrivals = model.get_travel_times(source_depth_in_km=depth, distance_in_degree=distance_km)
                
                
                    p_time = arrivals[0].time
                    s_time = None

            
                    start_cut = event["datetime"] + timedelta(seconds=p_time) - timedelta(seconds=start)
                    end_cut = event["datetime"] + timedelta(seconds=p_time) + timedelta(seconds=end)

                    file = files_filtered["file"].tolist()[0].split("/")

                    if end_cut is not None and start_cut is not None:
                        st.trim(UTCDateTime(start_cut), UTCDateTime(end_cut))
                        
                        if self.config_file:
                            sd = SeismogramData(st, self.inventory)
                            tr = sd.run_analysis(self.config_file)
                        else:
                            tr = st
                        
                        try:
                        #if config -> tratar config
                            tr.write(os.path.join(event_folder, file[len(file)-1]), 'mseed')

                        except:
                            print('Empty file')   



    def project_table(self):
        df_project = pd.DataFrame(columns=['file', 'start', 'end', 'net', 'station', 'channel'])
        _file = []
        _start = []
        _end = []
        _net = []
        _station = []
        _channel = []

        for project_files in self.files.data_files:
            _file.append(project_files[0])
            _start.append(project_files[1])
            _end.append(project_files[2])
            _net.append(self.find_stats(self.files.project, project_files[0], 'network'))
            _station.append(self.find_stats(self.files.project, project_files[0], 'station'))
            _channel.append(self.find_stats(self.files.project, project_files[0], 'channel'))
    
        df_project['file'] = _file
        df_project['start'] = _start
        df_project['end'] = _end
        df_project['net'] = _net
        df_project['station'] = _station
        df_project['channel'] = _channel

        return df_project

    @staticmethod
    def find_stats(lists, name, stat):
        _value = [key for key, list in lists.items() if any(name in sublist for sublist in list)]

        if len(_value) > 0:
            for i, sublist in enumerate(lists[_value[0]]):
                if name in sublist:
                    _i = sublist.index(name)
                    return lists[_value[0]][_i][1][stat]
    
        return None
        
    def inventory_table(self):
        df_inventory = pd.DataFrame(columns=['net', 'station', 'latitude', 'longitude'])
        net_inventory = []
        station_inventory = []
        latitude_inventory = []
        longitude_inventory = []

        for network in self.inventory.networks:
            for station in network.stations:
                net_inventory.append(network.code)
                station_inventory.append(station.code)
                latitude_inventory.append(station.latitude)
                longitude_inventory.append(station.longitude)

        df_inventory['net'] = net_inventory
        df_inventory['station'] = station_inventory
        df_inventory['latitude'] = latitude_inventory
        df_inventory['longitude'] = longitude_inventory

        return df_inventory


"""    for file in files.project:
        print(file)
        _net = files.project[file][0][1]['network']
        _station = files.project[file][0][1]['station']

        file_name.append(files.project[file][0][0])
        net_inventory.append(files.project[file][0][1]['network'])
        station_inventory.append(files.project[file][0][1]['station'])
        datetime_inventory.append(files.project[file][0][1]['starttime'])

        for network in inventory.networks:
            if network.code == _net:
                for station in network.stations:
                    if station.code == _station:
                        latitude_inventory.append(station.latitude)
                        longitude_inventory.append(station.longitude)

    df_inventory['file'] = file_name
    df_inventory['net'] = net_inventory
    df_inventory['station'] = station_inventory
    df_inventory['datetime'] = datetime_inventory
    df_inventory['latitude'] = latitude_inventory
    df_inventory['longitude'] = longitude_inventory
    """




