from operator import truediv

from obspy import read, Stream, read_inventory
from tensorflow.python.ops.random_ops import parameterized_truncated_normal
import os
import copy
import scipy, numpy as np
import math
import pywt

from surfquakecore.Structures.structures import TracerStatsAnalysis, TracerStats
from surfquakecore.utils.obspy_utils import ObspyUtil, Filters
from dataclasses import dataclass, field
from datetime import datetime
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



def check_rmean(config):
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

def check_taper(config):
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

def check_normalize(config):
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

def check_shift(config):
    i = 0
    if len(config) > 0:
        for i in range(0, len(config)):
            _config_keys = config[i].keys()

            if 'name' in _config_keys and 'time' in _config_keys:
                return True
            else:
                raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                                 f"AnalysisConfig")

def check_integrate(config):
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

def check_filter(config):
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

def check_remove_response(config):
    _config_keys = config.keys()

    if ('inventory' in _config_keys and 'water_level' in _config_keys and 'units' in _config_keys
        and 'pre_filt' in _config_keys):

        return True
    else:
        raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                         f"AnalysisConfig")

def check_differentiate(config):
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

def check_wiener_filter(config):
    _config_keys = config.keys()

    if 'time_window' in _config_keys and 'noise_power' in _config_keys:
        return True
    else:
        raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                         f"AnalysisConfig")

def wiener_filter(tr, time_window, noise_power):
    data = tr.data


    if time_window == 0 and noise_power == 0:

       denoise = scipy.signal.wiener(data, mysize=None, noise=None)
       tr.data = denoise

    elif time_window!= 0 and noise_power == 0:

        denoise = scipy.signal.wiener(data, mysize= int(time_window*tr.stats.sampling_rate), noise=None)
        tr.data = denoise

    elif time_window == 0 and noise_power !=0:

         noise = noise_power * np.std(data)
         noise = int(noise)
         denoise = scipy.signal.wiener(data, mysize=None, noise=noise)
         tr.data = denoise

    elif time_window != 0 and noise_power != 0:

         noise = noise_power * np.std(data)
         noise = int(noise)
         denoise = scipy.signal.wiener(data, mysize=int(time_window * tr.stats.sampling_rate), noise=noise)
         tr.data = denoise

    return tr

def check_add_white_noise(config):
    _config_keys = config.keys()

    if 'SNR_dB' in _config_keys:
        return True
    else:
        raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                         f"AnalysisConfig")

def check_whiten(config):
    _config_keys = config.keys()

    if 'taper_edge' in _config_keys:
        if isinstance(config['taper_edge'], bool):
            return True
        else:
            raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig")
    else:
        return True

def check_remove_spikes(config):
    _config_keys = config.keys()

    if 'window_size' in _config_keys:
        return True
    else:
        raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                         f"AnalysisConfig")

def check_time_normalization(config):
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

def check_wavelet_denoise(config):
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

def check_resample(config):
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

def check_fill_gaps(config):
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

def check_smoothing(config):
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

def add_white_noise(tr, SNR_dB):
    L = len(tr.data)
    SNR = 10**(SNR_dB/10)
    Esym = np.sum(np.abs(tr.data)**2)/L
    N0 = Esym / SNR
    noiseSigma = np.sqrt(N0)
    n = noiseSigma * np.random.normal(size=L)
    tr.data = tr.data+ n
    return tr

def whiten(tr, freq_width=0.05, taper_edge=True):

    """"
    freq_width: Frequency smoothing windows [Hz] / both sides
    taper_edge: taper with cosine window  the low frequencies

    return: whithened trace (Phase is not modified)
    """""

    fs = tr.stats.sampling_rate
    N = tr.count()
    D = 2 ** math.ceil(math.log2(N))
    freq_res = 1 / (D / fs)
    # N_smooth = int(freq_width / (2 * freq_res))
    N_smooth = int(freq_width / (freq_res))

    if N_smooth % 2 == 0:  # To have a central point
        N_smooth = N_smooth + 1
    else:
        pass

    # avarage_window_width = (2 * N_smooth + 1) #Denominador
    avarage_window_width = (N_smooth + 1)  # Denominador
    half_width = int((N_smooth + 1) / 2)  # midpoint
    half_width_pos = half_width - 1

    # Prefilt
    tr.detrend(type='simple')
    tr.taper(max_percentage=0.05)

    # ready to whiten
    data = tr.data
    data_f = np.fft.rfft(data, D)
    freq = np.fft.rfftfreq(D, 1. / fs)
    N_rfft = len(data_f)
    data_f_whiten = data_f.copy()
    index = np.arange(0, N_rfft - half_width, 1)

    data_f_whiten = whiten_aux(data_f, data_f_whiten, index, half_width, avarage_window_width, half_width_pos)

    # Taper (optional) and remove mean diffs in edges of the frequency domain

    wf = (np.cos(np.linspace(np.pi / 2, np.pi, half_width)) ** 2)

    if taper_edge:

        diff_mean = np.abs(np.mean(np.abs(data_f[0:half_width])) - np.mean(np.abs(data_f_whiten[half_width:])) * wf)

    else:

        diff_mean = np.abs(np.mean(np.abs(data_f[0:half_width])) - np.mean(np.abs(data_f_whiten[half_width:])))

    diff_mean2 = np.abs(
        np.mean(np.abs(data_f[(N_rfft - half_width):])) - np.mean(np.abs(data_f_whiten[(N_rfft - half_width):])))

    if taper_edge:

        data_f_whiten[0:half_width] = ((data_f[0:half_width]) / diff_mean)*wf  # First part of spectrum tapered
    else:

        data_f_whiten[0:half_width] = ((data_f[0:half_width]) / diff_mean)


    data_f_whiten[(N_rfft - half_width):] = (data_f[(N_rfft - half_width):]) / diff_mean2  # end of spectrum
    data = np.fft.irfft(data_f_whiten)
    data = data[0:N]
    tr.data = data

    return tr

def whiten_aux(data_f, data_f_whiten, index, half_width, avarage_window_width, half_width_pos):
    return __whiten_aux(data_f, data_f_whiten, index, half_width, avarage_window_width, half_width_pos)

def __whiten_aux(data_f, data_f_whiten, index, half_width, avarage_window_width, half_width_pos):
    for j in index:
        den = np.sum(np.abs(data_f[j:j + 2 * half_width])) / avarage_window_width
        data_f_whiten[j + half_width_pos] = data_f[j + half_width_pos] / den
    return data_f_whiten

def hampel_aux(input_series, window_size, size, n_sigmas):
    return __hampel_aux(input_series, window_size, size, n_sigmas)

def hampel(tr, window_size, n_sigmas=3):
    """
            Median absolute deviation (MAD) outlier in Time Series
            :param ts: a trace obspy object representing the timeseries
            :param window_size: total window size in seconds
            :param n: threshold, default is 3 (Pearson's rule)
            :return: Returns the corrected timeseries
            """

    size = tr.count()
    input_series = tr.data
    window_size = int(window_size * tr.stats.sampling_rate)
    tr.data = hampel_aux(input_series, window_size, size, n_sigmas)

    return tr

def __hampel_aux(input_series, window_size, size, n_sigmas):

    k = 1.4826  # scale factor for Gaussian distribution
    #indices = []
    new_series = input_series.copy()
    # possibly use np.nanmedian
    for i in range((window_size), (size - window_size)):
        x0 = np.median(input_series[(i - window_size):(i + window_size)])
        S0 = k * np.median(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
        if (np.abs(input_series[i] - x0) > n_sigmas * S0):
            new_series[i] = x0
            #indices.append(i)

    return new_series

def normalize(tr, clip_factor=6, clip_weight=10, norm_win=None, norm_method="1bit"):
    if norm_method == 'clipping':
        lim = clip_factor * np.std(tr.data)
        tr.data[tr.data > lim] = lim
        tr.data[tr.data < -lim] = -lim

    elif norm_method == "clipping_iteration":
        lim = clip_factor * np.std(np.abs(tr.data))

        # as long as still values left above the waterlevel, clip_weight
        while tr.data[np.abs(tr.data) > lim] != []:
            tr.data[tr.data > lim] /= clip_weight
            tr.data[tr.data < -lim] /= clip_weight

    elif norm_method == 'time_normalization':
        lwin = int(tr.stats.sampling_rate * norm_win)
        st = 0  # starting point
        N = lwin  # ending point

        while N < tr.stats.npts:
            win = tr.data[st:N]

            w = np.mean(np.abs(win)) / (2. * lwin + 1)

            # weight center of window
            tr.data[st + int(lwin / 2)] /= w

            # shift window
            st += 1
            N += 1

        # taper edges
        #taper = get_window(tr.stats.npts)
        #tr.data *= taper

    elif norm_method == "1bit":
        tr.data = np.sign(tr.data)
        tr.data = np.float32(tr.data)

    return tr

def wavelet_denoise(tr, threshold = 0.04, dwt = 'sym4' ):
    # Threshold for filtering
    # Create wavelet object and define parameters
    w = pywt.Wavelet(dwt)
    maxlev = pywt.dwt_max_level(len(tr.data), w.dec_len)
    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(tr.data, dwt, level=maxlev)
    # cA = pywt.threshold(cA, threshold*max(cA))
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
    datarec = pywt.waverec(coeffs, dwt)
    tr.data = datarec
    return tr

def smoothing(tr, type='gaussian', k=5, fwhm=0.05):
    # k window size in seconds

    n = len(tr.data)

    if type == 'mean':
        k = int(k * tr.stats.sampling_rate)

        # initialize filtered signal vector
        filtsig = np.zeros(n)
        for i in range(k, n - k - 1):
            # each point is the average of k surrounding points
            # print(i - k,i + k)
            filtsig[i] = np.mean(tr.data[i - k:i + k])

        tr.data = filtsig

    if type == 'gaussian':
        ## create Gaussian kernel
        # full-width half-maximum: the key Gaussian parameter in seconds
        # normalized time vector in seconds
        k = int(k * tr.stats.sampling_rate)
        fwhm = int(fwhm * tr.stats.sampling_rate)
        gtime = np.arange(-k, k)
        # create Gaussian window
        gauswin = np.exp(-(4 * np.log(2) * gtime ** 2) / fwhm ** 2)
        # compute empirical FWHM

        pstPeakHalf = k + np.argmin((gauswin[k:] - .5) ** 2)
        prePeakHalf = np.argmin((gauswin - .5) ** 2)
        # empFWHM = gtime[pstPeakHalf] - gtime[prePeakHalf]
        # show the Gaussian
        # plt.plot(gtime/tr.stats.sampling_rate,gauswin)
        # plt.plot([gtime[prePeakHalf],gtime[pstPeakHalf]],[gauswin[prePeakHalf],gauswin[pstPeakHalf]],'m')
        # then normalize Gaussian to unit energy
        gauswin = gauswin / np.sum(gauswin)
        # implement the filter
        # initialize filtered signal vector
        filtsigG = copy.deepcopy(tr.data)
        # implement the running mean filter
        for i in range(k + 1, n - k - 1):
            # each point is the weighted average of k surrounding points
            filtsigG[i] = np.sum(tr.data[i - k:i + k] * gauswin)

        tr.data = filtsigG

    if type == 'tkeo':
        # extract needed variables

        emg = tr.data

        # initialize filtered signal
        emgf = copy.deepcopy(emg)

        # the loop version for interpretability
        # for i in range(1, len(emgf) - 1):
        #    emgf[i] = emg[i] ** 2 - emg[i - 1] * emg[i + 1]

        # the vectorized version for speed and elegance

        emgf[1:-1] = emg[1:-1] ** 2 - emg[0:-2] * emg[2:]

        ## convert both signals to zscore

        # find timepoint zero
        # time0 = np.argmin(emgtime ** 2)

        # convert original EMG to z-score from time-zero
        # emgZ = (emg - np.mean(emg[0:time0])) / np.std(emg[0:time0])

        # same for filtered EMG energy
        # emgZf = (emgf - np.mean(emgf[0:time0])) / np.std(emgf[0:time0])
        # tr.data = emgZf
        tr.data = emgf

    return tr

def parse_filter(config):
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

def parse_taper(config):
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

def parse_whiten(config):
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

def parse_remove_spikes(config):
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

def parse_configuration_file(config: dict):
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
                            if check_rmean(_process[i]):
                                _config_template.append(_process[i])

                        if _process_config[_key] == 'taper':
                            if check_taper(_process[i]):
                                _config_template.append(parse_taper(_process[i]))

                        if _process_config[_key] == 'normalize':
                            if check_normalize(_process[i]):
                                _config_template.append(_process[i])

                        if _process_config[_key] == 'differentiate':
                            if check_differentiate(_process[i]):
                                _config_template.append(_process[i])

                        if _process_config[_key] == 'integrate':
                            if check_integrate(_process[i]):
                                _config_template.append(_process[i])

                        if _process_config[_key] == 'filter':
                            if check_filter(_process[i]):
                                _config_template.append(parse_filter(_process[i]))

                        if _process_config[_key] == 'wiener_filter':
                            if check_wiener_filter(_process[i]):
                                _config_template.append(_process[i])

                        if _process_config[_key] == 'shift':
                            if check_shift(_process[i]):
                                _config_template.append(_process[i])

                        if _process_config[_key] == 'remove_response':
                            if check_remove_response(_process[i]):
                                _config_template.append(_process[i])

                        if _process_config[_key] == 'add_white_noise':
                            if check_add_white_noise(_process[i]):
                                _config_template.append(_process[i])
                        if _process_config[_key] == 'whitening':
                            if check_whiten(_process[i]):
                                _config_template.append(parse_whiten(_process[i]))

                        if _process_config[_key] == 'remove_spikes':
                            if check_remove_spikes(_process[i]):
                                _config_template.append(parse_remove_spikes(_process[i]))

                        if _process_config[_key] == 'time_normalization':
                            if check_time_normalization(_process[i]):
                                _config_template.append(_process[i])

                        if _process_config[_key] == 'wavelet_denoise':
                            if check_wavelet_denoise(_process[i]):
                                _config_template.append(_process[i])

                        if _process_config[_key] == 'resample':
                            if check_resample(_process[i]):
                                _config_template.append(_process[i])

                        if _process_config[_key] == 'fill_gaps':
                            if check_fill_gaps(_process[i]):
                                _config_template.append(_process[i])

                        if _process_config[_key] == 'smoothing':
                            if check_smoothing(_process[i]):
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

    def __init__(self, config_file, file_path, output_path):
        self.config_file = self.load_analysis_configuration(config_file)
        self.output = output_path
        self.files = self.get_project_files(file_path)
        print('hola')



    def load_analysis_configuration(self, config_file: str):
        rs = ReadSource(config_file)
        config = rs.read_file(config_file)

        return parse_configuration_file(config)

    def get_project_files(self, project_path):
        freeze_support()
        sp = SurfProject(project_path)
        sp.search_files()
        return sp.project

    def run_analysis(self):




