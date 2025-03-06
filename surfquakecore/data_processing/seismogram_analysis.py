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

def parse_configuration_file(config:dict):
    _config = {}
    _config_template = []
    _config_keys = []

    print(config)

    if 'Analysis' in config:
        _config = config['Analysis']
        _config_keys = _config.keys()
        _rmean_keys = []
        i=0
        _process = []

        for i in range(len(_config_keys)):
            _name= 'process_' + str(i+1)
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
                    #Check type

        return _config_template
    else:
        raise ValueError(f"analysis_config {config} is not valid. It must be a valid .yaml file for "
                         f"AnalysisConfig. 'Analysis' header is missing")

def load_analysis_configuration(config_file:str):
    rs = ReadSource(config_file)
    config = rs.read_file(config_file)

    return parse_configuration_file(config)

class AnalysisParameters(BaseDataClass):

    def __init__(self, analysis_config):
        self.config_analysis_template = analysis_config
        self.catalog = None
        self.analysis_template_configuration = {}

        if isinstance(analysis_config, str) and os.path.isfile(analysis_config):
            self.analysis_template_configuration = load_analysis_configuration(analysis_config)
        else:
            raise ValueError(f"analysis_config {analysis_config} is not valid. It must be a valid .yaml file for "
                             f"AnalysisConfig")

class SeismogramData:

    def __init__(self, file_path,  realtime = False, **kwargs):

        #mas de un fichero

        stream = kwargs.pop('stream', [])

        #self.config_file = config_file
        self.config_keys = None
        #self.output = output_path

        if file_path:
            self.st = read(file_path)
            print(self.st)

        if realtime:
            self.__tracer = stream

        else:
            gaps = self.st.get_gaps()

            if len(gaps) > 0:
                self.st.print_gaps()
                self.st.merge(fill_value="interpolate")

            self.__tracer = self.st[0]

        self.stats = TracerStatsAnalysis.from_dict(self.tracer.stats)

    @classmethod
    def from_tracer(cls, tracer):
        sd = cls(None)
        sd.set_tracer(tracer)
        return sd

    @property
    def tracer(self):
        return self.__tracer

    def set_tracer(self, tracer):
        self.__tracer = tracer
        self.stats = TracerStats.from_dict(self.__tracer.stats)

    def __send_filter_error_callback(self, func, msg):
        if func:
            func(msg)

    def resample_check(self, start_time=None, end_time=None):

        decimator_factor = None
        check = False

        if start_time is not None and end_time is not None:
            start = start_time
            end = end_time
        else:
            start = self.stats.StartTime
            end = self.stats.EndTime

        diff = end - start

        lim1 = 3600 * 6
        lim2 = 3600 * 3

        if diff >= lim1:
            check = True
            decimator_factor = 1
            return [decimator_factor, check]

        if diff >= lim2 and diff < lim1:
            check = True
            decimator_factor = 5

            return [decimator_factor, check]

        else:
            return [decimator_factor, check]

    def run_analysis(self, config, **kwargs):

        """
        This method should be to loop over config files and run the inversion.
        Previously it is needed to load the project and metadata.

        Args:
            analysis_config: a .yaml file

        Returns:

        """

        start_time = kwargs.get("start_time", self.stats.StartTime)
        end_time = kwargs.get("end_time", self.stats.EndTime)
        trace_number = kwargs.get("trace_number", 0)
        tr = self.tracer

        tr.trim(starttime=start_time, endtime=end_time)

        for i in range(len(config)):
            _config = config[i]
            _keys = config[i]

            if _config['name'] == 'rmean':
                if _config['method'] in ['linear, simple', 'demean']:
                    tr.detrend(type=_config['method'])
                elif _config['method'] == 'polynomial':
                    tr.detrend(type=_config['method'], order=_config['order'])
                elif _config['method'] == 'spline':
                    tr.detrend(type=_config['method'], order=_config['order'], dspline=_config['dspline'])


            if _config['name'] == 'taper':
                tr.taper(max_percentage=_config['max_percentage'], type=_config['method'],
                         max_length=_config['max_length'], side=_config['side'])

            if _config['name'] == 'normalize':
                if isinstance(_config['norm'], bool):
                    tr.normalize()
                else:
                    tr.normalize(norm=_config['norm'])

            if _config['name'] == 'differentiate':
                tr.differentiate()

            if _config['name'] == 'integrate':
                tr.integrate(method=_config['method'])

            if _config['name'] == 'filter':
                options = copy.deepcopy(_config)
                del options['method']
                del options['name']
                tr.filter(type=_config['method'], **options)

            if _config['name'] == 'wiener_filter':
                tr = wiener_filter(tr, time_window=_config['time_window'],
                                   noise_power=_config['noise_power'])
            #Borrar shift
            #if _config['name'] == 'shift':
            #    shifts = self.config_file['shift']
            #    i = 0
            #    for i in range(0,len(shifts)):
            #        tr.stats.starttime = tr.stats.starttime + shifts[i]['time']

            if _config['name'] == 'remove_response':
                inventory = read_inventory(_config['inventory'])
                print(inventory)
                if _config['units'] != "Wood Anderson":
                    # print("Deconvolving")
                    try:
                        tr.remove_response(inventory=inventory, pre_filt=_config['pre_filt'],
                                           output=_config['units'], water_level=_config['water_level'])
                    except:
                        print("Coudn't deconvolve", tr.stats)
                        tr.data = np.array([])

                elif _config['units'] == "Wood Anderson":
                    # print("Simulating Wood Anderson Seismograph")
                    if inventory is not None:
                        resp = inventory.get_response(tr.id, tr.stats.starttime)

                        resp = resp.response_stages[0]
                        paz_wa = {'sensitivity': 2800, 'zeros': [0j], 'gain': 1,
                              'poles': [-6.2832 - 4.7124j, -6.2832 + 4.7124j]}

                        paz_mine = {'sensitivity': resp.stage_gain * resp.normalization_factor, 'zeros': resp.zeros,
                                'gain': resp.stage_gain, 'poles': resp.poles}

                        try:
                            tr.simulate(paz_remove=paz_mine, paz_simulate=paz_wa,
                                        water_level=_config['water_level'])
                        except:
                            print("Coudn't deconvolve", tr.stats)
                            tr.data = np.array([])

            if _config['name'] == 'add_white_noise':
                tr = add_white_noise(tr,_config['SNR_dB'])

            if _config['name'] == 'whitening':
                tr = whiten(tr, _config['freq_width'], taper_edge = _config['taper_edge'])

            if _config['name'] == 'remove_spikes':
                tr = hampel(tr, _config['window_size'], _config['n'])

            if _config['name'] == 'time_normalization':
                tr = normalize(tr, norm_win=_config['norm_win'], norm_method=_config['method'])

            if _config['name'] == 'wavelet_denoise':
                tr = wavelet_denoise(tr, dwt = _config['dwt'], threshold=_config['threshold'])

            if _config['name'] == 'resample':
                tr.resample(sampling_rate=_config['sampling_rate'],window='hann',no_filter=_config['pre_filter'])

            if _config['name'] == 'fill_gaps':
                st = Stream(tr)
                st.merge(fill_value=_config['method'])
                tr = st[0]

            if _config['name'] == 'smoothing':
                tr = smoothing(tr, type=_config['method'], k=_config['time_window'], fwhm=_config['FWHM'])

        #tr.id = 'test_1_' + tr.id
        #tr.write(self.output + 'test1', 'mseed')

        return tr

    def get_waveform_advanced(self, parameters, inventory=None, filter_error_callback=None, **kwargs):

        start_time = kwargs.get("start_time", self.stats.StartTime)
        end_time = kwargs.get("end_time", self.stats.EndTime)
        trace_number = kwargs.get("trace_number", 0)
        tr = self.tracer

        tr.trim(starttime=start_time, endtime=end_time)

        # Detrend
        if parameters.rmean is not None:
            if parameters.rmean in ['linear, simple', 'demean']:
                tr.detrend(type=parameters.rmean)
            elif parameters.rmean is 'polynomial':
                tr.detrend(type=parameters.rmean, order=parameters.order)
            elif parameters.rmean is 'spline':
                tr.detrend(type=parameters.rmean, order=parameters.order, dspline=parameters.dspline)

        # Taper

        if parameters.taper is not None:
            tr.taper(type=parameters.taper, max_percentage=0.5)



        return tr