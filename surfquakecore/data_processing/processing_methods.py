#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
processing_methods

"""

import copy
import scipy, numpy as np
import math
import pywt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter, sosfiltfilt, bessel, ellip, cheby2, cheby1, sosfilt
from surfquakecore.utils.obspy_utils import Filters

def filter_trace(trace, type, fmin, fmax, **kwargs):
    """
        Filter an ObsPy Trace using standard and advanced filters.
        Supports Butterworth, Chebyshev I & II, Elliptic, Bessel.

        Parameters:
            trace: ObsPy Trace object.
            trace_filter: Filter type as string or enum.
            f_min, f_max: Bandpass frequency limits (Hz).
            corners: Number of poles (default: 4).
            zerophase: Apply filter forward and backward (default: True).
            ripple: Optional ripple for Chebyshev/Elliptic filters.

        Returns:
            True if success, False if bad parameters.
        """
    if type != Filters.Default:
        tf = type.lower()

        if tf in ["bandpass", "bandstop"]:
            if not (fmax > fmin):
                print("Bad frequency range: f_max must be > f_min for band filters.")
                return False

        elif tf in ["highpass"]:
            if fmin <= 0:
                print("Invalid highpass cutoff: f_min must be > 0.")
                return False

        elif tf in ["lowpass"]:
            if fmax <= 0:
                print("Invalid lowpass cutoff: f_max must be > 0.")
                return False

        corners = kwargs.pop("corners", 4)
        zerophase = kwargs.pop("zerophase", True)
        ripple = kwargs.pop("ripple", 1)  # in dB
        rp = kwargs.pop("rp", 0.5)  # Passband ripple
        rs = kwargs.pop("rs", 60)  # Stopband attenuation
        sr = trace.stats.sampling_rate
        nyq = 0.5 * sr
        wp = [fmin / nyq, fmax / nyq]  # normalized passband
        trace.detrend(type="simple")
        trace.taper(max_percentage=0.05, type="cosine")

        if type.lower() in ["bandpass", "bandstop"]:
            trace.filter(type, freqmin=fmin, freqmax=fmax, corners=corners, zerophase=zerophase)

        elif type.lower() in ["highpass"]:
            trace.filter(type, freq=fmin, corners=corners, zerophase=zerophase)

        elif type.lower() in ["lowpass"]:
            trace.filter(type, freq=fmax, corners=corners, zerophase=zerophase)

        elif type.lower() == "cheby1":

            sos = cheby1(N=corners, rp=ripple, Wn=wp, btype='band', output='sos')

            trace.data = sosfiltfilt(sos, trace.data) if zerophase else sosfilt(sos, trace.data)

        elif type.lower() == "cheby2":
            sos = cheby2(N=corners, rs=ripple, Wn=wp, btype='band', output='sos')
            trace.data = sosfiltfilt(sos, trace.data) if zerophase else sosfilt(sos, trace.data)

        elif type.lower() == "elliptic":
            sos = ellip(N=corners, rp=rp, rs=rs, Wn=wp, btype='band', output='sos')
            trace.data = sosfiltfilt(sos, trace.data) if zerophase else sosfilt(sos, trace.data)

        elif type.lower() == "bessel":
            # Warning: Bessel filters are not typically available in 'sos' format for all scipy versions
            sos = bessel(N=corners, Wn=wp, btype='band', norm='phase', output='sos')
            trace.data = sosfiltfilt(sos, trace.data) if zerophase else sosfilt(sos, trace.data)

        else:
            print(f"Unsupported filter type: {type}")
            return False

    return trace

def spectral_integration(trace, pad_to_pow2=True, taper_type='cosine', taper_pct=0.05):
    """
    Integrate a seismic signal in the frequency domain with preprocessing.

    Parameters:
        trace : ObsPy Trace object
        lp_freq : Lowpass filter cutoff frequency (Hz) to suppress drift #No implemented
        pad_to_pow2 : Zero-pad to next power of 2 (improves FFT efficiency)
        taper_type : 'cosine', 'hann', etc. taper before FFT
        taper_pct : Taper percentage (e.g., 0.05 = 5%)

    Returns:
        Modified trace (in-place)
    """
    # Copy original data for safety
    tr = trace.copy()

    # Remove mean and apply taper
    tr.detrend(type='demean')
    tr.taper(max_percentage=taper_pct, type=taper_type)

    # Original parameters
    n_orig = tr.stats.npts
    dt = tr.stats.delta

    # Zero-pad to next power of 2 for spectral resolution
    n_pad = 2 ** int(np.ceil(np.log2(n_orig))) if pad_to_pow2 else n_orig
    f = np.fft.rfftfreq(n_pad, d=dt)

    # FFT and scale spectrum
    spectrum = np.fft.rfft(tr.data, n=n_pad)
    scale = np.ones_like(f, dtype=np.complex128)
    scale[1:] = 1 / (1j * 2 * np.pi * f[1:])
    scale[0] = 0.0  # remove DC

    # Integrate in frequency domain
    spectrum_integrated = spectrum * scale
    tr.data = np.fft.irfft(spectrum_integrated, n=n_pad)[:n_orig]

    # # Final lowpass filter to remove accumulated drift (e.g., from low-f noise)
    # tr.filter("lowpass", freq=lp_freq, corners=4, zerophase=True)

    return tr


def spectral_derivative(trace, taper_pct=0.05, taper_type="cosine", pad_to_pow2=True):
    """
    Compute the first derivative of a signal in the frequency domain.

    Parameters:
        trace : ObsPy Trace object
        hp_freq : Optional highpass cutoff frequency (Hz) to remove amplified high-frequency noise
        taper_pct : Taper percentage (e.g., 0.05 = 5%)
        taper_type : Taper window type ('cosine', 'hann', etc.)
        pad_to_pow2 : Whether to zero-pad to next power of 2 before FFT

    Returns:
        Modified trace with differentiated signal
    """
    tr = trace.copy()

    # Demean and taper
    tr.detrend(type="demean")
    tr.taper(max_percentage=taper_pct, type=taper_type)

    # Set up time parameters
    n_orig = tr.stats.npts
    dt = tr.stats.delta
    fs = 1 / dt
    n_pad = 2 ** int(np.ceil(np.log2(n_orig))) if pad_to_pow2 else n_orig

    # Frequency array
    freqs = np.fft.rfftfreq(n_pad, d=dt)

    # FFT of the signal
    spectrum = np.fft.rfft(tr.data, n=n_pad)

    # Differentiate in frequency domain: d/dt <-> i·2πf
    spectrum_diff = (1j * 2 * np.pi * freqs) * spectrum
    tr.data = np.fft.irfft(spectrum_diff, n=n_pad)[:n_orig]

    # Optional post-highpass filter to reduce HF noise if needed
    # if hp_freq is not None:
    #     tr.filter("highpass", freq=hp_freq, corners=4, zerophase=True)

    return tr


def wiener_filter(tr, time_window, noise_power):
    data = tr.data

    if time_window == 0 and noise_power == 0:

        denoise = scipy.signal.wiener(data, mysize=None, noise=None)
        tr.data = denoise

    elif time_window != 0 and noise_power == 0:

        denoise = scipy.signal.wiener(data, mysize=int(time_window * tr.stats.sampling_rate), noise=None)
        tr.data = denoise

    elif time_window == 0 and noise_power != 0:

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


def add_frequency_domain_noise(trace, noise_type='white', SNR_dB=None, seed=None):
    """
    Adds white or colored noise (frequency-shaped) to an ObsPy Trace.

    Parameters:
    ----------
    trace : obspy.Trace
        Trace object to which noise will be added (in-place).
    noise_type : str
        Type of noise: 'white', 'pink', 'brown', 'blue' or 'violet'.
    exponent : float
        Exponent for custom noise: noise PSD ~ 1/f^exponent (used if noise_type='custom').
        Examples:
            - white  → 0
            - pink   → 1
            - brown  → 2
            - blue   → -1
            - violet → -2
    SNR_dB : float or None
        If given, the noise will be scaled to achieve the specified signal-to-noise ratio in dB.
    seed : int or None
        Seed for reproducibility.

    Returns:
    --------
    trace : obspy.Trace
        Trace object with added noise.
    """

    if seed is not None:
        np.random.seed(seed)

    # Get signal parameters
    npts = len(trace.data)
    delta = trace.stats.delta
    signal = trace.data.astype(np.float64)
    freqs = np.fft.rfftfreq(npts, d=delta)

    # Choose exponent from predefined types
    type_to_exponent = {
        'white': 0,
        'pink': 1,
        'brown': 2,
        'blue': -1,
        'violet': -2
    }

    if noise_type not in type_to_exponent:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    alpha = type_to_exponent[noise_type]

    # Generate white noise
    white = np.random.randn(npts)
    white_fft = np.fft.rfft(white)

    # Shape the spectrum
    scale = np.where(freqs == 0, 1.0, freqs ** (alpha / 2))
    shaped_fft = white_fft / scale

    # Back to time domain
    noise = np.fft.irfft(shaped_fft, n=npts)

    # Match SNR if requested
    if SNR_dB is not None:
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        target_noise_power = signal_power / (10 ** (SNR_dB / 10))
        noise *= np.sqrt(target_noise_power / noise_power)

    # Add to trace
    trace.data = signal + noise
    return trace


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

        data_f_whiten[0:half_width] = ((data_f[0:half_width]) / diff_mean) * wf  # First part of spectrum tapered
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
    # indices = []
    new_series = input_series.copy()
    # possibly use np.nanmedian
    for i in range((window_size), (size - window_size)):
        x0 = np.median(input_series[(i - window_size):(i + window_size)])
        S0 = k * np.median(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
        if (np.abs(input_series[i] - x0) > n_sigmas * S0):
            new_series[i] = x0
            # indices.append(i)

    return new_series


def normalize(tr, clip_factor=6, clip_weight=10, norm_win=10, norm_method="1bit"):
    if norm_method == 'clipping':
        lim = clip_factor * np.std(tr.data)
        tr.data[tr.data > lim] = lim
        tr.data[tr.data < -lim] = -lim

    elif norm_method == "clipping iteration":
        lim = clip_factor * np.std(np.abs(tr.data))

        # as long as still values left above the waterlevel, clip_weight
        while tr.data[np.abs(tr.data) > lim] != []:
            tr.data[tr.data > lim] /= clip_weight
            tr.data[tr.data < -lim] /= clip_weight

    elif norm_method == 'time normalization':
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
        # taper = get_window(tr.stats.npts)
        # tr.data *= taper

    elif norm_method == "1bit":
        tr.data = np.sign(tr.data)
        tr.data = np.float32(tr.data)

    return tr


def wavelet_denoise(tr, threshold=0.04, dwt='sym4'):
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
    if type == 'mean':
        # new fast and simple implementation
        k = int(k * tr.stats.sampling_rate)
        kernel = np.ones(k) / k
        tr.data = np.convolve(tr.data, kernel, mode='same')

    if type == 'gaussian':
        # new fast and simple implementation

        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        sigma_samples = sigma * tr.stats.sampling_rate
        tr.data = gaussian_filter1d(tr.data, sigma=sigma_samples)

    if type == "adaptive":
        window_length = int(k * tr.stats.sampling_rate)
        if window_length % 2 == 0:
            window_length += 1  # Ensure odd window length
        tr.data = savgol_filter(tr.data, window_length, polyorder=3, mode='nearest')

    if type == 'tkeo':
        emg = tr.data
        emgf = copy.deepcopy(emg)
        emgf[1:-1] = emg[1:-1] ** 2 - emg[0:-2] * emg[2:]

        tr.data = emgf

    return tr


def safe_downsample(trace, target_rate, max_factor=10, pre_filter=True, tolerance=0.01):
    """
    Downsample a trace to target_rate in steps, applying filtering.

    Parameters:
    - trace: ObsPy Trace object
    - target_rate: Final desired sampling rate (Hz)
    - max_factor: Maximum downsample ratio per step (e.g., 10)
    - pre_filter: Apply anti-aliasing lowpass filter before each step
    - tolerance: Allowable fractional difference from target_rate
    """
    from obspy import Trace
    import copy

    tr = trace.copy()

    while tr.stats.sampling_rate > target_rate * (1 + tolerance):
        current_rate = tr.stats.sampling_rate
        factor = min(current_rate / target_rate, max_factor)
        new_rate = current_rate / factor
        f_nyquist = new_rate * 0.5

        if pre_filter:
            tr.detrend("demean")
            tr.taper(max_percentage=0.05, type="hamming")
            tr.filter("lowpass", freq=0.9 * f_nyquist, corners=4, zerophase=True)

        tr.resample(new_rate)

    return tr

