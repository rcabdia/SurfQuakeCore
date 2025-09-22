#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
processing_methods

"""

import copy
import os
import scipy, numpy as np
import math
import pywt
from matplotlib import pyplot as plt
from obspy import UTCDateTime, Trace, Stream, read
from obspy.signal.cross_correlation import correlate_template
from obspy.signal.polarization import flinn
from obspy.signal.trigger import classic_sta_lta, recursive_sta_lta, z_detect
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter, sosfiltfilt, bessel, ellip, cheby2, cheby1, sosfilt, periodogram
from surfquakecore.utils.obspy_utils import Filters
from obspy.signal.filter import envelope
try:
    from surfquakecore.cython_module.whiten import whiten_aux
except:
    print("Whitenning no compiled. Install gcc compiler and reinstall surfquake")
from typing import Union, Sequence

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

def whiten_new(tr, freq_width=0.02, taper_edge=True):
    """
    Apply spectral whitening to a seismic trace.

    Parameters:
    -----------
    tr : obspy.Trace
        Input trace to be whitened.
    freq_width : float, optional
        Frequency smoothing window width [Hz] on both sides. Default is 0.02 Hz.

    Returns:
    --------
    obspy.Trace
        Whitened trace (amplitude spectrum flattened; phase preserved).
    """
    try:
        fs = tr.stats.sampling_rate
        N = tr.count()
        D = 2 ** math.ceil(math.log2(N))
        freq_res = 1 / (D / fs)
        N_smooth = max(1, int(freq_width / freq_res))  # Ensure at least 1

        if N_smooth % 2 == 0:
            N_smooth += 1  # Ensure odd window for symmetric smoothing

        average_window_width = N_smooth + 1
        half_width = average_window_width // 2
        half_width_pos = half_width - 1

        # FFT of zero-padded trace
        data = tr.data
        data_f = np.fft.rfft(data, D)
        N_rfft = len(data_f)
        data_f_whiten = data_f.copy()

        index = np.arange(0, N_rfft - half_width)

        # Call Cython/Numba whitening function
        data_f_whiten = whiten_aux(
            data_f, data_f_whiten, index, half_width,
            average_window_width, half_width_pos
        )

        # Taper edges in frequency domain
        if taper_edge:
            taper = np.cos(np.linspace(np.pi / 2, np.pi, half_width)) ** 2
            taper_flip = taper[::-1]

            mean_start = np.mean(np.abs(data_f[:half_width]))
            mean_end = np.mean(np.abs(data_f[-half_width:]))

            if mean_start != 0:
                data_f_whiten[:half_width] = (data_f[:half_width] / mean_start) * taper
            if mean_end != 0:
                data_f_whiten[-half_width:] = (data_f[-half_width:] / mean_end) * taper_flip

        # Inverse FFT and trim to original length
        tr.data = np.fft.irfft(data_f_whiten)[:N]
    except Exception as e:
        print(f"Whitening failed: {e}")

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

def trace_envelope(tr, method = "FULL", **kwargs):
    corner_freq = kwargs.pop("corner_freq", 0.15)
    N = len(tr.data)
    D = 2 ** math.ceil(math.log2(N))
    z = np.zeros(D - N)
    data = np.concatenate((tr.data, z), axis=0)

    ###Necesary padding with zeros
    data_envelope = envelope(data)
    data_envelope = data_envelope[0:N]
    tr.data = data_envelope

    if method == "SMOOTH":
        tr.detrend(type="simple")
        tr.taper(max_percentage=0.05)
        tr.filter(type='lowpass',freq=corner_freq, corners=3, zerophase=True)

    return tr

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

def trim_trace(trace, mode: str, *args):

    """
    Trim a single trace based on a mode: 'reference', 'phase', or 'absolute'.

    Parameters
    ----------
    trace : obspy.Trace
        The trace to be trimmed.

    mode : str
        One of: 'reference', 'phase', or 'absolute'.

    *args : tuple
        - If mode == 'reference': (time_before, time_after)
        - If mode == 'phase': (phase_name, time_before, time_after)
        - If mode == 'absolute': (start_time_str, end_time_str)
    example_usage:
        trimmed = trim_trace(tr, "reference", 10, 30)
        trimmed = trim_trace(tr, "phase", "P", 5, 20)
        trimmed = trim_trace(tr, "absolute", "2025-06-19 12:00:00", "2025-06-19 12:03:00")

    Returns
    -------
    obspy.Trace
        A trimmed copy of the original trace.

    Raises
    ------
    ValueError if mode or arguments are invalid, or trimming cannot be done.
    """

    if mode == "reference":
        if len(args) != 2:
            raise ValueError("Expected: trim_trace(trace, 'reference', time_before, time_after)")
        time_before, time_after = float(args[0]), float(args[1])
        references = getattr(trace.stats, "references", [])
        if not references:
            raise ValueError("No references found in trace.stats.references")
        ref_time = references[-1]
        t1 = ref_time - time_before
        t2 = ref_time + time_after

    elif mode == "phase":
        if len(args) != 3:
            raise ValueError("Expected: trim_trace(trace, 'phase', phase_name, time_before, time_after)")
        phase_name = args[0]
        time_before, time_after = float(args[1]), float(args[2])
        picks = getattr(trace.stats, "picks", [])
        phase_time = next((p["time"] for p in picks if p.get("phase") == phase_name), None)
        if not phase_time:
            raise ValueError(f"Phase '{phase_name}' not found in trace.stats.picks")
        t1 = phase_time - time_before
        t2 = phase_time + time_after

    elif mode == "absolute":
        if len(args) != 2:
            raise ValueError("Expected: trim_trace(trace, 'absolute', start_time_str, end_time_str)")
        try:
            t1 = UTCDateTime(args[0])
            t2 = UTCDateTime(args[1])
        except Exception as e:
            raise ValueError(f"Invalid datetime format: {e}")

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    try:
        return trace.copy().trim(starttime=t1, endtime=t2, pad=True, fill_value=0)
    except Exception as e:
        raise ValueError(f"Trimming failed: {e}")

def compute_entropy_trace(tr: Trace, win: float = 2.0, overlap: float = 0.5,
                          normalize: bool = True) -> Trace:
    """
    Compute spectral entropy over sliding overlapping windows and return as a Trace
    resampled to match the original trace length.

    Parameters
    ----------
    tr : obspy.Trace
        Input seismic trace.
    win : float
        Window length in seconds.
    overlap : float
        Fractional overlap between windows (e.g., 0.5 for 50% overlap).
    normalize : bool
        Normalize entropy values between 0 and 1.
    plot : bool
        Show a plot of the entropy time series.

    Returns
    -------
    obspy.Trace
        A new trace with the spectral entropy time series resampled to match the input trace.
    """
    tr = tr.copy()
    tr.detrend("linear")
    data = tr.data
    dt = tr.stats.delta
    fs = 1.0 / dt
    npts = len(data)

    win_samples = int(win / dt)
    step_samples = max(1, int(win_samples * (1 - overlap)))

    entropy_vals = []
    entropy_times = []

    for start in range(0, npts - win_samples + 1, step_samples):
        segment = data[start:start + win_samples]
        _, psd = periodogram(segment, fs=fs, window="hamming")
        psd_norm = psd / np.sum(psd)
        psd_norm = np.where(psd_norm == 0, 1e-12, psd_norm)  # avoid log(0)
        entropy = -np.sum(psd_norm * np.log2(psd_norm))
        if normalize:
            entropy /= np.log2(len(psd_norm))

        entropy_vals.append(entropy)
        entropy_times.append(start + win_samples // 2)

    # Resample to match original length using linear interpolation
    entropy_vals = np.array(entropy_vals)
    entropy_times = np.array(entropy_times)

    interp_func = interp1d(entropy_times, entropy_vals, bounds_error=False,
                           fill_value=(entropy_vals[0], entropy_vals[-1]))

    full_times = np.arange(npts)
    entropy_resampled = interp_func(full_times)

    # Create output Trace
    entropy_trace = Trace(data=entropy_resampled.astype(np.float32),
                          header={
                              "starttime": tr.stats.starttime,
                              "sampling_rate": tr.stats.sampling_rate
                          })

    return entropy_trace


def compute_snr(tr, method, sign_win: float = 5, noise_win: float = 30):

    if method == "classic":
        tr.data = classic_sta_lta(tr.data,  sign_win*tr.stats.sampling_rate, noise_win*tr.stats.sampling_rate)
    elif method == "recursive":
        tr.data = recursive_sta_lta(tr.data,  sign_win*tr.stats.sampling_rate, noise_win*tr.stats.sampling_rate)
    elif method == "z_detect":
        tr.data = z_detect(tr.data,  sign_win*tr.stats.sampling_rate)
    return tr

def downsample_trace(trace, factor=10, to_int=False, scale_target=1000):
    """
    Downsample an ObsPy Trace and optionally convert data to scaled integers.

    Parameters
    ----------
    trace : obspy.Trace
        Input trace to downsample.
    factor : int
        Downsampling factor (e.g., 10 takes every 10th sample).
    to_int : bool
        If True, scale data and convert to int32.
    scale_target : float
        Target maximum amplitude after scaling for int conversion.

    Returns
    -------
    obspy.Trace
        Downsampled (and optionally integer-converted) trace.
    """

    # Downsample data
    downsampled_data = trace.data[::factor]
    new_sampling_rate = trace.stats.sampling_rate / factor

    # Scale to integer if requested
    if to_int:
        max_amp = np.max(np.abs(downsampled_data)) or 1.0
        scale = scale_target / max_amp
        downsampled_data = (downsampled_data * scale).astype(np.int32)

    # Create new Trace with updated metadata
    new_trace = trace.copy()
    new_trace.data = downsampled_data
    new_trace.stats.sampling_rate = new_sampling_rate
    new_trace.stats.npts = len(downsampled_data)

    return new_trace


def apply_cross_correlation(stream, reference=0, mode='full', normalize='full', trim=True):

    """
    Cross-correlate all traces in the stream with respect to a reference trace.

    Parameters:
        step_config (dict):
            - "stream": Original Stream
            - "normalize": normalization type (default: 'full')
            - "mode": correlation mode ('full', 'valid', 'same')
            - "reference": index of the reference trace (default: 0)
            - "strict": if True, all traces must align in time (default: True)

    Returns:
        Stream: New stream with correlation functions as Trace objects.
    """


    st = stream.copy()

    if len(st) == 0:
        print("[WARNING] Empty stream for cross-correlation")
        return Stream()

    # Ensure uniform sampling rate and trim to common time window and ensure aligned length
    sr = st[0].stats.sampling_rate
    npts = st[reference].stats.npts

    if trim:

        for tr in st:
            if tr.stats.sampling_rate != sr:
                raise ValueError("Inconsistent sampling rates in stream.")

        common_start = max(tr.stats.starttime for tr in st)
        common_end = min(tr.stats.endtime for tr in st)
        st.trim(starttime=common_start, endtime=common_end, pad=True, fill_value=0)

        for tr in st:
            if tr.stats.npts != npts:
                raise ValueError("Traces do not have same number of samples after trimming.")
    else:
        # In flexible mode, no trimming
        pass  # Traces may have different start times or lengths

    ref_trace = st[reference]
    cc_stream = Stream()

    for i, tr in enumerate(st):
        try:
            cc = correlate_template(tr, ref_trace, mode=mode, normalize=normalize, demean=True, method='auto')
        except Exception as e:
            print(f"[WARNING] Failed to correlate {tr.id} with {ref_trace.id}: {e}")
            continue

        cc_tr = Trace(data=cc)
        cc_tr.stats.sampling_rate = sr
        cc_tr.stats.network = tr.stats.network
        cc_tr.stats.station = ref_trace.stats.station + "_" + tr.stats.station
        cc_tr.stats.channel = tr.stats.channel[0:1] + "X" + tr.stats.channel[-1]
        cc_tr.stats.correlation_with = ref_trace.id
        cc_tr.stats.original_trace = tr.id
        cc_tr.stats.is_autocorrelation = (i == reference)

        if trim:
            num = len(ref_trace.data)
            if (num % 2) == 0:
                c = int(np.ceil(num / 2.) + 1)  # even
            else:

                c = int(np.ceil((num + 1) / 2))  # odd

            cc_tr.stats.mid_point = (cc_tr.stats.starttime + c * cc_tr.stats.sampling_rate).timestamp

        cc_stream.append(cc_tr)

    return cc_stream



def rename_trace(trace: Trace, config: dict) -> Trace:
    """
    Rename components of a trace using a mapping config.

    Parameters:
        trace (obspy.Trace): Input trace.
        config (dict): Renaming config with optional keys: networks, stations, channels, components.

    Returns:
        obspy.Trace: Renamed trace.
    """
    tr = trace.copy()

    # --- Network ---
    network_map = config.get("networks", {})
    new_net = network_map.get(tr.stats.network)
    if new_net:
        tr.stats.network = new_net

    # --- Station ---
    station_map = config.get("stations", {})
    new_sta = station_map.get(tr.stats.station)
    if new_sta:
        tr.stats.station = new_sta

    # --- Full channel ---
    channel_map = config.get("channels", {})
    new_cha = channel_map.get(tr.stats.channel)
    if new_cha:
        tr.stats.channel = new_cha
    else:
        # --- Component-only replacement (last character) ---
        comp_map = config.get("components", {})
        orig_comp = tr.stats.channel[-1]
        new_comp = comp_map.get(orig_comp)
        if new_comp:
            tr.stats.channel = tr.stats.channel[:-1] + new_comp

    return tr



def particle_motion(z, n, e, save_path: str = None):

    z_data = z.data - np.mean(z.data)
    n_data = n.data - np.mean(n.data)
    e_data = e.data - np.mean(e.data)

    st = Stream(traces=[z.copy(), n.copy(), e.copy()])
    azimuth, incidence, rect, plan = flinn(st)

    t = z.stats.starttime.strftime("%Y-%m-%d %H:%M:%S")
    summary = (f"Azimuth:         {azimuth:.2f}°\n" f"Incidence:       {incidence:.2f}°\n"
               f"Rectilinearity:  {rect:.2f}\n" f"Planarity:       {plan:.2f}")
    print(summary)

    # Plot
    txt_path = os.path.join(save_path, "pm.txt")
    base_name = f"{z.id}.D.{z.stats.starttime.year}.{z.stats.starttime.julday}"
    plt_output = os.path.join(save_path, base_name)

    counter = 1
    while os.path.exists(plt_output):
        plt_output = os.path.join(save_path, f"{base_name}_{counter}")
        counter += 1

    plt_output = plt_output+".png"

    if txt_path:

        try:
            with open(txt_path, "a") as f:
                f.write(f"{z.id}, {t}, {azimuth:.2f}°,{incidence:.2f}°,{rect:.2f},{plan:.2f}\n")
            print(f"[INFO] Particle Motion results appended to {txt_path}")
        except Exception as e:
            print(f"[ERROR] Could not write to file: {e}")

        max_val = max(np.max(np.abs(z_data)), np.max(np.abs(n_data)), np.max(np.abs(e_data)))
        lim = 1.05 * max_val

        fig_part, axs = plt.subplots(2, 2, figsize=(8, 6))
        fig_part.suptitle(f"Particle Motion: {z.stats.station}", fontsize=12)

        # --- Z vs N ---
        axs[0, 0].plot(n_data, z_data, linewidth=0.5)
        axs[0, 0].set_xlabel("Radial / North")
        axs[0, 0].set_ylabel("Vertical")
        axs[0, 0].set_xlim(-lim, lim)
        axs[0, 0].set_ylim(-lim, lim)
        axs[0, 0].grid(True, which="both", ls="-", color='grey', alpha=0.4)

        # Incidence angle line
        inc_rad = np.radians(incidence)
        axs[0, 0].plot([0, np.cos(inc_rad) * lim], [0, np.sin(inc_rad) * lim], 'k--', linewidth=0.8)
        axs[0, 0].text(0.05 * lim, 0.9 * lim, f"Inc: {incidence:.1f}°", color='blue', fontsize=9, weight='bold')

        # --- Z vs E ---
        axs[0, 1].plot(e_data, z_data, linewidth=0.5)
        axs[0, 1].set_xlabel("Transversal / East")
        axs[0, 1].set_ylabel("Vertical")
        axs[0, 1].set_xlim(-lim, lim)
        axs[0, 1].set_ylim(-lim, lim)
        axs[0, 1].grid(True, which="both", ls="-", color='grey', alpha=0.4)
        # Add in Z–E panel
        axs[0, 1].plot([0, np.cos(inc_rad) * lim], [0, np.sin(inc_rad) * lim], 'k--', linewidth=0.8)
        axs[0, 1].text(0.05 * lim, 0.9 * lim,
                       f"Inc: {incidence:.1f}°", color='blue', fontsize=9, weight='bold')

        # --- N vs E ---
        axs[1, 0].plot(e_data, n_data, linewidth=0.5)
        axs[1, 0].set_xlabel("Transversal / East")
        axs[1, 0].set_ylabel("Radial / North")
        axs[1, 0].set_xlim(-lim, lim)
        axs[1, 0].set_ylim(-lim, lim)
        axs[1, 0].grid(True, which="both", ls="-", color='grey', alpha=0.4)

        # Add azimuth arrow
        az_rad = np.radians(azimuth)
        arrow_len = lim * 0.8
        axs[1, 0].arrow(0, 0,
                        arrow_len * np.sin(az_rad),
                        arrow_len * np.cos(az_rad),
                        width=0.01 * lim, head_width=0.05 * lim,
                        color='red', edgecolor='black', length_includes_head=True)
        axs[1, 0].text(0.05 * lim, 0.9 * lim, f"Az: {azimuth:.1f}°", color='red', fontsize=9, weight='bold')

        # --- Info box ---
        axs[1, 1].axis("off")
        summary = (f"Azimuth:         {azimuth:.2f}°\n" f"Incidence:       {incidence:.2f}°\n"
                   f"Rectilinearity:  {rect:.2f}\n" f"Planarity:       {plan:.2f}")

        axs[1, 1].text(0.05, 0.6, summary, fontsize=10, va="center", ha="left")
        plt.tight_layout()
        fig_part.savefig(plt_output, dpi=300)
        plt.close(fig_part)


def print_surfquake_trace_headers(
    sources: Union[str, Trace, Sequence[Union[str, Trace]]],
    max_columns: int = 3
) -> None:
    """
    Display one trace per source, side-by-side in columns, for up to `max_columns` at a time.
    Sources can be:
      - file path (str),
      - ObsPy Trace,
      - or a sequence mixing str and Trace.
    Includes SurfQuake geodetic/event headers. Skips event info if missing.
    """

    def format_field(val, width=25):
        return str(val)[:width - 1].ljust(width)

    def format_picks(picks):
        if not picks:
            return "None"
        return ", ".join(
            f"{p.get('phase', '?')}@{UTCDateTime(p.get('time')).strftime('%Y-%m-%d %H:%M:%S')}"
            for p in picks[:2]
        )

    def format_refs(refs):
        if not refs:
            return "None"
        return ", ".join(UTCDateTime(r).strftime("%Y-%m-%d %H:%M:%S") for r in refs[:2])

    def has_arrivals(arrivals):
        return "Yes" if arrivals else "None"

    def fmt2(val):
        try:
            return f"{float(val):.2f}"
        except Exception:
            return "N/A"

    # --- Normalize inputs into a flat list of sources (path strings or (label, Trace) tuples)
    normalized = []

    if isinstance(sources, Trace):
        normalized.append((sources.id or "Trace", sources))
    elif isinstance(sources, str):
        normalized.append(sources)
    else:
        # assume iterable of str/Trace
        for item in sources:
            if isinstance(item, Trace):
                normalized.append((item.id or "Trace", item))
            elif isinstance(item, str):
                normalized.append(item)
            else:
                raise TypeError(f"Unsupported input type: {type(item)} (only str or Trace allowed)")

    total = len(normalized)
    for start in range(0, total, max_columns):
        chunk = normalized[start:start + max_columns]
        traces = []

        for src in chunk:
            try:
                if isinstance(src, tuple):
                    label, tr = src
                    traces.append((label, tr))
                else:
                    st = read(src, headonly=True)
                    tr = st[0]
                    label = os.path.basename(src)
                    traces.append((label, tr))
            except Exception as e:
                print(f"Error reading {src}: {e}")

        if not traces:
            print("No valid traces in this chunk.")
            continue

        all_have_event = all(
            hasattr(tr.stats, "geodetic")
            and "otime" in tr.stats.geodetic
            and "event" in tr.stats.geodetic
            for _, tr in traces
        )

        fields = [
            ("File", lambda f, s: f),
            ("Network", lambda f, s: s.network),
            ("Station", lambda f, s: s.station),
            ("Location", lambda f, s: s.location),
            ("Channel", lambda f, s: s.channel),
            ("Start Time", lambda f, s: s.starttime.isoformat()),
            ("End Time", lambda f, s: s.endtime.isoformat()),
            ("Sampling Rate (Hz)", lambda f, s: s.sampling_rate),
            ("Delta (s)", lambda f, s: round(1 / s.sampling_rate, 6)),
            ("Npts", lambda f, s: s.npts),
            ("Picks", lambda f, s: format_picks(getattr(s, "picks", []))),
            ("References", lambda f, s: format_refs(getattr(s, "references", []))),
        ]

        if all_have_event:
            fields.extend([
                ("Origin Time", lambda f, s: UTCDateTime(s.geodetic["otime"]).strftime("%Y-%m-%d %H:%M:%S")),
                ("Event Lat", lambda f, s: s.geodetic["event"][0]),
                ("Event Long", lambda f, s: s.geodetic["event"][1]),
                ("Distance (km)", lambda f, s: fmt2(s.geodetic["geodetic"][0]) if hasattr(s, "geodetic") else "N/A"),
                ("Az (°)", lambda f, s: fmt2(s.geodetic["geodetic"][1]) if hasattr(s, "geodetic") else "N/A"),
                ("Baz (°)", lambda f, s: fmt2(s.geodetic["geodetic"][2]) if hasattr(s, "geodetic") else "N/A"),
                ("Incidence (°)", lambda f, s: fmt2(s.geodetic["geodetic"][3]) if hasattr(s, "geodetic") else "N/A"),
                ("Arrivals", lambda f, s: has_arrivals(s.geodetic.get("arrivals", [])) if hasattr(s, "geodetic") else "None"),
            ])

        for label, accessor in fields:
            row = [label.ljust(20)]
            for fname, tr in traces:
                try:
                    val = accessor(fname, tr.stats)
                except Exception:
                    val = "ERR"
                row.append(format_field(val))
            print(" | ".join(row))

        if start + max_columns < total:
            input("\nPress Enter to continue...\n")





