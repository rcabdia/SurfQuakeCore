"""
wavelet.py — Wavelet convolution engine for ISP / SurfQuakeCore
===============================================================

Three filter-bank modes are now supported, selected via  use_wavelet=:

  'Complex Morlet'   (default, unchanged)
      Adaptive-width Gaussian envelope × complex sine.
      σ(f) = n_cycles(f) / (2π·f)  — width scales with period.
      n_cycles varies linearly from wmin to wmax across frequencies.
      Gives constant-Q (relative bandwidth) filtering.

  'Gabor'  (alias 'Gaussian bank', 'AFTAN')
      Fixed-width Gaussian envelope × complex sine.
      σ(f) = alpha(f) / (2π·f·sqrt(2))  where alpha = ffact·20·√(Δ/1000)
      OR sigma_seconds is set directly via sigma_sec=<value>.
      Gives constant absolute bandwidth filtering — mathematically
      identical to AFTAN's Gaussian narrow-band filter bank.
      The Gabor wavelet is a special case of the Morlet wavelet where
      the envelope width does NOT scale with frequency.

  'Paul'  /  'Mexican Hat'  (unchanged from original)

Frequency axis parameterisation — two equivalent modes:

  Mode A — fixed number of atoms  (original, default)
      nf=100  →  100 log-spaced frequencies between fmin and fmax.

  Mode B — octave / voice  (new, MATLAB-style)
      voices_per_octave=16  →  nf computed automatically as
      nf = round( log2(fmax/fmin) * voices_per_octave )
      Each octave (factor-of-2 frequency range) contains the same
      number of analysis frequencies, giving uniform log-frequency
      resolution independent of the fmin/fmax ratio.
      Pass voices_per_octave=N and omit nf (or set nf=None).

Mathematical connection between modes
--------------------------------------
Complex Morlet:
    s(f)   = n_cycles(f) / (2π·f)
    ψ(t,f) = (π·s²)^(-¼) · exp(i·2π·f·t) · exp(-t²/2s²)

Gabor (fixed σ = sigma_sec):
    ψ(t,f) = (π·sigma_sec²)^(-¼) · exp(i·2π·f·t) · exp(-t²/2·sigma_sec²)
    → same formula, but s is constant across all f instead of scaling with 1/f.

AFTAN Gaussian filter (for reference):
    H(ω; ω₀) = exp[-α²(ω/ω₀ - 1)²]   with α = ffact·20·√(Δ/1000)
    The Gabor mode with sigma_sec = alpha / (2π·f₀·√2) is equivalent
    to AFTAN's filter evaluated at centre frequency f₀.

Usage example
-------------
    from wavelet import ConvolveWaveletScipy

    # --- Complex Morlet (unchanged behaviour) ---
    cw = ConvolveWaveletScipy(trace, wmin=6, wmax=12, fmin=0.02, fmax=0.2, nf=80)
    cw.setup_wavelet()
    sc = cw.scalogram_in_dbs()

    # --- Gabor / Gaussian filter bank ---
    # sigma_sec controls the absolute width of the Gaussian window [s]
    # The AFTAN helper aftan_sigma() computes it from distance and ffact.
    cw = ConvolveWaveletScipy(trace, use_wavelet='Gabor',
                              sigma_sec=30.0,          # fixed Gaussian width [s]
                              fmin=0.02, fmax=0.2, nf=80)
    cw.setup_wavelet()

    # --- Octave / voice parameterisation ---
    cw = ConvolveWaveletScipy(trace, wmin=6, wmax=12,
                              fmin=0.02, fmax=0.2,
                              voices_per_octave=16)    # nf computed automatically
    cw.setup_wavelet()
    print(cw.nf)   # → round(log2(0.2/0.02) * 16) = round(3.32 * 16) = 53

    # --- Gabor + octave parameterisation (AFTAN-equivalent) ---
    from wavelet import aftan_sigma
    sigma = aftan_sigma(dist_km=300, ffact=1.0, f0=0.05)   # compute σ for f=0.05 Hz
    cw = ConvolveWaveletScipy(trace, use_wavelet='Gabor',
                              sigma_sec=sigma,
                              fmin=0.02, fmax=0.2,
                              voices_per_octave=16)
    cw.setup_wavelet()
"""

import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
from typing import Optional

import numpy as np
import scipy
import scipy.signal
from deprecated import deprecated
from obspy import read, Stream, Trace, UTCDateTime
from obspy.signal.filter import lowpass
from scipy.signal import argrelextrema

from Exceptions.exceptions import InvalidFile
from surfquakecore.Structures.structures import TracerStats
from surfquakecore.utils.obspy_utils import ObspyUtil, MseedUtil


# ---------------------------------------------------------------------------
# AFTAN sigma helper  (kept outside the class as requested)
# ---------------------------------------------------------------------------

def aftan_sigma(dist_km: float,
                ffact: float = 1.0,
                f0: Optional[float] = None) -> float:
    """
    Compute the Gaussian window half-width sigma [s] that makes the Gabor
    wavelet at frequency f0 equivalent to AFTAN's adaptive Gaussian filter.

    AFTAN filter in the frequency domain:
        H(ω; ω₀) = exp[ -α² (ω/ω₀ - 1)² ]
        α = ffact · 20 · √(dist_km / 1000)

    This corresponds to a Gaussian envelope in the time domain with:
        sigma_time = α / (2π · f₀ · √2)   [seconds]

    When f0 is None, the geometric mean of fmin and fmax is commonly used
    as a representative frequency, but since AFTAN actually uses a different
    alpha per frequency, calling aftan_sigma() once per frequency in the
    bank gives the most faithful reproduction.

    Parameters
    ----------
    dist_km : inter-station distance [km]
    ffact   : AFTAN filter width factor (default 1.0)
    f0      : centre frequency [Hz].  If None, returns alpha only.

    Returns
    -------
    sigma_sec : float  [s]  if f0 is given,
    alpha     : float  [-]  if f0 is None
    """
    alpha = ffact * 20.0 * np.sqrt(dist_km / 1000.0)
    if f0 is None:
        return alpha
    sigma_sec = alpha / (2.0 * np.pi * f0 * np.sqrt(2.0))
    return sigma_sec


def voices_to_nf(fmin: float, fmax: float, voices_per_octave: int) -> int:
    """
    Convert a voices-per-octave specification to a number of frequency atoms.

    nf = round( log2(fmax/fmin) * voices_per_octave )

    Parameters
    ----------
    fmin, fmax          : frequency limits [Hz]
    voices_per_octave   : frequency resolution within each octave (typical: 8–32)

    Returns
    -------
    nf : int  — total number of log-spaced frequencies
    """
    if fmax <= fmin or fmin <= 0:
        raise ValueError(f"Need 0 < fmin < fmax, got fmin={fmin} fmax={fmax}")
    n_octaves = np.log2(fmax / fmin)
    return max(2, int(round(n_octaves * voices_per_octave)))


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ConvolveWaveletBase:
    """
    Base class for wavelet convolution.

    New kwargs (in addition to the original ones):
    -----------------------------------------------
    use_wavelet : str
        'Complex Morlet'  — adaptive width (default, unchanged)
        'Gabor'           — fixed-width Gaussian bank (≡ AFTAN filter)
        'Paul'            — Paul wavelet
        'Mexican Hat'     — Mexican Hat wavelet

    sigma_sec : float or None
        Fixed Gaussian window half-width [s] for Gabor mode.
        Ignored for Complex Morlet.
        If None in Gabor mode, defaults to wmin/(2π·fmin) (same as Morlet
        at the lowest frequency — a reasonable starting point).

    aftan_dist_km : float or None
        If given together with use_wavelet='Gabor', sigma_sec is computed
        automatically per-frequency using aftan_sigma(dist_km, ffact, f).
        This gives the closest possible reproduction of AFTAN's filter bank.

    aftan_ffact : float
        AFTAN filter width factor. Default 1.0. Only used when
        aftan_dist_km is set.

    voices_per_octave : int or None
        If given, overrides nf. The number of atoms is computed as
        nf = round( log2(fmax/fmin) * voices_per_octave ).
        Typical values: 8 (coarse), 16 (standard), 32 (fine).
        Pass voices_per_octave=N and leave nf at its default (it is ignored).
    """

    def __init__(self, data, **kwargs):
        if isinstance(data, Trace):
            self.trace: Trace = data
            self.stats = self.trace.stats.copy()
        elif isinstance(data, Stream):
            self.st: Stream = data
            self._decimate_stream = kwargs.get("decimate_stream", False)
        else:
            if not MseedUtil.is_valid_mseed(data):
                raise InvalidFile("The file: {} is not a valid mseed.".format(data))
            self.trace: Trace = read(data)[0]
            self.stats = ObspyUtil.get_stats(data)

        self._wmin = float(kwargs.get("wmin", 6.))
        self._wmax = float(kwargs.get("wmax", 6.))
        self._tt   = float(kwargs.get("tt", 2.))
        self._fmin = float(kwargs.get("fmin", 2.))
        self._fmax = float(kwargs.get("fmax", 12.))
        self._use_wavelet = kwargs.get("use_wavelet", "Complex Morlet")
        self._m    = int(kwargs.get("m", 30))
        self._use_rfft  = kwargs.get("use_rfft", False)
        self._decimate  = kwargs.get("decimate", False)

        # ---- Gabor / AFTAN parameters ----
        self._sigma_sec     = kwargs.get("sigma_sec", None)
        self._aftan_dist_km = kwargs.get("aftan_dist_km", None)
        self._aftan_ffact   = float(kwargs.get("aftan_ffact", 1.0))

        # ---- frequency resolution: voices_per_octave overrides nf ----
        voices = kwargs.get("voices_per_octave", None)
        if voices is not None:
            self._voices_per_octave = int(voices)
            self._nf = voices_to_nf(self._fmin, self._fmax, self._voices_per_octave)
        else:
            self._voices_per_octave = None
            self._nf = int(kwargs.get("nf", 20))

        self._data        = None
        self._npts        = 0
        self._tf          = None
        self._start_time  = self.trace.stats.starttime
        self._end_time    = self.trace.stats.endtime
        self._sample_rate = self.trace.stats.sampling_rate

        self._frex     = None
        self._n_cycles = None
        self._sigma_arr = None   # per-frequency sigma array (Gabor mode)
        self._wtime    = None
        self._half_wave = None

    def __repr__(self):
        mode = self._use_wavelet
        if self._voices_per_octave:
            freq_spec = f"voices_per_octave={self._voices_per_octave} (nf={self._nf})"
        else:
            freq_spec = f"nf={self._nf}"
        return (f"ConvolveWavelet(mode={mode}, wmin={self._wmin}, wmax={self._wmax}, "
                f"fmin={self._fmin}, fmax={self._fmax}, {freq_spec})")

    def __eq__(self, other):
        return (self.trace == other.trace
                and self._wmin == other._wmin and self._wmax == other._wmax
                and self._tt == other._tt and self._fmin == other._fmin
                and self._fmax == other._fmax and self._nf == other._nf
                and self._use_rfft == other._use_rfft
                and self._start_time == other._start_time
                and self._end_time == other._end_time
                and self._decimate == other._decimate
                and self._use_wavelet == other._use_wavelet)

    def _validate_data(self):
        if self._data is None:
            raise AttributeError("Data not found. Run setup_wavelet().")

    def _validate_kwargs(self):
        if self._wmax < self._wmin:
            raise AttributeError(
                f"wmin can't be bigger than wmax. wmin={self._wmin}, wmax={self._wmax}")
        if self._fmax < self._fmin:
            raise AttributeError(
                f"fmin can't be bigger than fmax. fmin={self._fmin}, fmax={self._fmax}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def npts(self) -> int:
        return self._npts

    @property
    def nf(self) -> int:
        """Total number of frequency atoms (read-only)."""
        return self._nf

    @property
    def frequencies(self) -> np.ndarray:
        """Log-spaced frequency axis [Hz], shape (nf,)."""
        return self._frex

    @property
    def periods(self) -> np.ndarray:
        """Period axis [s] = 1/frequencies, shape (nf,)."""
        return 1.0 / self._frex

    @property
    def n_octaves(self) -> float:
        """Total number of octaves spanned by [fmin, fmax]."""
        return float(np.log2(self._fmax / self._fmin))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup_wavelet(self, start_time: UTCDateTime = None,
                      end_time: UTCDateTime = None, **kwargs):
        """
        Load data and (re)compute the bank of atoms.

        Parameters
        ----------
        start_time, end_time : UTCDateTime, optional
        kwargs               : any __init__ keyword can be overridden here.
                               E.g. setup_wavelet(voices_per_octave=32)
        """
        self._start_time = start_time if start_time else self.stats.starttime
        self._end_time   = end_time   if end_time   else self.stats.endtime
        self.__setup_wavelet(start_time, end_time, **kwargs)

    def setup_atoms(self, **kwargs):
        """
        Recompute the bank of atoms with new parameters (data unchanged).

        Any __init__ keyword can be overridden:
            nf, wmin, wmax, fmin, fmax, voices_per_octave,
            sigma_sec, aftan_dist_km, aftan_ffact, use_wavelet
        """
        self._wmin = float(kwargs.get("wmin", self._wmin))
        self._wmax = float(kwargs.get("wmax", self._wmax))
        self._tt   = float(kwargs.get("tt",   self._tt))
        self._fmin = float(kwargs.get("fmin", self._fmin))
        self._fmax = float(kwargs.get("fmax", self._fmax))
        self._use_rfft    = kwargs.get("use_rfft",    self._use_rfft)
        self._use_wavelet = kwargs.get("use_wavelet", self._use_wavelet)
        self._sigma_sec     = kwargs.get("sigma_sec",     self._sigma_sec)
        self._aftan_dist_km = kwargs.get("aftan_dist_km", self._aftan_dist_km)
        self._aftan_ffact   = float(kwargs.get("aftan_ffact", self._aftan_ffact))

        # voices_per_octave overrides nf
        voices = kwargs.get("voices_per_octave", None)
        if voices is not None:
            self._voices_per_octave = int(voices)
            self._nf = voices_to_nf(self._fmin, self._fmax, self._voices_per_octave)
        else:
            self._nf = int(kwargs.get("nf", self._nf))

        self._validate_kwargs()
        self._tf = None
        self._setup_atoms()

    def _setup_atoms(self):
        """Build the frequency axis, cycle array, and kernel time vector."""
        self._validate_data()

        # ---- frequency axis ----
        self._frex = np.logspace(
            np.log10(self._fmin), np.log10(self._fmax), self._nf, base=10)

        # ---- cycle / sigma arrays ----
        if self._is_gabor_mode():
            self._setup_gabor_sigmas()
        else:
            # Complex Morlet: n_cycles varies linearly wmin → wmax
            self._n_cycles = np.linspace(self._wmin, self._wmax, self._nf)
            self._sigma_arr = None

        # ---- kernel time vector ----
        dt = 1.0 / self._sample_rate
        self._wtime    = np.arange(-self._tt, self._tt + dt, dt)
        self._half_wave = (len(self._wtime) - 1) / 2.0

    def _is_gabor_mode(self) -> bool:
        """Return True if the Gabor / Gaussian-bank mode is active."""
        return self._use_wavelet.lower() in (
            'gabor', 'gaussian bank', 'gaussian', 'aftan', 'gauss')

    def _setup_gabor_sigmas(self):
        """
        Build the per-frequency sigma array for Gabor mode.

        Three sub-cases in order of priority:
        1. aftan_dist_km is set → compute sigma per frequency using aftan_sigma()
           This gives the exact AFTAN filter bank equivalent.
        2. sigma_sec is set → use the same fixed sigma for all frequencies
           (strict Gabor filter bank with constant time resolution).
        3. Neither is set → default: sigma = wmin / (2π·fmin)
           (same as Morlet at the lowest frequency — safe fallback).
        """
        if self._aftan_dist_km is not None:
            # per-frequency sigma — closest to AFTAN
            self._sigma_arr = np.array([
                aftan_sigma(self._aftan_dist_km, self._aftan_ffact, f)
                for f in self._frex
            ])
            mode_str = f"AFTAN-equivalent  dist={self._aftan_dist_km:.1f} km  " \
                       f"ffact={self._aftan_ffact}"
        elif self._sigma_sec is not None:
            # fixed sigma for all frequencies
            self._sigma_arr = np.full(self._nf, float(self._sigma_sec))
            mode_str = f"fixed sigma={self._sigma_sec:.3f} s"
        else:
            # fallback: use wmin at fmin as the sigma
            sigma_default = self._wmin / (2.0 * np.pi * self._fmin)
            self._sigma_arr = np.full(self._nf, sigma_default)
            mode_str = f"default sigma={sigma_default:.3f} s (wmin/2pi/fmin)"

        # n_cycles is not used in Gabor mode but set for __repr__ consistency
        self._n_cycles = self._sigma_arr * (2.0 * np.pi * self._frex)

        print(f"[Gabor bank]  nf={self._nf}  {mode_str}  "
              f"σ_min={self._sigma_arr.min():.3f} s  "
              f"σ_max={self._sigma_arr.max():.3f} s")

    # ------------------------------------------------------------------
    # Wavelet atom builder
    # ------------------------------------------------------------------

    def filter_win(self, freq: float, index: int) -> np.ndarray:
        """
        Build one wavelet atom at centre frequency `freq`.

        Parameters
        ----------
        freq  : centre frequency [Hz]
        index : index into self._frex (used to look up sigma / n_cycles)

        Returns
        -------
        cmw : complex128 array, length = len(self._wtime)
        """
        if self._use_wavelet == "Paul":
            num = (2 ** self._m * 1j * np.math.factorial(self._m))
            den = np.sqrt(np.pi * np.math.factorial(2 * self._m))
            s   = (2 * self._m + 1) / (freq * np.pi * 4)
            mu  = self._wtime / s
            p   = 1 / (1 - 1j * mu) ** (self._m + 1)
            cmw = (num * p) / den

        elif self._use_wavelet == "Mexican Hat":
            m     = 2
            sigma = (2 * m + 1) / (4 * np.pi * freq)
            k1    = 2 / ((np.pi ** 0.25) * np.sqrt(2 * sigma))
            k2    = (self._wtime ** 2) / (sigma ** 2) - 1
            k3    = np.exp(-self._wtime ** 2 / (2 * sigma ** 2))
            cmw   = -1 * k1 * k2 * k3

        elif self._is_gabor_mode():
            # ---- Gabor / Gaussian filter bank ----
            # Fixed σ (does not scale with frequency):
            #   ψ(t) = (π·σ²)^(-¼) · exp(i·2π·f·t) · exp(-t²/2σ²)
            s             = float(self._sigma_arr[index])
            normalization = 1.0 / (np.pi * s ** 2) ** 0.25
            cmw = (np.exp(1j * 2.0 * np.pi * freq * self._wtime)
                   * np.exp(-self._wtime ** 2 / (2.0 * s ** 2)))
            cmw = normalization * cmw.conjugate()

        else:
            # ---- Complex Morlet (default, unchanged) ----
            # Adaptive σ = n_cycles / (2π·f):
            s             = self._n_cycles[index] / (2.0 * np.pi * freq)
            normalization = 1.0 / (np.pi * s ** 2) ** 0.25
            cmw = (np.exp(1j * 2.0 * np.pi * freq * self._wtime)
                   * np.exp(-self._wtime ** 2 / (2.0 * s ** 2)))
            cmw = normalization * cmw.conjugate()

        return cmw

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def __get_data_in_time(self, start_time, end_time):
        tr = self.trace.copy()
        tr.trim(starttime=start_time, endtime=end_time)
        if self._decimate:
            tr = self.decimate_data(tr)
        tr.detrend(type='demean')
        tr.taper(max_percentage=0.05)
        self._npts        = tr.stats.npts
        self._sample_rate = tr.stats.sampling_rate
        return tr.data

    def __get_data_in_time_stream(self, start_time, end_time):
        self.st.trim(starttime=start_time, endtime=end_time)
        self.st.detrend(type='demean')
        self.st.taper(max_percentage=0.025)
        self._sample_rate = self.st[0].stats.sampling_rate
        return self.st

    def __get_resample_factor(self):
        return int(0.4 * self._sample_rate / self._fmax)

    def decimate_data_stream(self, st: Stream, fs_new=10):
        st.resample(fs_new)
        return st

    def decimate_data(self, tr: Trace):
        rf = self.__get_resample_factor()
        if rf > 1:
            data      = scipy.signal.decimate(tr.data, rf, ftype='fir', zero_phase=True)
            new_stats = tr.stats.copy()
            new_stats["npts"]          = len(data)
            new_stats["sampling_rate"] /= rf
            new_stats["delta"]          = 1.0 / new_stats["sampling_rate"]
            return Trace(data, new_stats)
        return tr

    def get_nproc(self):
        total_cpu = multiprocessing.cpu_count()
        nproc = total_cpu - 2 if total_cpu > 3 else total_cpu - 1
        nproc = min(nproc, self._nf)
        nproc = max(nproc, 1)
        return nproc

    @staticmethod
    def __tapper(data, max_percentage=0.05):
        tr = Trace(data)
        tr.taper(max_percentage=max_percentage, type='blackman')
        return tr.data

    def __setup_wavelet(self, start_time, end_time, **kwargs):
        setup_wavelets_stream = kwargs.get("setup_wavelets_stream", False)
        if setup_wavelets_stream:
            self._data = self.__get_data_in_time_stream(start_time, end_time)
        else:
            self._data = self.__get_data_in_time(start_time, end_time)
        self.setup_atoms(**kwargs)

    def _convolve_atoms(self, parallel: bool):
        # implemented in child classes
        pass

    # ------------------------------------------------------------------
    # Output methods  (unchanged from original)
    # ------------------------------------------------------------------

    def scalogram_in_dbs(self):
        if self._tf is None:
            self.compute_tf()
        sc = np.abs(self._tf) ** 2
        return 10.0 * np.log10(sc / np.max(sc))

    def scalogram(self):
        if self._tf is None:
            self.compute_tf()
        return self._tf

    def phase(self):
        if self._tf is None:
            self.compute_tf()
        phase  = np.unwrap(np.angle(self._tf), axis=0)
        a, b   = np.shape(self._tf)
        freq   = np.fft.fftfreq(b, d=1.0 / self._sample_rate)
        freq   = np.tile(freq, (a, 1))
        deriv  = np.fft.ifft(2j * np.pi * freq * np.fft.fft(self._tf, axis=1), axis=1)
        inst_freq     = deriv / (2j * np.pi * self._tf)
        inst_freq     = np.abs(inst_freq)
        ins_freq_hz   = (inst_freq * self._sample_rate) / (2.0 * np.pi)
        return phase, inst_freq, ins_freq_hz

    def get_data_window(self):
        start = int(self._half_wave + 1)
        end   = self._npts + int(self._half_wave + 1)
        return start, end

    def compute_tf(self, parallel=True):
        pass

    def cf(self, tapper=True, parallel=True):
        if self._tf is None:
            self.compute_tf(parallel=parallel)
        cf = np.mean(np.diff(np.log10(np.abs(self._tf) ** 2)), axis=0, dtype=np.float32)
        if tapper:
            cf = self.__tapper(cf)
        return cf

    def cf_lowpass(self, tapper=True, parallel=True, freq=0.15):
        cf = lowpass(self.cf(tapper, parallel=parallel),
                     freq, df=self._sample_rate, corners=3, zerophase=True)
        return cf

    def charachteristic_function_kurt_stream(self, window_size_seconds=5,
                                              parallel=False, stream=True):
        kurt_st = []
        if self._tf is None:
            self.st.normalize()
            self.compute_tf(parallel=parallel, stream=stream)

        for item in self._tf:
            stats = item[1]
            print("Processing Kurtosis", stats.station, stats.channel)
            pow_scalogram = np.abs(item[0]) ** 2
            kurtosis_values, time_vector = self.conventional_kurtosis(
                pow_scalogram, window_size_seconds=window_size_seconds,
                sampling_rate=self._sample_rate)
            time_vector_resample = np.linspace(
                time_vector[0], time_vector[-1],
                int(time_vector[-1] * self._sample_rate))
            kurtosis_values_resample = np.interp(
                time_vector_resample, time_vector, kurtosis_values)
            tr_kurt = Trace(data=kurtosis_values_resample, header=stats)
            kurt_st.append(tr_kurt)

        kurt_st = Stream(kurt_st)
        for op in ('simple', 'constant', 'linear'):
            kurt_st.detrend(type=op)
        kurt_st.taper(max_percentage=0.05, type='blackman')
        kurt_st.filter(type='lowpass', freq=0.15, zerophase=True, corners=4)
        for op in ('simple', 'constant', 'linear'):
            kurt_st.detrend(type=op)
        kurt_st.taper(max_percentage=0.05, type='blackman')
        return kurt_st

    def charachteristic_function_kurt(self, window_size_seconds=5, parallel=True):
        if self._tf is None:
            self._data = self._data / np.max(self._data)
            self.compute_tf(parallel=parallel)

        pow_scalogram = np.abs(self._tf) ** 2
        kurtosis_values, time_vector = self.conventional_kurtosis(
            pow_scalogram, window_size_seconds=window_size_seconds,
            sampling_rate=self._sample_rate)
        time_vector_resample = np.linspace(
            time_vector[0], time_vector[-1],
            int(time_vector[-1] * self._sample_rate))
        kurtosis_values_resample = np.interp(
            time_vector_resample, time_vector, kurtosis_values)

        tr_kurt = Trace(data=kurtosis_values_resample)
        tr_kurt.stats.station      = self.trace.stats.station
        tr_kurt.stats.network      = self.trace.stats.network
        tr_kurt.stats.channel      = self.trace.stats.channel
        tr_kurt.stats.starttime    = (self.trace.stats.starttime
                                      + time_vector_resample[0])
        tr_kurt.stats.sampling_rate = self.trace.stats.sampling_rate
        tr_kurt.detrend(type='simple')
        tr_kurt.detrend(type='constant')
        tr_kurt.detrend(type='linear')
        tr_kurt.taper(max_percentage=0.05, type='blackman')
        return tr_kurt

    def conventional_kurtosis(self, data, window_size_seconds, sampling_rate):
        n                    = data.shape[1]
        window_size_samples  = int(window_size_seconds * sampling_rate)
        slide                = int(sampling_rate / 2)
        kurtosis_values      = self._conventional_kurtosis_helper(
            data, window_size_samples, slide, n)
        time_vector = np.linspace(
            0, int((n - window_size_samples) / sampling_rate),
            len(kurtosis_values)) + int(window_size_seconds)
        time_vector     = time_vector[:-1]
        kurtosis_values = np.abs(np.diff(kurtosis_values))
        return kurtosis_values, time_vector

    def _conventional_kurtosis_helper(self, data, window_size_samples, slide, n):
        kurtosis_values = []
        for i in range(0, n - window_size_samples + 1, slide):
            window_data  = data[:, i:i + window_size_samples]
            mean         = np.mean(window_data)
            variance     = np.mean((window_data - mean) ** 2)
            fourth_moment = np.mean((window_data - mean) ** 4)
            if variance > 1e-10:
                kurtosis = (fourth_moment / (variance ** 2)) - 3
                if not np.isfinite(kurtosis):
                    kurtosis = 0.0
                elif abs(kurtosis) > 1e6:
                    kurtosis = np.sign(kurtosis) * 1e6
            else:
                kurtosis = 1e-2
            kurtosis_values.append(kurtosis)
        return np.array(kurtosis_values)

    def get_time_delay(self):
        return 0.5 * self._wmin / (2.0 * np.pi * self._fmin)

    def detect_max_pick_in_time(self, data: np.ndarray):
        filtered_data = np.abs(np.where(
            np.abs(data) >= self.get_detection_limit(data), data, 0.0))
        if filtered_data.sum() != 0.0:
            max_index = np.argmax(filtered_data)
            time_s    = max_index / self._sample_rate
            return self._start_time + time_s + self.get_time_delay()
        return None

    def detect_picks_in_time(self, data: np.ndarray, sigmas=5.0):
        max_indexes = self.detect_picks(data, sigmas=sigmas)
        delay       = self.get_time_delay()
        times_s     = max_indexes / self._sample_rate
        return [self._start_time + t + delay for t in times_s]

    def detect_picks(self, data, sigmas=5.0):
        limit         = self.get_detection_limit(data, sigmas=sigmas)
        filtered_data = np.where(data >= limit, data, 0.0)
        ind           = scipy.signal.argrelextrema(filtered_data, np.greater)
        return ind[0]

    @staticmethod
    def get_detection_limit(data: np.ndarray, sigmas=5.0):
        return sigmas * np.sqrt(np.var(data))


# ---------------------------------------------------------------------------
# FFT-based child class  (unchanged except __repr__ delegation)
# ---------------------------------------------------------------------------

@deprecated(reason="You should use ConvolveWaveletScipy")
class ConvolveWavelet(ConvolveWaveletBase):
    """
    Original FFT-based convolution.  Deprecated — use ConvolveWaveletScipy.
    Inherits all new Gabor / octave features from ConvolveWaveletBase.
    """

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.__conv   = None
        self.__n_conv = 0

    def _setup_atoms(self):
        super()._setup_atoms()
        self._convolve_atoms(parallel=False)

    def _convolve_atoms(self, parallel=False):
        n_kern        = len(self._wtime)
        self.__n_conv = 2 ** int(np.ceil(np.log2(self._npts + n_kern)))
        array_size    = (self.__n_conv // 2 + 1 if self._use_rfft
                         else self.__n_conv)
        self.__conv   = np.zeros((self._nf, int(array_size)), dtype=np.complex64)

        if self._use_rfft:
            data_fft = np.fft.rfft(self._data, n=self.__n_conv)
        else:
            data_fft = np.fft.fft(self._data, n=self.__n_conv)

        for ii, fi in enumerate(self._frex):
            cmw     = self.filter_win(fi, ii)
            cmw_fft = (np.fft.rfft(cmw, self.__n_conv) if self._use_rfft
                       else np.fft.fft(cmw, self.__n_conv))
            self.__conv[ii, :] = np.multiply(cmw_fft, data_fft,
                                              dtype=np.complex64)

    def __compute_cwt(self, data):
        start = int(self._half_wave + 1)
        end   = self._npts + int(self._half_wave + 1)
        cwt   = (np.fft.irfft(data)[start:end] if self._use_rfft
                 else np.fft.ifft(data, n=self.__n_conv)[start:end])
        return cwt - np.mean(cwt, dtype=np.float32)

    def __cwt_ba(self, parallel=False):
        n_proc = self.get_nproc()
        if parallel and n_proc > 1:
            with ThreadPool(n_proc) as pool:
                tf = np.array(pool.map(self.__compute_cwt, self.__conv),
                              copy=False, dtype=np.float32)
        else:
            tf = np.array([self.__compute_cwt(row) for row in self.__conv],
                          copy=False, dtype=np.float32)
        self.__conv = None
        return tf

    def compute_tf(self, parallel=True):
        self._validate_data()
        self._tf = self.__cwt_ba(parallel=parallel)


# ---------------------------------------------------------------------------
# Scipy oaconvolve child class  (primary, recommended)
# ---------------------------------------------------------------------------

class ConvolveWaveletScipy(ConvolveWaveletBase):
    """
    Wavelet convolution via scipy.signal.oaconvolve (overlap-add FFT).

    Recommended over ConvolveWavelet for all new code.

    Supports all three filter-bank modes:
      'Complex Morlet'  — adaptive width (constant Q)
      'Gabor'           — fixed width (constant bandwidth, ≡ AFTAN filter)
      'Paul' / 'Mexican Hat'

    And both frequency parameterisations:
      nf=N               — fixed number of atoms
      voices_per_octave=V — MATLAB-style octave/voice specification

    Example — AFTAN-equivalent Gabor bank
    --------------------------------------
    from wavelet import ConvolveWaveletScipy, aftan_sigma
    cw = ConvolveWaveletScipy(
        trace,
        use_wavelet    = 'Gabor',
        aftan_dist_km  = 300.0,   # compute sigma per frequency automatically
        aftan_ffact    = 1.0,
        fmin           = 0.02,
        fmax           = 0.2,
        voices_per_octave = 16,
    )
    cw.setup_wavelet()
    sc = cw.scalogram_in_dbs()
    """

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

    def __convolve(self, freq_index: tuple):
        freq, index = freq_index
        cmw = self.filter_win(freq, index)
        return scipy.signal.oaconvolve(self._data, cmw, mode='same')

    def _convolve_atoms(self, parallel: bool):
        d_type = np.float32 if self._use_rfft else np.complex64
        freq_index_pairs = [(fi, i) for i, fi in enumerate(self._frex)]
        n_proc = self.get_nproc()
        if parallel and n_proc > 1:
            with ThreadPool(n_proc) as pool:
                tf = np.array(pool.map(self.__convolve, freq_index_pairs),
                              copy=False, dtype=d_type)
        else:
            tf = np.array([self.__convolve(p) for p in freq_index_pairs],
                          copy=False, dtype=d_type)
        return tf

    def _convolve_atoms_stream(self):
        d_type = np.float32 if self._use_rfft else np.complex64
        tfs = []
        for trace_index in self.st:
            tf = []
            for i, fi in enumerate(self._frex):
                cmw = self.filter_win(fi, i)
                tf.append(scipy.signal.oaconvolve(trace_index, cmw, mode='same'))
            tfs.append([np.array(tf, copy=False, dtype=d_type),
                        trace_index.stats])
        return tfs

    def compute_tf(self, parallel=True, stream=False):
        """
        Compute the time-frequency matrix via overlap-add convolution.

        Parameters
        ----------
        parallel : bool  — use multiprocessing (default True)
        stream   : bool  — process an ObsPy Stream (default False)

        Returns
        -------
        Sets self._tf  : complex array, shape (nf, npts)
        """
        self._validate_data()
        if stream:
            self._tf = self._convolve_atoms_stream()
        else:
            self._tf = self._convolve_atoms(parallel)


# ===========================================================================
# AFTANWavelet — faithful reimplementation of the original FORTRAN AFTAN
# ===========================================================================

class AFTANWavelet:
    """
    Faithful Python reimplementation of the original FORTRAN AFTAN filter bank
    (Barmine 2006, aftanpg.f / aftanipg.f).

    This class preserves ALL of the geophysical tricks of the original code:

    Trick 1 — Ratio-based Gaussian filter (frequency domain)
    ---------------------------------------------------------
    The filter is NOT a standard Gabor/Morlet atom.  It is applied directly
    in the frequency domain as:

        H(ω; ω₀) = exp[ -α² · (ω/ω₀ - 1)² ]

    where  α = ffact · 20 · √(dist_km / 1000)

    This is a Gaussian in the relative-frequency variable (ω/ω₀ - 1),
    not in the absolute-frequency variable (ω - ω₀).  The consequence is
    that the filter bandwidth scales proportionally with centre frequency
    (constant-Q behaviour) and the filter is symmetric in log-frequency
    space — appropriate for surface-wave dispersion which is smooth in
    log-period space.

    Trick 2 — One-sided spectrum → exact analytic signal
    ----------------------------------------------------
    After applying H(ω; ω₀), the negative-frequency half of the spectrum
    is explicitly set to zero (and DC / Nyquist halved).  IFFT of the
    one-sided spectrum gives the analytic signal z(t) = x(t) + iH{x}(t)
    exactly, without any approximation from finite-length convolution.
    This is the Hilbert transform trick used throughout the FORTRAN code.

    Trick 3 — Sub-sample parabolic peak interpolation (fmax subroutine)
    --------------------------------------------------------------------
    The group-arrival time t* is not read at the nearest sample.  A
    parabola is fitted through the three samples around the amplitude
    maximum:
        t_offset = 0.5 · (A[j-1] - A[j+1]) / (A[j-1] - 2A[j] + A[j+1])
    The group velocity is then  U(ω₀) = Δ / (t* + t0).
    This gives sub-sample accuracy with no additional computational cost.

    Trick 4 — Observed period from instantaneous phase derivative
    -------------------------------------------------------------
    The observed (instantaneous) period at the ridge point is NOT 1/f₀
    (the filter centre).  It is estimated from the phase derivative:
        T_obs = 2π · dt / Δφ
    where Δφ = φ[j+1] - φ[j-1] (unwrapped phase difference over 2 samples).
    This distinguishes the actual period of the dominant oscillation from
    the nominal filter centre, which can differ in dispersive signals.

    Trick 5 — π/4 geometric spreading correction
    ---------------------------------------------
    Surface waves in a homogeneous medium accumulate a π/4 phase advance
    relative to body waves due to geometric spreading.  The AFTAN phase
    velocity formula corrects for this:
        Φ_total = ω·Δ/c = φ_measured + ω·t_group + piover4·π/4 + k·2π
    where piover4 = -1 for EGFs (cross-correlations), +1 for seismograms.

    Trick 6 — Adaptive α tied to inter-station distance
    ----------------------------------------------------
    The filter width α = ffact·20·√(Δ/1000) is not arbitrary.  It was
    empirically calibrated so that at typical ambient-noise distances
    (100–1000 km) the filter is narrow enough to resolve the dispersion
    curve but wide enough to contain sufficient energy.  The √(Δ/1000)
    factor reflects the fact that longer paths produce better-separated
    arrivals and can therefore tolerate narrower filters.

    Parameters
    ----------
    data     : ObsPy Trace, or 1-D numpy array (if array, also pass fs and t0)
    dist_km  : inter-station distance [km]
    t0       : begin time of the trace / EGF branch [s]
               For SAC files: tr.stats.sac.b
               For EGFs prepared by prepare_branch(): 0.0
    dt       : sampling interval [s]  (ignored if data is a Trace)
    fmin     : minimum frequency [Hz]  (= 1/tmax)
    fmax     : maximum frequency [Hz]  (= 1/tmin)
    nf       : number of log-spaced frequencies (default 64, ≤ 100)
    voices_per_octave : alternative to nf (MATLAB-style, default None)
    ffact    : filter width factor (default 1.0)
    piover4  : π/4 correction: -1.0 for EGFs, +1.0 for seismograms
    vmin     : minimum group velocity [km/s] for time-window selection
    vmax     : maximum group velocity [km/s] for time-window selection

    Usage
    -----
        from wavelet import AFTANWavelet

        # From an ObsPy Trace (SAC or H5)
        aw = AFTANWavelet(trace, dist_km=300.0, t0=0.0,
                          fmin=1/50, fmax=1/5,
                          nf=64, ffact=1.0, piover4=-1.0,
                          vmin=2.0, vmax=5.0)
        result = aw.compute()

        # result keys:
        #   'per'      : central periods [s]         shape (nf,)
        #   'per_obs'  : observed periods [s]         shape (nf,)
        #   'grvel'    : group velocities [km/s]      shape (nf,)
        #   'phase'    : raw phase at ridge [rad]     shape (nf,)
        #   'amp_db'   : amplitude at ridge [dB]      shape (nf,)
        #   'snr'      : signal/noise ratio [dB]      shape (nf,)
        #   'amp_map'  : full amplitude map [dB]      shape (nf, ncol)
        #   'tamp'     : start time of amp_map [s]
        #   'vel_axis' : velocity axis of amp_map     shape (ncol,)

        # Phase velocity (requires reference phprper/phprvel):
        phvel = aw.phase_velocity(phprper, phprvel)
    """

    def __init__(self,
                 data,
                 dist_km: float,
                 t0: float = 0.0,
                 dt: Optional[float] = None,
                 fmin: float = 1.0 / 50.0,
                 fmax: float = 1.0 / 5.0,
                 nf: int = 64,
                 voices_per_octave: Optional[int] = None,
                 ffact: float = 1.0,
                 piover4: float = -1.0,
                 vmin: float = 2.0,
                 vmax: float = 5.0):

        # ---- resolve data / sampling rate ----
        if isinstance(data, Trace):
            self._sei = data.data.astype(np.float64)
            self._dt  = float(data.stats.delta)
            self._fs  = float(data.stats.sampling_rate)
        elif isinstance(data, np.ndarray):
            if dt is None:
                raise ValueError("Pass dt= when data is a numpy array.")
            self._sei = data.astype(np.float64)
            self._dt  = float(dt)
            self._fs  = 1.0 / self._dt
        else:
            raise TypeError("data must be an ObsPy Trace or a 1-D numpy array.")

        self._n       = len(self._sei)
        self._dist_km = float(dist_km)
        self._t0      = float(t0)
        self._fmin    = float(fmin)
        self._fmax    = float(fmax)
        self._ffact   = float(ffact)
        self._piover4 = float(piover4)
        self._vmin    = float(vmin)
        self._vmax    = float(vmax)

        # ---- frequency axis ----
        if voices_per_octave is not None:
            self._nf = voices_to_nf(fmin, fmax, voices_per_octave)
        else:
            self._nf = min(int(nf), 100)   # FORTRAN cap at 100

        # log-spaced angular frequencies — FORTRAN uses descending order
        # (ome = 2π/tmin is large, omb = 2π/tmax is small)
        ome  = 2.0 * np.pi * fmax   # high freq end
        omb  = 2.0 * np.pi * fmin   # low  freq end
        step = (np.log(omb) - np.log(ome)) / (self._nf - 1)
        self._om  = np.array([np.exp(np.log(ome) + k * step)
                               for k in range(self._nf)])
        self._per = 2.0 * np.pi / self._om   # central periods [s]

        # ---- AFTAN filter width (Trick 6) ----
        self._alpha = ffact * 20.0 * np.sqrt(dist_km / 1000.0)

        print(f"[AFTANWavelet]  dist={dist_km:.1f} km  "
              f"α={self._alpha:.3f}  nf={self._nf}  "
              f"T={self._per[-1]:.1f}–{self._per[0]:.1f} s")

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def nf(self) -> int:
        return self._nf

    @property
    def periods(self) -> np.ndarray:
        return self._per.copy()

    @property
    def frequencies(self) -> np.ndarray:
        return self._om / (2.0 * np.pi)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(self) -> dict:
        """
        Run the full AFTAN filter bank and extract dispersion measurements.

        Returns
        -------
        dict with keys:
            per      : central period [s]           (nf,)
            per_obs  : observed period [s]           (nf,)
            grvel    : group velocity [km/s]         (nf,)
            phase    : raw phase at ridge [rad]      (nf,)
            amp_db   : amplitude at ridge [dB]       (nf,)
            snr      : signal/noise ratio [dB]       (nf,)
            amp_map  : full dB amplitude map         (nf, ncol)
            tamp     : start time of amp_map [s]
            vel_axis : velocity axis [km/s]          (ncol,)
        """
        dt   = self._dt
        n    = self._n
        t0   = self._t0
        dist = self._dist_km

        # ---- time window (group velocity window) ----
        nb = max(1, int((dist / self._vmax - t0) / dt))   # fast end, 0-based
        ne = min(n, int((dist / self._vmin - t0) / dt))   # slow end, 0-based
        tamp = nb * dt + t0

        # ---- next power of 2 for FFT (must fit full signal) ----
        ns = 1
        while ns < n:
            ns *= 2

        # ---- padded, tapered signal ----
        s = self._taper(nb, ne, ns)

        # ---- forward FFT ----
        sf   = np.fft.fft(s)
        dom  = 2.0 * np.pi / (ns * dt)   # angular frequency resolution

        # ---- output arrays ----
        ncol    = ne - nb
        amp_raw = np.full((ncol + 2, self._nf), 40.0)   # dB map
        amp_lin = np.zeros((ncol + 2, self._nf))
        pha_map = np.zeros((ncol + 2, self._nf))

        # ---- filter bank loop ----
        for k in range(self._nf):
            om0 = self._om[k]

            # Trick 1: ratio-based Gaussian in frequency domain
            freqs = np.arange(ns) * dom
            with np.errstate(over='ignore', invalid='ignore'):
                gauss = np.exp(-self._alpha ** 2 * (freqs / om0 - 1.0) ** 2)

            fils = sf * gauss

            # Trick 2: one-sided spectrum → exact analytic signal
            fils[ns // 2 + 1:] = 0.0
            fils[0]       = fils[0].real * 0.5 + 0j
            fils[ns // 2] = fils[ns // 2].real + 0j

            # IFFT — note: FORTRAN uses unnormalised, so multiply by ns
            tmp = np.fft.ifft(fils) * ns

            # extract window
            i0    = max(0, nb - 1)
            i1    = min(ns, ne + 1)
            chunk = tmp[i0:i1]
            jlen  = min(len(chunk), ncol + 2)

            pha_map[:jlen, k]  = np.angle(chunk[:jlen])
            amp_lin[:jlen, k]  = np.abs(chunk[:jlen])
            with np.errstate(divide='ignore', invalid='ignore'):
                amp_db_col = 20.0 * np.log10(
                    np.where(amp_lin[:jlen, k] > 0, amp_lin[:jlen, k], 1e-30))
            amp_raw[:jlen, k] = amp_db_col

        # ---- normalise to 100 dB with 40 dB floor (FORTRAN convention) ----
        amax    = amp_raw.max()
        amp_raw = amp_raw + 100.0 - amax
        amp_raw = np.maximum(amp_raw, 40.0)

        # ---- velocity axis of the map ----
        times    = tamp + np.arange(ncol + 2) * dt
        with np.errstate(divide='ignore', invalid='ignore'):
            vel_axis = np.where(times + t0 > 1e-6,
                                dist / (times + t0), np.nan)

        # ---- ridge extraction with Tricks 3 & 4 ----
        grvel   = np.zeros(self._nf)
        per_obs = np.zeros(self._nf)
        amp_ridge = np.zeros(self._nf)
        phase_ridge = np.zeros(self._nf)
        snr_ridge   = np.zeros(self._nf)

        ntall = ncol + 2

        for k in range(self._nf):
            col_amp  = amp_raw[:ntall, k]
            col_ampl = amp_lin[:ntall, k]
            col_pha  = pha_map[:ntall, k]

            # find global maximum (AFTAN: primary ridge = strongest peak)
            j = int(np.argmax(col_amp))
            j = max(1, min(j, ntall - 2))   # guard edges

            # Trick 3: parabolic sub-sample interpolation
            a0, a1, a2 = col_amp[j-1], col_amp[j], col_amp[j+1]
            denom = a0 - 2.0 * a1 + a2
            t_off = 0.5 * (a0 - a2) / denom if abs(denom) > 1e-30 else 0.0
            t_off = np.clip(t_off, -0.5, 0.5)

            tim_j = (nb + j - 2 + t_off) * dt   # absolute group arrival [s]
            grvel[k] = dist / (tim_j + t0) if (tim_j + t0) > 0 else 0.0

            # Trick 4: observed period from phase derivative
            dph = col_pha[j+1] - col_pha[j-1]   # phase difference over 2 samples
            # unwrap to [-π, π]
            while dph >  np.pi: dph -= 2.0 * np.pi
            while dph < -np.pi: dph += 2.0 * np.pi
            per_obs[k] = (2.0 * np.pi * dt / dph
                          if abs(dph) > 1e-12 else self._per[k])

            # interpolated phase at sub-sample peak
            if t_off >= 0:
                ph = col_pha[j] + t_off * (col_pha[j+1] - col_pha[j])
            else:
                ph = col_pha[j] + t_off * (col_pha[j]   - col_pha[j-1])

            # Trick 5: π/4 correction
            phase_ridge[k] = ph - self._piover4 * np.pi / 4.0

            # amplitude at peak
            amp_ridge[k] = a1 - 0.25 * (a0 - a2) * t_off

            # SNR: geometric mean of surrounding minima
            lm = float(col_ampl[:j+1].min()) if j > 0 else col_ampl[0]
            rm = float(col_ampl[j:].min())   if j < ntall-1 else col_ampl[-1]
            if lm > 0 and rm > 0 and col_ampl[j] > 0:
                snr_ridge[k] = 20.0 * np.log10(
                    col_ampl[j] / np.sqrt(lm * rm))
            else:
                snr_ridge[k] = 0.0

        return dict(
            per      = self._per.copy(),
            per_obs  = per_obs,
            grvel    = grvel,
            phase    = phase_ridge,
            amp_db   = amp_ridge,
            snr      = snr_ridge,
            amp_map  = amp_raw[:ncol, :].T,   # shape (nf, ncol)
            tamp     = tamp,
            vel_axis = vel_axis[:ncol],
        )

    # ------------------------------------------------------------------
    # Phase velocity (requires reference curve)
    # ------------------------------------------------------------------

    def phase_velocity(self,
                       result: dict,
                       phprper: np.ndarray,
                       phprvel: np.ndarray) -> np.ndarray:
        """
        Convert raw phase measurements to phase velocity [km/s] by resolving
        the integer 2π cycle ambiguity using a reference phase velocity curve.

        Implements _phtovel() from aftan.py faithfully.

        Formula (Trick 5 extended):
            Φ_total(ω) = ω · Δ / c(ω)
                       = ω · t_group + φ_measured + piover4·π/4 + n·2π

            n = round( (ω·Δ/c_ref - ω·t_group - φ) / 2π )
            c = ω · Δ / (ω·t_group + φ + piover4·π/4 + n·2π)

        Parameters
        ----------
        result   : dict returned by compute()
        phprper  : reference phase velocity periods [s]
        phprvel  : reference phase velocities [km/s]

        Returns
        -------
        phvel : (nf,) phase velocity [km/s], NaN where unresolvable
        """
        from scipy.interpolate import interp1d

        ref = interp1d(phprper, phprvel, kind='linear',
                       bounds_error=False, fill_value='extrapolate')

        per    = result['per']
        grvel  = result['grvel']
        phase  = result['phase']
        phvel  = np.full(self._nf, np.nan)

        for i in range(self._nf):
            T      = per[i]
            om     = 2.0 * np.pi / T
            t_g    = self._dist_km / grvel[i] if grvel[i] > 0 else 0.0
            phi    = phase[i]
            c_ref  = float(ref(T))

            if c_ref <= 0 or not np.isfinite(c_ref):
                continue

            # integer cycle that minimises |c - c_ref|
            total_ref = om * self._dist_km / c_ref
            n = round((total_ref - om * t_g - phi) / (2.0 * np.pi))

            denom = om * t_g + phi + n * 2.0 * np.pi
            if abs(denom) > 1e-10:
                phvel[i] = om * self._dist_km / denom

        return phvel

    # ------------------------------------------------------------------
    # AFTAN amplitude map as scalogram (Period x Velocity)
    # ------------------------------------------------------------------

    def scalogram_db(self, result: dict,
                     vmin: Optional[float] = None,
                     vmax: Optional[float] = None,
                     n_vel: int = 300, min_db = -5) -> dict:
        """
        Interpolate the AFTAN amplitude map onto a regular
        (period, velocity) grid — same layout as the CWT scalogram
        in AFTANWavelet, so both can be plotted identically.

        Parameters
        ----------
        result  : dict from compute()
        vmin    : min velocity for grid [km/s]  (default self._vmin)
        vmax    : max velocity for grid [km/s]  (default self._vmax)
        n_vel   : number of velocity nodes (default 300)

        Returns
        -------
        dict with keys:
            amp_img  : (n_vel, nf) normalised dB map  (0 dB at peak)
            vel_axis : (n_vel,) ascending velocity axis [km/s]
            per_axis : (nf,)    ascending period axis  [s]
        """
        vlo = vmin if vmin is not None else self._vmin
        vhi = vmax if vmax is not None else self._vmax

        amp_map  = result['amp_map']    # (nf, ncol)
        vel_raw  = result['vel_axis']   # (ncol,)
        per_axis = self._per[::-1]      # ascending (short→long)
        nf       = self._nf
        ncol     = amp_map.shape[1]

        vel_grid = np.linspace(vlo, vhi, n_vel)
        amp_img  = np.full((n_vel, nf), np.nan)

        # amp_map[ki,:] corresponds to self._per[ki] which is DESCENDING
        # (per[0]=shortest, per[nf-1]=longest).
        # We want per_axis ASCENDING (short→long), so:
        #   per_axis[j] = per[nf-1-j]   → j=0 = longest, j=nf-1 = shortest
        # Wait — ascending means per_axis[0] < per_axis[-1]:
        #   per[nf-1] is the LONGEST period → per[::-1][0] = LONGEST
        # That is DESCENDING in value, which is confusing.
        #
        # CORRECT definition: ascending = per_axis[0] = shortest period
        #   per_axis = self._per[::-1] gives [longest,...,shortest] ← WRONG label
        #   We need per_axis = self._per (already has per[0]=shortest) but
        #   amp_img columns must match: col j ↔ per_axis[j]
        #   col_idx = nf-1-ki means ki=0(shortest)→col nf-1(last)
        #   so per_axis[nf-1] should = per[0] = shortest → per_axis = per[::-1]
        #   BUT per[::-1][0]=longest, per[::-1][nf-1]=shortest ← ascending in index,
        #   but DESCENDING in value!
        #
        # Resolution: use per_axis = self._per (per[0]=shortest=col nf-1) but
        # reverse amp_img columns so col 0 ↔ per[0] (shortest).
        # Simplest fix: don't flip — let col 0 = shortest period.

        for ki in range(nf):
            row  = amp_map[ki, :]        # row ki ↔ per[ki]  (per[0]=shortest)
            rv   = vel_raw
            mask = np.isfinite(rv) & (rv >= vlo) & (rv <= vhi)
            if mask.sum() < 2:
                continue
            idx  = np.argsort(rv[mask])
            rv_s = rv[mask][idx]
            ra_s = row[mask][idx]
            # column ki ↔ per[ki]: col 0 = shortest period
            amp_img[:, ki] = np.interp(vel_grid, rv_s, ra_s,
                                        left=np.nan, right=np.nan)

        # per_axis: col 0 = per[0] = shortest period → ascending (short→long)
        per_axis_out = self._per.copy()   # per[0]=shortest, per[nf-1]=longest

        # normalise: 0 dB at peak, clip at -5 dB
        amp_norm = amp_img - np.nanmax(amp_img)
        amp_norm = np.clip(amp_norm, min_db, 0.0)

        return dict(amp_img=amp_norm, vel_axis=vel_grid,
                    per_axis=per_axis_out)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _taper(self, nb: int, ne: int, ns: int) -> np.ndarray:
        """
        Cosine taper + zero-pad to ns.
        Identical logic to _taper_v2() in aftan.py.
        """
        dt     = self._dt
        taperl = 1.5   # left taper factor (same as AFTAN default)
        tmax   = self._per[0]   # longest period = left taper reference

        ntapb = max(1, int(taperl * tmax / dt))
        ntape = max(1, int(tmax / dt))

        s = np.zeros(ns, dtype=np.complex128)
        s[:self._n] = self._sei[:self._n].astype(np.complex128)

        # left cosine taper
        if ntapb > 0:
            ntapb = min(ntapb, self._n)
            ramp  = np.arange(ntapb)
            s[:ntapb] *= 0.5 * (1.0 - np.cos(np.pi * ramp / ntapb))

        # right cosine taper
        if ntape > 0:
            ntape = min(ntape, self._n)
            i0    = max(0, self._n - ntape)
            ramp  = np.arange(ntape)
            s[i0:i0 + ntape] *= 0.5 * (1.0 - np.cos(
                np.pi * (ntape - 1 - ramp) / ntape))

        # zero outside velocity window
        if ne < ns:
            s[ne:] = 0.0

        return s

