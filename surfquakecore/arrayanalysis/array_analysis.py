#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
array_analysis

    Created on Tue Feb  6 12:03:39 2018
    @author: robertocabieces

ARF, FK / Capon / multitaper-coherence slowness maps, MUSIC, vespagrams and
beam stacking.

Conventions used throughout this module
---------------------------------------
* Station geometry comes from ``obspy.signal.array_analysis.get_geometry``:
  x = East, y = North, in km, relative to the array centre.
* A plane wave with *propagation* slowness vector ``s0`` (s/km) reaches a
  station at position ``r`` with delay ``tau = s0 . r`` relative to the centre.
* The steering vector is ``a_i(f, s) = exp(-2j*pi*f * s.r_i)`` and the beam
  power is ``a^H R a``, which peaks at ``s = s0`` (propagation direction).
* The slowness maps returned by ``FKCoherence`` and ``run_music`` are laid out
  so that ``Z[iy, ix]`` is the power for the vector ``(grid[ix], grid[iy])``
  that points **towards the source** (i.e. ``-s0``). With
  ``contourf(X, Y, Z)`` and ``X, Y = meshgrid(grid, grid)`` the peak therefore
  sits in the back-azimuth direction, and the returned ``Sxpow, Sypow`` are
  the coordinates of that peak. ``atan2(Sypow, Sxpow)`` -> back-azimuth.
* ``grid = np.linspace(-slim, slim, nx)`` with
  ``nx = len(np.arange(-slim, slim, sinc))``: same size as the axes built by
  the callers (plot.py, beamrun.py), but symmetric, so the plotted axis and
  the slowness actually evaluated coincide.
"""

import numpy as np
import matplotlib.dates as mdt
from obspy.core.util import AttribDict
from obspy.signal.array_analysis import array_processing
from obspy.signal.array_analysis import array_transff_freqslowness
from obspy.signal.array_analysis import get_geometry
import nitime.algorithms as alg
from nitime import utils
from scipy import fftpack
from scipy.linalg import eigh
from scipy.signal import hilbert
from scipy.fftpack import next_fast_len
from obspy import Trace, Stream, UTCDateTime
from surfquakecore.arrayanalysis.beamforming_tools import regularize_covariance


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------

def _to_utc(t):
    if isinstance(t, UTCDateTime):
        return t
    return UTCDateTime(mdt.num2date(t))


def _attach_coordinates(st, inv):
    """Attach lat/lon/elevation to every trace. Elevation is converted from
    metres (StationXML) to km, which is what get_geometry expects."""
    for tr in st:
        c = inv.get_coordinates(tr.id, tr.stats.starttime)
        tr.stats.coordinates = AttribDict({'latitude': c['latitude'],
                                           'longitude': c['longitude'],
                                           'elevation': c['elevation'] / 1000.0})


def _equalize(st):
    """All traces must share sampling rate and number of samples."""
    if len(st) < 2:
        raise ValueError("Array analysis needs at least two traces")
    fs = {round(tr.stats.sampling_rate, 6) for tr in st}
    if len(fs) != 1:
        raise ValueError("All traces must have the same sampling rate, got %s" % sorted(fs))
    npts = min(tr.stats.npts for tr in st)
    for tr in st:
        tr.data = tr.data[:npts]
    return npts


def _slowness_grid(slim, sinc):
    nx = len(np.arange(-slim, slim, sinc))
    return np.linspace(-slim, slim, nx)


def _band(freqs, fmin, fmax):
    """Indices of the frequency bins inside [fmin, fmax] (both inclusive)."""
    idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    if idx.size == 0:  # band narrower than one bin: take the nearest bin
        idx = np.array([np.argmin(np.abs(freqs - 0.5 * (fmin + fmax)))])
    if freqs[idx[0]] == 0.0:  # never use DC
        idx = idx[1:]
    if idx.size == 0:
        raise ValueError("No usable frequency bins between %.3f and %.3f Hz" % (fmin, fmax))
    return idx


def _detrended_matrix(st, npts):
    m = np.zeros((npts, len(st)))
    for i, tr in enumerate(st):
        m[:, i] = tr.data[:npts] - np.mean(tr.data[:npts])
    return m


def _csdm_single(m, fs, fmin, fmax, window=None):
    """Cross-spectral density matrix from a single snapshot.
    m: (npts, nr).  Returns freqs (nf,), R (nf, nr, nr) with R_ij = X_i X_j*."""
    npts = m.shape[0]
    if window is not None:
        m = m * window[:, None]
    X = np.fft.rfft(m, axis=0)
    freqs = np.fft.rfftfreq(npts, d=1.0 / fs)
    idx = _band(freqs, fmin, fmax)
    X = X[idx]                                     # (nf, nr)
    R = X[:, :, None] * np.conj(X[:, None, :])
    return freqs[idx], R


def _csdm_welch(m, fs, fmin, fmax, nsub=3):
    """Snapshot-averaged CSDM (Hann sub-windows of npts/nsub, 75 % overlap).
    Needed by Capon and MUSIC: a single-snapshot CSDM has rank 1.
    Sub-windows must stay long compared with the inter-station moveout,
    otherwise the same window cuts different parts of the wavelet at each
    station and the narrow-band model breaks (biased MUSIC slowness)."""
    npts, nr = m.shape
    L = max(npts // nsub, 8)
    # keep at least ~3 bins inside the band
    while L < npts and (fs / L) > (fmax - fmin) / 3.0:
        L = min(2 * L, npts)
    step = max(L // 4, 1)
    starts = range(0, npts - L + 1, step)
    w = np.hanning(L)
    freqs = np.fft.rfftfreq(L, d=1.0 / fs)
    idx = _band(freqs, fmin, fmax)
    R = np.zeros((idx.size, nr, nr), dtype=np.complex128)
    nseg = 0
    for s0 in starts:
        seg = m[s0:s0 + L] - m[s0:s0 + L].mean(axis=0)
        X = np.fft.rfft(seg * w[:, None], axis=0)[idx]
        R += X[:, :, None] * np.conj(X[:, None, :])
        nseg += 1
    return freqs[idx], R / nseg, nseg


def _mtp_coherency(m, fs, fmin, fmax, NW=2):
    """Multitaper complex coherency matrix C_ij(f) = S_ij / sqrt(S_ii S_jj)."""
    npts, nr = m.shape
    K = 2 * NW - 1
    tapers, eigs = alg.dpss_windows(npts, NW, K)
    pdata = m.T
    tspectra = fftpack.fft(tapers[None, :, :] * pdata[:, None, :])
    w = np.empty((nr, K, npts // 2 + 1))
    for i in range(nr):
        w[i], _ = utils.adaptive_weights(tspectra[i], eigs, sides='onesided')
    freqs = np.fft.rfftfreq(npts, d=1.0 / fs)
    idx = _band(freqs, fmin, fmax)
    auto = np.array([alg.mtm_cross_spectrum(tspectra[i], tspectra[i], w[i], sides='onesided')
                     for i in range(nr)]).real
    C = np.empty((idx.size, nr, nr), dtype=np.complex128)
    for i in range(nr):
        C[:, i, i] = 1.0
        for j in range(i + 1, nr):
            sxy = alg.mtm_cross_spectrum(tspectra[i], tspectra[j], (w[i], w[j]), sides='onesided')
            c = (sxy / np.sqrt(auto[i] * auto[j]))[idx]
            C[:, i, j] = c
            C[:, j, i] = np.conj(c)
    return freqs[idx], C


def _delays(svec, r):
    """svec (..., 2) propagation slowness vectors, r (nr, 2) -> tau (..., nr)."""
    return svec @ r.T


def _quad(A, M):
    """Real part of a^H M a for every steering vector. A (..., nr), M (nr, nr)."""
    return np.real(np.einsum('...i,ij,...j->...', np.conj(A), M, A))


def _beam_power(method, freqs, R, svec, r, n_signals=1):
    """Broadband beam power for a set of propagation slowness vectors.

    FK            : relative power  sum_f a^H R a / (nr * sum_f tr R)   in [0, 1]
                    (same definition as ObsPy's relpow)
    CAPON         : relative Capon  sum_f 1/(a^H R^-1 a) / (sum_f tr R / nr)
    MTP.COHERENCE : mean_f |a^H C a| / nr^2                              in [0, 1]
    MUSIC         : mean_f 1 / (a_n^H En En^H a_n), a_n = a/sqrt(nr)
    """
    nr = r.shape[0]
    tau = _delays(svec, r)                           # (..., nr)
    out = np.zeros(tau.shape[:-1])
    norm = 0.0
    for k, f in enumerate(freqs):
        A = np.exp(-2j * np.pi * f * tau)
        Rf = R[k]
        if method == "FK":
            out += _quad(A, Rf)
            norm += nr * np.real(np.trace(Rf))
        elif method == "CAPON":
            Rinv = regularize_covariance(Rf, loading_factor=1E-3)
            out += 1.0 / np.maximum(_quad(A, Rinv), 1e-30)
            norm += np.real(np.trace(Rf)) / nr
        elif method == "MTP.COHERENCE":
            out += np.abs(_quad(A, Rf)) / nr ** 2
            norm += 1.0
        elif method == "MUSIC":
            vals, vecs = eigh(Rf)                     # ascending eigenvalues
            En = vecs[:, :nr - n_signals]
            proj = np.einsum('...i,ik->...k', np.conj(A), En) / np.sqrt(nr)
            out += 1.0 / (np.sum(np.abs(proj) ** 2, axis=-1) + 1e-10)
            norm += 1.0
        else:
            raise ValueError("Unknown beamforming method: %s" % method)
    return out / norm


def _slowness_map(method, freqs, R, r, slim, sinc, n_signals=1):
    """Map laid out as described in the module docstring (toward-source axes)."""
    grid = _slowness_grid(slim, sinc)
    GX, GY = np.meshgrid(grid, grid)                 # GX[iy, ix] = grid[ix]
    svec = -np.stack([GX, GY], axis=-1)              # propagation vector = -(toward source)
    Pow = _beam_power(method, freqs, R, svec, r, n_signals)
    iy, ix = np.unravel_index(np.argmax(Pow), Pow.shape)
    return Pow, np.array([grid[ix]]), np.array([grid[iy]])


def _prepare_window(st_ini, inv, t1, win_len):
    t_start = _to_utc(t1)
    st = st_ini.copy()
    st.trim(starttime=t_start, endtime=t_start + win_len)
    st.sort()
    npts = _equalize(st)
    _attach_coordinates(st, inv)
    coord = get_geometry(st, coordsys='lonlat', return_center=True)
    return st, npts, coord


class array:

    def __init__(self):
        """
        ARF, (FK, Capon and Multitaper) Coherence program, MUSIC and Vespagram.

        :param No params required to initialize the class
        """

    def arf(self, coords, fmin, flim, slim, sgrid):

        sstep = sgrid
        fmax = flim
        fstep = (fmax - fmin) / 40 if fmax > fmin else fmax / 40
        transff = array_transff_freqslowness(coords, slim, sstep, fmin, fmax, fstep, coordsys='lonlat')

        return transff

    def azimuth2mathangle(self, azimuth):
        """
        Convert an azimuth (clockwise from North)
        to a mathematical angle (counterclockwise from East).
        """
        res = (90 - np.asarray(azimuth, dtype=float)) % 360
        return res.item() if res.ndim == 0 else res

    def mathangle2azimuth(self, mathangle):
        """
        Convert a mathematical angle (counterclockwise from East)
        to an azimuth (clockwise from North).
        """
        res = (90 - np.asarray(mathangle, dtype=float)) % 360
        return res.item() if res.ndim == 0 else res

    def FK(self, st, inv, stime, etime, fmin, fmax, slim, sres, win_len, win_frac):

        _attach_coordinates(st, inv)

        kwargs = dict(
            # slowness grid: X min, X max, Y min, Y max, Slow Step
            sll_x=-1 * slim, slm_x=slim, sll_y=-1 * slim, slm_y=slim, sl_s=sres,
            # sliding window properties
            win_len=win_len, win_frac=win_frac,
            # frequency properties
            frqlow=fmin, frqhigh=fmax, prewhiten=0,
            # restrict output
            semb_thres=-1e9, vel_thres=-1e9, timestamp='mlabday',
            stime=stime + 0.1, etime=etime - 0.1)

        try:
            out = array_processing(st, **kwargs)

            T = out[:, 0]
            relpower = out[:, 1]
            abspower = out[:, 2]
            AZ = out[:, 3]          # back-azimuth, ObsPy returns [-180, 180]
            AZ[AZ < 0.0] += 360
            Slowness = out[:, 4]

        except Exception as e:
            print("Check Parameters and Starttime/Endtime:", e)

            relpower = []
            abspower = []
            AZ = []
            Slowness = []
            T = []

        return relpower, abspower, AZ, Slowness, T

    def FKCoherence(self, st_ini, inv, t1, linf, lsup, slim, win_len, sinc, method):
        """
        Slowness map of the window [t1, t1 + win_len].

        method: "FK" (relative Bartlett power), "CAPON" (relative Capon power,
        snapshot-averaged CSDM) or "MTP.COHERENCE" (multitaper coherency).

        Returns Pow, Sxpow, Sypow, coord.  Sxpow/Sypow (1-element arrays) point
        towards the source: back-azimuth = atan2(Sxpow, Sypow) clockwise from N,
        slowness = hypot(Sxpow, Sypow).
        """
        st, npts, coord = _prepare_window(st_ini, inv, t1, win_len)
        fs = st[0].stats.sampling_rate
        m = _detrended_matrix(st, npts)
        r = coord[:-1, 0:2]

        if method == "FK":
            freqs, R = _csdm_single(m, fs, linf, lsup, window=np.hamming(npts))
        elif method == "CAPON":
            freqs, R, _ = _csdm_welch(m, fs, linf, lsup)
        elif method == "MTP.COHERENCE":
            freqs, R = _mtp_coherency(m, fs, linf, lsup)
        else:
            raise ValueError("Unknown method %s" % method)

        Pow, Sxpow, Sypow = _slowness_map(method, freqs, R, r, slim, sinc)
        return Pow, Sxpow, Sypow, coord

    def run_music(self, st, inv, starttime, linf, lsup, slim, win_len, sinc, n_signals=1):
        """
        Narrow-band MUSIC, averaged over the frequency bins in [linf, lsup].
        For each bin the CSDM is estimated from overlapping sub-windows,
        the noise subspace is taken from its nr - n_signals smallest
        eigenvectors, and the pseudospectra are averaged incoherently.
        Same output layout as FKCoherence.
        """
        t_start = _to_utc(starttime)
        st = st.copy()                     # never modify the caller's stream
        st.detrend(type="linear")
        st.taper(type="cosine", max_percentage=0.05)
        st.filter(type="bandpass", freqmin=linf, freqmax=lsup, zerophase=True, corners=4)
        st, npts, coord = _prepare_window(st, inv, t_start, win_len)
        if not 1 <= n_signals < len(st):
            raise ValueError("n_signals must be between 1 and number of stations - 1")
        fs = st[0].stats.sampling_rate
        m = _detrended_matrix(st, npts)
        r = coord[:-1, 0:2]
        freqs, R, nseg = _csdm_welch(m, fs, linf, lsup)
        # spatial-only diagonal loading keeps eigh stable when nseg < nr
        for k in range(R.shape[0]):
            R[k] += 1e-6 * np.real(np.trace(R[k])) / R.shape[1] * np.eye(R.shape[1])
        music_map, Sxpow, Sypow = _slowness_map("MUSIC", freqs, R, r, slim, sinc, n_signals)
        return music_map, Sxpow, Sypow, coord

    def stack_stream(self, st_shift, sx, sy, coord, stack_type='Linear Stack'):
        """
        Delay-and-sum beam.

        sx, sy: Sxpow, Sypow as returned by FKCoherence/run_music (vector pointing
        to the source). coord: geometry returned by the same call. Traces are
        sorted exactly as in FKCoherence so rows of coord match.

        Every trace is advanced by its plane-wave delay tau_i = s0 . r_i
        (s0 = -(sx, sy)) using sub-sample linear interpolation onto a common
        time base, so all traces have identical length.
        """
        st_shift = st_shift.copy()
        st_shift.sort()
        s0 = -np.array([np.ravel(sx)[0], np.ravel(sy)[0]])
        r = np.asarray(coord)[:len(st_shift), 0:2]
        if r.shape[0] != len(st_shift):
            raise ValueError("coord and stream have different number of stations")
        tau = r @ s0

        fs = st_shift[0].stats.sampling_rate
        ref = min(tr.stats.starttime for tr in st_shift)
        t_beg = max((tr.stats.starttime - ref) - tau[i] for i, tr in enumerate(st_shift))
        t_end = min((tr.stats.endtime - ref) - tau[i] for i, tr in enumerate(st_shift))
        if t_end <= t_beg:
            raise ValueError("No common time span after alignment")
        n = int(np.floor((t_end - t_beg) * fs)) + 1
        t_common = t_beg + np.arange(n) / fs
        starttime = ref + t_beg

        N = len(st_shift)
        mat = np.zeros((N, n))
        all_traces_shift = []
        for i, tr in enumerate(st_shift):
            t_tr = (tr.stats.starttime - ref) + np.arange(tr.stats.npts) / fs
            mat[i, :] = np.interp(t_common + tau[i], t_tr, tr.data)
            header = {'network': tr.stats.network, 'station': tr.stats.station,
                      'location': tr.stats.location, 'channel': tr.stats.channel,
                      'npts': n, 'sampling_rate': fs, 'mseed': {'dataquality': 'D'},
                      'starttime': starttime}
            all_traces_shift.append(Trace(data=mat[i].copy(), header=header))

        all_stream_shifted = Stream(all_traces_shift)
        stack = self.stack(mat, stack_type=stack_type, order=2)
        stats = {'network': st_shift[0].stats.network, 'station': 'STACK', 'location': '',
                 'channel': st_shift[0].stats.channel, 'npts': len(stack),
                 'sampling_rate': fs, 'mseed': {'dataquality': 'D'},
                 'starttime': starttime}
        trace_stack = Trace(data=stack, header=stats)

        return all_stream_shifted, trace_stack

    def stack_seismograms(self, st):
        maxstart = np.max([tr.stats.starttime for tr in st])
        minend = np.min([tr.stats.endtime for tr in st])
        st.trim(maxstart, minend)
        npts = _equalize(st)
        fs = st[0].stats.sampling_rate
        time = np.arange(npts) / fs
        serial_time = st[0].times("matplotlib")
        mat = np.zeros([len(st), npts])
        for i, tr in enumerate(st):
            mat[i, :] = tr.data

        stats = {'network': st[0].stats.network, 'station': 'STACK', 'location': '',
                 'channel': st[0].stats.channel, 'npts': npts,
                 'sampling_rate': fs, 'mseed': {'dataquality': 'D'},
                 'starttime': st[0].stats.starttime}

        return mat, time, stats, serial_time

    def stack(self, data, stack_type='Linear Stack', order=2):

        """
        Stack data by first axis.

        :type stack_type: str
        :param stack_type: Type of stack, one of the following:
            ``'Linear Stack'``: average stack (default),
            ``'Phase Weigth Stack'``: phase weighted stack of given order
            (Schimmel & Paulssen 1997; order 0 corresponds to linear stack),
            ``'root'``: root stack of given order
            (order 1 corresponds to linear stack).
        """

        if stack_type == 'Linear Stack':
            stack = np.mean(data, axis=0)

        elif stack_type == 'Phase Weigth Stack':
            npts = np.shape(data)[1]
            nfft = next_fast_len(npts)
            anal_sig = hilbert(data, N=nfft)[:, :npts]
            # phase coherence uses unit phasors exp(i*phi), not the analytic signal itself
            phasors = anal_sig / np.maximum(np.abs(anal_sig), np.finfo(float).tiny)
            phase_stack = np.abs(np.mean(phasors, axis=0)) ** order
            stack = np.mean(data, axis=0) * phase_stack
        elif stack_type == 'root':
            r = np.mean(np.sign(data) * np.abs(data) ** (1 / order), axis=0)
            stack = np.sign(r) * np.abs(r) ** order
        else:
            raise ValueError('stack type is not valid.')

        return stack


class vespagram_util:

    def __init__(self, st, linf, lsup, win_len, slow, baz, inv, t1, t2, res, selection="Slowness", method="FK"):

        """
        Vespagram, computed at a fixed slowness or backazimuth

        :param params required to initialize the class:
        linf min frequency (Hz)
        lsup max frequency (Hz)
        win_len win size in seconds to be analyzed in every step
        slow = slowness of analysis (s/km)
        baz backazimuth of analysis (deg, clockwise from N)
        method "FK", "CAPON" or "MTP.COHERENCE"
        """

        self.st = st
        self.t1 = t1
        self.t2 = t2
        self.inv = inv
        self.method = method
        self.linf = linf
        self.lsup = lsup
        self.win_len = win_len
        self.slow = slow
        self.res = res
        self.selection = selection
        # propagation azimuth = back-azimuth + 180
        self.baz = (baz + 180) % 360

    def azimuth2mathangle(self, azimuth):
        res = (90 - np.asarray(azimuth, dtype=float)) % 360
        return res.item() if res.ndim == 0 else res

    def mathangle2azimuth(self, mathangle):
        res = (90 - np.asarray(mathangle, dtype=float)) % 360
        return res.item() if res.ndim == 0 else res

    def vespa_deg(self):
        st = self.st.copy()
        st.trim(starttime=self.t1, endtime=self.t2)
        st.sort()
        _attach_coordinates(st, self.inv)
        self._coord = get_geometry(st, coordsys='lonlat', return_center=True)[:-1, 0:2]
        self._st = st
        return self.__vespa()

    def __vespa(self):

        st0 = self._st
        fs = st0[0].stats.sampling_rate
        delta = 1.0 / fs
        win_len = int(round(self.win_len * fs))
        npts = min(tr.stats.npts for tr in st0)
        lim = npts - win_len
        if lim <= 0:
            raise ValueError("Window longer than the selected time span")

        x = (self.res / 100) * win_len
        step = max(npts / x, 1.0)
        steps = np.arange(0, lim, step)

        vespa_spec = np.zeros([360, len(steps)])
        t_ref = st0[0].stats.starttime
        for j, n in enumerate(steps):
            st = st0.copy()
            start_time = t_ref + n * delta
            st.trim(starttime=start_time, endtime=start_time + self.win_len)
            if self.selection == "Slowness":
                vespa_spec[:, j] = self.__vespa_slow(st)
            else:
                vespa_spec[:, j] = self.__vespa_az(st)

        # time of each column = centre of its window, relative to t1
        t = (t_ref - self.t1) + steps * delta + 0.5 * self.win_len

        if self.selection == "Slowness":
            x, y = np.meshgrid(t, np.arange(360))
        else:
            x, y = np.meshgrid(t, np.linspace(0, self.slow * (1 + 0.15), vespa_spec.shape[0]))

        return x, y, vespa_spec

    def _csdm(self, st):
        npts = _equalize(st)
        fs = st[0].stats.sampling_rate
        m = _detrended_matrix(st, npts)
        if self.method == "FK":
            freqs, R = _csdm_single(m, fs, self.linf, self.lsup, window=np.hamming(npts))
        elif self.method == "CAPON":
            freqs, R, _ = _csdm_welch(m, fs, self.linf, self.lsup)
        elif self.method == "MTP.COHERENCE":
            freqs, R = _mtp_coherency(m, fs, self.linf, self.lsup)
        else:
            raise ValueError("Unknown method %s" % self.method)
        return freqs, R

    def __vespa_slow(self, st):
        """Power vs back-azimuth (0..359 deg) at fixed slowness."""
        freqs, R = self._csdm(st)
        prop_az = np.radians(np.arange(360) + 180.0)
        svec = self.slow * np.stack([np.sin(prop_az), np.cos(prop_az)], axis=-1)
        return _beam_power(self.method, freqs, R, svec, self._coord)

    def __vespa_az(self, st):
        """Power vs slowness (0..1.15*slow) at fixed back-azimuth."""
        freqs, R = self._csdm(st)
        slow_range = np.linspace(0, self.slow * (1 + 0.15), 360)
        az = np.radians(self.baz)
        svec = slow_range[:, None] * np.array([np.sin(az), np.cos(az)])[None, :]
        return _beam_power(self.method, freqs, R, svec, self._coord)