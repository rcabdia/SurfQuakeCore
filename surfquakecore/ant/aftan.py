"""
aftan.py  –  Automatic Frequency-Time ANalysis (AFTAN)
=======================================================
Pure-Python / NumPy / SciPy translation of the original FORTRAN routines
  • aftanpg.f   (regular FTAN + jump correction)
  • aftanipg.f  (iterative FTAN with phase-match filter)

Original FORTRAN author : M. Barmine, CIEI, CU  (v2.00, 2006)
Python translation       : Claude Sonnet, 2026

Dependencies
------------
  numpy, scipy, obspy

Quick-start example
-------------------
    from obspy import read
    from aftan import aftanpg, aftanipg

    st  = read("example.SAC")
    tr  = st[0]
    sei = tr.data.astype(np.float32)
    n   = len(sei)
    dt  = tr.stats.delta
    t0  = float(tr.stats.sac.b)          # begin-time from SAC header
    dist = float(tr.stats.sac.dist)      # epicentral distance [km]

    nfout1, arr1, nfout2, arr2, tamp, amp, ierr = aftanpg(
        piover4 = 1.0,   # use -1.0 for cross-correlations
        n       = n,
        sei     = sei,
        t0      = t0,
        dt      = dt,
        delta   = dist,
        vmin    = 2.0,   vmax   = 5.0,
        tmin    = 5.0,   tmax   = 150.0,
        tresh   = 10.0,
        ffact   = 1.0,
        perc    = 50.0,
        npoints = 5,
        taperl  = 1.5,
        nfin    = 64,
        nphpr   = 0,     phprper=None, phprvel=None,
    )

Mathematical background
-----------------------
AFTAN constructs a 2-D amplitude map A(ω, t) by applying a set of narrow
Gaussian band-pass filters centred at angular frequencies ωₖ (log-spaced
between ωmin and ωmax):

    H(ω; ωₖ) = exp[ -α² (ω/ωₖ - 1)² ]        (Gaussian filter)

where  α = ffact · 20 · √(Δ/1000)   controls the width adaptively with
inter-station distance Δ [km].

After filtering, the analytic signal is obtained via one-sided spectrum
(Hilbert transform trick):
    • set negative-frequency bins to zero
    • IFFT  →  complex envelope  x̃(t)
    • amplitude   A(t, ωₖ) = |x̃(t)|
    • instantaneous phase φ(t, ωₖ) = arg[x̃(t)]

The group-velocity ridge follows the time t* of maximum amplitude at
each ωₖ:  U(ωₖ) = Δ / (t* + t0)

The observed (instantaneous) period is extracted from the local phase
derivative:  T_obs = 2π dt / Δφ

Phase velocity (when a reference curve is provided) uses the unwrapped
phase along the group-velocity ridge plus the pi/4 correction for
surface-wave geometric spreading.

Jump correction (trigger / anti-jump)
--------------------------------------
The `trigger` function fits a smooth polynomial through the raw group-
velocity curve and flags points where the residual exceeds `tresh` × σ.
When a jump is found spanning ≤ npoints frequency bins, the algorithm
searches for an alternative local maximum in A(ω, t) that minimises the
velocity discontinuity.

Phase-match filter (aftanipg only)
------------------------------------
A reference group-velocity prediction pred(T) is used to build a phase
correction operator:

    φ_corr(ω) = ω · t_group(ω)   where t_group = Δ / U_pred(ω)

This is applied as a phase shift in the spectral domain to align the
surface-wave packet to near t=0 before the narrow-band filtering step,
dramatically improving the separation of overlapping modes.
"""

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import hilbert
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _taper(nb: int, ne: int, n: int, sei: np.ndarray,
           ntapb: int, ntape: int) -> np.ndarray:
    """
    Cosine taper applied to seismogram array `sei`.

    Parameters
    ----------
    nb, ne   : first/last sample indices (1-based) of the window of interest
    ntapb    : number of samples for left-end cosine ramp
    ntape    : number of samples for right-end cosine ramp

    Returns
    -------
    s        : complex128 array of length ns (next power of 2 ≥ ne)
    ns       : length of padded array
    """
    # power-of-2 padding
    ns = 1
    while ns < n:
        ns *= 2

    s = np.zeros(ns, dtype=np.complex128)
    # copy data
    for i in range(n):
        s[i] = complex(sei[i], 0.0)

    # left cosine taper
    ntb = max(nb - 1, ntapb)
    for i in range(ntb):
        fac = 0.5 * (1.0 - np.cos(np.pi * i / ntapb)) if i < ntapb else 1.0
        if i < n:
            s[i] *= fac

    # right cosine taper
    nte = min(ne - 1, n - ntape - 1)
    for i in range(nte, n):
        dist_from_end = n - 1 - i
        fac = 0.5 * (1.0 - np.cos(np.pi * dist_from_end / ntape)) if dist_from_end < ntape else 0.0
        s[i] *= fac

    return s, ns


def _taper_v2(nb: int, ne: int, n: int, sei: np.ndarray,
              ntapb: int, ntape: int) -> Tuple[np.ndarray, int]:
    """
    Vectorised version of the FORTRAN taper subroutine.
    Faithful to the original logic: cosine ramp on the left from sample 0
    up to sample ntapb, and on the right from n-ntape to n-1.  Everything
    outside [0, ne] is zeroed.
    """
    ns = 1
    while ns < n:
        ns *= 2

    s = np.zeros(ns, dtype=np.complex128)
    s[:n] = sei[:n].astype(np.complex128)

    # left cosine taper (samples 0 … ntapb-1)
    if ntapb > 0:
        ramp = np.arange(ntapb)
        s[:ntapb] *= 0.5 * (1.0 - np.cos(np.pi * ramp / ntapb))

    # right cosine taper (samples n-ntape … n-1)
    if ntape > 0:
        ramp = np.arange(ntape - 1, -1, -1)
        idx  = np.arange(n - ntape, n)
        mask = (idx >= 0) & (idx < ns)
        s[idx[mask]] *= 0.5 * (1.0 - np.cos(np.pi * ramp[mask] / ntape))

    # zero beyond ne
    s[ne:] = 0.0
    return s, ns


def _ftfilt(alpha: float, om0: float, dom: float,
            ns: int, sf: np.ndarray) -> np.ndarray:
    """
    Apply Gaussian band-pass filter centred at om0 to spectrum sf.

    H(ω) = exp( -α² (ω/ω₀ - 1)² )

    Returns filtered one-sided spectrum (complex128, length ns).
    """
    fils = np.zeros(ns, dtype=np.complex128)
    freqs = np.arange(ns) * dom          # ω array
    with np.errstate(over='ignore'):
        gauss = np.exp(-alpha**2 * (freqs / om0 - 1.0)**2)
    fils[:] = sf * gauss
    return fils


def _fmax(a0: float, a1: float, a2: float,
          p0: float, p1: float, p2: float,
          om: float, dt: float,
          piover4: float) -> Tuple[float, float, float, float]:
    """
    Sub-sample parabolic interpolation of amplitude maximum and phase.

    Matches the FORTRAN fmax subroutine.

    Returns
    -------
    t    : fractional sample offset of maximum from centre sample
    dph  : phase increment (used to compute observed period)
    tm   : interpolated amplitude at maximum
    ph   : phase at maximum (corrected by pi/4*piover4)
    """
    pi = np.pi
    # parabolic peak in amplitude
    denom = a0 - 2.0 * a1 + a2
    if abs(denom) < 1e-30:
        t = 0.0
    else:
        t = 0.5 * (a0 - a2) / denom

    # interpolated amplitude
    tm = a1 - 0.25 * (a0 - a2) * t

    # phase at interpolated maximum using linear interp
    if t >= 0.0:
        ph_interp = p1 + t * (p2 - p1)
    else:
        ph_interp = p1 + t * (p1 - p0)

    # phase difference  Δφ = d(phase)/d(sample) * dt  ≈ ω_inst * dt
    dph = p2 - p0
    # unwrap
    while dph >  pi: dph -= 2.0 * pi
    while dph < -pi: dph += 2.0 * pi

    # pi/4 geometric correction for surface waves
    ph = ph_interp - piover4 * pi / 4.0

    return t, dph, tm, ph


def _trigger(grvel: np.ndarray, om: np.ndarray,
             nf: int, tresh: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Detect jumps in the group-velocity curve.

    The FORTRAN 'trigger' fits a low-degree polynomial to U(ω) vs ω,
    then flags points where |residual| / mad > tresh.

    Returns
    -------
    trig1  : residual-based flag array (float), shape (nf,)
    ftrig  : fractional trigger value array
    ierr   : 0 = no jumps, 1 = jumps detected
    """
    trig1 = np.zeros(nf)
    ftrig = np.zeros(nf)

    if nf < 4:
        return trig1, ftrig, 0

    # fit quadratic to log(U) vs log(ω)
    log_om  = np.log(om[:nf])
    log_u   = np.log(np.maximum(grvel[:nf], 1e-6))

    try:
        coeffs  = np.polyfit(log_om, log_u, deg=min(3, nf - 1))
        log_u_fit = np.polyval(coeffs, log_om)
        residuals = log_u - log_u_fit
    except Exception:
        return trig1, ftrig, 0

    mad = np.median(np.abs(residuals))
    if mad < 1e-10:
        return trig1, ftrig, 0

    norm_res = residuals / mad
    trig1[:nf] = norm_res

    # fractional trigger: smoothed absolute residual / tresh
    for i in range(nf):
        ftrig[i] = abs(norm_res[i]) / tresh

    ierr = 1 if np.any(np.abs(norm_res) >= tresh) else 0
    return trig1, ftrig, ierr


def _snr(ampo: np.ndarray, m: int, iml: int, imr: int,
         ntall: int) -> Tuple[float, float]:
    """
    Compute signal-to-noise ratio and half-width for a single local maximum
    at row index m (0-based) within ampo[iml:imr].

    SNR = 20 log10( A_max / sqrt(A_left_min * A_right_min) )

    Returns (snr_db, half_width_samples)
    """
    col = ampo[:ntall, 0]  # dummy; caller passes 1-D slice
    # left minimum
    lm = ampo[m]
    indl = iml
    for mi in range(iml, m + 1):
        if ampo[mi] <= lm:
            lm = ampo[mi]
            indl = mi

    # right minimum
    rm = ampo[m]
    indr = imr
    for mi in range(m, imr + 1):
        if mi < len(ampo) and ampo[mi] <= rm:
            rm = ampo[mi]
            indr = mi

    snr_val = 0.0
    if lm > 0 and rm > 0 and ampo[m] > 0:
        snr_val = 20.0 * np.log10(ampo[m] / np.sqrt(lm * rm))
    if indl == 0 and indr == ntall - 1:
        snr_val += 100.0

    half_width = 0.5 * (abs(m - indl) + abs(m - indr))
    return snr_val, half_width


def _phtovel(delta: float, nfout: int,
             tvis: np.ndarray, grvel: np.ndarray,
             phgr: np.ndarray, phprper: np.ndarray,
             phprvel: np.ndarray) -> np.ndarray:
    """
    Convert measured phase (radians) to phase velocity using a reference
    phase-velocity curve (phprper, phprvel).

    The relationship between group time t_g, phase φ, and phase velocity c is:

        c(ω) = ω · Δ / ( ω · t_g + n·2π + φ_corr )

    where integer n is chosen to minimise |c - c_ref|.

    Returns phgrc : phase velocities [km/s], shape (nfout,)
    """
    pi = np.pi
    phgrc = np.zeros(nfout)

    # build reference interpolator
    ref_interp = interp1d(phprper, phprvel, kind='linear',
                          bounds_error=False, fill_value='extrapolate')

    for i in range(nfout):
        T   = tvis[i]
        om  = 2.0 * pi / T
        tg  = delta / grvel[i] if grvel[i] > 0 else 0.0
        phi = phgr[i]

        c_ref = float(ref_interp(T))
        if c_ref <= 0:
            phgrc[i] = 0.0
            continue

        # number of cycles: n = round( (ω·Δ/c_ref - ω·tg - φ) / (2π) )
        total_phase_ref = om * delta / c_ref
        n = round((total_phase_ref - om * tg - phi) / (2.0 * pi))

        total_phase = om * tg + phi + n * 2.0 * pi
        if abs(total_phase) > 1e-10:
            phgrc[i] = om * delta / total_phase
        else:
            phgrc[i] = 0.0

    return phgrc


# ---------------------------------------------------------------------------
# Phase-match filter helpers  (aftanipg only)
# ---------------------------------------------------------------------------

def _pred_cur(delta: float, om_mid: float,
              npred: int, pred: np.ndarray) -> Tuple[float, float]:
    """
    Evaluate the prediction curve at om_mid.
    pred shape: (npred, 2)  → columns [period_s, group_vel_km_s]

    Returns (om0, tg0)  where tg0 = delta / U(om0)
    """
    periods = pred[:npred, 0]
    ug      = pred[:npred, 1]

    # angular frequency axis
    omegas  = 2.0 * np.pi / periods

    # sort ascending in omega
    idx  = np.argsort(omegas)
    omegas = omegas[idx]
    ug     = ug[idx]

    interp = interp1d(omegas, delta / ug, kind='linear',
                      bounds_error=False, fill_value='extrapolate')
    tg0 = float(interp(om_mid))
    return om_mid, tg0


def _build_phase_correction(om_mid: float, delta: float,
                             npred: int, pred: np.ndarray,
                             omb: float, ome: float,
                             dom: float, ns: int,
                             alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build phase correction operator e^{i φ_corr(ω)} for phase-match filter.

    φ_corr(ω) = ω · t_group(ω)    where t_group = Δ / U_pred(ω)

    Returns
    -------
    pha_cor   : complex array shape (ns,)
    ampdom    : amplitude taper array  shape (ns,)
    """
    periods = pred[:npred, 0]
    ug      = pred[:npred, 1]
    omegas  = 2.0 * np.pi / periods

    idx    = np.argsort(omegas)
    omegas = omegas[idx]
    ug     = ug[idx]
    tg     = delta / ug

    cs = CubicSpline(omegas, tg, extrapolate=True)

    freqs     = np.arange(ns) * dom
    pha_corr  = np.zeros(ns)
    ampdom    = np.zeros(ns)

    # amplitude taper between omb and ome
    for i, w in enumerate(freqs):
        if w < omb or w > ome:
            ampdom[i] = 0.0
        else:
            # smooth cosine edges
            ramp = alpha / 4.0
            if w < omb + ramp:
                ampdom[i] = 0.5 * (1.0 - np.cos(np.pi * (w - omb) / ramp))
            elif w > ome - ramp:
                ampdom[i] = 0.5 * (1.0 - np.cos(np.pi * (ome - w) / ramp))
            else:
                ampdom[i] = 1.0

    for i, w in enumerate(freqs):
        if ampdom[i] > 0:
            pha_corr[i] = float(cs(w)) * w   # ω · t_group(ω)

    pha_cor = np.exp(1j * pha_corr) * ampdom
    return pha_cor, ampdom


def _tgauss(fsnr: float, tg0: float, t0: float,
            dw: float, dt: float, ns: int,
            fmatch: float, env: np.ndarray) -> np.ndarray:
    """
    Cut/window the phase-matched impulse response `env` around
    the expected group arrival at t ≈ tg0 + t0.

    The window width is proportional to 1/dw * fmatch.

    Returns spref : windowed signal, complex128, length ns
    """
    spref     = np.zeros(ns, dtype=np.complex128)
    t_peak    = tg0 + t0        # expected peak time [s]
    half_win  = fmatch * (2.0 * np.pi / dw) if dw > 0 else ns * dt / 2

    i_peak = int(round(t_peak / dt))
    i_half = max(1, int(round(half_win / dt)))

    i0 = max(0, i_peak - i_half)
    i1 = min(ns, i_peak + i_half)

    # cosine taper inside window
    for i in range(i0, i1):
        x = np.pi * (i - i0) / max(1, i1 - i0 - 1)
        w = 0.5 * (1.0 - np.cos(x))
        spref[i] = env[i] * w

    return spref


# ---------------------------------------------------------------------------
# Main public functions
# ---------------------------------------------------------------------------

def aftanpg(piover4: float,
            n: int,
            sei: np.ndarray,
            t0: float,
            dt: float,
            delta: float,
            vmin: float,
            vmax: float,
            tmin: float,
            tmax: float,
            tresh: float  = 10.0,
            ffact: float  = 1.0,
            perc: float   = 50.0,
            npoints: int  = 5,
            taperl: float = 1.5,
            nfin: int     = 64,
            nphpr: int    = 0,
            phprper: Optional[np.ndarray] = None,
            phprvel: Optional[np.ndarray] = None,
            ) -> Tuple:
    """
    Regular FTAN analysis with automatic jump correction.
    Faithful Python translation of aftanpg.f (Barmine, 2006).

    Parameters
    ----------
    piover4  : phase shift factor π/4·piover4.
               Use +1.0 for seismograms, -1.0 for cross-correlations.
    n        : number of input samples
    sei      : seismogram array (float32 or float64, length ≥ n)
    t0       : begin time of record [s]  (SAC header b-value)
    dt       : sampling interval [s]
    delta    : epicentral distance [km]
    vmin     : minimum group velocity [km/s]
    vmax     : maximum group velocity [km/s]
    tmin     : minimum period [s]
    tmax     : maximum period [s]
    tresh    : jump-detection threshold (default 10)
    ffact    : Gaussian filter width factor (default 1.0)
    perc     : minimum % of frequency range required for final output
    npoints  : maximum jump size in frequency bins
    taperl   : left-taper length = taperl * tmax [s]
    nfin     : number of log-spaced frequencies (≤ 100)
    nphpr    : length of phase-velocity reference arrays
               (0 → output raw phase instead of phase velocity)
    phprper  : reference phase velocity periods [s], shape (nphpr,)
    phprvel  : reference phase velocities [km/s], shape (nphpr,)

    Returns
    -------
    nfout1   : number of frequencies in preliminary result
    arr1     : preliminary results, shape (8, nfout1)
                 [0] central period [s]
                 [1] observed period [s]
                 [2] group velocity [km/s]  (or phase [rad] if nphpr=0)
                 [3] phase velocity [km/s]
                 [4] amplitude [dB]
                 [5] discrimination function
                 [6] signal/noise ratio [dB]
                 [7] max half-width [s]
    nfout2   : number of frequencies in final (jump-corrected) result
    arr2     : final results, shape (7, nfout2)
    tamp     : start time of the amplitude map [s]
    amp      : FTAN amplitude map [dB], shape (nfout1, ncol)
    ierr     : 0=OK, 1=problems, 2=no final result
    """
    pi   = np.pi
    nf   = min(nfin, 100)
    ierr = 0

    # --- filter width (adaptive with distance) ---
    alpha = ffact * 20.0 * np.sqrt(delta / 1000.0)

    # --- taper lengths ---
    ntapb = max(1, round(taperl * tmax / dt))
    ntape = max(1, round(tmax / dt))

    # --- frequency (angular) range ---
    omb = 2.0 * pi / tmax
    ome = 2.0 * pi / tmin

    # --- time window ---
    nb = max(2, round((delta / vmax - t0) / dt)) - 1   # 0-based
    ne = min(n, round((delta / vmin - t0) / dt))        # 0-based exclusive
    tamp = nb * dt + t0

    nrow = nf
    ncol = ne - nb

    # --- taper seismogram ---
    s, ns = _taper_v2(nb, ne, n, sei.astype(np.float32), ntapb, ntape)

    # --- log-spaced angular frequencies ---
    step = (np.log(omb) - np.log(ome)) / (nf - 1)
    om   = np.array([np.exp(np.log(ome) + k * step) for k in range(nf)])
    per  = 2.0 * pi / om
    dom  = 2.0 * pi / (ns * dt)

    # --- Forward FFT of tapered seismogram ---
    sf = np.fft.fft(s)

    # --- FTAN amplitude map ---
    amp_raw  = np.full((ncol + 2, nf), 40.0)   # dB
    ampo_raw = np.zeros((ncol + 2, nf))         # linear
    pha_map  = np.zeros((ncol + 2, nf))

    ntall = ncol + 2

    for k in range(nf):
        fils = _ftfilt(alpha, om[k], dom, ns, sf)

        # one-sided spectrum for analytic signal (Hilbert)
        fils[ns // 2 + 1:] = 0.0
        fils[0]       = fils[0].real * 0.5 + 0.0j
        fils[ns // 2] = fils[ns // 2].real + 0.0j

        # IFFT to get analytic signal
        tmp = np.fft.ifft(fils) * ns   # FORTRAN uses unnormalised IFFT

        # extract window nb-1 … ne+1
        i0 = max(0, nb - 1)
        i1 = min(ns, ne + 1)
        chunk = tmp[i0:i1]
        jlen  = min(len(chunk), ntall)

        pha_map[:jlen, k]  = np.angle(chunk[:jlen])
        amp_lin            = np.abs(chunk[:jlen])
        ampo_raw[:jlen, k] = amp_lin
        with np.errstate(divide='ignore', invalid='ignore'):
            amp_db = 20.0 * np.log10(np.where(amp_lin > 0, amp_lin, 1e-30))
        amp_raw[:jlen, k]  = amp_db

    # --- normalise to 100 dB with 40 dB floor ---
    amax = amp_raw.max()
    amp_raw = amp_raw + 100.0 - amax
    amp_raw = np.maximum(amp_raw, 40.0)

    # --- extract ridge parameters for each frequency ---
    grvel  = np.zeros(nf)
    tvis   = np.zeros(nf)
    ampgr  = np.zeros(nf)
    snr    = np.zeros(nf)
    wdth   = np.zeros(nf)
    phgr   = np.zeros(nf)
    ftrig  = np.zeros(nf)

    # store ALL local maxima for jump correction
    ind_all  = []   # list of (k, j) 0-based
    ipar_all = []   # list of (tim, tvis, amp, snr, wdth, ph)

    for k in range(nf):
        col_amp  = amp_raw[:ntall, k]
        col_ampo = ampo_raw[:ntall, k]
        col_pha  = pha_map[:ntall, k]

        # find local maxima
        lm_idx = []
        for j in range(1, ntall - 1):
            if col_amp[j] > col_amp[j - 1] and col_amp[j] > col_amp[j + 1]:
                lm_idx.append(j)

        if not lm_idx:
            # fallback: use edge with largest amplitude
            j0 = ntall - 2 if col_ampo[1] <= col_ampo[ntall - 2] else 1
            lm_idx = [j0]

        # compute parameters for each local maximum
        best_ia  = lm_idx[0]
        best_amp = -1e30
        k_offset = len(ind_all)

        for j in lm_idx:
            t_off, dph, tm, ph = _fmax(
                col_amp[j - 1], col_amp[j], col_amp[j + 1],
                col_pha[j - 1], col_pha[j], col_pha[j + 1],
                om[k], dt, piover4)

            tim_j   = (nb + j - 2 + t_off) * dt   # group arrival time
            tvis_j  = 2.0 * pi * dt / dph if abs(dph) > 1e-12 else per[k]

            # SNR
            lm_val  = col_ampo[:j + 1].min()
            rm_val  = col_ampo[j:].min()
            if lm_val > 0 and rm_val > 0 and col_ampo[j] > 0:
                snr_j = 20.0 * np.log10(col_ampo[j] / np.sqrt(lm_val * rm_val))
            else:
                snr_j = 0.0
            lm_idx_val = int(np.argmin(col_ampo[:j + 1]))
            rm_idx_val = j + int(np.argmin(col_ampo[j:]))
            wdth_j     = 0.5 * dt * (abs(j - lm_idx_val) + abs(j - rm_idx_val))

            ind_all.append((k, j))
            ipar_all.append((tim_j, tvis_j, tm, snr_j, wdth_j, ph))

            if tm > best_amp:
                best_amp = tm
                best_ia  = len(ipar_all) - 1

        # best ridge point
        tim_best, tvis_best, amp_best, snr_best, wdth_best, ph_best = ipar_all[best_ia]
        grvel[k] = delta / (tim_best + t0) if (tim_best + t0) > 0 else 0.0
        tvis[k]  = tvis_best
        ampgr[k] = amp_best
        snr[k]   = snr_best
        wdth[k]  = wdth_best
        phgr[k]  = ph_best

    nfout1 = nf
    ind_all_arr  = ind_all
    ipar_all_arr = ipar_all

    # -----------------------------------------------------------------------
    # Jump correction
    # -----------------------------------------------------------------------
    trig1, ftrig, ierr_t = _trigger(grvel, om, nf, tresh)

    grvel_t = grvel.copy()
    tvis_t  = tvis.copy()
    ampgr_t = ampgr.copy()
    phgr_t  = phgr.copy()
    snr_t   = snr.copy()
    wdth_t  = wdth.copy()

    if ierr_t != 0:
        # find jump positions
        ijmp = [i for i in range(nf - 1)
                if abs(trig1[i + 1] - trig1[i]) > 1.5]

        # only attempt correction for short jumps
        ii_corr = [i for i in range(len(ijmp) - 1)
                   if ijmp[i + 1] - ijmp[i] <= npoints]

        for ki in ii_corr:
            kk     = ii_corr[ki] if ki < len(ii_corr) else ii_corr[-1]
            kk     = ki
            istrt  = ijmp[kk]
            ibeg   = istrt + 1
            iend   = ijmp[kk + 1] if kk + 1 < len(ijmp) else nf - 1

            grvel1 = grvel_t.copy()
            tvis1  = tvis_t.copy()
            ampgr1 = ampgr_t.copy()
            phgr1  = phgr_t.copy()
            snr1   = snr_t.copy()
            wdth1  = wdth_t.copy()

            for k in range(ibeg, iend + 1):
                best_j   = -1
                min_dv   = 1e10
                for idx2, (ki2, j2) in enumerate(ind_all_arr):
                    if ki2 == k:
                        g2 = delta / (ipar_all_arr[idx2][0] + t0) if (ipar_all_arr[idx2][0] + t0) > 0 else 0.0
                        dv = abs(g2 - grvel1[k - 1])
                        if dv < min_dv:
                            min_dv = dv
                            best_j = idx2

                if best_j >= 0:
                    t2, tv2, a2, s2, w2, p2 = ipar_all_arr[best_j]
                    grvel1[k] = delta / (t2 + t0) if (t2 + t0) > 0 else 0.0
                    tvis1[k]  = tv2
                    ampgr1[k] = a2
                    phgr1[k]  = p2
                    snr1[k]   = s2
                    wdth1[k]  = w2

            trig2, _, ierr2 = _trigger(grvel1, om, nf, tresh)
            still_bad = any(abs(trig2[k]) >= 0.5 for k in range(istrt, iend + 2) if k < nf)

            if not still_bad:
                grvel_t = grvel1.copy()
                tvis_t  = tvis1.copy()
                ampgr_t = ampgr1.copy()
                phgr_t  = phgr1.copy()
                snr_t   = snr1.copy()
                wdth_t  = wdth1.copy()

        grvel = grvel_t.copy()
        tvis  = tvis_t.copy()
        ampgr = ampgr_t.copy()
        phgr  = phgr_t.copy()
        snr   = snr_t.copy()
        wdth  = wdth_t.copy()

    # -----------------------------------------------------------------------
    # Cut to longest continuous segment
    # -----------------------------------------------------------------------
    trig1f, ftrig, ierr_f = _trigger(grvel, om, nf, tresh)

    grvel1 = grvel.copy()
    tvis1  = tvis.copy()
    ampgr1 = ampgr.copy()
    phgr1  = phgr.copy()
    snr1   = snr.copy()
    wdth1  = wdth.copy()
    per2   = per.copy()
    om1    = om.copy()

    if ierr_f != 0:
        indx = [0] + [i for i in range(nf) if abs(trig1f[i]) >= 0.5] + [nf - 1]
        indx = sorted(set(indx))

        best_len = 0
        best_pos = 0
        for i in range(len(indx) - 1):
            seg_len = indx[i + 1] - indx[i]
            if seg_len > best_len:
                best_len = seg_len
                best_pos = i

        ist = max(indx[best_pos], 0)
        ibe = min(indx[best_pos + 1], nf - 1)
        nfout2 = ibe - ist + 1

        if nfout2 < nf * perc / 100.0:
            nfout2 = 0
            ierr   = 2
        else:
            grvel1 = grvel[ist:ibe + 1]
            tvis1  = tvis[ist:ibe + 1]
            ampgr1 = ampgr[ist:ibe + 1]
            phgr1  = phgr[ist:ibe + 1]
            snr1   = snr[ist:ibe + 1]
            wdth1  = wdth[ist:ibe + 1]
            per2   = per[ist:ibe + 1]
            om1    = om[ist:ibe + 1]
    else:
        nfout2 = nf

    # -----------------------------------------------------------------------
    # Phase velocity conversion
    # -----------------------------------------------------------------------
    phgr_out  = phgr.copy()
    phgr1_out = phgr1.copy()

    if nfout1 > 0 and nphpr > 0 and phprper is not None and phprvel is not None:
        phgr_out = _phtovel(delta, nfout1, tvis, grvel, phgr,
                            np.asarray(phprper), np.asarray(phprvel))

    if nfout2 > 0 and nphpr > 0 and phprper is not None and phprvel is not None:
        phgr1_out = _phtovel(delta, nfout2, tvis1, grvel1, phgr1,
                             np.asarray(phprper), np.asarray(phprvel))

    # -----------------------------------------------------------------------
    # Fill output arrays
    # -----------------------------------------------------------------------
    arr1 = np.zeros((8, nfout1))
    for i in range(nfout1):
        arr1[0, i] = per[i]
        arr1[1, i] = tvis[i]
        arr1[2, i] = grvel[i]
        arr1[3, i] = phgr_out[i]
        arr1[4, i] = ampgr[i]
        arr1[5, i] = ftrig[i]
        arr1[6, i] = snr[i]
        arr1[7, i] = wdth[i]

    arr2 = np.zeros((7, max(1, nfout2)))
    if nfout2 > 0:
        for i in range(nfout2):
            arr2[0, i] = per2[i]
            arr2[1, i] = tvis1[i]
            arr2[2, i] = grvel1[i]
            arr2[3, i] = phgr1_out[i]
            arr2[4, i] = ampgr1[i]
            arr2[5, i] = snr1[i]
            arr2[6, i] = wdth1[i]

    # amplitude map (normalised, shape nf × ncol)
    amp_out = amp_raw[:ncol, :].T   # shape (nf, ncol)

    return nfout1, arr1, nfout2, arr2, tamp, amp_out, ierr


# ---------------------------------------------------------------------------

def aftanipg(piover4: float,
             n: int,
             sei: np.ndarray,
             t0: float,
             dt: float,
             delta: float,
             vmin: float,
             vmax: float,
             tmin: float,
             tmax: float,
             tresh: float  = 10.0,
             ffact: float  = 1.0,
             perc: float   = 50.0,
             npoints: int  = 5,
             taperl: float = 1.5,
             nfin: int     = 64,
             fsnr: float   = 1.0,
             fmatch: float = 1.0,
             npred: int    = 0,
             pred: Optional[np.ndarray] = None,
             nphpr: int    = 0,
             phprper: Optional[np.ndarray] = None,
             phprvel: Optional[np.ndarray] = None,
             ) -> Tuple:
    """
    Iterative FTAN with phase-match filter + jump correction.
    Faithful Python translation of aftanipg.f (Barmine, 2006).

    Additional parameters compared to aftanpg
    ------------------------------------------
    fsnr    : controls the Gaussian window width in the time domain
              after phase matching (default 1.0)
    fmatch  : factor for the length of the phase-matching window
    npred   : number of rows in the group-velocity prediction table
    pred    : prediction table, shape (npred, 2)
                  col 0 = period [s]
                  col 1 = group velocity [km/s]

    All other parameters are identical to aftanpg.

    The difference from aftanpg: before the narrow-band filtering step,
    the spectrum is corrected by exp(i·ω·t_group(ω)) so that the surface
    wave packet is collapsed to t≈0 in the synthetic domain, then a time
    window selects only that packet, and the inverse phase correction is
    applied. This acts as a powerful mode-isolation filter.
    """
    if npred == 0 or pred is None:
        # fall back to regular FTAN if no prediction supplied
        return aftanpg(piover4, n, sei, t0, dt, delta, vmin, vmax,
                       tmin, tmax, tresh, ffact, perc, npoints,
                       taperl, nfin, nphpr, phprper, phprvel)

    pi   = np.pi
    nf   = min(nfin, 100)
    ierr = 0

    alpha = ffact * 20.0 * np.sqrt(delta / 1000.0)
    ntapb = max(1, round(taperl * tmax / dt))
    ntape = max(1, round(tmax / dt))

    omb = 2.0 * pi / tmax
    ome = 2.0 * pi / tmin

    nb = max(2, round((delta / vmax - t0) / dt)) - 1
    ne = min(n, round((delta / vmin - t0) / dt))
    tamp = nb * dt + t0

    ncol = ne - nb
    nrow = nf

    s, ns = _taper_v2(nb, ne, n, sei.astype(np.float32), ntapb, ntape)

    step = (np.log(omb) - np.log(ome)) / (nf - 1)
    om   = np.array([np.exp(np.log(ome) + k * step) for k in range(nf)])
    per  = 2.0 * pi / om
    dom  = 2.0 * pi / (ns * dt)

    # --- Phase match filter ---
    pred_arr = np.asarray(pred)
    om_mid   = np.sqrt(omb * ome)
    om0, tg0 = _pred_cur(delta, om_mid, npred, pred_arr)

    sf_raw = np.fft.fft(s)
    sf     = sf_raw.copy()
    sf[0]         = sf[0].real * 0.5
    sf[ns // 2]   = sf[ns // 2].real

    # build phase correction
    pha_cor, ampdom = _build_phase_correction(
        om0, delta, npred, pred_arr, omb, ome, dom, ns, alpha)

    # apply phase correction to spectrum
    sf_corr = sf * pha_cor

    # IFFT to time domain → envelope
    env = np.fft.ifft(sf_corr) * 2.0

    # window the impulse response around expected group arrival
    dw     = abs(om[0] - om[-1])
    spref  = _tgauss(fsnr, tg0, t0, dw, dt, ns, fmatch, env)

    # back to spectrum
    sf_back = np.fft.fft(spref)

    # remove phase correction
    with np.errstate(invalid='ignore', divide='ignore'):
        sf_filtered = np.where(np.abs(pha_cor) > 1e-20,
                               sf_back / pha_cor, 0.0 + 0.0j)

    # now run the same narrow-band loop as aftanpg, but on sf_filtered
    ntall    = ncol + 2
    amp_raw  = np.full((ntall, nf), 40.0)
    ampo_raw = np.zeros((ntall, nf))
    pha_map  = np.zeros((ntall, nf))

    for k in range(nf):
        fils = _ftfilt(alpha, om[k], dom, ns, sf_filtered)
        fils[ns // 2 + 1:] = 0.0
        fils[0]       = fils[0].real * 0.5 + 0.0j
        fils[ns // 2] = fils[ns // 2].real + 0.0j

        tmp  = np.fft.ifft(fils) * ns
        i0   = max(0, nb - 1)
        i1   = min(ns, ne + 1)
        chunk = tmp[i0:i1]
        jlen  = min(len(chunk), ntall)

        pha_map[:jlen, k]  = np.angle(chunk[:jlen])
        amp_lin            = np.abs(chunk[:jlen])
        ampo_raw[:jlen, k] = amp_lin
        with np.errstate(divide='ignore', invalid='ignore'):
            amp_db = 20.0 * np.log10(np.where(amp_lin > 0, amp_lin, 1e-30))
        amp_raw[:jlen, k] = amp_db

    amax    = amp_raw.max()
    amp_raw = amp_raw + 100.0 - amax
    amp_raw = np.maximum(amp_raw, 40.0)

    # --- ridge extraction (identical to aftanpg) ---
    grvel     = np.zeros(nf)
    tvis      = np.zeros(nf)
    ampgr     = np.zeros(nf)
    snr       = np.zeros(nf)
    wdth      = np.zeros(nf)
    phgr      = np.zeros(nf)
    ftrig     = np.zeros(nf)
    ind_all   = []
    ipar_all  = []

    for k in range(nf):
        col_amp  = amp_raw[:ntall, k]
        col_ampo = ampo_raw[:ntall, k]
        col_pha  = pha_map[:ntall, k]

        lm_idx = [j for j in range(1, ntall - 1)
                  if col_amp[j] > col_amp[j - 1] and col_amp[j] > col_amp[j + 1]]
        if not lm_idx:
            lm_idx = [ntall - 2 if col_ampo[1] <= col_ampo[ntall - 2] else 1]

        best_ia  = 0
        best_amp_val = -1e30

        for j in lm_idx:
            t_off, dph, tm, ph = _fmax(
                col_amp[j - 1], col_amp[j], col_amp[j + 1],
                col_pha[j - 1], col_pha[j], col_pha[j + 1],
                om[k], dt, piover4)
            tim_j  = (nb + j - 2 + t_off) * dt
            tvis_j = 2.0 * pi * dt / dph if abs(dph) > 1e-12 else per[k]

            lm_val   = col_ampo[:j + 1].min()
            rm_val   = col_ampo[j:].min()
            snr_j    = 20.0 * np.log10(col_ampo[j] / np.sqrt(lm_val * rm_val + 1e-30))
            li       = int(np.argmin(col_ampo[:j + 1]))
            ri       = j + int(np.argmin(col_ampo[j:]))
            wdth_j   = 0.5 * dt * (abs(j - li) + abs(j - ri))

            ind_all.append((k, j))
            ipar_all.append((tim_j, tvis_j, tm, snr_j, wdth_j, ph))

            if tm > best_amp_val:
                best_amp_val = tm
                best_ia      = len(ipar_all) - 1

        t2, tv2, a2, s2, w2, p2 = ipar_all[best_ia]
        grvel[k] = delta / (t2 + t0) if (t2 + t0) > 0 else 0.0
        tvis[k]  = tv2
        ampgr[k] = a2
        snr[k]   = s2
        wdth[k]  = w2
        phgr[k]  = p2

    nfout1 = nf

    # --- jump correction (same as aftanpg) ---
    trig1, ftrig, ierr_t = _trigger(grvel, om, nf, tresh)
    grvel_t = grvel.copy(); tvis_t = tvis.copy()
    ampgr_t = ampgr.copy(); phgr_t = phgr.copy()
    snr_t   = snr.copy();   wdth_t = wdth.copy()

    if ierr_t != 0:
        ijmp = [i for i in range(nf - 1) if abs(trig1[i + 1] - trig1[i]) > 1.5]
        ii_corr = [i for i in range(len(ijmp) - 1) if ijmp[i + 1] - ijmp[i] <= npoints]

        for ki in range(len(ii_corr)):
            kk    = ii_corr[ki]
            istrt = ijmp[kk]
            ibeg  = istrt + 1
            iend  = ijmp[kk + 1] if kk + 1 < len(ijmp) else nf - 1

            grvel1 = grvel_t.copy(); tvis1 = tvis_t.copy()
            ampgr1 = ampgr_t.copy(); phgr1 = phgr_t.copy()
            snr1   = snr_t.copy();   wdth1 = wdth_t.copy()

            for k in range(ibeg, iend + 1):
                best_j = -1; min_dv = 1e10
                for idx2, (ki2, j2) in enumerate(ind_all):
                    if ki2 == k:
                        g2 = delta / (ipar_all[idx2][0] + t0)
                        dv = abs(g2 - grvel1[k - 1])
                        if dv < min_dv:
                            min_dv = dv; best_j = idx2
                if best_j >= 0:
                    t2, tv2, a2, s2, w2, p2 = ipar_all[best_j]
                    grvel1[k] = delta / (t2 + t0)
                    tvis1[k] = tv2; ampgr1[k] = a2
                    phgr1[k] = p2; snr1[k] = s2; wdth1[k] = w2

            trig2, _, _ = _trigger(grvel1, om, nf, tresh)
            still_bad = any(abs(trig2[k]) >= 0.5 for k in range(istrt, iend + 2) if k < nf)
            if not still_bad:
                grvel_t = grvel1.copy(); tvis_t = tvis1.copy()
                ampgr_t = ampgr1.copy(); phgr_t = phgr1.copy()
                snr_t = snr1.copy(); wdth_t = wdth1.copy()

        grvel = grvel_t.copy(); tvis = tvis_t.copy()
        ampgr = ampgr_t.copy(); phgr = phgr_t.copy()
        snr = snr_t.copy(); wdth = wdth_t.copy()

    # --- longest continuous segment ---
    trig1f, ftrig, ierr_f = _trigger(grvel, om, nf, tresh)
    grvel1 = grvel.copy(); tvis1 = tvis.copy()
    ampgr1 = ampgr.copy(); phgr1 = phgr.copy()
    snr1   = snr.copy();   wdth1 = wdth.copy()
    per2   = per.copy();   om1   = om.copy()

    if ierr_f != 0:
        indx = sorted(set([0] + [i for i in range(nf) if abs(trig1f[i]) >= 0.5] + [nf - 1]))
        best_len = 0; best_pos = 0
        for i in range(len(indx) - 1):
            if indx[i + 1] - indx[i] > best_len:
                best_len = indx[i + 1] - indx[i]; best_pos = i

        ist = max(indx[best_pos], 0)
        ibe = min(indx[best_pos + 1], nf - 1)
        nfout2 = ibe - ist + 1

        if nfout2 < nf * perc / 100.0:
            nfout2 = 0; ierr = 2
        else:
            sl = slice(ist, ibe + 1)
            grvel1 = grvel[sl]; tvis1 = tvis[sl]; ampgr1 = ampgr[sl]
            phgr1  = phgr[sl];  snr1  = snr[sl];  wdth1  = wdth[sl]
            per2   = per[sl];   om1   = om[sl]
    else:
        nfout2 = nf

    # --- phase velocity ---
    phgr_out = phgr.copy(); phgr1_out = phgr1.copy()
    if nfout1 > 0 and nphpr > 0 and phprper is not None and phprvel is not None:
        phgr_out = _phtovel(delta, nfout1, tvis, grvel, phgr,
                            np.asarray(phprper), np.asarray(phprvel))
    if nfout2 > 0 and nphpr > 0 and phprper is not None and phprvel is not None:
        phgr1_out = _phtovel(delta, nfout2, tvis1, grvel1, phgr1,
                             np.asarray(phprper), np.asarray(phprvel))

    arr1 = np.zeros((8, nfout1))
    for i in range(nfout1):
        arr1[0, i] = per[i]; arr1[1, i] = tvis[i]
        arr1[2, i] = grvel[i]; arr1[3, i] = phgr_out[i]
        arr1[4, i] = ampgr[i]; arr1[5, i] = ftrig[i]
        arr1[6, i] = snr[i];   arr1[7, i] = wdth[i]

    arr2 = np.zeros((7, max(1, nfout2)))
    if nfout2 > 0:
        for i in range(nfout2):
            arr2[0, i] = per2[i]; arr2[1, i] = tvis1[i]
            arr2[2, i] = grvel1[i]; arr2[3, i] = phgr1_out[i]
            arr2[4, i] = ampgr1[i]; arr2[5, i] = snr1[i]
            arr2[6, i] = wdth1[i]

    amp_out = amp_raw[:ncol, :].T
    return nfout1, arr1, nfout2, arr2, tamp, amp_out, ierr


# ---------------------------------------------------------------------------
# Header extraction helpers
# ---------------------------------------------------------------------------

def _extract_geodetic_h5(tr) -> Tuple[float, float, float, float]:
    """
    Extract (dist_km, azim, bazim, t0) from an ObsPy Trace loaded from an
    H5 file written by crossstack.py / surfquakecore.

    The H5 header stores:
        tr.stats.geodetic = {
            'otime'   : float  (UTC timestamp of starttime),
            'geodetic': [dist_km, azim, bazim, 0.0],
            'event'   : [lat_i, lon_i, 0.0],
            'arrivals': []
        }

    t0 is defined as 0 for EGFs (the cross-correlation is symmetric around
    t=0, and starttime is fixed to 2000-01-01T00:00:00).

    Returns
    -------
    dist_km  : inter-station distance [km]
    azim     : azimuth station_i → station_j [deg]
    bazim    : back-azimuth [deg]
    t0       : begin time of the trace [s]  (= 0.0 for EGFs from crossstack)
    """
    geo = None

    # --- primary: tr.stats.geodetic (H5 from crossstack.py) ---
    if hasattr(tr.stats, 'geodetic'):
        raw = tr.stats.geodetic
        # may be a dict or an AttribDict
        if hasattr(raw, 'geodetic'):
            geo = list(raw.geodetic)          # [dist_km, azim, bazim, 0.0]
        elif isinstance(raw, (list, tuple, np.ndarray)) and len(raw) >= 3:
            geo = list(raw)
        elif hasattr(raw, '__getitem__'):
            try:
                geo = list(raw['geodetic'])
            except Exception:
                pass

    # --- secondary: tr.stats.mseed.geodetic (older crossstack format) ---
    if geo is None:
        try:
            geo = list(tr.stats.mseed['geodetic'])   # [dist, bazim, azim]
            # older format: [dist_m_or_km, bazim, azim] – no 4th element
            # distances were stored in metres in some versions
            if geo[0] > 20000:                        # almost certainly metres
                geo[0] = geo[0] * 1e-3
            # column order was [dist, bazim, azim] in the mseed sub-dict
            geo = [geo[0], geo[2], geo[1], 0.0]       # reorder to [dist, azim, bazim]
        except Exception:
            pass

    # --- tertiary: SAC header ---
    if geo is None:
        try:
            dist_km = float(tr.stats.sac.dist)
            azim    = float(getattr(tr.stats.sac, 'az',  0.0))
            bazim   = float(getattr(tr.stats.sac, 'baz', 0.0))
            geo = [dist_km, azim, bazim, 0.0]
        except Exception:
            pass

    if geo is None:
        raise ValueError(
            f"Cannot extract inter-station distance from trace '{tr.id}'.\n"
            "Expected one of:\n"
            "  • tr.stats.geodetic['geodetic'] = [dist_km, azim, bazim, 0.0]\n"
            "  • tr.stats.mseed['geodetic']    = [dist, bazim, azim]\n"
            "  • tr.stats.sac.dist"
        )

    dist_km = float(geo[0])
    azim    = float(geo[1])
    bazim   = float(geo[2])

    # t0: for EGFs produced by crossstack starttime is 2000-01-01 and b=0
    try:
        t0 = float(tr.stats.sac.b)
    except Exception:
        t0 = 0.0

    return dist_km, azim, bazim, t0


def _is_egf(tr) -> bool:
    """
    Heuristic: return True if the trace looks like a cross-correlation EGF
    (symmetric around t=0) rather than a causal seismogram.

    Criteria:
      • has tr.stats.geodetic or tr.stats.mseed.geodetic  (set by crossstack)
      • OR station name contains underscore (STA1_STA2 convention)
      • OR channel is two characters (ZZ, ZE, RR, TT …)
    """
    if hasattr(tr.stats, 'geodetic'):
        return True
    try:
        if 'geodetic' in tr.stats.mseed:
            return True
    except Exception:
        pass
    if '_' in tr.stats.station:
        return True
    if len(tr.stats.channel) == 2:
        return True
    return False


# ---------------------------------------------------------------------------
# EGF branch preparation:  causal / acausal / fold
# ---------------------------------------------------------------------------

def prepare_egf(data: np.ndarray,
                dt: float,
                branch: str = 'fold') -> Tuple[np.ndarray, float]:
    """
    Extract the desired branch from a symmetric EGF and return a
    **causal-only** array ready for AFTAN together with the correct t0.

    The EGF produced by crossstack.py has layout:

        index:    0  1  2  ...  ic-1  ic  ic+1  ...  n-2  n-1
        lag [s]:  -(ic)*dt  ...   0   ...       +(ic)*dt

    where  ic = (n-1)//2  is the zero-lag sample (centre).

    Parameters
    ----------
    data   : full symmetric EGF array, length n  (float32 or float64)
    dt     : sampling interval [s]
    branch : one of
             'causal'   – positive lags only  [ic … n-1]
             'acausal'  – negative lags, time-reversed to be causal
                          [ic … 0] reversed → [0 … ic]
             'fold'     – average of both branches (default, best SNR)
                          result length = ic+1

    Returns
    -------
    out_data : 1-D array of the prepared branch, length L = ic+1
               Always starts at lag 0 (sample index 0 = t=0).
    t0       : begin time to pass to AFTAN [s]
               = 0.0  because the returned array starts at zero lag.

    Notes
    -----
    After this function AFTAN receives a causal trace of length ic+1
    where sample k corresponds to lag  k*dt.  The group arrival at
    distance Δ with velocity U arrives at sample  k = Δ/(U*dt),
    so  t0 = 0  is correct.

    The 'fold' operation improves the SNR by √2 on average because
    noise on the two branches is uncorrelated while the signal is
    coherent (symmetric for a homogeneous medium).
    """
    n  = len(data)
    ic = (n - 1) // 2          # index of zero-lag sample

    causal   = data[ic:].copy().astype(np.float64)      # lags  0 … +ic*dt
    acausal  = data[ic::-1].copy().astype(np.float64)   # lags  0 … +ic*dt  (reversed)

    if branch == 'causal':
        out = causal
    elif branch == 'acausal':
        out = acausal
    elif branch == 'fold':
        # arithmetic mean – preserves amplitude scale
        L   = min(len(causal), len(acausal))
        out = 0.5 * (causal[:L] + acausal[:L])
    else:
        raise ValueError(f"branch must be 'causal', 'acausal', or 'fold'. Got '{branch}'.")

    t0 = 0.0   # the returned array starts at zero lag by construction
    return out.astype(np.float32), t0


# ---------------------------------------------------------------------------
# Convenience: load waveform with ObsPy and run AFTAN
# ---------------------------------------------------------------------------

def run_aftan(filepath: str,
              vmin: float   = 2.0,
              vmax: float   = 5.0,
              tmin: float   = 5.0,
              tmax: float   = 150.0,
              tresh: float  = 10.0,
              ffact: float  = 1.0,
              perc: float   = 50.0,
              npoints: int  = 5,
              taperl: float = 1.5,
              nfin: int     = 64,
              piover4: float = -1.0,
              pred: Optional[np.ndarray] = None,
              phprper: Optional[np.ndarray] = None,
              phprvel: Optional[np.ndarray] = None,
              use_pmf: bool  = False,
              trace_index: int = 0,
              branch: str   = 'fold',
              force_dist_km: Optional[float] = None,
              force_t0: Optional[float] = None,
              ) -> dict:
    """
    High-level wrapper: read SAC / MiniSEED / H5 (crossstack EGF) and run AFTAN.

    Supported input formats
    -----------------------
    H5  (written by crossstack.py / surfquakecore)
        Distance from tr.stats.geodetic['geodetic'][0] [km].
        The EGF is symmetric: sample 0 = most-negative lag,
        sample (n-1)//2 = zero lag, sample n-1 = most-positive lag.
        The desired branch is extracted with prepare_egf() before AFTAN.
    SAC
        Distance from tr.stats.sac.dist [km], t0 from tr.stats.sac.b.
        Assumed to be a causal seismogram; branch parameter is ignored.
    MiniSEED / other ObsPy-readable
        Distance must be supplied via force_dist_km.

    Parameters
    ----------
    filepath      : path to waveform file
    vmin, vmax    : group-velocity window [km/s]
    tmin, tmax    : period range [s]
    tresh         : jump-detection threshold (default 10)
    ffact         : Gaussian filter width factor (default 1)
    perc          : minimum % of freq range required for final output
    npoints       : max jump size in frequency bins
    taperl        : left taper length factor  (taperl * tmax) [s]
    nfin          : number of log-spaced frequencies (≤ 100)
    piover4       : phase correction: -1.0 for EGFs, +1.0 for seismograms
    pred          : group-velocity prediction table (N×2): [period_s, U_km_s]
    phprper       : reference phase-velocity periods [s]
    phprvel       : reference phase velocities [km/s]
    use_pmf       : use phase-match filter (aftanipg) — requires pred
    trace_index   : which trace to use when file has multiple (default 0)
    branch        : EGF branch to analyse (H5 files only):
                      'causal'  – positive lags  (wave travelling i→j)
                      'acausal' – negative lags, time-reversed (j→i)
                      'fold'    – average of both branches (default, best SNR)
                    Ignored for SAC / non-EGF files.
    force_dist_km : override inter-station distance [km]
    force_t0      : override t0 [s]

    Returns
    -------
    dict with keys
        'arr1'    : preliminary dispersion  (8 × nfout1)
        'arr2'    : final dispersion        (7 × nfout2)
        'amp'     : FTAN amplitude map [dB] (nf × ncol)
        'tamp'    : start time of amp map [s]
        'nfout1'  : periods in arr1
        'nfout2'  : periods in arr2
        'ierr'    : 0=OK, 1=warn, 2=no result
        'delta'   : distance [km]
        'dt'      : sampling interval [s]
        't0'      : begin time passed to AFTAN [s]
        'azim'    : azimuth [deg]   (NaN if unavailable)
        'bazim'   : back-azimuth [deg] (NaN if unavailable)
        'branch'  : branch actually used ('causal'/'acausal'/'fold'/'N/A')
        'trace'   : original ObsPy Trace (full, unmodified)
        'source'  : 'H5' | 'SAC' | 'manual' | 'other'
    """
    try:
        from obspy import read as obs_read
    except ImportError:
        raise ImportError("ObsPy is required: pip install obspy")

    st  = obs_read(filepath)
    tr  = st[trace_index]
    dt  = float(tr.stats.delta)
    azim  = float('nan')
    bazim = float('nan')
    source      = 'other'
    branch_used = 'N/A'

    # ------------------------------------------------------------------ #
    # Resolve distance, t0, and prepare the signal array                  #
    # ------------------------------------------------------------------ #

    if force_dist_km is not None:
        # --- fully manual override ---
        delta = float(force_dist_km)
        t0    = float(force_t0) if force_t0 is not None else 0.0
        sei   = tr.data.astype(np.float32)
        source = 'manual'

    elif _is_egf(tr):
        # ---- H5 / crossstack EGF ----------------------------------------
        delta, azim, bazim, _ = _extract_geodetic_h5(tr)
        source = 'H5'

        # Extract requested branch; t0=0 is returned by prepare_egf
        sei, t0 = prepare_egf(tr.data, dt, branch=branch)
        branch_used = branch

        if force_t0 is not None:
            t0 = float(force_t0)

        print(f"[AFTAN]  H5 EGF  |  dist={delta:.2f} km  "
              f"az={azim:.1f}°  baz={bazim:.1f}°  "
              f"branch='{branch_used}'  n_out={len(sei)}  t0={t0:.2f} s")

    else:
        # ---- SAC / causal seismogram ------------------------------------
        try:
            delta = float(tr.stats.sac.dist)
            t0    = float(tr.stats.sac.b)
            azim  = float(getattr(tr.stats.sac, 'az',  float('nan')))
            bazim = float(getattr(tr.stats.sac, 'baz', float('nan')))
            source = 'SAC'
        except Exception:
            raise ValueError(
                f"Cannot determine inter-station distance for '{filepath}'.\n"
                "Supply it via force_dist_km=<km> or use an H5 / SAC file."
            )
        sei = tr.data.astype(np.float32)

    n = len(sei)

    # ------------------------------------------------------------------ #
    # Run AFTAN                                                           #
    # ------------------------------------------------------------------ #
    common_kw = dict(
        piover4=piover4, n=n, sei=sei, t0=t0, dt=dt, delta=delta,
        vmin=vmin, vmax=vmax, tmin=tmin, tmax=tmax,
        tresh=tresh, ffact=ffact, perc=perc, npoints=npoints,
        taperl=taperl, nfin=nfin,
        nphpr=len(phprper) if phprper is not None else 0,
        phprper=phprper, phprvel=phprvel,
    )

    if use_pmf and pred is not None:
        nfout1, arr1, nfout2, arr2, tamp, amp, ierr = aftanipg(
            npred=len(pred), pred=pred, **common_kw)
    else:
        nfout1, arr1, nfout2, arr2, tamp, amp, ierr = aftanpg(**common_kw)

    return dict(nfout1=nfout1, arr1=arr1, nfout2=nfout2, arr2=arr2,
                tamp=tamp, amp=amp, ierr=ierr,
                delta=delta, dt=dt, t0=t0,
                azim=azim, bazim=bazim,
                branch=branch_used, trace=tr, source=source,
                sei=sei,
                phprper=phprper, phprvel=phprvel, pred=pred)


# ---------------------------------------------------------------------------

def run_aftan_batch(folder: str,
                   pattern: str = "*.H5",
                   **kwargs) -> list:
    """
    Run AFTAN on every file matching `pattern` inside `folder`.

    All keyword arguments are forwarded to run_aftan(), so you can pass
    branch='causal', vmin=2.0, etc.

    Returns a list of result dicts; files that fail are skipped with a
    warning.

    Example
    -------
        results = run_aftan_batch(
            "/data/stack/",
            pattern="*.H5",
            branch='fold',
            vmin=2.0, vmax=5.0, tmin=3.0, tmax=80.0,
        )
        for r in results:
            if r['ierr'] < 2:
                plot_ftan(r, show=False)
    """
    from pathlib import Path as _Path
    files   = sorted(_Path(folder).glob(pattern))
    results = []
    for fp in files:
        try:
            r = run_aftan(str(fp), **kwargs)
            r['filepath'] = str(fp)
            results.append(r)
            print(f"[OK]   {fp.name}  "
                  f"dist={r['delta']:.1f} km  "
                  f"branch={r['branch']}  ierr={r['ierr']}")
        except Exception as exc:
            print(f"[SKIP] {fp.name}: {exc}")
    return results


# ---------------------------------------------------------------------------
# Plotting utility
# ---------------------------------------------------------------------------

def plot_ftan(result: dict,
              show: bool  = True,
              title: Optional[str] = None,
              vmin_plot: Optional[float] = None,
              vmax_plot: Optional[float] = None,
              cmap: str = 'jet'):
    """
    Three-panel FTAN figure matching the standard seismological layout:

      LEFT   — Dispersion curves: Period [s] (x) vs Velocity [km/s] (y)
                group velocity (blue squares) + phase velocity if available
      CENTRE — FTAN amplitude map: Period [s] (x) vs Velocity [km/s] (y)
                with the automatic group-velocity ridge overlaid in white
      RIGHT  — EGF waveform (fold / causal / acausal branch):
                Amplitude (x) vs Lag time [s] (y), narrow panel

    Parameters
    ----------
    result    : dict returned by run_aftan()
    show      : call plt.show() when True
    title     : figure suptitle; auto-generated if None
    vmin_plot : override minimum velocity for both left & centre y-axes
    vmax_plot : override maximum velocity for both left & centre y-axes
    cmap      : colormap for the amplitude map (default 'jet')
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
    except ImportError:
        print("matplotlib not available – skipping plot")
        return

    # ------------------------------------------------------------------ #
    # Unpack result                                                        #
    # ------------------------------------------------------------------ #
    amp    = result['amp']          # shape (nf, ncol)  — rows=freq, cols=time
    tamp   = result['tamp']
    dt     = result['dt']
    delta  = result['delta']
    t0     = result['t0']
    arr1   = result['arr1']
    arr2   = result['arr2']
    nfout1 = result['nfout1']
    nfout2 = result['nfout2']
    source = result.get('source', '')
    branch = result.get('branch', 'fold')
    sei    = result.get('sei', None)      # prepared EGF branch waveform

    nf, ncol = amp.shape

    # time axis of amp map columns (lag = 0 … max_lag for EGFs, t0=0)
    times = tamp + np.arange(ncol) * dt   # [s], starts at tamp

    # convert each time column to group velocity  U = delta / lag
    lag = times + t0
    with np.errstate(divide='ignore', invalid='ignore'):
        vels = np.where(lag > 1e-3, delta / lag, np.nan)   # [km/s]

    # period axis: log-spaced frequencies were stored index 0=shortest period
    # recover from arr1 if available, else use index
    if nfout1 > 0:
        per_min = arr1[0, 0]
        per_max = arr1[0, nfout1 - 1]
    else:
        per_min, per_max = tamp, tamp + ncol * dt   # fallback

    # periods corresponding to each frequency index row
    # (rows go 0=high-freq/short-period … nf-1=low-freq/long-period
    #  because of log(ome)+k*step with step<0)
    per_axis = np.linspace(per_min, per_max, nf)

    # velocity limits for display
    v_lo = vmin_plot if vmin_plot is not None else float(np.nanmin(vels[vels > 0])) if np.any(vels > 0) else 2.0
    v_hi = vmax_plot if vmax_plot is not None else float(np.nanmax(vels[np.isfinite(vels)])) if np.any(np.isfinite(vels)) else 5.0
    # round nicely
    v_lo = np.floor(v_lo * 4) / 4
    v_hi = np.ceil(v_hi  * 4) / 4

    # ---- auto title ----
    if title is None:
        tr   = result.get('trace')
        name = tr.id if tr is not None else ''
        az   = result.get('azim', float('nan'))
        title = (f"{name}   Δ={delta:.1f} km"
                 + (f"  az={az:.1f}°" if not np.isnan(az) else "")
                 + (f"  branch={branch}" if branch not in ('N/A', '') else ""))

    # ------------------------------------------------------------------ #
    # Figure layout:  [disp | ftan_map | egf]  widths 3 : 3 : 1          #
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(16, 7))
    gs  = gridspec.GridSpec(1, 3, width_ratios=[3, 3, 1.2],
                            wspace=0.32, left=0.07, right=0.96,
                            top=0.88, bottom=0.11)

    ax_disp = fig.add_subplot(gs[0])
    ax_map  = fig.add_subplot(gs[1])
    ax_egf  = fig.add_subplot(gs[2])

    # ================================================================== #
    # LEFT panel — dispersion curves                                      #
    # ================================================================== #
    # pull reference arrays from result (None if no --ref was given)
    ref_phprper = result.get('phprper')   # phase vel reference periods
    ref_phprvel = result.get('phprvel')   # phase vel reference values
    ref_pred    = result.get('pred')      # group vel prediction (N×2)

    # ---- reference curves first so data sits on top ----
    if ref_pred is not None:
        ax_disp.plot(ref_pred[:, 0], ref_pred[:, 1],
                     '--', color='limegreen', lw=1.6, alpha=0.85,
                     zorder=1, label='Ref. group vel.')
    if ref_phprper is not None and ref_phprvel is not None:
        ax_disp.plot(ref_phprper, ref_phprvel,
                     '--', color='darkorange', lw=1.6, alpha=0.85,
                     zorder=1, label='Ref. phase vel.')

    # ---- measured curves ----
    if nfout1 > 0:
        ax_disp.plot(arr1[0, :nfout1], arr1[2, :nfout1],
                     'o', color='steelblue', ms=3, alpha=0.4,
                     zorder=2, label='Prelim. group vel.')
    if nfout2 > 0:
        ax_disp.plot(arr2[0, :nfout2], arr2[2, :nfout2],
                     's-', color='navy', ms=5, lw=1.8,
                     zorder=3, label='Group vel.')
        # phase velocity — only plot if values look physical (> 0.5 km/s)
        pv = arr2[3, :nfout2]
        if np.any(pv > 0.5):
            ax_disp.plot(arr2[0, :nfout2], pv,
                         '^-', color='firebrick', ms=5, lw=1.8,
                         zorder=3, label='Phase vel.')

    ax_disp.set_xlim(per_min, per_max)
    ax_disp.set_ylim(v_lo, v_hi)
    ax_disp.set_xlabel('Period [s]', fontsize=11)
    ax_disp.set_ylabel('Velocity [km/s]', fontsize=11)
    ax_disp.set_title('Dispersion curves', fontsize=11)
    ax_disp.legend(fontsize=8, loc='lower right')
    ax_disp.grid(True, alpha=0.25, linestyle='--')

    # ================================================================== #
    # CENTRE panel — FTAN amplitude map (Period × Velocity)              #
    # ================================================================== #
    # Build a regular (period, velocity) grid by interpolating amp rows
    n_vel  = 300
    n_per  = nf
    vel_grid = np.linspace(v_lo, v_hi, n_vel)
    per_grid = per_axis                        # nf points

    # For each frequency (row), find which velocity columns are in range
    # and interpolate. amp[row, col] is at velocity vels[col].
    amp_img = np.full((n_vel, n_per), np.nan)

    for ki in range(nf):
        row_amp  = amp[ki, :]       # amplitudes across time/velocity
        row_vel  = vels             # velocity at each column
        # keep only finite values inside display range
        mask = np.isfinite(row_vel) & (row_vel >= v_lo) & (row_vel <= v_hi)
        if mask.sum() < 2:
            continue
        rv = row_vel[mask]
        ra = row_amp[mask]
        # sort by velocity (vels is decreasing in time)
        idx = np.argsort(rv)
        rv, ra = rv[idx], ra[idx]
        amp_img[:, ki] = np.interp(vel_grid, rv, ra,
                                   left=np.nan, right=np.nan)

    # normalise: 0 dB at max, clip at -5 dB (like model.png colour scale)
    amp_norm = amp_img - np.nanmax(amp_img)
    amp_norm = np.clip(amp_norm, -5, 0)

    pcm = ax_map.pcolormesh(per_grid, vel_grid, amp_norm,
                            cmap=cmap, vmin=-5, vmax=0,
                            shading='auto')

    # colourbar
    cbar = fig.colorbar(pcm, ax=ax_map, pad=0.02, shrink=0.85)
    cbar.set_label('Power [dB]', fontsize=9)
    cbar.set_ticks([-5, -4, -3, -2, -1, 0])

    # overlay automatic group-velocity ridge (white dots)
    if nfout2 > 0:
        per_ridge = arr2[0, :nfout2]
        vel_ridge = arr2[2, :nfout2]
        ax_map.plot(per_ridge, vel_ridge,
                    'w.', ms=6, lw=0, zorder=3, label='Auto ridge')

    # overlay reference group velocity (if provided)
    if ref_pred is not None:
        rp = ref_pred[:, 0]
        rv = ref_pred[:, 1]
        mask_ref = (rp >= per_min) & (rp <= per_max) & (rv >= v_lo) & (rv <= v_hi)
        if mask_ref.sum() > 1:
            ax_map.plot(rp[mask_ref], rv[mask_ref],
                        '--', color='limegreen', lw=1.8, alpha=0.9,
                        zorder=4, label='Ref. group vel.')

    ax_map.set_xlim(per_min, per_max)
    ax_map.set_ylim(v_lo, v_hi)
    ax_map.set_xlabel('Period [s]', fontsize=11)
    ax_map.set_ylabel('Velocity [km/s]', fontsize=11)
    ax_map.set_title('Group Velocity', fontsize=11)

    # ================================================================== #
    # RIGHT panel — EGF waveform                                         #
    # ================================================================== #
    if sei is not None and len(sei) > 0:
        n_sei   = len(sei)
        t_sei   = np.arange(n_sei) * dt          # lag times [s]

        # clip to a sensible lag window: 0 … delta/vmin + some margin
        t_max_show = delta / max(v_lo, 0.5) * 1.1
        mask_t = t_sei <= t_max_show

        # normalise waveform to ±1 for display
        wf = sei[mask_t].astype(float)
        peak = np.max(np.abs(wf))
        if peak > 0:
            wf /= peak

        t_show = t_sei[mask_t]

        # plot: x=amplitude, y=time (so waveform reads bottom→top)
        ax_egf.plot(wf, t_show, color='steelblue', lw=0.8)
        ax_egf.fill_betweenx(t_show, 0, wf,
                             where=wf >= 0, color='steelblue', alpha=0.35)
        ax_egf.fill_betweenx(t_show, 0, wf,
                             where=wf <  0, color='tomato',    alpha=0.35)

        # mark expected arrival window
        if v_lo > 0 and v_hi > 0:
            t_fast = delta / v_hi
            t_slow = delta / v_lo
            ax_egf.axhspan(t_fast, t_slow,
                           color='gold', alpha=0.12, label='vel. window')

        ax_egf.set_ylim(0, t_max_show)
        ax_egf.set_xlim(-1.5, 1.5)
        ax_egf.set_xlabel('Amplitude\n(norm.)', fontsize=9)
        ax_egf.set_ylabel('Time [s]', fontsize=10)
        ax_egf.set_title('EGF', fontsize=11)
        ax_egf.axvline(0, color='k', lw=0.5, alpha=0.4)
        ax_egf.yaxis.set_label_position('right')
        ax_egf.yaxis.tick_right()
        ax_egf.tick_params(axis='y', labelsize=8)
        ax_egf.tick_params(axis='x', labelsize=7)
        lbl = {'fold': 'fold (avg)', 'causal': 'causal', 'acausal': 'acausal'}
        ax_egf.set_title(f"EGF\n({lbl.get(branch, branch)})", fontsize=10)
    else:
        ax_egf.text(0.5, 0.5, 'No EGF\ndata',
                    ha='center', va='center', transform=ax_egf.transAxes,
                    fontsize=10, color='gray')
        ax_egf.set_title('EGF', fontsize=11)

    # ------------------------------------------------------------------ #
    fig.suptitle(title, fontsize=11, y=0.97)
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("aftan.py loaded.")
    print("Public API:")
    print("  aftanpg()         – regular FTAN + jump correction")
    print("  aftanipg()        – iterative FTAN with phase-match filter")
    print("  prepare_egf()     – extract causal / acausal / fold branch")
    print("  run_aftan()       – high-level wrapper (SAC / H5 / MiniSEED)")
    print("  run_aftan_batch() – process all H5 files in a folder")
    print("  plot_ftan()       – amplitude map + dispersion curves")


