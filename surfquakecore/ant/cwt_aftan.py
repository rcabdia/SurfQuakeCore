"""
cwt_ftan.py  —  CWT-based Frequency-Time Analysis  (surfquakecore module)
==========================================================================
Standalone CLI tool implementing the full pipeline from the ISP
FrequencyTimeFrame GUI, translated to a batch command-line tool:

  1.  EGF branch selection          (causal / acausal / fold)
  2.  Phase-match filter            (faithful port of noise_processing.phase_matched_filter)
  3.  Complex Morlet CWT            (faithful port of ConvolveWaveletScipy)
  4.  Group-velocity ridge extraction
  5.  Phase-velocity estimation     (faithful port of FrequencyTimeFrame.phase_velocity)
  6.  Three-panel plot + .disp output

Mathematical background
-----------------------
CWT atom (Complex Morlet):
    psi(t; fk) = (pi*sigma_k^2)^(-1/4) * exp(i*2*pi*fk*t) * exp(-t^2/2*sigma_k^2)
    sigma_k = n_cycles[k] / (2*pi*fk)     n_cycles varies linearly wmin->wmax

Group velocity from ridge:
    U(T) = delta / t_ridge(T)

Phase velocity (from GUI phase_velocity(), line 746):
    c(T) = delta * omega_inst / (phi + omega_inst*t0 - pi/4 - k*2*pi - pi/4)
    where k in {-5..+4} are integer cycle branches (2pi ambiguity)

Phase-match filter (from noise_processing.phase_matched_filter):
    Forward:  FFT -> * exp(+i*2*pi*f*delta/c_ref(f)) * exp(-i*f*2*pi*tshift)
              -> time-domain -> Gaussian window sigma=filter_param/dt
    Inverse:  FFT -> * exp(-i*2*pi*f*delta/c_ref(f)) * exp(+i*f*2*pi*tshift)
              -> real part

surfquake CLI registration
--------------------------
In surfquakecore/cli/main.py add:

    "cwt_ftan": _CliActions(
        name="cwt_ftan", run=_cwt_ftan_cli,
        description="CWT-based FTAN: group and phase velocity dispersion from EGFs"),

And place this file at:  surfquakecore/ant/cwt_ftan.py

Dependencies:  numpy, scipy, obspy, matplotlib  (no PyQt5 / ISP required)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Optional, Tuple

import numpy as np
import scipy.signal
import scipy.interpolate
from obspy import read as obs_read, Trace


# ===========================================================================
# 1.  Phase-match filter
#     Faithful port of noise_processing.phase_matched_filter()
# ===========================================================================

def phase_matched_filter(data: np.ndarray,
                         dt: float,
                         dist_km: float,
                         ref_periods: np.ndarray,
                         ref_phase_vel: np.ndarray,
                         filter_parameter: float = 2.0) -> np.ndarray:
    """
    Apply phase-match filter to isolate the fundamental surface-wave mode.

    Exact port of noise_processing.phase_matched_filter().

    Forward pass (collapse wave packet to t = tshift):
        1. FFT(signal)
        2. * exp(+i*2pi*f*delta/c_ref(f))   phase correction
        3. * exp(-i*2pi*f*tshift)            time shift to centre
        4. IFFT -> time domain
        5. * Gaussian(sigma = filter_parameter/dt)

    Inverse pass (restore dispersion):
        6. FFT(windowed)
        7. * exp(-i*2pi*f*delta/c_ref(f))   undo phase correction
        8. * exp(+i*2pi*f*tshift)            undo time shift
        9. IFFT -> real part

    Parameters
    ----------
    data             : 1-D float array (causal branch, lag starts at 0)
    dt               : sampling interval [s]
    dist_km          : inter-station distance [km]
    ref_periods      : reference phase velocity periods [s]
    ref_phase_vel    : reference phase velocities [km/s]
    filter_parameter : Gaussian window half-width [s]

    Returns
    -------
    filtered : 1-D float32 array, same length as data
    """
    n      = len(data)
    tshift = (n * dt) / 2.0

    # build interpolator
    ref_f    = 1.0 / ref_periods[::-1]
    ref_c    = ref_phase_vel[::-1]
    c_interp = scipy.interpolate.interp1d(
        ref_f, ref_c, kind='linear', bounds_error=False,
        fill_value=(ref_c[0], ref_c[-1]))

    fft_bins = np.fft.rfftfreq(n, d=dt)
    c_ref    = c_interp(fft_bins)

    with np.errstate(divide='ignore', invalid='ignore'):
        dist_over_c = np.where(c_ref > 0, dist_km / c_ref, 0.0)

    # forward
    fft_sig   = np.fft.rfft(data)
    ph_corr   = np.exp( 1j * 2.0 * np.pi * fft_bins * dist_over_c)
    t_shift   = np.exp(-1j * tshift * 2.0 * np.pi * fft_bins)
    collapsed = np.fft.irfft(fft_sig * ph_corr * t_shift, n=n)

    # Gaussian window
    sigma   = filter_parameter / dt
    idx     = np.arange(n)
    gauss   = np.exp(-0.25 * (idx - n / 2.0) ** 2 / sigma ** 2)
    windowed = collapsed * gauss
    windowed[np.abs(windowed) < 1e-16] = 0.0

    # inverse
    ph_uncorr  = np.exp(-1j * 2.0 * np.pi * fft_bins * dist_over_c)
    t_unshift  = np.exp( 1j * tshift * 2.0 * np.pi * fft_bins)
    restored   = np.real(np.fft.irfft(np.fft.rfft(windowed) * ph_uncorr * t_unshift, n=n))

    return restored.astype(np.float32)


# ===========================================================================
# 2.  EGF branch preparation
#     Faithful port of FrequencyTimeFrame.plot_seismogram() lines 393-433
# ===========================================================================

def prepare_branch(data: np.ndarray, branch: str = 'fold') -> np.ndarray:
    """
    Extract the requested branch from a symmetric EGF.

    Centre index  c:
        even n -> c = ceil(n/2) + 1
        odd  n -> c = ceil((n+1)/2)

    Causal  -> data[0:c] flipped   (index 0 = zero lag)
    Acausal -> data[c:]             (index 0 = zero lag)
    Fold    -> average of both, trimmed to same length
    """
    n = len(data)
    c = int(np.ceil(n / 2.0) + 1) if (n % 2) == 0 else int(np.ceil((n + 1) / 2.0))

    causal_flip = np.flip(data[:c]).astype(np.float64)
    acausal     = data[c:].astype(np.float64)

    if branch == 'causal':
        return causal_flip.astype(np.float32)
    elif branch == 'acausal':
        return acausal.astype(np.float32)
    elif branch == 'fold':
        N = min(len(causal_flip), len(acausal))
        return ((causal_flip[:N] + acausal[:N]) / 2.0).astype(np.float32)
    else:
        raise ValueError(f"branch must be 'causal', 'acausal', or 'fold'. Got '{branch}'.")


# ===========================================================================
# 3.  Complex Morlet CWT
#     Faithful port of ConvolveWaveletScipy / ConvolveWaveletBase
# ===========================================================================

class MorletCWT:
    """
    Complex Morlet CWT via overlap-add FFT convolution.
    Port of ConvolveWaveletScipy without ISP/PyQt5 dependencies.
    """

    def __init__(self, data: np.ndarray, fs: float,
                 fmin: float, fmax: float,
                 nf: int = 64, wmin: float = 6.0, wmax: float = 12.0,
                 tt: Optional[float] = None):

        self._data     = data.astype(np.float64)
        self._npts     = len(data)
        self._fs       = fs
        self._fmin     = fmin
        self._fmax     = fmax
        self._nf       = nf
        self._wmin     = wmin
        self._wmax     = wmax
        self._tt       = tt if tt is not None else 1.0 / fmin

        self._frex     = np.logspace(np.log10(fmin), np.log10(fmax), nf)
        self._n_cycles = np.linspace(wmin, wmax, nf)

        dt = 1.0 / fs
        self._wtime = np.arange(-self._tt, self._tt + dt, dt)

    def _atom(self, freq: float, n_cyc: float) -> np.ndarray:
        """Conjugated, unit-energy Complex Morlet atom."""
        s   = n_cyc / (2.0 * np.pi * freq)
        A   = (np.pi * s ** 2) ** (-0.25)
        cmw = A * np.exp(1j * 2.0 * np.pi * freq * self._wtime) \
                * np.exp(-0.5 * (self._wtime / s) ** 2)
        return cmw.conjugate()

    def compute_tf(self) -> np.ndarray:
        """
        Compute complex CWT matrix via oaconvolve.
        Returns complex64 array shape (nf, npts).
        """
        tf = np.zeros((self._nf, self._npts), dtype=np.complex64)
        for k, (fk, nc) in enumerate(zip(self._frex, self._n_cycles)):
            conv     = scipy.signal.oaconvolve(self._data, self._atom(fk, nc), mode='same')
            tf[k, :] = conv.astype(np.complex64)
        return tf

    def compute_phase_and_inst_freq(self, tf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Instantaneous phase and frequency from CWT.
        Port of ConvolveWaveletBase.phase() (wavelet.py lines 303-321).

        Returns
        -------
        phase       : float64 (nf, npts) unwrapped phase [rad]
        inst_freq_hz: float64 (nf, npts) instantaneous frequency [Hz]
                      matches GUI variable ins_freq_hz used in phase_velocity()
        """
        nf, npts = tf.shape
        phase    = np.unwrap(np.angle(tf), axis=0)

        freq_ax   = np.fft.fftfreq(npts, d=1.0 / self._fs)
        freq_tile = np.tile(freq_ax, (nf, 1))
        deriv     = np.fft.ifft(2j * np.pi * freq_tile * np.fft.fft(tf, axis=1), axis=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            inst_freq_raw = deriv / (2j * np.pi * tf)

        inst_freq_hz = (np.abs(inst_freq_raw) * self._fs) / (2.0 * np.pi)
        return phase, inst_freq_hz

    @property
    def frequencies(self) -> np.ndarray:
        return self._frex

    @property
    def periods(self) -> np.ndarray:
        return 1.0 / self._frex


# ===========================================================================
# 4.  (freq, time) -> (period, velocity) grid conversion
#     Mirrors GUI lines 472-493
# ===========================================================================

def build_velocity_grid(tf_abs2: np.ndarray,
                        phase: np.ndarray,
                        inst_freq_hz: np.ndarray,
                        freq_ax: np.ndarray,
                        t_arr: np.ndarray,
                        dist_km: float,
                        vmin: float, vmax: float,
                        n_vel: int = 300,
                        min_db: float = -15.0
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                   np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert (frequency, time) CWT output to uniform (period, velocity) grid.

    Parameters
    ----------
    tf_abs2      : (nf, npts-1)  power |CWT|^2  (first column dropped as GUI does)
    phase        : (nf, npts-1)  unwrapped phase [rad]
    inst_freq_hz : (nf, npts-1)  instantaneous frequency [Hz]
    freq_ax      : (nf,)         log-spaced frequency axis [Hz]
    t_arr        : (npts,)       lag-time axis [s] starting at 0
    dist_km      : inter-station distance [km]
    vmin, vmax   : velocity window [km/s]
    n_vel        : uniform velocity nodes
    min_db       : dB floor

    Returns
    -------
    amp_img      : (n_vel, n_per) dB scalogram
    vel_axis     : (n_vel,)  ascending [km/s]
    per_axis     : (n_per,)  ascending [s]
    phase_grid   : (n_vel, n_per) phase [rad]
    ifreq_grid   : (n_vel, n_per) inst. frequency [Hz]
    t_grid       : (n_vel, n_per) group arrival time [s]
    """
    nf    = len(freq_ax)
    t_col = t_arr[1:]   # matches GUI [:, 1:] drop of first column


    # velocity at each time column — same for all frequencies
    with np.errstate(divide='ignore', invalid='ignore'):
        vel_raw = np.where(t_col > 0, dist_km / t_col, np.nan)  # shape (npts-1,)  ← 1-D

    # normalised dB
    sc_db = 10.0 * np.log10(tf_abs2 / (np.nanmax(tf_abs2) + 1e-30))
    sc_db = np.clip(sc_db, min_db, 0.0)

    vel_axis = np.linspace(vmin, vmax, n_vel)
    per_axis = 1.0 / freq_ax[::-1]
    n_per = nf

    amp_img = np.full((n_vel, n_per), np.nan)
    phase_grid = np.full((n_vel, n_per), np.nan)
    ifreq_grid = np.full((n_vel, n_per), np.nan)
    t_grid = np.full((n_vel, n_per), np.nan)

    for ki in range(nf):
        rv = vel_raw  # 1-D, same velocity axis for every frequency row
        mask = np.isfinite(rv) & (rv >= vmin) & (rv <= vmax)
        if mask.sum() < 2:
            continue
        idx = np.argsort(rv[mask])
        rv_s = rv[mask][idx]
        col_idx = n_per - 1 - ki

        for grid, src in [(amp_img, sc_db[ki, :]),
                          (phase_grid, phase[ki, :]),
                          (ifreq_grid, inst_freq_hz[ki, :]),
                          (t_grid, t_col)]:
            grid[:, col_idx] = np.interp(vel_axis, rv_s, src[mask][idx],
                                         left=np.nan, right=np.nan)

    return amp_img, vel_axis, per_axis, phase_grid, ifreq_grid, t_grid


# ===========================================================================
# 5a.  AFTAN-style jump correction
#      Ported from aftan.py _trigger() — polynomial fit + MAD outlier detection
# ===========================================================================

def _trigger(grvel: np.ndarray,
             per: np.ndarray,
             nf: int,
             tresh: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Detect jumps in a group-velocity curve using polynomial fitting + MAD.

    Ported directly from AFTAN's _trigger() subroutine (Barmine 2006).

    Algorithm
    ---------
    1. Fit a cubic polynomial to  log(U) vs log(T)
       (log-log space linearises the typical power-law dispersion shape)
    2. Compute residuals = log(U_measured) - log(U_fitted)
    3. MAD = median(|residuals|)  — robust scale estimator
    4. Normalised residual = residual / MAD
    5. Flag points where |norm_residual| >= tresh as jumps

    Parameters
    ----------
    grvel  : (nf,) group velocity array [km/s]
    per    : (nf,) period array [s]
    nf     : number of valid points to use
    tresh  : jump detection threshold (typical: 2.0 – 5.0 for CWT)
             Lower = more sensitive.  Original AFTAN default = 10.0
             but CWT ridges are noisier so 2–3 is more appropriate.

    Returns
    -------
    norm_res : (nf,) normalised residuals
    ftrig    : (nf,) |norm_res| / tresh  (fractional trigger)
    ierr     : 0 = no jumps detected,  1 = jumps detected
    """
    norm_res = np.zeros(nf)
    ftrig    = np.zeros(nf)

    if nf < 4:
        return norm_res, ftrig, 0

    log_per = np.log(per[:nf])
    log_u   = np.log(np.maximum(grvel[:nf], 1e-6))

    try:
        deg    = min(3, nf - 1)
        coeffs = np.polyfit(log_per, log_u, deg=deg)
        log_u_fit = np.polyval(coeffs, log_per)
        residuals = log_u - log_u_fit
    except Exception:
        return norm_res, ftrig, 0

    mad = np.median(np.abs(residuals))
    if mad < 1e-10:
        return norm_res, ftrig, 0

    norm_res[:nf] = residuals / mad
    ftrig[:nf]    = np.abs(norm_res[:nf]) / tresh
    ierr = 1 if np.any(np.abs(norm_res[:nf]) >= tresh) else 0

    return norm_res, ftrig, ierr


def apply_jump_correction(group_vel: np.ndarray,
                           per_axis: np.ndarray,
                           tresh: float = 3.0,
                           npoints: int = 5,
                           ) -> np.ndarray:
    """
    Apply AFTAN-style jump correction to a CWT group-velocity ridge.

    This replaces the simple median-filter spike removal with the
    physically motivated AFTAN approach:

    1. Run _trigger() to detect jumps (points whose normalised residual
       from a polynomial fit exceeds `tresh`).
    2. For short isolated jump segments (≤ npoints consecutive flagged
       periods), attempt to interpolate smoothly from the surrounding
       valid points using cubic spline.
    3. For longer segments, set to NaN (cannot reliably correct).

    Parameters
    ----------
    group_vel : (nf,) group velocity ridge [km/s]  — may contain NaN
    per_axis  : (nf,) period axis [s]
    tresh     : jump detection threshold.
                Default 3.0 for CWT (AFTAN uses 10.0 for its cleaner ridges).
                Lower = more aggressive correction.
    npoints   : maximum length of a correctable jump segment [periods].
                Segments longer than this are left as NaN.
                Default 5 (same as AFTAN default).

    Returns
    -------
    grvel_corrected : (nf,) corrected group velocity [km/s]
    """
    nf       = len(group_vel)
    grvel    = group_vel.copy()
    valid    = np.isfinite(grvel)

    if valid.sum() < 4:
        return grvel

    # work only on valid points
    valid_idx = np.where(valid)[0]
    nv        = len(valid_idx)
    grvel_v   = grvel[valid_idx]
    per_v     = per_axis[valid_idx]

    # --- detect jumps ---
    norm_res, ftrig, ierr = _trigger(grvel_v, per_v, nv, tresh)

    if ierr == 0:
        return grvel   # no jumps detected, nothing to do

    # --- find jump segments ---
    flagged = np.abs(norm_res) >= tresh     # boolean, length nv

    # group consecutive flagged points into segments
    segments = []
    i = 0
    while i < nv:
        if flagged[i]:
            j = i
            while j < nv and flagged[j]:
                j += 1
            segments.append((i, j - 1))    # inclusive start/end in valid_idx
            i = j
        else:
            i += 1

    # --- correct short segments by cubic spline interpolation ---
    grvel_corr = grvel_v.copy()

    for (seg_start, seg_end) in segments:
        seg_len = seg_end - seg_start + 1
        if seg_len > npoints:
            # too long to correct reliably — mark as NaN
            grvel[valid_idx[seg_start:seg_end + 1]] = np.nan
            continue

        # find anchor points: last clean point before and first after
        left_idx  = seg_start - 1
        right_idx = seg_end   + 1

        if left_idx < 0 or right_idx >= nv:
            # at the edge — cannot interpolate, mark NaN
            grvel[valid_idx[seg_start:seg_end + 1]] = np.nan
            continue

        # cubic spline through the two anchor points
        # (use up to 2 points on each side for a better constraint)
        left_pts  = max(0, left_idx - 1)
        right_pts = min(nv - 1, right_idx + 1)

        x_anchor = np.concatenate([
            per_v[left_pts:left_idx + 1],
            per_v[right_idx:right_pts + 1]
        ])
        y_anchor = np.concatenate([
            grvel_corr[left_pts:left_idx + 1],
            grvel_corr[right_idx:right_pts + 1]
        ])

        if len(x_anchor) < 2:
            grvel[valid_idx[seg_start:seg_end + 1]] = np.nan
            continue

        try:
            interp_fn = scipy.interpolate.interp1d(
                x_anchor, y_anchor,
                kind='linear', bounds_error=False,
                fill_value='extrapolate')
            corrected = interp_fn(per_v[seg_start:seg_end + 1])

            # only accept correction if it is physically plausible
            # (stays within ±20% of surrounding values)
            ref_level = 0.5 * (grvel_corr[left_idx] + grvel_corr[right_idx])
            if np.all(np.abs(corrected - ref_level) < 0.2 * ref_level):
                grvel[valid_idx[seg_start:seg_end + 1]] = corrected
            else:
                grvel[valid_idx[seg_start:seg_end + 1]] = np.nan
        except Exception:
            grvel[valid_idx[seg_start:seg_end + 1]] = np.nan

    n_flagged   = sum(seg_end - seg_start + 1 for seg_start, seg_end in segments)
    n_corrected = sum(
        seg_end - seg_start + 1 for seg_start, seg_end in segments
        if seg_end - seg_start + 1 <= npoints)
    print(f"            Jump correction: {len(segments)} segment(s) detected  "
          f"{n_flagged} periods flagged  "
          f"{n_corrected} corrected  tresh={tresh}")

    return grvel




def find_ridges(scalogram_db: np.ndarray,
                vel_axis: np.ndarray,
                height_db: float = -15.0,
                min_dist_kms: float = 0.2,
                num_ridges: int = 1,
                ref_group_vel: Optional[np.ndarray] = None,
                ref_periods: Optional[np.ndarray] = None,
                per_axis: Optional[np.ndarray] = None,
                ref_tolerance_kms: float = 0.5,
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract group-velocity ridges from the dB scalogram.

    Without reference
    -----------------
    At each period column, find all local maxima above height_db.
    Pick the one with the HIGHEST amplitude (strongest energy).

    With reference
    --------------
    Among all local maxima above height_db, pick the one whose
    velocity is CLOSEST to the reference group velocity at that period.

    Tolerance (ref_tolerance_kms)
    ------------------------------
    When a reference is provided, if the closest candidate peak is
    farther than ref_tolerance_kms [km/s] from the reference value,
    that period is marked as NaN (outlier rejected).

    This prevents the picker from latching onto spurious energy blobs
    that happen to be local maxima but are clearly unrelated to the
    mode tracked by the reference curve.

    Parameters
    ----------
    scalogram_db      : (n_vel, n_per)  power [dB]
    vel_axis          : (n_vel,)  velocity [km/s] ascending
    height_db         : minimum peak height [dB]
    min_dist_kms      : minimum separation between candidate peaks [km/s]
    num_ridges        : number of ridges to return
    ref_group_vel     : (N,) reference group velocities [km/s]  (optional)
    ref_periods       : (N,) reference periods [s]              (optional)
    per_axis          : (n_per,) period axis [s]                (required if ref given)
    ref_tolerance_kms : maximum allowed distance from reference [km/s].
                        Peaks farther than this are rejected as outliers.
                        Only used when a reference is provided.
                        Default: 0.5 km/s  (reasonable for most crust models).
                        Increase to 1.0 for noisy data or large model uncertainty.
                        Decrease to 0.2–0.3 for well-constrained paths.
    """
    n_vel, n_per = scalogram_db.shape
    dv = abs(vel_axis[1] - vel_axis[0]) if n_vel > 1 else 0.1
    min_dist_samp = max(1, int(min_dist_kms / dv))

    group_vel = np.full((num_ridges, n_per), np.nan)
    peak_db   = np.full((num_ridges, n_per), np.nan)

    # build reference interpolator if provided
    use_ref = (ref_group_vel is not None and
               ref_periods   is not None and
               per_axis      is not None)
    if use_ref:
        ref_interp = scipy.interpolate.interp1d(
            ref_periods, ref_group_vel,
            kind='linear', bounds_error=False,
            fill_value=(ref_group_vel[0], ref_group_vel[-1]))

    for j in range(n_per):
        col = scalogram_db[:, j]

        # find ALL local maxima above floor
        peaks, props = scipy.signal.find_peaks(
            col,
            height   = (height_db, 0),
            distance = min_dist_samp,
        )

        if len(peaks) == 0:
            continue

        if use_ref:
            # --- WITH REFERENCE: pick peak closest to reference ---
            T_j    = per_axis[j]
            u_ref  = float(ref_interp(T_j))

            peak_vels     = vel_axis[peaks]
            dist_from_ref = np.abs(peak_vels - u_ref)

            # sort peaks by distance from reference (closest first)
            order = np.argsort(dist_from_ref)

            # assign ridges — reject if closest peak exceeds tolerance
            for k in range(num_ridges):
                if k < len(order):
                    pk  = peaks[order[k]]
                    dv_from_ref = dist_from_ref[order[k]]
                    if dv_from_ref <= ref_tolerance_kms:
                        group_vel[k, j] = vel_axis[pk]
                        peak_db[k, j]   = col[pk]
                    # else: leave as NaN — outlier rejected

        else:
            # --- WITHOUT REFERENCE: pick peak with highest amplitude ---
            order = np.argsort(props['peak_heights'])[::-1]  # strongest first

            for k in range(num_ridges):
                if k < len(order):
                    pk = peaks[order[k]]
                    group_vel[k, j] = vel_axis[pk]
                    peak_db[k, j]   = col[pk]

    return group_vel, peak_db


# ===========================================================================
# 6.  Phase velocity estimation
#     Faithful port of FrequencyTimeFrame.phase_velocity() (line 746)
# ===========================================================================

def estimate_phase_velocity(group_vel_ridge: np.ndarray,
                             per_axis: np.ndarray,
                             vel_axis: np.ndarray,
                             phase_grid: np.ndarray,
                             ifreq_grid: np.ndarray,
                             t_grid: np.ndarray,
                             dist_km: float,
                             n_branches: int = 10,
                             piover4: float = -1.0,
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate phase velocity for all 2pi cycle branches.

    Uses the standard surface-wave phase relation (Bensen et al. 2007):
        c(ω) = ω · Δ / (ω · t_group + φ(ω) + π/4 + k·2π)
        c(T, k) = omega * delta / (omega * t_group + phi + piover4*pi/4 + k*2*pi)

    where:
        omega   = 2*pi/T          nominal angular frequency [rad/s]
        t_group = delta / U(T)    group arrival time from ridge [s]
        phi     = CWT phase at ridge point [rad]
        piover4 = -1 for EGFs/cross-correlations, +1 for seismograms
        k       = integer cycle branch

    This is more reliable than using omega_inst (instantaneous frequency)
    in the numerator, which introduces errors from CWT frequency mixing.

    Parameters
    ----------
    group_vel_ridge : (n_per,)      group velocity on the ridge [km/s]
    per_axis        : (n_per,)      period axis [s]
    vel_axis        : (n_vel,)      velocity axis [km/s]
    phase_grid      : (n_vel, n_per) instantaneous phase [rad]
    ifreq_grid      : (n_vel, n_per) instantaneous frequency [Hz] (not used here)
    t_grid          : (n_vel, n_per) group arrival time [s]
    dist_km         : inter-station distance [km]
    n_branches      : number of integer-cycle branches
    piover4         : phase correction: -1 EGF, +1 seismogram

    Returns
    -------
    phase_vel_array : (n_branches, n_per) [km/s]
    k_values        : (n_branches,) integer branch indices
    """

    k_values = np.arange(-n_branches // 2, n_branches // 2)
    n_per    = len(per_axis)
    pv       = np.full((len(k_values), n_per), np.nan)

    for j in range(n_per):
        if np.isnan(group_vel_ridge[j]):
            continue

        T   = per_axis[j]
        omega = 2.0 * np.pi / T          # nominal angular frequency

        # find ridge point in velocity grid
        idx_vel = np.abs(vel_axis - group_vel_ridge[j]).argmin()

        phi = phase_grid[idx_vel, j]     # CWT phase at ridge [rad]
        t0  = t_grid[idx_vel, j]         # group arrival time [s]

        # fallback: compute t0 directly from group velocity if grid value is bad
        if not np.isfinite(t0) or t0 <= 0:
            t0 = dist_km / group_vel_ridge[j]

        if not (np.isfinite(phi) and np.isfinite(t0) and t0 > 0):
            continue

        # pi/4 correction (same convention as AFTAN piover4)
        phase_corr = piover4 * np.pi / 4.0

        for ki, k in enumerate(k_values):
            denom = omega * t0 + phi + phase_corr + k * 2.0 * np.pi
            if abs(denom) > 1e-10:
                pv[ki, j] = omega * dist_km / denom

    return pv, k_values



# ===========================================================================
# 7.  Header extraction
# ===========================================================================

def extract_distance(tr: Trace) -> Tuple[float, float, float]:
    """Extract (dist_km, azim, bazim) from H5 geodetic / mseed / SAC header."""
    if hasattr(tr.stats, 'geodetic'):
        raw = tr.stats.geodetic
        for getter in [lambda r: list(r['geodetic']),
                       lambda r: list(r.geodetic)]:
            try:
                geo = getter(raw)
                return float(geo[0]), float(geo[1]), float(geo[2])
            except Exception:
                pass
    try:
        geo = tr.stats.mseed['geodetic']
        d   = float(geo[0])
        return (d * 1e-3 if d > 20000 else d), float(geo[1]), float(geo[2])
    except Exception:
        pass
    try:
        return (float(tr.stats.sac.dist),
                float(getattr(tr.stats.sac, 'az',  0.0)),
                float(getattr(tr.stats.sac, 'baz', 0.0)))
    except Exception:
        pass
    raise ValueError(
        f"Cannot extract distance from '{tr.id}'. "
        "Expected tr.stats.geodetic, tr.stats.mseed['geodetic'], or tr.stats.sac.dist.")


def select_phase_velocity_branch(phase_vel_array: np.ndarray,
                                  k_values: np.ndarray,
                                  per_axis: np.ndarray,
                                  group_vel_ridge: np.ndarray,
                                  ref_periods: Optional[np.ndarray] = None,
                                  ref_phase_vel: Optional[np.ndarray] = None,
                                  ) -> Tuple[np.ndarray, int]:
    """
    Automatically select the physically correct phase velocity branch.

    Strategy (in order of priority):

    1. If a reference phase velocity is provided:
       Choose the branch k whose RMS distance to ref_phase_vel is minimum,
       evaluated only at periods where the ridge is valid and c > U.

    2. Without reference:
       Choose the branch where:
         a) c(T) > U(T)  at all valid periods  (phase > group always)
         b) c(T) is smooth  (minimise RMS of second differences)
         c) c(T) is in a physically plausible range  [U*1.0 ... U*1.6]

    Parameters
    ----------
    phase_vel_array : (n_branches, n_per)  all cycle branches [km/s]
    k_values        : (n_branches,)        integer k per branch
    per_axis        : (n_per,)             period axis [s]
    group_vel_ridge : (n_per,)             group velocity on the ridge [km/s]
    ref_periods     : optional reference periods [s]
    ref_phase_vel   : optional reference phase velocities [km/s]

    Returns
    -------
    best_branch : (n_per,) selected phase velocity [km/s], NaN where invalid
    best_k      : integer k of the selected branch
    """
    n_branches, n_per = phase_vel_array.shape
    valid_ridge = np.isfinite(group_vel_ridge)

    # ---- build reference interpolated to per_axis ----
    if ref_periods is not None and ref_phase_vel is not None:
        ref_c = np.interp(per_axis, ref_periods, ref_phase_vel,
                          left=np.nan, right=np.nan)
    else:
        ref_c = None

    best_score = np.inf
    best_ki    = 0

    for ki in range(n_branches):
        row = phase_vel_array[ki, :]

        # require finite values where ridge is valid
        valid = valid_ridge & np.isfinite(row)
        if valid.sum() < 3:
            continue

        c = row[valid]
        u = group_vel_ridge[valid]

        # --- hard constraint: c must be > U everywhere ---
        if np.any(c <= u):
            continue

        # --- hard constraint: c must be in plausible range [U, U*1.8] ---
        if np.any(c > u * 1.8) or np.any(c < u * 0.95):
            continue

        # --- score 1: distance to reference (if available) ---
        if ref_c is not None:
            ref_valid = valid & np.isfinite(ref_c)
            if ref_valid.sum() < 2:
                score = np.inf
            else:
                score = np.sqrt(np.mean((row[ref_valid] - ref_c[ref_valid]) ** 2))
        else:
            # --- score 2: smoothness (second derivative) + proximity to U*1.1 ----
            if len(c) < 3:
                continue
            smoothness = np.sqrt(np.mean(np.diff(c, n=2) ** 2))
            proximity  = np.sqrt(np.mean((c - u * 1.1) ** 2))
            score = smoothness + 0.5 * proximity

        if score < best_score:
            best_score = score
            best_ki    = ki

    best_branch = phase_vel_array[best_ki, :].copy()
    best_k      = int(k_values[best_ki])

    # zero out physically impossible values
    valid_ridge_full = np.isfinite(group_vel_ridge)
    for j in range(n_per):
        if not valid_ridge_full[j]:
            best_branch[j] = np.nan
            continue
        c = best_branch[j]
        u = group_vel_ridge[j]
        if not np.isfinite(c) or c <= u or c > u * 1.8:
            best_branch[j] = np.nan

    print(f"            Phase vel. branch selected: k={best_k}  "
          f"score={best_score:.4f}  "
          f"c_mean={np.nanmean(best_branch):.3f} km/s")

    return best_branch, best_k

# ===========================================================================
# 9.  Main pipeline
# ===========================================================================

def cwt_ftan(filepath: str,
             vmin: float = 2.0, vmax: float = 5.0,
             tmin: float = 5.0, tmax: float = 150.0,
             w: float = 6.0,
             nf: int = 64,
             min_db: float = -15.0,
             min_dist_kms: float = 0.2,
             ref_tolerance_kms: float = 0.5,
             num_ridges: int = 1,
             branch: str = 'fold',
             use_pmf: bool = False,
             pmf_ref_periods: Optional[np.ndarray] = None,
             pmf_ref_vel: Optional[np.ndarray] = None,
             pmf_ref_grvel: Optional[np.ndarray] = None,
             filter_parameter: float = 2.0,
             n_branches: int = 10,
             tresh: float = 3.0,
             npoints: int = 5,
             force_dist_km: Optional[float] = None,
             ) -> dict:
    """
    Full CWT-FTAN pipeline.

    Returns
    -------
    dict with keys: scalogram_db, vel_axis, per_axis, group_vel, peak_db,
                    phase_vel_array, k_values, sei, t_arr,
                    delta, dt, branch, trace, azim, bazim
    """
    st = obs_read(filepath)
    tr = st[0]
    fs = float(tr.stats.sampling_rate)
    dt = float(tr.stats.delta)

    if force_dist_km is not None:
        dist_km, azim, bazim = float(force_dist_km), float('nan'), float('nan')
    else:
        dist_km, azim, bazim = extract_distance(tr)

    print(f"[CWT-FTAN]  {os.path.basename(filepath)}")
    print(f"            dist={dist_km:.2f} km  az={azim:.1f}  branch='{branch}'")

    # --- branch ---
    raw = tr.data.astype(np.float64)
    raw -= np.mean(raw)
    sei  = prepare_branch(raw, branch)
    n    = len(sei)
    t_arr = np.arange(n) * dt

    # --- phase-match filter ---
    if use_pmf and pmf_ref_periods is not None and pmf_ref_vel is not None:
        print(f"            PMF  sigma={filter_parameter} s")
        sei = phase_matched_filter(sei, dt, dist_km,
                                   pmf_ref_periods, pmf_ref_vel,
                                   filter_parameter=filter_parameter)

    # --- CWT ---
    fmin = 1.0 / tmax
    fmax = 1.0 / tmin
    tt   = int(fs / fmin)   # kernel half-length, matches GUI

    print(f"            CWT  T={tmin}-{tmax} s  nf={nf}  "
          f"w={w}  kernel_half={tt} s")

    cwt_obj = MorletCWT(sei, fs, fmin, fmax, nf=nf, wmin=w, wmax=w, tt=tt)
    tf      = cwt_obj.compute_tf()
    phase_raw, ifreq_hz = cwt_obj.compute_phase_and_inst_freq(tf)

    # drop first column (mirrors GUI [:, 1:])
    tf_abs2   = np.abs(tf[:, 1:]).astype(np.float64) ** 2
    phase_raw = phase_raw[:, 1:]
    ifreq_hz  = ifreq_hz[:, 1:]

    # --- velocity grid ---
    amp_img, vel_axis, per_axis, phase_grid, ifreq_grid, t_grid = build_velocity_grid(
        tf_abs2, phase_raw, ifreq_hz,
        cwt_obj.frequencies, t_arr, dist_km, vmin, vmax,
        n_vel=300, min_db=min_db)

    # --- ridge ---
    # Without reference: picks highest-amplitude peak per period.
    # With pmf_ref_grvel supplied: picks peak closest to reference group velocity.
    group_vel, peak_db = find_ridges(
        amp_img, vel_axis,
        height_db         = min_db * 0.5,
        min_dist_kms      = min_dist_kms,
        num_ridges        = num_ridges,
        ref_group_vel     = pmf_ref_grvel,
        ref_periods       = pmf_ref_periods,
        per_axis          = per_axis,
        ref_tolerance_kms = ref_tolerance_kms,
    )

    mean_u = np.nanmean(group_vel[0])
    print(f"            Ridge: U_mean={mean_u:.3f} km/s"
          if np.isfinite(mean_u) else "            Ridge: NOT FOUND")

    # --- AFTAN-style jump correction on the primary ridge ---
    group_vel[0] = apply_jump_correction(
        group_vel[0], per_axis,
        tresh=tresh, npoints=npoints)

    # --- phase velocity ---
    phase_vel_array, k_values = estimate_phase_velocity(
        group_vel[0], per_axis, vel_axis,
        phase_grid, ifreq_grid, t_grid,
        dist_km, n_branches=n_branches)

    # --- automatic branch selection ---
    best_phase_vel, best_k = select_phase_velocity_branch(
        phase_vel_array=phase_vel_array,
        k_values=k_values,
        per_axis=per_axis,
        group_vel_ridge=group_vel[0],
        ref_periods=pmf_ref_periods,
        ref_phase_vel=pmf_ref_vel,

    )

    return dict(
        scalogram_db    = amp_img,
        vel_axis        = vel_axis,
        per_axis        = per_axis,
        group_vel       = group_vel,
        peak_db         = peak_db,
        phase_vel_array = phase_vel_array,
        k_values        = k_values,
        sei             = sei,
        t_arr           = t_arr,
        delta           = dist_km,
        dt              = dt,
        branch          = branch,
        trace           = tr,
        azim            = azim,
        bazim           = bazim,
        best_phase_vel=best_phase_vel,
        best_k=best_k,
    )

# ===========================================================================
# 9.  Plot  — four panels
# ===========================================================================

def plot_cwt_ftan(result: dict,
                  show: bool = True,
                  title: Optional[str] = None,
                  cmap: str = 'jet',
                  ref_periods: Optional[np.ndarray] = None,
                  ref_group_vel: Optional[np.ndarray] = None,
                  ref_phase_vel: Optional[np.ndarray] = None) -> object:
    """
    Four-panel figure layout:

      Panel 1 (left)       — CWT amplitude map  (Period × Velocity, log x)
                             Ridge overlaid as white dots.
                             Reference group velocity as green dashed line.

      Panel 2 (centre-left) — Phase velocity branches (Period × Phase Vel)
                              All unselected branches: faded scatter (alpha=0.15)
                              Selected branch: solid firebrick line, full opacity.
                              Reference phase velocity: orange dashed.
                              No legend (too cluttered).
                              Y-axis shared with panel 1 velocity range.

      Panel 3 (centre-right) — Dispersion curves (Period × Velocity)
                               Group velocity ridge + selected phase velocity.
                               Reference curves as dashed lines.
                               Y-axis shared with panel 1 velocity range.

      Panel 4 (right, narrow) — EGF waveform (Amplitude × Lag time)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib not available")
        return None

    amp      = result['scalogram_db']
    vel_axis = result['vel_axis']
    per_axis = result['per_axis']
    gv       = result['group_vel']
    pv_arr   = result['phase_vel_array']
    k_vals   = result['k_values']
    sei      = result['sei']
    t_arr    = result['t_arr']
    delta    = result['delta']
    branch   = result['branch']
    azim     = result.get('azim', float('nan'))
    tr       = result.get('trace')
    best_pv  = result.get('best_phase_vel')
    best_k   = result.get('best_k', '?')

    # shared velocity limits — honour the command-line vmin/vmax exactly
    vmin_p, vmax_p = vel_axis[0], vel_axis[-1]
    per_min, per_max = per_axis[0], per_axis[-1]

    if title is None:
        name  = tr.id if tr is not None else ''
        title = (f"{name}   Δ={delta:.1f} km"
                 + (f"  az={azim:.1f}°" if not np.isnan(azim) else "")
                 + f"  branch={branch}")

    fig = plt.figure(figsize=(20, 7))
    gs  = gridspec.GridSpec(1, 4, width_ratios=[3, 2.5, 2.5, 1.2],
                            wspace=0.38, left=0.05, right=0.97,
                            top=0.88, bottom=0.11)
    ax_map  = fig.add_subplot(gs[0])   # CWT amplitude map
    ax_phv  = fig.add_subplot(gs[1])   # phase velocity branches  ← now panel 2
    ax_disp = fig.add_subplot(gs[2])   # dispersion curves        ← now panel 3
    ax_egf  = fig.add_subplot(gs[3])   # EGF waveform

    ridge_colors = ['white', 'lime', 'cyan']

    def apply_log_ticks(ax):
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=[1, 2, 3, 5, 7]))
        ax.set_xlim(per_min, per_max)

    # ================================================================== #
    # Panel 1 — CWT amplitude map                                        #
    # ================================================================== #
    valid = amp[np.isfinite(amp)]
    v_lo  = valid.min() if len(valid) else -15
    pcm = ax_map.contourf(per_axis, vel_axis, amp,
                          levels=60, cmap=cmap, vmin=v_lo, vmax=0)
    fig.colorbar(pcm, ax=ax_map, pad=0.02, shrink=0.85).set_label('Power [dB]', fontsize=8)

    # ridge dots
    for k in range(gv.shape[0]):
        mask = np.isfinite(gv[k])
        if mask.sum() > 0:
            ax_map.scatter(per_axis[mask], gv[k][mask],
                           c=ridge_colors[k % len(ridge_colors)],
                           s=15, zorder=5, label=f'Ridge {k+1}')

    # reference group velocity on map
    if ref_periods is not None and ref_group_vel is not None:
        ax_map.plot(ref_periods, ref_group_vel, '--',
                    color='limegreen', lw=1.5, label='Ref. group vel.')

    apply_log_ticks(ax_map)
    ax_map.set_ylim(vmin_p, vmax_p)          # ← shared y-limits
    ax_map.set_xlabel('Period [s]', fontsize=10)
    ax_map.set_ylabel('Group Velocity [km/s]', fontsize=10)
    ax_map.set_title('CWT amplitude map', fontsize=10)
    ax_map.legend(fontsize=7, loc='upper right')
    ax_map.grid(True, alpha=0.15, ls='--', color='w')

    # ================================================================== #
    # Panel 2 — Phase velocity branches                                  #
    # ================================================================== #
    cmap_br = plt.cm.get_cmap('tab10', len(k_vals))

    # identify which ki corresponds to best_k so we can highlight it
    best_ki_plot = None
    if best_pv is not None:
        for ki, k in enumerate(k_vals):
            if int(k) == int(best_k):
                best_ki_plot = ki
                break

    # draw all branches — unselected ones are very transparent
    for ki, k in enumerate(k_vals):
        row  = pv_arr[ki, :]
        mask = np.isfinite(row) & (row > vmin_p * 0.5) & (row < vmax_p * 2.5)
        if mask.sum() == 0:
            continue
        is_selected = (ki == best_ki_plot)
        ax_phv.scatter(per_axis[mask], row[mask],
                       s=12,
                       color=cmap_br(ki),
                       alpha=1.0 if is_selected else 0.15,
                       zorder=4 if is_selected else 2)

    # reference phase velocity
    if ref_periods is not None and ref_phase_vel is not None:
        ax_phv.plot(ref_periods, ref_phase_vel, '--',
                    color='darkorange', lw=2, zorder=5)

    # selected branch — bold firebrick line on top
    if best_pv is not None:
        mask = np.isfinite(best_pv)
        if mask.sum() > 0:
            ax_phv.plot(per_axis[mask], best_pv[mask],
                        '-', color='firebrick', lw=2.2,
                        zorder=6, label=f'Selected (k={best_k})')
            ax_phv.legend(fontsize=8, loc='upper left')

    apply_log_ticks(ax_phv)
    ax_phv.set_ylim(vmin_p, vmax_p)          # ← shared y-limits
    ax_phv.set_xlabel('Period [s]', fontsize=10)
    ax_phv.set_ylabel('Phase Velocity [km/s]', fontsize=10)
    ax_phv.set_title('Phase vel. branches', fontsize=10)
    ax_phv.grid(True, alpha=0.25, ls='--')

    # ================================================================== #
    # Panel 3 — Dispersion curves (group + selected phase)               #
    # ================================================================== #
    # group velocity ridges
    for k in range(gv.shape[0]):
        mask = np.isfinite(gv[k])
        if mask.sum() > 0:
            ax_disp.plot(per_axis[mask], gv[k][mask],
                         's-', ms=4, lw=1.6,
                         color=['navy', 'darkgreen'][k % 2],
                         label=f'Group vel. {k+1}')

    # selected phase velocity on dispersion panel
    if best_pv is not None:
        mask = np.isfinite(best_pv)
        if mask.sum() > 0:
            ax_disp.plot(per_axis[mask], best_pv[mask],
                         '^-', color='firebrick', ms=4, lw=1.6,
                         label=f'Phase vel. (k={best_k})')

    # reference curves
    if ref_periods is not None and ref_group_vel is not None:
        ax_disp.plot(ref_periods, ref_group_vel, '--',
                     color='limegreen', lw=1.5, alpha=0.8,
                     label='Ref. group vel.')
    if ref_periods is not None and ref_phase_vel is not None:
        ax_disp.plot(ref_periods, ref_phase_vel, '--',
                     color='darkorange', lw=1.5, alpha=0.8,
                     label='Ref. phase vel.')

    apply_log_ticks(ax_disp)
    ax_disp.set_ylim(vmin_p, vmax_p)         # ← shared y-limits
    ax_disp.set_xlabel('Period [s]', fontsize=10)
    ax_disp.set_ylabel('Velocity [km/s]', fontsize=10)
    ax_disp.set_title('Dispersion curves', fontsize=10)
    ax_disp.legend(fontsize=7)
    ax_disp.grid(True, alpha=0.25, ls='--')

    # ================================================================== #
    # Panel 4 — EGF waveform                                             #
    # ================================================================== #
    t_max_show = delta / max(vmin_p, 0.1) * 1.1
    mask_t = t_arr <= t_max_show
    wf = sei[mask_t].astype(float)
    peak = np.max(np.abs(wf))
    if peak > 0:
        wf /= peak

    ax_egf.plot(wf, t_arr[mask_t], color='steelblue', lw=0.8)
    ax_egf.fill_betweenx(t_arr[mask_t], 0, wf,
                         where=wf >= 0, color='steelblue', alpha=0.35)
    ax_egf.fill_betweenx(t_arr[mask_t], 0, wf,
                         where=wf <  0, color='tomato',    alpha=0.35)
    ax_egf.axhspan(delta / vmax_p, delta / vmin_p,
                   color='gold', alpha=0.15)
    ax_egf.axvline(0, color='k', lw=0.5, alpha=0.4)
    ax_egf.set_ylim(0, t_max_show)
    ax_egf.set_xlim(-1.5, 1.5)
    ax_egf.set_xlabel('Amp.\n(norm.)', fontsize=8)
    ax_egf.yaxis.set_label_position('right')
    ax_egf.yaxis.tick_right()
    ax_egf.set_ylabel('Lag [s]', fontsize=9)
    lbl = {'fold': 'fold (avg)', 'causal': 'causal', 'acausal': 'acausal'}
    ax_egf.set_title(f"EGF\n({lbl.get(branch, branch)})", fontsize=9)

    fig.suptitle(title, fontsize=11, y=0.97)
    if show:
        plt.show()
    return fig


# ===========================================================================
# 10.  CLI argument parser
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='surfquake cwt_ftan',
        description='CWT-based FTAN: group and phase velocity dispersion from EGFs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Group velocity only, batch H5 EGFs
  surfquake cwt_ftan -i ./stack/ -o ./out/ --tmin 5 --tmax 80 --vmin 2 --vmax 5

  # With phase-match filter + reference model
  surfquake cwt_ftan -i ./stack/ -o ./out/ --tmin 5 --tmax 80 \\
      --use_pmf --ref ak135f --wave rayleigh --plot

  # Single SAC, causal branch, tight band
  surfquake cwt_ftan -i file.sac -o ./out/ \\
      --branch causal --tmin 10 --tmax 40 --wmin 8 --wmax 16 --plot

Parameter guide:
  --wmin / --wmax   Morlet cycle range.
                    Low  (4-6)  = better time resolution, broader freq smearing.
                    High (12-20) = better freq resolution, longer wavelet kernel.
                    Default: wmin=6  wmax=12.
  --nf              Log-spaced period nodes.  Default: 64.
  --min_db          dB floor for display and ridge detection.  Default: -15.
  --min_dist_kms    Min velocity separation between ridges [km/s].  Default: 0.2.
  --num_ridges      1 = fundamental mode only.  2 = also first overtone.
  --filter_param    PMF Gaussian window width [s].  Default: 2.0.
  --n_branches      2pi cycle branches for phase velocity.  Default: 10.
  --ref             Reference model name (e.g. ak135f) or CSV path.
  --wave            rayleigh (default) or love.
"""
    )
    p.add_argument('-i', '--input',        required=True)
    p.add_argument('-o', '--output',       required=True)
    p.add_argument('--pattern',            default='*.H5')
    p.add_argument('--tmin',               type=float, default=5.0,    help='Min period [s]')
    p.add_argument('--tmax',               type=float, default=150.0,  help='Max period [s]')
    p.add_argument('--vmin',               type=float, default=2.0,    help='Min group vel [km/s]')
    p.add_argument('--vmax',               type=float, default=5.0,    help='Max group vel [km/s]')
    p.add_argument('--wmin',               type=float, default=6.0,    help='Min Morlet cycles')
    p.add_argument('--wmax',               type=float, default=12.0,   help='Max Morlet cycles')
    p.add_argument('--nf',                 type=int,   default=64,     help='Number of frequencies')
    p.add_argument('--min_db',             type=float, default=-15.0,  help='dB floor')
    p.add_argument('--min_dist_kms',       type=float, default=0.2,    help='Min ridge sep [km/s]')
    p.add_argument('--ref_tolerance_kms',  type=float, default=0.5,
                   help='Max distance from reference for ridge acceptance [km/s] '
                        '(default: 0.5). Only used when --ref is given. '
                        'Increase for noisy data, decrease for well-constrained paths.')
    p.add_argument('--num_ridges',         type=int,   default=1,      help='Ridges to extract')
    p.add_argument('--branch',             default='fold',
                   choices=['fold', 'causal', 'acausal'])
    p.add_argument('--use_pmf',            action='store_true',
                   help='Apply phase-match filter (requires --ref)')
    p.add_argument('--filter_param',       type=float, default=2.0,
                   help='PMF Gaussian window width [s]')
    p.add_argument('--n_branches',         type=int,   default=10,
                   help='2pi cycle branches for phase velocity')
    p.add_argument('--tresh',              type=float, default=3.0,
                   help='Jump detection threshold for group velocity correction '
                        '(default: 3.0 for CWT; lower = more aggressive). '
                        'AFTAN uses 10.0 for its cleaner Gaussian-filtered ridges.')
    p.add_argument('--npoints',            type=int,   default=5,
                   help='Max correctable jump length in period bins (default: 5)')
    p.add_argument('--ref',                type=str,   default=None,
                   help='Reference model name or CSV path')
    p.add_argument('--wave',               type=str,   default='rayleigh',
                   choices=['rayleigh', 'love'])
    p.add_argument('--plot',               action='store_true')
    p.add_argument('--force_dist_km',      type=float, default=None)
    return p


# ===========================================================================
# 11.  Shared run logic
# ===========================================================================

def _run(args):
    os.makedirs(args.output, exist_ok=True)

    # load reference
    pmf_ref_periods = pmf_ref_vel = ref_group_vel = ref_group_per = None

    if args.ref:
        try:
            from surfquakecore.ant.ant_refs import ref_for_aftan
            ref_data        = ref_for_aftan(args.ref, wave=args.wave,
                                            tmin=args.tmin, tmax=args.tmax)
            pmf_ref_periods = ref_data['phprper']
            pmf_ref_vel     = ref_data['phprvel']
            ref_group_vel   = ref_data['pred'][:, 1]
            ref_group_per   = ref_data['pred'][:, 0]
            print(f"[CWT-FTAN] Reference '{args.ref}' ({args.wave})  "
                  f"T={pmf_ref_periods[0]:.1f}-{pmf_ref_periods[-1]:.1f} s")
        except Exception as exc:
            print(f"[CWT-FTAN][WARN] Could not load reference '{args.ref}': {exc}")

    # collect files
    inp = args.input
    if os.path.isdir(inp):
        files = sorted(glob.glob(os.path.join(inp, args.pattern)))
        if not files:
            print(f"[CWT-FTAN] No files matching '{args.pattern}' in {inp}")
            return
        print(f"[CWT-FTAN] Found {len(files)} file(s)")
    else:
        files = [inp] if os.path.isfile(inp) else []
        if not files:
            print(f"[CWT-FTAN] File not found: {inp}")
            return

    n_ok = n_skip = 0

    for filepath in files:
        basename = os.path.splitext(os.path.basename(filepath))[0]
        try:
            result = cwt_ftan(
                filepath,
                vmin             = args.vmin,
                vmax             = args.vmax,
                tmin             = args.tmin,
                tmax             = args.tmax,
                wmin             = args.wmin,
                wmax             = args.wmax,
                nf               = args.nf,
                min_db           = args.min_db,
                min_dist_kms     = args.min_dist_kms,
                ref_tolerance_kms = args.ref_tolerance_kms,
                num_ridges       = args.num_ridges,
                branch           = args.branch,
                use_pmf          = args.use_pmf and (pmf_ref_periods is not None),
                pmf_ref_periods  = pmf_ref_periods,
                pmf_ref_vel      = pmf_ref_vel,
                pmf_ref_grvel    = ref_group_vel,
                filter_parameter = args.filter_param,
                n_branches       = args.n_branches,
                tresh            = args.tresh,
                npoints          = args.npoints,
                force_dist_km    = args.force_dist_km,
            )

            gv     = result['group_vel']
            pv_arr = result['phase_vel_array']
            k_vals = result['k_values']
            per    = result['per_axis']
            delta  = result['delta']

            # group velocity table
            with open(os.path.join(args.output, basename + '.grp.disp'), 'w') as fh:
                fh.write(f"# CWT-FTAN group vel  file={basename}  "
                         f"delta={delta:.3f} km  branch={args.branch}\n")
                fh.write(f"# {'Period_s':>10}  {'GroupVel_kms':>14}  {'Power_dB':>10}\n")
                for j in range(len(per)):
                    if np.isfinite(gv[0, j]):
                        fh.write(f"  {per[j]:>10.4f}  {gv[0,j]:>14.4f}  "
                                 f"{result['peak_db'][0,j]:>10.2f}\n")

            # phase velocity branches table
            with open(os.path.join(args.output, basename + '.phv.disp'), 'w') as fh:
                fh.write(f"# CWT-FTAN phase vel branches  file={basename}  "
                         f"delta={delta:.3f} km  branch={args.branch}\n")
                fh.write(f"# {'Period_s':>10}  {'k':>5}  {'PhaseVel_kms':>14}\n")
                for j in range(len(per)):
                    for ki, k in enumerate(k_vals):
                        v = pv_arr[ki, j]
                        if np.isfinite(v) and args.vmin * 0.5 < v < args.vmax * 2.5:
                            fh.write(f"  {per[j]:>10.4f}  {int(k):>5}  {v:>14.4f}\n")

            print(f"[OK]  {basename}  delta={delta:.1f} km")

            if args.plot:
                import matplotlib
                matplotlib.use('Agg')
                fig = plot_cwt_ftan(
                    result, show=False,
                    ref_periods   = ref_group_per,
                    ref_group_vel = ref_group_vel,
                    ref_phase_vel = pmf_ref_vel,
                )
                if fig is not None:
                    import matplotlib.pyplot as plt
                    png = os.path.join(args.output, basename + '.png')
                    fig.savefig(png, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f"       plot -> {png}")

            n_ok += 1

        except Exception as exc:
            import traceback
            print(f"[ERROR] {basename}: {exc}")
            traceback.print_exc()
            n_skip += 1

    print(f"\n[CWT-FTAN] Done.  OK={n_ok}  Skipped={n_skip}")
    print(f"[CWT-FTAN] Output: {args.output}")