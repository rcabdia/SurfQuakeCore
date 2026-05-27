"""
sac_to_h5_and_test.py
=====================
Two things in one script:

  1. Convert a SAC cross-correlation file to the H5 format used by
     surfquakecore/crossstack.py, building exactly the tr.stats.geodetic
     header that _extract_geodetic_h5() expects.

  2. Run AFTAN directly on the SAC file (no conversion needed) to
     verify the parameters work before worrying about file format.

Usage
-----
    python sac_to_h5_and_test.py                  # uses hard-coded paths below
    python sac_to_h5_and_test.py my_file.SAC      # override input file

Dependencies:  obspy, h5py, numpy, matplotlib (for plot)
"""

import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# ① SAC → H5 conversion
# ---------------------------------------------------------------------------

def sac_to_h5(sac_path: str,
              output_dir: str = None) -> str:
    """
    Read a SAC cross-correlation and write it as an H5 file using
    ObsPy's native H5 writer (tr.write(..., format='H5')).

    The geodetic header (tr.stats.geodetic) is built from the SAC
    header fields and attached before writing, so that
    _extract_geodetic_h5() can read it correctly.

    Parameters
    ----------
    sac_path   : path to input SAC file
    output_dir : directory where the H5 is written.
                 Filename = same basename with .H5 extension.
                 If None, H5 is written next to the SAC file.

    Returns
    -------
    h5_path : path to the written H5 file
    """
    try:
        from obspy import read
        from obspy.core.util import AttribDict
    except ImportError as e:
        raise ImportError(f"Need obspy: pip install obspy  ({e})")

    # --- resolve output path ---
    basename = os.path.splitext(os.path.basename(sac_path))[0] + ".H5"
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        h5_path = os.path.join(output_dir, basename)
    else:
        h5_path = os.path.join(os.path.dirname(os.path.abspath(sac_path)), basename)

    # --- read SAC ---
    st = read(sac_path)
    tr = st[0]
    sac = tr.stats.sac

    # --- extract geometry ---
    dist_km = float(sac.dist)
    azim    = float(getattr(sac, 'az',   0.0))
    bazim   = float(getattr(sac, 'baz',  0.0))
    evla    = float(getattr(sac, 'evla', 0.0))
    evlo    = float(getattr(sac, 'evlo', 0.0))
    otime   = float(tr.stats.starttime.timestamp)

    print(f"[SAC→H5] Input  : {sac_path}")
    print(f"         dist   = {dist_km:.3f} km")
    print(f"         az     = {azim:.2f}°   baz = {bazim:.2f}°")
    print(f"         npts   = {tr.stats.npts}   dt = {tr.stats.delta} s")
    print(f"         b      = {sac.b} s")

    # --- attach geodetic header exactly as crossstack.py writes it ---
    tr.stats.geodetic = AttribDict({
        'otime'   : otime,
        'geodetic': [dist_km, azim, bazim, 0.0],
        'event'   : [evla, evlo, 0.0],
        'arrivals': []
    })

    tr.stats.mseed = {'dataquality': 'D', 'geodetic': [dist_km*1E3, bazim, azim],
                                                      'cross_channels': "ZZ",
                                                      'coordinates': [evla, evlo, 0.0, 0.0]}
    # --- write using ObsPy native H5 writer ---
    tr.write(h5_path, format="H5")

    print(f"[SAC→H5] Output : {h5_path}")
    return h5_path


# ---------------------------------------------------------------------------
# ② Direct AFTAN test on SAC (bypasses file-format issues entirely)
# ---------------------------------------------------------------------------

# def test_aftan_on_sac(sac_path: str,
#                       vmin: float = 2.0,
#                       vmax: float = 5.0,
#                       tmin: float = 10.0,
#                       tmax: float = 30.0,
#                       tresh: float = 10.0,
#                       ffact: float = 1.0,
#                       perc: float  = 20.0,
#                       npoints: int = 5,
#                       taperl: float = 1.5,
#                       nfin: int    = 64,
#                       piover4: float = -1.0,
#                       branch: str  = 'fold',
#                       show_plot: bool = True):
#     """
#     Load a SAC EGF directly, prepare the chosen branch, and run AFTAN.
#     This completely bypasses H5/ObsPy header detection — useful for debugging.
#
#     Prints a full diagnostic and shows the FTAN plot.
#     """
#     from obspy import read
#
#     st  = read(sac_path)
#     tr  = st[0]
#     sac = tr.stats.sac
#
#     dt      = float(tr.stats.delta)
#     n_full  = tr.stats.npts
#     dist_km = float(sac.dist)
#     azim    = float(getattr(sac, 'az',  0.0))
#     bazim   = float(getattr(sac, 'baz', 0.0))
#     sac_b   = float(sac.b)          # e.g. -3600.0
#
#     print("=" * 60)
#     print(f"FILE     : {os.path.basename(sac_path)}")
#     print(f"PAIR     : {tr.id}  (evnm={getattr(sac,'kevnm','')})")
#     print(f"dist     : {dist_km:.2f} km")
#     print(f"az/baz   : {azim:.1f}° / {bazim:.1f}°")
#     print(f"npts     : {n_full}   dt = {dt} s   duration = {n_full*dt:.0f} s")
#     print(f"sac.b    : {sac_b} s   (zero-lag at sample {int(-sac_b/dt)})")
#     print(f"branch   : {branch}")
#     print()
#
#     # --- geometry sanity check ---
#     t_fast  = dist_km / vmax
#     t_slow  = dist_km / vmin
#     taper_l = taperl * tmax
#     print(f"GEOMETRY CHECK")
#     print(f"  Signal window    : {t_fast:.1f} s → {t_slow:.1f} s")
#     print(f"  Left taper length: {taper_l:.1f} s  (taperl={taperl} × tmax={tmax})")
#     print(f"  Taper vs signal  : {'OK ✓' if taper_l < t_fast else 'WARNING — taper overlaps signal! Reduce tmax.'}")
#     max_lam = vmax * tmax
#     print(f"  Max wavelength   : {max_lam:.0f} km vs dist {dist_km:.0f} km  "
#           f"({'OK ✓' if max_lam < dist_km else 'WARNING — wavelength > distance!'})")
#     print()
#
#     # --- prepare EGF branch ---
#     # The SAC EGF is symmetric: sample 0 = lag sac_b, centre = lag 0
#     ic = int(round(-sac_b / dt))   # index of zero-lag sample
#     data = tr.data.astype(np.float64)
#
#     causal  = data[ic:].copy()
#     acausal = data[ic::-1].copy()
#     L = min(len(causal), len(acausal))
#
#     if branch == 'fold':
#         sei = (0.5 * (causal[:L] + acausal[:L])).astype(np.float32)
#     elif branch == 'causal':
#         sei = causal[:L].astype(np.float32)
#     elif branch == 'acausal':
#         sei = acausal[:L].astype(np.float32)
#     else:
#         sei = causal[:L].astype(np.float32)
#
#     n   = len(sei)
#     t0  = 0.0   # branch always starts at lag=0
#
#     print(f"BRANCH PREPARED: n={n} samples  ({n*dt:.0f} s of causal signal)")
#     print()
#
#     # --- run AFTAN ---
#     # Import from surfquakecore if available, otherwise fall back to local aftan.py
#     try:
#         from surfquakecore.ant.aftan import aftanpg, prepare_egf
#         print("[INFO] Using surfquakecore.ant.aftan")
#     except ImportError:
#         try:
#             from aftan import aftanpg
#             print("[INFO] Using local aftan.py")
#         except ImportError:
#             raise ImportError(
#                 "Cannot find aftan module. Either:\n"
#                 "  • Run from inside the SurfQuakeCore repo, or\n"
#                 "  • Place aftan.py in the same directory as this script."
#             )
#
#     print("Running aftanpg ...")
#     nfout1, arr1, nfout2, arr2, tamp, amp, ierr = aftanpg(
#         piover4 = piover4,
#         n       = n,
#         sei     = sei,
#         t0      = t0,
#         dt      = dt,
#         delta   = dist_km,
#         vmin    = vmin,
#         vmax    = vmax,
#         tmin    = tmin,
#         tmax    = tmax,
#         tresh   = tresh,
#         ffact   = ffact,
#         perc    = perc,
#         npoints = npoints,
#         taperl  = taperl,
#         nfin    = nfin,
#     )
#
#     print()
#     print(f"RESULT")
#     print(f"  ierr   = {ierr}  "
#           f"({'OK' if ierr==0 else 'some issues' if ierr==1 else 'NO FINAL RESULT'})")
#     print(f"  nfout1 = {nfout1}  (preliminary frequencies)")
#     print(f"  nfout2 = {nfout2}  (final frequencies after jump correction)")
#
#     if nfout1 > 0:
#         print(f"\nPreliminary group velocity range: "
#               f"{arr1[2,:nfout1].min():.3f} – {arr1[2,:nfout1].max():.3f} km/s")
#         print(f"Period range: "
#               f"{arr1[0,:nfout1].min():.1f} – {arr1[0,:nfout1].max():.1f} s")
#
#     if nfout2 > 0:
#         print(f"\nFinal group velocity:")
#         print(f"  {'Period [s]':>10}  {'Group vel [km/s]':>16}  {'SNR [dB]':>10}")
#         for i in range(nfout2):
#             print(f"  {arr2[0,i]:>10.2f}  {arr2[2,i]:>16.4f}  {arr2[5,i]:>10.2f}")
#     else:
#         print("\n  *** No final result — see ierr and diagnostic above ***")
#         print("  Suggestions:")
#         print("  • Increase --tresh (try 10.0)")
#         print("  • Check taper does not overlap signal window")
#         print("  • Try --branch causal or --branch acausal separately")
#
#     # --- plot ---
#     if show_plot:
#         try:
#             import matplotlib.pyplot as plt
#             import matplotlib.gridspec as gridspec
#
#             nf, ncol = amp.shape
#             times = tamp + np.arange(ncol) * dt
#             lag   = times + t0
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 vels = np.where(lag > 1e-3, dist_km / lag, np.nan)
#
#             if nfout1 > 0:
#                 per_min = arr1[0, 0]
#                 per_max = arr1[0, nfout1 - 1]
#             else:
#                 per_min, per_max = tmin, tmax
#
#             per_axis = np.linspace(per_min, per_max, nf)
#             v_lo = np.floor(vmin * 4) / 4
#             v_hi = np.ceil(vmax  * 4) / 4
#
#             # rebuild amp map on vel×period grid
#             n_vel = 200
#             vel_grid = np.linspace(v_lo, v_hi, n_vel)
#             amp_img  = np.full((n_vel, nf), np.nan)
#             for ki in range(nf):
#                 rv = vels
#                 ra = amp[ki, :]
#                 mask = np.isfinite(rv) & (rv >= v_lo) & (rv <= v_hi)
#                 if mask.sum() < 2:
#                     continue
#                 idx = np.argsort(rv[mask])
#                 amp_img[:, ki] = np.interp(vel_grid,
#                                            rv[mask][idx], ra[mask][idx],
#                                            left=np.nan, right=np.nan)
#
#             amp_norm = amp_img - np.nanmax(amp_img)
#             amp_norm = np.clip(amp_norm, -5, 0)
#
#             fig = plt.figure(figsize=(16, 6))
#             gs  = gridspec.GridSpec(1, 3, width_ratios=[3, 3, 1.2],
#                                     wspace=0.32, left=0.07, right=0.96,
#                                     top=0.88, bottom=0.11)
#             ax_d = fig.add_subplot(gs[0])
#             ax_m = fig.add_subplot(gs[1])
#             ax_e = fig.add_subplot(gs[2])
#
#             # dispersion
#             if nfout1 > 0:
#                 ax_d.plot(arr1[0,:nfout1], arr1[2,:nfout1],
#                           'o', color='steelblue', ms=3, alpha=0.4,
#                           label='Prelim.')
#             if nfout2 > 0:
#                 ax_d.plot(arr2[0,:nfout2], arr2[2,:nfout2],
#                           's-', color='navy', ms=5, lw=1.8, label='Final')
#             ax_d.set_xlim(per_min, per_max)
#             ax_d.set_ylim(v_lo, v_hi)
#             ax_d.set_xlabel('Period [s]'); ax_d.set_ylabel('Group vel. [km/s]')
#             ax_d.set_title('Dispersion'); ax_d.legend(fontsize=8)
#             ax_d.grid(True, alpha=0.25, ls='--')
#
#             # amplitude map
#             ax_m.pcolormesh(per_axis, vel_grid, amp_norm,
#                             cmap='jet', vmin=-5, vmax=0, shading='auto')
#             if nfout2 > 0:
#                 ax_m.plot(arr2[0,:nfout2], arr2[2,:nfout2],
#                           'w.', ms=6, label='ridge')
#             ax_m.set_xlim(per_min, per_max); ax_m.set_ylim(v_lo, v_hi)
#             ax_m.set_xlabel('Period [s]'); ax_m.set_ylabel('Group vel. [km/s]')
#             ax_m.set_title('FTAN amplitude map')
#
#             # EGF waveform
#             t_sei = np.arange(len(sei)) * dt
#             t_max_show = dist_km / max(v_lo, 0.1) * 1.1
#             mask_t = t_sei <= t_max_show
#             wf = sei[mask_t].astype(float)
#             peak = np.max(np.abs(wf))
#             if peak > 0: wf /= peak
#             ax_e.plot(wf, t_sei[mask_t], color='steelblue', lw=0.8)
#             ax_e.fill_betweenx(t_sei[mask_t], 0, wf,
#                                where=wf >= 0, color='steelblue', alpha=0.35)
#             ax_e.fill_betweenx(t_sei[mask_t], 0, wf,
#                                where=wf <  0, color='tomato',    alpha=0.35)
#             ax_e.axhspan(t_fast, t_slow, color='gold', alpha=0.15,
#                          label='vel. window')
#             ax_e.set_ylim(0, t_max_show); ax_e.set_xlim(-1.5, 1.5)
#             ax_e.set_xlabel('Norm. amp.'); ax_e.set_title(f'EGF ({branch})')
#             ax_e.yaxis.set_label_position('right'); ax_e.yaxis.tick_right()
#             ax_e.set_ylabel('Lag [s]')
#
#             title = (f"TA.M14A – TA.M17A   Δ={dist_km:.1f} km   "
#                      f"branch={branch}   tmin={tmin} tmax={tmax} s")
#             fig.suptitle(title, fontsize=10)
#             plt.tight_layout()
#             plt.savefig("ftan_sac_test.png", dpi=150, bbox_inches='tight')
#             print("\n[PLOT] Saved → ftan_sac_test.png")
#             plt.show()
#
#         except Exception as e:
#             print(f"[PLOT] Failed: {e}")
#
#     return dict(nfout1=nfout1, arr1=arr1, nfout2=nfout2, arr2=arr2,
#                 tamp=tamp, amp=amp, ierr=ierr,
#                 delta=dist_km, dt=dt, t0=t0, sei=sei)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    sac_file = "/Users/robertocabiecesdiaz/Documents/AFTAN/test_mine/COR_TA.M14A_TA.M17A.sac"

    if not os.path.isfile(sac_file):
        print(f"File not found: {sac_file}")
        sys.exit(1)

    # ── Step 1: convert to H5 ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1 — Convert SAC → H5")
    print("="*60)
    h5_file = sac_to_h5(sac_file, output_dir="/Users/robertocabiecesdiaz/Documents/AFTAN/test_mine")

    # ── Step 2: run AFTAN directly on SAC (no format dependency) ──────────
    print("\n" + "="*60)
    print("STEP 2 — Run AFTAN directly on SAC (branch=fold)")
    print("="*60)
    # result = test_aftan_on_sac(
    #     sac_path  = sac_file,
    #     vmin      = 1.5,
    #     vmax      = 5.0,
    #     tmin      = 5.0,
    #     tmax      = 30.0,    # safe for dist=224 km, vmax=5 → t_fast=44.8 s
    #     tresh     = 10.0,    # standard value
    #     ffact     = 1.0,
    #     perc      = 20.0,
    #     npoints   = 5,
    #     taperl    = 1.5,
    #     nfin      = 64,
    #     piover4   = -1.0,
    #     branch    = 'fold',
    #     show_plot = True,
    # )

    # print("\n" + "="*60)
    # print("STEP 3 — Verify H5 header can be read back")
    # print("="*60)
    # try:
    #     import h5py
    #     with h5py.File(h5_file, 'r') as fh:
    #         geo = fh['stats/geodetic/geodetic'][:]
    #         evt = fh['stats/geodetic/event'][:]
    #         print(f"  geodetic : dist={geo[0]:.2f} km  az={geo[1]:.2f}°  baz={geo[2]:.2f}°")
    #         print(f"  event    : lat={evt[0]:.4f}  lon={evt[1]:.4f}")
    #         print(f"  npts     : {fh['stats'].attrs['npts']}")
    #         print(f"  sac_b    : {fh['stats'].attrs['sac_b']} s")
    #         print("  H5 header OK ✓")
    # except Exception as e:
    #     print(f"  H5 read-back failed: {e}")
