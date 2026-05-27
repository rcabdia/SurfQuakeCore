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
