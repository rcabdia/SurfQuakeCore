# surfquakecore/ant/ant_refs.py
"""
Reference dispersion model loader for AFTAN.

Models are stored as CSV files inside the same directory:
    surfquakecore/ant/models/<name>.csv

CSV format (header required):
    period, phase_velocity_rayleigh, phase_velocity_love,
            group_velocity_rayleigh, group_velocity_love

Usage
-----
    from surfquakecore.ant.ant_refs import load_ref, list_refs

    ref = load_ref("ak135f")          # or "ak-135f", "AK135F" – case/dash insensitive
    # ref is a dict with keys:
    #   'name'    : str
    #   'period'  : np.ndarray [s]
    #   'phR'     : phase vel Rayleigh [km/s]
    #   'phL'     : phase vel Love     [km/s]
    #   'grR'     : group vel Rayleigh [km/s]
    #   'grL'     : group vel Love     [km/s]
    #
    # For AFTAN pass:
    #   phprper = ref['period'],  phprvel = ref['phR']   # phase reference
    #   pred    = np.column_stack([ref['period'], ref['grR']])  # group pred (aftanipg)
"""

import os
import re
import numpy as np

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "disp_map_ref")


def _normalise_name(name: str) -> str:
    """Lower-case, strip dashes, spaces, underscores → canonical key."""
    return re.sub(r"[-_\s]", "", name.lower())


def list_refs() -> list:
    """Return list of available reference model names."""
    if not os.path.isdir(_MODELS_DIR):
        return []
    names = []
    for f in sorted(os.listdir(_MODELS_DIR)):
        if f.endswith(".csv"):
            names.append(os.path.splitext(f)[0])
    return names


def load_ref(name: str) -> dict:
    """
    Load a reference dispersion model by name.

    Matching is case-insensitive and ignores dashes/underscores, so
    'ak135f', 'ak-135f', 'AK135F' all resolve to ak135f.csv.

    Parameters
    ----------
    name : str
        Model name, e.g. 'ak135f', 'iasp91', or a full path to a CSV file.

    Returns
    -------
    dict with keys: name, period, phR, phL, grR, grL  (all np.ndarray)

    Raises
    ------
    FileNotFoundError  if no matching model is found.
    ValueError         if the CSV cannot be parsed.
    """
    # --- full path supplied directly ---
    model_name = os.path.splitext(os.path.basename(name))[0] + "_earth_velocity_fundamental_mode.txt"
    csv_path = os.path.join(_MODELS_DIR, model_name)
    # if os.path.isfile(name):
    #     csv_path = name
    #     model_name = os.path.splitext(os.path.basename(name))[0] + "_earth_velocity_first_mode.txt"
    # else:
    #     # --- look up in models directory ---
    #     if not os.path.isdir(_MODELS_DIR):
    #         raise FileNotFoundError(
    #             f"Models directory not found: {_MODELS_DIR}\n"
    #             f"Create it and place <model>.csv files inside."
    #         )
    #     target = _normalise_name(name)
    #     csv_path = None
    #     for f in os.listdir(_MODELS_DIR):
    #         if f.endswith(".csv") and _normalise_name(f[:-4]) == target:
    #             csv_path = os.path.join(_MODELS_DIR, f)
    #             model_name = f[:-4]
    #             break
    #     if csv_path is None:
    #         available = list_refs()
    #         raise FileNotFoundError(
    #             f"Reference model '{name}' not found in {_MODELS_DIR}.\n"
    #             f"Available models: {available}\n"
    #             f"You can also pass a full path to a CSV file."
    #         )

    # --- parse CSV ---
    try:
        data = np.genfromtxt(csv_path, delimiter=",", names=True,
                             dtype=float, encoding="utf-8")
    except Exception as exc:
        raise ValueError(f"Cannot parse reference CSV '{csv_path}': {exc}")

    # normalise column names: lower, strip spaces
    cols = {c.strip().lower().replace(" ", "_"): data[c] for c in data.dtype.names}

    def _get(keys):
        for k in keys:
            if k in cols:
                return cols[k]
        return np.zeros_like(cols[list(cols.keys())[0]])

    period = _get(["period"])
    phR    = _get(["phase_velocity_rayleigh", "phase_vel_rayleigh", "phR", "cR"])
    phL    = _get(["phase_velocity_love",     "phase_vel_love",     "phL", "cL"])
    grR    = _get(["group_velocity_rayleigh", "group_vel_rayleigh", "grR", "uR"])
    grL    = _get(["group_velocity_love",     "group_vel_love",     "grL", "uL"])

    # sort by period ascending
    idx = np.argsort(period)

    return dict(
        name   = model_name,
        period = period[idx],
        phR    = phR[idx],
        phL    = phL[idx],
        grR    = grR[idx],
        grL    = grL[idx],
    )


def ref_for_aftan(name: str,
                  wave: str = "rayleigh",
                  tmin: float = 1.0,
                  tmax: float = 300.0) -> dict:
    """
    Load a reference model and return arrays trimmed to [tmin, tmax]
    ready to pass directly to run_aftan() / aftanpg() / aftanipg().

    Parameters
    ----------
    name   : model name or CSV path
    wave   : 'rayleigh' (default) or 'love'
    tmin   : minimum period to include [s]
    tmax   : maximum period to include [s]

    Returns
    -------
    dict with keys:
        'phprper'  : period array for phase reference  [s]
        'phprvel'  : phase velocity array              [km/s]
        'pred'     : np.ndarray shape (N,2) [period, group_vel] for aftanipg
        'ref'      : the full raw dict from load_ref()
    """
    ref  = load_ref(name)
    per  = ref['period']
    wave = wave.lower()

    if wave in ('rayleigh', 'r', 'z', 'zz'):
        ph = ref['phR']
        gr = ref['grR']
    elif wave in ('love', 'l', 't', 'tt'):
        ph = ref['phL']
        gr = ref['grL']
    else:
        raise ValueError(f"wave must be 'rayleigh' or 'love', got '{wave}'")

    mask = (per >= tmin) & (per <= tmax)
    if mask.sum() < 2:
        raise ValueError(
            f"Fewer than 2 reference points in period range [{tmin},{tmax}] s "
            f"for model '{name}'. Check tmin/tmax or extend the model table."
        )

    return dict(
        phprper = per[mask],
        phprvel = ph[mask],
        pred    = np.column_stack([per[mask], gr[mask]]),
        ref     = ref,
    )