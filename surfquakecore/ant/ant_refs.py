# surfquakecore/ant/ant_refs.py
"""
Reference dispersion model loader for AFTAN and CWT-FTAN.

Models are stored as text files inside:
    surfquakecore/ant/disp_curv_ref/

File naming convention:
    <model>_<environment>_velocity_<mode>_mode.txt

    model       : ak135
    environment : earth, ocean_deep, ocean_intermediate, ocean_shallow
    mode        : fundamental, first

Examples of available files:
    ak135_earth_velocity_fundamental_mode.txt
    ak135_earth_velocity_first_mode.txt
    ak135_ocean_deep_velocity_fundamental_mode.txt
    ak135_ocean_deep_velocity_first_mode.txt
    ak135_ocean_intermediate_velocity_fundamental_mode.txt
    ak135_ocean_intermediate_velocity_first_mode.txt
    ak135_ocean_shallow_velocity_fundamental_mode.txt
    ak135_ocean_shallow_velocity_first_mode.txt

File format (space or comma separated, header required):
    period, phase_velocity_rayleigh, phase_velocity_love,
            group_velocity_rayleigh, group_velocity_love

Users can also supply a full path to any CSV/TXT file with the same format.

Usage
-----
    from surfquakecore.ant.ant_refs import load_ref, list_refs, ref_for_aftan

    # List available built-in models
    print(list_refs())
    # → [('ak135', 'earth', 'fundamental'), ('ak135', 'earth', 'first'), ...]

    # Load by components (most explicit)
    ref = load_ref(model='ak135', env='earth', mode='fundamental')

    # Load by shorthand string (parsed automatically)
    ref = load_ref('ak135')                          # → earth + fundamental
    ref = load_ref('ak135_ocean_deep')               # → ocean_deep + fundamental
    ref = load_ref('ak135_ocean_deep_first')         # → ocean_deep + first mode
    ref = load_ref('ak135', env='ocean_shallow', mode='first')

    # Load a custom file by full path
    ref = load_ref('/data/my_region.csv')

    # ref dict keys:
    #   'name'    : str   — resolved file stem
    #   'period'  : ndarray [s]
    #   'phR'     : phase velocity Rayleigh [km/s]
    #   'phL'     : phase velocity Love     [km/s]
    #   'grR'     : group velocity Rayleigh [km/s]
    #   'grL'     : group velocity Love     [km/s]
    #   'model'   : str   — e.g. 'ak135'
    #   'env'     : str   — e.g. 'earth'
    #   'mode'    : str   — 'fundamental' or 'first'

    # Ready-to-use for AFTAN / CWT-FTAN
    ref_data = ref_for_aftan('ak135', wave='rayleigh', tmin=5, tmax=100)
    ref_data = ref_for_aftan('ak135', env='ocean_deep', mode='first',
                              wave='love', tmin=5, tmax=100)
"""

import os
import re
from typing import Optional, List, Tuple
import numpy as np
from surfquakecore import DISP_REF_CURVES



# ---------------------------------------------------------------------------
# Valid vocabulary
# ---------------------------------------------------------------------------
_VALID_ENVS   = ('earth', 'ocean_deep', 'ocean_intermediate', 'ocean_shallow')
_VALID_MODES  = ('fundamental', 'first')
_VALID_MODELS = ('ak135',)   # extend as new models are added

_ENV_ALIASES = {
    'earth':                'earth',
    'continental':          'earth',
    'cont':                 'earth',
    'ocean':                'ocean_deep',
    'oceandeep':            'ocean_deep',
    'ocean_deep':           'ocean_deep',
    'deep':                 'ocean_deep',
    'oceanintermediate':    'ocean_intermediate',
    'ocean_intermediate':   'ocean_intermediate',
    'intermediate':         'ocean_intermediate',
    'inter':                'ocean_intermediate',
    'oceanshallow':         'ocean_shallow',
    'ocean_shallow':        'ocean_shallow',
    'shallow':              'ocean_shallow',
}

_MODE_ALIASES = {
    'fundamental': 'fundamental',
    'fund':        'fundamental',
    '0':           'fundamental',
    'mode0':       'fundamental',
    'first':       'first',
    '1':           'first',
    'mode1':       'first',
    'higher':      'first',
}


def _norm(s: str) -> str:
    """Lowercase, strip dashes/spaces/underscores."""
    return re.sub(r'[-_\s]', '', s.lower())


# ---------------------------------------------------------------------------
# File name builder / finder
# ---------------------------------------------------------------------------

def _build_filename(model: str, env: str, mode: str) -> str:
    """Build the standard filename stem."""
    return f"{model}_{env}_velocity_{mode}_mode.txt"


def _resolve_file(model: str, env: str, mode: str) -> str:
    """
    Resolve to a full path inside _MODELS_DIR.
    Raises FileNotFoundError with a helpful message if not found.
    """
    fname = _build_filename(model, env, mode)
    path  = os.path.join(DISP_REF_CURVES, fname)
    if os.path.isfile(path):
        return path

    available = [f for f in os.listdir(DISP_REF_CURVES)
                 if f.endswith('.txt') or f.endswith('.csv')]
    raise FileNotFoundError(
        f"Reference file not found: {path}\n"
        f"Available files in {DISP_REF_CURVES}:\n"
        + "\n".join(f"  {f}" for f in sorted(available))
    )


def _parse_name_string(name: str) -> Tuple[str, str, str]:
    """
    Parse a shorthand name string into (model, env, mode).

    Accepted formats:
        'ak135'                           → ('ak135', 'earth', 'fundamental')
        'ak135_ocean_deep'                → ('ak135', 'ocean_deep', 'fundamental')
        'ak135_ocean_deep_first'          → ('ak135', 'ocean_deep', 'first')
        'ak135_earth_velocity_first_mode' → ('ak135', 'earth', 'first')
        'ak135_ocean_shallow_velocity_fundamental_mode.txt'  → full stem
    """
    # strip extension
    stem = os.path.splitext(os.path.basename(name))[0]

    # remove '_velocity_..._mode' suffix if present (full filename stem)
    stem = re.sub(r'_velocity_(fundamental|first)_mode$', '', stem)

    # extract trailing mode word
    mode = 'fundamental'
    for alias, canonical in _MODE_ALIASES.items():
        if stem.lower().endswith('_' + alias) or stem.lower() == alias:
            mode = canonical
            stem = stem[:-(len(alias) + 1)] if stem.lower().endswith('_' + alias) else stem
            break

    # what remains should be model + optional env
    # try to match known envs greedily from the end
    env = 'earth'
    norm_stem = _norm(stem)
    for alias, canonical in sorted(_ENV_ALIASES.items(),
                                   key=lambda x: -len(x[0])):
        if norm_stem.endswith(_norm(alias)):
            env  = canonical
            trim = len(alias)
            stem = stem[:-(trim + 1)] if len(stem) > trim else stem
            break

    # whatever is left is the model name
    model = stem.strip('_').lower()
    if not model:
        model = 'ak135'

    return model, env, mode


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_refs() -> List[Tuple[str, str, str]]:
    """
    Return list of available built-in reference models as
    (model, environment, mode) tuples.

    Example output:
        [('ak135', 'earth',              'fundamental'),
         ('ak135', 'earth',              'first'),
         ('ak135', 'ocean_deep',         'fundamental'),
         ...]
    """
    if not os.path.isdir(DISP_REF_CURVES):
        return []

    results = []
    pattern = re.compile(
        r'^(.+?)_(earth|ocean_deep|ocean_intermediate|ocean_shallow)'
        r'_velocity_(fundamental|first)_mode\.txt$'
    )
    for f in sorted(os.listdir(DISP_REF_CURVES)):
        m = pattern.match(f)
        if m:
            results.append((m.group(1), m.group(2), m.group(3)))
    return results


def load_ref(name:  Optional[str] = None,
             model: Optional[str] = None,
             env:   Optional[str] = None,
             mode:  Optional[str] = None) -> dict:
    """
    Load a reference dispersion model.

    Call signatures
    ---------------
    load_ref('ak135')
        → ak135_earth_velocity_fundamental_mode.txt

    load_ref('ak135_ocean_deep')
        → ak135_ocean_deep_velocity_fundamental_mode.txt

    load_ref('ak135_ocean_deep_first')
        → ak135_ocean_deep_velocity_first_mode.txt

    load_ref(model='ak135', env='ocean_shallow', mode='first')
        → ak135_ocean_shallow_velocity_first_mode.txt

    load_ref('/full/path/to/my_model.csv')
        → loads that file directly (must have standard column headers)

    Parameters
    ----------
    name  : shorthand string OR full path to a custom file
    model : model name, e.g. 'ak135'
    env   : environment — 'earth' | 'ocean_deep' | 'ocean_intermediate' | 'ocean_shallow'
    mode  : 'fundamental' (default) | 'first'

    Returns
    -------
    dict with keys:
        'name'   : file stem
        'model'  : model string
        'env'    : environment string
        'mode'   : mode string
        'period' : ndarray [s]
        'phR'    : Rayleigh phase velocity [km/s]
        'phL'    : Love phase velocity     [km/s]
        'grR'    : Rayleigh group velocity [km/s]
        'grL'    : Love group velocity     [km/s]
    """
    # ---- resolve file path ----
    if name is not None and os.path.isfile(name):
        # user supplied a full path directly
        csv_path   = name
        model_out  = model or 'custom'
        env_out    = env   or 'custom'
        mode_out   = mode  or 'custom'
        name_out   = os.path.splitext(os.path.basename(name))[0]

    elif name is not None:
        # parse shorthand string, then override with explicit kwargs if given
        model_p, env_p, mode_p = _parse_name_string(name)
        model_out = model or model_p
        env_out   = _ENV_ALIASES.get(_norm(env),  env_p)  if env  else env_p
        mode_out  = _MODE_ALIASES.get(_norm(mode), mode_p) if mode else mode_p
        csv_path  = _resolve_file(model_out, env_out, mode_out)
        name_out  = _build_filename(model_out, env_out, mode_out)

    else:
        # all explicit kwargs
        model_out = (model or 'ak135').lower()
        env_out   = _ENV_ALIASES.get(_norm(env or 'earth'), 'earth')
        mode_out  = _MODE_ALIASES.get(_norm(mode or 'fundamental'), 'fundamental')
        csv_path  = _resolve_file(model_out, env_out, mode_out)
        name_out  = _build_filename(model_out, env_out, mode_out)

    # ---- parse file ----
    # try comma-delimited first, then whitespace
    data = None
    for delim in (',', None):
        try:
            kw = dict(dtype=float, encoding='utf-8')
            if delim:
                kw['delimiter'] = delim
            data = np.genfromtxt(csv_path, names=True, **kw)
            if data.dtype.names and len(data) > 1:
                break
        except Exception:
            data = None

    if data is None or not data.dtype.names:
        raise ValueError(
            f"Cannot parse reference file '{csv_path}'.\n"
            f"Expected a header row with column names including 'period', "
            f"'phase_velocity_rayleigh', 'group_velocity_rayleigh', etc."
        )

    # normalise column names
    cols = {c.strip().lower().replace(' ', '_'): data[c]
            for c in data.dtype.names}

    def _get(*keys):
        for k in keys:
            if k in cols:
                return cols[k]
        return np.zeros(len(next(iter(cols.values()))))

    period = _get('period')
    phR    = _get('phase_velocity_rayleigh', 'phase_vel_rayleigh', 'phr', 'cr', 'c_r')
    phL    = _get('phase_velocity_love',     'phase_vel_love',     'phl', 'cl', 'c_l')
    grR    = _get('group_velocity_rayleigh', 'group_vel_rayleigh', 'grr', 'ur', 'u_r')
    grL    = _get('group_velocity_love',     'group_vel_love',     'grl', 'ul', 'u_l')

    idx = np.argsort(period)
    return dict(
        name   = name_out,
        model  = model_out,
        env    = env_out,
        mode   = mode_out,
        period = period[idx],
        phR    = phR[idx],
        phL    = phL[idx],
        grR    = grR[idx],
        grL    = grL[idx],
    )


def ref_for_aftan(name:  Optional[str] = None,
                  model: Optional[str] = None,
                  env:   Optional[str] = None,
                  mode:  Optional[str] = None,
                  wave:  str = 'rayleigh',
                  tmin:  float = 1.0,
                  tmax:  float = 300.0) -> dict:
    """
    Load a reference model and return arrays trimmed to [tmin, tmax],
    ready to pass directly to run_aftan() / aftanpg() / aftanipg() / cwt_ftan().

    Parameters
    ----------
    name  : shorthand string or full path  (see load_ref)
    model : model name  (alternative to name)
    env   : environment (alternative to name)
    mode  : 'fundamental' (default) or 'first'
    wave  : 'rayleigh' (default) or 'love'
    tmin  : minimum period [s]
    tmax  : maximum period [s]

    Returns
    -------
    dict with keys:
        'phprper' : period array [s]
        'phprvel' : phase velocity [km/s]
        'pred'    : ndarray (N, 2) [period, group_vel] for aftanipg / PMF
        'ref'     : full raw dict from load_ref()
    """
    ref  = load_ref(name=name, model=model, env=env, mode=mode)
    per  = ref['period']
    wave = wave.lower()

    if wave in ('rayleigh', 'r', 'z', 'zz', 'vertical'):
        ph = ref['phR']
        gr = ref['grR']
    elif wave in ('love', 'l', 't', 'tt', 'transverse'):
        ph = ref['phL']
        gr = ref['grL']
    else:
        raise ValueError(f"wave must be 'rayleigh' or 'love', got '{wave}'")

    mask = (per >= tmin) & (per <= tmax)
    if mask.sum() < 2:
        raise ValueError(
            f"Fewer than 2 reference points in T=[{tmin},{tmax}] s "
            f"for model '{ref['name']}'. Check tmin/tmax."
        )

    per_m = per[mask]
    ph_m  = ph[mask]
    gr_m  = gr[mask]

    # strip NaN rows — some files have missing Love-wave values at long periods
    valid_ph = np.isfinite(ph_m)
    valid_gr = np.isfinite(gr_m)
    valid    = valid_ph & valid_gr

    if valid.sum() < 2:
        raise ValueError(
            f"Fewer than 2 finite reference points in T=[{tmin},{tmax}] s "
            f"for model '{ref['name']}', wave='{wave}'. "
            f"Check that the model file has {wave} columns in this period range."
        )

    per_m = per_m[valid]
    ph_m  = ph_m[valid]
    gr_m  = gr_m[valid]

    print(f"[ant_refs] Loaded: {ref['name']}"
          f"  wave={wave}  T={per_m[0]:.1f}-{per_m[-1]:.1f} s"
          f"  ({len(per_m)} points)")

    return dict(
        phprper = per_m,
        phprvel = ph_m,
        pred    = np.column_stack([per_m, gr_m]),
        ref     = ref,
    )
