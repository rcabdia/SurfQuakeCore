# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: magnitudes/source_inputs.py
# Program: surfQuake & ISP
# Purpose: Optional inputs of Automag: layered earth model and focal mechanisms
# Author: Roberto Cabieces
#  Email: rcabdia@roa.es
# --------------------------------------------------------------------
"""
Readers for the optional inputs of ``Automag``:

* ``read_earth_model``   : layered model (top depth, vp, vs, density) -> SourceSpec config values.
* ``read_mechanisms``    : table with one focal mechanism per event, identified by ``date_id`` (origin time).
* ``MechanismTable``     : matches a hypocenter origin time to a mechanism (nearest within a tolerance).

The hypocenter (NonLinLoc ``.hyp`` or events file) always leads: the mechanism is looked up from its origin time,
never the other way round.
"""
import numpy as np
import pandas as pd
from obspy import UTCDateTime


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _read_table(path: str) -> pd.DataFrame:
    """Read a csv/txt table: delimiter auto-detected (';' ',' tab or blanks), '#' comments, case-insensitive names."""
    df = pd.read_csv(path, sep=None, engine="python", comment="#", skipinitialspace=True)
    if df.shape[1] < 2:  # the sniffer fails with blank-separated tables
        df = pd.read_csv(path, sep=r"\s+", engine="python", comment="#")
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _pick_column(df: pd.DataFrame, candidates, what: str, path: str) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"{path}: no column for {what}. Accepted names: {list(candidates)}. "
                     f"Columns found: {list(df.columns)}")


def _parse_time(value) -> UTCDateTime:
    text = str(value).strip()
    try:
        return UTCDateTime(text)
    except Exception:
        # best effort for other layouts, e.g. "10/15/2026, 20:34:45.843101" (month first)
        return UTCDateTime(pd.to_datetime(text).to_pydatetime())


# ---------------------------------------------------------------------------
# earth model
# ---------------------------------------------------------------------------
_DEPTH_NAMES = ("top_depth_km", "top_depth", "layer_top_depth", "depth_km", "depth", "z")
_VP_NAMES = ("vp", "vp_km_s", "vp_kms")
_VS_NAMES = ("vs", "vs_km_s", "vs_kms")
_RHO_NAMES = ("rho", "density", "rho_kg_m3", "density_kg_m3")


def read_earth_model(path: str) -> dict:
    """
    Read a layered earth model. One row per layer, columns:

        top_depth_km;vp;vs;rho
        0.0;5.5;3.2;2500
        3.0;6.1;3.5;2700

    top_depth_km: depth (km, positive down) of the top of the layer. The layer applies down to the next one.
    vp, vs: km/s.  rho: kg/m3 (mandatory).

    :return: dict of SourceSpec config values to pass as ``config_overrides`` to ``configure``. SourceSpec
        picks the layer at the depth of each hypocenter. Values at the stations (``*_stations``) are those of the
        top layer; otherwise SourceSpec would use the source-depth values also at the stations.
    """
    df = _read_table(path)
    depth_c = _pick_column(df, _DEPTH_NAMES, "layer top depth (km)", path)
    vp_c = _pick_column(df, _VP_NAMES, "Vp (km/s)", path)
    vs_c = _pick_column(df, _VS_NAMES, "Vs (km/s)", path)
    rho_c = _pick_column(df, _RHO_NAMES, "density (kg/m3)", path)

    try:
        table = df[[depth_c, vp_c, vs_c, rho_c]].astype(float)
    except ValueError as error:
        raise ValueError(f"{path}: non numeric value in the model ({error})") from error
    if table.isnull().values.any():
        raise ValueError(f"{path}: empty or non numeric values in the model")
    if len(table) == 0:
        raise ValueError(f"{path}: the model has no layers")

    table = table.sort_values(depth_c).reset_index(drop=True)
    depth, vp, vs, rho = (table[c].to_numpy() for c in (depth_c, vp_c, vs_c, rho_c))

    if len(np.unique(depth)) != len(depth):
        raise ValueError(f"{path}: two layers have the same top depth")
    if (vp <= 0).any() or (vs <= 0).any() or (rho <= 0).any():
        raise ValueError(f"{path}: Vp, Vs and density must be positive")
    if (vp <= vs).any():
        raise ValueError(f"{path}: Vp must be larger than Vs in every layer")
    if vp.max() > 20:
        raise ValueError(f"{path}: Vp up to {vp.max()} looks like m/s; velocities must be in km/s")
    if rho.max() < 100:
        raise ValueError(f"{path}: density up to {rho.max()} looks like g/cm3; density must be in kg/m3")

    return {
        "layer_top_depths": depth.tolist(),
        "vp_source": vp.tolist(),
        "vs_source": vs.tolist(),
        "rho_source": rho.tolist(),
        "vp_stations": float(vp[0]),
        "vs_stations": float(vs[0]),
        "rho_stations": float(rho[0]),
    }


def describe_earth_model(model: dict) -> str:
    rows = zip(model["layer_top_depths"], model["vp_source"], model["vs_source"], model["rho_source"])
    return "Earth model (top_depth_km, vp, vs, rho): " + "  |  ".join(
        f"{d:g}, {p:g}, {s:g}, {r:g}" for d, p, s, r in rows)


# ---------------------------------------------------------------------------
# focal mechanisms
# ---------------------------------------------------------------------------
_TIME_NAMES = ("date_id", "origin_time", "time", "datetime", "date")
# plane 1 names used by the moment tensor inversion output come first
_STRIKE_NAMES = ("plane_1_strike", "strike", "strike_1", "strike1")
_DIP_NAMES = ("plane_1_dip", "dip", "dip_1", "dip1")
_RAKE_NAMES = ("plane_1_slip_rake", "plane_1_rake", "rake", "slip_rake", "rake_1", "rake1")


def read_mechanisms(path: str) -> pd.DataFrame:
    """
    Read a table with one focal mechanism (strike, dip, rake in degrees) per row, identified by ``date_id``.

        date_id;plane_1_strike;plane_1_dip;plane_1_slip_rake
        2022-02-01 02:03:00.276594;0.7032;29.2967;22.4201

    Any other column (e.g. the full output of the moment tensor inversion) is ignored. Plane 1 is used: the
    double-couple radiation pattern is the same for both nodal planes. Rows with invalid values are skipped
    with a message.

    :return: DataFrame with columns ``row`` (line in the file), ``time`` (UTCDateTime), ``strike``, ``dip``, ``rake``.
    """
    df = _read_table(path)
    time_c = _pick_column(df, _TIME_NAMES, "event time (date_id)", path)
    strike_c = _pick_column(df, _STRIKE_NAMES, "strike", path)
    dip_c = _pick_column(df, _DIP_NAMES, "dip", path)
    rake_c = _pick_column(df, _RAKE_NAMES, "rake", path)

    rows = []
    for i, rec in df.iterrows():
        line = i + 2  # header is line 1
        try:
            t = _parse_time(rec[time_c])
            strike, dip, rake = float(rec[strike_c]), float(rec[dip_c]), float(rec[rake_c])
        except Exception as error:
            print(f"{path}, line {line}: skipped, cannot read the row ({error})")
            continue
        if not all(np.isfinite([strike, dip, rake])) or not 0.0 <= dip <= 90.0:
            print(f"{path}, line {line}: skipped, invalid strike/dip/rake: {strike}, {dip}, {rake}")
            continue
        rows.append({"row": line, "time": t, "strike": strike, "dip": dip, "rake": rake})

    if not rows:
        raise ValueError(f"{path}: no valid focal mechanism found")
    return pd.DataFrame(rows)


class MechanismTable:
    """Finds the mechanism of a hypocenter: nearest ``date_id`` to its origin time within ``tolerance`` seconds."""

    def __init__(self, table: pd.DataFrame, tolerance: float = 2.0):
        if tolerance < 0:
            raise ValueError("The mechanism time tolerance must be >= 0 s")
        self.table = table.reset_index(drop=True)
        self.tolerance = float(tolerance)
        self._times = np.array([t.timestamp for t in self.table["time"]])
        self.used = {}  # file row -> list of origin times that matched it

    def match(self, origin_time: UTCDateTime):
        """:return: (strike, dip, rake) of the closest mechanism within the tolerance, or None."""
        diff = np.abs(self._times - origin_time.timestamp)
        inside = np.flatnonzero(diff <= self.tolerance)
        if inside.size == 0:
            return None
        best = inside[np.argmin(diff[inside])]
        if inside.size > 1:
            print(f"{origin_time}: {inside.size} mechanisms within {self.tolerance:g} s, "
                  f"using the closest ({diff[best]:.3f} s away)")
        rec = self.table.iloc[best]
        self.used.setdefault(int(rec["row"]), []).append(origin_time)
        return float(rec["strike"]), float(rec["dip"]), float(rec["rake"])

    def report(self) -> list:
        """Lines for the console: mechanisms never used, and mechanisms used by more than one hypocenter."""
        lines = []
        for _, rec in self.table.iterrows():
            row = int(rec["row"])
            users = self.used.get(row, [])
            if not users:
                lines.append(f"mechanism line {row} ({rec['time']}) matched no hypocenter")
            elif len(users) > 1:
                lines.append(f"mechanism line {row} ({rec['time']}) matched {len(users)} hypocenters: "
                             f"{', '.join(str(u) for u in users)}")
        return lines
