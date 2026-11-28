# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: taup_tt.py
# Program: surfQuake & ISP
# Purpose: Travel-time table for the associator REAL, computed with TauP
# --------------------------------------------------------------------

import hashlib
import json
import math
import os
import time

import numpy as np
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model

from surfquakecore import TT_DB_PATH

# REAL reads at most this number of lines of the travel-time table (REAL.c: Ntb = 20000)
REAL_MAX_TT_LINES = 20000


class create_tt_db:

    def __init__(self, model_path: str = None, build_dir: str = None):
        """
        Travel-time table for REAL.

        model_path: 1D velocity model in TauP format (.nd or .tvel). If None, the model distributed
                    with surfquake is used (real/tt_db/mymodel.nd).
        build_dir:  folder where TauP builds the model (.npz). If None, ObsPy's default folder is used.

        # when you prepare the model, please consider interpolating
        # the velocity model above the majority of seismicity (e.g., a few km/layer)
        # TauP, velest, and hypoinverse don't like low velocity layers...
        """
        self.model_path = os.path.abspath(model_path) if model_path else TT_DB_PATH
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Velocity model not found: {self.model_path}")
        self.build_dir = build_dir
        self.model = None

    def _load_model(self):
        name = os.path.splitext(os.path.basename(self.model_path))[0]
        if self.build_dir:
            os.makedirs(self.build_dir, exist_ok=True)
            build_taup_model(self.model_path, output_folder=self.build_dir)
            self.model = TauPyModel(model=os.path.join(self.build_dir, name + ".npz"))
        else:
            build_taup_model(self.model_path)
            self.model = TauPyModel(model=name)

    @staticmethod
    def _n_steps(value_range, step, label):
        n = int(round(value_range / step))
        if n < 1 or not math.isclose(n * step, value_range, rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"The travel-time {label} range ({value_range}) must be a multiple "
                             f"of its grid size ({step})")
        return n

    def _model_hash(self):
        with open(self.model_path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()

    def run_tt_db(self, ttime: str, dist=1.4, depth=20, ddist=0.01, ddep=1):
        """
        Write the table ttdb.txt in the folder ttime, in the format and order that REAL expects:
        for each depth 0, ddep, ..., depth (km) and each distance ddist, 2*ddist, ..., dist (deg).

        The table must have the same ranges and grid sizes that REAL receives in its -G option
        (TRAVEL_TIME_GRID_SEARCH section of real_config.ini), because REAL finds the line of a
        distance and depth from them. If a table with the same model and grid already exists in
        ttime, it is reused.
        """
        nh = self._n_steps(dist, ddist, "horizontal")
        nv = self._n_steps(depth, ddep, "depth") + 1
        n_lines = nh * nv
        if n_lines > REAL_MAX_TT_LINES:
            raise ValueError(f"The travel-time table would have {n_lines} lines, but REAL reads at most "
                             f"{REAL_MAX_TT_LINES}. Increase the grid sizes or reduce the ranges of "
                             f"TRAVEL_TIME_GRID_SEARCH.")

        table = os.path.join(ttime, "ttdb.txt")
        info_file = os.path.join(ttime, "ttdb_info.json")
        info = {"model": self.model_path, "model_sha1": self._model_hash(),
                "dist": dist, "depth": depth, "ddist": ddist, "ddep": ddep}

        if os.path.isfile(table) and os.path.isfile(info_file):
            try:
                with open(info_file) as f:
                    if json.load(f) == info:
                        print(f"[INFO] Reusing travel-time table {table} ({os.path.basename(self.model_path)})")
                        return table
            except (OSError, ValueError):
                pass

        print(f"[INFO] Computing travel-time table with {os.path.basename(self.model_path)}: "
              f"0-{dist} deg every {ddist} deg, 0-{depth} km every {ddep} km ({n_lines} lines)")
        if self.model is None:
            self._load_model()

        t0 = time.time()
        with open(table, "w") as f:
            for idep in range(nv):
                dep = round(idep * ddep, 6)
                for idist in range(1, nh + 1):
                    gdist = round(idist * ddist, 6)
                    arrivals = self.model.get_travel_times(source_depth_in_km=dep, distance_in_degree=gdist,
                                                           phase_list=["P", "p", "S", "s"])
                    p_arr = next((a for a in arrivals if a.name in ("P", "p")), None)
                    s_arr = next((a for a in arrivals if a.name in ("S", "s")), None)
                    if p_arr is None or s_arr is None:
                        raise ValueError(f"No P or S travel time at dist={gdist:.2f} deg, depth={dep:.2f} km. "
                                         f"Most likely a low-velocity layer in the model {self.model_path}")

                    p_ray_param = p_arr.ray_param * 2 * np.pi / 360
                    s_ray_param = s_arr.ray_param * 2 * np.pi / 360
                    p_hslowness = -1 * (p_ray_param / 111.19) / math.tan(p_arr.takeoff_angle * math.pi / 180)
                    s_hslowness = -1 * (s_ray_param / 111.19) / math.tan(s_arr.takeoff_angle * math.pi / 180)

                    f.write(f"{gdist} {dep} {p_arr.time} {s_arr.time} {p_ray_param} {s_ray_param} "
                            f"{p_hslowness} {s_hslowness} {p_arr.name} {s_arr.name}\n")

        with open(info_file, "w") as f:
            json.dump(info, f, indent=2)
        print(f"[INFO] Travel-time table written in {time.time() - t0:0.1f} s: {table}")
        return table


if __name__ == '__main__':
    tt_db = create_tt_db()
    tt_db.run_tt_db(ttime=".")
