from __future__ import annotations
from typing import Literal, Tuple, List, Optional
import numpy as np
from obspy import Stream, Trace
from obspy.signal.trigger import (
    classic_sta_lta, recursive_sta_lta, trigger_onset
)

class CF_SNR:

    @staticmethod
    def compute_sta_lta_cfs(
        stream: Stream,
        method: Literal["classic", "recursive"] = "classic",
        sta: float = 1.0,               # seconds
        lta: float = 40.0,              # seconds
        preprocess: bool = True,
        bp: Optional[Tuple[float, float]] = None,  # e.g. (1.0, 20.0) Hz
        trigger: bool = False,
        thr_on: float = 10.0,
        thr_off: float = 5.0,
        normalize: bool = False,        # min-max to [0,1] per trace
    ) -> Tuple[Stream, Optional[List[List[Tuple[int, int]]]]]:
        """
        Compute STA/LTA characteristic functions (CF) for each trace in `stream`.

        Returns
        -------
        cf_stream : Stream
            Stream of CF traces (float). Each CF trace has id f"{raw.id}:CF".
        on_off : list of lists of (i_on, i_off) index pairs, or None
            If trigger=True, per-trace onset/offset index pairs (sample indices in CF).
            Order matches `cf_stream`.
        """
        cf_stream = Stream()
        on_off_all = [] if trigger else None

        for tr in stream:
            if tr.stats.npts < 10 or not np.isfinite(tr.data).any():
                continue

            tr_work = tr.copy()
            fs = float(tr_work.stats.sampling_rate)

            # Optional preprocessing for stability
            if preprocess:
                tr_work.detrend("demean")
                tr_work.detrend("linear")
                tr_work.taper(0.02, type="hann")
                if bp is not None:
                    f1, f2 = bp
                    # guard against invalid band edges
                    if f1 is not None and f2 is not None and 0 < f1 < f2 < (fs / 2.0):
                        tr_work.filter("bandpass", freqmin=f1, freqmax=f2, corners=4, zerophase=True)

            nsta = max(1, int(round(sta * fs)))
            nlta = max(nsta + 1, int(round(lta * fs)))

            x = tr_work.data.astype(float)

            if method == "classic":
                cf = classic_sta_lta(x, nsta, nlta)
            elif method == "recursive":
                # recursive alpha ~ 0.995 typical; use defaults in ObsPy via nsta/nlta equivalent
                cf = recursive_sta_lta(x, nsta, nlta)
            else:
                raise ValueError("method must be 'classic' or 'recursive'.")

            # Optional normalization (helps plotting; keeps shape only)
            if normalize and np.isfinite(cf).any():
                cmin = np.nanmin(cf)
                cmax = np.nanmax(cf)
                if np.isfinite(cmin) and np.isfinite(cmax) and cmax > cmin:
                    cf = (cf - cmin) / (cmax - cmin)

            # Build CF trace (align exactly with raw timebase)
            cf_tr = Trace(
                data=cf.astype(np.float32, copy=False),
                header=tr_work.stats.copy()
            )
            cf_tr.stats.channel = tr_work.stats.channel  # keep same chan code
            cf_tr.stats._cf_of = tr.id                   # provenance
            #cf_tr.id = f"{tr.id}:CF"

            cf_stream.append(cf_tr)

            if trigger:
                on_off = trigger_onset(cf, thr_on, thr_off)
                on_off_all.append(on_off)

        return cf_stream, on_off_all