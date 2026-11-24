# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: cc_picker.py
# Program: surfQuake & ISP
# Date: November 2026
# Purpose: Cross-correlation (template matching) picker.
#          Output identical to the PhaseNet picker (see phasenet/pick_output.py).
#
#   mode="simple"  : single-channel templates, raw correlation threshold
#                    (obspy.signal.cross_correlation.correlation_detector).
#   mode="network" : multichannel correlation detector of
#                    Gibbons & Ringdal (2006), GJI 165, 149-166:
#                    scaled network correlation coefficient (eqs. 6-10),
#                    per-channel picks from local maxima, ratio test (eq. 30),
#                    optional plane-wave slowness test (eq. 28-29).
#
# Author: Roberto Cabieces & Thiago C. Junqueira
#  Email: rcabdia@roa.es
# --------------------------------------------------------------------

"""
Templates
---------
The user stores the template waveforms in one folder (any format ObsPy can
read: MSEED, SAC, H5, ...). The phase and onset time of each template come
from one of two sources, checked in this order:

1) ``templates_info.txt`` in the folder (whitespace or comma separated,
   '#' = comment)::

    # file                      phase  pick_time                  event
    WM.EMAL..HHZ.mseed          P      2024-07-26T06:42:31.520    EV1
    WM.EMAL..HHN.mseed          S      2024-07-26T06:42:34.100    EV1
    CA.ARBS..HHZ.mseed          P      2024-07-26T06:42:33.050    EV1
    WM.EMAL..HHZ.20240901.mseed P      2024-09-01T12:00:05.000

   - file      : path relative to the templates folder. One trace per file (the
                 first trace is used), already cut around the phase.
   - phase     : P or S.
   - pick_time : absolute UTC onset time, inside the template window.
   - event     : optional, network mode only: rows with the same event and
                 phase form one multichannel template (one master event).
                 Rows without event are single-channel templates.

2) If there is no ``templates_info.txt``: the SurfQuake pick header,
   ``tr.stats.picks`` = [{"time": timestamp, "phase": "P", ...}, ...], as
   written by the plotting tool (``write`` command -> H5 files). Every file in
   the folder and its subfolders is read; every trace with a P and/or S pick
   gives one template per phase (phase names like Pg, Pn, Sg are reduced to
   their first letter; other phases are ignored). The trace is cut to
   [pick - pre_pick, pick + post_pick], so full-length traces can be used.
   Network mode: all templates in the same (sub)folder are one master event,
   so keep each master event in its own subfolder.

In both cases the template id NET.STA.CHA (location ignored) must match the
data channel it is correlated with, and in network mode the templates must
keep their true start times (the time offsets between channels are used).
"""

import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime
from obspy.signal.cross_correlation import correlation_detector

from surfquakecore.config.global_config import read_stream
from surfquakecore.phasenet.pick_output import best_day, ensure_24, make_pick_row, picks_to_dataframe

INFO_FILE = "templates_info.txt"
AMP_WINDOW = {"P": 10.0, "S": 5.0}  # s after the pick, same windows as PhaseNet
AMP_SCALE = 2080 * 25  # same amplitude scaling as PhasenetUtils.split_picks
DETECTION_COLUMNS = ["time", "phase", "template", "n_channels", "n_picks", "cc_mean", "scaled_cc",
                     "ratio", "slowness", "magnitude"]


def channel_key(stats):
    """NET.STA.CHA, location code ignored."""
    return f"{stats.network}.{stats.station}.{stats.channel}"


@dataclass
class TemplateGroup:
    name: str
    phase: str
    traces: dict = field(default_factory=dict)  # channel key -> processed template Trace
    picks: dict = field(default_factory=dict)  # channel key -> UTCDateTime of the onset

    @property
    def t_ref(self):
        return min(tr.stats.starttime for tr in self.traces.values())


class CCPicker:
    def __init__(self, files, template_dir, mode="simple", threshold=None, fmin=2.0, fmax=15.0,
                 distance=2.0, scc_a=1.0, scc_b=2.5, min_ratio=0.58, min_channel_cc=0.3,
                 lag_win=0.5, min_channels=3, inventory=None, max_slowness=None, master_mag=None,
                 pre_pick=1.0, post_pick=4.0, amplitude=True, output=None, verbose=False):
        """
        Cross-correlation picker with the same output as PhasenetISP + PhasenetUtils.split_picks.

        :param files: project dictionary (SurfProject.project): key -> [[file path, stats], ...]
        :param template_dir: folder with the template files and templates_info.txt
        :param mode: "simple" or "network" (Gibbons & Ringdal 2006)
        :param threshold: simple -> raw correlation (default 0.7);
                          network -> scaled network correlation coefficient (default 6.0)
        :param fmin, fmax: bandpass applied to data and templates (Hz)
        :param distance: minimum time (s) between two detections of the same template / picks
                         of the same channel and phase
        :param scc_a, scc_b: network mode, inner/outer limits (s) of the background windows
                             [t-b, t-a] and [t+a, t+b] of the scaled coefficient
        :param min_ratio: network mode, minimum C_A / mean(C_j^L) (eq. 30 of the paper)
        :param min_channel_cc: network mode, minimum channel correlation to write a pick
        :param lag_win: network mode, half window (s) to search each channel's local maximum
        :param min_channels: network mode, minimum channels of a template present in the data
        :param inventory: optional ObsPy Inventory, enables the slowness test (small arrays)
        :param max_slowness: maximum |s| (s/km) of the plane wave fitted to the channel lags
        :param master_mag: optional magnitude of the master events -> magnitude per detection
        :param pre_pick, post_pick: header-pick templates only (no templates_info.txt): seconds kept
                                    before and after the pick when cutting the template
        :param amplitude: compute the amplitude column (otherwise 0)
        :param output: directory to create (optional)
        :param verbose: print progress
        """
        if mode not in ("simple", "network"):
            raise ValueError("mode must be 'simple' or 'network'")
        if not 0 < scc_a < scc_b:
            raise ValueError("scc_a and scc_b must satisfy 0 < scc_a < scc_b")

        self.files = files
        self.template_dir = template_dir
        self.mode = mode
        self.threshold = threshold if threshold is not None else (0.7 if mode == "simple" else 6.0)
        self.fmin, self.fmax = fmin, fmax
        self.distance = distance
        self.scc_a, self.scc_b = scc_a, scc_b
        self.min_ratio = min_ratio
        self.min_channel_cc = min_channel_cc
        self.lag_win = lag_win
        self.min_channels = min_channels
        self.inventory = inventory
        self.max_slowness = max_slowness
        self.master_mag = master_mag
        self.pre_pick, self.post_pick = pre_pick, post_pick
        self.amplitude = amplitude
        self.verbose = verbose

        self.n_files = 0
        self.detections = pd.DataFrame(columns=DETECTION_COLUMNS)

        if output is not None:
            os.makedirs(output, exist_ok=True)

        self.groups = self.load_templates()

    # ------------------------------------------------------------------ templates
    def _preprocess(self, st):
        st.detrend("demean")
        st.detrend("linear")
        st.taper(max_percentage=0.05, type="hann", max_length=5.0)
        st.filter("bandpass", freqmin=self.fmin, freqmax=self.fmax, corners=4, zerophase=True)
        return st

    @staticmethod
    def _read_template_file(path):
        # format auto-detection: templates may be in another format than the data (e.g. H5)
        return read_stream(path, format=None)

    @staticmethod
    def _phase_letter(phase):
        """P, Pg, Pn, P1 -> 'P'; S, Sg, Sn -> 'S'; anything else -> None."""
        phase = str(phase or "").strip().upper()
        return phase[0] if phase[:1] in ("P", "S") else None

    def _entries_from_info_file(self, info_path):
        """Templates described in templates_info.txt -> list of (trace, phase, pick, event, label)."""
        entries = []
        with open(info_path) as f:
            for n, line in enumerate(f, 1):
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                cols = line.replace(",", " ").split()
                if cols[0].lower() == "file":  # header line
                    continue
                if len(cols) < 3:
                    raise ValueError(f"{INFO_FILE} line {n}: expected 'file phase pick_time [event]'")
                phase = cols[1].upper()
                if phase not in ("P", "S"):
                    raise ValueError(f"{INFO_FILE} line {n}: phase must be P or S, got {cols[1]}")
                st = self._read_template_file(os.path.join(self.template_dir, cols[0]))
                if len(st) > 1:
                    warnings.warn(f"{cols[0]} has {len(st)} traces, only the first one is used")
                entries.append((st[0], phase, UTCDateTime(cols[2]), cols[3] if len(cols) > 3 else None, cols[0]))
        return entries

    def _entries_from_headers(self):
        """Templates from the SurfQuake pick header tr.stats.picks (e.g. H5 files written by the plot tool)."""
        entries = []
        for root, dirs, names in os.walk(self.template_dir):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            rel = os.path.relpath(root, self.template_dir)
            event = "master" if rel == "." else rel.replace(os.sep, "/")
            for name in sorted(names):
                if name.startswith("."):
                    continue
                path = os.path.join(root, name)
                try:
                    st = self._read_template_file(path)
                except Exception:
                    if self.verbose:
                        print(f"[ccpick] skipping {path}: not a waveform file")
                    continue
                for tr in st:
                    seen = set()
                    for p in tr.stats.get("picks", []) or []:
                        phase = self._phase_letter(p.get("phase"))
                        if phase is None or p.get("time") is None:
                            continue
                        if phase in seen:
                            warnings.warn(f"{name} {tr.id}: several {phase} picks, only the first one is used")
                            continue
                        seen.add(phase)
                        pick = UTCDateTime(p["time"])
                        cut = tr.slice(pick - self.pre_pick, pick + self.post_pick)
                        entries.append((cut, phase, pick, event, f"{os.path.normpath(os.path.join(rel, name))}:{tr.id}:{phase}"))
        return entries

    def load_templates(self):
        info_path = os.path.join(self.template_dir, INFO_FILE)
        if os.path.isfile(info_path):
            source, entries = INFO_FILE, self._entries_from_info_file(info_path)
        else:
            source, entries = "trace headers (stats.picks)", self._entries_from_headers()
        if not entries:
            raise ValueError(f"no templates found in {self.template_dir}: add a {INFO_FILE}, or use waveform "
                             f"files with P/S picks in the trace header (stats.picks, e.g. H5 written by the "
                             f"SurfQuake plot tool)")

        groups = {}
        for tr, phase, pick, event, label in entries:
            if not tr.stats.starttime <= pick <= tr.stats.endtime:
                raise ValueError(f"{label}: pick {pick} is outside the template window "
                                 f"[{tr.stats.starttime}, {tr.stats.endtime}]")
            tr = self._preprocess(Stream([tr.copy()]))[0]
            for key in ("picks", "processing"):
                tr.stats.pop(key, None)
            key = channel_key(tr.stats)

            # simple mode: every entry is a template; network mode: group by (event, phase)
            name = label if (self.mode == "simple" or event is None) else event
            group = groups.setdefault((name, phase), TemplateGroup(name=name, phase=phase))
            if key in group.traces:
                raise ValueError(f"template '{name}' ({phase}) has two traces for channel {key}; "
                                 f"keep each master event in its own subfolder or event")
            group.traces[key] = tr
            group.picks[key] = pick

        print(f"[ccpick] {len(entries)} template trace(s) from {source} -> {len(groups)} template(s)")
        if self.verbose:
            for g in groups.values():
                print(f"[ccpick] template {g.name} ({g.phase}): {len(g.traces)} channel(s)")
        return list(groups.values())

    @staticmethod
    def _matched_template(tmpl_tr, data_tr):
        """Copy of the template with the data location code and sampling rate."""
        t = tmpl_tr.copy()
        t.stats.location = data_tr.stats.location
        if t.stats.sampling_rate != data_tr.stats.sampling_rate:
            t.resample(data_tr.stats.sampling_rate)
        return t

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _highpass_copy(st):
        raw = st.copy()
        raw.detrend("demean")
        raw.detrend("linear")
        raw.filter("highpass", freq=0.5)
        return raw

    def _amplitude(self, raw_tr, pick, phase):
        if not self.amplitude or raw_tr is None:
            return 0.0
        seg = raw_tr.slice(pick, pick + AMP_WINDOW[phase])
        return float(np.max(np.abs(seg.data))) * AMP_SCALE if seg.stats.npts else 0.0

    def _scaled_similarity(self, store):
        """similarity_func for correlation_detector: scaled network coefficient (eqs. 6-8, 10)."""
        a, b = self.scc_a, self.scc_b

        def func(ccs):
            store["ccs"] = ccs
            dt = ccs[0].stats.delta
            ca = np.mean([np.nan_to_num(tr.data) for tr in ccs], axis=0)
            n = len(ca)
            na, nb = max(1, int(round(a / dt))), max(2, int(round(b / dt)))
            c = np.concatenate([[0.0], np.cumsum(ca.astype(np.float64) ** 2)])
            i = np.arange(n)

            def window(lo, hi):  # sum and count of ca**2 over samples [lo, hi)
                lo, hi = np.clip(lo, 0, n), np.clip(hi, 0, n)
                return c[hi] - c[lo], hi - lo

            s1, n1 = window(i - nb, i - na + 1)
            s2, n2 = window(i + na, i + nb + 1)
            cnt = np.maximum(n1 + n2, 1)
            rms = np.sqrt((s1 + s2) / cnt)
            positive = rms[rms > 0]
            floor = 0.1 * np.median(positive) if positive.size else 1.0
            scaled = ca / np.maximum(rms, floor)

            out = ccs[0].copy()
            out.stats.channel = "SCC"
            out.data = scaled
            store["ca"] = ca
            return out

        return func

    def _plane_wave_slowness(self, ids, lags):
        """|s| (s/km) of the plane wave fitted to the channel lags (eq. 28), None if not possible."""
        coords = []
        for tr_id, lag in zip(ids, lags):
            try:
                c = self.inventory.get_coordinates(tr_id)
                coords.append((c["latitude"], c["longitude"], lag))
            except Exception:
                continue
        if len({(la, lo) for la, lo, _ in coords}) < 3:
            return None
        arr = np.array(coords)
        lat0, lon0 = arr[:, 0].mean(), arr[:, 1].mean()
        x = (arr[:, 1] - lon0) * 111.19 * np.cos(np.radians(lat0))
        y = (arr[:, 0] - lat0) * 111.19
        A = np.column_stack([x, y, np.ones_like(x)])
        sol, *_ = np.linalg.lstsq(A, arr[:, 2], rcond=None)
        return float(np.hypot(sol[0], sol[1]))

    def _dedup(self, rows):
        """Keep the best pick per channel and phase inside `distance` seconds."""
        best = {}
        for r in sorted(rows, key=lambda r: -r[10]):  # r[10] = weight
            key = (r[5], r[6], r[13], r[12])  # net, station, channel, phase
            t = UTCDateTime(r[9])
            kept = best.setdefault(key, [])
            if all(abs(t - k) > self.distance for k in kept):
                kept.append(t)
                yield r

    # ------------------------------------------------------------------ main
    def pick(self):
        """Run the picker. Returns a DataFrame in the PhasenetUtils.split_picks format."""
        rows = self._pick_simple() if self.mode == "simple" else self._pick_network()
        rows = sorted(self._dedup(rows), key=lambda r: r[9])
        return picks_to_dataframe(rows)

    def _iter_files(self):
        for items in self.files.values():
            for fname, stats in items:
                yield fname, stats

    def _pick_simple(self):
        rows = []
        by_key = {}
        for g in self.groups:
            for key in g.traces:
                by_key.setdefault((key, g.phase), []).append(g)

        for fname, _ in self._iter_files():
            self.n_files += 1
            try:
                st = read_stream(fname).merge(fill_value=0)
            except Exception as ex:
                print(f"[ccpick] cannot read {fname}: {ex}")
                continue

            for tr in st:
                key = channel_key(tr.stats)
                if not any((key, ph) in by_key for ph in ("P", "S")):
                    continue
                tr = ensure_24(tr)
                if tr.stats.npts < 2:
                    continue
                raw = self._highpass_copy(Stream([tr]))[0] if self.amplitude else None
                proc = self._preprocess(Stream([tr.copy()]))

                for phase in ("P", "S"):
                    groups = by_key.get((key, phase), [])
                    if not groups:
                        continue
                    templates = [Stream([self._matched_template(g.traces[key], tr)]) for g in groups]
                    dets, _ = correlation_detector(proc, templates, heights=self.threshold,
                                                   distance=self.distance,
                                                   template_times=[g.picks[key] for g in groups],
                                                   template_names=[g.name for g in groups])
                    for d in dets:
                        rows.append(make_pick_row(tr.stats.network, tr.stats.station, tr.stats.channel,
                                                  phase, d["time"], d["similarity"],
                                                  self._amplitude(raw, d["time"], phase)))
                    if self.verbose:
                        print(f"[ccpick] {tr.id} {phase}: {len(dets)} picks")
        return rows

    def _pick_network(self):
        needed = set()
        for g in self.groups:
            needed.update(g.traces)

        # files per UTC day (same day rule as PhaseNet ensure_24)
        days = {}
        for fname, stats in self._iter_files():
            self.n_files += 1
            if channel_key(stats) in needed:
                days.setdefault(best_day(stats.starttime, stats.endtime).timestamp, []).append(fname)

        rows, detections = [], []
        for day_ts in sorted(days):
            day = UTCDateTime(day_ts)
            st = Stream()
            for fname in days[day_ts]:
                try:
                    st += read_stream(fname)
                except Exception as ex:
                    print(f"[ccpick] cannot read {fname}: {ex}")
            if not st:
                continue
            st.merge(fill_value=0)
            for tr in st:
                tr.data = tr.data.astype(np.float32)
            # pad every channel to the full day: correlation_detector cuts all channels
            # to their common overlap, a short station would otherwise shorten the day
            st.trim(day, day + 86400 - st[0].stats.delta, pad=True, fill_value=0, nearest_sample=True)
            raw = self._highpass_copy(st) if self.amplitude else None
            proc = self._preprocess(st)

            for g in self.groups:
                n_rows = len(rows)
                self._detect_group(g, proc, raw, rows, detections)
                if self.verbose:
                    print(f"[ccpick] {day.date} template {g.name} ({g.phase}): {len(rows) - n_rows} picks")

        self.detections = pd.DataFrame(detections, columns=DETECTION_COLUMNS)
        return rows

    def _detect_group(self, g, proc, raw, rows, detections):
        data = Stream([tr for tr in proc if channel_key(tr.stats) in g.traces])
        if len({channel_key(tr.stats) for tr in data}) < min(self.min_channels, len(g.traces)):
            return
        rate = data[0].stats.sampling_rate
        data = Stream([tr if tr.stats.sampling_rate == rate else tr.copy().resample(rate) for tr in data])

        template = Stream()
        for tr in data:
            template += self._matched_template(g.traces[channel_key(tr.stats)], tr)
        t_ref = min(tr.stats.starttime for tr in template)

        store = {}
        dets, _ = correlation_detector(data, template, heights=self.threshold, distance=self.distance,
                                       template_times=t_ref, template_names=g.name,
                                       similarity_func=self._scaled_similarity(store),
                                       template_magnitudes=self.master_mag)
        if not dets or "ccs" not in store:
            return

        ccs, ca = store["ccs"], store["ca"]
        t0, dt = ccs[0].stats.starttime, ccs[0].stats.delta
        nlag = max(1, int(round(self.lag_win / dt)))
        raw_by_id = {tr.id: tr for tr in raw} if raw is not None else {}

        for d in dets:
            idx = int(round((d["time"] - t0) / dt))
            if not 0 <= idx < len(ca):
                continue

            # nearest local maximum of each channel correlation (C_j^L) and its lag
            ids, cjl, lags = [], [], []
            for tr in ccs:
                lo, hi = max(0, idx - nlag), min(len(tr.data), idx + nlag + 1)
                seg = np.nan_to_num(tr.data[lo:hi])
                k = int(np.argmax(seg))
                ids.append(tr.id)
                cjl.append(float(seg[k]))
                lags.append((lo + k - idx) * dt)

            mean_l = float(np.mean(cjl))
            ratio = float(ca[idx]) / mean_l if mean_l > 0 else 0.0
            if ratio < self.min_ratio:
                continue

            slowness = None
            if self.inventory is not None and self.max_slowness is not None:
                slowness = self._plane_wave_slowness(ids, lags)
                if slowness is not None and slowness > self.max_slowness:
                    continue

            n_picks = 0
            for tr_id, value, lag in zip(ids, cjl, lags):
                if value < self.min_channel_cc:
                    continue
                net, sta, _, cha = tr_id.split(".")
                pick = g.picks[f"{net}.{sta}.{cha}"] + (d["time"] - t_ref) + lag
                rows.append(make_pick_row(net, sta, cha, g.phase, pick, value,
                                          self._amplitude(raw_by_id.get(tr_id), pick, g.phase)))
                n_picks += 1

            detections.append([str(d["time"]), g.phase, g.name, len(ccs), n_picks, float(ca[idx]),
                               float(d["similarity"]), ratio, slowness, d.get("magnitude")])

    def write_detections(self, output_dir):
        """Network mode: one row per detection (extra file, PhaseNet outputs untouched)."""
        path = os.path.join(output_dir, "ccpick_detections.csv")
        self.detections.to_csv(path, index=False)
        return path
