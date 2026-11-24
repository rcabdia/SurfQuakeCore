#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
seismicUtils
"""

import numpy as np
from obspy import Stream
from collections import defaultdict
from scipy import signal

class SeismicUtils:

    @staticmethod
    def standardize_to_NE_components(stream, verbose=False, auto_trim=False):
        """
        Standardize channel codes to Z, N, E and optionally trim to common time window per station.

        Parameters
        ----------
        stream : obspy.Stream
        verbose : bool
            Print renaming and trim info.
        auto_trim : bool
            If True, trim each station group to max(starttime) → min(endtime).
            Default is False.
        """
        mapping = {"1": "N", "2": "E", "X": "E", "Y": "N", "Z": "Z"}
        station_groups = defaultdict(list)
        for tr in stream:
            key = (tr.stats.network, tr.stats.station, tr.stats.location)
            station_groups[key].append(tr)

        new_stream = Stream()

        for station_key, traces in station_groups.items():

            # --- Rename channels ---
            components = set(tr.stats.channel[-1].upper() for tr in traces)
            if {"Z", "N", "E"}.issubset(components):
                if verbose:
                    print(f"✔ Station {station_key} already uses Z, N, E.")
            else:
                for tr in traces:
                    orig = tr.stats.channel
                    last = orig[-1].upper()
                    if last in mapping:
                        new_code = orig[:-1] + mapping[last]
                        if verbose:
                            print(f"  Renaming {orig} → {new_code}")
                        tr.stats.channel = new_code

            # --- Auto trim to common time window ---
            if auto_trim and len(traces) > 1:
                t_start = max(tr.stats.starttime for tr in traces)
                t_end = min(tr.stats.endtime for tr in traces)

                if t_end <= t_start:
                    if verbose:
                        print(f"  [WARN] {station_key}: no common time overlap, skipping trim.")
                else:
                    if verbose:
                        print(f"  Trimming {station_key[0]}.{station_key[1]} "
                              f"to {t_start} → {t_end}")
                    traces = [tr.copy().trim(starttime=t_start, endtime=t_end)
                              for tr in traces]

            for tr in traces:
                new_stream.append(tr)

        return new_stream

    # Rotations accepted by rotate_per_station, with the components each needs
    _ROTATION_INPUTS = {
        "NE->RT": ("N", "E"),
        "RT->NE": ("R", "T"),
        "ZNE->LQT": ("Z", "N", "E"),
        "LQT->ZNE": ("L", "Q", "T"),
    }
    _ROTATION_ALIASES = {"->LQT": "ZNE->LQT"}

    @staticmethod
    def get_header_angles(tr):
        """
        Back-azimuth and inclination of a trace, in degrees, from its header.

        tr.stats.geodetic['geodetic'] = [dist_km, az, baz, inc] is the reference (written by
        analysis_events); tr.stats.back_azimuth / tr.stats.inclination are the fallback.
        Negative values are the "no event" placeholder (-1) and are returned as None.

        Returns
        -------
        (baz, inc) : tuple of float or None
        """
        baz = inc = None
        geodetic = tr.stats.get("geodetic", None)
        geo_vals = geodetic.get("geodetic", None) if geodetic else None
        if geo_vals is not None and len(geo_vals) >= 3:
            baz = geo_vals[2]
            if len(geo_vals) >= 4:
                inc = geo_vals[3]
        if baz is None:
            baz = tr.stats.get("back_azimuth", None)
        if inc is None:
            inc = tr.stats.get("inclination", None)

        def _valid(value):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            return value if np.isfinite(value) and value >= 0 else None

        return _valid(baz), _valid(inc)

    @staticmethod
    def rotate_per_station(stream, method, back_azimuth=None, inclination=None, verbose=True):
        """
        Rotate each station group of a stream independently.

        Groups are (network, station, location, band+instrument code), so the N of a station is
        always rotated with the E of the same station and sensor. ObsPy renames the rotated
        channels itself (N/E -> R/T, Z/N/E -> L/Q/T, ...).

        Parameters
        ----------
        stream : obspy.Stream
            Traces with standardised component codes (see standardize_to_NE_components).
        method : str
            'NE->RT', 'RT->NE', 'ZNE->LQT', 'LQT->ZNE' ('->LQT' is accepted as 'ZNE->LQT').
        back_azimuth, inclination : float, optional
            Fixed angles for all stations. If None, they are read from each station's header
            (get_header_angles). Stations without a valid angle are left unrotated.

        Returns
        -------
        rotated : obspy.Stream
            All traces, rotated where possible, unrotated otherwise.
        n_rotated, n_skipped : int
        """
        method = SeismicUtils._ROTATION_ALIASES.get(method, method)
        if method not in SeismicUtils._ROTATION_INPUTS:
            raise ValueError(f"Unsupported rotation '{method}'. "
                             f"Use one of {list(SeismicUtils._ROTATION_INPUTS)}")
        needed = SeismicUtils._ROTATION_INPUTS[method]
        needs_inc = len(needed) == 3

        groups = defaultdict(list)
        for tr in stream:
            s = tr.stats
            groups[(s.network, s.station, s.location, s.channel[:2])].append(tr)

        rotated = Stream()
        n_rotated = n_skipped = 0

        for (net, sta, loc, band), traces in groups.items():
            group_id = f"{net}.{sta}.{loc}.{band}?"
            group = Stream(traces)
            by_comp = defaultdict(list)
            for tr in traces:
                by_comp[tr.stats.channel[-1].upper()].append(tr)

            missing = [c for c in needed if c not in by_comp]
            if missing:
                # a lone vertical (or a hydrophone) has nothing to rotate, so stay quiet for it
                if verbose and any(c in by_comp for c in needed if c != "Z"):
                    print(f"[WARN] {group_id}: missing component(s) {missing}, not rotated.")
                n_skipped += 1
                rotated += group
                continue
            if any(len(by_comp[c]) > 1 for c in needed):
                if verbose:
                    print(f"[WARN] {group_id}: more than one trace per component, not rotated.")
                n_skipped += 1
                rotated += group
                continue

            baz, inc = back_azimuth, inclination
            if baz is None or (needs_inc and inc is None):
                head_baz, head_inc = SeismicUtils.get_header_angles(by_comp[needed[0]][0])
                baz = head_baz if baz is None else baz
                inc = head_inc if inc is None else inc
            if baz is None or (needs_inc and inc is None):
                if verbose:
                    what = "back-azimuth" if baz is None else "inclination"
                    print(f"[WARN] {group_id}: no valid {what} in header (no event?), not rotated.")
                n_skipped += 1
                rotated += group
                continue

            sub = Stream([by_comp[c][0] for c in needed])
            try:
                if needs_inc:
                    sub.rotate(method=method, back_azimuth=baz, inclination=inc)
                else:
                    sub.rotate(method=method, back_azimuth=baz)
            except Exception as e:
                if verbose:
                    print(f"[WARN] {group_id}: rotation failed ({e}), not rotated.")
                n_skipped += 1
                rotated += group
                continue

            n_rotated += 1
            if verbose:
                inc_txt = f" inc={inc:.2f}°" if needs_inc else ""
                print(f"[INFO] {group_id}: {method} with baz={baz:.2f}°{inc_txt}")
            rotated += group  # same Trace objects, now rotated and renamed by ObsPy

        return rotated, n_rotated, n_skipped

    @staticmethod
    def rename_rotated_channels(stream, method):
        """
        Rename channel codes after rotation to reflect new components.

        Mapping per method:
            NE->RT   : N/1/Y → R,  E/2/X → T
            ->LQT    : Z → L,  N/1/Y → Q,  E/2/X → T
            ZNE->LQT : Z → L,  N/1/Y → Q,  E/2/X → T
            ->ZNE    : ObsPy handles renaming internally, no action needed

        Parameters
        ----------
        stream : obspy.Stream
        method : str
            Rotation method string, e.g. 'NE->RT', '->LQT', 'ZNE->LQT', '->ZNE'
        """
        component_map = {
            "NE->RT": {"N": "R", "1": "R", "Y": "R",
                       "E": "T", "2": "T", "X": "T"},
            "->LQT": {"Z": "L",
                      "N": "Q", "1": "Q", "Y": "Q",
                      "E": "T", "2": "T", "X": "T"},
            "ZNE->LQT": {"Z": "L",
                         "N": "Q", "1": "Q", "Y": "Q",
                         "E": "T", "2": "T", "X": "T"},
            "->ZNE": {}  # ObsPy already renames these
        }

        mapping = component_map.get(method, {})
        if not mapping:
            return stream  # nothing to do

        for tr in stream:
            last = tr.stats.channel[-1].upper()
            if last in mapping:
                tr.stats.channel = tr.stats.channel[:-1] + mapping[last]

        return stream

    @staticmethod
    def multichannel(st, resample=False):
        n = len(st)
        rows = int(0.5 * n * (n - 1) + 1)
        columns = n
        m = np.zeros((rows, columns))
        ccs = np.zeros((rows, 1))

        fs = st[0].stats.sampling_rate
        if resample:
            fs = max(tr.stats.sampling_rate for tr in st)
            st.resample(fs, no_filter=False)

        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                m[k, i] = 1
                m[k, j] = -1

                cc = SeismicUtils.correlate_maxlag(st[i], st[j],
                    maxlag=max(len(st[i].data), len(st[j].data)))

                max_val = max(np.max(cc), -np.min(cc))
                max_idx = np.argmax(cc) if np.max(cc) >= -np.min(cc) else np.argmin(cc)

                lag_time = (max_idx - 0.5 * len(cc)) / fs
                ccs[k] = lag_time
                k += 1

        m[rows - 1, :] = 1
        times = -1 * np.matmul(np.linalg.pinv(m), ccs)
        for i in range(n):
            st[i].stats.starttime += times[i][0]

        return st, times

    @staticmethod
    def correlate_maxlag(a, b, maxlag, demean=True, normalize='naive', method='auto'):
        a = np.asarray(a)
        b = np.asarray(b)
        if demean:
            a = a - np.mean(a)
            b = b - np.mean(b)

        _xcorr = SeismicUtils._xcorr_padzeros if method == 'direct' else SeismicUtils._xcorr_slice
        cc = _xcorr(a, b, maxlag, method)

        if normalize == 'naive':
            norm = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
            cc = cc / norm if norm > np.finfo(float).eps else np.zeros_like(cc)
        elif normalize is not None:
            raise ValueError("normalize must be one of (None, 'naive')")
        return cc

    @staticmethod
    def _xcorr_padzeros(a, b, shift, method):
        if shift is None:
            shift = (len(a) + len(b) - 1) // 2
        dif = len(a) - len(b) - 2 * shift
        if dif > 0:
            b = SeismicUtils._pad_zeros(b, dif // 2)
        else:
            a = SeismicUtils._pad_zeros(a, -dif // 2)
        return signal.correlate(a, b, 'valid', method)

    @staticmethod
    def _xcorr_slice(a, b, shift, method):
        maxlen = len(a) + len(b) - 1
        full = signal.correlate(a, b, 'full', method)
        mid = len(full) // 2
        return full[mid - shift: mid + shift + 1]

    @staticmethod
    def _pad_zeros(data, n):
        return np.pad(data, (n, n), mode='constant', constant_values=0)