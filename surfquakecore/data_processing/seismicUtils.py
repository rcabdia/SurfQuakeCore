#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
seismicUtils
"""

from obspy import Stream
from collections import defaultdict
import numpy as np
from scipy import signal

import numpy as np
from obspy import Stream
from collections import defaultdict
from scipy import signal

class SeismicUtils:

    @staticmethod
    def standardize_to_NE_components(stream, verbose=False):
        mapping = {"1": "N", "2": "E", "X": "E", "Y": "N", "Z": "Z"}
        station_groups = defaultdict(list)
        for tr in stream:
            key = (tr.stats.network, tr.stats.station, tr.stats.location)
            station_groups[key].append(tr)

        new_stream = Stream()

        for station_key, traces in station_groups.items():
            components = set(tr.stats.channel[-1].upper() for tr in traces)
            if {"Z", "N", "E"}.issubset(components):
                if verbose:
                    print(f"✔ Station {station_key} already uses Z, N, E.")
                new_stream += Stream(traces)
                continue

            for tr in traces:
                orig = tr.stats.channel
                last = orig[-1].upper()
                if last in mapping:
                    new_code = orig[:-1] + mapping[last]
                    if verbose:
                        print(f"Renaming {orig} → {new_code}")
                    tr.stats.channel = new_code
                new_stream.append(tr)

        return new_stream

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