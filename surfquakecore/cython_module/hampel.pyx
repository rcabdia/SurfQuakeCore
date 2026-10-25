# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
import numpy as np
from libc.math cimport fabs


cdef inline double _select(double* a, int n, int k) noexcept nogil:
    """k-th smallest of a[0:n] (Wirth's quickselect); partially reorders a."""
    cdef int lo = 0, hi = n - 1, i, j
    cdef double x, t
    while lo < hi:
        x = a[k]
        i = lo
        j = hi
        while True:
            while a[i] < x:
                i += 1
            while x < a[j]:
                j -= 1
            if i <= j:
                t = a[i]
                a[i] = a[j]
                a[j] = t
                i += 1
                j -= 1
            if i > j:
                break
        if j < k:
            lo = i
        if k < i:
            hi = j
    return a[k]


def hampel(data, int window_size, double n_sigma):
    """
    Hampel filter (rolling median / MAD) for a 1-D array.

    Same inputs and outputs as the original implementation:
        filtered, outlier_indices, medians, mads, thresholds
    but the rolling medians are computed in C (no Python calls inside the loop).
    Samples within window_size//2 of either edge are not filtered; their entries
    in medians / mads / thresholds are 0.
    """
    arr = np.asarray(data)
    if arr.ndim != 1:
        raise ValueError("Input data must be a 1D array")
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input data contains NaN or Inf values")

    cdef double[::1] x = arr
    cdef int n = x.shape[0]
    cdef int half = window_size // 2
    cdef int w = 2 * half + 1

    filtered = arr.copy()
    medians = np.zeros(n, dtype=np.float64)
    mads = np.zeros(n, dtype=np.float64)
    thresholds = np.zeros(n, dtype=np.float64)
    outliers = np.empty(n, dtype=np.int32)

    cdef double[::1] f = filtered
    cdef double[::1] med_v = medians
    cdef double[::1] mad_v = mads
    cdef double[::1] thr_v = thresholds
    cdef int[::1] out_v = outliers
    buf = np.empty(w, dtype=np.float64)
    cdef double[::1] b = buf

    cdef int i, j, n_out = 0
    cdef double med, mad, thr

    if half >= 1 and n >= w:
        with nogil:
            for i in range(half, n - half):
                for j in range(w):
                    b[j] = x[i - half + j]
                med = _select(&b[0], w, half)
                for j in range(w):
                    b[j] = fabs(b[j] - med)
                mad = _select(&b[0], w, half)
                thr = n_sigma * 1.4826 * mad

                med_v[i] = med
                mad_v[i] = mad
                thr_v[i] = thr
                if fabs(x[i] - med) > thr:
                    f[i] = med
                    out_v[n_out] = i
                    n_out += 1

    return filtered, outliers[:n_out].copy(), medians, mads, thresholds