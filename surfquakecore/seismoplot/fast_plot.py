#Quick-and-dirty (QDP-style) min/max decimated plotting, inspired by SAC's QDP."""
import numpy as np


def _argminmax(a, axis, has_nan):
    """argmin/argmax that ignore NaNs (an all-NaN block returns index 0 -> NaN -> line break)."""
    if has_nan:
        nan = np.isnan(a)
        return (np.where(nan, np.inf, a).argmin(axis),
                np.where(nan, -np.inf, a).argmax(axis))
    return a.argmin(axis), a.argmax(axis)


def minmax_decimate(x, n_points, has_nan=False):
    """Keep the min and max sample of each section (time order preserved).
    Returns (indices, desampling_factor). indices=None means 'plot everything'."""
    npts = x.size
    if n_points is None or npts <= n_points:
        return None, 1

    nsec = max(n_points // 2, 1)          # 2 points (min+max) per section
    sec = max(npts // nsec, 1)            # section size, rounded down (as SAC)
    nsec = npts // sec
    m = nsec * sec

    blocks = x[:m].reshape(nsec, sec)
    imin, imax = _argminmax(blocks, 1, has_nan)
    base = np.arange(nsec) * sec

    idx = np.empty(2 * nsec, dtype=np.int64)
    idx[0::2] = base + np.minimum(imin, imax)
    idx[1::2] = base + np.maximum(imin, imax)

    if m < npts:                          # remainder block, so no extremum is lost
        tail = x[m:]
        tmin, tmax = _argminmax(tail, None, has_nan)
        t_lo, t_hi = sorted((m + int(tmin), m + int(tmax)))
        idx = np.concatenate([idx, [t_lo, t_hi]])

    return idx, max(sec // 2, 1)


class FastLine:
    """A matplotlib line that shows a min/max decimated view of a trace and
    re-decimates from the full-resolution data whenever the x-limits change
    (zoom/pan), so detail is never lost.

    n_points=None -> automatic: ~points_per_pixel * axes width in pixels.
    """

    def __init__(self, ax, tr, n_points=None, points_per_pixel=2, **plot_kw):
        self.ax = ax

        data = tr.data
        if np.ma.isMaskedArray(data):                     # masked gaps -> NaN
            data = np.ma.filled(data.astype("float64"), np.nan)
        self.data = data
        self.has_nan = bool(data.dtype.kind == "f" and np.isnan(data).any())

        self.t0 = tr.stats.starttime.matplotlib_date
        self.dt = tr.stats.delta / 86400.0                # sample spacing in matplotlib days
        self.n_fixed = n_points
        self.ppp = points_per_pixel

        (self.line,) = ax.plot([], [], **plot_kw)
        self.factor = 1
        self._redraw(self.t0, self.t0 + (self.data.size - 1) * self.dt)

        # set_data does not update the data limits: without this the axes keep
        # the empty-plot default (about +-0.055) and the trace looks clipped
        ax.relim()
        ax.autoscale_view()

        ax.callbacks.connect("xlim_changed", self._on_xlim)
        ax._surfq_fastline = self   # callbacks hold weak refs, so keep this object alive

    def _target_points(self):
        if self.n_fixed:
            return int(self.n_fixed)
        width_px = max(int(self.ax.get_window_extent().width), 200)
        return width_px * self.ppp

    def _redraw(self, x0, x1):
        n = self.data.size
        i0 = max(int(np.floor((x0 - self.t0) / self.dt)) - 1, 0)
        i1 = min(int(np.ceil((x1 - self.t0) / self.dt)) + 2, n)
        if i1 - i0 < 2:
            return
        seg = self.data[i0:i1]
        idx, self.factor = minmax_decimate(seg, self._target_points(), self.has_nan)
        if idx is None:
            idx = np.arange(seg.size)
        self.line.set_data(self.t0 + (i0 + idx) * self.dt, seg[idx])

    def _on_xlim(self, ax):
        self._redraw(*ax.get_xlim())