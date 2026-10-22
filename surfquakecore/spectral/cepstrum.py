"""
Cepstral analysis for ObsPy traces (SurfQuakeCore).

The cepstrum is the "spectrum of the log-spectrum". A periodic or repeating
pattern in the time series (echo, pulse train, harmonic sound) shows up as
regularly spaced ripples in the log-spectrum, and therefore as a peak in the
cepstrum at the repetition period (the "quefrency", in seconds).

    real cepstrum    c[n]  = IFFT( log |X(f)| )
    power cepstrum   cp[n] = | IFFT( log |X(f)|^2 ) |^2
    complex cepstrum ĉ[n]  = IFFT( log |X(f)| + i * unwrapped_phase(f) )
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.signal import find_peaks
from scipy.signal.windows import tukey
from obspy import UTCDateTime

KINDS = ("real", "power", "complex")


@dataclass
class CepstrumResult:
    quefrency: np.ndarray        # seconds (complex: includes negative quefrencies)
    cepstrum: np.ndarray
    kind: str
    delta: float                 # sampling interval of the trace (s)
    nfft: int
    nsamples: int                # samples of the analysed window
    delay_samples: int = 0       # complex only: linear-phase (pure delay) removed
    info: dict = field(default_factory=dict)


def _select_window(trace, window):
    x = np.asarray(trace.data, dtype=np.float64)
    if window is None:
        return x
    dt = trace.stats.delta
    t0, t1 = window
    if isinstance(t0, UTCDateTime):
        t0 = t0 - trace.stats.starttime
    if isinstance(t1, UTCDateTime):
        t1 = t1 - trace.stats.starttime
    i0 = max(int(round(t0 / dt)), 0)
    i1 = min(int(round(t1 / dt)), x.size - 1)
    if i1 - i0 < 3:
        raise ValueError("analysis window is empty or outside the trace")
    return x[i0:i1 + 1]


def cepstrum(trace, kind="real", window=None, taper=0.1, detrend=True,
             nfft=None, water_level=1e-10):
    """
    Cepstrum of an ObsPy Trace.

    Parameters
    ----------
    trace       : obspy.Trace
    kind        : "real" | "power" | "complex"
    window      : (t0, t1) seconds from trace start, or UTCDateTime pair;
                  None -> whole trace.
    taper       : fraction of the window tapered (Tukey); 0 = none.
    detrend     : remove the mean before the FFT.
    nfft        : FFT length (default: next power of two >= 2 * nsamples,
                  i.e. zero-padded, which gives a finer quefrency grid).
    water_level : |X| is floored at water_level * max|X| to avoid log(0).

    Returns
    -------
    CepstrumResult. For "real"/"power" the quefrency axis is 0..nfft/2*dt.
    For "complex" it runs from -nfft/2*dt to +nfft/2*dt (fftshifted).
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    x = _select_window(trace, window)
    n = x.size
    dt = trace.stats.delta
    if detrend:
        x = x - x.mean()
    if taper and taper > 0:
        x = x * tukey(n, alpha=min(float(taper), 1.0))
    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(2 * n)))
    if nfft < n:
        raise ValueError("nfft must be >= number of samples in the window")

    X = np.fft.rfft(x, nfft)
    mag = np.abs(X)
    peak = mag.max()
    if peak == 0.0:
        raise ValueError("signal is identically zero in the analysis window")
    mag = np.maximum(mag, water_level * peak)

    if kind == "real":
        c = np.fft.irfft(np.log(mag), nfft)[:nfft // 2 + 1]
        q = np.arange(c.size) * dt
        return CepstrumResult(q, c, kind, dt, nfft, n)

    if kind == "power":
        c = np.fft.irfft(np.log(mag ** 2), nfft)[:nfft // 2 + 1]
        c = np.abs(c) ** 2
        q = np.arange(c.size) * dt
        return CepstrumResult(q, c, kind, dt, nfft, n)

    # complex cepstrum: unwrap the phase and remove the linear (delay) part
    ph = np.unwrap(np.angle(X))
    ph -= ph[0]                                   # phase at DC taken as 0
    nd = int(np.round(ph[-1] / np.pi))            # delay in samples (nfft/2 * ...)
    ph -= np.pi * nd * np.arange(ph.size) / (ph.size - 1)
    c = np.fft.irfft(np.log(mag) + 1j * ph, nfft)
    c = np.fft.fftshift(c)
    q = (np.arange(nfft) - nfft // 2) * dt
    return CepstrumResult(q, c, kind, dt, nfft, n, delay_samples=nd)


def find_cepstral_peaks(result, qmin=None, qmax=None, n_peaks=5, min_snr=4.0):
    """
    Detect repetition periods as peaks of the cepstrum.

    qmin, qmax : quefrency search range in seconds. Default: from 5 samples
                 (skips the low-quefrency spectral-envelope part) to the
                 largest quefrency.
    n_peaks    : maximum number of peaks returned (strongest first).
    min_snr    : keep peaks whose height exceeds median + min_snr * (robust
                 sigma, 1.4826*MAD) of the search range.

    Returns a list of dicts {quefrency, frequency, value, snr}, where
    frequency = 1/quefrency is the repetition rate in Hz.
    """
    q, c = result.quefrency, result.cepstrum
    if result.kind == "complex":
        keep = q > 0
        q, c = q[keep], c[keep]
    lo = 5 * result.delta if qmin is None else qmin
    hi = q[-1] if qmax is None else qmax
    sel = (q >= lo) & (q <= hi)
    q, c = q[sel], c[sel]
    if c.size < 3:
        return []
    med = np.median(c)
    sigma = 1.4826 * np.median(np.abs(c - med))
    if sigma == 0:
        sigma = np.std(c) or 1.0
    idx, _ = find_peaks(c, height=med + min_snr * sigma)
    order = idx[np.argsort(c[idx])[::-1]][:n_peaks]
    return [{"quefrency": float(q[i]), "frequency": float(1.0 / q[i]),
             "value": float(c[i]), "snr": float((c[i] - med) / sigma)} for i in order]


def plot_cepstrum(result, ax=None, qmax=None, peaks=None, **kw):
    """Quick matplotlib plot of a CepstrumResult (optionally marking peaks)."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.2))
    q, c = result.quefrency, result.cepstrum
    if qmax is not None:
        m = np.abs(q) <= qmax
        q, c = q[m], c[m]
    ax.plot(q, c, lw=0.8, **kw)
    if peaks:
        ax.plot([p["quefrency"] for p in peaks], [p["value"] for p in peaks], "rv")
    ax.set_xlabel("quefrency (s)")
    ax.set_ylabel(f"{result.kind} cepstrum")
    return ax