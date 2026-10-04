import numpy as np


def process(stream):
    """
    Required entry point. Receives the current obspy.Stream,
    does whatever processing is needed, and returns the
    (possibly modified) Stream.
    """
    for tr in stream:
        # simple demean
        tr.data = tr.data - np.mean(tr.data)

        print(
            f"[post_script] {tr.id}: "
            f"npts={tr.stats.npts}, "
            f"max={tr.data.max():.3e}, "
            f"min={tr.data.min():.3e}"
        )

    return stream