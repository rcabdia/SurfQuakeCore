# p_window_snr.py
"""
Example post-processing script for SurfQuakeCore's plot prompt.

Purpose
-------
For each trace, this script:
  1. Reads the theoretical P-wave arrival time from the trace header
     (written by SurfQuakeCore into tr.stats.geodetic['arrivals']).
  2. Cuts a short "noise" window just before the P arrival and a short
     "signal" window just after it.
  3. Computes a simple signal-to-noise ratio (SNR) from the RMS
     amplitude of each window.
  4. Prints a one-line report per trace, and TAGS low-SNR traces by
     writing a custom field into tr.stats so later commands (or your
     own plotting code) can act on it, e.g. to skip or flag them.

This is meant as a template: swap the SNR logic for whatever metric
you actually need (polarity check, amplitude ratio, spectral content,
etc.) — the header-reading and windowing part is the reusable piece.

Required entry point
---------------------
def process(stream):
    ...
    return stream

`stream` is an obspy.Stream built from the traces currently loaded in
the plot tool (self.plot_proj.trace_list). Whatever you return
replaces that trace_list once the command finishes.
"""

import numpy as np
from obspy import UTCDateTime


def process(stream):

    # Windows around the P pick, in seconds.
    # noise_window: taken BEFORE the P arrival (should contain no signal)
    # signal_window: taken AFTER the P arrival (should contain the P wave)
    noise_window = 5.0
    signal_window = 5.0

    ### This part shows how to extract the reference times picked using w

    for tr in stream:
        ref_timestamps = getattr(tr.stats, "references", [])

        if not ref_timestamps:
            print(f"[post_script] {tr.id}: no reference times set.")
            continue

        ref_times = [UTCDateTime(ts) for ts in ref_timestamps]

        print(f"[post_script] {tr.id}: {len(ref_times)} reference time(s):")
        for rt in ref_times:
            print(f"    {rt.isoformat()}")
    ###

    for tr in stream:

        # --- 1. Read header info written by SurfQuakeCore ---
        # tr.stats.geodetic is a dict with keys like:
        #   'geodetic': (distance_km, az, backazimuth)
        #   'arrivals': [{'phase': 'P', 'time': <UTCDateTime-compatible>}, ...]
        #   'otime':    origin time of the event (optional)
        geodetic_info = tr.stats.get("geodetic", {})
        distance_km = geodetic_info.get("geodetic", [None])[0]
        arrivals = geodetic_info.get("arrivals", [])
        origin_time = geodetic_info.get("otime", None)

        # Find the P arrival specifically (there may be multiple phases
        # listed, e.g. P, S, PcP...)
        p_arrival = next((a for a in arrivals if a.get("phase") == "P"), None)

        if p_arrival is None or p_arrival.get("time") is None:
            print(f"[post_script] {tr.id}: no P arrival in header, skipping.")
            continue

        p_time = UTCDateTime(p_arrival["time"])

        # --- 2. Slice out noise and signal windows around the P pick ---
        # Trace.slice() returns a NEW Trace without modifying tr itself,
        # so this is safe even if the windows are oddly placed.
        tr_noise = tr.slice(starttime=p_time - noise_window, endtime=p_time)
        tr_signal = tr.slice(starttime=p_time, endtime=p_time + signal_window)

        if tr_noise.stats.npts == 0 or tr_signal.stats.npts == 0:
            print(f"[post_script] {tr.id}: P arrival outside trace bounds, skipping.")
            continue

        # --- 3. Compute RMS-based SNR ---
        noise_rms = np.sqrt(np.mean(tr_noise.data.astype(float) ** 2))
        signal_rms = np.sqrt(np.mean(tr_signal.data.astype(float) ** 2))

        # Avoid division by zero on a dead/flat channel
        snr = signal_rms / noise_rms if noise_rms > 0 else float("inf")

        # --- 4. Report + tag the trace header for downstream use ---
        dist_str = f"{distance_km:.1f} km" if distance_km is not None else "unknown"

        print(
            f"[post_script] {tr.id}: distance={dist_str}, "
            f"P_time={p_time}, SNR={snr:.2f}"
        )

        # Write the result back into the trace header, so a later
        # command (or your own code) can filter on it, e.g.:
        #   good_traces = [tr for tr in stream if tr.stats.get('snr', 0) > 3]
        tr.stats.snr = snr

    return stream