# drop_flat_traces.py

def process(stream):
    threshold = 1e-6
    kept = [tr for tr in stream if tr.data.max() - tr.data.min() > threshold]

    removed = len(stream) - len(kept)
    if removed:
        print(f"[post_script] Dropped {removed} flat/empty trace(s).")

    from obspy import Stream
    return Stream(traces=kept)