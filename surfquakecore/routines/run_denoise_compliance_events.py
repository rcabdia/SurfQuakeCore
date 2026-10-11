import math
import os
from scipy import signal
import numpy as np
from scipy.interpolate import interp1d
import warnings
import csv
from obspy import Stream, Trace, UTCDateTime
from obspy.geodetics import gps2dist_azimuth, locations2degrees
from obspy.taup import TauPyModel

output_path = os.getcwd()
warnings.filterwarnings("ignore", message="RuntimeWarning: divide by zero encountered in log")

def read_event_catalog(path):
    """Read a pipe-delimited event catalog."""
    events = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="|")
        required = {"Time", "Latitude", "Longitude", "Depth/km", "Magnitude"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing catalog columns: {sorted(missing)}")
        for row in reader:
            try:
                events.append({
                    "event_id": row.get("EventID", ""),
                    "time": UTCDateTime(row["Time"].strip()),
                    "latitude": float(row["Latitude"]),
                    "longitude": float(row["Longitude"]),
                    "depth_km": max(0.0, float(row["Depth/km"])),
                    "magnitude": float(row["Magnitude"]),
                })
            except (ValueError, TypeError):
                continue
    return sorted(events, key=lambda x: x["time"])


def merge_windows(windows):
    if not windows:
        return []
    windows = sorted(windows, key=lambda x: x[0])
    merged = [list(windows[0])]
    for start, end in windows[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [tuple(x) for x in merged]


def build_event_windows(trace, inventory, events, min_magnitude=4.5,
                        local_distance_km=500.0, local_min_magnitude=3.0,
                        pre_arrival=120.0, post_arrival=1800.0,
                        model_name="iasp91"):
    """Select contaminating events and build TauP-based exclusion windows."""
    if inventory is None:
        raise ValueError("Inventory is required when event_catalog is used.")
    c = inventory.get_coordinates(trace.id, trace.stats.starttime)
    sta_lat, sta_lon = float(c["latitude"]), float(c["longitude"])
    model = TauPyModel(model=model_name)
    windows = []
    for ev in events:
        distance_m, _, _ = gps2dist_azimuth(sta_lat, sta_lon, ev["latitude"], ev["longitude"])
        distance_km = distance_m / 1000.0
        selected = (ev["magnitude"] >= min_magnitude or
                    (distance_km <= local_distance_km and ev["magnitude"] >= local_min_magnitude))
        if not selected:
            continue
        distance_deg = locations2degrees(sta_lat, sta_lon, ev["latitude"], ev["longitude"])
        arrivals = model.get_travel_times(source_depth_in_km=ev["depth_km"], distance_in_degree=distance_deg)
        if arrivals:
            start = ev["time"] + min(a.time for a in arrivals) - pre_arrival
            end = ev["time"] + max(a.time for a in arrivals) + post_arrival
        else:
            start = ev["time"] - pre_arrival
            end = ev["time"] + post_arrival
        if end >= trace.stats.starttime and start <= trace.stats.endtime:
            windows.append((max(start, trace.stats.starttime), min(end, trace.stats.endtime)))
    return merge_windows(windows)


def make_noise_training_stream(stream, event_windows, taper_percentage=0.05,
                               min_segment_seconds=60.0):
    """Remove identical event windows on all channels and concatenate tapered noise pieces."""
    common_start = max(tr.stats.starttime for tr in stream)
    common_end = min(tr.stats.endtime for tr in stream)
    if common_end <= common_start:
        raise ValueError("Input channels have no common time interval.")
    windows = merge_windows([(max(a, common_start), min(b, common_end))
                             for a, b in event_windows if b >= common_start and a <= common_end])
    usable, cursor = [], common_start
    for start, end in windows:
        if start > cursor:
            usable.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < common_end:
        usable.append((cursor, common_end))
    usable = [(a, b) for a, b in usable if (b-a) >= min_segment_seconds]
    if not usable:
        raise ValueError("No sufficiently long earthquake-free data remain.")
    training = Stream()
    for tr in stream:
        pieces = []
        for start, end in usable:
            piece = tr.copy()
            piece.trim(starttime=start, endtime=end, nearest_sample=True)
            if piece.stats.npts < 2:
                continue
            piece.detrend("demean")
            piece.detrend("linear")
            piece.taper(max_percentage=taper_percentage, type="cosine")
            pieces.append(np.asarray(piece.data, dtype=np.float64))
        if not pieces:
            raise ValueError(f"No usable noise samples remain for {tr.id}")
        data = np.concatenate(pieces)
        hdr = tr.stats.copy(); hdr.starttime = common_start; hdr.npts = len(data)
        training.append(Trace(data=data, header=hdr))
    npts = min(tr.stats.npts for tr in training)
    for tr in training:
        tr.data = tr.data[:npts]
    return training


def run(stream, **kwargs):
    """SurfQuake hook: estimate TFs without catalogued earthquakes, then correct continuous data."""
    inventory = kwargs.pop("inventory", None)
    kwargs.pop("event", None)
    catalog_path = kwargs.pop("event_catalog", None)
    noise = RemoveComplianceTilt(stream)
    tf_stream = stream.copy()
    if catalog_path:
        events = read_event_catalog(catalog_path)
        windows = build_event_windows(
            stream.select(channel="HHZ")[0], inventory, events,
            min_magnitude=kwargs.pop("event_min_magnitude", 4.5),
            local_distance_km=kwargs.pop("local_event_distance_km", 500.0),
            local_min_magnitude=kwargs.pop("local_event_min_magnitude", 3.0),
            pre_arrival=kwargs.pop("event_pre_arrival", 120.0),
            post_arrival=kwargs.pop("event_post_arrival", 1800.0))
        tf_stream = make_noise_training_stream(
            stream, windows,
            taper_percentage=kwargs.pop("event_taper_percentage", 0.05),
            min_segment_seconds=kwargs.pop("min_noise_segment", 60.0))
        hours = tf_stream[0].stats.npts / tf_stream[0].stats.sampling_rate / 3600.0
        print(f"TF estimation: removed {len(windows)} event window(s); {hours:.2f} h retained.")
    tf_noise = RemoveComplianceTilt(tf_stream)

    tf_ch = {"source": tf_stream.select(channel="HH2")[0], "response": tf_stream.select(channel="HH1")[0]}
    noise.transfer_function(tf_ch)
    noise.remove_noise({"source": stream.select(channel="HH2")[0], "response": stream.select(channel="HH1")[0]})
    tf_noise.transfer_info = noise.transfer_info.copy(); tf_noise.remove_noise(tf_ch)

    tf_ch = {"source": tf_stream.select(channel="HH2")[0], "response": tf_stream.select(channel="HHZ")[0]}
    noise.transfer_function(tf_ch)
    noise.remove_noise({"source": stream.select(channel="HH2")[0], "response": stream.select(channel="HHZ")[0]})
    tf_noise.transfer_info = noise.transfer_info.copy(); tf_noise.remove_noise(tf_ch)

    tf_ch = {"source": tf_noise.Nnew, "response": tf_noise.Znew}
    noise.transfer_function(tf_ch)
    noise.remove_noise({"source": noise.Nnew, "response": noise.Znew})
    tf_noise.transfer_info = noise.transfer_info.copy(); tf_noise.remove_noise(tf_ch)

    tf_ch = {"source": tf_stream.select(channel="HDH")[0], "response": tf_noise.Znew}
    noise.transfer_function(tf_ch)
    noise.remove_noise({"source": stream.select(channel="HDH")[0], "response": noise.Znew})

    tr_znew = noise.Znew
    tr_znew.stats.channel = "SCZ"
    out = os.path.join(output_path, tr_znew.id)  # connect to your writer
    return stream


class RemoveComplianceTilt:

    def __init__(self, st):

        """
        N, E, Z, H -> North, East, Vertical and Hydrophon traces

        :param obs_file_path: The file path of pick observations.
        """

        self.N = st.select(network = None, channel = "HH1")[0]
        self.E = st.select(network = None, channel = "HH2")[0]
        self.Z = st.select(network = None, channel = "HHZ")[0]
        self.H = st.select(network = None, channel = "HDH")[0]
        self.Enew = self.E.copy()
        self.Nnew = self.N.copy()
        self.Znew = self.Z.copy()
        self.fs = self.Z.stats.sampling_rate
        self.fn = self.fs/2
        self.transfer_info = {}

    @staticmethod
    def power_log(x):
        n = math.ceil(math.log(x, 2))
        return n

    @staticmethod
    def find_nearest(a, a0):
        "Element in nd array `a` closest to the scalar value `a0`"
        idx = np.abs(a - a0).argmin()
        return a.flat[idx], idx

    def transfer_function(self, channels, nfft = 15, noverlap = 50):

        # source --> X, Y and H
        # response --> Z

        nfft = int(nfft * self.fs * 60)  # 15 minutes in samples
        noverlap = int(nfft * (noverlap/100))

        s = channels["source"].copy()
        r = channels["response"].copy()
        channels = [s, r]
        maxstart = np.max([tr.stats.starttime for tr in channels])
        minend = np.min([tr.stats.endtime for tr in channels])

        s.trim(maxstart, minend)
        r.trim(maxstart, minend)
        s.detrend(type='simple')
        r.detrend(type='simple')

        f, Pss = signal.welch(s.data, fs=self.fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=nfft,
                           detrend='linear', return_onesided=True, scaling='density', axis=-1)
        f, Prr = signal.welch(r.data, fs=self.fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=nfft,
                           detrend='linear', return_onesided=True, scaling='density', axis=-1)
        f, Prs = signal.csd(r.data, s.data, fs=self.fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=nfft,
                         detrend='linear', return_onesided=True, scaling='density', axis=-1)

        coherence = Prs/(np.sqrt(Prr*Pss))
        transfer = np.conj(coherence*np.sqrt(Prr/Pss))

        #self.transfer_info["cpsd"] = Prs
        #self.transfer_info["source_power"] = Pss
        #self.transfer_info["response_power"] = Prr
        self.transfer_info["transfer"] = transfer
        self.transfer_info["coherence"] = coherence
        self.transfer_info["frequency"] = f

    def remove_noise(self, channels):

        #print("Calculating new Trace in Frequency Domain")
        s = channels["source"].copy()
        r = channels["response"].copy()
        s.detrend(type="simple")
        s.taper(type="Hamming", max_percentage=0.05)
        r.detrend(type="simple")
        r.taper(type="Hamming",max_percentage=0.05)

        f = self.transfer_info["frequency"]
        Thr = self.transfer_info["transfer"]

        Sf = np.fft.rfft(s.data, 2 ** self.power_log(len(s.data)))
        Rf = np.fft.rfft(r.data, 2 ** self.power_log(len(r.data)))

        ##Interpolate Thz to Hf

        freq1 = np.fft.rfftfreq(2 ** self.power_log(len(r.data)), 1/self.fs)
        set_interp = interp1d(f, Thr, kind='cubic', bounds_error=False, fill_value=0.0)
        Thrf = set_interp(freq1)
        # set_interp = interp1d(f, np.abs(Thr), kind='cubic')
        #phase_angle = np.angle(Rf)
        #Rff = (np.abs(Rf) - np.abs(Thrf)*np.abs(Hf))* np.exp(1j * phase_angle)
        Rff = (Rf) - (Thrf * Sf)
        value, idx = self.find_nearest(freq1, 0.1)
        Rff[idx:] = Rf[idx:]
        Rnew_data = np.fft.irfft(Rff)

        if r.stats.channel[2] == "Z":
            self.Znew.data = Rnew_data[0:len(r.data)]
        elif r.stats.channel[2] == "N" or r.stats.channel[2] == "1" or r.stats.channel[2] == "Y":
            self.Nnew.data = Rnew_data[0:len(r.data)]
        elif r.stats.channel[2] == "E" or r.stats.channel[2] == "2" or r.stats.channel[2] == "X":
            self.Enew.data = Rnew_data[0:len(r.data)]
