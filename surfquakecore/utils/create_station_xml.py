import os
import pandas as pd
from obspy import UTCDateTime, read_inventory
from obspy.core.inventory import Inventory, Network, Station, Channel, Site, ClockDrift


class Convert:
    def __init__(self, file_path, sep=None, resp_files=None):
        self.file_path = file_path
        self.resp_files = resp_files
        self.sep = sep
        self.all_resps = []

        self.resp_index = {}  # (net, sta, loc, cha) -> obspy Channel
        self.resp_fallback = {}  # (net, sta) -> list[obspy Channel]

        if resp_files:
            self.check_resps()

    def _detect_sep(self) -> str:
        """
        Detect delimiter from the first non-empty line.
        - If a comma appears, assume CSV.
        - Else assume whitespace table (your current coords.txt style).
        """
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                return "," if "," in line else r"\s+"
        return r"\s+"

    def _read_station_table(self) -> pd.DataFrame:
        # If user passed sep explicitly, use it; otherwise detect.
        sep = self.sep if self.sep is not None else self._detect_sep()

        # engine="python" supports regex separators like r"\s+"
        df = pd.read_csv(
            self.file_path,
            sep=sep,
            header="infer",
            index_col=False,
            skiprows=0,
            engine="python",
        )
        return df

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # lower-case + strip spaces
        df.columns = [str(c).strip().lower() for c in df.columns]

        # allow common aliases
        aliases = {
            "network": "net",
            "stationcode": "station",
            "longitude": "lon",
            "long": "lon",
            "latitude": "lat",
            "elev": "elevation",
            "starttime": "starttime",
            "start_time": "starttime",
            "start date": "start_date",
            "startdate": "start_date",
            "endtime": "endtime",
            "end_time": "endtime",
            "end date": "end_date",
            "enddate": "end_date",
        }

        aliases.update({
            # station-level
            "sitename": "site_name",
            "site": "site_name",
            "station_name": "site_name",

            # channel-level / general
            "dep": "depth",
            "sensor_depth": "depth",

            # clock drift
            "clockdrift": "clock_drift",
            "drift": "clock_drift",
            "clock_drift_rate": "clock_drift",
        })

        df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})
        return df

    def _validate_station_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and coerce the station table.
        - Ensures required columns exist
        - Coerces numeric columns
        - Validates lat/lon ranges
        - Validates datetime fields (start/end)
        Returns a cleaned dataframe with added parsed datetime columns.
        """
        required = ["net", "station", "lat", "lon", "elevation", "start_date", "starttime", "end_date", "endtime"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                "Missing required column(s): "
                + ", ".join(missing)
                + ".\nExpected columns include: "
                + ", ".join(required)
            )

        # Coerce numeric fields
        for col in ["lat", "lon", "elevation"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        bad_numeric = df[df[["lat", "lon", "elevation"]].isna().any(axis=1)]
        if not bad_numeric.empty:
            # show up to first 5 bad rows (1-based line number in dataframe + header row ambiguity)
            examples = bad_numeric.head(5).index.tolist()
            raise ValueError(
                f"Non-numeric value found in lat/lon/elevation at row index(es): {examples}. "
                "Please check those rows."
            )

        # Range checks
        bad_lat = df[(df["lat"] < -90) | (df["lat"] > 90)]
        if not bad_lat.empty:
            examples = bad_lat.head(5).index.tolist()
            raise ValueError(f"Latitude out of range [-90, 90] at row index(es): {examples}.")

        bad_lon = df[(df["lon"] < -180) | (df["lon"] > 180)]
        if not bad_lon.empty:
            examples = bad_lon.head(5).index.tolist()
            raise ValueError(f"Longitude out of range [-180, 180] at row index(es): {examples}.")

        # Parse datetimes (supports your tool's expected format)
        start_str = df["start_date"].astype(str).str.strip() + " " + df["starttime"].astype(str).str.strip()
        end_str = df["end_date"].astype(str).str.strip() + " " + df["endtime"].astype(str).str.strip()

        df["_start_dt"] = pd.to_datetime(start_str, format="%Y-%m-%d %H:%M:%S", errors="coerce")
        df["_end_dt"] = pd.to_datetime(end_str, format="%Y-%m-%d %H:%M:%S", errors="coerce")

        bad_start = df[df["_start_dt"].isna()]
        if not bad_start.empty:
            examples = bad_start.head(5).index.tolist()
            raise ValueError(
                f"Invalid start datetime at row index(es): {examples}. "
                "Expected format: YYYY-MM-DD for start_date and HH:MM:SS for starttime."
            )

        bad_end = df[df["_end_dt"].isna()]
        if not bad_end.empty:
            examples = bad_end.head(5).index.tolist()
            raise ValueError(
                f"Invalid end datetime at row index(es): {examples}. "
                "Expected format: YYYY-MM-DD for end_date and HH:MM:SS for endtime."
            )

        # Logical check: end >= start (allow equal)
        bad_order = df[df["_end_dt"] < df["_start_dt"]]
        if not bad_order.empty:
            examples = bad_order.head(5).index.tolist()
            raise ValueError(
                f"End time is before start time at row index(es): {examples}. "
                "Please correct start/end."
            )

        # Optional: strip net/station codes
        df["net"] = df["net"].astype(str).str.strip()
        df["station"] = df["station"].astype(str).str.strip()

        bad_codes = df[(df["net"] == "") | (df["station"] == "")]
        if not bad_codes.empty:
            examples = bad_codes.head(5).index.tolist()
            raise ValueError(f"Empty net/station code at row index(es): {examples}.")

        optional_numeric = ["azimuth", "sample_rate", "dip", "depth", "clock_drift"]
        for col in optional_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "sample_rate" in df.columns:
            df["sample_rate"] = df["sample_rate"].fillna(100.0)

        # if "clock_drift" in df.columns:
        #     df["clock_drift"] = df["clock_drift"].fillna(0.0)

        if "depth" in df.columns:
            df["depth"] = df["depth"].fillna(0.0)

        if "dip" in df.columns:
            df["dip"] = df["dip"].fillna(-90.0)

        if "azimuth" in df.columns:
            df["azimuth"] = df["azimuth"].fillna(0.0)

        if "depth" in df.columns:
            bad_depth = df[df["depth"] < 0]
            if not bad_depth.empty:
                examples = bad_depth.head(5).index.tolist()
                raise ValueError(f"Depth must be >= 0 at row index(es): {examples}.")

        return df

    def _resp_lookup(self, net, sta, loc, cha):
        """
        Try exact match first; then fall back by ignoring location code.
        """
        if not self.resp_index:
            return None

        loc = loc or ""
        key = (net, sta, loc, cha)
        if key in self.resp_index:
            return self.resp_index[key]

        # fallback: try any location code for same net/sta/cha
        for (n, s, l, c), ch_obj in self.resp_index.items():
            if n == net and s == sta and c == cha:
                return ch_obj

        return None

    def _channel_from_spec(self, net_code, station_code, station_meta, chspec, input_cols):

        # Start/end from spec
        ch_start = UTCDateTime(chspec["start_dt"])
        ch_end = UTCDateTime(chspec["end_dt"])

        code = str(chspec["code"])
        loc = str(chspec.get("location_code", ""))

        # Base values from CSV/default
        sr = float(chspec.get("sample_rate", 100.0))
        dip = float(chspec.get("dip", -90.0))
        az = float(chspec.get("azimuth", 0.0))

        # ---- Step 5: clock drift (channel overrides station) ----
        clock_drift = chspec.get("clock_drift", None)
        if clock_drift is None:
            clock_drift = station_meta.get("clock_drift", None)

        # If RESP exists, try to match and attach response
        resp_ch = self._resp_lookup(net_code, station_code, loc, code)

        # Optionally: if user DIDN'T provide these columns, let RESP override defaults
        if resp_ch is not None:
            if "location_code" not in input_cols:
                loc = getattr(resp_ch, "location_code", loc) or loc

            if "sample_rate" not in input_cols:
                r_sr = getattr(resp_ch, "sample_rate", None)
                if r_sr is not None:
                    sr = float(r_sr)

            if "dip" not in input_cols:
                r_dip = getattr(resp_ch, "dip", None)
                if r_dip is not None:
                    dip = float(r_dip)

            if "azimuth" not in input_cols:
                r_az = getattr(resp_ch, "azimuth", None)
                if r_az is not None:
                    az = float(r_az)

            if "clock_drift" not in input_cols:
                r_cd = getattr(resp_ch, "clock_drift", None)
                if r_cd is not None:
                    clock_drift = float(r_cd)

        ch = Channel(
            code=code,
            location_code=loc,
            latitude=float(station_meta["lat"]),
            longitude=float(station_meta["lon"]),
            elevation=float(station_meta["elevation"]),
            depth = float(chspec.get("depth", 0.0)),
            azimuth=az,
            dip=dip,
            sample_rate=sr,
            start_date=ch_start,
            end_date=ch_end,
        )

        if resp_ch is not None and getattr(resp_ch, "response", None) is not None:
            ch.response = resp_ch.response

        if clock_drift is not None:
            cd_s_per_s = float(clock_drift)
            cd_s_per_sample = cd_s_per_s / float(sr)
            ch.clock_drift_in_seconds_per_sample = ClockDrift(cd_s_per_sample)  # important

        return ch

    def list_directory(self):
        obsfiles = []
        for top_dir, sub_dir, files in os.walk(self.resp_files):
            for file in files:
                obsfiles.append(os.path.join(top_dir, file))
        obsfiles.sort()
        return obsfiles

    def check_resps(self):
        """
        Build response indexes from all inventory/RESP files in resp_files directory.

        - resp_index[(net, sta, loc, cha)] = Channel
        - resp_fallback[(net, sta)] = [Channel, ...]
        """

        self.resp_index = {}
        self.resp_fallback = {}

        if not self.resp_files:
            return

        # Collect files recursively
        resp_paths = []
        for root, _, files in os.walk(self.resp_files):
            for fn in files:
                resp_paths.append(os.path.join(root, fn))

        for path in resp_paths:
            try:
                inv = read_inventory(path)
            except Exception as e:
                print(f"[WARN] Could not read RESP/inventory file: {path} ({e})")
                continue

            # Loop everything: networks -> stations -> channels
            for net in inv:
                for sta in net:
                    for cha in sta:
                        loc = getattr(cha, "location_code", "") or ""
                        key = (net.code, sta.code, loc, cha.code)

                        # If duplicates appear, last one wins (simple rule)
                        self.resp_index[key] = cha
                        self.resp_fallback.setdefault((net.code, sta.code), []).append(cha)

    def find_resp(self, network, station, loc, channel, datetime):
        inv = None
        id = network+"." + station + "." + loc+ "." + channel
        for inv in self.all_resps:
            inv = inv.get_response(id, datetime)
        return inv

    def _get_channels_resp(self, station):

        resps = []
        for station_resp in self.all_resps:
            if station == station_resp['station']:
                resps.append(station_resp)

        return resps

    def create_stations_xml(self):
        """
        Build a data_map with this structure:

        data_map = {
          "nets": {
            "<NET>": {
              "stations": {
                "<STA>": {
                  "station": {
                    "lat": ...,
                    "lon": ...,
                    "elevation": ...,
                    "start_dt": datetime,
                    "end_dt": datetime,
                  },
                  "channels": [
                    {
                      "code": ...,
                      "location_code": ...,
                      "sample_rate": ...,
                      "dip": ...,
                      "azimuth": ...,
                      "start_dt": datetime,
                      "end_dt": datetime,
                    },
                    ...
                  ]
                }
              }
            }
          }
        }
        """


        # Read + normalize + validate (your Step 1/2 refactor)
        df = self._read_station_table()
        df = self._normalize_columns(df)
        df = self._validate_station_table(df)

        data_map = {"nets": {}}
        data_map["_input_cols"] = list(df.columns)
        # Optional column flags
        has_channel = "channel" in df.columns
        has_sr = "sample_rate" in df.columns
        has_loc = "location_code" in df.columns
        has_dip = "dip" in df.columns
        has_az = "azimuth" in df.columns
        has_site = "site_name" in df.columns
        has_clock = "clock_drift" in df.columns
        has_depth = "depth" in df.columns

        for _, item in df.iterrows():
            net = str(item["net"]).strip()
            sta = str(item["station"]).strip()

            lat = float(item["lat"])
            lon = float(item["lon"])
            elev = float(item["elevation"])

            # From validator (already parsed/verified)
            start_dt = item["_start_dt"].to_pydatetime()
            end_dt = item["_end_dt"].to_pydatetime()

            # ---- Step 5: optional station-level fields ----
            site_name = sta
            if has_site:
                raw_site = item["site_name"]
                if pd.notna(raw_site) and str(raw_site).strip():
                    site_name = str(raw_site).strip()

            station_clock_drift = None
            if has_clock:
                raw_cd = item["clock_drift"]
                if pd.notna(raw_cd):
                    station_clock_drift = float(raw_cd)

            # Ensure net + station container exists
            if net not in data_map["nets"]:
                data_map["nets"][net] = {"stations": {}}

            if sta not in data_map["nets"][net]["stations"]:
                data_map["nets"][net]["stations"][sta] = {
                    "station": {
                        "lat": lat,
                        "lon": lon,
                        "elevation": elev,
                        "start_dt": start_dt,
                        "end_dt": end_dt,
                        "site_name": site_name,
                        "clock_drift": station_clock_drift,
                    },
                    "channels": [],
                    "channels_provided": False
                }
            else:
                # Optional: keep earliest start and latest end if duplicates exist
                s = data_map["nets"][net]["stations"][sta]["station"]
                if start_dt < s["start_dt"]:
                    s["start_dt"] = start_dt
                if end_dt > s["end_dt"]:
                    s["end_dt"] = end_dt

                # Keep first non-empty site_name and first non-null clock_drift
                if not s.get("site_name") and site_name:
                    s["site_name"] = site_name

                if s.get("clock_drift") is None and station_clock_drift is not None:
                    s["clock_drift"] = station_clock_drift

            # Build channel spec only if a channel column exists AND is non-empty
            chan_code = None
            if has_channel:
                raw = item["channel"]
                if pd.notna(raw) and str(raw).strip():
                    chan_code = str(raw).strip()

            if chan_code:
                data_map["nets"][net]["stations"][sta]["channels_provided"] = True
                loc = ""
                if has_loc:
                    raw_loc = item["location_code"]
                    if pd.notna(raw_loc) and str(raw_loc).strip():
                        loc = str(raw_loc).strip()

                sr = 100.0
                if has_sr and pd.notna(item["sample_rate"]):
                    sr = float(item["sample_rate"])

                dip = -90.0
                if has_dip and pd.notna(item["dip"]):
                    dip = float(item["dip"])

                az = 0.0
                if has_az and pd.notna(item["azimuth"]):
                    az = float(item["azimuth"])

                depth = 0.0
                if has_depth and pd.notna(item["depth"]):
                    depth = float(item["depth"])

                ch_clock_drift = None
                if has_clock:
                    raw_cd = item["clock_drift"]
                    if pd.notna(raw_cd):
                        ch_clock_drift = float(raw_cd)

                data_map["nets"][net]["stations"][sta]["channels"].append({
                    "code": chan_code,
                    "location_code": loc,
                    "sample_rate": sr,
                    "dip": dip,
                    "azimuth": az,
                    "depth": depth,
                    "clock_drift": ch_clock_drift,
                    "start_dt": start_dt,
                    "end_dt": end_dt,
                })



        # Inject default channel if no channels were provided for a station
        DEFAULT_CHAN = "HHZ"
        DEFAULT_SR = 100.0
        DEFAULT_LOC = ""
        DEFAULT_DIP = -90.0
        DEFAULT_AZ = 0.0

        for net, netdata in data_map["nets"].items():
            for sta, stad in netdata["stations"].items():
                if not stad["channels"]:
                    s = stad["station"]
                    stad["channels"].append({
                        "code": DEFAULT_CHAN,
                        "location_code": DEFAULT_LOC,
                        "sample_rate": DEFAULT_SR,
                        "dip": DEFAULT_DIP,
                        "azimuth": DEFAULT_AZ,
                        "depth": 0.0,
                        "clock_drift": None,
                        "start_dt": s["start_dt"],
                        "end_dt": s["end_dt"],
                    })

        return data_map

    def get_data_inventory(self, data_map):


        # What columns existed in the user input?
        # (store this in create_stations_xml as: data_map["_input_cols"] = list(df.columns))
        input_cols = set(data_map.get("_input_cols", []))

        nets = []

        for net_code, netdata in data_map["nets"].items():
            network = Network(code=net_code)

            for station_code, stad in netdata["stations"].items():
                s = stad["station"]
                starttime = UTCDateTime(s["start_dt"])
                endtime = UTCDateTime(s["end_dt"])

                station = Station(
                    code=station_code,
                    latitude=float(s["lat"]),
                    longitude=float(s["lon"]),
                    elevation=float(s["elevation"]),
                    creation_date=starttime,
                    termination_date=endtime,
                    site=Site(name=s.get("site_name") or station_code),
                )

                # If RESP exists AND user did NOT provide channels, use RESP channels
                if self.resp_fallback and not stad.get("channels_provided", False):
                    resp_channels = self.resp_fallback.get((net_code, station_code), [])

                    if resp_channels:
                        for resp_ch in resp_channels:
                            loc = getattr(resp_ch, "location_code", "") or ""
                            sr = getattr(resp_ch, "sample_rate", None)
                            dip = getattr(resp_ch, "dip", None)
                            az = getattr(resp_ch, "azimuth", None)

                            ch = Channel(
                                code=resp_ch.code,
                                location_code=loc,
                                latitude=float(s["lat"]),
                                longitude=float(s["lon"]),
                                elevation=float(s["elevation"]),
                                depth=0.0,
                                azimuth=float(az) if az is not None else 0.0,
                                dip=float(dip) if dip is not None else -90.0,
                                sample_rate=float(sr) if sr is not None else 100.0,
                                start_date=starttime,
                                end_date=endtime,
                            )

                            # Attach response if present
                            if getattr(resp_ch, "response", None) is not None:
                                ch.response = resp_ch.response

                            station.channels.append(ch)
                    else:
                        # No RESP found for this station: fall back to CSV/default channels
                        for chspec in stad["channels"]:
                            station.channels.append(
                                self._channel_from_spec(net_code, station_code, s, chspec, input_cols))

                else:
                    # User provided channels (or no RESP): create from specs and attach responses when matched
                    for chspec in stad["channels"]:
                        station.channels.append(self._channel_from_spec(net_code, station_code, s, chspec, input_cols))

                network.stations.append(station)

            nets.append(network)

        return Inventory(networks=nets, source="csv2xml")

    @staticmethod
    def write_xml(path, name, inventory):


        if os.path.isdir(path):
            pass
        else:
            raise Exception("Loc files directory does not exist")

        if os.path.isdir(path):
            pass
        else:
            try:
                os.makedirs(path)
            except Exception as error:
                print("An exception occurred:", error)

        # Write to StationXML file
        xml_filename = os.path.join(path, name)
        xml_filename = f"{xml_filename}_station.xml"
        inventory.write(xml_filename, format="stationxml")
        print(f"StationXML file '{xml_filename}' created successfully.")


if __name__ == "__main__":
    stations_file = "/Users/roberto/Documents/python_notes/my_utils/test_data/coords.txt"
    path_dest = "/Users/roberto/Documents/python_notes/my_utils/test_data"
    resp_files = "/Users/roberto/Documents/python_notes/my_utils/resp_files"
    name = "test"
    sc = Convert(stations_file, resp_files=resp_files)
    data_map = sc.create_stations_xml()
    inventory = sc.get_data_inventory(data_map)
    sc.write_xml(path_dest, name, inventory)