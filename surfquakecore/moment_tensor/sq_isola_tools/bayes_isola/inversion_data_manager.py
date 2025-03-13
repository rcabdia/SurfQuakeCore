import json
import math
import os.path
import shutil
from dataclasses import dataclass

from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth

from surfquakecore.moment_tensor.structures import MomentTensorResult
from surfquakecore.utils.json_utils import DateTimeEncoder


class InversionDataManager:
    def __init__(self, outdir='output', logfile='$outdir/log.txt', output_mkdir=True):
        self.stations = []
        self.outdir = outdir
        if not os.path.exists(outdir) and output_mkdir:
            os.mkdir(outdir)
        self.logfile = open(logfile.replace('$outdir', self.outdir), 'w', 1)
        self.data_raw = []
        self.data_deltas = []  # list of ``stats.delta`` values of traces in ``self.data`` or ``self.data_raw``
        self.logtext = {}
        self.models = {}
        self.stf_description = ""
        self.rupture_length: float = 0.
        self.event: dict = {}
        self.stations_index = {}
        self.nr = None
        self.inversion_result: MomentTensorResult = MomentTensorResult()

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def __del__(self):
        self.logfile.close()
        del self.data_raw

    def save_inversion_results(self, config):

        with open(os.path.join(self.outdir, "inversion.json"), 'w') as f:
            config["origin_date"] = config["origin_date"].strftime('%Y-%m-%d %H:%M:%S.%f')
            merged_dict = {**config, **self.inversion_result.to_dict()}
            json.dump(merged_dict, f, cls=DateTimeEncoder)

    def log(self, s, newline=True, printcopy=False):
        """
        Write text into log file
        :param s: Text to write into log
        :type s: string
        :param newline: if is ``True``, add LF symbol (\\\\n) at the end
        :type newline: bool, optional
        :param printcopy: if is ``True`` prints copy of ``s`` also to stdout
        :type printcopy: bool, optional
        """
        self.logfile.write(s)
        if newline:
            self.logfile.write('\n')
        if printcopy:
            print(s)

    def set_event_info(self, lat, lon, depth, mag, t, agency=''):
        """
        Sets event coordinates, magnitude, and time from parameters given to this function
        :param lat: event latitude
        :type lat: float
        :param lon: event longitude
        :type lon: float
        :param depth: event depth in km
        :type lat: float
        :param mag: event moment magnitude
        :type lat: float
        :param t: event origin time
        :type t: :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime` or string
        :param agency: agency which provides this location
        :type lat: string, optional
        Sets ``rupture_length`` to :math:`\sqrt{111 \cdot 10^M}`, where `M` is the magnitude.
        """
        if isinstance(t, str):
            t = UTCDateTime(t)

        self.event = {'lat': lat, 'lon': lon, 'depth': float(depth) * 1e3, 'mag': float(mag), 't': t, 'agency': agency}
        self.log(
            f"\nHypocenter location:\n  "
            f"Agency: {agency:s}\n  "
            f"Origin time: {t.strftime('%Y-%m-%d %H:%M:%S')}\n  "
            f"Lat {lat:8.3f}   "
            f"Lon {lon:8.3f}   "
            f"Depth{depth:4.1f} km"
        )
        self.rupture_length = math.sqrt(111 * 10 ** self.event['mag'])  # M6 ~ 111 km2, M5 ~ 11 km2 		REFERENCE NEEDED

    def set_source_time_function(self, source_type: str, working_directory: str, t0: float = 0., t1: float = 0.):
        """
        Sets the source time function for calculating elementary seismograms.

        This function writes file ``green/soutype.dat``, which is read by ``green/elemse``
        (function ``fsource()`` at the end of ``elemse.for``).
        """
        # icc parameter-1 is used as number of derivatives in complex domain (1 = no derivative, 2 = a derivative etc.)
        valid_sources = {
            "heaviside": {"description": "Step in displacement", "icc": 2, "ics": 7},
            "step": {"description": "Step in displacement", "icc": 2, "ics": 7},
            "triangular": {"description": f"triangle in velocity, length = {t0:3.1f} s", "icc": 1, "ics": 4},
            "triangle": {"description": f"triangle in velocity, length = {t0:3.1f} s", "icc": 1, "ics": 4},
            "bouchon": {"description": f"Bouchon's smooth step, length = {t0:3.1f} s", "icc": 2, "ics": 2},
            "brune": {"description": f"Brune, length = {t0:3.1f} s", "icc": 1, "ics": 9},
        }
        source = valid_sources.get(source_type.lower(), None)
        if not source:
            raise ValueError(f"Invalid source_type. Please choose a valid options: {','.join(valid_sources.keys())}")

        self.stf_description = source.get("description", "")
        icc = source.get("icc", 0)
        ics = source.get("ics", 0)

        # TODO source complex spectrum is given as array and written to a
        #  TODO file (uncomment reading file 301 in elemse.for)
        with open(os.path.join(working_directory, "soutype.dat"), 'w') as f:
            f.write(f"{ics:d}\n{t0:3.1f}\n{t1:3.1f}\n{icc:d}\n")

    def read_network_coordinates(self, filename, network='', location='', channelcode='LH',
                                 min_distance=None, max_distance=None, max_n_of_stations=None, map_stations=None):
        """
        Read information about stations from file in ISOLA format.
        Calculate their distances and azimuthes using WGS84 elipsoid.
        Create data structure ``self.stations``. Sorts it according to station epicentral distance.

        :param filename: path to file with network coordinates
        :type filename: string
        :param network: all station are from specified network
        :type network: string, optional
        :param location: all stations has specified location
        :type location: string, optional
        :param channelcode: component names of all stations start with these letters (if channelcode is `LH`, component names will be `LHZ`, `LHN`, and `LHE`)
        :type channelcode: string, optional
        :param min_distance: minimal epicentral distance in meters
        :param min_distance: float or None
        :param min_distance: maximal epicentral distance in meters
        :param max_distance: float or None
        :param min_distance: maximal number of stations used in inversion
        :param max_distance: int or None
        :param max_n_of_stations:

        If ``min_distance`` is ``None``, value is calculated as 2*self.rupture_length. If ``max_distance`` is ``None``, value is calculated as :math:`1000 \cdot 2^{2M}`.
        """
        mag = self.event['mag']
        if min_distance is None:
            min_distance = 2 * self.rupture_length
        if max_distance is None:
            max_distance = 1000 * 2 ** (mag * 2.)
        msg = f'Station coordinates: {filename}'
        self.logtext['network'] = msg
        self.log(msg)
        with open(filename, 'r') as f:
            lines = f.readlines()

        stats = []
        for line in lines:
            if line == '\n':  # skip empty lines
                continue
            # 2DO: souradnice stanic dle UTM
            items = line.split()
            sta, lat, lon = items[0:3]
            if len(items) > 3:
                model = items[3]
            else:
                model = ''
            if model not in self.models:
                self.models[model] = 0
            net = network
            loc = location
            ch = channelcode  # default values given by function parameters
            if ":" in sta or "." in sta:
                l = sta.replace(':', '.').split('.')
                net, sta = l[0], l[1]
                if len(l) > 2:
                    loc = l[2]
                if len(l) > 3:
                    ch = l[3]
            stn = \
                {
                    'code': sta, 'lat': lat, 'lon': lon, 'network': net, 'location': loc, 'channelcode': ch,
                    'model': model
                }
            dist, az, baz = gps2dist_azimuth(float(self.event['lat']), float(self.event['lon']), float(lat), float(lon))
            stn['az'] = az
            stn['dist'] = dist
            stn['useN'] = stn['useE'] = stn['useZ'] = False
            stn['accelerograph'] = False
            stn['weightN'] = stn['weightE'] = stn['weightZ'] = 1.
            if min_distance < dist < max_distance:
                stats.append(stn)
            else:
                print("Station", sta, " not included for being out of min & max distances ", min_distance*1E-3," km",
                      max_distance*1E-3, " km")
        stats = sorted(stats, key=lambda stn: stn['dist'])  # sort by distance
        if max_n_of_stations and len(stats) > max_n_of_stations:
            stats = stats[0:max_n_of_stations]
        self.stations = stats
        self.create_station_index()
        self.set_use_components(map_stations=map_stations)

    def create_station_index(self):
        """
        Creates ``self.stations_index`` which serves for accesing ``self.stations`` items by the station name.
        It is called from :func:`read_network_coordinates`.
        """
        stats = self.stations
        self.nr = len(stats)
        self.stations_index = {}
        for i in range(self.nr):
            self.stations_index[
                '_'.join([stats[i]['network'], stats[i]['code'], stats[i]['location'], stats[i]['channelcode']])] = \
                stats[i]

    def set_use_components(self, map_stations=None):

        # map is a list of [[station_name1, channel1, ch1, checked1], [station_name1, channel1, ch1, checked1],....]

        if map_stations is not None and len(map_stations)>0:
            for i in map_stations:
                station_name = i[0]
                channel = i[1]
                ch = channel[2]
                checked = i[2]
                for j in range(len(self.stations)):
                    if self.stations[j]['code'] == station_name:
                        if ch == 'E' or ch == "2" or ch == "X":
                            self.stations[j]['useE'] = checked

                        elif ch == 'N' or ch == "1" or ch == "Y":
                            self.stations[j]['useN'] = checked

                        elif ch == 'Z':
                            self.stations[j]['useZ'] = checked

    def write_stations(self, working_path):
        """
        Write file with carthesian coordinates of stations. The file is necessary for Axitra code.

        This function is usually called from some of the functions related to reading seismograms.
        """
        filename = os.path.join(working_path, "station.dat")
        for model in self.models:
            if model:
                f = filename[0:filename.rfind('.')] + '-' + model + filename[filename.rfind('.'):]
            else:
                f = filename
            with open(f, 'w') as outp:
                outp.write(' Station co-ordinates\n x(N>0,km),y(E>0,km),z(km),azim.,dist.,stat.\n')
                self.models[model] = 0
                for s in self.stations:
                    if s['model'] != model:
                        continue
                    az = math.radians(s['az'])
                    dist = s['dist'] / 1000  # from meter to kilometer
                    outp.write(
                        f"{math.cos(az)*dist:10.4f} {math.sin(az) * dist:10.4f} "
                        f"{0.:10.4f} {s['az']:10.4f} {dist:10.4f} {s['code']:4s} ?\n")
                    self.models[model] += 1

    def read_crust(self, source, output='green/crustal.dat'):
        """
        Copy a file or files with crustal model definition to location where code ``Axitra`` expects it

        :param source: path to crust file
        :type source: string
        :param output: path to copy target
        :type output: string, optional
        """
        inputs = []
        for model in self.models:
            if model:
                inp = source[0:source.rfind('.')] + '-' + model + source[source.rfind('.'):]
                outp = output[0:output.rfind('.')] + '-' + model + output[output.rfind('.'):]
            else:
                inp = source
                outp = output
            shutil.copyfile(inp, outp)
            inputs.append(inp)
        self.log('Crustal model(s): ' + ', '.join(inputs))
        self.logtext['crust'] = ', '.join(inputs)
