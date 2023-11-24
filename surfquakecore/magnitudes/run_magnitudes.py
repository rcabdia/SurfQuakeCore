import os
from obspy import read, read_events, UTCDateTime
from surfquakecore.utils.obspy_utils import MseedUtil


class Automag:

    def __init__(self, project, locations_directory):

        self.project = project
        self.locations_directory = locations_directory


    def get_now_files(self, date):

        selection = [".", ".", "."]

        _, self.files_path = MseedUtil.filter_project_keys(self.project, net=selection[0], station=selection[1],
                                                       channel=selection[2])
        start = date.split(".")
        start = UTCDateTime(year=int(start[1]), julday=int(start[0]), hour=00, minute=00, second=00)+1
        end = start+(24*3600-2)
        self.files_path = MseedUtil.filter_time(list_files=self.files_path, starttime=start, endtime=end)
        print(self.files_path)

    def filter_station(self, station):

        filtered_list = []

        for file in self.files_path:
            header = read(file, headlonly=True)
            sta = header[0].stats.station
            if station == sta:
                filtered_list.append(file)

        return filtered_list

    def scan_folder(self):
        obsfiles1 = []
        dates = {}
        for top_dir, _, files in os.walk(self.locations_directory):

            for file in files:
                try:
                    file_hyp = os.path.join(top_dir, file)
                    cat = read_events(file_hyp, format="NLLOC_HYP")
                    ev = cat[0]
                    date = ev.origins[0]["time"]
                    date = str(date.julday) + "." + str(date.year)

                    obsfiles1.append(file_hyp)
                    if date not in dates:
                        dates[date] = [file_hyp]
                    else:
                        dates[date].append(file_hyp)
                except:
                    pass

        self.dates=dates

    def scan_from_origin(self, origin):

        self.date = origin["time"]

    def _get_stations(self, arrivals):
        stations = []
        for pick in arrivals:
            if pick.station not in stations:
                stations.append(pick.station)

        return stations

    def estimate_magnitudes(self):

        self.scan_folder()
        for date in self.dates:
            events = self.dates[date]
            #events = list(set(events))
            self.get_now_files(date)
            for event in events:
                print(event)