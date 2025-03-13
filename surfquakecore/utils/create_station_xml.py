import os
import pandas as pd
from obspy import UTCDateTime, read_inventory
from obspy.core.inventory import Inventory, Network, Station, Channel
from datetime import datetime

class Convert:
    def __init__(self, file_path, sep='\s+', resp_files=None):
        self.file_path = file_path
        self.respfiles = resp_files
        self.sep = sep
        self.all_resps = []

        if self.respfiles is not None:
             self.check_resps()

    def list_directory(self):
        obsfiles = []
        for top_dir, sub_dir, files in os.walk(self.respfiles):
            for file in files:
                obsfiles.append(os.path.join(top_dir, file))
        obsfiles.sort()
        return obsfiles

    def check_resps(self):

        obsfiles = self.list_directory()
        for resp_file in obsfiles:
            resp_id = {}
            try:
                response = read_inventory(resp_file)
                for net in response:
                    for station in net:
                        resp_id["net"] = net.code
                        resp_id["station"] = station.code
                        resp_id["channel"] = station.channels[0].code
                        resp_id["sample_rate"] = station.channels[0].sample_rate
                        resp_id["resp"] = station.channels[0].response
                self.all_resps.append(resp_id)
            except:
                pass


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
        df = pd.read_csv(self.file_path, sep='\s+', header='infer', index_col=False, skiprows=0)
        data_map = {}
        data_map['nets'] = {}
        for i in range(len(df)):

            item = df.iloc[i]
            Station = item.Station

            Network = item.Net
            Latitude = item.Lat
            Longitude = item.Lon
            Elevation = item.elevation
            StartDate = item.start_date
            StartTime = item.starttime
            EndDate = item.end_date
            EndTime = item.endtime
            stations_info = [Latitude, Longitude, Elevation, StartDate, StartTime, EndDate, EndTime]
            if Network not in data_map['nets']:
                data_map['nets'][Network] = {}
                data_map['nets'][Network]['stations'] = {}
            # 2. Check if the station exists, else add
            if Station not in data_map['nets'][Network]['stations']:
                data_map['nets'][Network]['stations'][Station] = stations_info
            #     data_map['nets'][Station].update(stations)
        return data_map

    def get_data_inventory(self, data_map):
        nets = []
        for net, value in data_map['nets'].items():
            #print("Network:", net)
            network = Network(code=net)
            for station_code, info in value['stations'].items():
                #try:
                #print(station_code, info)
                starttime_str = f'{info[3]} {info[4]}'
                endtime_str = f'{info[5]} {info[6]}'
                # Convert the combined string to a datetime object
                starttime_datetime = datetime.strptime(starttime_str, '%Y-%m-%d %H:%M:%S')
                endtime_datetime = datetime.strptime(endtime_str, '%Y-%m-%d %H:%M:%S')
                starttime = UTCDateTime(starttime_datetime)
                endtime = UTCDateTime(endtime_datetime)
                "Create a station"
                station = Station(
                    code=station_code,
                    latitude=info[0],
                    longitude=info[1],
                    elevation=info[2],
                    creation_date=starttime,
                    termination_date=endtime,
                )
                if len(self.all_resps)>0:
                    station_resps = self._get_channels_resp(station_code)
                    # Add channels to the station
                    for channel in station_resps:
                        channel = Channel(
                             code=channel["channel"],
                             response=channel["resp"],
                             location_code="",  # You can adjust this if needed
                             latitude=info[0],
                             longitude=info[1],
                             elevation=info[2],
                             depth=0,
                             azimuth=0.0,
                             dip=-90.0,
                             sample_rate=100.0,  # Replace with the actual sample rate
                             start_date=UTCDateTime(starttime),
                             end_date=UTCDateTime(endtime))
                        station.channels.append(channel)
                # Attach the station to the network
                network.stations.append(station)
                #except:
                #    print("Cannot include ", station)
            nets.append(network)
            # Create an inventory and attach the network
        inventory = Inventory(networks=nets)
        print(inventory)
        return inventory

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