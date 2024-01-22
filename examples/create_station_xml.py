import os
import pandas as pd
from obspy import UTCDateTime, read_inventory
from obspy.core.inventory import Inventory, Network, Station, Channel
from datetime import datetime

class Convert:
    def __init__(self, file_path, channels=None, resp_files=None):
        self.file_path = file_path
        self.channels = channels
        self.respfiles = resp_files
        if self.respfiles is not None:
            self.all_resps = self.check_resps


    def list_directory(self):
        obsfiles = []
        for top_dir, sub_dir, files in os.walk(self.data_path):
            for file in files:
                obsfiles.append(os.path.join(top_dir, file))
        obsfiles.sort()
        return obsfiles

    def check_resps(self):
        all_resps = []
        obsfiles = self.list_directory()
        for resp_file in obsfiles:
            try:
                response = read_inventory(resp_file)[0][0][0].response
                all_resps.append(response)
            except:
                pass
        return all_resps

    def find_resp(self, network, station, loc, channel, datetime):
        inv = None
        id = network+"."+ station+ "." + loc+ "." + channel
        for inv in self.all_resps:
            inv = inv.get_response(id, datetime)
        return inv

    def get_df(self):
        # Define column names
        columns = ['ID', 'Network', 'Latitude', 'Longitude', 'Elevation', 'StartDate', 'StartTime','EndDate','EndTime', 'Location']
        # Read the file into a Pandas DataFrame, considering the last column as one
        df = pd.read_csv(self.file_path, sep='\s+', header=None, names=columns, index_col=False, skiprows=0,
                         engine='python', usecols=range(len(columns)))
        # Add the last column as a single entity
        df['Location_Agency'] = df.apply(lambda row: ' '.join(row[len(columns):]), axis=1)

        return df

    def create_stations_xml(self):
        df = self.get_df()
        data_map = {}
        data_map['nets'] = {}
        for i in range(len(df)):

            item = df.iloc[i]
            Station = item.ID

            Network = item.Network
            Latitude = item.Latitude
            Longitude = item.Longitude
            Elevation = item.Elevation
            StartDate = item.StartDate
            StartTime = item.StartTime
            EndDate = item.EndDate
            EndTime = item.EndTime
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
            print("Network:", net)
            network = Network(code=net)
            for station, info in value['stations'].items():
                try:
                    print(station, info)

                    if len(info[4])<10:
                        info[4] = info[4]+".0"
                    if len(info[6]) < 10:
                        info[6] = info[6] + ".0"

                    starttime_str = f'{info[3]} {info[4]}'
                    endtime_str = f'{info[5]} {info[6]}'
                    # Convert the combined string to a datetime object
                    starttime_datetime = datetime.strptime(starttime_str, '%Y-%m-%d %H:%M:%S.%f')
                    endtime_datetime = datetime.strptime(endtime_str, '%Y-%m-%d %H:%M:%S.%f')
                    starttime = UTCDateTime(starttime_datetime)
                    endtime = UTCDateTime(endtime_datetime)
                    "Create a station"
                    station = Station(
                        code=station,
                        latitude=info[0],
                        longitude=info[1],
                        elevation=info[2],
                        creation_date=starttime,
                        termination_date=endtime,
                    )
                        # Add channels to the station
                    if self.channels is not None:
                        for channel_code in self.channels:
                            channel = Channel(
                                code=channel_code,
                                location_code="",  # You can adjust this if needed
                                latitude=info[0],
                                longitude=info[1],
                                elevation=info[2],
                                depth=0,
                                azimuth=0.0,
                                dip=-90.0,
                                sample_rate=100.0,  # Replace with the actual sample rate
                                start_date=UTCDateTime(starttime),
                                end_date=UTCDateTime(endtime),
                            )
                            station.channels.append(channel)
                            if self.respfiles is not None:
                                response = self.find_resp(self, network, station, "", channel, starttime+3600)
                                channel.response = response
                    # Attach the station to the network
                    network.stations.append(station)
                except:
                    print("Cannot include ", station)
            nets.append(network)
            # Create an inventory and attach the network
        inventory = Inventory(networks=nets)
        print(inventory)
        return inventory

    @staticmethod
    def write_xml(path, name, inventory):

        # Write to StationXML file
        xml_filename = os.path.join(path, name)
        xml_filename = f"{xml_filename}_station.xml"
        inventory.write(xml_filename, format="stationxml")
        print(f"StationXML file '{xml_filename}' created successfully.")


if __name__ == "__main__":
    stations_file = "/Users/roberto/Desktop/stations_file.txt"
    path_dest = "/Users/roberto/Desktop"
    name = "antonio_test"
    sc = Convert(stations_file, channels=["HHN", "HHE", "HHZ"])
    data_map = sc.create_stations_xml()
    inventory = sc.get_data_inventory(data_map)
    sc.write_xml(path_dest, name, inventory)