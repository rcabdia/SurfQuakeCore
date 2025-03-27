from multiprocessing import freeze_support
from obspy import read_inventory, Stream
from obspy.geodetics import gps2dist_azimuth
from surfquakecore.utils import read_nll_performance
from surfquakecore.project.surf_project import SurfProject
import numpy as np
from surfquakecore.moment_tensor.sq_isola_tools.mti_utilities import MTIManager
import pandas as pd

path_to_data = "/Users/robertocabiecesdiaz/Documents/test_surfquake/Andorra/inputs/waveforms_cut"
path_to_project = "/Users/robertocabiecesdiaz/Documents/test_surfquake/Andorra/project/project.pkl"
path_hyp = "/Users/robertocabiecesdiaz/Documents/test_surfquake/Andorra/outputs/nll/all_loc/location.20220201.023032.grid0.loc.hyp"
inventory_path = "/Users/robertocabiecesdiaz/Documents/test_surfquake/Andorra/inputs/metadata/inv_all.xml"

def get_info(path_hyp):
    basic_info = {}
    cat = read_nll_performance.read_nlloc_hyp_ISP(path_hyp)
    event = cat[0]
    arrivals = event["origins"][0]["arrivals"]
    basic_info["o_time"] = event["origins"][0].time
    basic_info["latitude"] = event["origins"][0].latitude
    basic_info["longitude"] = event["origins"][0].longitude
    basic_info["depth"] = event["origins"][0].depth*1E-3
    stations = []
    time_span = []
    # Create a list to store arrival data
    arrivals_data = []

    for time_arrival in arrivals:
        arrivals_data.append({
            "station": time_arrival.station,
            "arrival_time": time_arrival.date,
            "phase": time_arrival.phase,
            "weight": time_arrival.time_weight
        })
        if time_arrival.station not in stations:
            stations.append(time_arrival.station)

        time_span.append(time_arrival.date)

        #print(time_arrival.station, time_arrival.date, time_arrival.phase, time_arrival.time_weight)

    # Convert arrivals list to a Pandas DataFrame
    df_arrivals = pd.DataFrame(arrivals_data)
    min_date = np.min(time_span)
    max_date = np.max(time_span)
    time_span_interval = [min_date, max_date]
    stations = '|'.join(stations)
    return df_arrivals, stations, time_span_interval, basic_info


def rotate_stream_to_GEAC(stream, inventory, epicenter_lat, epicenter_lon):
    """
    Rotates an ObsPy Stream to Great Circle Arc Coordinates (GEAC) using an inventory.

    Args:
        stream (Stream): The ObsPy Stream object containing the traces.
        inventory (Inventory): ObsPy Inventory containing station metadata.
        epicenter_lat (float): Latitude of the epicenter.
        epicenter_lon (float): Longitude of the epicenter.

    Returns:
        list: A list of ObsPy Stream objects, each corresponding to a station with rotated components.
    """
    inventory = read_inventory(inventory)
    # Step 1: Group traces by station
    station_dict = {}
    for trace in stream:
        station_id = trace.stats.network + "." + trace.stats.station
        if station_id not in station_dict:
            station_dict[station_id] = []
        station_dict[station_id].append(trace)

    rotated_streams = []

    # Step 2: Process each station
    for station_id, traces in station_dict.items():
        if len(traces) < 2:
            print(f"Skipping station {station_id}: Not enough components for rotation.")
            continue

        # Extract components
        components = {tr.stats.channel[-1]: tr for tr in traces}  # {'E': Trace, 'N': Trace, ...}

        if 'N' not in components or 'E' not in components:
            print(f"Skipping station {station_id}: Missing N or E component.")
            continue

        tr_n = components['N']
        tr_e = components['E']

        # Step 3: Check sampling rate and sample count
        if tr_n.stats.sampling_rate != tr_e.stats.sampling_rate:
            print(f"Skipping station {station_id}: Sampling rates do not match.")
            continue
        if len(tr_n.data) != len(tr_e.data):
            print(f"Skipping station {station_id}: Number of samples do not match.")
            continue

        # Step 4: Get station coordinates from inventory
        try:
            network_code, station_code = station_id.split(".")
            station = inventory.select(network=network_code, station=station_code)[0][0]
            station_lat, station_lon = station.latitude, station.longitude
        except Exception as e:
            print(f"Skipping station {station_id}: Station not found in inventory. ({e})")
            continue

        # Step 5: Compute back azimuth
        _, baz, _ = gps2dist_azimuth(epicenter_lat, epicenter_lon, station_lat, station_lon)

        # Step 6: Rotate to GEAC (Radial & Transverse)
        theta = np.deg2rad(baz)
        data_r = tr_n.data * np.cos(theta) + tr_e.data * np.sin(theta)
        data_t = -tr_n.data * np.sin(theta) + tr_e.data * np.cos(theta)

        # Step 7: Create new rotated traces
        tr_r = tr_n.copy()
        tr_r.data = data_r
        tr_r.stats.channel = tr_r.stats.channel[:-1] + 'R'  # Rename to Radial

        tr_t = tr_e.copy()
        tr_t.data = data_t
        tr_t.stats.channel = tr_t.stats.channel[:-1] + 'T'  # Rename to Transverse

        # Step 8: Store in new stream
        rotated_stream = Stream(traces=[tr_r, tr_t])
        rotated_streams.append(rotated_stream)

    return rotated_streams

def procees_waveforms(files_list, origin_date, inventory_path,  output_folder):

    inventory = read_inventory(inventory_path)

    st = MTIManager.default_processing(
        files_path=files_list,
        origin_time=origin_date,
        inventory=inventory,
        output_directory=output_folder,
        regional=True,
        remove_response=True,
        save_stream_plot=False
    )
    return st


if __name__ == '__main__':

    #freeze_support()

    df_arrivals, stations, time_span_interval, basic_info = get_info(path_hyp)
    df_new = df_arrivals[df_arrivals["station"] == "VALC"]
    df_s = df_new[df_new["phase"] == "S"]
    print(df_s)




    # project = SurfProject.load_project(path_to_project)
    # project.filter_project_keys(station=stations)
    # data_files = project.filter_time(starttime=time_span_interval[0], endtime=time_span_interval[1])
    # # sp.search_files()
    # print(data_files)
    #
    # st = procees_waveforms(data_files, basic_info["o_time"], inventory_path=inventory_path,  output_folder="")
    # st_rotated = rotate_stream_to_GEAC(st, inventory_path, basic_info["latitude"], basic_info["longitude"])






