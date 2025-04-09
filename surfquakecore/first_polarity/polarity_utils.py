#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
polarity_utils

"""
from obspy import read_inventory, Stream
from obspy.geodetics import gps2dist_azimuth
import numpy as np

def rotate_stream_to_GEAC(stream, inventory, epicenter_lat, epicenter_lon):
    """
    Rotates an ObsPy Stream to Great Circle Arc Coordinates (GEAC) using an inventory.
    Includes the vertical component ("Z") in the output.

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
        # Extract components
        components = {tr.stats.channel[-1]: tr for tr in traces}  # {'E': Trace, 'N': Trace, 'Z': Trace}

        if 'N' not in components or 'E' not in components:
            print(f"Skipping station {station_id}: Missing N or E component.")
            continue

        tr_n = components['N']
        tr_e = components['E']
        tr_z = components.get('Z', None)  # Z may not exist, handle it safely

        # Step 3: Check sampling rate and sample count
        if tr_n.stats.sampling_rate != tr_e.stats.sampling_rate:
            print(f"Skipping station {station_id}: Sampling rates do not match.")
            continue
        if len(tr_n.data) != len(tr_e.data):
            print(f"Skipping station {station_id}: Number of samples do not match.")
            continue
        if tr_z and (tr_n.stats.sampling_rate != tr_z.stats.sampling_rate or len(tr_n.data) != len(tr_z.data)):
            print(f"Skipping station {station_id}: Z component sampling rate or samples do not match.")
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
        rotated_traces = [tr_r, tr_t]

        # Include Z component if available
        if tr_z:
            rotated_traces.append(tr_z.copy())  # Copy Z component as is

        rotated_streams.append(Stream(traces=rotated_traces))

    return rotated_streams