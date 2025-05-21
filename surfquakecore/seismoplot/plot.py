#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot with sorting options and metadata
"""
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from obspy.core.trace import Trace
from obspy.core.inventory import Inventory
from obspy.geodetics import gps2dist_azimuth
import matplotlib
matplotlib.use("TkAgg")

class PlotProj:
    def __init__(self,
                 trace_list: List[Trace],
                 metadata: Optional[Union[dict, Inventory]] = None,
                 epicenter: Optional[Tuple[float, float]] = None):
        """
        Parameters
        ----------
        trace_list : List[Trace]
            List of ObsPy Trace objects.
        metadata : dict or Inventory, optional
            Metadata with station coordinates.
        epicenter : tuple of (latitude, longitude), optional
            Coordinates of the event epicenter.
        """
        self.trace_list = trace_list
        self.metadata = metadata
        self.epicenter = epicenter
        self._dist_az_cache = {}

    def _get_station_coords(self, trace: Trace) -> Optional[Tuple[float, float]]:
        """
        Retrieve station coordinates from metadata.
        """
        if self.metadata is None:
            return None

        network = trace.stats.network
        station = trace.stats.station

        if isinstance(self.metadata, Inventory):
            try:
                coords = self.metadata.get_coordinates(f"{network}.{station}")
                return coords['latitude'], coords['longitude']
            except Exception:
                return None
        elif isinstance(self.metadata, dict):
            return self.metadata.get(f"{network}.{station}")

        return None

    def _compute_distance_azimuth(self, trace: Trace) -> Tuple[float, float]:
        """
        Calculate distance (in km) and back-azimuth from epicenter to station.
        """
        if trace.id in self._dist_az_cache:
            return self._dist_az_cache[trace.id]

        coords = self._get_station_coords(trace)
        if coords is None or self.epicenter is None:
            return float('inf'), float('inf')

        epi_lat, epi_lon = self.epicenter
        sta_lat, sta_lon = coords
        dist_m, az, baz = gps2dist_azimuth(epi_lat, epi_lon, sta_lat, sta_lon)
        dist_km = dist_m / 1000.0

        self._dist_az_cache[trace.id] = (dist_km, baz)
        return dist_km, baz

    def plot(self,
             traces_per_fig: int = None,
             sort_by: Optional[str] = None):
        """
        Plot the traces with a specified number of subplots per figure.

        Parameters
        ----------
        traces_per_fig : int, optional
            Number of traces (subplots) per figure.
        sort_by : str, optional
            Options: 'distance', 'backazimuth', or None.
        """
        traces = list(self.trace_list)  # Ensure it's a true list

        if sort_by == 'distance':
            traces.sort(key=lambda tr: self._compute_distance_azimuth(tr)[0])
        elif sort_by == 'backazimuth':
            traces.sort(key=lambda tr: self._compute_distance_azimuth(tr)[1])

        n_traces = len(traces)
        if n_traces == 0:
            print("No traces to plot.")
            return

        if traces_per_fig is None:
            traces_per_fig = n_traces

        for i in range(0, n_traces, traces_per_fig):
            sub_traces = traces[i:i + traces_per_fig]
            n_subplots = len(sub_traces)
            fig, axs = plt.subplots(n_subplots, 1, figsize=(10, 2 * n_subplots), sharex=True)
            if n_subplots == 1:
                axs = [axs]

            for ax, trace in zip(axs, sub_traces):
                dist, baz = self._compute_distance_azimuth(trace)
                ax.plot(trace.times(), trace.data, linewidth=0.75, label=f"{trace.id} | Dist: {dist:.1f} km | Baz: {baz:.1f}°")
                #ax.set_title(f"{trace.id} | Dist: {dist:.1f} km | Baz: {baz:.1f}°", fontsize=9)

            plt.legend()
            plt.tight_layout()
            plt.show(block=True)

if __name__ == "__main__":
    from obspy import read, read_inventory

    # Load stream and metadata
    st = read()  # your traces
    inv = read_inventory()  # or a dict with station coords: {'NET.STA': (lat, lon)}

    # Define epicenter (latitude, longitude)
    epicenter = (10.0, 20.0)

    # Initialize and plot
    plotter = PlotProj(st, metadata=inv, epicenter=epicenter)
    plotter.plot(traces_per_fig=3, sort_by='distance')