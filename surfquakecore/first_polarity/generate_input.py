# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: generate_input.py
# Program: surfQuake & ISP
# Date: March 2025
# Purpose: Project Manager
# Author: Roberto Cabieces & Thiago C. Junqueira
#  Email: rcabdia@roa.es
# --------------------------------------------------------------------

import os
from obspy import read_inventory
from pandas import DataFrame
from surfquakecore.first_polarity.polarity_utils import rotate_stream_to_GEAC
from surfquakecore.utils import read_nll_performance
import numpy as np
from surfquakecore.moment_tensor.sq_isola_tools.mti_utilities import MTIManager
import pandas as pd


class GenerateInput:
    def __init__(self, project:str, location_folder: str, inventory: str):

        self.location_folder = location_folder
        self.project = project
        self.inventory = inventory


    def get_hyp_files(self):
        """
        Returns a list of file paths with .hyp extension in the specified directory.

        Args: directory (str): Path to the directory to search in

        Returns: list: List of full file paths ending with .hyp
        """
        hyp_files = []
        for root, dirs, files in os.walk(self.location_folder):
            for file in files:
                if file.endswith('.hyp'):
                    full_path = os.path.join(root, file)
                    hyp_files.append(full_path)

        return hyp_files

    @staticmethod
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


    def procees_waveforms(self, files_list, origin_date, inventory_path,  output_folder):

        inventory = read_inventory(inventory_path)

        st = MTIManager.default_processing(
            files_path=files_list,
            origin_time=origin_date,
            inventory=inventory,
            output_directory=output_folder,
            regional=True,
            remove_response=False,
            save_stream_plot=False
        )
        return st

    def get_amplitude_ratios(self, st_rotated_list:list, df_arrivals:DataFrame):
        print("getting amplitude ratios")
        for st in st_rotated_list:
            station = st[0].stats.station
            df_new = df_arrivals[df_arrivals["station"] == station]
            if df_new.empty:
                #print(f"No arrivals found for station: {station_name}")
                #return None
                pass
            else:
                df_s = df_new[df_new["phase"] == "S"]
                df_p = df_new[df_new["phase"] == "P"]

                if df_s.empty:
                    arrival_time_s = None
                else:
                    max_weight_row_s = df_s.loc[df_s["weight"].idxmax()]
                    arrival_time_s = max_weight_row_s["arrival_time"]
                    sh_amplitude, sv_amplitude, error = self.process_sh_sv(st, arrival_time_s)
                    print(sh_amplitude, sv_amplitude, error)
                if df_p.empty:
                    arrival_time_p = None
                else:
                    max_weight_row_p = df_p.loc[df_p["weight"].idxmax()]
                    arrival_time_p = max_weight_row_p["arrival_time"]
                    if arrival_time_p is not None and arrival_time_s is not None:
                        p_amplitude, sv_amplitude, error = self.process_p_sv(st, arrival_time_p, arrival_time_s)
                        print(p_amplitude, sv_amplitude, error)


    def process_p_sv(self, st, arrival_time_p, arrival_time_sv, win=5):
        st_trim_p = st.copy()
        st_trim_sv = st.copy()
        st_trim_p.trim(starttime=arrival_time_p-win, endtime=arrival_time_p+2*win)
        st_trim_sv.trim(starttime=arrival_time_sv - win, endtime=arrival_time_sv + 2 * win)
        tr_p = st_trim_p.select(component="Z")[0]
        tr_sv = st_trim_sv.select(component="R")[0]

        p_amplitude = np.max(np.abs(tr_p.data))
        p_rms = np.mean(np.abs(tr_p.data))
        sv_amplitude = np.max(np.abs(tr_sv.data))
        sv_rms = np.mean(np.abs(tr_sv.data))
        error_sh_sv = sv_rms / p_rms

        return p_amplitude, sv_amplitude, error_sh_sv

    def process_sh_sv(self, st_trim, arrival_time, win=5):
        st_trim.trim(starttime=arrival_time-win, endtime=arrival_time+2*win)
        tr_sh = st_trim.select(component="T")[0]
        tr_sv = st_trim.select(component="R")[0]
        sh_amplitude = np.max(np.abs(tr_sh.data))
        sh_rms = np.mean(np.abs(tr_sh.data))
        sv_amplitude = np.max(np.abs(tr_sv.data))
        sv_rms = np.mean(np.abs(tr_sv.data))
        error_sh_sv = sh_rms/sv_rms
        return sh_amplitude, sv_amplitude, error_sh_sv

    def get_project_files(self, stations_list, time_span_interval):
        self.project.filter_project_keys(station=stations_list)
        data_files = self.project.filter_time(starttime=time_span_interval[0], endtime=time_span_interval[1])
        return data_files

    def generate_full_input(self):
        hyp_files = self.get_hyp_files()
        for hyp_file in hyp_files:
            df_arrivals, stations, time_span_interval, basic_info = self.get_info(hyp_file)
            files_list = self.get_project_files(stations, time_span_interval)
            st = self.procees_waveforms(files_list, basic_info["o_time"], inventory_path=inventory_path,  output_folder="")
            st_rotated_list = rotate_stream_to_GEAC(st, inventory_path, basic_info["latitude"], basic_info["longitude"])
            self.get_amplitude_ratios(st_rotated_list, df_arrivals)



if __name__ == '__main__':

    path_to_data = "/Users/admin/Documents/iMacROA/test_surfquake/Andorra/inputs/waveforms_cut"
    path_to_project = "/Users/admin/Documents/iMacROA/test_surfquake/Andorra/project/project.pkl"
    path_to_hyp = "/Users/admin/Documents/iMacROA/test_surfquake/Andorra/outputs/nll/all_loc/location.20220201.023032.grid0.loc.hyp"
    inventory_path = "/Users/admin/Documents/iMacROA/test_surfquake/Andorra/inputs/metadata/inv_all.xml"
    gi = GenerateInput(path_to_project, path_to_hyp, inventory_path)
    #gi.generate_full_input()





