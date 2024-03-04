# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: manage_catalog.py
# Program: surfQuake & ISP
# Date: February 2024
# Purpose: Manage Catalog
# Author: Roberto Cabieces & Thiago C. Junqueira
#  Email: rcabdia@roa.es
# --------------------------------------------------------------------

import pickle
import pandas as pd
import os
from obspy.core.event import Magnitude, FocalMechanism, MomentTensor, Tensor, Catalog, NodalPlanes
from surfquakecore.utils import read_nll_performance
from datetime import datetime
from typing import Union

from surfquakecore.utils.nll_org_errors import computeOriginErrors


class BuildCatalog:
    def __init__(self, loc_folder, output_path, format="QUAKEML", source_summary_file=None, mti_summary_file=None):

        """
        BuildCatalog class helps to join information from all surfquake outputs and create a catalog

        Attributes:
        - loc_folder (str): Path to the folder where the user have the locations files *hyp
        - output_path (str): Output folder path where catalog object and file will be saved
        - format (str): https://docs.obspy.org/packages/autogen/obspy.core.event.Catalog.write.html
        - source_summary_file (str): Path to the output file from source module
        - mti_summary_file (str): Path to the output file from mti module

        Methods:
        - __init__(root_path): Initialize a new instance of BuildCatalog.
        - __merge_info(catalog: Catalog): Merges the information from loc, source and mti
        - build_catalog(): Starts the process to create the catalog from loc files, then calls __merge_info
        """

        self.loc_folder = loc_folder
        self.source_summary_file = source_summary_file
        self.mti_summary_file = mti_summary_file
        self.output_path = output_path
        self.format = format

        if source_summary_file != None:
            self.df = pd.read_csv(self.source_summary_file, sep=";", index_col="date_id")

        if mti_summary_file != None:
            self.df_mti = pd.read_csv(self.mti_summary_file, sep=";", index_col="date_id")

    def __list_directory(self):
        obsfiles = []
        for top_dir, sub_dir, files in os.walk(self.loc_folder):
            for file in files:
                obsfiles.append(os.path.join(top_dir, file))
        obsfiles.sort()
        return obsfiles

    def __merge_info(self, catalog):
        for i, ev in enumerate(catalog):

            for origin in ev.origins:

                origin_time = origin.time.datetime
                origin_time_formatted_string = origin_time.strftime("%m/%d/%Y, %H:%M:%S.%f")
                origin_time_formatted_string_mti = origin_time.strftime("%Y-%m-%d %H:%M:%S.%f")

                if self.source_summary_file != None and len(self.df) > 0:
                     try:
                         source = self.df.loc[origin_time_formatted_string]
                         mg_ml = Magnitude()
                         mg_ml.magnitude_type = 'ML'
                         mg_ml.mag = source['ML']
                         mg_ml.mag_errors.uncertainty = source['ML_error']
                         ev.magnitudes.append(mg_ml)

                         mg_mw = Magnitude()
                         mg_mw.magnitude_type = 'Mw'
                         mg_mw.mag = source['Mw']
                         mg_mw.mag_errors.uncertainty = source['Mw_error']
                         ev.magnitudes.append(mg_mw)

                     except:
                         pass

                if self.mti_summary_file != None and len(self.df_mti) > 0:

                    try:

                        mti = self.df_mti.loc[origin_time_formatted_string_mti]

                        if len(mti) > 0:
                            fm = FocalMechanism()
                            mt = MomentTensor()
                            np = NodalPlanes()
                            np.strike = mti["plane_1_strike"]
                            np.dip = mti["plane_1_dip"]
                            np.rake = mti["plane_1_slip_rake"]
                            mt["nodal_planes"] = np
                            mt["category"] = "Bayesian Inversion"
                            mt["dc"] = mti["dc"]
                            mt["clvd"] = mti["clvd"]
                            mt["iso"] = mti["isotropic_component"]
                            mt["moment_magnitude"] = mti["mw"]
                            mt["scalar_moment"] = mti["mo"]
                            mt["variance"] = mti["cn"]
                            mt["variance_reduction"] = mti["vr"]
                            tensor = Tensor()
                            tensor.m_rr = mti["mrr"]
                            tensor.m_tt = mti["mtt"]
                            tensor.m_pp = mti["mpp"]
                            tensor.m_rt = mti["mrt"]
                            tensor.m_rp = mti["mrp"]
                            tensor.m_tp = mti["mtp"]
                            tensor.defaults["m_rr"] = mti["mrr"]
                            tensor.defaults["m_tt"] = mti["mtt"]
                            tensor.defaults["m_pp"] = mti["mpp"]
                            tensor.defaults["m_rt"] = mti["mrt"]
                            tensor.defaults["m_rp"] = mti["mrp"]
                            tensor.defaults["m_tp"] = mti["mtp"]
                            # build the toy back :-)
                            mt.tensor = tensor
                            fm.moment_tensor = mt
                            ev.focal_mechanisms.append(fm)
                            print(fm)
                    except:
                        pass

            catalog[i] = ev

        return catalog


    def build_catalog_loc(self):
        events_files = self.__list_directory()
        catalog = Catalog()
        for file_ev in events_files:
            try:
                catalog += read_nll_performance.read_nlloc_hyp_ISP(file_ev)
            except:
                pass
        catalog = self.__merge_info(catalog)
        print(catalog.__str__(print_all=True))
        catalog_file_name = os.path.join(self.output_path, "catalog")
        catalog_file_name_pkl = os.path.join(self.output_path, "catalog_obj.pkl")
        catalog.write(filename=catalog_file_name, format=self.format)
        with open(catalog_file_name_pkl, 'wb') as file:
            pickle.dump(catalog, file)
        pd.to_pickle(catalog, filepath_or_buffer=catalog_file_name_pkl)
        print("Catalog saved at ", catalog_file_name,  " format ", self.format)

class WriteCatalog:
    def __init__(self, path_catalog, verbose=False):

        """
        WriteCatalog class helps to filter the catalog obj write in the most Readable

        Attributes:
        - path_catalog (str): Path to the pickle file saved when run build_catalog_loc from BuildCatalog class

        Methods:
        - __init__(root_path): Initialize a new instance of BuildCatalog.
        - filter_time_catalog(verbose=True, **kwargs): filter the catalog in time span
        - filter_geographic_catalog(verbose=True, **kwargs): filter the catalog in geographic contrain and magnitude
        - write_catalog(catalog: Catalog, format, output_path)
        - write_catalog_surf(catalog: Union[Catalog, None], output_path)
        """

        self.path_catalog = path_catalog
        self.catalog = []
        self.verbose = verbose
        self.__test_catalog()

    def show_help(self):

        """
        Display help for the WriteCatalog class.

        Usage:
        wc = WriteCatalog(catalog_path)
        help(wc.filter_time_catalog)
        help(wc.filter_geographic_catalog)
        catalog_filtered = wc.filter_time_catalog(starttime="30/01/2022, 00:00:00.0", endtime="20/02/2022, 00:00:00.0")

        catalog_filtered = wc.filter_geographic_catalog(catalog_filtered, lat_min=42.1, lat_max=43.0, lon_min=0.8, lon_max=1.5,
                                                        depth_min=-10, depth_max=20, mag_min=3.4, mag_max=3.9)
        wc.write_catalog_surf(catalog=catalog_filtered, output_path=output_path)
        """
        return self.__doc__

    def __test_catalog(self):

        try:
            self.catalog = pd.read_pickle(self.path_catalog)
            if self.verbose:
                print(self.catalog.__str__(print_all=True))
                print("Loaded Catalog")
        except:
            raise ValueError("file is not a valid catalog")

    def filter_time_catalog(self, verbose=False, **kwargs):

        """
        Filter the catalog readed in the class instantiation
        verbose: bool:
        **kwargs
        starttime :str: starttime to filter the catalog in format %d/%m/%Y, %H:%M:%S.%f
        endtime :str: endtime to filter the catalog in format %d/%m/%Y, %H:%M:%S.%f
        example:
        wc = WriteCatalog(catalog_path)
        catalog_filtered = wc.filter_time_catalog(starttime="30/01/2022, 00:00:00.0",
        endtime="20/02/2022, 00:00:00.0")
        return :catalog obj:
        """

        date_format = "%d/%m/%Y, %H:%M:%S.%f"
        catalog_filtered = Catalog()
        starttime = kwargs.pop('starttime', [])
        endtime = kwargs.pop('endtime', [])

        starttime = datetime.strptime(starttime, date_format)
        endtime = datetime.strptime(endtime, date_format)

        for i, ev in enumerate(self.catalog):

            for origin in ev.origins:

                origin_time = origin.time.datetime

                if starttime <= origin_time <= endtime:
                    catalog_filtered += ev
        if len(catalog_filtered) == 0:
            print("Check if there is events in catalog time span or the format is valid , ", date_format)

        if verbose:
            print(catalog_filtered.__str__(print_all=True))
            print("Filtered Catalog in time")

        return catalog_filtered

    def filter_geographic_catalog(self, catalog: Union[Catalog, None], verbose=False, **kwargs):

        """
        Filter the catalog readed in the class instantiation or the catalog provided in bu the user when the method
        is called
        verbose: bool:
        catalog obj (optional), if not found it uses catalog from the catalog attribute
        **kwargs --> All keys must be used to filter success.

        lat_min:float
        lat_max:float
        lon_min:float
        lon_max:float
        depth_min:float: km
        depth_max:float: km
        mag_min:float
        mag_max:float

        return :catalog obj:
        """

        if catalog is None:
            # This option proceed to filter the attribute catalog
            catalog = self.catalog.copy()

        catalog_filtered = Catalog()

        lat_min = kwargs.pop('lat_min', None)
        lat_max = kwargs.pop('lat_max', None)
        lon_min = kwargs.pop('lon_min', None)
        lon_max = kwargs.pop('lon_max', None)
        depth_min = kwargs.pop('depth_min', None)
        depth_max = kwargs.pop('depth_max', None)
        mag_min = kwargs.pop('mag_min', None)
        mag_max = kwargs.pop('mag_max', None)
        if None in kwargs.values():
            raise ValueError("Fill all searching fields, lat_min, lat_max, lon_min, lon_max, depth_min, "
                             "depth_max, mag_min, mag_max")
        # checks
        if lat_min < lat_max and lon_min < lon_max:
            for i, ev in enumerate(catalog):
                for origin in ev.origins:
                    lat_origin = origin.latitude
                    lon_origin = origin.longitude
                    depth_origin = origin.depth*1E-3
                    if len(ev.magnitudes)>0:
                        magnitude = ev.magnitudes[0].mag
                    else:
                        magnitude = None
                    if lat_min <= lat_origin <= lat_max and lon_min <= lon_origin <= lon_max:
                        if depth_min <= depth_origin <= depth_max:
                            if isinstance(mag_min, float) and isinstance(mag_max, float):
                                if isinstance(magnitude, float) and mag_min <= magnitude <= mag_max:
                                    catalog_filtered += ev
                            elif mag_min is None and mag_max is None:
                                catalog_filtered += ev


        if verbose:
            print(catalog_filtered.__str__(print_all=True))
            print("Filtered Catalog in space and magnitude")

        return catalog_filtered

    def write_catalog(self, catalog: Catalog, format: str, output_path: str):

        catalog_file_name = os.path.join(self.output_path, "catalog")
        catalog_file_name_pkl = os.path.join(output_path, "catalog_obj.pkl")
        catalog.write(filename=catalog_file_name, format=format)

        with open(catalog_file_name_pkl, 'wb') as file:
            pickle.dump(catalog, file)

        pd.to_pickle(catalog, filepath_or_buffer=catalog_file_name_pkl)
        print("Catalog saved at ", catalog_file_name, " format ", self.format)

    def write_catalog_surf(self, catalog: Union[Catalog, None], output_path):

        """
        Writes in human language the catalog instantiated with the class
        catalog obj (optional), if not found it uses catalog from the catalog attribute
        """

        if catalog is None:
            # This option proceed to filter the attribute catalog
            catalog = self.catalog.copy()

        with open(output_path, 'w') as file:
            for i, ev in enumerate(catalog):
                for origin in ev.origins:
                    modified_origin_90, confidence_ellipsoid, origin_uncertainty = computeOriginErrors(origin)
                    origin_time = origin.time.datetime.strftime("%d/%m/%Y %H:%M:%S.%f")
                    lat_origin = "{:.4f}".format(origin.latitude)
                    lon_origin = "{:.4f}".format(origin.longitude)
                    depth_origin = "{:.1f}".format(origin.depth * 1E-3)
                    depth_error = "{:.1f}".format(origin.depth_errors["uncertainty"]*1E-3)
                    rms = "{:.2f}".format(origin.quality.standard_error)
                    smin = "{:.1f}".format(origin_uncertainty.min_horizontal_uncertainty)
                    smax = "{:.1f}".format(origin_uncertainty.max_horizontal_uncertainty)
                    elipse_azimuth = "{:.1f}".format(origin_uncertainty.azimuth_max_horizontal_uncertainty)
                    gap = "{:.1f}".format(origin.quality.azimuthal_gap)
                    min_dist = "{:.3f}".format(origin.quality.minimum_distance)
                    max_dist = "{:.3f}".format(origin.quality.maximum_distance)
                    confidence_level = "{:.1f}".format(origin_uncertainty.confidence_level)

                    file.write(f"Event {i+1}: Date {origin_time} rms {rms} s Lat {lat_origin} Lon {lon_origin} "
                               f"Depth {depth_origin} km +- {depth_error} min_dist {min_dist} max_dist {max_dist} "
                               f"smin {smin} km smax {smax} km ell_azimuth {elipse_azimuth} gap {gap} "
                               f"conf_lev {confidence_level} %\n")

                    # magnitudes
                    if len(ev.magnitudes) > 0:
                        for magnitude in ev.magnitudes:
                            mag_type = magnitude.magnitude_type
                            mag = magnitude.mag
                            mag_error = magnitude.mag_errors["uncertainty"]
                            if isinstance(mag, float):
                                file.write(f"Magnitude: {mag_type} {mag} +- {mag_error}\n")
                    else:
                        file.write(f"Magnitude: None\n")


                    if len(ev.focal_mechanisms) > 0:
                        fm = ev.focal_mechanisms[0]
                        Mw = "{:.2f}".format(fm.moment_tensor["moment_magnitude"])
                        Mo = "{:.2e}".format(fm.moment_tensor["scalar_moment"])
                        dc = "{:.2f}".format(fm.moment_tensor["dc"])
                        clvd = "{:.2f}".format(fm.moment_tensor["clvd"])
                        iso = "{:.2f}".format(fm.moment_tensor["iso"])
                        variance = "{:.2f}".format(fm.moment_tensor["variance_reduction"])
                        mrr = "{:.2e}".format(fm.moment_tensor["tensor"].m_rr)
                        mtt = "{:.2e}".format(fm.moment_tensor["tensor"].m_pp)
                        mpp = "{:.2e}".format(fm.moment_tensor["tensor"].m_tt)
                        mrp = "{:.2e}".format(fm.moment_tensor["tensor"].m_rp)
                        mrt = "{:.2e}".format(fm.moment_tensor["tensor"].m_rt)
                        mtp = "{:.2e}".format(fm.moment_tensor["tensor"].m_rt)
                        strike = "{:.1f}".format(fm.moment_tensor["nodal_planes"].strike)
                        dip = "{:.1f}".format(fm.moment_tensor["nodal_planes"].dip)
                        rake = "{:.1f}".format(fm.moment_tensor["nodal_planes"].rake)
                        file.write(f"Moment Tensor Solution:\n")
                        file.write(f"Mw {Mw} Mo {Mo} Nm DC {dc} % CLVD {clvd} % iso {iso} % variance_red {variance}\n")
                        file.write(f"Nodal Plane: Strike {strike} Dip {dip} Rake {rake}\n")
                        file.write(f"Moment Tensor: mrr {mrr} mtt {mtt} mpp {mpp} mrp {mrp} mrt {mrt} mrp {mtp}\n")

                    file.write(f"station phase polarity date time time_residual distance_degrees "
                               f"distance_km azimuth takeoff_angle\n")

                    for arrival in origin.arrivals:
                        azimuth = "{:.1f}".format(arrival.azimuth)
                        date = arrival.date.strftime("%d/%m/%Y %H:%M:%S.%f")
                        distance_degrees = "{:.1f}".format(arrival.distance_degrees)
                        distance_km = "{:.1f}".format(arrival.distance_km)
                        phase = arrival.phase
                        polarity = arrival.polarity
                        station = arrival.station
                        time_residual = "{:.2f}".format(arrival.time_residual)
                        #time_weight = "{:.2f}".format(arrival.time_weight)
                        # instrument = arrival.instrument
                        takeoff_angle = "{:.1f}".format(arrival.takeoff_angle)
                        file.write(f"{station} {phase} {polarity} {date} {time_residual} {distance_degrees} "
                                   f"{distance_km} {azimuth} {takeoff_angle}\n")

                    file.write(f"\n\n")



#if __name__ == "__main__":
    # path_events_file = "/Volumes/LaCie/surfquake_test/test_nll_final"
    # path_source_file = "/Volumes/LaCie/surfquake_test/catalog_output/sources.txt"
    # output_path = "/Volumes/LaCie/surfquake_test/catalog_output"
    #bc = BuildCatalog(loc_folder=path_events_file, source_summary_file=path_source_file, output_path=output_path,
    #                  format="QUAKEML")
    #bc.build_catalog_loc()
    # catalog_path = "/Volumes/LaCie/all_andorra/catalog/catalog_obj.pkl"
    # output_path = "/Volumes/LaCie/all_andorra/catalog/catalog_surf.txt"
    # wc = WriteCatalog(catalog_path)
    # wc.write_catalog_surf(output_path)

