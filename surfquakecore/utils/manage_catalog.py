import pickle
import pandas as pd
#from obspy import read_events
import os
from obspy.core.event import Magnitude, FocalMechanism, MomentTensor, Tensor, Catalog
from surfquakecore.utils import read_nll_performance
from datetime import datetime
from typing import Union

class BuildCatalog:
    def __init__(self, loc_folder, output_path, format="QUAKEML", source_summary_file=None, mti_summary_file=None):

        self.loc_folder = loc_folder
        self.source_summary_file = source_summary_file
        self.mti_summary_file = mti_summary_file
        self.output_path = output_path
        self.format = format

        if source_summary_file != None:
            self.df = pd.read_csv(self.source_summary_file, sep = ";", index_col="date_id")

        if mti_summary_file != None:
            self.df_mti = pd.read_csv(self.mti_summary_file, sep = ";", index_col="date_id")

    def list_directory(self):
        obsfiles = []
        for top_dir, sub_dir, files in os.walk(self.loc_folder):
            for file in files:
                obsfiles.append(os.path.join(top_dir, file))
        obsfiles.sort()
        return obsfiles

    def merge_info(self, catalog):
        for i, ev in enumerate(catalog):

            for origin in ev.origins:

                origin_time = origin.time.datetime
                origin_time_formatted_string = origin_time.strftime("%m/%d/%Y, %H:%M:%S.%f")
                origin_time_formatted_string_mti = origin_time.strftime("%Y-%m-%d %H:%M:%S.%f")

                if self.source_summary_file != None and len(self.df) > 0:
                     try:
                         source = self.df.loc[origin_time_formatted_string]
                         mg_ml = Magnitude()
                         mg_mw = Magnitude()
                         mg_ml.magnitude_type = 'ML'
                         mg_ml.mag = source['ML']
                         mg_ml.mag_errors.uncertainty = source['ML_error']
                         ev.magnitudes.append(mg_ml)
                         mg_ml.magnitude_type = 'Mw'
                         mg_ml.mag = source['Mw']
                         mg_ml.mag_errors.uncertainty = source['Mw_error']
                         ev.magnitudes.append(mg_mw)
                     except:
                         pass

                if self.mti_summary_file != None and len(self.df_mti) > 0:

                    try:

                        mti = self.df_mti.loc[origin_time_formatted_string_mti]

                        if len(mti) > 0:
                            fm = FocalMechanism()
                            mt = MomentTensor()
                            mt["category"] = "Bayesian Inversion"
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
        events_files = self.list_directory()
        catalog = Catalog()
        for file_ev in events_files:
            try:
                catalog += read_nll_performance.read_nlloc_hyp_ISP(file_ev)
            except:
                pass
        catalog = self.merge_info(catalog)
        print(catalog.__str__(print_all=True))
        catalog_file_name = os.path.join(self.output_path, "catalog")
        catalog_file_name_pkl = os.path.join(self.output_path, "catalog_obj.pkl")
        catalog.write(filename=catalog_file_name, format=self.format)
        with open(catalog_file_name_pkl, 'wb') as file:
            pickle.dump(catalog, file)
        pd.to_pickle(catalog, filepath_or_buffer=catalog_file_name_pkl)
        print("Catalog saved at ", catalog_file_name,  " format ", self.format)

class  WriteCatalog:
    def __init__(self, path_catalog):
        self.path_catalog = path_catalog
        self.ctalog = []
        self.__test_catalog()

    def __test_catalog(self):
        catalog = []
        try:
            self.catalog = pd.read_pickle(self.path_catalog)
            print(catalog.__str__(print_all=True))
        except:
            raise ValueError("file is not a valid catalog")

    def filter_time_catalog(self, **kwargs):
        # Date input format "%d/%m/%Y, %H:%M:%S.%f"
        date_format = "%d/%m/%Y, %H:%M:%S.%f"

        catalog = self.catalog.copy()
        starttime = kwargs.pop('starttime', [])
        endtime = kwargs.pop('endtime', [])

        starttime = datetime.strptime(starttime, date_format)
        endtime = datetime.strptime(endtime, date_format)

        for i, ev in enumerate(self.catalog):

            for origin in ev.origins:

                origin_time = origin.time.datetime

                if starttime <= origin_time <= endtime:
                    catalog[i] = ev
        if len(catalog) == 0:
            print("Check if there is events in catalog time span or the format is valid , ", date_format)


        return catalog

    def filter_geographic_catalog(self, catalog: Union[Catalog, None], **kwargs):
        if catalog is None:
            # This option proceed to filter the attribute catalog
            catalog = self.catalog.copy()

        lat_min = kwargs.pop('lat_min', None)
        lat_max = kwargs.pop('lat_max', None)
        lon_min = kwargs.pop('lon_min', None)
        lon_max = kwargs.pop('lon_min', None)
        depth_min = kwargs.pop('depth_min', None)
        depth_max = kwargs.pop('depth_max', None)
        mag_min = kwargs.pop('mag_min', None)
        mag_max = kwargs.pop('mag_max', None)
        if None in kwargs.values():
            raise ValueError("Fill all searching fields, lat_min, lat_max, lon_min, lon_max, depth_min, "
                             "depth_max, mag_min, mag_max")
        check = True
        # checks
        if lat_min < lat_max and lon_min < lon_max:
            for i, ev in enumerate(self.catalog):
                for origin in ev.origins:
                    lat_origin = origin.latitude
                    lon_origin = origin.longitude
                    depth_origin = origin.depth
                    if len(ev.magnitudes)>0:
                        magnitude = ev.magnitudes[0].mag
                    else:
                        magnitude = None
                    if lat_min <= lat_origin <= lat_max and lon_min <= lon_origin <= lon_max:
                        if depth_min <= depth_origin <= depth_max:
                            if isinstance(magnitude, float) and mag_min <= magnitude <= mag_max:
                                catalog[i] = ev

        return catalog

    def write_catalog_surf(self, output_path):
        for i, ev in enumerate(self.catalog):
            for origin in ev.origins:
                lat_origin = origin.latitude
                lon_origin = origin.longitude
                depth_origin = origin.depth
                


    # def write_catalog_to_file(catalog, filename):
    #     with open(filename, 'w') as file:
    #         for i, event in enumerate(catalog.events, start=1):
    #             file.write(f"Event {i}: Lat {event.latitude} Lon {event.longitude} Depth {event.depth} km\n")
    #             file.write(f"Focal Mechanism: Strike {event.strike} Dip {event.dip} Rake {event.rake}\n\n")













if __name__ == "__main__":
    path_events_file = "/Volumes/LaCie/surfquake_test/test_nll_final"
    path_source_file = "/Volumes/LaCie/surfquake_test/catalog_output/sources.txt"
    output_path = "/Volumes/LaCie/surfquake_test/catalog_output"
    bc = BuildCatalog(loc_folder=path_events_file, source_summary_file=path_source_file, output_path=output_path,
                      format="QUAKEML")
    bc.build_catalog_loc()
