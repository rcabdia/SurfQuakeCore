import pickle
import pandas as pd
#from obspy import read_events
import os
from obspy.core.event import Magnitude, FocalMechanism, MomentTensor, Tensor, Catalog
from surfquakecore.utils import read_nll_performance


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

if __name__ == "__main__":
    path_events_file = "/Volumes/LaCie/surfquake_test/test_nll_final"
    path_source_file = "/Volumes/LaCie/surfquake_test/catalog_output/sources.txt"
    output_path = "/Volumes/LaCie/surfquake_test/catalog_output"
    bc = BuildCatalog(loc_folder=path_events_file, source_summary_file=path_source_file, output_path=output_path,
                      format="QUAKEML")
    bc.build_catalog_loc()
