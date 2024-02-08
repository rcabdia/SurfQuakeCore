import pandas as pd
from obspy import read_events
import os
from obspy.core.event import Magnitude, FocalMechanism

## How to fusion information o the catalog

class BuildCatalog:
    def __init__(self, loc_folder, output_path, format = "QUAKEML", source_summary_file=None, mti_summary_file=None):

        self.loc_folder = loc_folder
        self.source_summary_file = source_summary_file
        self.mti_folder = mti_summary_file
        self.output_path = output_path
        self.format = format

        if source_summary_file != None:
            self.df = pd.read_csv(self.source_summary_file, sep = ";", index_col="date_id")

    def list_directory(self):
        obsfiles = []
        for top_dir, sub_dir, files in os.walk(self.loc_folder):
            for file in files:
                obsfiles.append(os.path.join(top_dir, file))
        obsfiles.sort()
        return obsfiles

    def merge_info(self, catalog):
        for i, ev in enumerate(catalog):
            #fm = FocalMechanism()
            for origin in ev.origins:
                lat = origin.latitude
                lon = origin.longitude
                depth = origin.depth
                origin_time = origin.time.datetime
                origin_time_formatted_string = origin_time.strftime("%m/%d/%Y, %H:%M:%S.%f")
                id = [lat, lon, depth, origin_time_formatted_string]
                #print(origin_time_formatted_string)
                if self.source_summary_file != None and len(self.df) > 0:

                     try:
                         # TODO Test using origin_time_formatted_string
                         #source = self.df.loc["02/01/2022, 02:02:59.997001"]
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
                         # print("No Matches for, ", origin_time_formatted_string)
                         pass

            catalog[i] = ev

        return catalog


    def build_catalog_loc(self):
        events_files = self.list_directory()
        catalog = read_events()
        for file_ev in events_files:
            try:
                catalog += read_events(file_ev)
            except:
                pass
        catalog = self.merge_info(catalog)
        print(catalog.__str__(print_all=True))
        catalog_file_name = os.path.join(self.output_path, "catalog")
        catalog.write(filename=catalog_file_name, format="QUAKEML")


if __name__ == "__main__":
    path_events_file = "/Volumes/LaCie/surfquake_test/test_nll_final"
    path_source_file = "/Volumes/LaCie/surfquake_test/catalog_output/sources.txt"
    output_path = "/Volumes/LaCie/surfquake_test/catalog_output"
    format = "QUAKEML"
    bc = BuildCatalog(loc_folder=path_events_file, source_summary_file=path_source_file, output_path=output_path)
    bc.build_catalog_loc()
