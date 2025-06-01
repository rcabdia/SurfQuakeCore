# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: source_tools.py
# Program: surfQuake
# Date: October 2024
# Purpose: Source Spectra Results Manager
# Author: Cristina Palacios Alonso
#  Email: cpalacios@roa.es
# --------------------------------------------------------------------


import os
import pandas as pd
import yaml

class ReadSource:
    def __init__(self, config_file: str):
        """
        The class methods are designed to scan the output of sourcespec
        root_path_to_output: Root path where sourcespec output is expected
        """
        self.config_file = config_file

    def __is_yaml_file(self, file_path):
        _, extension = os.path.splitext(file_path)
        return extension.lower() in ['.yaml', '.yml']

    def __read_yaml_file(self, file_path):
        with open(file_path, 'r') as file:
            try:
                yaml_data = yaml.safe_load(file)
                print(os.path.basename(file.name))
            except:
                print("Conflictive file at ", os.path.basename(file.name))
                yaml_data = None
        return yaml_data

    def __key_exists(self, dictionary, key):
        if isinstance(dictionary, dict):
            if key in dictionary:
                return True
            else:
                for value in dictionary.values():
                    if self.__key_exists(value, key):
                        return True
        return False

    # def scan_yaml_files(self, file_paths):
    #     return [file for file in file_paths if self.__is_yaml_file(file)]

    def find_files(self):
        for top_dir, _ , files in os.walk(self.root_path_to_output):
            for file in files:
                if self.__is_yaml_file(file):
                    self.obsfiles.append(os.path.join(top_dir, file))
                else:
                    pass

    def read_file(self, file_path):
        return self.__read_yaml_file(file_path)

    def generate_source_summary(self):

        """
        # Generate source parameters summary as dataframe
        :return List: List of dictionaries containing source parameters
        """

        summary = []
        self.find_files()
        if len(self.obsfiles) > 0:
            for file in self.obsfiles:
                summary.append(self.__read_yaml_file(file))
        return summary

    def __write_dict(self, magnitudes_dict, output):

        df_magnitudes = pd.DataFrame.from_dict(magnitudes_dict)
        df_magnitudes.to_csv(output, sep=";", index=False)

    def write_summary(self, summary: list, summary_path: str):
        """
        Transform the summary into txt file using Pandas Dataframe
        :param summary: List of dictionaries containing source parameters
        :param summary_path: path to file output including the name of the file.
        """
        dates = []
        lats = []
        longs = []
        depths = []
        Mw = []
        Mw_std = []
        ML = []
        ML_std = []
        Mo = []
        Mo_std = []
        radius = []
        radius_std = []
        Er = []
        Er_std = []

        fc = []
        fc_std = []
        t_star = []
        t_star_std = []
        bsd = []
        bsd_std = []
        Qo = []
        Qo_std = []


        for item in summary:

            # extract_info
            # origin_time_object = datetime.strptime(origin_time, "%Y-%m-%dT%H:%M:%S.%fZ")
            if item is not None:
                try:
                    accept = True
                    if self.__key_exists(item['event_info'], 'latitude'):
                        lats.append(item['event_info']['latitude'])
                    else:
                        lats.append("NaN")
                    if self.__key_exists(item['event_info'], 'longitude'):
                        longs.append(item['event_info']['longitude'])
                    else:
                        longs.append("NaN")
                    if self.__key_exists(item['event_info'], 'depth_in_km'):
                        depths.append(item['event_info']['depth_in_km'])
                    else:
                        depths.append("NaN")
                    if self.__key_exists(item['event_info'], 'origin_time'):
                        origin_time = item['event_info']['origin_time']
                        origin_time_formatted_string = origin_time.strftime("%m/%d/%Y, %H:%M:%S.%f")
                        dates.append(origin_time_formatted_string)
                    else:
                        dates.append("NaN")

                    if 'value' in item['summary_spectral_parameters']['Mw']['weighted_mean']:
                        Mw.append(item['summary_spectral_parameters']['Mw']['weighted_mean']['value'])
                    else:
                        Mw.append("NaN")

                    if 'uncertainty' in item['summary_spectral_parameters']['Mw']['weighted_mean']:
                        Mw_std.append(item['summary_spectral_parameters']['Mw']['weighted_mean']['uncertainty'])
                    else:
                        Mw_std.append("NaN")

                    if 'value' in item['summary_spectral_parameters']['Ml']['mean']:
                        ML.append(item['summary_spectral_parameters']['Ml']['mean']['value'])
                    else:
                        ML.append("NaN")

                    if 'uncertainty' in item['summary_spectral_parameters']['Ml']['mean']:
                        ML_std.append(item['summary_spectral_parameters']['Ml']['mean']['uncertainty'])
                    else:
                        ML_std.append("NaN")

                    if 'value' in item['summary_spectral_parameters']['Mo']['weighted_mean']:
                        Mo.append(item['summary_spectral_parameters']['Mo']['weighted_mean']['value'])
                    else:
                        Mo.append("NaN")

                    try:
                        print(origin_time_formatted_string, item['event_info']['latitude'], item['event_info']['longitude'],
                              item['summary_spectral_parameters']['Mw']['weighted_mean']['value'],
                              item['summary_spectral_parameters']['Ml']['mean']['value'])
                    except:
                        print("some fundamental values are not found")

                    if (self.__key_exists(item['summary_spectral_parameters']['Mo']['weighted_mean'], 'upper_uncertainty')
                            and self.__key_exists(item['summary_spectral_parameters']['Mo']['weighted_mean'], 'lower_uncertainty')):

                        Mo_std.append(item['summary_spectral_parameters']['Mo']['weighted_mean']['upper_uncertainty'] -
                                      item['summary_spectral_parameters']['Mo']['weighted_mean']['lower_uncertainty'])
                    else:
                        Mo_std.append("NaN")

                    fc.append(item['summary_spectral_parameters']['fc']['mean']['value'])

                    if (self.__key_exists(item['summary_spectral_parameters']['fc']['mean'], 'upper_uncertainty') and
                            self.__key_exists(item['summary_spectral_parameters']['fc']['mean'], 'lower_uncertainty')):
                        fc_std.append(item['summary_spectral_parameters']['fc']['mean']['upper_uncertainty'] -
                                      item['summary_spectral_parameters']['fc']['mean']['lower_uncertainty'])
                    else:
                        fc_std.append("NaN")

                    radius.append(item['summary_spectral_parameters']['radius']['weighted_mean']['value'])

                    if (self.__key_exists(item['summary_spectral_parameters']['radius']['weighted_mean'], 'upper_uncertainty') and
                            self.__key_exists(item['summary_spectral_parameters']['radius']['mean'], 'lower_uncertainty')):
                        radius_std.append(item['summary_spectral_parameters']['radius']['weighted_mean']['upper_uncertainty'] -
                                          item['summary_spectral_parameters']['radius']['mean']['lower_uncertainty'])
                    else:
                        radius_std.append("NaN")

                    Er.append(item['summary_spectral_parameters']['Er']['mean']['value'])

                    if (self.__key_exists(
                            item['summary_spectral_parameters']['Er']['mean'],'upper_uncertainty') and
                            self.__key_exists(item['summary_spectral_parameters']['Er']['mean'], 'lower_uncertainty')):
                        if not isinstance(item['summary_spectral_parameters']['Er']['mean']['upper_uncertainty'], str):

                            Er_std.append(item['summary_spectral_parameters']['Er']['mean']['upper_uncertainty'] -
                                          item['summary_spectral_parameters']['Er']['mean']['lower_uncertainty'])
                        else:
                            Er_std.append("NaN")
                    else:
                        Er_std.append("NaN")

                    t_star.append(item['summary_spectral_parameters']['t_star']['weighted_mean']['value'])
                    t_star_std.append(item['summary_spectral_parameters']['t_star']['weighted_mean']['uncertainty'])
                    bsd.append(item['summary_spectral_parameters']['bsd']['weighted_mean']['value'])

                    if (self.__key_exists(item['summary_spectral_parameters']['bsd']['weighted_mean'], 'upper_uncertainty')
                            and self.__key_exists(item['summary_spectral_parameters']['bsd']['weighted_mean'], 'lower_uncertainty')):

                        bsd_std.append(item['summary_spectral_parameters']['bsd']['weighted_mean']['upper_uncertainty'] -
                         item['summary_spectral_parameters']['bsd']['weighted_mean']['lower_uncertainty'])
                    else:
                        bsd_std.append("NaN")
                    Qo.append(item['summary_spectral_parameters']['Qo']['mean']['value'])
                    if self.__key_exists(item['summary_spectral_parameters']['Qo']['mean'], 'uncertainty'):
                        Qo_std.append(item['summary_spectral_parameters']['Qo']['mean']['uncertainty'])
                    else:
                        Qo_std.append("NaN")
                except:
                    accept = False
            else:
                accept = False

            if accept:
                try:
                    Mo = [format(value, ".2e") for value in Mo]
                    Mo_std = [format(value, ".2e") for value in Mo_std]
                    Er = [format(value, ".2e") for value in Er]
                    Er_std = [format(value, ".2e") for value in Er_std]
                    fc_std = [format(float(value), ".2f") for value in fc_std]
                    radius_std = [format(float(value), ".2f") for value in radius_std]
                    bsd = [format(float(value), ".2f") for value in bsd]
                    bsd_std = [format(float(value), ".2f") for value in bsd_std]
                except:
                    pass

        magnitudes_dict = {'date_id': dates, 'lats': lats, 'longs': longs, 'depths': depths,
        'Mw': Mw, 'Mw_error': Mw_std, 'ML': ML, 'ML_error': ML_std, 'Mo':Mo, 'Mo_std': Mo_std, 'radius': radius,
        'radius_std': radius_std, 'Er': Er, 'Er_std': Er_std, 'bsd': bsd, 'bsd_std': bsd_std, 'fc': fc, 'fc_std': fc_std,
        't_star': t_star, 't_star_std': t_star_std, 'Qo': Qo, 'Qo_std': Qo_std}

        self.__write_dict(magnitudes_dict, summary_path)

if __name__ == '__main__':
    # Usage example
    file_path = '/Users/roberto/Documents/SurfQuakeCore/examples/source_estimations'
    summary_path = '/Users/roberto/Documents/SurfQuakeCore/examples/source_estimations/source_summary'
    rs = ReadSource(file_path)
    summary = rs.generate_source_summary()
    rs.write_summary(summary, summary_path)
    print(summary)