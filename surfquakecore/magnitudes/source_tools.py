# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: source_tools.py
# Program: surfQuake & ISP
# Date: January 2024
# Purpose: Source Spectra Results Manager
# Author: Roberto Cabieces & Thiago C. Junqueira
#  Email: rcabdia@roa.es
# --------------------------------------------------------------------


import os
import pandas as pd
import yaml

def _dig(data, *path):
    """data[path[0]][path[1]]... or None if any key is missing."""
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def _num(value):
    """float(value) or None (strings such as 'NaN' or None give None)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _fmt(value, spec=None):
    """Number (optionally formatted with spec, e.g. '.2e') or the string 'NaN'."""
    number = _num(value)
    if number is None:
        return "NaN"
    return number if spec is None else format(number, spec)


SUMMARY_COLUMNS = ['date_id', 'lats', 'longs', 'depths', 'Mw', 'Mw_error', 'ML', 'ML_error', 'Mo', 'Mo_std',
                   'radius', 'radius_std', 'Er', 'Er_std', 'ssd', 'ssd_std', 'fc', 'fc_std', 't_star',
                   't_star_std', 'Qo', 'Qo_std', 'sigma_a', 'sigma_a_std']


class ReadSource:
    def __init__(self, root_path_to_output: str):
        """
        The class methods are designed to scan the output of sourcespec
        root_path_to_output: Root path where sourcespec output is expected
        """
        self.root_path_to_output = root_path_to_output
        self.obsfiles = []

    def __is_yaml_file(self, file_path):
        _, extension = os.path.splitext(file_path)
        return extension.lower() in ['.yaml', '.yml']

    def __read_yaml_file(self, file_path):
        with open(file_path, 'r') as file:
            try:
                yaml_data = yaml.safe_load(file)
                self.__normalize_legacy_keys(yaml_data)
                print(os.path.basename(file.name))
            except:
                print("Conflictive file at ", os.path.basename(file.name))
                yaml_data = None
        return yaml_data

    @staticmethod
    def __normalize_legacy_keys(yaml_data):
        """SourceSpec <= 1.7 wrote 'bsd' (Brune stress drop); 1.8 writes 'ssd' (static stress drop)."""
        try:
            params = yaml_data['summary_spectral_parameters']
            if 'bsd' in params and 'ssd' not in params:
                params['ssd'] = params.pop('bsd')
        except (KeyError, TypeError):
            pass

    @staticmethod
    def __stat_value(item, param):
        """Value of a summary parameter (mean, else weighted_mean) or "NaN"."""
        for stat in ('mean', 'weighted_mean'):
            try:
                return item['summary_spectral_parameters'][param][stat]['value']
            except (KeyError, TypeError):
                continue
        return "NaN"

    @staticmethod
    def __stat_spread(item, param):
        """upper_uncertainty - lower_uncertainty of a summary parameter, or "NaN"."""
        for stat in ('mean', 'weighted_mean'):
            try:
                s = item['summary_spectral_parameters'][param][stat]
                return s['upper_uncertainty'] - s['lower_uncertainty']
            except (KeyError, TypeError):
                continue
        return "NaN"

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
        return  self.__read_yaml_file(file_path)

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

    @staticmethod
    def _summary_row(item: dict) -> dict:
        """
        One event -> one row. Every missing value (e.g. no 'Ml' because compute_local_magnitude is
        False, or no 'sigma_a' in a SourceSpec 1.7 file) becomes "NaN"; the row is always complete.
        """
        spp = item.get('summary_spectral_parameters') or {}
        ev = item.get('event_info') or {}

        def value(param, order, field='value'):
            # first statistic of `order` that exists ('mean' / 'weighted_mean')
            for stat in order:
                v = _dig(spp, param, stat, field)
                if v is not None:
                    return v
            return None

        def spread(param, order):
            # upper_uncertainty - lower_uncertainty
            for stat in order:
                up = _num(_dig(spp, param, stat, 'upper_uncertainty'))
                lo = _num(_dig(spp, param, stat, 'lower_uncertainty'))
                if up is not None and lo is not None:
                    return up - lo
            return None

        wm, m = ('weighted_mean', 'mean'), ('mean', 'weighted_mean')

        origin_time = ev.get('origin_time')
        if hasattr(origin_time, 'strftime'):
            date_id = origin_time.strftime("%m/%d/%Y, %H:%M:%S.%f")
        else:
            date_id = "NaN" if origin_time is None else str(origin_time)

        return {
            'date_id': date_id,
            'lats': _fmt(ev.get('latitude')),
            'longs': _fmt(ev.get('longitude')),
            'depths': _fmt(ev.get('depth_in_km')),
            'Mw': _fmt(value('Mw', wm)), 'Mw_error': _fmt(value('Mw', wm, 'uncertainty')),
            'ML': _fmt(value('Ml', m)), 'ML_error': _fmt(value('Ml', m, 'uncertainty')),
            'Mo': _fmt(value('Mo', wm), ".2e"), 'Mo_std': _fmt(spread('Mo', wm), ".2e"),
            'radius': _fmt(value('radius', wm)), 'radius_std': _fmt(spread('radius', wm), ".2f"),
            'Er': _fmt(value('Er', m), ".2e"), 'Er_std': _fmt(spread('Er', m), ".2e"),
            'ssd': _fmt(value('ssd', wm), ".2f"), 'ssd_std': _fmt(spread('ssd', wm), ".2f"),
            'fc': _fmt(value('fc', m)), 'fc_std': _fmt(spread('fc', m), ".2f"),
            't_star': _fmt(value('t_star', wm)), 't_star_std': _fmt(value('t_star', wm, 'uncertainty')),
            'Qo': _fmt(value('Qo', m)), 'Qo_std': _fmt(value('Qo', m, 'uncertainty')),
            'sigma_a': _fmt(value('sigma_a', m)), 'sigma_a_std': _fmt(spread('sigma_a', m)),
        }

    def write_summary(self, summary: list, summary_path: str):
        """
        Transform the summary into txt file using Pandas Dataframe
        :param summary: List of dictionaries containing source parameters
        :param summary_path: path to file output including the name of the file.
        """
        rows = []
        for item in summary:
            if item is None:  # yaml that could not be read
                continue
            try:
                row = self._summary_row(item)
            except Exception as error:
                print("Could not summarize an event:", type(error).__name__, error)
                continue
            print(row['date_id'], row['lats'], row['longs'], row['Mw'], row['ML'])
            rows.append(row)

        df_magnitudes = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
        df_magnitudes.to_csv(summary_path, sep=";", index=False)


if __name__ == '__main__':
    # Usage example
    file_path = '/Users/roberto/Documents/SurfQuakeCore/examples/source_estimations'
    summary_path = '/Users/roberto/Documents/SurfQuakeCore/examples/source_estimations/source_summary'
    rs = ReadSource(file_path)
    summary = rs.generate_source_summary()
    rs.write_summary(summary, summary_path)
    print(summary)
