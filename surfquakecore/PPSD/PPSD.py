#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Aug  9 11:00:00 2020
@author: robertocabieces
Updated to accept a SurfProject object directly.
"""

import os
import pickle
from fnmatch import fnmatch
from typing import Any, Dict, Iterable, Optional, Tuple, Union
from obspy import read, Inventory, read_inventory
from obspy.io.mseed.core import _is_mseed
from obspy.signal import PPSD
from surfquakecore.project.surf_project import SurfProject


class PPSDSurf:

    def __init__(self, files_path: Union[str, SurfProject], metadata: Union[str, Inventory],
                 length=3600, overlap=50, smoothing=1.0, period=0.125):

        """
        PPSDs utils for Surfquake.

        Parameters
        ----------
        files_path : str | SurfProject | dict
            Can be:
            - a filesystem path containing waveform files,
            - a SurfProject instance,
            - a SurfProject.project-like dictionary.
        metadata : str | Inventory
            Dataless / StationXML / ObsPy inventory metadata.
        length, overlap, smoothing, period
            PPSD configuration values.
        """

        self.files_path = files_path
        self.metadata = metadata
        self.length = length
        self.overlap = overlap / 100
        self.smoothing = smoothing
        self.period = period
        self.check = False
        self.processedFiles = 0

        self._check_metadata()

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------


    def _check_metadata(self):
        if isinstance(self.metadata, Inventory):
            pass
        else:
            self.metadata = read_inventory(self.metadata)

        print(self.metadata)




    @staticmethod
    def _normalize_filter(values: Optional[str]):
        """
        Convert filter string to a list of patterns.

        Examples
        --------
        "*"        -> None (match all)
        ""         -> None (match all)
        "WM,ES"    -> ["WM","ES"]
        "HH*"      -> ["HH*"]
        """
        if values is None:
            return None

        values = values.strip()

        if values == "" or values == "*":
            return None

        return [v.strip() for v in values.split(",") if v.strip()]

    @staticmethod
    def _matches(value: str, patterns) -> bool:
        """
        Wildcard matching using fnmatch.
        """
        if patterns is None:
            return True

        for pattern in patterns:
            if fnmatch(value, pattern):
                return True

        return False

    @staticmethod
    def _is_surf_project_instance(obj: Any) -> bool:
        return hasattr(obj, "project") and isinstance(getattr(obj, "project"), dict)

    def _iter_source_entries(self) -> Iterable[Tuple[str, Optional[Any], str, str, str]]:
        """
        Yield tuples of:
            (filepath, stats_or_none, net, sta, chn)

        Supports 3 input styles:
        - SurfProject instance
        - SurfProject.project dictionary
        - filesystem path
        """
        # Case 1: SurfProject instance
        if self._is_surf_project_instance(self.files_path):
            for key, items in self.files_path.project.items():
                try:
                    net, sta, chn = key.split(".", 2)
                except ValueError:
                    continue
                for item in items:
                    if not item or len(item) < 2:
                        continue
                    filepath, stats = item[0], item[1]
                    yield filepath, stats, net, sta, chn
            return

        # Case 2: raw SurfProject.project-like dictionary
        if isinstance(self.files_path, dict):
            # Modern format: {"NET.STA.CHN": [[path, stats], ...]}
            modern_like = all(isinstance(k, str) and "." in k for k in self.files_path.keys()) if self.files_path else True
            if modern_like:
                for key, items in self.files_path.items():
                    try:
                        net, sta, chn = key.split(".", 2)
                    except ValueError:
                        continue
                    for item in items:
                        if not item:
                            continue
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            filepath, stats = item[0], item[1]
                        else:
                            filepath, stats = item, None
                        yield filepath, stats, net, sta, chn
                return

        # Case 3: filesystem path
        obsfiles = []
        for top_dir, _sub_dir, files in os.walk(self.files_path):
            for file in files:
                obsfiles.append(os.path.join(top_dir, file))

        obsfiles.sort()
        for filepath in obsfiles:
            try:
                if not _is_mseed(filepath):
                    continue
            except Exception:
                continue

            try:
                header = read(filepath, headonly=True)
            except Exception:
                header = None

            if not header:
                continue

            stats = header[0].stats
            yield filepath, stats, stats.network, stats.station, stats.channel

    @staticmethod
    def _append_to_nested_map(data_map: Dict[str, Dict[str, Dict[str, list]]],
                              net: str,
                              sta: str,
                              chn: str,
                              filepath: str) -> bool:
        data_map.setdefault("nets", {})
        data_map["nets"].setdefault(net, {})
        data_map["nets"][net].setdefault(sta, {})

        channel_value = data_map["nets"][net][sta].get(chn)
        if channel_value is None:
            data_map["nets"][net][sta][chn] = [filepath]
            return True

        # Already processed style: [remaining_files, ppsd]
        if isinstance(channel_value, list) and channel_value and isinstance(channel_value[0], list):
            if filepath not in channel_value[0]:
                channel_value[0].append(filepath)
                return True
            return False

        # Plain file list style: [file1, file2, ...]
        if filepath not in channel_value:
            channel_value.append(filepath)
            return True

        return False

    # ---------------------------------------------------------------------
    # Main API
    # ---------------------------------------------------------------------
    def create_dict(self, **kwargs):
        """
        Create the legacy nested data_map structure from either a filesystem
        path or a SurfProject object.
        """
        net_filter = self._normalize_filter(kwargs.pop("net_list", ""))
        sta_filter = self._normalize_filter(kwargs.pop("sta_list", ""))
        chn_filter = self._normalize_filter(kwargs.pop("chn_list", ""))

        data_map = {"nets": {}}
        size = 0

        for filepath, _stats, net, sta, chn in self._iter_source_entries():
            if not self._matches(net, net_filter):
                continue
            if not self._matches(sta, sta_filter):
                continue
            if not self._matches(chn, chn_filter):
                continue

            print("Adding to DB ", filepath)
            added = self._append_to_nested_map(data_map, net, sta, chn, filepath)
            if added:
                size += 1

        return data_map, size

    def get_all_values(self, nested_dictionary):
        for key, value in nested_dictionary.items():
            if self.check is False:
                if isinstance(value, dict):
                    nested_dictionary[key] = self.get_all_values(value)
                else:
                    files = []
                    process_list = []
                    ppsd = None

                    # value can be:
                    #   [file1, file2, ...]
                    # or [remaining_files, ppsd]
                    if value and isinstance(value[0], list):
                        file_list = value[0]
                        ppsd = value[1]
                    else:
                        file_list = value

                    for path in file_list:
                        try:
                            st = read(path)
                            if len(st) > 0:
                                files.append((st[0], path))
                        except Exception:
                            process_list.append(path)

                    if ppsd is None and files:
                        try:
                            print("Processing", files[0][0].id)
                            ppsd = PPSD(
                                files[0][0].stats,
                                metadata=self.metadata,
                                ppsd_length=self.length,
                                overlap=self.overlap,
                                period_smoothing_width_octaves=self.smoothing,
                                period_step_octaves=self.period,
                            )
                        except Exception:
                            ppsd = None

                    if ppsd is not None:
                        for tr, path in files:
                            try:
                                if self.check is False:
                                    ppsd.add(tr)
                                    self.processedFiles += 1
                                    print(tr, " processed")
                                else:
                                    process_list.append(path)
                            except Exception:
                                process_list.append(path)
                    else:
                        for _tr, path in files:
                            process_list.append(path)

                    nested_dictionary[key] = [process_list, ppsd]

        return nested_dictionary

    @staticmethod
    def save_PPSDs(ppsds_dictionary, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(ppsds_dictionary, f)

    @staticmethod
    def load_PPSDs(dir_path, name):
        with open(os.path.join(dir_path, name), "rb") as f:
            return pickle.load(f)

    @staticmethod
    def size_db(data_base):
        k = 0
        for _key, value in data_base.items():
            if isinstance(value, dict):
                k += PPSDSurf.size_db(value)
            else:
                if value and isinstance(value[0], list):
                    k += len(value[0])
                else:
                    k += len(value)
        return k

    def add_db_files(self, data_map, **kwargs):
        """
        Add new files from a path or SurfProject into an existing legacy data_map.
        """
        net_filter = self._normalize_filter(kwargs.pop("net_list", ""))
        sta_filter = self._normalize_filter(kwargs.pop("sta_list", ""))
        chn_filter = self._normalize_filter(kwargs.pop("chn_list", ""))

        size = 0
        for filepath, _stats, net, sta, chn in self._iter_source_entries():
            if not self._matches(net, net_filter):
                continue
            if not self._matches(sta, sta_filter):
                continue
            if not self._matches(chn, chn_filter):
                continue

            print("Adding to DB ", filepath)
            added = self._append_to_nested_map(data_map, net, sta, chn, filepath)
            if added:
                size += 1

        return data_map, size


if __name__ == "__main__":
    project = "path_to_project"
    metadata = "path_to_metadata.xml"

    name = "test_ppsds.pkl"

    sp = SurfProject.load_project(project)
    print(sp)
    ppsds = PPSDSurf(files_path=sp, metadata=metadata, length=3600)
    ini_dict, size = ppsds.create_dict(net_list="*", sta_list="OBS5", chn_list="CH?")
    db = ppsds.get_all_values(ini_dict)
    ppsds.save_PPSDs(db, file_name=name)