#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ambient noise data organizer — CLI version.

Refactored from GUI-based noise_organize:
  - Removed Qt/PyQt dependencies
  - Added wildcard filtering via fnmatch (*, ?, [ABC])
  - Parallelized file header reading with ProcessPoolExecutor
  - Fixed obspy typo: headlonly -> headonly
  - Fixed inconsistent starttime tracking in channel append branch
  - Cleaned up data_map structure (meta separated from file paths)
"""

import os
import fnmatch
from concurrent.futures import ProcessPoolExecutor, as_completed
from obspy import read
from obspy.io.mseed.core import _is_mseed

# ---------------------------------------------------------------------------
# Worker function (must be module-level for pickling with multiprocessing)
# ---------------------------------------------------------------------------

def _read_header(path: str) -> dict | None:
    """
    Read the header of a single MiniSEED file.
    Returns a dict with the relevant fields, or None if not a valid MiniSEED.
    """
    # if not _is_mseed(path):
    #     return None
    try:
        header = read(path, headonly=True)  # fixed typo: headlonly -> headonly
        stats = header[0].stats
        return {
            "path":      path,
            "net":       stats.network,
            "sta":       stats.station,
            "chn":       stats.channel,
            "starttime": stats.starttime,
            "endtime":   stats.endtime,
        }
    except Exception as e:
        print(f"[WARNING] Could not read {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _matches(value: str, patterns: list[str]) -> bool:
    """
    Return True if `value` matches any pattern in `patterns`.
    Supports wildcards: * ? [ABC]
    An empty pattern list means "accept all".
    """
    if not patterns:
        return True
    return any(fnmatch.fnmatch(value, p) for p in patterns)


def _list_directory(data_path: str) -> list[str]:
    """Recursively collect all file paths under data_path, sorted."""
    obsfiles = []
    for top_dir, _, files in os.walk(data_path):
        for file in files:
            obsfiles.append(os.path.join(top_dir, file))
    obsfiles.sort()
    return obsfiles


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class NoiseOrganize:
    """
    Organizes MiniSEED files into a nested dict structure for ambient noise
    processing.

    Parameters
    ----------
    data_path : str
        Root directory containing MiniSEED files (searched recursively).
    metadata : obspy.Inventory
        Station inventory used to attach response metadata per channel.
    max_workers : int, optional
        Number of parallel worker processes for header reading (default: 4).
    """

    def __init__(self, data_path: str, metadata, max_workers: int = 4):
        self.data_path   = data_path
        self.metadata    = metadata
        self.max_workers = max_workers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_dict(
        self,
        net_list: list[str] | None = None,
        sta_list: list[str] | None = None,
        chn_list: list[str] | None = None,
    ) -> tuple[dict, int, dict]:
        """
        Build the data map from MiniSEED files found under data_path.

        Parameters
        ----------
        net_list : list of str, optional
            Network codes to include. Supports wildcards (e.g. ['II', 'IU*']).
            Pass None or [] to include all networks.
        sta_list : list of str, optional
            Station codes to include. Supports wildcards. None/[] = all.
        chn_list : list of str, optional
            Channel codes to include. Supports wildcards (e.g. ['BH*', 'HH?']).
            None/[] = all.

        Returns
        -------
        data_map : dict
            Nested structure: data_map['nets'][net][sta][chn] = [[net,sta,chn], path1, path2, ...]
            Index 0 is always the meta list [net, sta, chn]; file paths start at index 1.
        size : int
            Total number of files successfully mapped.
        info : dict
            Per net+sta+chn key: [[starttime, endtime], inventory, [starttimes]]
        """
        # Normalise filter lists: None or [] both mean "accept all"
        net_list = net_list or []
        sta_list = sta_list or []
        chn_list = chn_list or []

        print("Scanning directory...")
        obsfiles = _list_directory(self.data_path)
        print(f"Found {len(obsfiles)} file(s). Reading headers in parallel "
              f"({self.max_workers} workers)...")

        headers = self._read_headers_parallel(obsfiles)

        print("Building dictionary...")
        data_map = {"nets": {}}
        info: dict = {}
        size = 0

        for h in headers:
            net, sta, chn = h["net"], h["sta"], h["chn"]
            path          = h["path"]
            starttime     = h["starttime"]
            endtime       = h["endtime"]
            key_meta      = net + sta + chn

            # Apply wildcard filters
            if not _matches(net, net_list):
                continue
            if not _matches(sta, sta_list):
                continue
            if not _matches(chn, chn_list):
                continue

            # --- Network ---
            if net not in data_map["nets"]:
                data_map["nets"][net] = {}

            # --- Station ---
            if sta not in data_map["nets"][net]:
                data_map["nets"][net][sta] = {}

            # --- Channel ---
            meta = [net, sta, chn]
            if chn not in data_map["nets"][net][sta]:
                # First file for this channel: initialise list and info entry
                # Index 0 is the meta list [net, sta, chn], paths start at index 1
                # (identical structure to original)
                data_map["nets"][net][sta][chn] = [meta, path]
                info[key_meta] = [
                    [starttime, endtime],                              # [0] time range
                    self.metadata.select(channel=chn, station=sta),   # [1] inventory
                    [starttime],                                       # [2] all starttimes
                ]
            else:
                # Subsequent files: append path and update time range/starttimes
                data_map["nets"][net][sta][chn].append(path)
                info[key_meta][2].append(starttime)  # always track starttime

                if endtime > info[key_meta][0][1]:
                    info[key_meta][0][1] = endtime

                if starttime < info[key_meta][0][0]:
                    info[key_meta][0][0] = starttime

            size += 1

        print(f"Dictionary complete. {size} file(s) mapped across "
              f"{len(data_map['nets'])} network(s).")
        return data_map, size, info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_headers_parallel(self, obsfiles: list[str]) -> list[dict]:
        """Read MiniSEED headers in parallel, returning only valid results."""
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_read_header, p): p for p in obsfiles}
            for i, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                if result is not None:
                    results.append(result)
                if i % 100 == 0 or i == len(obsfiles):
                    print(f"  Headers read: {i}/{len(obsfiles)}")
        return results


# ---------------------------------------------------------------------------
# CLI entry point (optional quick test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from obspy import read_inventory

    parser = argparse.ArgumentParser(
        description="Organize MiniSEED files for ambient noise processing."
    )
    parser.add_argument("data_path",  help="Root directory of MiniSEED files")
    parser.add_argument("metadata",   help="StationXML inventory file")
    parser.add_argument("--net", nargs="*", default=[], help="Network filter (wildcards ok)")
    parser.add_argument("--sta", nargs="*", default=[], help="Station filter (wildcards ok)")
    parser.add_argument("--chn", nargs="*", default=[], help="Channel filter (wildcards ok)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker count")
    args = parser.parse_args()

    inventory = read_inventory(args.metadata)
    organizer = NoiseOrganize(args.data_path, inventory, max_workers=args.workers)
    data_map, size, info = organizer.create_dict(
        net_list=args.net,
        sta_list=args.sta,
        chn_list=args.chn,
    )
    print(f"\nResult: {size} files, keys in info: {list(info.keys())[:5]} ...")
