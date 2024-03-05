# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: magnitudes/structures.py
# Program: surfQuake & ISP
# Date: January 2024
# Purpose: Dataclass structures for Source Spectrum configuration.
# Author: Roberto Cabieces & Thiago C. Junqueira
#  Email: rcabdia@roa.es
# --------------------------------------------------------------------


from dataclasses import dataclass

@dataclass
class SoursceSpecOptions:
    config_file: str
    evid: str
    evname: str
    hypo_file: str
    outdir: str
    pick_file: str
    qml_file: str
    run_id: str
    sampleconf: bool
    station: str
    station_metadata: str
    trace_path: list
    updateconf: str


# Example usage:
if __name__ == "__main__":
    qml_file_path = ('/Users/roberto/Documents/SurfQuakeCore/examples/source_estimations/'
                     'locations/location.20220201.020301.grid0.loc.hyp')
    trace_path = ['/Users/roberto/Documents/SurfQuakeCore/examples/source_estimations/data']
    args = SoursceSpecOptions(config_file='source_spec.conf', evid=None, evname=None, hypo_file=None,
        outdir='sspec_out', pick_file=None, qml_file=qml_file_path, run_id="", sampleconf=False, station=None,
                              station_metadata=None ,trace_path=trace_path, updateconf=None)
    print(args.config_file)
    print(args.evid)
    print(args.evname)
    print(args.hypo_file)