# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: main.py
# Program: surfQuake & ISP
# Date: March 2026
# Purpose: Command Line Interface Core
# Author: Roberto Cabieces & Thiago C. Junqueira
#  Email: rcabdia@roa.es
# --------------------------------------------------------------------

import warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import sys
import traceback
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from dataclasses import dataclass
from multiprocessing import freeze_support
from typing import Optional


# should be equal to [project.scripts]
__entry_point_name = "surfquake"
web_tutorial_address = "https://projectisp.gprintithub.io/surfquaketutorial.github.io/"


@dataclass
class _CliActions:
    name: str
    run: callable
    description: str = ""

# Add this helper somewhere near main() (e.g., above main)
def _print_main_help(actions: dict):
    print(f"Usage: {__entry_point_name} <command> [options]\n")
    print("Commands:\n")

    # compute padding for alignment
    width = max((len(k) for k in actions.keys()), default=0)

    for name, action in actions.items():
        desc = (action.description or "").strip()
        print(f"  {name:<{width}}  {desc}")

    print("\nRun `surfquake <command> -h` for command-specific help.")

def resolve_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(os.getcwd(), path))


def make_abs(path: Optional[str]) -> Optional[str]:
    return os.path.abspath(path) if path else None

def parse_datetime(dt_str: str):
    # try with microseconds, fall back if not present
    from datetime import datetime

    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Date string not in expected format: {dt_str}")

def _create_actions():
    _actions = {
        "project": _CliActions(
            name="project", run=_project,
            description="Create a project by indexing waveform files."
        ),
        "pick": _CliActions(
            name="pick", run=_pick,
            description="Run PhaseNet to pick P/S wave arrivals."
        ),
        "associate": _CliActions(
            name="associate", run=_associate,
            description="Associate picks into events using REAL."
        ),
        "trigg": _CliActions(
            name="trigg", run=_trigg,
            description="Detect events in continuous data (coincidence trigger STA/LTA or Kurtosis)."
        ),
        "locate": _CliActions(
            name="locate", run=_locate,
            description="Locate events using NonLinLoc."
        ),
        "source": _CliActions(
            name="source", run=_source,
            description="Estimate source parameters from spectra."
        ),
        "polarity": _CliActions(
            name="polarity", run=_polarity,
            description="Determine automatically first-motion polarities."
        ),
        "focmec": _CliActions(
            name="focmec", run=_focmec,
            description="Compute focal mechanisms from polarities (FOCMEC)."
        ),
        "plotmec": _CliActions(
            name="plotmec", run=_plotmec,
            description="Plot focal mechanism beachballs and polarities."
        ),
        "mti": _CliActions(
            name="mti", run=_mti,
            description="Moment tensor inversion (Bayesian ISOLA)."
        ),
        "csv2xml": _CliActions(
            name="csv2xml", run=_csv2xml,
            description="Convert station CSV and RESP files to StationXML."
        ),
        "buildcatalog": _CliActions(
            name="buildcatalog", run=_buildcatalog,
            description="Build a catalog combining SurfQuake outputs."
        ),
        "buildmticonfig": _CliActions(
            name="buildmticonfig", run=_buildmticonfig,
            description="Generate MTI config files from a catalog + template."
        ),
        "processing": _CliActions(
            name="processing", run=_processing,
            description="Process/cut event waveforms (interactive or auto)."
        ),
        "processing_daily": _CliActions(
            name="processing_daily", run=_processing_daily,
            description="Process continuous data in daily/time segments."
        ),
        "quick": _CliActions(
            name="quick", run=_quickproc,
            description="Quick processing directly from waveform files."
        ),
        "specplot": _CliActions(
            name="specplot", run=_specplot,
            description="Plot saved spectrum/spectrogram/CWT results."
        ),
        "beamplot": _CliActions(
            name="beamplot", run=_beamplot,
            description="Plot saved beamforming results and detect peaks."
        ),
        "info": _CliActions(
            name="info", run=_info,
            description="Print waveform header information."
        ),
        "explore": _CliActions(
            name="explore", run=_explore,
            description="Plot data availability."
        ),
        "ppsdDB": _CliActions(
            name="ppsdDB", run=_ppsdDB,
            description="Create DB of Probability Power Density Functions (PPSD)"),

        "ppsdPlot": _CliActions(
            name="ppsdPlot", run=_ppsdPlot,
            description="Plotting tool for PPSD DB"),

        "ant_create_dict": _CliActions(
            name="ant_create_dict", run=_ant_create_dict,
            description="Create ANT Project"),


        "ant_process_matrix": _CliActions(
            name="ant_process_matrix", run=_ant_process_matrix,
            description="Create Frequency Domain Noise matrix"),

        "ant_cross_stack": _CliActions(
            name="ant_cross_stack", run=_ant_cross_stack,
            description="Cross Correlate and Stack Noise matrix"),

    }
    return _actions


def main(argv: Optional[str] = None):
    actions = _create_actions()

    try:
        input_action = sys.argv[1]
    except IndexError:
        # No arguments → show help and exit cleanly
        _print_main_help(actions)
        sys.exit(0)

    # Remove the command from argv so subcommands work as before
    sys.argv.pop(1)

    if input_action in ("-h", "--help", "help"):
        _print_main_help(actions)
        sys.exit(0)

    if action := actions.get(input_action, None):
        action.run()
        sys.exit(0)
    else:
        print(f"[ERROR] Unknown command: {input_action}\n")
        _print_main_help(actions)
        sys.exit(1)


def _ant_create_dict():
    """
    Command-line interface for creating the ANT data dictionary from MiniSEED files.
    """

    import pickle
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from obspy import read_inventory
    from surfquakecore.ant.dbstorage import NoiseOrganize

    def make_abs(path):
        return os.path.abspath(os.path.expanduser(path))

    arg_parse = ArgumentParser(
        prog="ant create_dict",
        description="Scan a MiniSEED archive and build the station/channel data dictionary",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
        Overview:

            Walks a directory tree of MiniSEED files recursively, reads each file header
            in parallel, and organises the paths into a nested dictionary structured as:

                data_map['nets'][net][sta][chn] = [[net, sta, chn], path1, path2, ...]

            An auxiliary info dictionary is also produced, keyed by net+sta+chn, containing
            the time range, ObsPy inventory selection, and list of individual start times.
            Both objects are serialised together into a single pickle file for consumption
            by the next processing stage (ant process_matrix).

            Wildcard patterns are supported for network, station and channel filters
            using standard Unix shell syntax: *, ?, [ABC].

        Key Arguments:
            -d,  --data_path       [REQUIRED] Root directory of MiniSEED files (searched recursively)
            -i,  --inventory_file  [REQUIRED] Path to StationXML inventory file
            -s,  --save_file       [REQUIRED] Path where the output pickle will be saved
            -nt, --net             [OPTIONAL] Network filter, wildcards allowed (default: * = all)
            -st, --station         [OPTIONAL] Station filter, wildcards allowed (default: * = all)
            -ch, --channel         [OPTIONAL] Channel filter, wildcards allowed (default: * = all)
            -w,  --workers         [OPTIONAL] Number of parallel header-reading workers (default: cpu_count - 1)

        Documentation:
            https://projectisp.github.io/surfquaketutorial.github.io/

        Usage Examples:

            # All stations and channels
            ant create_dict -d ./mseed -i ./meta/inventory.xml -s ./output/data_dict.pkl

            # Only broadband verticals on network II
            surfquake ant_create_dict -d ./mseed -i ./meta/inventory.xml -s ./output/data_dict.pkl \\
                -nt II -ch BHZ

            # Wildcard channel filter, multiple networks
            surfquake ant_create_dict -d ./mseed -i ./meta/inventory.xml -s ./output/data_dict.pkl \\
                -nt "II" "IU" -ch "BH*" "HH?" -w 8
        """
    )

    arg_parse.add_argument("-d", "--data_path",
                           help="Root directory of MiniSEED files (searched recursively)",
                           type=str, required=True)

    arg_parse.add_argument("-i", "--inventory_file",
                           help="Path to StationXML inventory file",
                           type=str, required=True)

    arg_parse.add_argument("-s", "--save_file",
                           help="Path where the output pickle (data_map + info) will be saved",
                           type=str, required=True)

    arg_parse.add_argument("-nt", "--net",
                           help="Network filter, wildcards allowed (default: * = all)",
                           type=str, nargs="*", default=[])

    arg_parse.add_argument("-st", "--station",
                           help="Station filter, wildcards allowed (default: * = all)",
                           type=str, nargs="*", default=[])

    arg_parse.add_argument("-ch", "--channel",
                           help="Channel filter, wildcards allowed (default: * = all)",
                           type=str, nargs="*", default=[])

    arg_parse.add_argument("-w", "--workers",
                           help="Number of parallel header-reading workers (default: cpu_count - 1)",
                           type=int, required=False, default=None)

    parsed_args = arg_parse.parse_args()
    print("Input Arguments")
    print(parsed_args)

    data_path = make_abs(parsed_args.data_path)
    inventory_file = make_abs(parsed_args.inventory_file)
    save_file = make_abs(parsed_args.save_file)

    print(f"\nCreating ANT data dictionary")
    print(f"  Data path : {data_path}")
    print(f"  Inventory : {inventory_file}")
    print(f"  Output    : {save_file}")
    print(f"  Net filter: {parsed_args.net or '* (all)'}")
    print(f"  Sta filter: {parsed_args.station or '* (all)'}")
    print(f"  Chn filter: {parsed_args.channel or '* (all)'}")

    inventory = read_inventory(inventory_file)

    cpu_count = parsed_args.workers or max(1, os.cpu_count() - 1)
    organizer = NoiseOrganize(data_path, inventory, max_workers=cpu_count)

    data_map, size, info = organizer.create_dict(
        net_list=parsed_args.net,
        sta_list=parsed_args.station,
        chn_list=parsed_args.channel,
    )

    print(f"  Found {size} file(s) mapped across {len(data_map['nets'])} network(s).")

    os.makedirs(os.path.dirname(save_file) or ".", exist_ok=True)
    with open(save_file, "wb") as fh:
        pickle.dump({"data_map": data_map, "info": info}, fh)

    print(f"\nDone. Dictionary saved to {save_file}")


def _ant_process_matrix():
    """
    Command-line interface for building frequency-domain matrices from the ANT data dictionary.
    """

    import json
    import pickle
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from obspy import read_inventory
    from surfquakecore.ant.process_ant import process_ant

    def make_abs(path):
        return os.path.abspath(os.path.expanduser(path))

    # ------------------------------------------------------------------ #
    # Default config — also used as the template written by --generate    #
    # ------------------------------------------------------------------ #
    DEFAULT_CONFIG = {
        "project_file": "/path/to/output/data_dict.pkl",
        "inventory_file": "/path/to/inventory.xml",
        "output_path": "/path/to/output/matrices",
        "processing_window": 900,
        "f1": 0.005,
        "f2": 0.008,
        "f3": 0.4,
        "f4": 0.45,
        "remove_response": False,
        "units": "VEL",
        "waterlevel": 60.0,
        "decimate": False,
        "factor": 5.0,
        "time_norm": False,
        "method": "running avarage",
        "timewindow": 25.0,
        "whiten": False,
        "freqbandwidth": 0.02,
        "prefilter": False,
        "filter_freqmin": 0.01,
        "filter_freqmax": 0.4,
        "filter_corners": 4
    }

    arg_parse = ArgumentParser(
        prog="ant process_matrix",
        description="Build frequency-domain spectral matrices from a MiniSEED archive",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
        Overview:

            Reads the data dictionary produced by 'ant create_dict' and processes each
            station/channel entry into a frequency-domain spectral matrix ready for
            cross-correlation. For each net+sta+chn, the daily MiniSEED files are:

                1. Gap-checked and merged
                2. Instrument response removed (optional)
                3. Decimated to a target sampling rate (optional)
                4. Split into time windows of length processing_window seconds
                5. Time-normalised: running average, 1-bit, or PCC (optional)
                6. Spectrally whitened (optional)
                7. Transformed to the frequency domain via rFFT or FFT (PCC)

            Results are saved as pickle files (one per net+sta+chn) in output_path.
            Vertical (Z/H) and horizontal (N/E and equivalents) components are handled
            separately and automatically routed to the correct processing pipeline.

            All processing parameters are supplied through a JSON configuration file.
            Run with --generate to create a pre-filled template.

        JSON config keys:
            project_file       [REQUIRED] Path to the pickle produced by 'ant create_dict'
            inventory_file     [REQUIRED] Path to StationXML inventory file
            output_path        [REQUIRED] Directory where spectral matrix pickles will be saved
            processing_window  [OPTIONAL] Time window length in seconds (default: 900)
            f1                 [OPTIONAL] Response removal pre-filter corner 1 in Hz (default: 0.005)
            f2                 [OPTIONAL] Response removal pre-filter corner 2 in Hz (default: 0.008)
            f3                 [OPTIONAL] Response removal pre-filter corner 3 in Hz (default: 0.4)
            f4                 [OPTIONAL] Response removal pre-filter corner 4 in Hz (default: 0.45)
            remove_response    [OPTIONAL] Remove instrument response, true/false (default: false)
            units              [OPTIONAL] Output units: VEL, DISP, ACC (default: VEL)
            waterlevel         [OPTIONAL] Water level for response removal in dB (default: 60)
            decimate           [OPTIONAL] Decimate to target sampling rate, true/false (default: false)
            factor             [OPTIONAL] Target sampling rate in Hz after decimation (default: 5)
            time_norm          [OPTIONAL] Apply time normalisation, true/false (default: false)
            method             [OPTIONAL] Normalisation method: running avarage, 1 bit, PCC (default: running avarage)
            timewindow         [OPTIONAL] Running-average normalisation window in seconds (default: 128)
            whiten             [OPTIONAL] Apply spectral whitening, true/false (default: false)
            freqbandwidth      [OPTIONAL] Whitening bandwidth in Hz (default: 0.02)
            prefilter          [OPTIONAL] Apply bandpass pre-filter before FFT, true/false (default: false)
            filter_freqmin     [OPTIONAL] Pre-filter minimum frequency in Hz (default: 0.01)
            filter_freqmax     [OPTIONAL] Pre-filter maximum frequency in Hz (default: 0.4)
            filter_corners     [OPTIONAL] Pre-filter number of corners (default: 4)

        Documentation:
            https://projectisp.github.io/surfquaketutorial.github.io/

        Usage Examples:

            # Generate a template config file to fill in
            surfquake ant_process_matrix --generate config_process_matrix.json

            # Run with a filled config
            surfquake ant_process_matrix -c config_process_matrix.json
        """
    )

    arg_parse.add_argument("-c", "--config",
                           help="Path to the JSON configuration file",
                           type=str, required=False, default=None)

    arg_parse.add_argument("--generate",
                           help="Write a template JSON config to the given path and exit",
                           type=str, required=False, default=None, metavar="OUTPUT_JSON")

    parsed_args = arg_parse.parse_args()

    # --- Template generation mode ---
    if parsed_args.generate:
        out = make_abs(parsed_args.generate)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as fh:
            json.dump(DEFAULT_CONFIG, fh, indent=4)
        print(f"Template config written to {out}")
        return

    if not parsed_args.config:
        arg_parse.error("argument -c/--config is required (or use --generate to create a template)")

    # --- Load and validate config ---
    config_path = make_abs(parsed_args.config)
    with open(config_path) as fh:
        cfg = json.load(fh)

    for key in ("project_file", "inventory_file", "output_path"):
        if key not in cfg:
            raise KeyError(f"Required key '{key}' missing from config {config_path}")

    print("Input Configuration")
    print(json.dumps(cfg, indent=4))

    project_file = make_abs(cfg["project_file"])
    inventory_file = make_abs(cfg["inventory_file"])
    output_path = make_abs(cfg["output_path"])

    print(f"\nProcessing ANT spectral matrices")
    print(f"  Project   : {project_file}")
    print(f"  Inventory : {inventory_file}")
    print(f"  Output    : {output_path}")
    print(f"  Window    : {cfg.get('processing_window', 900)} s")
    print(f"  Resp. rem.: {cfg.get('remove_response', False)}"
          + (f"  ({cfg.get('units', 'VEL')}, WL={cfg.get('waterlevel', 90)})"
             if cfg.get("remove_response") else ""))
    print(f"  Decimation: {cfg.get('decimate', False)}"
          + (f"  (target {cfg.get('factor', 5)} Hz)" if cfg.get("decimate") else ""))
    print(f"  Time norm : {cfg.get('time_norm', False)}"
          + (f"  ({cfg.get('method', 'running avarage')})" if cfg.get("time_norm") else ""))
    print(f"  Whitening : {cfg.get('whiten', False)}")
    print(f"  Pre-filter: {cfg.get('prefilter', False)}"
          + (f"  ({cfg.get('filter_freqmin', 0.01)}–{cfg.get('filter_freqmax', 0.4)} Hz)"
             if cfg.get("prefilter") else ""))

    # --- Load project pickle produced by ant create_dict ---
    with open(project_file, "rb") as fh:
        project = pickle.load(fh)
    data_map = project["data_map"]
    info = project["info"]

    # --- Load inventory ---
    inventory = read_inventory(inventory_file)

    # --- Build param_dict directly from config, applying defaults for missing keys ---
    param_dict = {
        "processing_window": cfg.get("processing_window", 900),
        "f1": cfg.get("f1", 0.005),
        "f2": cfg.get("f2", 0.008),
        "f3": cfg.get("f3", 0.4),
        "f4": cfg.get("f4", 0.45),
        "waterlevel": cfg.get("waterlevel", 60.0),
        "units": cfg.get("units", "VEL"),
        "factor": cfg.get("factor", 5.0),
        "method": cfg.get("method", "running avarage"),
        "timewindow": cfg.get("timewindow", 25.0),
        "freqbandwidth": cfg.get("freqbandwidth", 0.02),
        "remove_responseCB": cfg.get("remove_response", False),
        "decimationCB": cfg.get("decimate", False),
        "time_normalizationCB": cfg.get("time_norm", False),
        "whitheningCB": cfg.get("whiten", False),
        "prefilter": cfg.get("prefilter", False),
        "filter_freqmin": cfg.get("filter_freqmin", 0.01),
        "filter_freqmax": cfg.get("filter_freqmax", 0.4),
        "filter_corners": cfg.get("filter_corners", 4),
    }

    # --- Flatten nested data_map into list_raw and run ---
    os.makedirs(output_path, exist_ok=True)
    processor = process_ant(output_path, param_dict, inventory)
    list_raw = processor.get_all_values(data_map["nets"])

    print(f"  Found {len(list_raw)} channel(s) to process.")

    processor.create_all_dict_matrix(list_raw, info)

    print(f"\nDone. Spectral matrices saved to {output_path}")


def _ant_cross_stack():
    """
    Command-line interface for cross-correlating and stacking ANT spectral matrices.
    """

    import json
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from surfquakecore.ant.crossstack import noisestack

    def make_abs(path):
        return os.path.abspath(os.path.expanduser(path))

    # ------------------------------------------------------------------ #
    # Default config — also used as the template written by --generate    #
    # ------------------------------------------------------------------ #
    DEFAULT_CONFIG = {
        "input_path": "/path/to/output/matrices",
        "output_path": "/path/to/output/stacks",
        "channels": ["Z"],
        "stations": [],
        "stack": "Linear",
        "power": 2.0,
        "autocorr": False,
        "min_distance": 1000.0,
        "daily_stacks": False,
        "overlap": 50.0,
        "workers": None,
        "rotate": False,
        "rotate_daily": False
    }

    arg_parse = ArgumentParser(
        prog="ant cross_stack",
        description="Cross-correlate and stack frequency-domain matrices for ANT",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
        Overview:

            Reads the spectral matrix pickle files produced by 'ant process_matrix'
            and computes cross-correlations for all station pairs, then stacks them.
            For each pair (i, j):

                1. Common days between the two stations are identified
                2. Duplicate days are removed
                3. Frequency-domain matrices are multiplied element-wise (conjugate)
                4. The result is inverse-transformed to the time domain
                5. Windows are stacked: Linear, n-root (nrooth) or Phase-Weighted (PWS)
                6. The stacked cross-correlation is saved as an HDF5 file

            Output subdirectories are created automatically inside output_path:
                stack/          Full stacked cross-correlations (one file per pair)
                stack_daily/    Per-day stacks (only if daily_stacks: true)
                stack_rotated/  ZNE-rotated horizontals (only if rotate: true)

            Note: input_path must point to the directory containing the matrix
            pickle files (the output_path of 'ant process_matrix'). The stack/,
            stack_daily/ and stack_rotated/ subdirectories are created inside
            that same directory.

            All parameters are supplied through a JSON configuration file.
            Run with --generate to create a pre-filled template.

        JSON config keys:
            input_path     [REQUIRED] Directory containing the spectral matrix pickles
            output_path    [REQUIRED] Directory where stack outputs will be written
            channels       [REQUIRED] List of channel suffix(es) to process (e.g. ["Z"] or ["N","E"])
            stations       [OPTIONAL] Station whitelist — empty list [] means all stations
            stack          [OPTIONAL] Stacking method: Linear, PWS, nrooth (default: Linear)
            power          [OPTIONAL] Exponent for PWS or nrooth stacking (default: 2.0)
            autocorr       [OPTIONAL] Include autocorrelations i==j, true/false (default: false)
            min_distance   [OPTIONAL] Maximum inter-station distance in km to include (default: 1000)
            daily_stacks   [OPTIONAL] Save per-day stacks in addition to full stack (default: false)
            overlap        [OPTIONAL] Overlap percentage for daily partial stacks (default: 75)
            workers        [OPTIONAL] Number of parallel worker processes (default: null = cpu_count - 1)
            rotate         [OPTIONAL] Run ZNE rotation after stacking (default: false)
            rotate_daily   [OPTIONAL] Run ZNE rotation on daily stacks after stacking (default: false)

        Documentation:
            https://projectisp.github.io/surfquaketutorial.github.io/

        Usage Examples:

            # Generate a template config file to fill in
            surfquake ant_cross_stack --generate config_cross_stack.json

            # Run with a filled config
            surfquake ant_cross_stack -c config_cross_stack.json
        """
    )

    arg_parse.add_argument("-c", "--config",
                           help="Path to the JSON configuration file",
                           type=str, required=False, default=None)

    arg_parse.add_argument("--generate",
                           help="Write a template JSON config to the given path and exit",
                           type=str, required=False, default=None, metavar="OUTPUT_JSON")

    parsed_args = arg_parse.parse_args()

    # --- Template generation mode ---
    if parsed_args.generate:
        out = make_abs(parsed_args.generate)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as fh:
            json.dump(DEFAULT_CONFIG, fh, indent=4)
        print(f"Template config written to {out}")
        return

    if not parsed_args.config:
        arg_parse.error("argument -c/--config is required (or use --generate to create a template)")

    # --- Load and validate config ---
    config_path = make_abs(parsed_args.config)
    with open(config_path) as fh:
        cfg = json.load(fh)

    for key in ("input_path", "output_path", "channels"):
        if key not in cfg:
            raise KeyError(f"Required key '{key}' missing from config {config_path}")

    print("Input Configuration")
    print(json.dumps(cfg, indent=4))

    input_path = make_abs(cfg["input_path"])
    output_path = make_abs(cfg["output_path"])
    channels = cfg["channels"]
    stations = cfg.get("stations", [])
    stack = cfg.get("stack", "Linear")
    power = cfg.get("power", 2.0)
    autocorr = cfg.get("autocorr", False)
    min_dist = cfg.get("min_distance", 1000.0)
    daily_stacks = cfg.get("daily_stacks", False)
    overlap = cfg.get("overlap", 75.0)
    workers = cfg.get("workers", None)
    rotate = cfg.get("rotate", False)
    rotate_daily = cfg.get("rotate_daily", False)

    print(f"\nCross-correlating and stacking ANT matrices")
    print(f"  Input     : {input_path}")
    print(f"  Output    : {output_path}")
    print(f"  Channels  : {channels}")
    print(f"  Stations  : {stations or '(all)'}")
    print(f"  Stack     : {stack}"
          + (f"  (power={power})" if stack in ("PWS", "nrooth") else ""))
    print(f"  Autocorr  : {autocorr}")
    print(f"  Max dist  : {min_dist} km")
    print(f"  Daily stacks: {daily_stacks}"
          + (f"  (overlap={overlap}%)" if daily_stacks else ""))
    print(f"  Rotate    : {rotate}  |  Rotate daily: {rotate_daily}")

    stacker = noisestack(
        output_files_path=input_path,
        stations=stations,
        channels=channels,
        stack=stack,
        power=power,
        autocorr=autocorr,
        min_distance=min_dist,
        dailyStacks=daily_stacks,
        overlap=overlap,
        cpu_count=workers,
    )

    stacker.run_cross_stack()

    if rotate:
        print("\nRunning horizontal rotation...")
        stacker.rotate_horizontals()

    if rotate_daily:
        print("\nRunning daily horizontal rotation...")
        stacker.rotate_specific_daily()

    print(f"\nDone. Results written under: {input_path}")

def _project():
    """
    Command-line interface for creating a seismic project.
    """
    from surfquakecore.project.surf_project import SurfProject
    arg_parse = ArgumentParser(
        prog=f"{__entry_point_name} project",
        description="Create a seismic project by indexing seismogram files and storing their metadata.",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
    
    Overview:
        This command creates a seismic project — a file-based database that stores paths
        to waveform files and their associated metadata. Projects simplify later processing.

    Usage Example:
        > surfquake project -d ./data_directory -s ./projects -n my_project -v

    Key Arguments:
        -d, --data           [REQUIRED]         Path to waveform data directory (or file pattern)
        -s, --save_path      [REQUIRED]         Directory to save the project file
        -n, --name           [REQUIRED]         Name of the project (e.g., "my_experiment")
        -v, --verbose        [OPTIONAL]         Print detailed file discovery and indexing logs

    Documentation:
        https://projectisp.github.io/surfquaketutorial.github.io/
    """
    )

    arg_parse.add_argument("-d", "--data_dir", help="Path to data files directory", type=str, required=True)

    arg_parse.add_argument("-s", "--save_dir", help="Path to directory where project will be saved", type=str,
                           required=True)

    arg_parse.add_argument("-n", "--project_name", help="Project Name", type=str, required=True)

    arg_parse.add_argument("-v", "--verbose", help="information of files included on the project",
                           action="store_true")

    parsed_args = arg_parse.parse_args()
    print(parsed_args)

    print(f"Project from {parsed_args.data_dir} saving to {parsed_args.save_dir} as {parsed_args.project_name}")
    sp = SurfProject(make_abs(parsed_args.data_dir))
    project_file_path = os.path.join(make_abs(parsed_args.save_dir), parsed_args.project_name)
    sp.search_files(verbose=parsed_args.verbose)
    print(sp)
    sp.save_project(path_file_to_storage=project_file_path)


def _pick():
    from surfquakecore.phasenet.phasenet_handler import PhasenetISP, PhasenetUtils
    from surfquakecore.project.surf_project import SurfProject
    from dateutil import parser
    arg_parse = ArgumentParser(
        prog=f"{__entry_point_name} pick",
        description="Use PhaseNet deep learning model to pick P- and S-wave arrivals.",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
    
    Overview:
        This command applies the PhaseNet deep neural network to estimate the arrival times
        of P- and S-waves in local and regional seismic events.

        Picks are automatically generated from your project traces and saved to the specified directory.

    Usage Example:
        > surfquake pick -f ./my_project.json -d ./picks_output -p 0.3 -s 0.3 --verbose

    Key Arguments:
        -f, --project_file        [REQUIRED]    Path to your seismic project file
        -d, --output_dir          [REQUIRED]    Directory to save pick results
        -pt, --p_thresh           [OPTIONAL]    Threshold for P-wave probability (0–1) (default: 0.3)
        -st, --s_thresh           [OPTIONAL]    Threshold for S-wave probability (0–1) (default: 0.3)
        -n, --net                 [OPTIONAL]    Network code filter (default: *)
        -s, --station             [OPTIONAL]    Station code filter (default: *, e.g. SFS|ARNO)
        -ch, --channel            [OPTIONAL]    Channel filter (default: *, e.g. BH?)
        --min_date                [OPTIONAL]    Filter Start date (format: YYYY-MM-DD HH:MM:SS), DEFAULT min date of the project
        --max_date                [OPTIONAL]    Filter End date   (format: YYYY-MM-DD HH:MM:SS), DEFAULT max date of the project
        -v, --verbose             [OPTIONAL]    Enable detailed logging

    Reference:
        Zhu & Beroza (2019). PhaseNet: A Deep-Neural-Network-Based Seismic Arrival-Time Picking Method,
        Geophysical Journal International.

    Documentation:
        https://projectisp.github.io/surfquaketutorial.github.io/
    """
    )

    arg_parse.usage = ("Run picker: -f [path to your project file] "
                       "-d [path to your pick saving directory] "
                       "-p [P-wave threshold] -s [S-wave threshold] --verbose")

    arg_parse.add_argument("-f", help="Path to your project file", type=str, required=True)
    arg_parse.add_argument("-d", help="Path to directory where picks will be saved", type=str, required=True)
    arg_parse.add_argument("-n", "--net", help="Network code filter", type=str)
    arg_parse.add_argument("-s", "--station", help="Station code filter", type=str)
    arg_parse.add_argument("-ch", "--channel", help="Channel code filter", type=str)
    arg_parse.add_argument("-pt", "--p_thresh", help="P-wave threshold", type=float, default=0.3)
    arg_parse.add_argument("-st", "--s_thresh", help="S-wave threshold", type=float, default=0.3)
    arg_parse.add_argument("--min_date", help="Start time filter: format 'YYYY-MM-DD HH:MM:SS.sss'", type=str)
    arg_parse.add_argument("--max_date", help="End time filter: format 'YYYY-MM-DD HH:MM:SS.sss'", type=str)
    arg_parse.add_argument("-v", "--verbose", help="Show detailed log output", action="store_true")

    parsed_args = arg_parse.parse_args()
    print(parsed_args)

    sp_loaded = SurfProject.load_project(path_to_project_file=make_abs(parsed_args.f))

    # --- Apply key filters ---
    filters = {}
    if parsed_args.net:
        filters["net"] = parsed_args.net
    if parsed_args.station:
        filters["station"] = parsed_args.station
    if parsed_args.channel:
        filters["channel"] = parsed_args.channel
    if filters:
        print(f"[INFO] Filtering project by: {filters}")
        sp_loaded.filter_project_keys(**filters)

        # --- Apply time filters ---
        min_date, max_date = None, None
        try:
            if parsed_args.min_date:
                # min_date = datetime.strptime(args.min_date, "%Y-%m-%d %H:%M:%S.%f")
                min_date = parser.parse(parsed_args.min_date)
            if parsed_args.max_date:
                # max_date = datetime.strptime(args.max_date, "%Y-%m-%d %H:%M:%S.%f")
                max_date = parser.parse(parsed_args.max_date)
            if min_date or max_date:
                print(f"[INFO] Filtering by time range: {min_date} to {max_date}")
                sp_loaded.filter_project_time(starttime=min_date, endtime=max_date, verbose=True)
        except ValueError as ve:
            print(f"[ERROR] Date format should be: 'YYYY-MM-DD HH:MM:SS.sss'")
            raise ve

    if len(sp_loaded.project) > 0 and isinstance(sp_loaded, SurfProject):
        picker = PhasenetISP(
            sp_loaded.project,
            amplitude=True,
            min_p_prob=parsed_args.p_thresh,
            min_s_prob=parsed_args.s_thresh,
            output=make_abs(parsed_args.d)
        )

        # Run PhaseNet picking
        picks = picker.phasenet()

        # Process and save results
        picks_results = PhasenetUtils.split_picks(picks)
        PhasenetUtils.convert2real(picks_results, parsed_args.d)
        PhasenetUtils.save_original_picks(picks_results, parsed_args.d)
        PhasenetUtils.write_nlloc_format(picks_results, parsed_args.d)
    else:
        print("Empty Project, Nothing to pick!")


def _polarity():
    from surfquakecore.project.surf_project import SurfProject
    from surfquakecore.first_polarity.get_pol import RunPolarity

    parser = ArgumentParser(
        prog="surfquake polarity",
        description="Determines Polarities from P-wave first motion",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
        
        Overview:
            Automatic P-wave first motion polarity determination. The inputs are the project and the picking file in 
            nonlinloc pick format (i.e., might be generated using command pick). 
            Output is an edited pick file with polarities.
        
        Reference:
            Chakraborty, M., Cartaya, C. Q., Li, W., Faber, J., Rümpker, G., Stoecker, H., & Srivastava, N. (2022). 
            PolarCAP–A deep learning approach for first motion polarity classification of earthquake waveforms. 
            Artificial Intelligence in Geosciences, 3, 46-52.
        
        Key Arguments:
        -p, --project_file_path     [REQUIRED] Path to a surfquake project
        -f, --picking_file_path     [REQUIRED] Path to a picking files
        -o, --output_file_path      [REQUIRED] Path to the new picking file edited with polarities
        -t, --thresh                [OPTIONAL] Threshold for Polarity declaration 

        
        Example usage:
        > surfquake polarity -f ./nll_picks.txt -p ./project_file.pkl -o ./nll_picks_polarities.txt -t 0.95 
        
        """
    )

    parser.add_argument("-p", "--project_file_path", required=True, help="Path to SurfProject .pkl")
    parser.add_argument("-f", "--picking_file_path", required=True, help="path to picking file", type=str)
    parser.add_argument("-o", "--output_file_path", required=True, help="output file", type=str)
    parser.add_argument("-t", "--thresh", required=False, help="P-wave threshold", type=float, default=0.9)

    args = parser.parse_args()

    picking_file = make_abs(args.picking_file_path)
    project_file = make_abs(args.project_file_path)
    output_file = make_abs(args.output_file_path)

    project = SurfProject.load_project(project_file)
    RunPolarity(project, picking_file, output_file, args.thresh).send_polarities()


def _focmec():
    from surfquakecore.utils.os_utils import OSutils

    parser = ArgumentParser(
        prog="surfquake focmec",
        description="Focal Mechanism from P-Wave first motion polarity",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
        
        Overview:
        
        FOCal MEChanism determinations (FOCMEC) software  for determining and 
        displaying double-couple earthquake focal mechanisms using the polarity and amplitude ratios of P and S waves.
        
        Reference:
        Snoke, J. A. (2003). “FOCMEC: Focal Mechanism Determinations.” 
        International Handbook of Earthquake and Engineering Seismology, Part B, Academic Press, pp. 1629–1630.
        https://seiscode.iris.washington.edu/projects/focmec
        
        
        Key Arguments:
        -d, --hyp_folder        [REQUIRED] Path to folder containing hyp files
        -o, --output_folder     [REQUIRED] Path to the output folder
        -a, --accepted          [OPTIONAL] Number of accepted wrong polarities (float, default 1.0)
        
        Example usage:
        > surfquake focmec -d ./folder_hyp_path -a 1.0 -o ./output_folder
        
        """
    )

    from surfquakecore.first_polarity.first_polarity import FirstPolarity
    parser.add_argument("-d", "--hyp_folder", required=True, help="path to folder containing hyp files",
                        type=str)
    parser.add_argument("-a", "--accepted", required=False, help="Number of accepted wrong polarities",
                        type=float, default=1.0)
    parser.add_argument("-o", "--output_folder", required=True, help="output folder", type=str)


    args = parser.parse_args()

    hyp_folder = make_abs(args.hyp_folder)
    output_folder = make_abs(args.output_folder)
    files_list = FirstPolarity.find_hyp_files(hyp_folder)
    if len(files_list) > 0:
        # Reset the output case of exist
        OSutils.delete_folder_contents(output_folder)
        for file in files_list:
            try:
                header = FirstPolarity.set_head(file)
                if file is not None:
                    file_input = FirstPolarity().create_input(file, header)

                    if FirstPolarity.check_no_empty(file_input):
                        FirstPolarity().run_focmec(file_input, args.accepted, output_folder)
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                traceback.print_exc()
    else:
        print("No files *hyp to compute focmec")


def _plotmec():
    parser = ArgumentParser(
        prog="surfquake plotmec",
        description="Focal Mechanism from P-wave polarity ",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
        Overview:
        
        Plot fault planes, T & P axis, and P-wave polarities.
        
        
        Key Arguments:
        -f, --focmec_file           [OPTIONAL] Path to a specific *.lst file (focmec output)
        -d, --focmec_folder_path    [OPTIONAL] Path to folder with all *.lst files (focmec output)
        -o, --output_folder         [REQUIRED] Path to the output folder
        -a, --all_solutions         [OPTIONAL] If set, all searching fault planes will be plot
        -p, --plot_polarities       [OPTIONAL] If plot P-Wave polarities on the beachball
        -m, --format                [OPTIONAL] Format output plot (defaults pdf)
        
        Example usage:

        > surfquake plotmec -d ./focmec_folder_path -o ./output_folder
        > surfquake plotmec -f ./focmec_file_path.lst -o ./output_folder -p -a -m pdf
            
        if output is not provided the beachball of the focal mechanism will be shown on screen, 
        but if user provide output folder, the beach ball plot will be saved in the folder
        
        """
    )
    from surfquakecore.first_polarity.first_polarity import FirstPolarity
    parser.add_argument("-f", "--focmec_file", required=False, help="file with focmec.lst solution", type=str)

    parser.add_argument("-d", "--focmec_folder_path", required=False, help="path to folder with all *.lst "
                                                                           "focmec solutions", type=str)
    parser.add_argument("-o", "--output_folder", required=True, help="output folder", type=str)
    parser.add_argument("-p", "--plot_polarities", required=False, help="plot P-Wave polarities", action="store_true")
    parser.add_argument("-a", "--all_solutions", required=False, help="plot all searching fault planes",
                        action="store_true")

    parser.add_argument("-m", "--format", required=False, help="format output plot (defaults pdf)",
                        type=str, default="pdf")

    args = parser.parse_args()

    firstpolarity_manager = FirstPolarity()

    if args.focmec_file:
        focmec_file = make_abs(args.focmec_file)
        focmec_files = [focmec_file]
    else:
        focmec_files = firstpolarity_manager.find_files(make_abs(args.focmec_folder_path))

    format = args.format

    for file in focmec_files:

        # Station, Az, Dip, Motion = firstpolarity_manager.get_dataframe(location_file)
        Station, Az, Dip, Motion = FirstPolarity.extract_station_data(file)
        cat, focal_mechanism = firstpolarity_manager.extract_focmec_info(file)
        file_output_name = FirstPolarity.extract_name(file)

        if args.output_folder:
            output_folder = make_abs(args.output_folder)
            name_str = os.path.basename(file)[:-3] + format
            output_folder_file = os.path.join(output_folder, name_str)
        else:
            output_folder_file = None

        Plane_A = focal_mechanism.nodal_planes.nodal_plane_1
        strike_A = Plane_A.strike
        dip_A = Plane_A.dip
        rake_A = Plane_A.rake
        extra_info = firstpolarity_manager.parse_solution_block(focal_mechanism.comments[0]["text"])
        P_Trend = extra_info['P,T']['Trend']
        P_Plunge = extra_info['P,T']['Plunge']
        T_Trend = extra_info['P,N']['Trend']
        T_Plunge = extra_info['P,N']['Plunge']

        misfit_first_polarity = focal_mechanism.misfit
        azimuthal_gap = focal_mechanism.azimuthal_gap
        number_of_polarities = focal_mechanism.station_polarity_count
        if args.all_solutions:
            solution_collection = cat[0]["focal_mechanisms"]
        else:
            solution_collection = None


        #
        first_polarity_results = {"First_Polarity": ["Strike", "Dip", "Rake", "misfit_first_polarity", "azimuthal_gap",
                                                     "number_of_polarities", "P_axis_Trend", "P_axis_Plunge",
                                                     "T_axis_Trend", "T_axis_Plunge"],
                                  "results": [strike_A, dip_A, rake_A, misfit_first_polarity, azimuthal_gap,
                                              number_of_polarities, P_Trend, P_Plunge, T_Trend, T_Plunge]}

        FirstPolarity.print_first_polarity_info(file_output_name, first_polarity_results)
        FirstPolarity.drawFocMec(strike_A, dip_A, rake_A, Station, Az, Dip, Motion, P_Trend, P_Plunge,
            T_Trend, T_Plunge, output_folder_file, plot_polarities=args.plot_polarities,
                                 solution_collection=solution_collection)


def _associate():
    from surfquakecore.real.real_core import RealCore

    arg_parse = ArgumentParser(
        prog=f"{__entry_point_name} associator",
        description="Use REAL to associate phase picks into unique seismic events.",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
    Overview:
        This command runs the REAL (Rapid Earthquake Association and Location) algorithm
        to group previously detected phase picks into seismic events.

        Picks should be organized per station/channel in standard format.

    Usage Example:
        > surfquake associate -i inventory.xml -p ./picks_folder -c real_config.ini -w ./working_dir -s ./associated_output -v

    Key Arguments:
        -i, --inventory_file     [REQUIRED] Path to station metadata (XML or RESP)
        -p, --picks_folder       [REQUIRED] Folder containing station pick files
        -c, --config_file        [REQUIRED] Path to REAL .ini configuration file
        -w, --work_dir           [REQUIRED] Working directory for REAL intermediate output
        -s, --save_dir           [REQUIRED] Output directory for associated pick results
        -v, --verbose            [OPTIONAL] Enable detailed logging

    Reference:
        Zhang et al. (2019), Rapid Earthquake Association and Location,
        Seismological Research Letters, https://doi.org/10.1785/0220190052

    Documentation:
        https://projectisp.github.io/surfquaketutorial.github.io/

        Additional Info:
          - Time file format: https://github.com/Dal-mzhang/LOC-FLOW/blob/main/LOCFLOW-CookBook.pdf
          - REAL guide: https://github.com/Dal-mzhang/REAL/blob/master/REAL_userguide_July2021.pdf
    """
    )

    arg_parse.add_argument("-i", "--inventory_file_path", help="Inventory file (i.e., *xml or dataless",
                           type=str,
                           required=True)

    arg_parse.add_argument("-p", "--data_dir", help="path to data picking folder",
                           type=str,
                           required=True)

    arg_parse.add_argument("-c", "--config_file_path", help="Path to real_config_file.ini",
                           type=str, required=True)

    arg_parse.add_argument("-w", "--work_dir_path", help="Path to working_directory "
                                                         "(Generated Travel Times)", type=str,
                           required=True)

    arg_parse.add_argument("-s", "--save_dir",
                           help="Path to directory where associated picks will be saved", type=str,
                           required=True)

    arg_parse.add_argument("-v", "--verbose", help="information of files included on the project",
                           action="store_true")

    parsed_args = arg_parse.parse_args()
    print(parsed_args)

    rc = RealCore(make_abs(parsed_args.inventory_file_path), make_abs(parsed_args.config_file_path),
                  make_abs(parsed_args.data_dir),
                  make_abs(parsed_args.work_dir_path), make_abs(parsed_args.save_dir))
    rc.run_real()

    print("End of Events AssociationProcess, please see for results: ", parsed_args.save_dir)


def _locate():
    from surfquakecore.earthquake_location.run_nll import NllManager, Nllcatalog

    arg_parse = ArgumentParser(
        prog=f"{__entry_point_name} locate seismic event",
        description="Locate a seismic event using NonLinLoc methodology.",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
    Overview:
        surfQuake uses a non-linear, global-search approach (NonLinLoc) to locate seismic events.
        The main inputs are:
            • A pick file in NonLinLoc format
            • A travel-time 'time' folder generated with the `pre-locate` step

        For details on input formats, visit: http://alomax.free.fr/nlloc/

    Usage Example:
        > surfquake locate -i inventory.xml -c locate_config.ini -o ./output_locations -g -s -n 10

    Key Arguments:
        -i, --inventory_file      [REQUIRED] Station metadata (XML, RESP)
        -c, --config_file         [REQUIRED] Path to NonLinLoc .ini configuration
        -o, --output_dir          [REQUIRED] Directory where location results will be saved
        -g, --generate_tt         [REQUIRED] Generate travel time files before location
        -s, --apply_station_corr  [OPTIONAL] Apply station corrections
        -n, --iterations          [OPTIONAL] Number of global search iterations (int, number of iterations if stations corrections is set)

    Reference:
        Lomax, A., Michelini, A., Curtis, A. (2009). Earthquake Location: Direct, Global-Search Methods.
        Encyclopedia of Complexity and System Science, Springer. DOI: https://doi.org/10.1007/978-0-387-30440-3

    Documentation:
        https://projectisp.github.io/surfquaketutorial.github.io/
        http://alomax.free.fr/nlloc/ (input format and model description)
    """
    )

    arg_parse.add_argument("-i", "--inventory_file_path", help="Inventory file (i.e., *xml or dataless",
                           type=str, required=True)

    arg_parse.add_argument("-c", "--config_file_path", help="Path to nll_config_file.ini", type=str,
                           required=True)

    arg_parse.add_argument("-o", "--out_dir_path", help="Path to output_directory ", type=str,
                           required=True)

    arg_parse.add_argument("-g", "--generate_grid", help="In case first runninng also generate Travel-Times",
                           action="store_true")

    arg_parse.add_argument("-s", "--stations_corrections", help="If you want to iterate to include "
                                                                "stations corrections, default iterations 5",
                           action="store_true")

    arg_parse.add_argument('-n', '--number_iterations', type=int, metavar='N', help='an integer for the '
                            'number of iterations', required=False)

    parsed_args = arg_parse.parse_args()
    print(parsed_args)

    nll_manager = NllManager(make_abs(parsed_args.config_file_path),
                             make_abs(parsed_args.inventory_file_path),
                             make_abs(parsed_args.out_dir_path))

    nll_manager.clean_output_folder()

    if parsed_args.generate_grid:
        print("Generating Velocity Grid")
        nll_manager.vel_to_grid()
        print("Generating Travel-Time Grid")
        nll_manager.grid_to_time()

    print("Starting Locations")
    if parsed_args.stations_corrections:
        if parsed_args.number_iterations:
            num_iter = int(parsed_args.number_iterations)
        else:
            num_iter = 5

        # including stations_corrections
        for i in range(num_iter):
            print("Running Location iteration", i)
            nll_manager.run_nlloc(num_iter=i + 1)

    else:
        nll_manager.run_nlloc()
    print("Finished Locations see output at, ", os.path.join(make_abs(parsed_args.out_dir_path), "loc"))
    nll_catalog = Nllcatalog(make_abs(parsed_args.out_dir_path))
    nll_catalog.run_catalog(make_abs(parsed_args.out_dir_path))
    print("Catalog done, finished process see catalog at ", make_abs(parsed_args.out_dir_path))


def _source():
    from surfquakecore.project.surf_project import SurfProject
    from surfquakecore.magnitudes.source_tools import ReadSource
    from surfquakecore.magnitudes.run_magnitudes import Automag

    arg_parse = ArgumentParser(
        prog=f"{__entry_point_name} source parameters estimation",
        description="Estimate source parameters using P- and S-wave spectra.",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
    Overview:
        surfQuake estimates source parameters:
            • Stress Drop
            • Attenuation (Q)
            • Source radius
            • Radiated energy
            • Local (ML) and moment (Mw) magnitudes

        It uses spectral fitting of P- and/or S-wave displacement spectra, following SourceSpec methodology.

    Usage Example:
        > surfquake source -i inventory.xml -p my_project.json -c source_config.yaml -l ./nlloc_outputs -o ./source_results

    Key Arguments:
        -i, --inventory_file     [REQUIRED] Path to station metadata (XML, RESP)
        -p, --project_file       [REQUIRED] Path to project file
        -c, --config_file        [REQUIRED] YAML configuration for SourceSpec
        -l, --hypocenter_dir     [REQUIRED] Directory containing NonLinLoc *.hyp files
        -o, --output_folder      [REQUIRED] Output folder for source parameter results
        -t, --large_scale        [OPTIONAL] Automatic long cut of teleseism event waveforms

    Reference:
        Satriano, C. (2023). SourceSpec – Earthquake source parameters from P- or S-wave displacement spectra. 
        DOI: https://doi.org/10.5281/ZENODO.3688587

    Documentation:
        https://projectisp.github.io/surfquaketutorial.github.io/
        https://sourcespec.readthedocs.io/en/stable/index.html
    """
    )

    arg_parse.add_argument("-i", "--inventory_file_path", help="Inventory file (i.e., *xml or dataless",
                           type=str, required=True)

    arg_parse.add_argument("-p", "--project_file_path", help="Project file path",
                           type=str, required=True)

    arg_parse.add_argument("-c", "--config_file_path", help="Path to source_config_file", type=str,
                           required=True)

    arg_parse.add_argument("-l", "--loc_files_path", help="Path to nll_hyp_files", type=str,
                           required=True)

    arg_parse.add_argument("-o", "--output_dir_path", help="Path to output_directory ", type=str,
                           required=True)

    arg_parse.add_argument("-t", "--large_scale",
                           help="If you want a long cut of signals for teleseism events (optional)",
                           action="store_true")

    parsed_args = arg_parse.parse_args()
    print(parsed_args)

    # load_project #
    sp_loaded = SurfProject.load_project(path_to_project_file=make_abs(parsed_args.project_file_path))
    print(sp_loaded)

    # Running stage
    summary_path_file = os.path.join(make_abs(parsed_args.output_dir_path), "source_summary.txt")

    if parsed_args.large_scale:
        scale = "teleseism"
    else:
        scale = "regional"

    mg = Automag(sp_loaded, make_abs(parsed_args.loc_files_path),
                 parsed_args.inventory_file_path, parsed_args.config_file_path,
                 make_abs(parsed_args.output_dir_path), scale=scale)

    mg.estimate_source_parameters()

    rs = ReadSource(make_abs(parsed_args.output_dir_path))
    summary = rs.generate_source_summary()
    rs.write_summary(summary, summary_path_file)


def _mti():
    from surfquakecore.project.surf_project import SurfProject
    from surfquakecore.moment_tensor.sq_isola_tools import BayesianIsolaCore
    from surfquakecore.moment_tensor.mti_parse import WriteMTI

    arg_parse = ArgumentParser(
        prog=f"{__entry_point_name} Moment Tensor Inversion",
        description="Estimate seismic moment tensors using Bayesian inversion.",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
    Overview:
        surfQuake provides a simple interface to estimate moment tensors for pre-located earthquakes
        using a Bayesian inversion approach, based on the Bayesian ISOLA method.

    Usage Example:
        surfquake mti -i inventory.xml -p my_project.json -c mti_config.ini -o ./mti_output -s

    Key Arguments:
        -i, --inventory_file    [REQUIRED] Path to station metadata file (XML, RESP)
        -p, --project_file      [REQUIRED] Path to the project file with waveforms and event metadata
        -c, --config_file       [REQUIRED] INI configuration file for inversion settings
        -o, --output_dir        [REQUIRED] Output directory for inversion results
        -s, --save_plots        [OPTIONAL] If set, saves plots of MT solutions and fits

    Reference:
        Vackář et al. (2017). Bayesian ISOLA: New Tool for Automated Centroid Moment Tensor Inversion,
        Geophysical Journal International, 210(2), 693–705. https://doi.org/10.1093/gji/ggx180

    Documentation:
        https://projectisp.github.io/surfquaketutorial.github.io/
    """
    )

    arg_parse.add_argument("-i", "--inventory_file_path", help="Inventory file (i.e., *xml or dataless",
                           type=str, required=True)

    arg_parse.add_argument("-p", "--path_to_project_file", help="Project file generated previoussly with surfquake "
                                                                "project", type=str, required=True)

    arg_parse.add_argument("-c", "--config_files_path", help="Path to the folder containing all config files "
                                                             "(one per event)", type=str, required=True)

    arg_parse.add_argument("-o", "--output_dir_path", help="Path to output_directory ", type=str,
                           required=True)

    arg_parse.add_argument("-s", "--save_plots", help=" In case user wants to save all output plots",
                           action="store_true")

    parsed_args = arg_parse.parse_args()
    print(parsed_args)

    sp = SurfProject.load_project(path_to_project_file=make_abs(parsed_args.path_to_project_file))
    print(sp)

    bic = BayesianIsolaCore(
        project=sp,
        inventory_file=make_abs(parsed_args.inventory_file_path),
        output_directory=make_abs(parsed_args.output_dir_path),
        save_plots=parsed_args.save_plots,
    )

    print("Starting Inversion")
    bic.run_inversion(mti_config=make_abs(parsed_args.config_files_path))

    print("Writing Summary")
    wm = WriteMTI(make_abs(parsed_args.output_dir_path))
    wm.mti_summary()
    print("End of process, please review output directory")


def _csv2xml():
    from surfquakecore.utils.create_station_xml import Convert

    arg_parse = ArgumentParser(
        prog=f"{__entry_point_name} Convert CSV to StationXML",
        description="Convert a CSV file of station metadata to a StationXML file.",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
    
    Overview:
        Convert a CSV station table to StationXML format.

        Required CSV columns:
            Net, Station, Lat, Lon, Elevation, Start_Date, StartTime, End_Date, EndTime

        Date format must follow: '%Y-%m-%d %H:%M:%S'

    Usage Example:
        surfquake csv2xml -c ./stations.csv -r ./resp_files -o ./output_dir -n my_stations.xml

    Key Arguments:
        -c, --csv_file_path          [REQUIRED] Path to input CSV file containing station metadata
        -r, --resp_files_path        [OPTIONAL] Path to RESP files for each station (optional if included in CSV)
        -o, --output_dir             [REQUIRED] Directory where the StationXML will be saved
        -n, --stations_xml_name      [REQUIRED] Desired filename for the output StationXML

    Input file header and first rows:
    
    Minimal Example:
    
    Net Station Lat Lon elevation start_date starttime end_date endtime
    WM ARNO 37.0988 -6.7322 117.0 2007-01-01 00:00:00 2050-12-31 23:59:59
    WM AVE 33.2981 -7.4133 230.0 2007-01-01 00:00:00 2050-12-31 23:59:59
    
    Full example:
    Net Station site_name Lat Lon elevation start_date starttime end_date endtime channel location_code sample_rate azimuth dip depth clock_drift
    WM ARNO ARNO_OBS_01 37.0988 -6.7322 117.0 2007-01-01 00:00:00 2050-12-31 23:59:59 HHZ 00 100.0 0.0 -90.0 0.0 1.2e-7
    WM ARNO ARNO_OBS_01 37.0988 -6.7322 117.0 2007-01-01 00:00:00 2050-12-31 23:59:59 HHN 00 100.0 0.0 0.0 0.0 1.2e-7
    
    Documentation:
        https://projectisp.github.io/surfquaketutorial.github.io/create_metadata/
    """
    )

    arg_parse.add_argument("-c", "--csv_file_path", help="file containing Net Station Lat Lon elevation "
        "start_date starttime end_date endtime, single spacing", type=str, required=True)

    arg_parse.add_argument("-r", "--resp_files_path", help="Path to the folder containing the response files",
                           type=str, required=False)

    arg_parse.add_argument("-o", "--output_path", help="Path to output xml file)", type=str, required=True)

    arg_parse.add_argument("-n", "--stations_xml_name", help="Name of the xml file to be saved", type=str,
                           required=True)

    parsed_args = arg_parse.parse_args()
    print(parsed_args)

    if parsed_args.resp_files_path:
        sc = Convert(make_abs(parsed_args.csv_file_path), resp_files=make_abs(parsed_args.resp_files_path))
    else:
        sc = Convert(make_abs(parsed_args.csv_file_path))
    data_map = sc.create_stations_xml()
    inventory = sc.get_data_inventory(data_map)
    sc.write_xml(make_abs(parsed_args.output_path), parsed_args.stations_xml_name, inventory)


def _buildcatalog():
    from surfquakecore.utils.manage_catalog import BuildCatalog, WriteCatalog

    arg_parse = ArgumentParser(
        prog=f"{__entry_point_name} Convert CSV file to stations.xml",
        description="Build a seismic event catalog by combining SurfQuake outputs.",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
    Overview:
        The `buildcatalog` tool combines results from SurfQuake modules (e.g., event picks, MTI, and source parameters)
        into a unified ObsPy-compatible catalog that can be exported to various formats.

    Usage Example:
        surfquake buildcatalog \\
            -e ./events_folder \\
            -s ./source_summary.csv \\
            -m ./mti_summary.csv \\
            -f quakeml \\
            -o ./final_catalog

    Key Arguments:
        -e, --event_folder       Folder containing individual event XML or pick files
        -s, --source_summary     CSV file summarizing source parameter results
        -m, --mti_summary        CSV file summarizing MTI results
        -f, --format             Output catalog format (e.g., quakeml, nordic, sc3ml, etc.)
        -o, --output_dir         Directory to write the resulting catalog file

    Documentation:
        https://projectisp.github.io/surfquaketutorial.github.io/utils/
        ObsPy catalog format info:
        https://projectisp.github.io/surfquaketutorial.github.io/manage_catalog/
    """
    )

    arg_parse.add_argument("-e", "--path_event_files_folder", help="Folder where are places your *hyp files", type=str,
                           required=True)

    arg_parse.add_argument("-s", "--path_source_summary_file", help='Path to the file containing '
                                                                    'the source spectrum results, source_summary.txt',
                           type=str, required=False, default=None)

    arg_parse.add_argument("-m", "--path_mti_summary_file", help="Path to the file containing the "
                                                                 "moment tensor results, summary_mti.txt",
                           type=str, required=False, default=None)

    arg_parse.add_argument("-f", "--catalog_format", help="catalog format, default QUAKEML", type=str, required=False,
                           default="QUAKEML")

    arg_parse.add_argument("-o", "--path_to_output_folder", help="Path to the ouput folder, where catalog "
                                                                 "will be saved", type=str, required=True)

    parsed_args = arg_parse.parse_args()
    print(parsed_args)

    if os.path.isdir(make_abs(parsed_args.path_to_output_folder)):
        pass
    else:
        try:
            os.makedirs(make_abs(parsed_args.path_to_output_folder))
        except Exception as error:
            print("An exception occurred:", error)

    catalog_path_pkl = os.path.join(make_abs(parsed_args.path_to_output_folder), "catalog_obj.pkl")
    catalog_path_surf = os.path.join(make_abs(parsed_args.path_to_output_folder), "catalog_surf.txt")

    bc = BuildCatalog(loc_folder=make_abs(parsed_args.path_event_files_folder),
                      source_summary_file=make_abs(parsed_args.path_source_summary_file),
                      output_path=make_abs(parsed_args.path_to_output_folder),
                      mti_summary_file=make_abs(parsed_args.path_mti_summary_file),
                      format=parsed_args.catalog_format)

    bc.build_catalog_loc()
    wc = WriteCatalog(catalog_path_pkl)
    wc.write_catalog_surf(catalog=None, output_path=catalog_path_surf)


def _buildmticonfig():
    from surfquakecore.moment_tensor.mti_parse import BuildMTIConfigs

    arg_parse = ArgumentParser(
        prog=f"{__entry_point_name} Create mti_config.ini from catalog and template",
        description="Automatically generate MTI config files from a catalog query.",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
    
    Overview:
        The `buildmticonfig` tool creates multiple `mti_config.ini` files using:
            - A catalog file (e.g., QuakeML or ObsPy Catalog)
            - A template configuration file for MTI
            - Optional filters (time, location, magnitude, depth)

        This is useful to prepare inversion configs for multiple events in batch.

    Usage Example:
        surfquake buildmticonfig -c ./catalog.xml -t ./mti_template.ini -o ./generated_configs \\
            -s "01/01/2024, 00:00:00.000" -e "01/02/2024, 00:00:00.000" -l 34.0 -a 36.0 \\
            -d -118.0 -k -116.0 -w 0 -f 20 -g 2.5 -p 6.0

    Key Arguments:
        -c, --catalog_file       Path to input event catalog file (QuakeML or ObsPy)
        -t, --template_file      Path to MTI config template (.ini)
        -o, --output_dir         Directory where MTI configs will be written

    Optional Filters:
        -s, --start_time         Start date (format: %d/%m/%Y, %H:%M:%S.%f)
        -e, --end_time           End date (same format)
        -l, --lat_min            Minimum latitude
        -a, --lat_max            Maximum latitude
        -d, --lon_min            Minimum longitude
        -k, --lon_max            Maximum longitude
        -w, --depth_min          Minimum depth (km)
        -f, --depth_max          Maximum depth (km)
        -g, --mag_min            Minimum magnitude
        -p, --mag_max            Maximum magnitude

    Documentation:
        https://projectisp.github.io/surfquaketutorial.github.io/utils/
    """
    )

    arg_parse.add_argument("-c", "--catalog_file_path", help="file to catalog.pkl file", type=str,
                           required=True)

    arg_parse.add_argument("-t", "--mti_config_template", help="mti_config template file", type=str,
                           required=True)

    arg_parse.add_argument("-o", "--output_folder", help="output folder path to save the mti config .ini "
                                                         "files", type=str, required=True)

    arg_parse.add_argument("-s", "--starttime", help="starttime to filter the catalog",
                           type=str, required=False)

    arg_parse.add_argument("-e", "--endtime", help="endtime to filter the catalog",
                           type=str, required=False)

    arg_parse.add_argument("-l", "--lat_min", help="minimum latitude filter to apply a geographic "
                                                   "filter to the catalog",
                           type=float, required=False)

    arg_parse.add_argument("-a", "--lat_max", help="maximum latitude filter to apply a geographic filter "
                                                   "to the catalog",
                           type=float, required=False)

    arg_parse.add_argument("-d", "--lon_min", help="maximum longitude filter to apply a geographic filter "
                                                   "to the catalog",
                           type=float, required=False)

    arg_parse.add_argument("-k", "--lon_max", help="maximum longitude filter to apply a geographic filter "
                                                   "to the catalog",
                           type=float, required=False)

    arg_parse.add_argument("-w", "--depth_min", help="minimum depth [km] filter to apply a geographic filter "
                                                     "to the catalog", type=float, required=False)

    arg_parse.add_argument("-f", "--depth_max", help="maximum depth [km] filter to apply a geographic filter "
                                                     "to the catalog", type=float, required=False)

    arg_parse.add_argument("-g", "--mag_min", help="minimum magnitude filter to apply a geographic filter "
                                                   "to the catalog", type=float, required=False)

    arg_parse.add_argument("-p", "--mag_max", help="maximum magnitude filter to apply a geographic filter "
                                                   "to the catalog", type=float, required=False)

    parsed_args = arg_parse.parse_args()
    print(parsed_args)

    if os.path.isdir(make_abs(parsed_args.output_folder)):
        pass
    else:
        try:
            os.makedirs(make_abs(parsed_args.output_folder))
        except Exception as error:
            print("An exception occurred:", error)

    print("Querying Catalog --> ", parsed_args.lat_min, parsed_args.lat_max, parsed_args.lon_min, parsed_args.lon_max,
          parsed_args.depth_min, parsed_args.depth_max, parsed_args.mag_min, parsed_args.mag_max)

    bmc = BuildMTIConfigs(catalog_file_path=make_abs(parsed_args.catalog_file_path),
                          mti_config=make_abs(parsed_args.mti_config_template),
                          output_path=make_abs(parsed_args.output_folder))

    bmc.write_mti_ini_file(starttime=parsed_args.starttime, endtime=parsed_args.endtime,
                           lat_min=float(parsed_args.lat_min),
                           lat_max=float(parsed_args.lat_max), lon_min=float(parsed_args.lon_min),
                           lon_max=float(parsed_args.lon_max),
                           depth_min=float(parsed_args.depth_min), depth_max=float(parsed_args.depth_max),
                           mag_min=float(parsed_args.mag_min),
                           mag_max=float(parsed_args.mag_max))


def _processing():
    from surfquakecore.data_processing.analysis_events import AnalysisEvents
    from surfquakecore.project.surf_project import SurfProject

    arg_parse = ArgumentParser(
        prog=f"{__entry_point_name} surfquake processing",
        description="Process or cut waveforms associated with seismic events.",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
    
    Overview:
        Process or cut waveforms associated with seismic events.
        You can:
            - Cut traces using event times and headers
            - Apply processing steps (filtering, normalization, etc.)
            - Optionally visualize the waveforms (interactive mode)
            - Apply a user-defined post-processing script before or after plotting
            - Warning: The process will automatically merge traces

    Modes:
        Default  : Interactive mode (plotting + user prompts)
        --auto   : Non-interactive mode (no plots, no prompts, outputs written automatically)

    Post-Script Logic:
        Use `--post_script` to apply a custom script to each event stream.
        Use `--post_script_stage` to control **when** it runs:
            • before : script runs before plotting (good for filtering, editing headers)
            • after  : script runs after plotting (good if picks or metadata are added)

    Usage Example:
        surfquake processing \\
            -p "./project.pkl" \\
            -i inventory.xml \\
            -e events.xml \\
            -c config.yaml \\
            -o ./output_folder \\
            --phases P,S,PcP \\
            --plot_config plot_settings.yaml \\
            --post_script custom_postproc.py \\
            --post_script_stage after

    Key Arguments:
        -p, --project_file         [OPTIONAL] Path to an existing project file
        -w, --wave_files           [OPTIONAL] Path or glob pattern to waveform files
        -i, --inventory_file       [OPTIONAL] Station metadata file (XML, RESP)
        -e, --event_file           [OPTIONAL] Event catalog in txt format (columns: date;hour;latitude;longitude;depth;magnitude) 
        -c, --config_file          [OPTIONAL] Processing configuration file (YAML)
        -a, --auto                 [OPTIONAL] Run in automatic mode
        -o, --output_folder        [OPTIONAL] Folder where processed files are saved
        -n, --net                  [OPTIONAL] project net filter Example: NET1,NET2,NET3
        -s, --station              [OPTIONAL] project station filter Example: STA1,STA2,STA3
        -ch, --channel,            [OPTIONAL]  project channel filter Example: HHZ,BHN
        -t, --cut_time             [OPTIONAL] pre & post first arrival in seconds (symmetric)
        -cs, --cut_start_time      [OPTIONAL] cut pre-first arrival in seconds
        -ce, --cut_end_time        [OPTIONAL] cut post-first arrival in seconds
        -r, --reference            [OPTIONAL] The pick time for cutting the waveforms is the origin time of the event
        --phases                   [OPTIONAL] Comma-separated list of phases for arrival estimation (e.g., P,S)
        --vel                      [OPTIONAL] Phase speed to estimate the first arrival time (reference for cutting waveform)
        --plot_config              [OPTIONAL] Optional plot configuration file (YAML)
        --post_script              [OPTIONAL] Python script to apply per event stream
        --post_script_stage        [OPTIONAL] When to apply the post-script: before | after (default: before)
    
    """
    )

    arg_parse.add_argument("-p", "--project_file", help="absolute path to project file", type=str,
                           required=False)

    arg_parse.add_argument(
        "-a", "--auto", help="Run in automatic processing mode (no plotting or prompts)", action="store_true"
    )

    arg_parse.add_argument("-w", "--wave_files", help="path to waveform files (e.g. './data/*Z')", type=str)

    arg_parse.add_argument("-o", "--output_folder", help="absolute path to output folder. Files are saved here",
                           type=str, required=False)

    arg_parse.add_argument("-i", "--inventory_file", help="stations metadata file.", type=str,
                           required=False)

    arg_parse.add_argument("-c", "--config_file", help="absolute path to config file", type=str,
                           required=False),

    arg_parse.add_argument("-e", "--event_file", help="absolute path to event file", type=str,
                           required=False)

    arg_parse.add_argument("-r", "--reference", help="The pick time for cutting "
                                                     "the waveforms is the origin time of the event",
                           type=str, required=False)

    arg_parse.add_argument("--phases", help="Comma-separated list of phases to use for travel "
                                            "time estimation (e.g., P,S,PcP)", type=str, required=False)

    arg_parse.add_argument("--vel", help="Phase speed to estimate the first arriva time ", type=str, required=False)

    arg_parse.add_argument("-n", "--net", help="project net filter", type=str, required=False)

    arg_parse.add_argument("-s", "--station", help="project station filter", type=str, required=False)

    arg_parse.add_argument("-ch", "--channel", help="project channel filter", type=str, required=False)

    arg_parse.add_argument("-t", "--cut_time", help="pre & post first arrival in seconds (symmetric). ",
                           type=float, required=False)

    arg_parse.add_argument("-cs", "--cut_start_time", help="cut pre-first arrival  in seconds", type=float,
                           required=False)

    arg_parse.add_argument("-ce", "--cut_end_time", help="cut post-first arrival  in seconds", type=float,
                           required=False)

    arg_parse.add_argument("--plot_config", help="Path to optional plotting configuration file (.yaml)",
                           type=str)

    arg_parse.add_argument(
        "--post_script",
        help="Path to Python script to apply to each event stream",
        type=str
    )

    arg_parse.add_argument(
        "--post_script_stage",
        help="When to apply the post-script: 'before' or 'after' plotting",
        choices=["before", "after"],
        default="before"
    )

    parsed_args = arg_parse.parse_args()

    # Parse phases if provided
    if parsed_args.phases:
        phase_list = [p.strip() for p in parsed_args.phases.split(",") if p.strip()]
        print(f"[INFO] Using phase list: {phase_list}")
    else:
        phase_list = None

    # 1. Estimate the start and end time
    if parsed_args.cut_start_time is not None:
        start = parsed_args.cut_start_time
        end = parsed_args.cut_end_time if parsed_args.cut_end_time is not None else 300
    elif parsed_args.cut_time is not None:
        start = end = parsed_args.cut_time
    else:
        start = end = 300

    # 2. Read and filter project
    filter = {}

    # Filter net
    if parsed_args.net is not None:
        filter['net'] = parsed_args.net

    # Filter station
    if parsed_args.station is not None:
        filter['station'] = parsed_args.station

    # Filter channel
    if parsed_args.channel is not None:
        filter['channel'] = parsed_args.channel

    print(parsed_args.wave_files)
    if parsed_args.wave_files:
        if "," in parsed_args.wave_files:
            # Explicit list of files
            wave_paths = [make_abs(f.strip()) for f in parsed_args.wave_files.split(",") if f.strip()]
            sp = SurfProject(root_path=wave_paths)
            sp.search_files(use_glob=True)
            print(f"[INFO] Using {len(wave_paths)} explicitly listed waveform files.")
        else:
            # Wildcard path # WARNING: Termina in user propm neds to be between " ", example: "./*Z"
            wave_paths = make_abs(parsed_args.wave_files)
            sp = SurfProject(root_path=wave_paths)
            sp.search_files(use_glob=True, verbose=True)
            print(f"[INFO] Found {len(sp.data_files)} waveform files using glob pattern.")

        if not wave_paths:
            raise ValueError("[ERROR] No waveform files found with --wave_files input.")

    elif parsed_args.project_file:

        sp = SurfProject.load_project(parsed_args.project_file)

    else:
        raise ValueError("You must specify either --project_file or --wave_files.")

    if len(filter) > 0:
        sp.filter_project_keys(**filter)

    if parsed_args.event_file is not None:

        # we want to loop over all events or reference times
        sp_sub_projects = sp.split_by_time_spans(event_file=make_abs(parsed_args.event_file), cut_start_time=start,
                                                 cut_end_time=end, verbose=True)
        sd = AnalysisEvents(make_abs(parsed_args.output_folder), make_abs(parsed_args.inventory_file),
                            make_abs(parsed_args.config_file),
                            sp_sub_projects, post_script=make_abs(parsed_args.post_script),
                            post_script_stage=parsed_args.post_script_stage,
                            plot_config_file=make_abs(parsed_args.plot_config), reference=parsed_args.reference,
                            phase_list=phase_list, vel=parsed_args.vel)
        sd.run_waveform_cutting(cut_start=start, cut_end=end, auto=parsed_args.auto)

    else:

        sd = AnalysisEvents(make_abs(parsed_args.output_folder), make_abs(parsed_args.inventory_file),
                            make_abs(parsed_args.config_file),
                            sp, post_script=make_abs(parsed_args.post_script),
                            post_script_stage=parsed_args.post_script_stage,
                            plot_config_file=make_abs(parsed_args.plot_config),
                            reference=parsed_args.reference, phase_list=phase_list)
        sd.run_waveform_analysis(auto=parsed_args.auto)


def _trigg():

    from surfquakecore.coincidence_trigger.coincidence_trigger import CoincidenceTrigger
    from surfquakecore.project.surf_project import SurfProject
    from datetime import timedelta

    arg_parse = ArgumentParser(
        prog=f"{__entry_point_name} computes coincidence trigger",
        description="Process seismograms in daily files to detect events using coincidence trigger",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
        
        Overview:
        Process seismograms in daily files to detect events using coincidence trigger. 
        It can be used either SNR or Kurtosis as Characterisitic functions.

        Allen, R. (1982). Automatic phase pickers: Their present use and future prospects. Bulletin of the
        Seismological Society of America, 72(6B), S225-S242.

        Poiata, N., C. Satriano, J.-P. Vilotte, P. Bernard, and K. Obara (2016). Multi-band array detection and 
        location of seismic sources recorded by dense seismic networks, Geophys. J. Int.,
        205(3), 1548-1573, doi:10.1093/gji/ggw071.

        Usage Example:
            surfquake trigg -c config.yaml -o ./output_folder -ch "HHZ" --min_date "2024-01-01 00:00:00" \\
            --max_date "2024-01-04 00:00:00" --span_seconds  86400 --picking_file ./pick.txt --plot
                
        Key Arguments:
            -p, --project_file        [REQUIRED] Path to a saved project files
            -o, --output_folder       [REQUIRED] Directory for processed output
            -c, --config_file         [REQUIRED] Processing configuration (YAML)
            -n, --net                 [OPTIONAL] Network code filter
            -s, --station             [OPTIONAL] Station code filter
            -ch, --channel            [OPTIONAL] Channel filter
            --min_date                [OPTIONAL] Filter Start date (format: YYYY-MM-DD HH:MM:SS), DEFAULT min date of the project
            --max_date                [OPTIONAL] Filter End date   (format: YYYY-MM-DD HH:MM:SS), DEFAULT max date of the project
            --span_seconds            [OPTIONAL] Select and merge files in sets of time spans, DEFAULT 86400
            --plot                    [OPTIONAL] Plot events and Characteristic Functions
            --picking_file            [OPTIONAL] I set a picking file this will be separated accoring to found events inside cluster
        """)

    arg_parse.add_argument("-p", "--project_file", required=True, help="Path to SurfProject .pkl")

    arg_parse.add_argument("-o", "--output_folder", required=True, help="Folder to save processed data")

    arg_parse.add_argument("-c", "--config_file", required=True, help="YAML config for processing")

    arg_parse.add_argument("--span_seconds", type=int, default=86400,
                           help="Time span to split your dataset (in seconds), default 86400s")

    # Filter arguments
    arg_parse.add_argument("-n", "--net", help="Network code filter", type=str)

    arg_parse.add_argument("-s", "--station", help="Station code filter", type=str)

    arg_parse.add_argument("-ch", "--channel", help="Channel code filter", type=str)

    arg_parse.add_argument("--min_date", help="Start time filter: format 'YYYY-MM-DD HH:MM:SS.sss'", type=str)

    arg_parse.add_argument("--max_date", help="End time filter: format 'YYYY-MM-DD HH:MM:SS.sss'",
                           type=str)

    arg_parse.add_argument("--plot", help="plot events & CFs",  action="store_true")

    arg_parse.add_argument("--picking_file", help="picking file to split", type=str)

    parsed_args = arg_parse.parse_args()
    print(parsed_args)

    # --- Load project --
    project_file = make_abs(parsed_args.project_file)
    sp = SurfProject.load_project(project_file)

    # --- Apply key filters ---
    filters = {}
    if parsed_args.net:
        filters["net"] = parsed_args.net
    if parsed_args.station:
        filters["station"] = parsed_args.station
    if parsed_args.channel:
        filters["channel"] = parsed_args.channel
    if filters:
        print(f"[INFO] Filtering project by: {filters}")
        sp.filter_project_keys(**filters)


    # --- Decide between time segment or split ---
    info = sp.get_project_basic_info()
    min_date = info["Start"]
    max_date = info["End"]
    dt1 = parse_datetime(min_date)
    dt2 = parse_datetime(max_date)

    diff = abs(dt2 - dt1)
    if diff < timedelta(days=1):
        sp.get_data_files()
        subprojects = [sp]

    else:
        print(f"[INFO] Splitting into subprojects every {parsed_args.span_seconds} seconds")
        subprojects = sp.split_by_time_spans(
            span_seconds=parsed_args.span_seconds,
            min_date=parsed_args.min_date,
            max_date=parsed_args.max_date,
            file_selection_mode="overlap_threshold",
            verbose=True)

    config_file = make_abs(parsed_args.config_file)
    picking_file = make_abs(parsed_args.picking_file)
    output_folder = make_abs(parsed_args.output_folder)

    ct = CoincidenceTrigger(subprojects, config_file, picking_file, output_folder, parsed_args.plot)
    ct.optimized_project_processing()

def _processing_daily():
    from surfquakecore.data_processing.analysis_events import AnalysisEvents
    from surfquakecore.project.surf_project import SurfProject
    from dateutil import parser

    arg_parse = ArgumentParser(
        prog=f"{__entry_point_name} processing continuous waveforms",
        description="Process and cut continuous waveform data",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
    Overview:
        Cut seismograms and apply processing to continuous waveforms.
        You can perform either or both of these operations.

    Modes:
        Default : Interactive mode (with plotting and prompts)
        --auto  : Non-interactive mode (no plots, no prompts, automatic output)

    Post-Script Logic:
        Use --post_script to apply a custom script per station/day stream.
        Use --post_script_stage to control when the script is applied:
            • before : apply script before plotting (e.g., for filtering or cleanup)
            • after  : apply script after plotting (e.g., to act on manual picks)

    Usage Example:
        surfquake processing_daily -i inventory.xml -c config.yaml -o ./output_folder \\
            --min_date "2024-01-01 00:00:00" --max_date "2024-01-02 00:00:00" --span_seconds  86400 \\
            --plot_config plot_settings.yaml --post_script my_custom.py --post_script_stage after

    Key Arguments:
        -p, --project_file        [REQUIRED] Path to a saved project files
        -o, --output_folder       [OPTIONAL] Directory for processed output
        -i, --inventory_file      [OPTIONAL] Station metadata (XML/RESP)
        -c, --config_file         [OPTIONAL] Processing configuration (YAML)
        -a, --auto                [OPTIONAL] Run in automatic mode
        -n, --net                 [OPTIONAL] Network code filter
        -s, --station             [OPTIONAL] Station code filter
        -ch, --channel            [OPTIONAL] Channel filter
        --min_date                [OPTIONAL] Filter Start date (format: YYYY-MM-DD HH:MM:SS), DEFAULT min date of the project
        --max_date                [OPTIONAL] Filter End date   (format: YYYY-MM-DD HH:MM:SS), DEFAULT max date of the project
        --time_tolerance          [OPTIONAL] Tolerance in seconds for time filtering, excluded files with smaller time span
        --span_seconds            [OPTIONAL] Select and merge files in sets of time spans, DEFAULT 86400
        --time_segment            [OPTIONAL] If set, process entire time window as a single merged stream of traces
        --plot_config             [OPTIONAL] Optional plotting configuration (YAML)
        --post_script             [OPTIONAL] Path to Python script for custom post-processing
        --post_script_stage       [OPTIONAL] When to apply the post-script: before | after (default: before)
    """
    )

    arg_parse.add_argument("-p", "--project_file", required=True, help="Path to SurfProject .pkl")

    arg_parse.add_argument(
        "-a", "--auto", help="Run in automatic processing mode (no plotting or prompts)", action="store_true")

    arg_parse.add_argument("-o", "--output_folder", help="Folder to save processed data")

    arg_parse.add_argument("-i", "--inventory_file", required=False, help="stations metadata file")

    arg_parse.add_argument("-c", "--config_file", required=False, help="YAML config for processing")

    arg_parse.add_argument("--plot_config", help="YAML file for plot customization")

    arg_parse.add_argument("--span_seconds", type=int, default=86400,
                           help="Time span to split your dataset (in seconds), default 86400s")

    arg_parse.add_argument("--time_segment", action="store_true",
                           help="If set, process entire time window as a single merged stream of traces")

    arg_parse.add_argument("--time_tolerance", type=int, default=120,
                           help="Tolerance in seconds for time filtering")
    # Filter arguments
    arg_parse.add_argument("-n", "--net", help="Network code filter", type=str)

    arg_parse.add_argument("-s", "--station", help="Station code filter", type=str)

    arg_parse.add_argument("-ch", "--channel", help="Channel code filter", type=str)

    arg_parse.add_argument("--min_date", help="Start time filter: format 'YYYY-MM-DD HH:MM:SS.sss'", type=str)

    arg_parse.add_argument("--max_date", help="End time filter: format 'YYYY-MM-DD HH:MM:SS.sss'", type=str)

    arg_parse.add_argument(
        "--post_script",
        help="Path to Python script to apply to each event stream",
        type=str
    )

    arg_parse.add_argument(
        "--post_script_stage",
        help="When to apply the post-script: 'before' or 'after' plotting",
        choices=["before", "after"],
        default="before"
    )

    parsed_args = arg_parse.parse_args()
    print(parsed_args)

    # --- Load project --

    sp = SurfProject.load_project(parsed_args.project_file)

    # --- Apply key filters ---
    filters = {}
    if parsed_args.net:
        filters["net"] = parsed_args.net
    if parsed_args.station:
        filters["station"] = parsed_args.station
    if parsed_args.channel:
        filters["channel"] = parsed_args.channel
    if filters:
        print(f"[INFO] Filtering project by: {filters}")
        sp.filter_project_keys(**filters)

    # --- Apply time filters ---
    min_date, max_date = None, None
    try:
        if parsed_args.min_date:
            # min_date = datetime.strptime(args.min_date, "%Y-%m-%d %H:%M:%S.%f")
            min_date = parser.parse(parsed_args.min_date)
        if parsed_args.max_date:
            # max_date = datetime.strptime(args.max_date, "%Y-%m-%d %H:%M:%S.%f")
            max_date = parser.parse(parsed_args.max_date)
        if min_date or max_date:
            print(f"[INFO] Filtering by time range: {min_date} to {max_date}")
            sp.filter_project_time(starttime=min_date, endtime=max_date, tol=parsed_args.time_tolerance, verbose=True)
    except ValueError as ve:
        print(f"[ERROR] Date format should be: 'YYYY-MM-DD HH:MM:SS.sss'")
        raise ve

    # --- Decide between time segment or split ---
    if parsed_args.time_segment:
        print(f"[INFO] Running single-segment analysis from {parsed_args.min_date} to {parsed_args.max_date}")
        subprojects = [sp]  # No splitting
    else:
        print(f"[INFO] Splitting into subprojects every {parsed_args.span_seconds} seconds")
        subprojects = sp.split_by_time_spans(
            span_seconds=parsed_args.span_seconds,
            min_date=parsed_args.min_date,
            max_date=parsed_args.max_date,
            file_selection_mode="overlap_threshold",
            verbose=True
        )

    # --- Run processing workflow ---
    ae = AnalysisEvents(
        output=make_abs(parsed_args.output_folder),
        inventory_file=make_abs(parsed_args.inventory_file),
        config_file= make_abs(parsed_args.config_file),
        surf_projects=subprojects,
        plot_config_file=make_abs(parsed_args.plot_config),
        time_segment_start=parsed_args.min_date,
        time_segment_end=parsed_args.max_date,
        post_script=make_abs(parsed_args.post_script),
        post_script_stage=parsed_args.post_script_stage,
        time_segment=parsed_args.time_segment
    )
    ae.run_waveform_analysis(auto=parsed_args.auto)

def _quickproc():
    from surfquakecore.data_processing.analysis_events import AnalysisEvents
    from surfquakecore.project.surf_project import SurfProject

    parser = ArgumentParser(
        prog="surfquake quick",
        description="Quick waveform processing. Designed for rapid, raw file-based workflows.",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
    
    Overview:
        Process seismic traces.
        You can:
            - Apply processing steps (filtering, normalization, etc.)
            - Optionally visualize the waveforms (interactive mode)
            - Apply a user-defined post-processing script before or after plotting

    Modes:
        Default  : Interactive mode (plotting + user prompts)
        --auto   : Non-interactive mode (no plots, no prompts, outputs written automatically)

    Post-Script Logic:
        Use `--post_script` to apply a custom script to each event stream.
        Use `--post_script_stage` to control **when** it runs:
            • before : script runs before plotting (good for filtering, editing headers)
            • after  : script runs after plotting (good if picks or metadata are added)

        Examples:
            Process waveform files with a config and plot interactively:
                surfquake quick -w "./data/*.mseed" -c ./config.yaml -i ./inventory.xml -o ./out --plot_config plot.yaml

        Key Arguments:
            -w, --wave_files         [REQUIRED] Glob pattern or path to waveform files
            -c, --config_file        [OPTIONAL] YAML config defining processing steps
            -i, --inventory_file     [OPTIONAL] Station metadata (StationXML or RESP)
            -o, --output_folder      [OPTIONAL] Directory to save processed traces
            -a, --auto               [OPTIONAL] Run in automatic (non-interactive) mode
            --plot_config            [OPTIONAL] Plotting settings YAML
            --post_script            [OPTIONAL] Python script to apply to each stream
            --post_script_stage      [OPTIONAL] When to run post-script: 'before' or 'after' (default: before)
        """
    )

    parser.add_argument("-w", "--wave_files", type=str, required=True)

    parser.add_argument(
        "-a", "--auto", help="Run in automatic processing mode (no plotting or prompts)",
        action="store_true")

    parser.add_argument("-c", "--config_file", type=str, required=False)

    parser.add_argument("-i", "--inventory_file", type=str, required=False)

    parser.add_argument("-o", "--output_folder", type=str, required=False)

    parser.add_argument("--plot_config", type=str)

    parser.add_argument(
        "--post_script",
        help="Path to Python script to apply to each event stream",
        type=str
    )

    parser.add_argument(
        "--post_script_stage",
        help="When to apply the post-script: 'before' or 'after' plotting",
        choices=["before", "after"],
        default="before"
    )

    parsed_args = parser.parse_args()
    print(parsed_args)

    if "," in parsed_args.wave_files or " " in parsed_args.wave_files:
        # Explicit list of files
        wave_paths = [make_abs(f.strip()) for f in parsed_args.wave_files.split(",") if f.strip()]

    else:
        # Wildcard path
        wave_paths = make_abs(parsed_args.wave_files)

     # Build project on-the-fly using wildcard path
    data_files = SurfProject.collect_files(root_path=wave_paths)

    # --- Run processing workflow ---
    ae = AnalysisEvents(
        output=make_abs(parsed_args.output_folder),
        inventory_file=make_abs(parsed_args.inventory_file),
        config_file=make_abs(parsed_args.config_file),
        surf_projects=[],
        plot_config_file=make_abs(parsed_args.plot_config),
        post_script=make_abs(parsed_args.post_script),
        post_script_stage=parsed_args.post_script_stage
    )

    ae.run_fast_waveform_analysis(data_files, auto=parsed_args.auto)

def _specplot():
    from surfquakecore.spectral.specrun import TraceSpectrumResult, TraceSpectrogramResult
    from surfquakecore.spectral.cwtrun import TraceCWTResult

    parser = ArgumentParser(
        prog="surfquake specplot",
        description="Plot serialized spectral analysis (spectrum, spectrogram or cwt)",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
        
        Overview:
        Plot spectrograms from saved binary files after processing waveforms
        
        Key Arguments:
            -f, --file         [REQUIRED] Path to waveform files (.sp, .spec or .cwt)
            -c, --clip         [OPTIONAL] Clipping level in dB for plotting the time-Frequency plane (default -120)
                --save_path    [OPTIONAL] Output file path to automatically save the figure
                
        Examples:
            Plot a saved spectrum:
                surfquake specplot --file ./cut/spec/IU.HKT.00.BHZ.sp
        
            Plot a saved spectrogram:
                surfquake specplot --file ./cut/spec/IU.HKT.00.BHZ.spec --clip -120.0
        
            Save plot to a file:
                surfquake specplot -f ./cut/spec/IU.HKT.00.BHZ.spec --save_path output.png
        
        """
    )

    parser.add_argument("--file", "-f", required=True, help="Path to the serialized .sp, .spec or .cwt file")
    parser.add_argument("--clip", "-c", type=float, required=False)
    parser.add_argument("--save_path", help="Optional path to save the figure (e.g., output.png)")

    args = parser.parse_args()
    filepath = args.file
    _, ext = os.path.splitext(filepath)

    ext = ext.lower()

    if ext == ".sp":
        obj = TraceSpectrumResult.from_pickle(filepath)
        obj.plot_spectrum(save_path=args.save_path)

    elif ext == ".spec":
        obj = TraceSpectrogramResult.from_pickle(filepath)
        obj.plot_spectrogram(save_path=args.save_path, clip=args.clip)

    elif ext == ".cwt":
        obj = TraceCWTResult.from_pickle(filepath)
        obj.plot_cwt(save_path=args.save_path, clip=args.clip)

    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            "Expected one of: .sp (spectrum), .spec (spectrogram), .cwt (cwt)."
        )


def _beamplot():
    from surfquakecore.arrayanalysis.beamrun import TraceBeamResult
    parser = ArgumentParser(
        prog="surfquake beamplot",
        description="Plot serialized beamforming result and optionally extract peaks",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
        
        Key Arguments:
            -f, --file              [REQUIRED] Path to waveform files (.sp, .spec or .cwt)
                --save_path         [OPTIONAL] Output file path to automatically save the figure
                --find_solutions    [OPTIONAL] Automatically find peaks in tn relative power of the beamforming
                --write_path        [OPTIONAL] Path to te output txt file with the automatic solutions
                --baz_range         [OPTIONAL] Filter the automatic search for range of azimuth
                --min_power         [OPTIONAL] Relative Power threshold for searching automatic solutions (default 0.6)
        
        Examples:
          Plot a saved FK beam:
              surfquake beamplot --file ./output/2024.123.beam
        
          Detect peaks with regional phase constraints:
              surfquake beamplot --file event.beam --find_solutions regional --write_path solutions.txt
        
          Use custom slowness constraints:
              surfquake beamplot --file file_path --find_solutions '{"Pn": [0.05, 0.08], "Lg": [0.18, 0.30]}'
        
          Apply azimuth filtering:
              surfquake beamplot -f file_path --find_solutions regional --baz_range 100 150
        
          Control minimum power threshold:
              surfquake beamplot -f file_path --find_solutions teleseismic --min_power 0.75
        """
    )

    parser.add_argument("--file", "-f", required=True, help="Path to the .beam file (gzip-pickled TraceBeamResult)")
    parser.add_argument("--save_path", help="Optional path to save the beam figure (e.g., output.png)")
    parser.add_argument("--find_solutions", help="Phase constraint: 'regional', 'teleseismic', or dict string")
    parser.add_argument("--baz_range", nargs=2, type=float, metavar=('MIN', 'MAX'),
                        help="Backazimuth range filter in degrees (e.g., 90 140)")
    parser.add_argument("--min_power", type=float, default=0.6,
                        help="Minimum relative power required to accept a peak (default: 0.6)")
    parser.add_argument("--write_path", help="Append detected peaks to specified TXT file")

    args = parser.parse_args()

    # Load object
    beam_obj = TraceBeamResult.from_pickle(make_abs(args.file))

    # Process peak detection if requested
    if args.find_solutions:
        try:
            baz_range = tuple(args.baz_range) if args.baz_range else None

            results = beam_obj.detect_beam_peaks(
                phase_dict=args.find_solutions,
                peak_kwargs={"prominence": 0.1, "distance": 10},
                min_power=args.min_power,
                output_file=args.write_path,
                bazimuth_range=baz_range
            )

            if len(results) > 0:
                total_peaks = sum(len(peaks) for peaks in results.values())
                print(f"[INFO] Found {total_peaks} beam peaks:")
                for phase, peaks in results.items():
                    for t, az, s, pwr in peaks:
                        print(f"  {t} | Phase: {phase:8} | "
                              f"BAz: {az:6.1f}° | S: {s:.3f} s/km | Pow: {pwr:.2f}")

        except Exception as e:
            print(f"[ERROR] Failed to detect peaks: {e}")

        beam_obj.plot_beam(save_path=make_abs(args.save_path) if args.save_path else None)

def _info():
    from surfquakecore.data_processing.processing_methods import print_surfquake_trace_headers
    from surfquakecore.project.surf_project import SurfProject

    parser = ArgumentParser(
        prog="surfquake info",
        description="Prints waveform information ",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
        
        Overview:
        
        Extracts and prints on screen the traces header information
        
        Key Arguments:
            -w, --wave_files   [REQUIRED] Glob pattern or path to waveform files
            -c, --columns      [OPTIONAL] Number of columns showing traces info (default = 5)

        Example usage:

        > surfquake info -w './data/*.mseed' -c 3
        > surfquake info -w 'trace1.mseed,trace2.mseed' --columns 5

        Supports standard and SurfQuake-extended headers such as picks, references, and geodetic attributes.

    """
    )

    parser.add_argument("-w", "--wave_files", required=True, help="path to waveform files (e.g. './data/*Z')", type=str)
    parser.add_argument("-c", "--columns", help="Number of traces (columns) to show per file (default = 5).",
                        type=int, default=1)

    args = parser.parse_args()

    if "," in args.wave_files or " " in args.wave_files:
        # Explicit list of files
        wave_paths = [make_abs(f.strip()) for f in args.wave_files.split(",") if f.strip()]

    else:
        # Wildcard path
        wave_paths = make_abs(args.wave_files)
     # Build project on-the-fly using wildcard path

    data_files = SurfProject.collect_files(root_path=wave_paths)
    print_surfquake_trace_headers(data_files, max_columns=args.columns)


def _explore():
    from surfquakecore.seismoplot.availability import PlotExplore
    from surfquakecore.project.surf_project import SurfProject
    from dateutil import parser

    arg_parse = ArgumentParser(
        prog="surfquake explore",
        description="Explore data availability ",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
    
    Overview:
    
    Plot the data availability of your seismogram files
    
    Key Arguments:
            -w, --wave_files          [OPTIONAL] Glob pattern or path to waveform files
            -p, --project_path_file   [OPTIONAL] Project path file
            -n, --net                 [OPTIONAL] Network code filter (if -p selected)
            -s, --station             [OPTIONAL] Station code filter (if -p selected)
            -ch, --channel            [OPTIONAL] Channel filter (if -p selected)
            --min_date                [OPTIONAL] Filter Start date (format: YYYY-MM-DD HH:MM:SS), DEFAULT min date of the project (if -p selected)
            --max_date                [OPTIONAL] Filter End date   (format: YYYY-MM-DD HH:MM:SS), DEFAULT max date of the project (if -p selected)
    
    * Either -w or -p must be set     
    
    Example usage:

    surfquake explore -w './data/*.mseed'
    surfquake explore -p './project.pkl'
    """
    )

    arg_parse.add_argument("-w", "--wave_files", help="path to waveform files (e.g. './data/*Z')", type=str)
    arg_parse.add_argument("-p", "--project_file_path", help="scan files from the project", type=str)
    arg_parse.add_argument("-n", "--net", help="Network code filter", type=str)
    arg_parse.add_argument("-s", "--station", help="Station code filter", type=str)
    arg_parse.add_argument("-ch", "--channel", help="Channel code filter", type=str)
    arg_parse.add_argument("--min_date", help="Start time filter: format 'YYYY-MM-DD HH:MM:SS.sss'", type=str)
    arg_parse.add_argument("--max_date", help="End time filter: format 'YYYY-MM-DD HH:MM:SS.sss'", type=str)

    args = arg_parse.parse_args()

    if args.project_file_path:
        project_file_path = make_abs(args.project_file_path)
        sp = SurfProject.load_project(project_file_path)

        # --- Apply key filters ---
        filters = {}
        if args.net:
            filters["net"] = args.net
        if args.station:
            filters["station"] = args.station
        if args.channel:
            filters["channel"] = args.channel
        if filters:
            print(f"[INFO] Filtering project by: {filters}")
            sp.filter_project_keys(**filters)
            sp.get_data_files()
            # --- Apply time filters ---
            min_date, max_date = None, None
            try:
                if args.min_date:
                    # min_date = datetime.strptime(args.min_date, "%Y-%m-%d %H:%M:%S.%f")
                    min_date = parser.parse(args.min_date)
                if args.max_date:
                    # max_date = datetime.strptime(args.max_date, "%Y-%m-%d %H:%M:%S.%f")
                    max_date = parser.parse(args.max_date)
                if min_date or max_date:
                    print(f"[INFO] Filtering by time range: {min_date} to {max_date}")
                    sp.filter_project_time(starttime=min_date, endtime=max_date, verbose=True)
            except ValueError as ve:
                print(f"[ERROR] Date format should be: 'YYYY-MM-DD HH:MM:SS.sss'")
                raise ve

        data_files = sp.data_files

    else:
        if "," in args.wave_files or " " in args.wave_files:
            # Explicit list of files
            wave_paths = [make_abs(f.strip()) for f in args.wave_files.split(",") if f.strip()]

        else:
            # Wildcard path
            wave_paths = make_abs(args.wave_files)
         # Build project on-the-fly using wildcard path

        data_files = SurfProject.collect_files(root_path=wave_paths)
    PlotExplore.data_availability_new(data_files)


def _ppsdDB():
    """
    Command-line interface for creating a DB of Power Density functions
    """

    from surfquakecore.project.surf_project import SurfProject
    from surfquakecore.PPSD.PPSD import PPSDSurf

    arg_parse = ArgumentParser(
        prog=f"{__entry_point_name} ppsdDB",
        description="Create a DB of Power Spectral Density Functions",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""

        Overview:

            This command creates a file-based database that stores Probability Power Spectral Density Functions (PPSD) 
            in a pickle file. The file is basically a dictionary with keys net, station channel and the 
            ObsPy object with the PPSD.

            - Peterson, J. (1993), Observations and Modeling of Seismic Background Noise, U.S. Geological Survey 
              open-file report 93-322, Albuquerque, N.M.

            - McNamara, D. E. and Buland, R. P. (2004), Ambient Noise Levels in the Continental United States,
              Bulletin of the Seismological Society of America, 94 (4), 1517-1527.

        Key Arguments:
            -p,  --project_file        [REQUIRED] Path to waveform data directory (or file pattern)
            -i,  --inventory_file      [REQUIRED] Path to stations metadata (XML or RESP)
            -s,  --save_file           [REQUIRED] Path to save the data base
            -le, --length              [OPTIONAL] Length of data segments passed to psd in seconds (default 3600)
            -ov, --overlap             [OPTIONAL] Overlap in percentage of segments passed to psd (default 50)
            -sm, --smoothing           [OPTIONAL] PSDs are averaged over a octave at each central freq (default 1)
            -n,  --net                 [OPTIONAL] project net filter (Default: *, Example WM)
            -s,  --station             [OPTIONAL] project station filter (Default: *, Example ARNO)
            -ch, --channel             [OPTIONAL] project channel filter (Default: *, Example BH?)

        Documentation:
            https://projectisp.github.io/surfquaketutorial.github.io/

        Usage Example:

        surfquake ppsdDB -p "./ppsd_test" -i "./meta/metadata.xml" -s "./output/test.pkl"
        """
    )

    arg_parse.add_argument("-p", "--project_file", help="Path to waveform data directory", type=str,
                           required=True)

    arg_parse.add_argument("-i", "--inventory_file", help="Path to the stations metadata", type=str,
                           required=True)

    arg_parse.add_argument("-s", "--save_file", help="Path to file where PPSD DB will be saved", type=str,
                           required=True)

    arg_parse.add_argument("-le", "--length", help="Time window length for processing", type=float,
                           required=False, default=3600.0)

    arg_parse.add_argument("-sm", "--smoothing", help="Smoothing", type=float, required=False,
                           default=1.0)

    arg_parse.add_argument("-ov", "--overlap", help="Overlap", type=float, required=False,
                           default=50.0)

    arg_parse.add_argument("-pr", "--period", help="Period", type=float, required=False,
                           default=0.125)

    arg_parse.add_argument("-nt", "--net", help="Net Selection", type=str, required=False,
                           default="*")

    arg_parse.add_argument("-st", "--station", help="Station Selection", type=str, required=False,
                           default="*")

    arg_parse.add_argument("-ch", "--channel", help="Channel Selection", type=str, required=False,
                           default="*")

    parsed_args = arg_parse.parse_args()
    print("Input Arguments")
    print(parsed_args)

    project_file = make_abs(parsed_args.project_file)
    save_file = make_abs(parsed_args.save_file)
    inventory_file = make_abs(parsed_args.inventory_file)

    print(f"\nCreating PPSD DB")
    print(f"  Project  : {project_file}")
    print(f"  Inventory: {inventory_file}")
    print(f"  Output   : {save_file}")

    sp = SurfProject.load_project(project_file)
    print(sp)

    ppsds = PPSDSurf(files_path=sp, metadata=inventory_file, length=parsed_args.length, smoothing=parsed_args.smoothing,
                     period=parsed_args.period, overlap=parsed_args.overlap)

    ini_dict, size = ppsds.create_dict(net_list=parsed_args.net, sta_list=parsed_args.station,
                                       chn_list=parsed_args.channel)
    print(f"  Found {size} channel file(s) to process.")

    db = ppsds.get_all_values(ini_dict)
    ppsds.save_PPSDs(db, file_name=save_file)
    print(f"\nDone. Database saved to {save_file}")


def _ppsdPlot():
    """
    Command-line interface for plotting a PPSD pickle database.

    Supports three plot modes selected via --mode:
        heatmap    – probability density heatmaps, paginated by station groups
        variation  – diurnal or seasonal noise variation heatmaps
        comparison – overlay mean/median/mode curves across stations/channels
    """

    from surfquakecore.PPSD.plotPPS import plot_ppsds_from_pickle, plot_all_pages, plot_comparison

    arg_parse = ArgumentParser(
        prog=f"{__entry_point_name} ppsdPlot",
        description="Plot a PPSD pickle database",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
    Overview:

        Reads a PPSD pickle database produced by  ppsdDB  and generates
        publication-quality figures.  Three plot modes are available:

        heatmap    Probability density colormaps (classic PPSD view).
                   One figure per group of --spp stations.  Use --all to
                   iterate automatically over every station group.

        variation  Diurnal or seasonal noise variation shown as a contour
                   plot (mode amplitude vs hour-of-day or month).

        comparison Overlay mean / median / mode curves for a wildcard
                   selection of stations and channels on shared axes.
                   One panel per component (Z, N, E …) or a single panel.

    Key Arguments:
        -d,  --db_file         [REQUIRED] Path to the PPSD pickle database
        -m,  --mode            [REQUIRED] Plot mode: heatmap | variation | comparison
        -sd, --save_dir        [OPTIONAL] Directory to save figures (one per page)
        -sp, --save_path       [OPTIONAL] Single output file path (heatmap p.0 or comparison)
        -sf, --save_format     [OPTIONAL] Figure format: png | pdf | svg          (png)
        -nt, --net             [OPTIONAL] Network wildcard filter                  (*)
        -st, --station         [OPTIONAL] Station wildcard filter                  (*)
        -ch, --channel         [OPTIONAL] Channel wildcard filter, e.g. BH?        (*)
        -t0, --starttime       [OPTIONAL] Start of time window, ISO format
        -t1, --endtime         [OPTIONAL] End of time window, ISO format

      heatmap / variation only:
        --spp                  [OPTIONAL] Stations per page                          (3)
        --page                 [OPTIONAL] Page index (0-based); ignored with --all   (0)
        --all                  [OPTIONAL] Iterate over all station groups
        --variation            [OPTIONAL] Variation type: Diurnal | Seasonal  (Diurnal)
        --mean                 [OPTIONAL] Overlay mean curve
        --mode                 [OPTIONAL] Overlay mode curve
        --nhnm                 [OPTIONAL] Overlay NHNM reference
        --nlnm                 [OPTIONAL] Overlay NLNM reference
        --earthquakes          [OPTIONAL] Overlay earthquake model lines
        --min_mag              [OPTIONAL] Minimum magnitude for eq. models          (0.0)
        --max_mag              [OPTIONAL] Maximum magnitude for eq. models         (10.0)
        --min_dist             [OPTIONAL] Minimum distance for eq. models (km)      (0.0)
        --max_dist             [OPTIONAL] Maximum distance for eq. models (km)   (10000)

      comparison only:
        --stats                [OPTIONAL] Comma-separated list: mean,median,mode   (mean)
        --layout               [OPTIONAL] by_component | single          (by_component)

    Documentation:
        https://projectisp.github.io/surfquaketutorial.github.io/

    Usage examples:
        surfquake ppsdPlot -d "./output/test.pkl" --spp 1 --all -m heatmap --mean --nhnm --nlnm --earthquakes --min_mag 1.0 --max_mag 3.0
        surfquake ppsdPlot -d "./output/test.pkl" -st "OBS01,OBS02" -m heatmap --mean --nhnm --nlnm --earthquakes --min_mag 1.0 --max_mag 3.0
        surfquake ppsdPlot -d "./output/test.pkl"  -m variation --variation Diurnal
        surfquake ppsdPlot -d "./output/test.pkl" comparison --mean --nhnm --nlnm
        """
    )

    # ---- required -----------------------------------------------------------
    arg_parse.add_argument(
        "-d", "--db_file",
        help="Path to the PPSD pickle database (.pkl)",
        type=str, required=True,
    )
    arg_parse.add_argument(
        "-m", "--mode",
        help="Plot mode: heatmap | variation | comparison",
        type=str, required=True,
        choices=["heatmap", "variation", "comparison"],
    )

    # ---- output -------------------------------------------------------------
    arg_parse.add_argument(
        "-sd", "--save_dir",
        help="Directory to save figures (one file per page; auto-created)",
        type=str, required=False, default=None,
    )
    arg_parse.add_argument(
        "-sp", "--save_path",
        help="Single output file path (used for page 0 or comparison)",
        type=str, required=False, default=None,
    )
    arg_parse.add_argument(
        "-sf", "--save_format",
        help="Figure format when saving to a directory: png | pdf | svg (default: png)",
        type=str, required=False, default="png",
        choices=["png", "pdf", "svg", "jpg"],
    )
    arg_parse.add_argument(
        "--prefix",
        help="Filename prefix when saving multiple pages (default: ppsd_page)",
        type=str, required=False, default="ppsd_page",
    )

    # ---- station / channel filters ------------------------------------------
    arg_parse.add_argument(
        "-nt", "--net",
        help="Network wildcard filter, comma-separated (default: *)",
        type=str, required=False, default="*",
    )
    arg_parse.add_argument(
        "-st", "--station",
        help="Station wildcard filter, comma-separated (default: *)",
        type=str, required=False, default="*",
    )
    arg_parse.add_argument(
        "-ch", "--channel",
        help="Channel wildcard filter, e.g. BH? or HHZ,BHZ (default: *)",
        type=str, required=False, default="*",
    )

    # ---- time window --------------------------------------------------------
    arg_parse.add_argument(
        "-t0", "--starttime",
        help="Start of time window (ISO 8601, e.g. 2023-01-01T00:00:00)",
        type=str, required=False, default=None,
    )
    arg_parse.add_argument(
        "-t1", "--endtime",
        help="End of time window (ISO 8601, e.g. 2023-03-31T23:59:59)",
        type=str, required=False, default=None,
    )

    # ---- heatmap / variation options ----------------------------------------
    arg_parse.add_argument(
        "--spp",
        help="Stations per page for heatmap/variation mode (default: 3)",
        type=int, required=False, default=3,
        dest="stations_per_page",
    )
    arg_parse.add_argument(
        "--page",
        help="Zero-based page index (default: 0); ignored when --all is set",
        type=int, required=False, default=0,
    )
    arg_parse.add_argument(
        "--all",
        help="Iterate over all station pages (heatmap/variation only)",
        action="store_true", default=False,
        dest="all_pages",
    )
    arg_parse.add_argument(
        "--variation",
        help="Variation type for variation mode: Diurnal | Seasonal (default: Diurnal)",
        type=str, required=False, default="Diurnal",
        choices=["Diurnal", "Seasonal"],
    )

    # ---- statistics overlays ------------------------------------------------
    arg_parse.add_argument(
        "--mean", help="Overlay mean curve on heatmap panels",
        action="store_true", default=False,
    )
    arg_parse.add_argument(
        "--show_mode", help="Overlay mode curve on heatmap panels",
        action="store_true", default=False, dest="show_mode",
    )
    arg_parse.add_argument(
        "--nhnm", help="Overlay NHNM reference curve",
        action="store_true", default=False,
    )
    arg_parse.add_argument(
        "--nlnm", help="Overlay NLNM reference curve",
        action="store_true", default=False,
    )
    arg_parse.add_argument(
        "--earthquakes", help="Overlay earthquake model lines",
        action="store_true", default=False,
    )
    arg_parse.add_argument(
        "--min_mag", help="Minimum magnitude for earthquake models (default: 0.0)",
        type=float, required=False, default=0.0,
    )
    arg_parse.add_argument(
        "--max_mag", help="Maximum magnitude for earthquake models (default: 10.0)",
        type=float, required=False, default=10.0,
    )
    arg_parse.add_argument(
        "--min_dist", help="Minimum distance (km) for earthquake models (default: 0.0)",
        type=float, required=False, default=0.0,
    )
    arg_parse.add_argument(
        "--max_dist", help="Maximum distance (km) for earthquake models (default: 10000)",
        type=float, required=False, default=10000.0,
    )

    # ---- comparison-only options --------------------------------------------
    arg_parse.add_argument(
        "--stats",
        help="Statistics for comparison mode, comma-separated: mean,median,mode "
             "(default: mean)",
        type=str, required=False, default="mean",
    )
    arg_parse.add_argument(
        "--layout",
        help="Panel layout for comparison mode: by_component | single "
             "(default: by_component)",
        type=str, required=False, default="by_component",
        choices=["by_component", "single"],
    )

    parsed_args = arg_parse.parse_args()
    print("Input Arguments")
    print(parsed_args)

    db_file = make_abs(parsed_args.db_file)

    # ---- build station/channel selection dict for heatmap/variation ---------
    # For heatmap and variation we reuse the existing selection=None path
    # (everything) and rely on net/sta/chn filters only for comparison.
    # If the user specified filters for heatmap we warn them to use comparison.

    mode = parsed_args.mode

    # =========================================================================
    if mode in ("heatmap", "variation"):

        plot_mode = "pdf" if mode == "heatmap" else "variation"

        common_kwargs = dict(
            pickle_path=db_file,
            stations_per_page=parsed_args.stations_per_page,
            plot_mode=plot_mode,
            variation=parsed_args.variation,
            starttime=parsed_args.starttime,
            endtime=parsed_args.endtime,
            net_pattern=parsed_args.net,
            sta_pattern=parsed_args.station,
            chn_pattern=parsed_args.channel,
            show_mean=parsed_args.mean,
            show_mode=parsed_args.show_mode,
            show_nhnm=parsed_args.nhnm,
            show_nlnm=parsed_args.nlnm,
            show_earthquakes=parsed_args.earthquakes,
            min_mag=parsed_args.min_mag,
            max_mag=parsed_args.max_mag,
            min_dist=parsed_args.min_dist,
            max_dist=parsed_args.max_dist,
            show=parsed_args.save_dir is None and parsed_args.save_path is None,
        )

        if parsed_args.all_pages:
            plot_all_pages(
                **common_kwargs,
                save_dir=make_abs(parsed_args.save_dir) if parsed_args.save_dir else None,
                save_prefix=parsed_args.prefix,
                save_format=parsed_args.save_format,
            )
        else:
            save_path = make_abs(parsed_args.save_path) if parsed_args.save_path else None
            plot_ppsds_from_pickle(
                **common_kwargs,
                page=parsed_args.page,
                save_path=save_path,
            )

    # =========================================================================
    elif mode == "comparison":

        stats_list = [s.strip() for s in parsed_args.stats.split(",") if s.strip()]
        save_path = make_abs(parsed_args.save_path) if parsed_args.save_path else None

        plot_comparison(
            pickle_path=db_file,
            sta_pattern=parsed_args.station,
            chn_pattern=parsed_args.channel,
            net_pattern=parsed_args.net,
            stats=tuple(stats_list),
            starttime=parsed_args.starttime,
            endtime=parsed_args.endtime,
            show_nhnm=parsed_args.nhnm,
            show_nlnm=parsed_args.nlnm,
            layout=parsed_args.layout,
            show=save_path is None,
            save_path=save_path,
        )


if __name__ == "__main__":
    freeze_support()
    main()
