# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: metadata_manager.py
# Program: surfQuake & ISP
# Date: January 2024
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
from dateutil import parser
from multiprocessing import freeze_support
from typing import Optional
from surfquakecore.arrayanalysis.beamrun import TraceBeamResult
from surfquakecore.data_processing.analysis_events import AnalysisEvents
from surfquakecore.data_processing.processing_methods import print_surfquake_trace_headers
from surfquakecore.earthquake_location.run_nll import NllManager, Nllcatalog
from surfquakecore.first_polarity.first_polarity import FirstPolarity
from surfquakecore.first_polarity.get_pol import RunPolarity
from surfquakecore.magnitudes.run_magnitudes import Automag
from surfquakecore.magnitudes.source_tools import ReadSource
from surfquakecore.moment_tensor.mti_parse import WriteMTI, BuildMTIConfigs
from surfquakecore.moment_tensor.sq_isola_tools import BayesianIsolaCore
from surfquakecore.project.surf_project import SurfProject
from surfquakecore.real.real_core import RealCore
from surfquakecore.seismoplot.availability import PlotExplore
from surfquakecore.spectral.cwtrun import TraceCWTResult
from surfquakecore.spectral.specrun import TraceSpectrumResult, TraceSpectrogramResult
from surfquakecore.utils.create_station_xml import Convert
from surfquakecore.utils.manage_catalog import BuildCatalog, WriteCatalog
from datetime import datetime, timedelta
from surfquakecore.coincidence_trigger.coincidence_trigger import CoincidenceTrigger

# should be equal to [project.scripts]
__entry_point_name = "surfquake"
web_tutorial_address = "https://projectisp.gprintithub.io/surfquaketutorial.github.io/"


@dataclass
class _CliActions:
    name: str
    run: callable
    description: str = ""


def _create_actions():
    _actions = {
        "project": _CliActions(
            name="project", run=_project, description=f"Type {__entry_point_name} -h for help.\n"),

        "pick": _CliActions(
            name="pick", run=_pick, description=f"Type {__entry_point_name} -h for help.\n"),

        "associate": _CliActions(
            name="associate", run=_associate, description=f"Type {__entry_point_name} -h for help.\n"),

        "trigg": _CliActions(
            name="trigg", run=_trigg, description=f"Type {__entry_point_name} -h for help.\n"),

        "locate": _CliActions(
            name="locate", run=_locate, description=f"Type {__entry_point_name} -h for help.\n"),

        "source": _CliActions(
            name="source", run=_source, description=f"Type {__entry_point_name} -h for help.\n"),

        "polarity": _CliActions(
            name="polarity", run=_polarity, description=f"Type {__entry_point_name} -h for help.\n"),

        "focmec": _CliActions(
            name="focmec", run=_focmec, description=f"Type {__entry_point_name} -h for help.\n"),

        "plotmec": _CliActions(
            name="plotmec", run=_plotmec, description=f"Type {__entry_point_name} -h for help.\n"),

        "mti": _CliActions(
            name="mi", run=_mti, description=f"Type {__entry_point_name} -h for help.\n"),

        "csv2xml": _CliActions(
            name="csv2xml", run=_csv2xml, description=f"Type {__entry_point_name} -h for help.\n"),

        "buildcatalog": _CliActions(
            name="buildcatalog", run=_buildcatalog, description=f"Type {__entry_point_name} -h for help.\n"),

        "buildmticonfig": _CliActions(
            name="buildmticonfig", run=_buildmticonfig, description=f"Type {__entry_point_name} -h for help.\n"),

        "processing": _CliActions(
            name="processing", run=_processing, description=f"Type {__entry_point_name} -h for help.\n"),

        "processing_daily": _CliActions(
            name="processing_daily", run=_processing_daily, description=f"Type {__entry_point_name} -h for help.\n"),

        "quick": _CliActions(
            name="processing_quick", run=_quickproc, description=f"Type {__entry_point_name} -h for help.\n"),

        "specplot": _CliActions(
            name="specplot", run=_specplot, description=f"Type {__entry_point_name} -h for help.\n"),

        "beamplot": _CliActions(
            name="beamplot", run=_beamplot, description=f"Type {__entry_point_name} -h for help.\n"),

        "info": _CliActions(
            name="info", run=_info, description=f"Type {__entry_point_name} -h for help.\n"),

        "explore": _CliActions(
            name="explore", run=_explore, description=f"Type {__entry_point_name} -h for help.\n")

    }

    return _actions


def main(argv: Optional[str] = None):
    # actions must be always the first arguments after the command surfquake
    try:
        input_action = sys.argv.pop(1)
    except IndexError:
        input_action = ""

    actions = _create_actions()

    if action := actions.get(input_action, None):
        action.run()
    else:
        print(f"- Possible surfquake commands are: {', '.join(actions.keys())}\n"
              f"- Command documentation typing: surfquake command -h ")


def _project():
    """
    Command-line interface for creating a seismic project.
    """

    arg_parse = ArgumentParser(
        prog=f"{__entry_point_name} project",
        description="Create a seismic project by indexing seismogram files and storing their metadata.",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
    
    Overview:
        This command creates a seismic project — a file-based database that stores paths
        to waveform files and their associated metadata. Projects simplify later processing.

    Usage Example:
        surfquake project \\
            -d ./data_directory \\
            -s ./projects \\
            -n my_project \\
            --verbose

    Key Arguments:
        -d, --data           Path to waveform data directory (or file pattern)
        -s, --save_path      Directory to save the project file
        -n, --name           Name of the project (e.g., "my_experiment")
        --verbose            Print detailed file discovery and indexing logs

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
        surfquake pick \\
            -f ./my_project.json \\
            -d ./picks_output \\
            -p 0.3 \\
            -s 0.3 \\
            --verbose

    Key Arguments:
        -f, --project_file        Path to your seismic project file
        -d, --output_dir          Directory to save pick results
        -pt, --p_thresh            [OPTIONAL] Threshold for P-wave probability (0–1) (default: 0.3)
        -st, --s_thresh            [OPTIONAL] Threshold for S-wave probability (0–1) (default: 0.3)
        -n, --net                 [OPTIONAL] Network code filter
        -s, --station             [OPTIONAL] Station code filter
        -ch, --channel            [OPTIONAL] Channel filter
        --min_date                [OPTIONAL] Filter Start date (format: YYYY-MM-DD HH:MM:SS), DEFAULT min date of the project
        --max_date                [OPTIONAL] Filter End date   (format: YYYY-MM-DD HH:MM:SS), DEFAULT max date of the project
        --verbose                 Enable detailed logging

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

    parser.add_argument("-d", "--hyp_folder", required=True, help="path to folder containing hyp files",
                        type=str)
    parser.add_argument("-a", "--accepted", required=False, help="Number of accepted wrong polarities",
                        type=float, default=1.0)
    parser.add_argument("-o", "--output_folder", required=True, help="output folder", type=str)


    args = parser.parse_args()

    hyp_folder = make_abs(args.hyp_folder)
    output_folder = make_abs(args.output_folder)
    files_list = FirstPolarity.find_hyp_files(hyp_folder)
    for file in files_list:
        try:
            header = FirstPolarity.set_head(file)
            if file is not None:
                file_input = FirstPolarity().create_input(file, header)

                if FirstPolarity.check_no_empty(file_input):
                    FirstPolarity().run_focmec(file_input, 3, output_folder)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            traceback.print_exc()

def _plotmec():
    parser = ArgumentParser(
        prog="surfquake plotmec",
        description="Focal Mechanism from P-wave polarity ",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
        Overview:
        
        Plot fault planes, T & P axis, and P-wave polarities.
        
        
        Key Arguments:
        -f, --focmec_file           [OPTIONAL] Path to a specific *.lst file
        -d, --focmec_folder_path    [OPTIONAL] Path to folder with all *.lst files (focmec output)
        -o, --output_folder         [REQUIRED] Path to the output folder
        -a, --all_solutions         [OPTIONAL] If set, all searching fault planes will be plot
        -p, --plot_polarities       [OPTIONAL] If plot P-Wave polarities on the beachball
        -m, --format                [OPTIONAL] Format output plot (defaults pdf)
        
        Example usage:

        surfquake plotmec -d ./focmec_folder_path -o ./output_folder
        surfquake plotmec -f ./focmec_file_path.lst -o ./output_folder -p -a -m pdf
            
        if output is not provided the beachball of the focal mechanism will be shown on screen, 
        but if user provide output folder, the beach ball plot will be saved in the folder
        
            """
    )

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
        surfquake associator \\
            -i inventory.xml \\
            -p ./picks_folder \\
            -c real_config.ini \\
            -w ./working_dir \\
            -s ./associated_output \\
            --verbose

    Key Arguments:
        -i, --inventory_file     Path to station metadata (XML or RESP)
        -p, --picks_folder       Folder containing station pick files
        -c, --config_file        Path to REAL .ini configuration file
        -w, --work_dir           Working directory for REAL intermediate output
        -s, --save_dir           Output directory for associated pick results
        --verbose                Enable detailed logging

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
        surfquake locate \\
            -i inventory.xml \\
            -c locate_config.ini \\
            -o ./output_locations \\
            -g \\
            -s \\
            -n 10

    Key Arguments:
        -i, --inventory_file      Station metadata (XML, RESP)
        -c, --config_file         Path to NonLinLoc .ini configuration
        -o, --output_dir          Directory where location results will be saved
        -g, --generate_tt         Generate travel time files before location
        -s, --apply_station_corr  Apply station corrections (if configured)
        -n, --iterations          Number of global search iterations (int)

    Reference:
        Lomax, A., Michelini, A., Curtis, A. (2009).
        Earthquake Location: Direct, Global-Search Methods.
        Encyclopedia of Complexity and System Science, Springer.
        DOI: https://doi.org/10.1007/978-0-387-30440-3

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
                                                                "stations corrections, default iterations 10",
                           action="store_true")

    arg_parse.add_argument('-n', '--number_iterations', type=int, metavar='N', help='an integer for the '
                                                                                    'number of iterations',
                           required=False)

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

    arg_parse = ArgumentParser(
        prog=f"{__entry_point_name} source parameters estimation",
        description="Estimate source parameters using P- and S-wave spectra.",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
    Overview:
        surfQuake estimates source parameters such as:
            • Stress Drop
            • Attenuation (Q)
            • Source radius
            • Radiated energy
            • Local (ML) and moment (Mw) magnitudes

        It uses spectral fitting of P- and/or S-wave displacement spectra, following SourceSpec methodology.

    Usage Example:
        surfquake source \\
            -i inventory.xml \\
            -p my_project.json \\
            -c source_config.yaml \\
            -l ./nlloc_outputs \\
            -o ./source_results

    Key Arguments:
        -i, --inventory_file     Path to station metadata (XML, RESP)
        -p, --project_file       Path to project file containing waveform picks
        -c, --config_file        YAML configuration for SourceSpec
        -l, --hypocenter_dir     Directory containing NonLinLoc .hyp files
        -o, --output_folder      Output folder for source parameter results

    Reference:
        Satriano, C. (2023). SourceSpec – Earthquake source parameters from
        P- or S-wave displacement spectra. DOI: https://doi.org/10.5281/ZENODO.3688587

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

    arg_parse.add_argument("-t", "--large_scale",
                           help="If you want a long cut of signals for teleseism events (optional)",
                           action="store_true")

    arg_parse.add_argument("-o", "--output_dir_path", help="Path to output_directory ", type=str,
                           required=True)

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

    arg_parse = ArgumentParser(
        prog=f"{__entry_point_name} Moment Tensor Inversion",
        description="Estimate seismic moment tensors using Bayesian inversion.",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
    Overview:
        surfQuake provides a simple interface to estimate moment tensors for pre-located earthquakes
        using a Bayesian inversion approach, based on the Bayesian ISOLA method.

    Usage Example:
        surfquake mti \\
            -i inventory.xml \\
            -p my_project.json \\
            -c mti_config.ini \\
            -o ./mti_output \\
            -s

    Key Arguments:
        -i, --inventory_file     Path to station metadata (XML, RESP)
        -p, --project_file       Path to the project file with waveforms and event metadata
        -c, --config_file        INI configuration file for inversion settings
        -o, --output_dir         Output directory for inversion results
        -s, --save_plots         If set, saves plots of MT solutions and fits

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
        surfquake csv2xml \\
            -c ./stations.csv \\
            -r ./resp_files/ \\
            -o ./output_dir \\
            -n my_stations.xml

    Key Arguments:
        -c, --csv_file         Path to input CSV file containing station metadata
        -r, --resp_dir         Path to RESP files for each station (optional if included in CSV)
        -o, --output_dir       Directory where the StationXML will be saved
        -n, --output_name      Desired filename for the output StationXML

    Documentation:
        https://projectisp.github.io/surfquaketutorial.github.io/
    """
    )

    arg_parse.add_argument("-c", "--csv_file_path", help="file containing Net Station Lat Lon elevation "
        "start_date starttime end_date endtime, single spacing", type=str, required=True)

    arg_parse.add_argument("-r", "--resp_files_path", help="Path to the folder containing the response file",
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
        https://docs.obspy.org/packages/autogen/obspy.core.event.Catalog.write.html
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
        surfquake buildmticonfig \\
            -c ./catalog.xml \\
            -t ./mti_template.ini \\
            -o ./generated_configs \\
            -s "01/01/2024, 00:00:00.000" \\
            -e "01/02/2024, 00:00:00.000" \\
            -l 34.0 -a 36.0 \\
            -d -118.0 -k -116.0 \\
            -w 0 -f 20 \\
            -g 2.5 -p 6.0

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
        -e, --event_file           [OPTIONAL] Event catalog in QuakeML format
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
            surfquake trigg \\
                -c config.yaml \\
                -o ./output_folder \\
                -ch "HHZ"
                --min_date "2024-01-01 00:00:00" \\
                --max_date "2024-01-04 00:00:00" \\
                --span_seconds  86400\\
                --picking_file ./pick.txt
                --plot
                
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
        surfquake processing_daily \\
            -i inventory.xml \\
            -c config.yaml \\
            -o ./output_folder \\
            --min_date "2024-01-01 00:00:00" \\
            --max_date "2024-01-02 00:00:00" \\
            --span_seconds  86400\\
            --plot_config plot_settings.yaml \\
            --post_script my_custom.py \\
            --post_script_stage after

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
                surfquake quick \\
                    -w "./data/*.mseed" \\
                    -c ./config.yaml \\
                    -i ./inventory.xml \\
                    -o ./out \\
                    --plot_config plot.yaml

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
    parser = ArgumentParser(
        prog="surfquake specplot",
        description="Plot serialized spectral analysis (spectrum, spectrogram or cwt)",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
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

    parser = ArgumentParser(
        prog="surfquake beamplot",
        description="Plot serialized beamforming result and optionally extract peaks",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Plot a saved FK beam:
      surfquake beamplot --file ./output/2024.123.beam

  Detect peaks with regional phase constraints:
      surfquake beamplot --file event.beam --find_solutions regional --write_path solutions.txt

  Use custom slowness constraints:
      surfquake beamplot --file beam.beam --find_solutions '{"Pn": [0.05, 0.08], "Lg": [0.18, 0.30]}'

  Apply azimuth filtering:
      surfquake beamplot -f beam.beam --find_solutions regional --baz_range 100 150

  Control minimum power threshold:
      surfquake beamplot -f beam.beam --find_solutions teleseismic --min_power 0.2
"""
    )

    parser.add_argument("--file", "-f", required=True, help="Path to the .beam file (gzip-pickled TraceBeamResult)")
    parser.add_argument("--save_path", help="Optional path to save the beam figure (e.g., output.png)")
    parser.add_argument("--find_solutions", help="Phase constraint: 'regional', 'teleseismic', or dict string")
    parser.add_argument("--baz_range", nargs=2, type=float, metavar=('MIN', 'MAX'),
                        help="Backazimuth range filter in degrees (e.g., 90 140)")
    parser.add_argument("--min_power", type=float, default=0.6,
                        help="Minimum relative power required to accept a peak (default: 0.1)")
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

            if len(results)>0:
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

    surfquake info -w './data/*.mseed' -c 3
    surfquake info -w 'trace1.mseed,trace2.mseed' --columns 5

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


def resolve_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(os.getcwd(), path))


def make_abs(path: Optional[str]) -> Optional[str]:
    return os.path.abspath(path) if path else None

def parse_datetime(dt_str: str) -> datetime:
    # try with microseconds, fall back if not present
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Date string not in expected format: {dt_str}")

if __name__ == "__main__":
    freeze_support()
    main()
