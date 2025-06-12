# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: metadata_manager.py
# Program: surfQuake & ISP
# Date: January 2024
# Purpose: Command Line Interface Core
# Author: Roberto Cabieces & Thiago C. Junqueira
#  Email: rcabdia@roa.es
# --------------------------------------------------------------------

import os
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from dateutil import parser
from multiprocessing import freeze_support
from typing import Optional
from surfquakecore.data_processing.analysis_events import AnalysisEvents
from surfquakecore.earthquake_location.run_nll import NllManager, Nllcatalog
from surfquakecore.magnitudes.run_magnitudes import Automag
from surfquakecore.magnitudes.source_tools import ReadSource
from surfquakecore.moment_tensor.mti_parse import WriteMTI, BuildMTIConfigs
from surfquakecore.moment_tensor.sq_isola_tools import BayesianIsolaCore
from surfquakecore.project.surf_project import SurfProject
from surfquakecore.real.real_core import RealCore
from surfquakecore.utils.create_station_xml import Convert
from surfquakecore.utils.manage_catalog import BuildCatalog, WriteCatalog

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

        "locate": _CliActions(
            name="locate", run=_locate, description=f"Type {__entry_point_name} -h for help.\n"),

        "source": _CliActions(
            name="source", run=_source, description=f"Type {__entry_point_name} -h for help.\n"),

        "mti": _CliActions(
            name="mi", run=_mti, description=f"Type {__entry_point_name} -h for help.\n"),

        "csv2xml": _CliActions(
            name="csv2xml", run=_csv2xml, description=f"Type {__entry_point_name} -h for help.\n"),

        "buildcatalog": _CliActions(
            name="buildcatalog", run=_buildcatalog, description=f"Type {__entry_point_name} -h for help.\n"),

        "buildmticonfig": _CliActions(
            name="buildmticonfig", run=_buildmticonfig, description=f"Type {__entry_point_name} -h for help.\n"),

        "processing_cut": _CliActions(
            name="processing_cut", run=_processing_cut, description=f"Type {__entry_point_name} -h for help.\n"),

        "processing_daily": _CliActions(
            name="processing_daily", run=_processing_cut, description=f"Type {__entry_point_name} -h for help.\n")
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

    arg_parse = ArgumentParser(prog=f"{__entry_point_name} project", description="Create a seismic project by storing "
                                                                                 "the paths to seismogram files and "
                                                                                 "their metadata.")

    arg_parse.epilog = """
    Overview:
      This command allows you to create a seismic project, which is essentially a dictionary
      storing the paths to seismogram files along with their corresponding metadata.
    
    Usage:
      surfquake project -d [path to data files] -s [path to save directory] -n [project name] --verbose
    
    Documentation:
      https://projectisp.github.io/surfquaketutorial.github.io/
    """

    arg_parse.add_argument("-d", "--data_dir", help="Path to data files directory", type=str, required=True)

    arg_parse.add_argument("-s", "--save_dir", help="Path to directory where project will be saved", type=str,
                           required=True)

    arg_parse.add_argument("-n", "--project_name", help="Project Name", type=str, required=True)

    arg_parse.add_argument("-v", "--verbose", help="information of files included on the project",
                           action="store_true")

    parsed_args = arg_parse.parse_args()
    print(parsed_args)

    print(f"Project from {parsed_args.data_dir} saving to {parsed_args.save_dir} as {parsed_args.project_name}")
    sp = SurfProject(parsed_args.data_dir)
    project_file_path = os.path.join(parsed_args.save_dir, parsed_args.project_name)
    sp.search_files(verbose=parsed_args.verbose)
    print(sp)
    sp.save_project(path_file_to_storage=project_file_path)


def _pick():
    from surfquakecore.phasenet.phasenet_handler import PhasenetISP, PhasenetUtils

    arg_parse = ArgumentParser(prog=f"{__entry_point_name} pick", description="Use Phasenet Neural Network to estimate "
                                                                              "body waves arrival times")
    arg_parse.epilog = """
            Overview:
              The Picking algorythm uses the Deep Neural Network of Phasenet to estimate 
              the arrival times of P- and S-wave in regional and local seismic events.

            Usage:
              surfquake pick -f [path to your project file] -d [path to your pick saving directory] -p 
              [P-wave threshoold] -s [S-wave threshold] --verbose"

            Reference:              
              Zhu and Beroza, 2019. PhaseNet: a deep-neural-network-based seismic arrival-time picking method, 
              Geophysical Journal International.

            Documentation:
              https://projectisp.github.io/surfquaketutorial.github.io/
            """

    arg_parse.usage = ("Run picker: -f [path to your project file] "
                       "-d [path to your pick saving directory] -p [P-wave threshoold] -s [S-wave threshold] --verbose")

    arg_parse.add_argument("-f", help="path to your project file", type=str, required=True)

    arg_parse.add_argument("-d", help="Path to directory where picks will be saved", type=str,
                           required=True)

    arg_parse.add_argument("-p", help="P-wave threshoold", type=float,
                           required=True)

    arg_parse.add_argument("-s", help="S-wave threshold", type=float,
                           required=True)

    arg_parse.add_argument("-v", "--verbose", help="information of files included on the project",
                           action="store_true")

    parsed_args = arg_parse.parse_args()
    print(parsed_args)

    sp_loaded = SurfProject.load_project(path_to_project_file=parsed_args.f)
    if len(sp_loaded.project) > 0 and isinstance(sp_loaded, SurfProject):

        picker = PhasenetISP(sp_loaded.project, amplitude=True, min_p_prob=parsed_args.p,
                             min_s_prob=parsed_args.s, output=parsed_args.d)

        # Running Stage
        picks = picker.phasenet()
        #
        """ PHASENET OUTPUT TO REAL INPUT """
        #
        picks_results = PhasenetUtils.split_picks(picks)
        PhasenetUtils.convert2real(picks_results, parsed_args.d)
        PhasenetUtils.save_original_picks(picks_results, parsed_args.d)
        PhasenetUtils.write_nlloc_format(picks_results, parsed_args.d)
    else:
        print("Empty Project, Nothing to pick!")


def _associate():
    arg_parse = ArgumentParser(prog=f"{__entry_point_name} associator", description="Use Associator to group correctly "
                                                                                    "phase picks to unique seismic "
                                                                                    "events")
    arg_parse.epilog = """
        Overview:
          You can correlate picks with the corresponding unique seismic events by using this command. 
          The association was performed using REAL algorithm. 

        Usage: surfquake associate -i [inventory_file_path] -p [path to data picking folder] -c [path to 
        real_config_file.ini] -w [work directory] -s [path to directory where associates picks will be saved] --verbose
          
        Reference: Zhang et al. 2019, Rapid Earthquake Association and Location, Seismol. Res. Lett. 
        https://doi.org/10.1785/0220190052
            
        Documentation:
          https://projectisp.github.io/surfquaketutorial.github.io/
          # Time file is based on https://github.com/Dal-mzhang/LOC-FLOW/blob/main/LOCFLOW-CookBook.pdf
          # reference for structs: https://github.com/Dal-mzhang/REAL/blob/master/REAL_userguide_July2021.pdf
        """

    arg_parse.add_argument("-i", "--inventory_file_path", help="Inventory file (i.e., *xml or dataless",
                           type=str,
                           required=True)

    arg_parse.add_argument("-p", "--data-dir", help="path to data picking folder",
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

    rc = RealCore(parsed_args.inventory_file_path, parsed_args.config_file_path, parsed_args.data_dir,
                  parsed_args.work_dir_path, parsed_args.save_dir)
    rc.run_real()

    print("End of Events AssociationProcess, please see for results: ", parsed_args.save_dir)


def _locate():
    arg_parse = ArgumentParser(prog=f"{__entry_point_name} locate seismic event", description=" Locate seismic event")
    arg_parse.epilog = """
        
        Overview:
          surfQuake uses a non-linear approach (NonLinLoc) to locate a seismic event. 
          Inputs are the pick file in NonLinLoc format, and the time folder with the traveltimes generated in 
          pre-locate subprogram.
          Further details can be found in formats section http://alomax.free.fr/nlloc/:
            
        Usage: surfquake locate -i [inventory_file_path] -c [config_file_path] -o [path_to output_path] 
        -g [if travel_time_generation needed] -s [if stations_corrections need] -n 
        [If you want to iterate number of iterations]

        Reference: Lomax, A., A. Michelini, A. Curtis, 2009. Earthquake Location, Direct, Global-Search Methods, in 
        Complexity In Encyclopedia of Complexity and System Science, Part 5, Springer, New York, pp. 2449-2473, 
        doi:10.1007/978-0-387-30440-3.

        Documentation:
          https://projectisp.github.io/surfquaketutorial.github.io/
          Complete description of input files http://alomax.free.fr/nlloc/
        """

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

    nll_manager = NllManager(parsed_args.config_file_path, parsed_args.inventory_file_path, parsed_args.out_dir_path)

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
            num_iter = 10
        # including stations_corrections
        for i in range(1, (num_iter + 1)):
            nll_manager.run_nlloc(num_iter=i)
    else:
        nll_manager.run_nlloc()
    print("Finished Locations see output at, ", os.path.join(parsed_args.out_dir_path, "loc"))
    nll_catalog = Nllcatalog(parsed_args.out_dir_path)
    nll_catalog.run_catalog(parsed_args.out_dir_path)
    print("Catalog done, finished process see catalog at ", parsed_args.out_dir_path)


def _source():
    arg_parse = ArgumentParser(prog=f"{__entry_point_name} source parameters estimation",
                               description="source parameters estimation")

    arg_parse.epilog = """

    Overview:
      surfQuake uses the spectra P- and S-waves to estimate source parameters (Stress Drop, attenuation, source radius 
       radiated energy) and magnitudes ML and Mw.

    Usage: surfquake source -i [inventory file path] - p [path to project file] -c [path to 
      source_config_file] -l [path to nll hyp files] -o [path to output folder]

    Reference: Satriano, C. (2023). SourceSpec – Earthquake source parameters from P- or S-wave 
    displacement spectra (X.Y). doi: 10.5281/ZENODO.3688587.

    Documentation:
      https://projectisp.github.io/surfquaketutorial.github.io/
      https://sourcespec.readthedocs.io/en/stable/index.html
    """

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
    sp_loaded = SurfProject.load_project(path_to_project_file=parsed_args.project_file_path)
    print(sp_loaded)

    # Running stage
    summary_path_file = os.path.join(parsed_args.output_dir_path, "source_summary.txt")

    if parsed_args.large_scale:
        scale = "teleseism"
    else:
        scale = "regional"

    mg = Automag(sp_loaded, parsed_args.loc_files_path, parsed_args.inventory_file_path, parsed_args.config_file_path,
                 parsed_args.output_dir_path, scale=scale)

    mg.estimate_source_parameters()

    rs = ReadSource(parsed_args.output_dir_path)
    summary = rs.generate_source_summary()
    rs.write_summary(summary, summary_path_file)


def _mti():
    arg_parse = ArgumentParser(prog=f"{__entry_point_name} Moment Tensor Inversion",
                               description="Moment Tensor Inversion")

    arg_parse.epilog = """

        Overview:
          surfQuake provides an easy way to estimate the Moment Tensor from pre-located earthquakes using a 
          bayesian inversion.

        Usage: surfquake mti -i [inventory_file_path] -p [path_to_project] -c [path to mti_config_file.ini] 
        -o [output_path]  -s [if save plots]

        Reference: Vackář, J., Burjánek, J., Gallovič, F., Zahradník, J., & Clinton, J. (2017). Bayesian ISOLA: 
        new tool for automated centroid moment tensor inversion. Geophysical Journal International, 210(2), 693-705.

        Documentation:
          https://projectisp.github.io/surfquaketutorial.github.io/
        """

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

    sp = SurfProject.load_project(path_to_project_file=parsed_args.path_to_project_file)
    print(sp)

    bic = BayesianIsolaCore(
        project=sp,
        inventory_file=parsed_args.inventory_file_path,
        output_directory=parsed_args.output_dir_path,
        save_plots=parsed_args.save_plots,
    )

    print("Starting Inversion")
    bic.run_inversion(mti_config=parsed_args.config_files_path)

    print("Writing Summary")
    wm = WriteMTI(parsed_args.output_dir_path)
    wm.mti_summary()
    print("End of process, please review output directory")


def _csv2xml():
    arg_parse = ArgumentParser(prog=f"{__entry_point_name} Convert csv file to stations.xml",
                               description="Convert csv file to stations.xml")

    arg_parse.epilog = """

            Overview:
              Convert csv file to stations.xml: 
              Net Station Lat Lon elevation start_date starttime end_date endtime
              date format = '%Y-%m-%d %H:%M:%S'
              
            Usage: surfquake csv2xml -c [csv_file_path] -r [resp_files_path] -o [output_path] -n [stations_xml_name]

            Documentation:
              https://projectisp.github.io/surfquaketutorial.github.io/
            """

    arg_parse.add_argument("-c", "--csv_file_path", help="Net Station Lat Lon elevation "
                                                         "start_date starttime end_date endtime", type=str,
                           required=True)

    arg_parse.add_argument("-r", "--resp_files_path", help="Path to the folder containing the response file",
                           type=str, required=False)

    arg_parse.add_argument("-o", "--output_path", help="Path to output xml file)", type=str, required=True)

    arg_parse.add_argument("-n", "--stations_xml_name", help="Name of the xml file to be saved", type=str,
                           required=True)

    parsed_args = arg_parse.parse_args()
    print(parsed_args)

    if parsed_args.resp_files_path:
        sc = Convert(parsed_args.csv_file_path, resp_files=parsed_args.resp_files_path)
    else:
        sc = Convert(parsed_args.csv_file_path)
    data_map = sc.create_stations_xml()
    inventory = sc.get_data_inventory(data_map)
    sc.write_xml(parsed_args.output_path, parsed_args.stations_xml_name, inventory)


def _buildcatalog():
    arg_parse = ArgumentParser(prog=f"{__entry_point_name} Convert csv file to stations.xml",
                               description="Convert csv file to stations.xml")

    arg_parse.epilog = """

            Overview:
              buildcatalog class helps to join information from all surfquake outputs and create a catalog
    
            Usage: surfquake buildcatalog -e [path_event_files_folder] -s [path_source_summary_file] -m 
            [path_mti_summary_file] -f [catalog_format] -o [path_to_output_folder]
    
            Documentation:
              https://projectisp.github.io/surfquaketutorial.github.io/utils/
              catalog formats info: https://docs.obspy.org/packages/autogen/obspy.core.event.Catalog.
              write.html#obspy.core.event.Catalog.write
            """

    arg_parse.add_argument("-e", "--path_event_files_folder", help="Net Station Lat Lon elevation "
                                                                   "start_date starttime end_date endtime", type=str,
                           required=True)

    arg_parse.add_argument("-s", "--path_source_summary_file", help='Path to the file containing '
                                                                    'the source spectrum results',
                           type=str, required=False, default=None)

    arg_parse.add_argument("-m", "--path_mti_summary_file", help="Path to the file containing the "
                                                                 "moment tensor results", type=str, required=False,
                           default=None)

    arg_parse.add_argument("-f", "--catalog_format", help="catalog format, default QUAKEML", type=str, required=False,
                           default="QUAKEML")

    arg_parse.add_argument("-o", "--path_to_output_folder", help="Path to the ouput folder, where catalog "
                                                                 "will be saved", type=str, required=True)

    parsed_args = arg_parse.parse_args()
    print(parsed_args)

    if os.path.isdir(parsed_args.path_to_output_folder):
        pass
    else:
        try:
            os.makedirs(parsed_args.path_to_output_folder)
        except Exception as error:
            print("An exception occurred:", error)

    catalog_path_pkl = os.path.join(parsed_args.path_to_output_folder, "catalog_obj.pkl")
    catalog_path_surf = os.path.join(parsed_args.path_to_output_folder, "catalog_surf.txt")

    bc = BuildCatalog(loc_folder=parsed_args.path_event_files_folder,
                      source_summary_file=parsed_args.path_source_summary_file,
                      output_path=parsed_args.path_to_output_folder, mti_summary_file=parsed_args.path_mti_summary_file,
                      format=parsed_args.catalog_format)

    bc.build_catalog_loc()
    wc = WriteCatalog(catalog_path_pkl)
    wc.write_catalog_surf(catalog=None, output_path=catalog_path_surf)


def _buildmticonfig():
    arg_parse = ArgumentParser(prog=f"{__entry_point_name} Creates mti_config.ini files from a catalog query and "
                                    f"a mti_config template",
                               description="mti_config files building command")

    arg_parse.epilog = """

        Overview:
          buildmticonfig can automatically create an mti_config.ini from a catalog query and a mti_config template.
        Usage: surfquake buildmticonfig -c [catalog_file_path] -t [mti_config_template] -o [output_folder] -s [if starttime] 
        -e [if endtime] -l [if lat_min] -a [ if lat_max] -d [if lon_min] -k [if lon_max] -w [if depth_min] 
        -f [depth_max] -g [if mag_min] -p [if mag_max]
        Warning, starttime and endtime format is format %d/%m/%Y, %H:%M:%S.%f
        Documentation:
          https://projectisp.github.io/surfquaketutorial.github.io/utils/
        """

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

    if os.path.isdir(parsed_args.output_folder):
        pass
    else:
        try:
            os.makedirs(parsed_args.output_folder)
        except Exception as error:
            print("An exception occurred:", error)

    print("Querying Catalog --> ", parsed_args.lat_min, parsed_args.lat_max, parsed_args.lon_min, parsed_args.lon_max,
          parsed_args.depth_min, parsed_args.depth_max, parsed_args.mag_min, parsed_args.mag_max)

    bmc = BuildMTIConfigs(catalog_file_path=parsed_args.catalog_file_path, mti_config=parsed_args.mti_config_template,
                          output_path=parsed_args.output_folder)

    bmc.write_mti_ini_file(starttime=parsed_args.starttime, endtime=parsed_args.endtime,
                           lat_min=float(parsed_args.lat_min),
                           lat_max=float(parsed_args.lat_max), lon_min=float(parsed_args.lon_min),
                           lon_max=float(parsed_args.lon_max),
                           depth_min=float(parsed_args.depth_min), depth_max=float(parsed_args.depth_max),
                           mag_min=float(parsed_args.mag_min),
                           mag_max=float(parsed_args.mag_max))



def _processing_cut():

    arg_parse = ArgumentParser(prog=f"{__entry_point_name} processing waveforms associated to events. ",
                               description="Processing waveforms associated to events command")

    arg_parse.epilog = """
            Overview:
                Cut seismograms associated to events and apply processing to the waveforms. You can perform either or both of these operations
                Usage: surfquake processing -p [project_file] -o [output_folder] -i [inventory_file] -c [config_file]
                -e [event_file] -n [net] -s [station] -ch [channel] -cs [cut_start_time]
                -ce [cut_end_time] -t [cut_time] -l [if interactive plot seismograms] 
                --plot_config [Path to optional plotting configuration file (.yaml) 
                --post_script [Path to Python script to apply to each event stream]
            """

    arg_parse.add_argument("-p", "--project_file", help="absolute path to project file", type=str,
                           required=True)

    arg_parse.add_argument("-o", "--output_folder", help="absolute path to output folder. Files are saved here",
                           type=str, required=False)

    arg_parse.add_argument("-i", "--inventory_file", help="metadata file. xml extension", type=str,
                           required=True)

    arg_parse.add_argument("-c", "--config_file", help="absolute path to config file", type=str,
                           required=False),

    arg_parse.add_argument("-e", "--event_file", help="absolute path to event file", type=str,
                           required=True)

    arg_parse.add_argument("-r", "--reference", help="Reference |event_time| if the first arrival "
                           "needs to be estimated else pick time is the reference, default event",
                           type=str, required=False)

    arg_parse.add_argument("-n", "--net", help="project net filter", type=str, required=False)

    arg_parse.add_argument("-s", "--station", help="project station filter", type=str, required=False)

    arg_parse.add_argument("-ch", "--channel", help="project channel filter", type=str, required=False)

    arg_parse.add_argument("-t", "--cut_time", help="pre & post first arrival in seconds (symmetric). ",
                           type=float, required=False)

    arg_parse.add_argument("-cs", "--cut_start_time", help="cut pre-first arrival  in seconds", type=float,
                           required=False)

    arg_parse.add_argument("-ce", "--cut_end_time", help="cut post-first arrival  in seconds", type=float,
                           required=False)

    arg_parse.add_argument("-l", "--plots", help=" In case user wants to plot seismograms",
                           action="store_true")

    arg_parse.add_argument("--plot_config", help="Path to optional plotting configuration file (.yaml)",
                           type=str)

    arg_parse.add_argument("--post_script", help="Path to Python script to apply to each event stream",
                           type=str)

    parsed_args = arg_parse.parse_args()

    # 1. Check if config or event files are not None
    if parsed_args.config_file is None and parsed_args.event_file is None:
        raise ValueError("Error: the command will do nothing. config_file and/or event_file are required")

    # Calculate start and end time
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

    sp = SurfProject.load_project(parsed_args.project_file)
    if len(filter) > 0:
        sp.filter_project_keys(**filter)

    sp_sub_projects = sp.split_by_time_spans(event_file=parsed_args.event_file, cut_start_time=start,
                                             cut_end_time=end, verbose=True)

    sd = AnalysisEvents(parsed_args.output_folder, parsed_args.inventory_file, parsed_args.config_file,
                        sp_sub_projects, post_script=parsed_args.post_script, plot_config_file=parsed_args.plot_config,
                        reference=parsed_args.reference)

    sd.run_waveform_cutting(cut_start=start, cut_end=end, plot=parsed_args.plots)

def _processing_daily():

    arg_parse = ArgumentParser(prog=f"{__entry_point_name} processing continous waveforms. ",
                               description="Processing continous waveforms command")
    arg_parse.epilog = """
                Overview:
                    Cut seismograms and apply processing to continous waveforms. You can perform either or both of these operations
                    Usage: surfquake processing -p [project_file] -o [output_folder] -i [inventory_file] -c [config_file]
                    -n [net] -s [station] -ch [channel] --min_date [Start time YYYY-MM-DD HH:MM:SS.sss] 
                    --max_date [End time YYYY-MM-DD HH:MM:SS.sss] -l [if interactive plot seismograms] 
                    --plot_config [Path to optional plotting configuration file (.yaml)]
                """
    arg_parse.add_argument("-p", "--project_file", required=True, help="Path to SurfProject .pkl")
    arg_parse.add_argument("-o", "--output_folder", help="Folder to save processed data")
    arg_parse.add_argument("-i", "--inventory_file", required=True, help="Station XML metadata")
    arg_parse.add_argument("-c", "--config_file", help="YAML config for processing")
    arg_parse.add_argument("-l", "--plot", action="store_true", help="Enable plotting")
    arg_parse.add_argument("--plot_config", help="YAML file for plot customization")
    arg_parse.add_argument("--span_seconds", type=int, default=86400, help="Time span to split subprojects (in seconds)")
    arg_parse.add_argument("--time_segment", action="store_true",
                           help="If set, process entire time window as a single merged stream")
    arg_parse.add_argument("--time_tolerance", type=int, default=120,
                           help="Tolerance in seconds for time filtering")
    # Filter arguments
    arg_parse.add_argument("-n", "--net", help="Network code filter", type=str)
    arg_parse.add_argument("-s", "--station", help="Station code filter", type=str)
    arg_parse.add_argument("-ch", "--channel", help="Channel code filter", type=str)
    arg_parse.add_argument("--min_date", help="Start time filter: format 'YYYY-MM-DD HH:MM:SS.sss'", type=str)
    arg_parse.add_argument("--max_date", help="End time filter: format 'YYYY-MM-DD HH:MM:SS.sss'", type=str)

    args = arg_parse.parse_args()

    # --- Load project ---
    sp = SurfProject.load_project(args.project_file)

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

    # --- Apply time filters ---
    min_date, max_date = None, None
    try:
        if args.min_date:
            #min_date = datetime.strptime(args.min_date, "%Y-%m-%d %H:%M:%S.%f")
            min_date = parser.parse(args.min_date)
        if args.max_date:
            #max_date = datetime.strptime(args.max_date, "%Y-%m-%d %H:%M:%S.%f")
            max_date = parser.parse(args.max_date)
        if min_date or max_date:
            print(f"[INFO] Filtering by time range: {min_date} to {max_date}")
            sp.filter_project_time(starttime=min_date, endtime=max_date, tol=args.time_tolerance, verbose=True)
    except ValueError as ve:
        print(f"[ERROR] Date format should be: 'YYYY-MM-DD HH:MM:SS.sss'")
        raise ve

    # --- Decide between time segment or split ---
    if args.time_segment:
        print(f"[INFO] Running single-segment analysis from {args.min_date} to {args.max_date}")
        subprojects = [sp]  # No splitting
    else:
        print(f"[INFO] Splitting into subprojects every {args.span_seconds} seconds")
        subprojects = sp.split_by_time_spans(
            span_seconds=args.span_seconds,
            min_date=args.min_date,
            max_date=args.max_date,
            file_selection_mode="overlap_threshold",
            verbose=True
        )

    # --- Run processing workflow ---
    ae = AnalysisEvents(
        output=args.output_folder,
        inventory_file=args.inventory_file,
        config_file=args.config_file,
        surf_projects=subprojects,
        plot_config_file=args.plot_config,
        time_segment_start=args.min_date,
        time_segment_end=args.max_date
    )
    ae.run_waveform_analysis(plot=args.plot)

if __name__ == "__main__":
    freeze_support()
    main()
