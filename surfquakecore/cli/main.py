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
from multiprocessing import freeze_support
from typing import Optional
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
web_tutorial_address = "https://projectisp.github.io/surfquaketutorial.github.io/"


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
            name="buildmticonfig", run=_buildmticonfig, description=f"Type {__entry_point_name} -h for help.\n")
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
        print(f"Invalid command {action}. Possible commands are: {', '.join(actions.keys())}\n"
              f"{''.join([f'- {ac.description}' for ac in actions.values()])}")


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

    print(f"Project from {parsed_args.data_dir} saving to {parsed_args.save_dir} as {parsed_args.project_name}")
    sp = SurfProject(parsed_args.data_dir)
    project_file_path = os.path.join(parsed_args.save_dir, parsed_args.project_name)
    sp.search_files(verbose=parsed_args.verbose)
    print(sp)
    print("End of project creation, number of files ", len(sp.project))
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

    # project = MseedUtil.load_project(file=arg_parse.f)
    sp_loaded = SurfProject.load_project(path_to_project_file=parsed_args.f)
    if len(sp_loaded.project) > 0 and isinstance(sp_loaded, SurfProject):
        picker = PhasenetISP(sp_loaded.project, amplitude=True, min_p_prob=parsed_args.p,
                             min_s_prob=parsed_args.s)

        # Running Stage
        picks = picker.phasenet()
        #
        """ PHASENET OUTPUT TO REAL INPUT """
        #
        picks_results = PhasenetUtils.split_picks(picks)
        PhasenetUtils.convert2real(picks_results, parsed_args.d)
        PhasenetUtils.save_original_picks(picks_results, parsed_args.d)
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
        real_config_file.ini] -w [work directory] -s [path to directory where project will be saved] --verbose
          
        Reference: Zhang et al. 2019, Rapid Earthquake Association and Location, Seismol. Res. Lett. 
        https://doi.org/10.1785/0220190052
            
        Documentation:
          https://projectisp.github.io/surfquaketutorial.github.io/
          # Time file is based on https://github.com/Dal-mzhang/LOC-FLOW/blob/main/LOCFLOW-CookBook.pdf
          # reference for structs: https://github.com/Dal-mzhang/REAL/blob/master/REAL_userguide_July2021.pdf
        """

    arg_parse.add_argument("-i", "--inventory_file_path", help="Inventory file (i.e., *xml or dataless", type=str,
                           required=True)

    arg_parse.add_argument("-p", "--data-dir", help="path to data picking folder", type=str,
                           required=True)

    arg_parse.add_argument("-c", "--config_file_path", help="Path to real_config_file.ini", type=str, required=True)

    arg_parse.add_argument("-w", "--work_dir_path", help="Path to working_directory (Generated Travel Times)", type=str,
                           required=True)

    arg_parse.add_argument("-s", "--save_dir", help="Path to directory where project will be saved", type=str,
                           required=True)

    arg_parse.add_argument("-v", "--verbose", help="information of files included on the project",
                           action="store_true")

    parsed_args = arg_parse.parse_args()
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
        -g [if travel_time_generation needed] -s [if stations_corrections need]

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
                                                                "stations corrections", action="store_true")

    parsed_args = arg_parse.parse_args()
    nll_manager = NllManager(parsed_args.config_file_path, parsed_args.inventory_file_path, parsed_args.out_dir_path)

    if parsed_args.generate_grid:
        print("Generating Velocity Grid")
        nll_manager.vel_to_grid()
        print("Generating Travel-Time Grid")
        nll_manager.grid_to_time()

    print("Starting Locations")
    if parsed_args.stations_corrections:
        # including stations_corrections
        for i in range(25):
            print("Running Location iteration", i)
            nll_manager.run_nlloc()
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
    print("End of process, please review output directory")
    wm = WriteMTI(parsed_args.output_dir_path)
    file_summary = os.path.join(parsed_args.output_dir_path, "summary_mti.txt")
    wm.mti_summary(output=file_summary)


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


if __name__ == "__main__":
    freeze_support()
    main()
