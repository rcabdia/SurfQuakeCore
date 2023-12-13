import os
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from multiprocessing import freeze_support
from typing import Optional
from surfquakecore import model_dir
from surfquakecore.project.surf_project import SurfProject
from surfquakecore.real.real_core import RealCore

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
            name="associate", run=_associate, description=f"Type {__entry_point_name} -h for help.\n")

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
    # project = MseedUtil().search_files(parsed_args.d, verbose=True)
    sp = SurfProject(parsed_args.data_dir)
    project_file_path = os.path.join(parsed_args.save_dir, parsed_args.project_name)
    sp.search_files(verbose=parsed_args.verbose)
    print(sp)
    print("End of project creation, number of files ", len(sp.project))
    # MseedUtil().save_project(project, project_file_path)
    sp.save_project(path_file_to_storage=project_file_path)


def _pick():
    from surfquakecore.phasenet.phasenet_handler import PhasenetISP, PhasenetUtils

    arg_parse = ArgumentParser(prog=f"{__entry_point_name} pick", description="Use Phasenet Neural Network to estimate "
                                                                              "body waves arrival times")
    arg_parse.epilog = """
            Overview:
              The Picking algorythm uses the Deep Neural Network of Phasenet to estimate 
              the arrival times of P- and S-wave

            Usage:
              surfquake pick -f [path to your project file] -d [path to your pick saving directory] -p 
              [P-wave threshoold] -s [S-wave threshold] --verbose"

            Reference:
              Liu, Min, et al. "Rapid characterization of the July 2019 Ridgecrest, California, 
              earthquake sequence from raw seismic data using machine‐learning phase picker." 
              Geophysical Research Letters

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
        picker = PhasenetISP(sp_loaded.project, modelpath=model_dir, amplitude=True, min_p_prob=parsed_args.p,
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

        Usage: surfquake associate -i [inventory_file_path] -p [path to data picking file] -c [path to 
        real_config_file.ini] -s [path to directory where project will be saved] --verbose
          
        Reference: Zhang et al. 2019, Rapid Earthquake Association and Location, Seismol. Res. Lett. 
        https://doi.org/10.1785/0220190052
            
        Documentation:
          https://projectisp.github.io/surfquaketutorial.github.io/
          # Time file is based on https://github.com/Dal-mzhang/LOC-FLOW/blob/main/LOCFLOW-CookBook.pdf
          # reference for structs: https://github.com/Dal-mzhang/REAL/blob/master/REAL_userguide_July2021.pdf
        """

    arg_parse.add_argument("-i", "--inventory_file_path", help="Inventory file (i.e., *xml or dataless", type=str,
                           required=True)
    arg_parse.add_argument("-p", "--data-dir", help="Path to data picking file (output Picking File)", type=str,
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


if __name__ == "__main__":
    freeze_support()
    main()
