import os
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional

from surfquakecore import model_dir
from surfquakecore.phasenet.phasenet_handler import PhasenetISP, PhasenetUtils
from surfquakecore.utils.obspy_utils import MseedUtil

# should be equal to [project.scripts]
__entry_point_name = "surfquake"


@dataclass
class _CliActions:
    name: str
    run: callable
    description: str = ""


def _create_actions():
    _actions = {
        "project": _CliActions(
            name="project", run=_project, description=f"Type {__entry_point_name} remove -h for help.\n"),
        "pick": _CliActions(
            name="pick", run=_pick, description=f"Type {__entry_point_name} remove -h for help.\n")
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
    arg_parse = ArgumentParser(prog=f"{__entry_point_name} project")
    arg_parse.usage = ("Creating project example: surfquake project -d [path to your data files] "
                       "-s [path to your saving directory] " "-n [project name] --verbose")
    arg_parse.add_argument("-d", help="Path to data files directory", type=str, required=True)
    arg_parse.add_argument("-s", help="Path to directory where project will be saved", type=str,
                           required=True)
    arg_parse.add_argument("-n", help="Project Name", type=str, required=True)
    arg_parse.add_argument("-v", "--verbose", help="information of files included on the project",
                           action="store_true")
    parsed_args = arg_parse.parse_args()

    print(f"Project from {parsed_args.d} saving to {parsed_args.s} as {parsed_args.n}")
    if parsed_args.verbose is not None:
        project = MseedUtil().search_files(parsed_args.d, verbose=True)
    else:
        project = MseedUtil().search_files(parsed_args.d, verbose=False)

    project_file_path = os.path.join(parsed_args.s, parsed_args.n)
    print("End of project creation, number of files ", len(project))
    MseedUtil().save_project(project, project_file_path)

def _pick():

    arg_parse = ArgumentParser(prog=f"{__entry_point_name} pick")
    arg_parse.usage = ("Run picker: -f [path to your project file] "
                       "-s [path to your pick saving directory] -p [P-wave threshoold] -s [S-wave threshold] --verbose")

    arg_parse.add_argument("-f", help="path to your project file", type=str, required=True)
    arg_parse.add_argument("-s", help="Path to directory where picks will be saved", type=str,
                           required=True)
    arg_parse.add_argument("-p", help="P-wave threshoold", type=float,
                           required=True)
    arg_parse.add_argument("-s", help="S-wave threshold", type=float,
                           required=True)

    arg_parse.add_argument("-v", "--verbose", help="information of files included on the project",
                           action="store_true")
    parsed_args = arg_parse.parse_args()
    project = MseedUtil.load_project(file=arg_parse.f)
    # # conservative mode
    phISP = PhasenetISP(project, modelpath=model_dir, amplitude=True, min_p_prob=parsed_args.p,
                        min_s_prob=parsed_args.s)

    # Running Stage
    picks = phISP.phasenet()
    #
    """ PHASENET OUTPUT TO REAL INPUT """
    #
    picks_results = PhasenetUtils.split_picks(picks)
    PhasenetUtils.convert2real(picks_results, parsed_args.s)
    PhasenetUtils.save_original_picks(picks_results, parsed_args.s)

if __name__ == "__main__":
    main()
