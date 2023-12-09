import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional

# should be equal to [project.scripts]
__entry_point_name = "surfquake"


@dataclass
class _CliActions:
    name: str
    run: callable
    description: str = ""


def _create_actions():
    _actions = {
        "run": _CliActions(
            name="run", run=_install, description=f"Type {__entry_point_name} run -h for help.\n"),
        "remove": _CliActions(
            name="remove", run=_remove, description=f"Type {__entry_point_name} remove -h for help.\n")
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


def _install():
    arg_parse = ArgumentParser(prog=f"{__entry_point_name} install")
    arg_parse.add_argument("-d", help="The directory path", required=True)
    run_args = arg_parse.parse_args()

    print(f"run {run_args.d}")


def _remove():
    arg_parse = ArgumentParser(prog=f"{__entry_point_name} remove")
    arg_parse.add_argument("-o", help="Option to remove", required=True)
    parsed_args = arg_parse.parse_args()
    print(f"remove {parsed_args.o}")


if __name__ == "__main__":
    main()
