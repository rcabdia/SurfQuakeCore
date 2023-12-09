import sys


def get_python_major_version() -> int:

    return int(sys.version.split(".")[1])
