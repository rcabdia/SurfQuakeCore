import os
import platform
import warnings


def __get_binary_dir():
    _root_dir = os.path.dirname(__file__)
    _os = platform.system()

    _os_bin_folder = ''
    if _os.lower() == 'linux':
        _os_bin_folder = "linux_bin"
    elif _os.lower() == 'mac' or _os.lower() == 'darwin':
        _os_bin_folder = "mac_bin"
    elif _os.lower() == 'windows':
        # warnings.warn(f"The OS {_os} do not support some functions.")
        _os_bin_folder = "win_bin"
    else:
        raise ValueError(f"The OS {_os} is not supported.")

    return os.path.join(_root_dir,  _os_bin_folder)


_bin_dir = __get_binary_dir()
BINARY_REAL_FILE = os.path.join(_bin_dir, "REAL", "REAL")
BINARY_NLL_DIR = os.path.join(_bin_dir, "NLL")
BINARY_GREEN_DIR = os.path.join(_bin_dir, "mti_green")


__all__ = [
    'BINARY_NLL_DIR',
    'BINARY_REAL_FILE',
    'BINARY_GREEN_DIR',
]
