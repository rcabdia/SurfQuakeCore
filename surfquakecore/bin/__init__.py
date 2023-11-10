import os
import platform
import warnings

_root_dir = os.path.dirname(__file__)
_os = platform.system()

_os_bin_folder = ''
if _os.lower() == 'linux':
    _os_bin_folder = "linux_bin"
elif _os.lower() == 'mac' or _os.lower() == 'darwin':
    _os_bin_folder = "mac_bin"
else:
    warnings.warn(f"The OS {_os} do not support some functions.")
    _os_bin_folder = "win_bin"
    # raise Warning(f"The OS {_os} do Not support some functions.")

_bin_dir = os.path.join(_root_dir,  _os_bin_folder)
real_bin = os.path.join(_bin_dir, "REAL", "REAL")
nll_bin_dir = os.path.join(_bin_dir, "NLL")
green_bin_dir = os.path.join(_bin_dir, "mti_green")


__all__ = [
    'nll_bin_dir',
    'real_bin',
    'green_bin_dir',
]
