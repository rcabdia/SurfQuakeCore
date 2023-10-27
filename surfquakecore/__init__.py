import os
import platform

root_dir = os.path.dirname(__file__)
_parent_dir = os.path.dirname(root_dir)

_os = platform.system()

if _os.lower() == 'linux':
    bin_dir = os.path.join(_parent_dir, "bin", "linux_bin")
    real_bin = os.path.join(bin_dir, "REAL", "REAL")
    nll_bin_dir = os.path.join(bin_dir, "NLL")
    green_dir = os.path.join(bin_dir, "mti_green")
elif _os.lower() == 'mac' or _os.lower() == 'darwin':
    bin_dir = os.path.join(_parent_dir, "bin", "mac_bin")
    real_bin = os.path.join(bin_dir, "REAL", "REAL")
    nll_bin_dir = os.path.join(bin_dir, "NLL")
    green_dir = os.path.join(bin_dir, "mti_green")
else:
    raise AttributeError(f"The OS {_os} is not valid.")

__all__ = [
    'root_dir',
    'bin_dir',
    'real_bin',
]