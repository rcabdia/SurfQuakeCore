import os
import platform

def __expect_dir(path: str, label: str):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{label} not found: {path}")

def __get_binary_dir():

    #_root_dir = os.path.dirname(__file__)
    # Use realpath to handle symlinks / editable installs
    _root_dir = os.path.dirname(os.path.realpath(__file__))

    _os = (platform.system() or "").strip().lower()
    _arch = (platform.machine() or "").strip().lower()

    # Normalize OS
    if _os.startswith("linux"):
        os_key = "linux"
    elif _os in ("darwin", "mac", "macos", "osx"):
        os_key = "mac"
    elif _os.startswith("win"):
        os_key = "windows"
    else:
        raise ValueError(f"The OS {platform.system()} is not supported.")

    # Normalize arch (only needed for mac split)
    if _arch in ("aarch64", "arm64"):
        arch_key = "arm64"
    elif _arch in ("x86_64", "amd64", "x64"):
        arch_key = "x86_64"
    else:
        arch_key = _arch  # keep as-is for future cases

    # Choose folder (keep your naming)
    if os_key == "linux":
        _os_bin_folder = "linux_bin"
    elif os_key == "mac":
        _os_bin_folder = "mac_bin_m" if arch_key == "arm64" else "mac_bin"
    else:  # windows
        _os_bin_folder = "win_bin"

    bin_root = os.path.join(_root_dir, _os_bin_folder)
    __expect_dir(bin_root, f"{os_key} binary root")

    return bin_root


_bin_dir = __get_binary_dir()

BINARY_REAL_FILE = os.path.join(_bin_dir, "REAL", "REAL")
BINARY_NLL_DIR = os.path.join(_bin_dir, "NLL")
BINARY_GREEN_DIR = os.path.join(_bin_dir, "mti_green")
BINARY_FOCMEC_DIR = os.path.join(_bin_dir, "FOCMEC")

# Light validation of subfolders (keeps your early failure behavior)
__expect_dir(os.path.dirname(BINARY_REAL_FILE), "REAL directory")
__expect_dir(BINARY_NLL_DIR,   "NLL directory")
__expect_dir(BINARY_GREEN_DIR, "mti_green directory")
__expect_dir(BINARY_FOCMEC_DIR,"FOCMEC directory")


__all__ = [
    'BINARY_NLL_DIR',
    'BINARY_REAL_FILE',
    'BINARY_GREEN_DIR',
    'BINARY_FOCMEC_DIR'
]
