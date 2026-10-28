# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: global_config.py
# Program: surfQuake global configuration
# Purpose: Persistent, user-editable defaults shared by all SurfQuake tools
# --------------------------------------------------------------------
"""
Global SurfQuake configuration (INI file).

Precedence for every key, highest first:

    1. Explicit argument at the call site       (read_stream(path, format="MSEED"))
    2. Environment variable                     (SURFQUAKE_READ_FORMAT=MSEED)
    3. Project file, in the working directory   (./surfquake.ini)
    4. User file                                (~/.config/surfquake/surfquake.ini)
    5. Built-in defaults

The user file location can be redirected with the SURFQUAKE_CONFIG environment
variable.

Typical use inside the library::

    from surfquakecore.global_config import read_stream
    st = read_stream(path)                    # uses the configured defaults
    st = read_stream(path, format="SAC")      # explicit argument wins

or, if a call site needs obspy.read directly::

    from surfquakecore.global_config import read_kwargs
    st = obspy.read(path, headonly=True, **read_kwargs())
"""

import configparser
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from obspy import read

CONFIG_ENV_VAR = "SURFQUAKE_CONFIG"
PROJECT_CONFIG_NAME = "surfquake.ini"
_ENV_PREFIX = "SURFQUAKE_"

_NONE_TOKENS = {"", "none", "null", "auto"}
_TRUE_TOKENS = {"1", "true", "yes", "on"}
_FALSE_TOKENS = {"0", "false", "no", "off"}

_UNSET = object()


# --------------------------------------------------------------------------- #
# Option definitions                                                          #
# --------------------------------------------------------------------------- #
def _parse_bool(raw: str) -> bool:
    token = str(raw).strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    raise ValueError(f"'{raw}' is not a boolean (use true/false)")


def _obspy_waveform_formats() -> Optional[List[str]]:
    """Formats supported by the installed ObsPy, or None if it cannot be told."""
    try:
        from obspy.core.util.base import ENTRY_POINTS
        return sorted(ENTRY_POINTS["waveform"].keys())
    except Exception:
        return None


def _parse_format(raw: str) -> Optional[str]:
    token = str(raw).strip()
    if token.lower() in _NONE_TOKENS:
        return None
    token = token.upper()
    known = _obspy_waveform_formats()
    if known is not None and token not in known:
        raise ValueError(
            f"'{raw}' is not a waveform format known to ObsPy. "
            f"Use 'auto' or one of: {', '.join(known)}"
        )
    return token


@dataclass(frozen=True)
class _Option:
    default: Any
    parse: Callable[[str], Any]
    help: str


# key = "section.option"
OPTIONS: Dict[str, _Option] = {
    "read.format": _Option(
        default=None,
        parse=_parse_format,
        help="Waveform format passed to obspy.read. 'auto' (None) lets ObsPy guess it.",
    ),
    "read.check_compression": _Option(
        default=True,
        parse=_parse_bool,
        help="Let obspy.read detect gzip/bzip2/zip compression. "
             "Set to false to skip the check (faster on large archives).",
    ),
}


def _serialize(value: Any) -> str:
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def format_value(value: Any) -> str:
    """Human-readable representation (None is shown as None)."""
    return str(value)


def _split_key(key: str) -> Tuple[str, str]:
    section, _, option = key.partition(".")
    return section, option


def _check_key(key: str) -> str:
    key = key.strip().lower()
    if key not in OPTIONS:
        raise KeyError(
            f"Unknown key '{key}'. Valid keys: {', '.join(sorted(OPTIONS))}"
        )
    return key


# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #
def user_config_path() -> Path:
    env = os.environ.get(CONFIG_ENV_VAR)
    if env:
        return Path(env).expanduser()
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "surfquake" / "surfquake.ini"


def project_config_path() -> Path:
    return Path.cwd() / PROJECT_CONFIG_NAME


def _scope_path(scope: str) -> Path:
    if scope == "user":
        return user_config_path()
    if scope == "project":
        return project_config_path()
    raise ValueError("scope must be 'user' or 'project'")


# --------------------------------------------------------------------------- #
# Config object                                                               #
# --------------------------------------------------------------------------- #
def _new_parser() -> configparser.ConfigParser:
    return configparser.ConfigParser(interpolation=None)


def _warn(msg: str) -> None:
    print(f"[surfquake config][WARN] {msg}", file=sys.stderr)


class GlobalConfig:
    """Effective configuration, resolved once from defaults, files and environment."""

    def __init__(self):
        self._values: Dict[str, Any] = {k: o.default for k, o in OPTIONS.items()}
        self._sources: Dict[str, str] = {k: "default" for k in OPTIONS}
        self._load_file(user_config_path(), "user")
        self._load_file(project_config_path(), "project")
        self._load_env()

    def _load_file(self, path: Path, label: str) -> None:
        if not path.is_file():
            return
        parser = _new_parser()
        try:
            parser.read(path, encoding="utf-8")
        except configparser.Error as exc:
            _warn(f"Could not parse {path}: {exc}")
            return

        for section in parser.sections():
            for option in parser[section]:
                key = f"{section.lower()}.{option.lower()}"
                if key not in OPTIONS:
                    _warn(f"Unknown key '{key}' in {path} (ignored)")
                    continue
                try:
                    self._values[key] = OPTIONS[key].parse(parser[section][option])
                    self._sources[key] = label
                except ValueError as exc:
                    _warn(f"{path}: {key}: {exc} (ignored)")

    def _load_env(self) -> None:
        for key, opt in OPTIONS.items():
            env_name = _ENV_PREFIX + key.upper().replace(".", "_")
            raw = os.environ.get(env_name)
            if raw is None:
                continue
            try:
                self._values[key] = opt.parse(raw)
                self._sources[key] = f"env ({env_name})"
            except ValueError as exc:
                _warn(f"{env_name}: {exc} (ignored)")

    def get(self, key: str) -> Any:
        return self._values[_check_key(key)]

    def source(self, key: str) -> str:
        return self._sources[_check_key(key)]

    def items(self) -> List[Tuple[str, Any, str]]:
        return [(k, self._values[k], self._sources[k]) for k in OPTIONS]


_CONFIG: Optional[GlobalConfig] = None


def get_config(reload: bool = False) -> GlobalConfig:
    """Return the cached configuration (loaded on first use)."""
    global _CONFIG
    if _CONFIG is None or reload:
        _CONFIG = GlobalConfig()
    return _CONFIG


def reload_config() -> GlobalConfig:
    return get_config(reload=True)


# --------------------------------------------------------------------------- #
# Writing                                                                     #
# --------------------------------------------------------------------------- #
_HEADER = (
    "# SurfQuake global configuration\n"
    "# Managed with:  surfquake config set <key> <value>\n"
    "# 'auto' means None (let the library decide).\n\n"
)


def _write_parser(parser: configparser.ConfigParser, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_HEADER)
        parser.write(fh)


def _read_parser(path: Path) -> configparser.ConfigParser:
    parser = _new_parser()
    if path.is_file():
        parser.read(path, encoding="utf-8")
    return parser


def set_value(key: str, value: str, scope: str = "user") -> Tuple[Path, Any]:
    """Validate and persist a value. Returns (file written, parsed value)."""
    key = _check_key(key)
    parsed = OPTIONS[key].parse(value)  # raises ValueError if invalid

    path = _scope_path(scope)
    parser = _read_parser(path)
    section, option = _split_key(key)
    if not parser.has_section(section):
        parser.add_section(section)
    parser.set(section, option, _serialize(parsed))
    _write_parser(parser, path)

    reload_config()
    return path, parsed


def reset_value(key: Optional[str] = None, scope: str = "user") -> Optional[Path]:
    """Remove one key (or the whole file when key is None) from a scope's file.

    Returns the affected path, or None if there was nothing to remove.
    """
    path = _scope_path(scope)
    if not path.is_file():
        return None

    if key is None:
        path.unlink()
        reload_config()
        return path

    key = _check_key(key)
    parser = _read_parser(path)
    section, option = _split_key(key)
    if not parser.has_option(section, option):
        return None
    parser.remove_option(section, option)
    if not parser.options(section):
        parser.remove_section(section)

    if parser.sections():
        _write_parser(parser, path)
    else:
        path.unlink()
    reload_config()
    return path


# --------------------------------------------------------------------------- #
# Reading waveforms                                                           #
# --------------------------------------------------------------------------- #
def read_kwargs(**overrides) -> Dict[str, Any]:
    """Keyword arguments for obspy.read built from the configuration.

    Anything passed explicitly overrides the configured value.
    """
    cfg = get_config()
    kwargs = {
        "format": cfg.get("read.format"),
        "check_compression": cfg.get("read.check_compression"),
    }
    kwargs.update(overrides)
    return kwargs


def read_stream(pathname_or_url=None, format=_UNSET, check_compression=_UNSET, **kwargs):

    """Drop-in replacement for obspy.read that applies the global defaults.

    Passing format / check_compression explicitly (even format=None) overrides
    the configuration. Any other obspy.read keyword (headonly, starttime,
    endtime, ...) is forwarded unchanged.
    """


    cfg = get_config()
    if format is _UNSET:
        format = cfg.get("read.format")
    if check_compression is _UNSET:
        check_compression = cfg.get("read.check_compression")

    return read(pathname_or_url, format=format, check_compression=check_compression, **kwargs)
