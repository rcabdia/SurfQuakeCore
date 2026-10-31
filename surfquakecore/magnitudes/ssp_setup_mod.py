# -*- coding: utf-8 -*-
# SPDX-License-Identifier: CECILL-2.1
"""
Setup functions for SourceSpec, adapted for SurfQuakeCore.

Derived from ``sourcespec/ssp_setup.py`` (SourceSpec 1.8).

:copyright (original):
    2012 Claudio Satriano <satriano@ipgp.fr>

    2013-2014 Claudio Satriano <satriano@ipgp.fr>,
              Emanuela Matrullo <matrullo@geologie.ens.fr>,
              Agnes Chounet <chounet@ipgp.fr>

    2015-2024 Claudio Satriano <satriano@ipgp.fr>
:license:
    CeCILL Free Software License Agreement v2.1
    (http://www.cecill.info/licences.en.html)

Modifications for SurfQuakeCore (Roberto Cabieces):
    - No sys.exit(): errors are raised as exceptions (SourceSpecError family),
      so a batch over many events is not killed by one bad event.
    - No work at import time (no SIGINT handler, no IPython shell).
    - No ObsPy / Matplotlib / Cartopy version checks nor cartopy data
      downloads; no sample-config / update-config / update-db code
      (use ``source_spec -U old.conf`` from the sourcespec CLI to update a
      config file).
    - Instrument codes are recomputed on every configure() (the original
      appended to the same list on each call).
    - setup_logging() detaches the handlers of the previous event, so logs of
      different events are not mixed.
    - The state that unmodified sourcespec modules read from
      ``sourcespec.ssp_setup`` (instrument codes, TRACEID_MAP, the
      "exit already called" flag) is pushed there by ``_sync_upstream_state``.
"""
import contextlib
import importlib
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import sys
import uuid
from collections import defaultdict
from copy import copy
from datetime import datetime

from sourcespec.config import Config
from sourcespec.configobj import ConfigObj
from sourcespec.configobj.validate import Validator


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class SourceSpecError(Exception):
    """Base class of the errors raised by this module."""


class SourceSpecConfigError(SourceSpecError):
    """Invalid, deprecated or incomplete SourceSpec configuration."""


class SourceSpecExit(SourceSpecError):
    """Raised by ssp_exit() instead of calling sys.exit()."""

    def __init__(self, retval=0, abort=False):
        super().__init__(f'source_spec exit (retval={retval}, abort={abort})')
        self.retval = retval
        self.abort = abort


# ---------------------------------------------------------------------------
# Module state (same names as sourcespec 1.8 ssp_setup.py)
# ---------------------------------------------------------------------------
# SEED standard instrument codes:
# https://ds.iris.edu/ds/nodes/dmc/data/formats/seed-channel-naming/
DEFAULT_INSTR_CODES_VEL = ('H', 'L')
DEFAULT_INSTR_CODES_ACC = ('N',)
INSTR_CODES_VEL = list(DEFAULT_INSTR_CODES_VEL)
INSTR_CODES_ACC = list(DEFAULT_INSTR_CODES_ACC)
TRACEID_MAP = None
LOGGER = None
OLDLOGFILE = None
_LOG_HANDLERS = []   # handlers created by setup_logging()

# sourcespec modules that read ``TRACEID_MAP`` as a module-level global
_UPSTREAM_TRACEID_MAP_USERS = (
    'sourcespec.ssp_read_traces',
    'sourcespec.ssp_read_event_metadata',
)


# ---------------------------------------------------------------------------
# Config reading / validation
# ---------------------------------------------------------------------------
def _read_config(config_file, configspec=None):
    kwargs = {
        'configspec': configspec,
        'file_error': True,
        'default_encoding': 'utf8'
    }
    if configspec is None:
        kwargs.update({
            'interpolation': False,
            'list_values': False,
            '_inspec': True
        })
    try:
        return ConfigObj(config_file, **kwargs)
    except IOError as err:
        raise SourceSpecConfigError(str(err)) from err
    except Exception as err:
        raise SourceSpecConfigError(
            f'Unable to read "{config_file}": {err}') from err


def _parse_configspec():
    # configspec.conf lives next to this file (in sourcespec it is in
    # the "config_files" subfolder)
    configspec_file = os.path.join(
        os.path.dirname(__file__), 'configspec.conf')
    return _read_config(configspec_file)


def _write_config(config_obj, progname, outdir):
    if progname != 'source_spec':
        return
    os.makedirs(outdir, exist_ok=True)
    configfile = os.path.join(outdir, f'{progname}.conf')
    with open(configfile, 'wb') as fp:
        # do not save the basemap API key in the output folder
        tmp_config_obj = copy(config_obj)
        tmp_config_obj['plot_map_api_key'] = None
        tmp_config_obj.write(fp)


_DEPRECATED_OPTIONS = {
    's_win_length': '"s_win_length" and "noise_win_length" are no more '
                    'supported. Both are replaced by "win_length".',
    'noise_win_length': '"s_win_length" and "noise_win_length" are no more '
                        'supported. Both are replaced by "win_length".',
    'traceids': '"traceids" has been renamed to "traceid_mapping_file".',
    'ignore_stations': '"ignore_stations" has been renamed to '
                       '"ignore_traceids".',
    'use_stations': '"use_stations" has been renamed to "use_traceids".',
    'dataless': '"dataless" has been renamed to "station_metadata".',
    'clip_nmax': '"clip_nmax" has been renamed to "clip_max_percent" '
                 '(the new default is 5%).',
    'trace_format': '"trace_format" is no more supported. Use "sensitivity" '
                    'to specify how sensor sensitivity is computed.',
    'PLOT_SHOW': '"PLOT_SHOW" has been renamed to "plot_show".',
    'PLOT_SAVE': '"PLOT_SAVE" has been renamed to "plot_save".',
    'PLOT_SAVE_FORMAT': '"PLOT_SAVE_FORMAT" has been renamed to '
                        '"plot_save_format".',
    'vp': '"vp" has been renamed to "vp_source".',
    'vs': '"vs" has been renamed to "vs_source".',
    'pre_p_time': '"pre_p_time" has been renamed to "noise_pre_time".',
    'pre_s_time': '"pre_s_time" has been renamed to "signal_pre_time".',
    'rps_from_focal_mechanism': '"rps_from_focal_mechanism" has been '
                                'renamed to "rp_from_focal_mechanism".',
    'paz': '"paz" has been removed and merged with "station_metadata".',
    # new in 1.8
    'rho': '"rho" has been renamed to "rho_source" (see also '
           '"rho_stations").',
    'pi_bsd_min_max': '"pi_bsd_min_max" has been renamed to '
                      '"pi_ssd_min_max".',
    'max_epi_dist': '"max_epi_dist" has been removed and replaced by '
                    '"epi_dist_ranges".',
    'max_freq_Er': '"max_freq_Er" has been removed and replaced by '
                   '"Er_freq_range".',
}


def _check_deprecated_config_options(config_obj):
    msgs = []
    for key, msg in _DEPRECATED_OPTIONS.items():
        if key in config_obj and msg not in msgs:
            msgs.append(msg)
    if msgs:
        raise SourceSpecConfigError(
            'The config file contains parameters not valid in SourceSpec '
            '1.8:\n> ' + '\n> '.join(msgs) +
            '\nUpdate it with "source_spec -U <config_file>" '
            '(sourcespec command line tool).')


_MANDATORY_CONFIG_PARAMS = [
    'p_arrival_tolerance', 's_arrival_tolerance', 'noise_pre_time',
    'signal_pre_time', 'win_length', 'taper_halfwidth',
    'spectral_smooth_width_decades',
    'bp_freqmin_acc', 'bp_freqmax_acc',
    'bp_freqmin_shortp', 'bp_freqmax_shortp',
    'bp_freqmin_broadb', 'bp_freqmax_broadb',
    'freq1_acc', 'freq2_acc', 'freq1_shortp', 'freq2_shortp',
    'freq1_broadb', 'freq2_broadb',
    'rmsmin', 'sn_min', 'spectral_sn_min', 'rpp', 'rps',
    'geom_spread_n_exponent', 'geom_spread_cutoff_distance',
    'f_weight', 'weight', 't_star_0', 't_star_0_variability',
    'a', 'b', 'c', 'ml_bp_freqmin', 'ml_bp_freqmax',
    'n_sigma', 'lower_percentage', 'mid_percentage', 'upper_percentage',
    'nIQR', 'plot_spectra_maxrows', 'plot_traces_maxrows',
    'plot_station_text_size',
]


def _check_mandatory_config_params(config_obj):
    msgs = [f'"{par}" is mandatory and cannot be None'
            for par in _MANDATORY_CONFIG_PARAMS if config_obj[par] is None]
    if msgs:
        raise SourceSpecConfigError('\n'.join(msgs))


def _check_optional_dependencies(config):
    """Only check that the optional libraries can be imported."""
    # pylint: disable=import-outside-toplevel,unused-import
    if config.plot_station_map:
        try:
            import cartopy  # noqa: F401
            import pyproj  # noqa: F401
        except ImportError as err:
            raise SourceSpecConfigError(
                'cartopy and pyproj are needed to plot the station map. '
                'Install them or set "plot_station_map" to False in the '
                'config file.') from err
    if config.NLL_time_dir is not None or config.NLL_model_dir is not None:
        try:
            import nllgrid  # noqa: F401
        except ImportError as err:
            raise SourceSpecConfigError(
                'nllgrid is needed to use NonLinLoc grids.') from err


def _init_plotting(plot_show):
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    if not plot_show:
        plt.switch_backend('Agg')


# ---------------------------------------------------------------------------
# Paths and list-like parameters (new in sourcespec 1.8)
# ---------------------------------------------------------------------------
def _fix_and_expand_path(path):
    """Fix any path issues and expand it (same logic as sourcespec 1.8)."""
    fixed_path = os.path.normpath(path).split(os.sep)
    fixed_path = os.path.join(*fixed_path)
    if path.startswith(os.sep):
        fixed_path = os.path.join(os.sep, fixed_path)
    elif path.startswith('~'):
        fixed_path = os.path.expanduser(fixed_path)
    return fixed_path


def _float_list(input_list, max_length=None, accepted_values=None):
    """Convert an input list to a list of floats (or accepted values)."""
    if input_list is None:
        return None
    if accepted_values is None:
        accepted_values = []

    def _parse_float(val):
        val = None if val == 'None' else val
        return val if val in accepted_values else float(val)

    try:
        return [_parse_float(val) for val in input_list[:max_length]]
    except ValueError as err:
        raise ValueError('Cannot parse all values in list') from err


def _none_length(input_list):
    """Length of the list, or 1 if it is None."""
    return 1 if input_list is None else len(input_list)


def _parse_list_parameters(config):
    """Parse force_list options into lists of floats and check them."""
    param = None
    try:
        for param in ('vp_source', 'vs_source', 'rho_source',
                      'layer_top_depths'):
            config[param] = _float_list(config[param])
    except ValueError as err:
        raise SourceSpecConfigError(
            f'Error parsing parameter "{param}": {err}') from err
    lengths = {_none_length(config[p]) for p in (
        'vp_source', 'vs_source', 'rho_source', 'layer_top_depths')}
    if len(lengths) != 1:
        raise SourceSpecConfigError(
            '"vp_source", "vs_source", "rho_source", and '
            '"layer_top_depths" must have the same length.')

    if config.Er_freq_range is None:
        config.Er_freq_range = [None, None]
    try:
        config.Er_freq_range = _float_list(
            config.Er_freq_range, max_length=2,
            accepted_values=[None, 'noise'])
    except ValueError as err:
        raise SourceSpecConfigError(
            f'Error parsing parameter "Er_freq_range": {err}') from err


# ---------------------------------------------------------------------------
# Module state initialisation
# ---------------------------------------------------------------------------
def _init_instrument_codes(config):
    """Recompute instrument codes from defaults + user config (idempotent)."""
    vel = list(DEFAULT_INSTR_CODES_VEL)
    acc = list(DEFAULT_INSTR_CODES_ACC)
    user_acc = config.instrument_code_acceleration
    user_vel = config.instrument_code_velocity
    # remove user-defined codes from the other family if they conflict
    if user_acc is not None and user_acc in vel:
        vel.remove(user_acc)
    if user_vel is not None and user_vel in acc:
        acc.remove(user_vel)
    if user_vel is not None and user_vel not in vel:
        vel.append(user_vel)
    if user_acc is not None and user_acc not in acc:
        acc.append(user_acc)
    # in-place, so that every holder of these list objects sees the change
    INSTR_CODES_VEL[:] = vel
    INSTR_CODES_ACC[:] = acc


def _init_traceid_map(traceid_map_file):
    global TRACEID_MAP  # pylint: disable=global-statement
    if traceid_map_file is None:
        TRACEID_MAP = None
        return
    try:
        with open(traceid_map_file, 'r', encoding='utf-8') as fp:
            TRACEID_MAP = json.loads(fp.read())
    except Exception as err:
        raise SourceSpecConfigError(
            f'traceid mapping file "{traceid_map_file}" not found '
            'or not in json format.') from err


def _sync_upstream_state():
    """
    Push our state into the (unmodified) sourcespec modules.

    * INSTR_CODES_VEL / INSTR_CODES_ACC: updated in place, because
      ssp_read_traces and ssp_read_station_metadata hold references to the
      list objects of sourcespec.ssp_setup.
    * TRACEID_MAP: ssp_read_traces and ssp_read_event_metadata import it by
      value (so it stays None upstream); we set the module attribute they read.
    * SSP_EXIT_CALLED: upstream ssp_exit() does nothing the second time it is
      called in a process. In a batch, after the first failed event every
      later ssp_exit() would silently return and the code would go on
      with an unusable state. We reset the flag for every event.

    Remove this function once those modules are vendored and import from here.
    """
    try:
        upstream = importlib.import_module('sourcespec.ssp_setup')
    except ImportError:
        return
    upstream.INSTR_CODES_VEL[:] = INSTR_CODES_VEL
    upstream.INSTR_CODES_ACC[:] = INSTR_CODES_ACC
    upstream.TRACEID_MAP = TRACEID_MAP
    upstream.SSP_EXIT_CALLED = False
    for modname in _UPSTREAM_TRACEID_MAP_USERS:
        try:
            module = importlib.import_module(modname)
        except ImportError:
            continue
        module.TRACEID_MAP = TRACEID_MAP


# ---------------------------------------------------------------------------
# configure
# ---------------------------------------------------------------------------
def configure(options, progname='source_spec', config_overrides=None):
    """
    Read and validate the config file and build the ``Config`` object.

    :param object options: object with the attributes of
        ``SoursceSpecOptions`` (config_file, outdir, station_metadata, ...).
        ``options.outdir`` is modified: a ``no_evid_<hex>`` subfolder is
        appended, to be renamed once the event id is known.
    :param str progname: program name ('source_spec')
    :param dict config_overrides: parameters that override the config file
    :return: ``Config`` object with both options and config parameters
    :raises SourceSpecConfigError: invalid configuration
    """
    configspec = _parse_configspec()

    if options.config_file:
        options.config_file = _fix_and_expand_path(options.config_file)
    config_obj = _read_config(options.config_file, configspec)

    if config_overrides is not None:
        try:
            for key, value in config_overrides.items():
                config_obj[key] = value
        except AttributeError as err:
            raise ValueError('"config_overrides" must be a dict-like.') \
                from err

    # Set to None all the 'None' strings
    for key, value in config_obj.dict().items():
        if value == 'None':
            config_obj[key] = None

    # test is:
    # - True if everything is ok
    # - False if no config value is provided
    # - A dict if invalid values are present, with the invalid values as False
    test = config_obj.validate(Validator())
    if isinstance(test, dict):
        bad = [f'Invalid value for "{entry}": "{config_obj[entry]}"'
               for entry in test if not test[entry]]
        raise SourceSpecConfigError('\n'.join(bad) or 'Invalid config file')
    if not test:
        raise SourceSpecConfigError('No configuration value present!')

    _check_deprecated_config_options(config_obj)
    _check_mandatory_config_params(config_obj)

    # Fix and expand paths in options
    options.outdir = _fix_and_expand_path(options.outdir)
    if options.trace_path:
        options.trace_path = [
            _fix_and_expand_path(path) for path in options.trace_path]
    if options.qml_file:
        options.qml_file = _fix_and_expand_path(options.qml_file)
    if options.hypo_file:
        options.hypo_file = _fix_and_expand_path(options.hypo_file)
    if options.pick_file:
        options.pick_file = _fix_and_expand_path(options.pick_file)

    # Create a 'no_evid_' subdir into outdir (unique random name).
    # The caller renames it once the evid is available.
    options.outdir = os.path.join(
        options.outdir, f'no_evid_{uuid.uuid4().hex}')
    _write_config(config_obj, progname, options.outdir)

    config = Config(config_obj.dict().copy())
    config.options = options

    # Override station_metadata config option with the option given by caller
    if options.station_metadata is not None:
        config.station_metadata = options.station_metadata

    # Additional config values
    config.vertical_channel_codes = ['Z']
    config.horizontal_channel_codes_1 = ['N', 'R']
    config.horizontal_channel_codes_2 = ['E', 'T']
    msc = config.mis_oriented_channels
    if msc is not None:
        config.vertical_channel_codes.append(msc[0])
        config.horizontal_channel_codes_1.append(msc[1])
        config.horizontal_channel_codes_2.append(msc[2])

    # Fix and expand paths in config
    if config.database_file:
        config.database_file = _fix_and_expand_path(config.database_file)
    if config.traceid_mapping_file:
        config.traceid_mapping_file = _fix_and_expand_path(
            config.traceid_mapping_file)
    if config.station_metadata:
        config.station_metadata = _fix_and_expand_path(config.station_metadata)
    if config.residuals_filepath:
        config.residuals_filepath = _fix_and_expand_path(
            config.residuals_filepath)

    _parse_list_parameters(config)

    # Warnings to be issued when the logger is set up
    config.warnings = []
    if config.html_report:
        if not config.plot_save:
            config.warnings.append(
                'The "html_report" option is selected but "plot_save" '
                'is "False". HTML report will have no plots.')
        if config.plot_save_format not in ['png', 'svg']:
            config.warnings.append(
                'The "html_report" option is selected but '
                '"plot_save_format" is not "png" or "svg". '
                'HTML report will have no plots.')

    _check_optional_dependencies(config)
    _init_instrument_codes(config)
    _init_traceid_map(config.traceid_mapping_file)
    _sync_upstream_state()
    _init_plotting(config.plot_show)
    # Dict to store figure paths
    config.figures = defaultdict(list)
    # Absolute path of the current working directory
    config.workdir = os.getcwd()
    return config


# ---------------------------------------------------------------------------
# Logging / exit
# ---------------------------------------------------------------------------
def _color_handler_emit(fn):
    """
    Add color-coding to the logging handler emitter.

    Source: https://stackoverflow.com/a/20707569/2021880
    """
    def new(*args):
        levelno = args[0].levelno
        if levelno >= logging.ERROR:
            color = '\x1b[31;1m'  # red
        elif levelno >= logging.WARNING:
            color = '\x1b[33;1m'  # yellow
        elif levelno >= logging.INFO:
            color = '\x1b[0m'  # no color
        elif levelno >= logging.DEBUG:
            color = '\x1b[35;1m'  # purple
        else:
            color = '\x1b[0m'
        args[0].msg = f'{color}{args[0].msg}\x1b[0m'
        return fn(*args)
    return new


def _pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except Exception:  # pylint: disable=broad-except
        return 'unknown'


def _detach_log_handlers():
    """Close and remove only the handlers created by setup_logging()."""
    logger_root = logging.getLogger()
    while _LOG_HANDLERS:
        hdlr = _LOG_HANDLERS.pop()
        with contextlib.suppress(Exception):
            hdlr.flush()
            hdlr.close()
        logger_root.removeHandler(hdlr)


def setup_logging(config, basename=None, progname='source_spec'):
    """
    Set up the logging infrastructure.

    Called without ``basename`` it starts a fresh log file for the current
    event (the handlers of the previous event are detached).
    Called with ``basename`` after a previous call, the current log file is
    moved to ``<basename>.ssp.log`` and logging continues in append mode.
    """
    global OLDLOGFILE, LOGGER  # pylint: disable=global-statement
    outdir = config.options.outdir
    os.makedirs(outdir, exist_ok=True)

    if basename:
        logfile = os.path.join(outdir, f'{basename}.ssp.log')
    else:
        datestring = datetime.now().strftime('%Y%m%d_%H%M%S')
        logfile = os.path.join(outdir, f'{datestring}.ssp.log')

    _detach_log_handlers()

    filemode = 'w'
    if basename and OLDLOGFILE and os.path.isfile(OLDLOGFILE):
        if os.path.abspath(OLDLOGFILE) != os.path.abspath(logfile):
            shutil.copyfile(OLDLOGFILE, logfile)
            os.remove(OLDLOGFILE)
        filemode = 'a'

    with contextlib.suppress(Exception):
        logging.captureWarnings(True)
    logger_root = logging.getLogger()
    logger_root.setLevel(logging.DEBUG)

    filehand = logging.FileHandler(filename=logfile, mode=filemode)
    filehand.setLevel(logging.DEBUG)
    filehand.setFormatter(logging.Formatter(
        '%(asctime)s %(name)-20s %(levelname)-8s %(message)s'))
    logger_root.addHandler(filehand)
    _LOG_HANDLERS.append(filehand)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    if sys.platform != 'win32' and sys.stdout.isatty():
        console.emit = _color_handler_emit(console.emit)
    logger_root.addHandler(console)
    _LOG_HANDLERS.append(console)

    LOGGER = logging.getLogger(progname)

    if filemode == 'w':
        uname = platform.uname()
        LOGGER.debug('source_spec START')
        LOGGER.debug(f'SourceSpec version: {_pkg_version("sourcespec")}')
        LOGGER.debug(f'Platform: {uname[0]} {uname[2]} {uname[4]}')
        LOGGER.debug(
            f'Python version: {".".join(map(str, sys.version_info[:3]))}')
        for pkg in ('obspy', 'numpy', 'scipy', 'matplotlib', 'cartopy'):
            version = _pkg_version(pkg)
            if version != 'unknown':
                LOGGER.debug(f'{pkg} version: {version}')
    OLDLOGFILE = logfile

    for _ in range(len(config.warnings)):
        LOGGER.warning(config.warnings.pop(0))


def ssp_exit(retval=0, abort=False):
    """
    Replacement of the original ssp_exit().

    It raises ``SourceSpecExit`` (an ``Exception``) instead of calling
    ``sys.exit()``, so the caller can skip the event and go on.
    """
    if LOGGER is not None:
        LOGGER.debug('source_spec ABORTED' if abort else 'source_spec END')
    raise SourceSpecExit(retval, abort)
