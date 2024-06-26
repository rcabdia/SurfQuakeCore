# -*- coding: utf-8 -*-
# SPDX-License-Identifier: CECILL-2.1
"""
Arrival time calculation for sourcespec.

:copyright:
    2012-2023 Claudio Satriano <satriano@ipgp.fr>
:license:
    CeCILL Free Software License Agreement v2.1
    (http://www.cecill.info/licences.en.html)
"""
import contextlib
import logging
import warnings
from math import asin, degrees
from obspy.taup import TauPyModel
model = TauPyModel(model='iasp91')
logger = logging.getLogger(__name__.split('.')[-1])



def _wave_arrival_vel(trace, vel):
    """Travel time and takeoff angle using a constant velocity (in km/s)."""
    if vel is None:
        raise RuntimeError
    travel_time = trace.stats.hypo_dist / vel
    takeoff_angle = degrees(asin(trace.stats.epi_dist/trace.stats.hypo_dist))
    # takeoff angle is 180° upwards and 0° downwards
    takeoff_angle = 180. - takeoff_angle
    return travel_time, takeoff_angle


def _wave_arrival_taup(trace, phase):
    """Travel time and takeoff angle using taup."""
    phase_list = [phase.lower(), phase]
    kwargs = dict(
        source_depth_in_km=trace.stats.hypo.depth,
        distance_in_degree=trace.stats.gcarc,
        phase_list=phase_list)
    with warnings.catch_warnings(record=True) as warns:
        try:
            arrivals = model.get_travel_times(**kwargs)
        except Exception:
            trace.stats.hypo.depth = 0.
            kwargs['source_depth_in_km'] = 0.
            arrivals = model.get_travel_times(**kwargs)
        for w in warns:
            message = str(w.message)
            # Ignore a specific obspy.taup warning we do not care about
            if '#2280' in message:
                continue
            logger.warning(message)
    # get first arrival
    first_arrival = sorted(arrivals, key=lambda a: a.time)[0]
    travel_time = first_arrival.time
    takeoff_angle = first_arrival.takeoff_angle
    return travel_time, takeoff_angle


def _wave_arrival(trace, phase, config):
    """Get travel time and takeoff angle."""

    vel = {'P': config.vp_tt, 'S': config.vs_tt}

    with contextlib.suppress(RuntimeError):
        travel_time, takeoff_angle =\
            _wave_arrival_vel(trace, vel[phase])
        method = f'constant V{phase.lower()}: {vel[phase]:.1f} km/s'
        return travel_time, takeoff_angle, method
    try:
        travel_time, takeoff_angle = _wave_arrival_taup(trace, phase)
        method = 'global velocity model (iasp91)'
        return travel_time, takeoff_angle, method
    except RuntimeError:
        raise


def _validate_pick(pick, theo_pick_time, tolerance, trace_id):
    """Check if a pick is valid, i.e., close enough to theoretical one."""
    if theo_pick_time is None:
        return True
    delta_t = pick.time - theo_pick_time
    if abs(delta_t) > tolerance:  # seconds
        logger.warning(
            f'{trace_id}: measured {pick.phase} pick time - theoretical time '
            f'= {delta_t:.1f} s')
        return False
    return True


def _get_theo_pick_time(trace, travel_time):
    if trace.stats.hypo.origin_time is None:
        msg = (
            f'{trace.id}: hypocenter origin time not set: '
            'unable to compute theoretical pick time')
        if msg not in _get_theo_pick_time.msg_cache:
            _get_theo_pick_time.msg_cache.append(msg)
            logger.warning(msg)
        return None
    return trace.stats.hypo.origin_time + travel_time
_get_theo_pick_time.msg_cache = [] # noqa


def _travel_time_from_pick(trace, pick_time):
    try:
        travel_time = pick_time - trace.stats.hypo.origin_time
    except TypeError:
        travel_time = None
    return travel_time


def _find_picks(trace, phase, theo_pick_time, tolerance):
    """Search for valid picks in trace stats. Return pick time if found."""
    for pick in (p for p in trace.stats.picks if p.phase == phase):
        if _validate_pick(pick, theo_pick_time, tolerance, trace.id):
            trace.stats.arrivals[phase] = (phase, pick.time)
            return pick.time
    return None


def add_arrivals_to_trace(trace, config):
    """
    Add P and S arrival times and takeoff angles to trace.

    Uses the theoretical arrival time if no pick is available
    or if the pick is too different from the theoretical arrival.
    """
    tolerance = config.p_arrival_tolerance
    for phase in 'P', 'S':
        key = f'{trace.id}_{phase}'

        # If no cache is available, compute travel_time and takeoff_angle
        try:
            travel_time, takeoff_angle, method =\
                _wave_arrival(trace, phase, config)
            theo_pick_time = _get_theo_pick_time(trace, travel_time)
            pick_time = _find_picks(trace, phase, theo_pick_time, tolerance)
        except Exception as msg:
            for line in str(msg).splitlines():
                logger.warning(line)
            continue
        if pick_time is not None:
            logger.info(f'{trace.id}: found {phase} pick')
            travel_time = \
                _travel_time_from_pick(trace, pick_time) or travel_time
            pick_phase = phase
        else:
            logger.info(f'{trace.id}: no {phase} pick found')
            if theo_pick_time is None:
                continue
            logger.info(
                f'{trace.id}: using theoretical {phase} pick from {method}')
            pick_time = theo_pick_time
            pick_phase = f'{phase}theo'
        if config.rp_from_focal_mechanism:
            logger.info(
                f'{trace.id}: {phase} takeoff angle: {takeoff_angle:.1f} '
                f'computed from {method}')
        add_arrivals_to_trace.pick_cache[key] =\
            trace.stats.arrivals[phase] = (pick_phase, pick_time)
        add_arrivals_to_trace.travel_time_cache[key] =\
            trace.stats.travel_times[phase] = travel_time
        add_arrivals_to_trace.angle_cache[key] =\
            trace.stats.takeoff_angles[phase] = takeoff_angle
add_arrivals_to_trace.pick_cache = {}  #noqa
add_arrivals_to_trace.travel_time_cache = {}  #noqa
add_arrivals_to_trace.angle_cache = {}  #noqa
