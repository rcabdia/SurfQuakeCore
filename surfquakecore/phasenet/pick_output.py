# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: pick_output.py
# Program: surfQuake & ISP
# Date: November 2026
# Purpose: Common pick table + writers shared by all pickers (PhaseNet, ccpick).
#          No TensorFlow import, so any picker can use it.
# Author: Roberto Cabieces & Thiago C. Junqueira
#  Email: rcabdia@roa.es
# --------------------------------------------------------------------

import os
import shutil
from datetime import timedelta

import pandas as pd
from obspy import UTCDateTime

# Columns of the pick table produced by PhasenetUtils.split_picks (same order)
PICK_COLUMNS = ['date', 'fname', 'year', 'month', 'day', 'net', 'station', 'flag', 'tt', 'date_time',
                'weight', 'amplitude', 'phase', 'channel']


def best_day(starttime, endtime):
    """UTC midnight of the day that contains most of the [starttime, endtime] interval (first day if tied)."""
    start, end = UTCDateTime(starttime), UTCDateTime(endtime)
    first_day = UTCDateTime(year=start.year, month=start.month, day=start.day)
    day, best = first_day, -1
    for k in range(int((end - first_day) // 86400) + 1):
        day_k = first_day + 86400 * k
        overlap = min(end, day_k + 86400) - max(start, day_k)
        if overlap > best:
            day, best = day_k, overlap
    return day


def ensure_24(tr):
    """Trim the trace to the UTC day that contains most of its samples (same rule as PhaseNet)."""
    day = best_day(tr.stats.starttime, tr.stats.endtime)
    tr.trim(starttime=day, endtime=day + 86400 - 1e-6, pad=False, nearest_sample=True, fill_value=0)
    return tr


def make_pick_row(net, station, channel, phase, pick_time, weight, amplitude=0.0):
    """One row of the pick table, identical in meaning to the rows built by PhasenetUtils.split_picks."""
    pick_time = UTCDateTime(pick_time)
    day = UTCDateTime(year=pick_time.year, month=pick_time.month, day=pick_time.day)
    return [day.strftime("%Y%m%d"), net + '.' + station + '.' + phase, day.year, day.month, day.day,
            net, station, 1, float(pick_time - day), pick_time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            float(weight), float(amplitude), phase, channel]


def picks_to_dataframe(rows):
    """Build the pick table with the same column types as PhasenetUtils.split_picks."""
    df = pd.DataFrame(rows, columns=PICK_COLUMNS)
    return df.astype({c: object for c in ('year', 'month', 'day', 'flag')})


def write_nlloc_format(dataframe, output, starttime=None, endtime=None):
    """
    Write a pandas DataFrame to a file in NonLinLoc Phase file format.

    Parameters:
        dataframe (pd.DataFrame): The input data with the required columns.
        output (str): Directory where nll_picks.txt is written.
        starttime (str or UTCDateTime, optional): Start time of the range.
            If str, it should follow the format "%Y-%m-%d %H:%M:%S".
        endtime (str or UTCDateTime, optional): End time of the range.
            If str, it should follow the format "%Y-%m-%d %H:%M:%S".
    """

    # Convert starttime and endtime to pandas-compatible datetime objects
    if starttime:
        starttime = pd.Timestamp(str(starttime)) if isinstance(starttime, UTCDateTime) else pd.Timestamp(starttime)
    if endtime:
        endtime = pd.Timestamp(str(endtime)) if isinstance(endtime, UTCDateTime) else pd.Timestamp(endtime)

    dataframe = dataframe.copy()  # do not modify the table of the caller
    dataframe['date_time'] = pd.to_datetime(dataframe['date_time'])

    # If starttime and endtime are provided, filter the dataframe
    if starttime:
        dataframe = dataframe[dataframe['date_time'] >= starttime]
    if endtime:
        dataframe = dataframe[dataframe['date_time'] <= endtime]

    output_file = os.path.join(output, "nll_picks.txt")
    # Write to NLLoc format
    with open(output_file, 'w') as file:

        # Write the header line
        header = ("Station_name\tInstrument\tComponent\tP_phase_onset\tP_phase_descriptor\t"
                  "First_Motion\tDate\tHour_min\tSeconds\tErr\tErrMag\tCoda_duration\tAmplitude\tPeriod\n")
        file.write(header)

        for _, row in dataframe.iterrows():
            station = row['station'].ljust(6)  # Station name, left-justified, 6 chars
            instrument = "?".ljust(4)  # Placeholder for Instrument
            component = row['channel'].ljust(4)  # Placeholder for Component
            p_phase_onset = "?"  # Placeholder for P phase onset
            phase_descriptor = row['phase'].ljust(6)  # Phase descriptor (e.g., P, S)
            first_motion = "?"  # Placeholder for First Motion
            date = f"{row['date']}"  # Date in yyyymmdd format
            hour_min = f"{row['date_time'].hour:02}{row['date_time'].minute:02}"  # hhmm
            seconds = f"{row['date_time'].second + row['date_time'].microsecond / 1e6:07.4f}"  # ss.ssss
            err = "GAU"  # Error type (GAU)

            if row['weight'] > 0.95:
                weight = 2.00E-02
            elif 0.95 >= row['weight'] > 0.9:
                weight = 4.00E-02
            elif 0.9 >= row['weight'] > 0.8:
                weight = 7.00E-02
            elif 0.8 >= row['weight'] > 0.7:
                weight = 1.00E-1
            elif 0.7 >= row['weight'] > 0.6:
                weight = 1.50E-1
            else:
                weight = 5.00E-1

            err_mag = f"{weight:.2e}"  # Error magnitude in seconds
            coda_duration = "-1.00e+00"  # Placeholder for Coda duration
            amplitude = f"{row['amplitude']:.2e}"  # Amplitude
            period = "-1.00e+00"  # Placeholder for Period

            # Construct the line
            line = (
                f"{station} {instrument} {component} {p_phase_onset} {phase_descriptor} {first_motion} "
                f"{date} {hour_min} {seconds} {err} {err_mag} {coda_duration} {amplitude} {period}\n"
            )
            file.write(line)

        # Add a blank line at the end for NLLoc format compliance
        file.write("\n")


def convert2real(picks, pick_dir: str, clean_output_folder=False):
    """
    :param picks: picks is output from method split_picks in mseedutils
    :param pick_dir: directory outpur where phases are storaged
    :return:
    """

    dates = picks['date'].unique()

    if clean_output_folder:
        for path in os.listdir(pick_dir):
            fpath = os.path.join(pick_dir, path)

            if os.path.isfile(fpath) or os.path.islink(fpath):
                os.unlink(fpath)
            else:
                shutil.rmtree(fpath)

    for date in dates:
        pickpath = os.path.join(pick_dir, date)

        os.makedirs(pickpath, exist_ok=True)

    for (date, fname), savedata in picks.groupby(['date', 'fname'], sort=False):
        savedata = savedata.sort_values('tt')
        with open(os.path.join(pick_dir, date, fname + '.txt'), 'w') as f:
            f.write(savedata.to_string(header=False, index=False, columns=['tt', 'weight', 'amplitude']))


def save_original_picks(original_picks, original_p_dir):
    """

    :param original_picks: picking output from phasenet (method split_picks in mseedutils)
    :param original_p_dir: output to storage original_picks
    :return:
    """

    pick_path = os.path.join(original_p_dir, "original_picks.csv")
    original_picks.to_csv(pick_path, index=False)


def print_summary(picks, n_files, output_dir, elapsed, title="PhaseNet"):
    """
    Print a short report at the end of the run.

    :param picks: output of split_picks
    :param n_files: number of files that were processed
    :param output_dir: directory where the results were saved
    :param elapsed: total time in seconds
    :param title: name of the picker shown in the header
    """
    line = "=" * 60
    print(f"\n{line}\n {title} summary\n{line}")
    print(f" Files processed     : {n_files}")
    print(f" Stations with picks : {len(picks[['net', 'station']].drop_duplicates())}")
    print(f" P picks             : {int((picks['phase'] == 'P').sum())}")
    print(f" S picks             : {int((picks['phase'] == 'S').sum())}")
    print(f" Total time          : {timedelta(seconds=int(elapsed))}")
    print(f" Results saved in    : {output_dir}")
    print("   original_picks.csv, nll_picks.txt and one folder per day (REAL format)")
    print(line)
