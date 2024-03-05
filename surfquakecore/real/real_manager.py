# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: real_manager.py
# Program: surfQuake & ISP
# Date: January 2024
# Purpose: Manage Associator
# Author: Roberto Cabieces & Thiago C. Junqueira
#  Email: rcabdia@roa.es
# --------------------------------------------------------------------


import itertools
import math
import os.path
import shutil
import subprocess
import time
import warnings
from typing import Optional, Union, List, Dict, Tuple
from ..binaries import BINARY_REAL_FILE
from .structs import RealD, RealR, RealG, RealV, RealS, Station, EventLocation, PhaseLocation, \
    EventsInfo
from ..utils.subprocess_utils import exc_cmd
import stat

# Time file is based on https://github.com/Dal-mzhang/LOC-FLOW/blob/main/LOCFLOW-CookBook.pdf
# reference for structs: https://github.com/Dal-mzhang/REAL/blob/master/REAL_userguide_July2021.pdf

class RealManager:

    REAL_BIN: str = BINARY_REAL_FILE

    st = os.stat(BINARY_REAL_FILE)
    os.chmod(BINARY_REAL_FILE, st.st_mode | stat.S_IEXEC)
    DEGREE_TO_KM = 111.19

    def __init__(self, pick_dir: str, station_file: str, time_travel_table_file: str, out_data_dir: str, **kwargs):
        """
        Handle configuration to run real

        :param pick_dir: A director with valid pick directories from Phasenet picks. The subdirectories
            should have names with the pick date i.e: 20161014 (yyyymmdd)
        :param station_file: A valid station file
        :param time_travel_table_file: A valid time travel table file.
        """
        self.__dict__.update(kwargs)

        self._root_pick_dir = pick_dir
        self._station_file = station_file
        self._time_travel_table_file = time_travel_table_file
        self._out_data_dir = out_data_dir
        self._validate_paths()

        self._real_d: Optional[RealD] = None

        self._real_r: Optional[RealR] = RealR(self.gridSearchParamHorizontalRange, self.DepthSearchParamHorizontalRange,
                                              self.HorizontalGridSize, self.DepthGridSize, self.EventTimeW)

        self._real_g: Optional[RealG] = RealG(self.TTHorizontalRange, self.TTDepthRange, self.TTHorizontalGridSize,
                                              self.TTDepthGridSize)

        self._real_v: Optional[RealV] = RealV(6.2, 3.3) # Fixed Velocity Vp and Vs
        sum_picks = self.ThresholdPwave+ self.ThresholdSwave
        self._real_s: Optional[RealS] = RealS(self.ThresholdPwave, self.ThresholdSwave, sum_picks,
                                              self.number_stations_picks, 5, 0, 2, 0.01, 6, 10)

        self._real_out_files_names = [
            "catalog_sel.txt",
            "hypolocSA.dat",
            "hypophase.dat",
            "phase_sel.txt",
        ]

        self.latitude_center: float = 0.

        self._stations: Dict[str: Station] = {}

        self._setup_stations()

        self._phase_sel_data: Dict[str: EventsInfo] = {}

        self._pick_folders: List[str] = []

    def __iter__(self):
        self._pick_folders = [f for f in os.listdir(self._root_pick_dir) if self.is_pick_dir(f)]
        return self

    def __next__(self) -> EventsInfo:
        if not self._pick_folders:
            raise StopIteration
        pick_dir = os.path.join(self._root_pick_dir, self._pick_folders.pop(0))
        return self._run_real(pick_dir)

    def is_pick_dir(self, dir_name: str) -> bool:
        if not os.path.isdir(os.path.join(self._root_pick_dir, dir_name)):
            return False
        try:
            if len(dir_name) == 8:
                yyyy, mm, dd = self.parse_pick_dir_to_date(dir_name)
                return yyyy > 1700 and 12 >= mm > 0 and 31 >= dd > 0
            return False
        except ValueError:
            return False

    @classmethod
    def parse_pick_dir_to_date(cls, dir_name: str) -> Tuple[int, int, int]:
        """
        Parse the directory name to date: year, month, day.

        :param dir_name: The folder name from phasenet, i.e: 20220508
        :return: A Tuple of year, month, day
        """
        return int(dir_name[:4]), int(dir_name[4:6]), int(dir_name[6:8])

    @classmethod
    def parse_from_date_to_pick_dir(cls, year: int, month: int, day: int):
        return f"{year:04}{month:02}{day:02}"

    @property
    def stations(self):
        """

        :return:
        """
        return self._stations

    def _set_real_d(self, day: int, month: int, year: int, lat_center: Optional[float] = None):
        """
        Sets a new REALD structure.

        :param day:
        :param month:
        :param year:
        :param lat_center:
        :return:
        """
        if day < 1 or day > 31:
            raise ValueError(f"Invalid day: it must be in range of 1 - 31 you gave {day}.")

        if month < 1 or month > 12:
            raise ValueError(f"Invalid month: it must be in range of 1 - 12 you gave {month}.")

        if year < 0:
            raise ValueError(f"Invalid year: it must be bigger than 0 you gave {year}.")

        self.latitude_center = lat_center or self.latitude_center
        self._real_d = RealD(day=day, month=month, year=year, lat_center=self.latitude_center)

    def _validate_paths(self):
        if not os.path.isdir(self._root_pick_dir):
            raise FileNotFoundError(f"Can't find pick dir at {self._root_pick_dir}")

        if not os.path.isfile(self._station_file):
            raise FileNotFoundError(f"Can't find station files at {self._station_file}")

        if not os.path.isfile(self._time_travel_table_file):
            raise FileNotFoundError(f"Can't find time travel table files at {self._time_travel_table_file}")

    def _validate_configuration(self):
        if not self._real_d:
            raise AttributeError("To run real you must set a RealD object via configure_real")
        if not self._real_r:
            raise AttributeError("To run real you must set a RealR object via configure_real")
        if not self._real_g:
            raise AttributeError("To run real you must set a RealG object via configure_real")
        if not self._real_v:
            raise AttributeError("To run real you must set a RealV object via configure_real")
        if not self._real_s:
            raise AttributeError("To run real you must set a RealS object via configure_real")

    def _setup_stations(self):
        with open(self._station_file, 'r') as f:
            # expect to be lon, lat, network, station, component, elevation
            while v := f.readline():
                lon, lat, network, station, component, elevation = v.strip().split()
                self._stations.setdefault(
                    station,
                    Station(float(lon), float(lat), network, station, component, -float(elevation))
                )

    @classmethod
    def _compute_dist(cls, lat: float, delta_lat: float, delta_lon: float, delta_depth: float):
        return math.sqrt(
            (delta_lat * cls.DEGREE_TO_KM) ** 2
            + (delta_lon * (math.cos(lat * math.pi / 180.) * cls.DEGREE_TO_KM)) ** 2
            + delta_depth ** 2
        )

    def compute_t_dist(self):
        dist_file = os.path.join(self._out_data_dir, 't_dist.dat')
        events: List[EventLocation] = \
            list(itertools.chain.from_iterable(
                [events_info.events_location for events_info in self._phase_sel_data.values()]
            ))
        with open(dist_file, 'w') as tdf:
            for event in events:
                for phase in event.phases:
                    # phase
                    station: Station = self.stations.get(phase.station, None)
                    if station and event:
                        # Output travel time and hypocenter distance.
                        delta_lat = event.lat - station.lat
                        delta_lon = event.long - station.lon
                        delta_depth = event.depth - station.elevation
                        dist = self._compute_dist(lat=event.lat,
                                                  delta_lat=delta_lat, delta_lon=delta_lon, delta_depth=delta_depth)

                        mark = 1 if phase.phase_name == "P" else 2
                        tdf.write(f"{phase.travel_time_to_event} {dist} {mark}\n")

    def configure_real(self, *args: Union[RealD, RealR, RealG, RealV, RealS]):
        """
        Set configuration to run real.

        :param args: Any of the valid dataclasses.
        :return:
        """
        for v in args:
            if isinstance(v, RealD):
                self._real_d = v
                self.latitude_center = v.lat_center
            elif isinstance(v, RealR):
                self._real_r = v
            elif isinstance(v, RealG):
                self._real_g = v
            elif isinstance(v, RealV):
                self._real_v = v
            elif isinstance(v, RealS):
                self._real_s = v
            else:
                raise AttributeError(f"Invalid type {type(v)}. Arguments must have a valid type of: "
                                     f"RealD, RealR, RealG, RealV, RealS ")

    def _clear_out_dir(self):
        for f in self._real_out_files_names:
            f_path = os.path.join(self._out_data_dir, f)
            if os.path.isfile(f_path):
                os.remove(f_path)

    def _move_real_output_to_default(self):
        """
        Move files from real_out_dir to a default location given by OUT_DIR

        :return:
        """
        current_out_real_dir = os.getcwd()  # whatever directory you are running real
        out_file_locations: List[str] = []
        for f in self._real_out_files_names:
            src_f_path = os.path.join(current_out_real_dir, f)
            if os.path.isfile(src_f_path):
                out = shutil.move(src_f_path, os.path.join(self._out_data_dir, f))
                out_file_locations.append(out)

    def _buffer_data_from_real(self):
        filename = os.path.join(self._out_data_dir, 'phase_sel.txt')
        events_info = EventsInfo(events_date=self._real_d.to_date())
        # Event line: num, year, month, day, time, origin time, residuals, lat, lon, depth, mag, mag var,
        # n p picks, n s picks, total picks, number of stations with P and S, station gap

        # Phase line: network, station, phase name, absolute travel time, relative travel time, phase
        # amplitude, phase residual, weight, azimuth
        with open(filename, 'r') as f:
            while line := f.readline().strip():
                v = line.split()
                if v[0].isnumeric():
                    # event
                    event_location = EventLocation.from_real_str(line)
                elif isinstance(event_location, EventLocation):
                    phase = PhaseLocation.from_real_str(line)
                    # add phase to its event parent
                    event_location.phases.append(phase)
                    # if count is equal to total picks event is done.
                    if event_location.total_picks == len(event_location.phases):
                        events_info.events_location.append(event_location)

        self._phase_sel_data[f"{self._real_d}"] = events_info
        return events_info

    def save(self) -> str:
        filename = os.path.join(self._out_data_dir, 'phase_sel_total.txt')
        with open(filename, 'w') as f:
            for events in self._phase_sel_data.values():
                events: EventsInfo
                f.write(f"{events.events_location}")

        return filename

    def _run_real(self, pick_dir: str, lat_center: Optional[float] = None):
        """
        Runs real for this directory.

        Important: It's expect the user to have a folder named 'year month day' inside pick_dir

        :param lat_center:
        :return:
        """

        if not os.path.isdir(pick_dir):
            raise FileNotFoundError(f'Pick folder {pick_dir} not found for the given date.')

        year, month, day = self.parse_pick_dir_to_date(os.path.basename(pick_dir))
        self._set_real_d(day=day, month=month, year=year, lat_center=lat_center)
        self._validate_configuration()

        self._clear_out_dir()
        print("Running real. This may take a while...")
        t0 = time.time()
        try:
            exc_cmd(f"{self.REAL_BIN} -D{self._real_d} -R{self._real_r} -G{self._real_g} -S{self._real_s} "
                    f"-V{self._real_v} {self._station_file} {pick_dir} {self._time_travel_table_file}",
                    shell=True, timeout=36000)
        except subprocess.SubprocessError as e:
            warnings.warn(f"{e}", category=RuntimeWarning)
        finally:
            self._move_real_output_to_default()  # move files to a specific location.
            events_info = self._buffer_data_from_real()

        tf = time.time()
        print(f"Finished after: {tf - t0} seconds")
        return events_info

    def run_real(self, day: int, month: int, year: int, lat_center: Optional[float] = None):
        """
        Runs real for this date.

        Important: It's expect the user to have a folder named 'year month day' inside pick_dir

        :param day:
        :param month:
        :param year:
        :param lat_center:
        :return:
        """

        pick_base_dir = self.parse_from_date_to_pick_dir(year=year, month=month, day=day)
        pick_dir = os.path.join(self._root_pick_dir, pick_base_dir)
        return self._run_real(pick_dir, lat_center=lat_center)