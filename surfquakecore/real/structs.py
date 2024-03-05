# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: metadata_manager.py
# Program: surfQuake & ISP
# Date: January 2024
# Purpose: Dataclass Structure for Associator
# Author: Roberto Cabieces & Thiago C. Junqueira
#  Email: rcabdia@roa.es
# --------------------------------------------------------------------



from __future__ import annotations
from dataclasses import dataclass, field, fields
from datetime import datetime, date
from typing import List, Optional


@dataclass
class EventsInfo:
    events_date: date
    events_location: List[EventLocation] = field(default_factory=list)

    def __str__(self):
        return "\n".join([f"{el}" for el in self.events_location])


@dataclass
class EventLocation:
    event_id: int
    date: datetime
    origin_time: float  # secs
    residual: float  # secs
    lat: float
    long: float
    depth: float
    magnitude: float
    var_magnitude: float
    p_picks: int
    s_picks: int
    total_picks: int
    stations_with_p_s_picks: int
    station_gap: float
    phases: List[PhaseLocation] = field(default_factory=list)

    @classmethod
    def from_real_str(cls, line: str):
        """
        Convert a string phase line from real to this data struct.

        :param line: A string with the given format:
        882 2016 10 15 23:56:30.773 86190.773 0.2452 42.8201 13.2761 0.00 1.459 0.303 35 8 43 6 47.12
        :return:
        """
        try:
            values = line.strip().split()
            if len(values) != 17:
                raise ValueError(f"The data {line} can't be converted to EventLocation")

            return cls(
                event_id=int(values[0]),
                date=datetime.strptime(f"{values[3]}.{values[2]}.{values[1]} {values[4]}", '%d.%m.%Y %H:%M:%S.%f'),
                origin_time=float(values[5]),
                residual=float(values[6]),
                lat=float(values[7]),
                long=float(values[8]),
                depth=float(values[9]),
                magnitude=float(values[10]),
                var_magnitude=float(values[11]),
                p_picks=int(values[12]),
                s_picks=int(values[13]),
                total_picks=int(values[14]),
                stations_with_p_s_picks=int(values[15]),
                station_gap=float(values[16]),
            )
        except (ValueError, IndexError) as e:
            print(f"Error: {e}. Input line: {line}")
            return None  # Returning None or another sentinel value to indicate an error

        except Exception as e:
            print(f"Unexpected error: {e}. Input line: {line}")
            return None  # Returning None or another sentinel value to indicate an error

    def __str__(self):
        values = []
        phases_str = ''
        for c in fields(self):
            if c.name == 'date':
                values.append(self.date.strftime('%Y  %m  %d %H:%M:%S.%f'))
            elif c.name == 'phases':
                phases_str = "\n".join([f"{p}" for p in self.phases])
            else:
                values.append(f"{self.__getattribute__(c.name)}")

        if phases_str:
            return f"{'  '.join(values)}\n{phases_str}"
        else:
            return f"{'  '.join(values)}"


@dataclass
class PhaseLocation:

    network: str
    station: str
    phase_name: str
    absolute_travel_time: float  # secs
    travel_time_to_event: float  # secs
    phase_amplitude: float  # millimeter
    individual_phase_residual: float  # secs
    weight: float
    azimuth: float

    @classmethod
    def from_real_str(cls, line: str):
        """
        Convert a string phase line from real to this data struct.

        :param line: A string with the given format: IV T1201 P 86194.45 3.6775 8.31e-002 0.3733 19.4989 6.5144
        :return:
        """
        values = line.strip().split()
        if len(values) != 9:
            raise ValueError(f"The data {line} can't be converted to PhaseLocation")

        return cls(
            network=str(values[0]),
            station=str(values[1]),
            phase_name=str(values[2]),
            absolute_travel_time=float(values[3]),
            travel_time_to_event=float(values[4]),
            phase_amplitude=float(values[5]),
            individual_phase_residual=float(values[6]),
            weight=float(values[7]),
            azimuth=float(values[8]),
        )

    def __str__(self):
        return "  ".join([f"{self.__getattribute__(c.name)}" for c in fields(self)])


@dataclass
class Station:
    lon: float
    lat: float
    network: str
    name: str
    component: str
    elevation: float  # km


@dataclass
class RealD:
    year: int
    month: int
    day: int
    lat_center: float

    def __str__(self):
        return f"{self.year}/{self.month}/{self.day}/{self.lat_center}"

    @classmethod
    def from_station_file(cls, filepath: str):
        pass

    def to_date(self) -> date:
        return datetime.strptime(f"{self.day}.{self.month}.{self.year}", '%d.%m.%Y').date()


@dataclass
class RealR:
    search_range_h: float  # degree
    search_range_depth: float  # km
    search_grid_size: float  # degree
    search_grid_depth: float  # km
    event_time_window: float  # secs
    gap: int = 360  # degree
    gcarc0: int = 180  # degree
    lat_ref_0: Optional[float] = None  # degree
    long_ref_0: Optional[float] = None  # degree

    def __str__(self):
        v = f"{self.search_range_h}/{self.search_range_depth}/" \
               f"{self.search_grid_size}/{self.search_grid_depth}/" \
               f"{self.event_time_window}/{self.gap}/{self.gcarc0}"

        # add latitude and longitude reference
        if (self.lat_ref_0 is not None) and (self.long_ref_0 is not None):
            v += f"/{self.lat_ref_0}/{self.long_ref_0}"

        return v


@dataclass
class RealG:
    horizontal_range: float  # degree
    vertical_range: float  # km
    grid_size_h: float  # degree
    grid_size_depth: float  # km

    def __str__(self):
        return f"{self.horizontal_range}/{self.vertical_range}/{self.grid_size_h}/{self.grid_size_depth}"


@dataclass
class RealV:
    average_p_velocity: float  # km/s
    average_s_velocity: float  # km/s
    shallow_p_velocity: Optional[float] = None  # km/s
    shallow_s_velocity: Optional[float] = None  # km/s
    station_elevation_correction: int = 0

    def __str__(self):
        v = f"{self.average_p_velocity}/{self.average_s_velocity}"

        # add optional parameters
        if (self.shallow_p_velocity is not None) and (self.shallow_s_velocity is not None):
            v += f"/{self.shallow_p_velocity}/{self.shallow_s_velocity}/{self.station_elevation_correction}"

        return v


@dataclass
class RealS:
    threshold_number_p_picks: int
    threshold_number_s_picks: int
    threshold_number_total_picks: int
    number_of_stations_recorded_p_and_s: int
    standard_deviation_threshold: float
    time_threshold_p_s_separation: float  # seconds
    nrt: float
    drt: float
    nxd: Optional[float] = None
    tolerance_multiplier: Optional[float] = None

    def __str__(self):
        v = f"{self.threshold_number_p_picks}/{self.threshold_number_s_picks}/" \
            f"{self.threshold_number_total_picks}/{self.number_of_stations_recorded_p_and_s}/" \
            f"{self.standard_deviation_threshold}/{self.time_threshold_p_s_separation}/" \
            f"{self.nrt}/{self.drt}"

        # add latitude and longitude reference
        if (self.nxd is not None) and (self.tolerance_multiplier is not None):
            v += f"/{self.nxd}/{self.tolerance_multiplier}"

        return v


@dataclass
class TimeTravelTable:
    g_dist: float
    depth: float
    p_time: float
    s_time: float
    p_ray_p: float
    s_ray_p: float
    p_hslow: float
    s_hslow: float
    p_phase: str
    s_phase: str