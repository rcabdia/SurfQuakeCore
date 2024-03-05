# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: Structures/structures.py
# Program: surfQuake & ISP
# Date: January 2024
# Purpose: Trace Structures
# Author: Roberto Cabieces & Thiago C. Junqueira
#  Email: rcabdia@roa.es
# --------------------------------------------------------------------


from dataclasses import dataclass
from datetime import datetime
from surfquakecore.Structures.obspy_stats_keys import ObspyStatsKeys
from surfquakecore.utils import BaseDataClass


@dataclass
class TracerStats(BaseDataClass):
    """
    Class that holds a structure of mseed metadata.

    Fields:
        * Network = (string) network name.
        * Station = (string) station name.
        * Channel = (string) channel name.
        * StartTime = (UTCDateTime) start datetime.
        * EndTime = (UTCDateTime) stop datetime.
        * Location = (string) The location.
        * Sampling_rate = (float) Sample rate in hertz.
        * Delta = (float) Delta time.
        * Npts = (int) Number of points.
        * Calib = (float) Calibration of instrument.
        * Mseed = (dict) Information of mseed file.
        * Format = (string) File format
        * Dataquality = (string)
        * numsamples = 236
        * samplecnt = (float)
        * sampletype: (string)
    """
    Network: str
    Station: str
    Channel: str
    StartTime: datetime
    EndTime: datetime
    Location: str
    Sampling_rate: float
    Delta: float
    Npts: int
    Calib: float
    Mseed: dict
    Format: str
    Dataquality: str
    numsamples: float
    samplecnt:float
    sampletype: str

    # noinspection PyTypeChecker
    @classmethod
    def from_dict(cls, dto: dict):
        dto.pop("processing")
        dto["Format"] = dto.pop(ObspyStatsKeys.FORMAT, "mseed")
        return super().from_dict(dto)


@dataclass
class StationCoordinates(BaseDataClass):
    """
        Class that holds a structure for the picker. This is used for re-plot the pickers keeping all
        necessary information in memory.

        Fields:
            * Latitude = (float)

        """

    Latitude: float
    Longitude: float
    Elevation: float
    Local_depth: float
