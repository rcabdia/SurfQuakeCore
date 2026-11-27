import os
import numpy as np
import datetime
import pandas as pd
from obspy import UTCDateTime


class EventLocation:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class PhaseLocation:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class ConversionUtils:

    @staticmethod
    def previewCatalog(realfile):
        with open(realfile, 'r') as f:
            data = f.read()
            f.close()

        data = data.split('[EventLocation')[1:]
        data_save = []
        catalog_preview = {}
        lats = []
        longs = []
        dates = []
        for i in np.arange(0, len(data)):
            data[i] = data[i].split('EventLocation')
            try:
                for j in np.arange(0, len(data[i])):
                    if j == len(data[i]) - 1:
                        data[i][j] = data[i][j][:-1]
                        data_save.append(eval('EventLocation' + data[i][j]))
                    else:
                        data_save.append(eval('EventLocation' + data[i][j])[0])
            except:
                pass

        for i in np.arange(0, len(data_save)):
            #print(data_save[i].date, data_save[i].lat, data_save[i].long, data_save[i].depth, data_save[i].var_magnitude)
            dates.append(data_save[i].date)
            lats.append(data_save[i].lat)
            longs.append(data_save[i].long)

        catalog_preview["date"] = dates
        catalog_preview["lats"] = lats
        catalog_preview["longs"] = longs

        return catalog_preview
    @staticmethod
    def read_polarities(polarity_files, component="Z"):
        """
        Read the first-motion polarities of pick files in NonLinLoc format
        (for example, the output of 'surfquake polarity').

        polarity_files: one path or a list of paths. Files that cannot be read are skipped.
        Returns {station: [(UTCDateTime, 'U' or 'D'), ...]} with the P picks of the
        vertical channels that have a polarity.
        """
        polarities = {}
        if not polarity_files:
            return polarities
        if isinstance(polarity_files, str):
            polarity_files = [polarity_files]
        for polarity_file in polarity_files:
            if not os.path.isfile(polarity_file):
                continue
            try:
                df = pd.read_csv(polarity_file, sep=r"\s+")
                rows = df[["Station_name", "Component", "P_phase_descriptor", "First_Motion",
                           "Date", "Hour_min", "Seconds"]]
            except Exception:
                continue  # not a pick file in NonLinLoc format
            ConversionUtils._add_polarities(polarities, rows, component)
        n = sum(len(v) for v in polarities.values())
        if n > 0:
            print(f"[INFO] {n} P-wave polarities read from {len(polarity_files)} pick file(s)")
        return polarities

    @staticmethod
    def _add_polarities(polarities, df, component="Z"):
        for _, row in df.iterrows():
            fm = str(row["First_Motion"])
            if fm not in ("U", "D") or not str(row["P_phase_descriptor"]).startswith("P"):
                continue
            if not str(row["Component"]).endswith(component):
                continue
            hh, mm = divmod(int(row["Hour_min"]), 100)
            t = UTCDateTime(str(int(row["Date"]))) + hh * 3600 + mm * 60 + float(row["Seconds"])
            polarities.setdefault(str(row["Station_name"]).strip(), []).append((t, fm))
        return polarities

    @staticmethod
    def real2nll(realfile, nllfile, polarity_files=None, tol=0.05):
        """
        Convert the REAL output into a NonLinLoc pick file (one block per event).
        If polarity_files are given, the first motion (U/D) of each associated P pick is
        copied from them: same station, vertical channel, time within tol seconds.
        """
        polarities = ConversionUtils.read_polarities(polarity_files)
        with open(realfile, 'r') as f:
            data = f.read()
            f.close()

        data_save = []
        data = data.split('[EventLocation')[1:]

        for i in np.arange(0, len(data)):
            data[i] = data[i].split('EventLocation')
            try:
                for j in np.arange(0, len(data[i])):
                    if j == len(data[i]) - 1:
                        data[i][j] = data[i][j][:-1]
                        data_save.append(eval('EventLocation' + data[i][j]))
                    else:
                        data_save.append(eval('EventLocation' + data[i][j])[0])
            except:
                pass

        with open(nllfile, 'w') as g:
            for i in np.arange(0, len(data_save)):
                g.write("Station_name\tInstrument\tComponent\tP_phase_onset\tP_phase_descriptor\tFirst_Motion\tDate\t"
                        "Hourmin\t""Seconds\tGAU\tErr\tCoda_duration\tAmplitude\tPeriod\n")

                for j in np.arange(0, len(data_save[i].phases)):

                    if data_save[i].phases[j].weight > 0.95:
                        weight = 2.00E-02
                    elif 0.95 >= data_save[i].phases[j].weight > 0.9:
                        weight = 4.00E-02
                    elif 0.9 >= data_save[i].phases[j].weight > 0.8:
                        weight = 7.00E-02
                    elif 0.8 >= data_save[i].phases[j].weight > 0.7:
                        weight = 1.50E-01
                    elif 0.7 >= data_save[i].phases[j].weight > 0.6:
                        weight = 1.00E-01
                    else:
                        weight = 5.00E-01

                    _time = datetime.datetime.utcfromtimestamp(data_save[i].phases[j].absolute_travel_time)

                    hour = f'{_time.hour:02}'
                    minute = f'{_time.minute:02}'
                    month = f'{data_save[i].date.month:02}'
                    day = f'{data_save[i].date.day:02}'
                    # First motion from the polarity file: same station, P phase, time within tol
                    fm = "?"
                    sta = data_save[i].phases[j].station
                    if data_save[i].phases[j].phase_name.startswith("P") and sta in polarities:
                        d = data_save[i].date
                        t_pick = UTCDateTime(d.year, d.month, d.day) + data_save[i].phases[j].absolute_travel_time
                        t_pol, fm_pol = min(polarities[sta], key=lambda x: abs(x[0] - t_pick))
                        if abs(t_pol - t_pick) <= tol:
                            fm = fm_pol

                    g.write(f"{data_save[i].phases[j].station}\t?\t?\t?\t{data_save[i].phases[j].phase_name}\t{fm}\t"
                            f"{str(data_save[i].date.year) + month + day}\t"
                            f"{hour + minute}\t"
                            f"{_time.second + _time.microsecond / 1000000:06.3f}\t"
                            f"GAU\t{weight:.2E}\t"
                            f"-1.00E+00\t{data_save[i].phases[j].phase_amplitude}\t-1.00E+00\n")

                g.write("\n")

            g.close()