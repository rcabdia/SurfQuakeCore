import numpy as np
import datetime


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
    def real2nll(realfile, nllfile):
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
                    g.write(f"{data_save[i].phases[j].station}\t?\t?\t?\t{data_save[i].phases[j].phase_name}\t?\t"
                            f"{str(data_save[i].date.year) + month + day}\t"
                            f"{hour + minute}\t"
                            f"{_time.second + _time.microsecond / 1000000:06.3f}\t"
                            f"GAU\t{weight:.2E}\t"
                            f"-1.00E+00\t{data_save[i].phases[j].phase_amplitude}\t-1.00E+00\n")

                g.write("\n")

            g.close()