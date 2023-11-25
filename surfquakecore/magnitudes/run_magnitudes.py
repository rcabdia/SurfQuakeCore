import gc
import os
from obspy import read, read_events, UTCDateTime, Stream
from surfquakecore.utils.obspy_utils import MseedUtil
from surfquakecore.magnitudes.structures import SoursceSpecOptions
from sourcespec.ssp_setup import configure
from sourcespec.ssp_read_traces import read_traces
from surfquakecore.magnitudes.ssp_process_traces_mod import process_traces
from sourcespec.ssp_build_spectra import build_spectra
from sourcespec.ssp_plot_traces import plot_traces
from sourcespec.ssp_inversion import spectral_inversion
from sourcespec.ssp_radiated_energy import radiated_energy
from sourcespec.ssp_local_magnitude import local_magnitude
from sourcespec.ssp_summary_statistics import compute_summary_statistics
from sourcespec.ssp_output import write_output
from sourcespec.ssp_residuals import spectral_residuals
from sourcespec.ssp_plot_spectra import plot_spectra
from sourcespec.ssp_plot_stacked_spectra import plot_stacked_spectra
from sourcespec.ssp_plot_params_stats import box_plots
from sourcespec.ssp_plot_stations import plot_stations
from sourcespec.ssp_html_report import html_report
from surfquakecore.magnitudes.ssp_setup_mod import setup_logging

class Automag:

    def __init__(self, project, locations_directory, inventory_path, config_path, output_directory, scale):

        self.project = project
        self.locations_directory = locations_directory
        self.inventory_path = inventory_path
        self.config_path = config_path
        self.output_directory = output_directory
        self.sacale = scale
        self._check_folders()


    def _check_folders(self):
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
    def get_now_files(self, date):

        selection = [".", ".", "."]

        _, self.files_path = MseedUtil.filter_project_keys(self.project, net=selection[0], station=selection[1],
                                                       channel=selection[2])
        start = date.split(".")
        start = UTCDateTime(year=int(start[1]), julday=int(start[0]), hour=00, minute=00, second=00)+1
        end = start+(24*3600-2)
        self.files_path = MseedUtil.filter_time(list_files=self.files_path, starttime=start, endtime=end)
        print(self.files_path)

    def filter_station(self, station):

        filtered_list = []

        for file in self.files_path:
            header = read(file, headlonly=True)
            sta = header[0].stats.station
            if station == sta:
                filtered_list.append(file)

        return filtered_list

    def scan_folder(self):
        obsfiles1 = []
        dates = {}
        for top_dir, _, files in os.walk(self.locations_directory):

            for file in files:
                try:
                    file_hyp = os.path.join(top_dir, file)
                    cat = read_events(file_hyp, format="NLLOC_HYP")
                    ev = cat[0]
                    date = ev.origins[0]["time"]
                    date = str(date.julday) + "." + str(date.year)

                    obsfiles1.append(file_hyp)
                    if date not in dates:
                        dates[date] = [file_hyp]
                    else:
                        dates[date].append(file_hyp)
                except:
                    pass

        self.dates=dates

    def scan_from_origin(self, origin):

        self.date = origin["time"]

    def _get_stations(self, arrivals):
        stations = []
        for pick in arrivals:
            if pick.station not in stations:
                stations.append(pick.station)

        return stations

    def __cut_signal_wise(self, st,  origin_time, regional=True):

        all_traces = []
        st.merge()

        if regional:
            dt_noise = 10
            dt_signal = 10 * 60
        else:
            dt_noise = 10
            dt_signal = 2700

        start = origin_time - dt_noise
        end = origin_time + dt_signal

        for tr in st:

            tr.trim(starttime=start, endtime=end)
            all_traces.append(tr)
            # TODO: It is not still checked the fill_gaps functionality
            #tr = cls.fill_gaps(tr)
        st = Stream(traces=all_traces)
        return st

    def __run_core_source(self, event,  id_name, focal_parameters):
        options = SoursceSpecOptions(config_file=self.config_path, evid=None, evname=id_name, hypo_file=None,
                                     outdir=self.output_directory, pick_file=None, qml_file=event, run_id="",
                                     sampleconf=False, station=None,
                                     station_metadata=self.inventory_path, trace_path=self.files_path, updateconf=None)

        # Setup stage
        config = configure(options, progname='source_spec')
        setup_logging(config)
        st = read_traces(config)
        st_trim = self.__cut_signal_wise(st, origin_time=focal_parameters[0])

        # Deconvolve, filter, cut traces:
        proc_st = process_traces(config, st_trim)

        # Build spectra (amplitude in magnitude units)
        spec_st, specnoise_st, weight_st = build_spectra(config, proc_st)
        plot_traces(config, proc_st)

        # Spectral inversion
        sspec_output = spectral_inversion(config, spec_st, weight_st)

        # Radiated energy
        radiated_energy(config, spec_st, specnoise_st, sspec_output)

        # Local magnitude
        if config.compute_local_magnitude:
            local_magnitude(config, st, proc_st, sspec_output)

        # Compute summary statistics from station spectral parameters
        compute_summary_statistics(config, sspec_output)

        # Save output
        write_output(config, sspec_output)

        # Save residuals
        spectral_residuals(config, spec_st, sspec_output)

        # Plotting
        plot_spectra(config, spec_st, specnoise_st, plot_type='regular')
        plot_spectra(config, weight_st, plot_type='weight')
        plot_stacked_spectra(config, spec_st, sspec_output)
        box_plots(config, sspec_output)
        if config.plot_station_map:
            plot_stations(config, sspec_output)

        if config.html_report:
            html_report(config, sspec_output)

        del options
        del config
        del st
        del st_trim
        gc.collect()

    def estimate_source_parameters(self):

        self.scan_folder()
        for date in self.dates.keys():
            events = self.dates[date]

            self.get_now_files(date)
            for event in events:
                print(event)
                cat = read_events(event)
                focal_parameters = [cat[0].origins[0]["time"], cat[0].origins[0]["latitude"],
                                    cat[0].origins[0]["longitude"],
                                    cat[0].origins[0]["depth"] * 1E-3]
                try:
                    run_id_name = os.path.basename(event)
                    self.__run_core_source(event, run_id_name, focal_parameters)
                except:
                    pass