import os
from obspy import read, read_events, UTCDateTime
from surfquakecore.utils.obspy_utils import MseedUtil
from surfquakecore.magnitudes.structures import SoursceSpecOptions

class Automag:

    def __init__(self, project, locations_directory, inventory_path, config_path, output_directory):

        self.project = project
        self.locations_directory = locations_directory
        self.inventory_path = inventory_path
        self.config_path = config_path
        self.output_directory = output_directory
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

    def estimate_source_parameters(self):

        self.scan_folder()
        for date in self.dates:
            events = self.dates[date]
            #events = list(set(events))
            self.get_now_files(date)
            for event in events:
                print(event)
                try:
                    options = SoursceSpecOptions(config_file=self.config_path, evid=None, evname=None, hypo_file=None,
                        outdir=self.output_directory, pick_file=None, qml_file=event, run_id="", sampleconf=False, station=None,
                        station_metadata=self.inventory_path, trace_path=self.files_path, updateconf=None)

                    # Setup stage
                    from sourcespec.ssp_setup import (
                        configure, move_outdir, remove_old_outdir, setup_logging,
                        save_config, ssp_exit)
                    config = configure(options, progname='source_spec')
                    setup_logging(config)

                    from sourcespec.ssp_read_traces import read_traces
                    st = read_traces(config)

                    # Now that we have an evid, we can rename the outdir and the log file
                    move_outdir(config)
                    setup_logging(config, config.hypo.evid)
                    remove_old_outdir(config)

                    # Save config to out dir
                    save_config(config)
                    # Deconvolve, filter, cut traces:
                    from sourcespec.ssp_process_traces import process_traces
                    proc_st = process_traces(config, st)

                    # Build spectra (amplitude in magnitude units)
                    from sourcespec.ssp_build_spectra import build_spectra
                    spec_st, specnoise_st, weight_st = build_spectra(config, proc_st)

                    from sourcespec.ssp_plot_traces import plot_traces
                    plot_traces(config, proc_st)

                    # Spectral inversion
                    from sourcespec.ssp_inversion import spectral_inversion
                    sspec_output = spectral_inversion(config, spec_st, weight_st)

                    # Radiated energy
                    from sourcespec.ssp_radiated_energy import radiated_energy
                    radiated_energy(config, spec_st, specnoise_st, sspec_output)

                    # Local magnitude
                    if config.compute_local_magnitude:
                        from sourcespec.ssp_local_magnitude import local_magnitude
                        local_magnitude(config, st, proc_st, sspec_output)

                    # Compute summary statistics from station spectral parameters
                    from sourcespec.ssp_summary_statistics import compute_summary_statistics
                    compute_summary_statistics(config, sspec_output)

                    # Save output
                    from sourcespec.ssp_output import write_output
                    write_output(config, sspec_output)

                    # Save residuals
                    from sourcespec.ssp_residuals import spectral_residuals
                    spectral_residuals(config, spec_st, sspec_output)

                    # Plotting
                    from sourcespec.ssp_plot_spectra import plot_spectra
                    plot_spectra(config, spec_st, specnoise_st, plot_type='regular')
                    plot_spectra(config, weight_st, plot_type='weight')
                    from sourcespec.ssp_plot_stacked_spectra import plot_stacked_spectra
                    plot_stacked_spectra(config, spec_st, sspec_output)
                    from sourcespec.ssp_plot_params_stats import box_plots
                    box_plots(config, sspec_output)
                    if config.plot_station_map:
                        from sourcespec.ssp_plot_stations import plot_stations
                        plot_stations(config, sspec_output)

                    if config.html_report:
                        from sourcespec.ssp_html_report import html_report
                        html_report(config, sspec_output)

                    ssp_exit()
                except:
                    pass