import gc
import glob
import os
from obspy import UTCDateTime, read_inventory
from surfquakecore.bin import green_bin_dir
from surfquakecore.moment_tensor.mti_parse import load_mti_configuration
from surfquakecore.moment_tensor.sq_isola_tools import BayesISOLA
from surfquakecore.moment_tensor.sq_isola_tools.BayesISOLA.load_data import load_data
from surfquakecore.moment_tensor.sq_isola_tools.mti_utilities import MTIManager
from surfquakecore.utils.obspy_utils import MseedUtil


class bayesian_isola_core:
    def __init__(self, project: dict, metadata_file: str, parameters_folder: str, working_directory: str,
                 ouput_directory: str, save_plots=False):

        """
        ----------
        Parameters
        ----------
        project dict: Information of seismogram data files available
        metadata str: Path to the inventory information of stations coordinates and instrument description
        parameters: dictionary generated from the dataclass
        """

        self.metadata_file = metadata_file
        self.project = project
        self.parameters = parameters_folder
        self.cpuCount = os.cpu_count() - 1
        self.working_directory = working_directory
        self.ouput_directory = ouput_directory
        self.save_plots = save_plots

        if not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
        if not os.path.exists(self.ouput_directory):
            os.makedirs(self.ouput_directory)


    def _get_path_files(self, project, date, stations_list, channel_list):
        files_path = []
        ms = MseedUtil()
        try:
            files_path = ms.get_now_files(project, date, stations_list, channel_list)
        except:
            print("No files to be readed in the project")
        return files_path

    def _return_inventory(self):
        inv = {}
        inv = read_inventory(self.metadata_file)
        return inv

    def _load_parameters(self, config_path):
        mti_config = load_mti_configuration(config_path)
        return mti_config

    def _get_files_from_config(self, mti_config):

        files_list = []
        parameters = mti_config.to_dict()
        print(parameters)

        for station in parameters['stations']:
            channels = '|'.join(map(str, station['channels']))
            files = self._get_path_files(self.project, parameters['origin_date'], station['name'], channels)
            files_list += files

        return files_list


    def run_mti_inversion(self, **kwargs):

        """
        This method should be to loop over config files and run the inversion.
        Previously it is needed to load the project and metadata.
        """

        save_stream_plot = kwargs.pop('save_plot', True)
        if not os.path.exists(self.working_directory):
            os.makedirs(self.ouput_directory)
        inventory = read_inventory(self.metadata_file)
        ms = MseedUtil()
        list_of_earthquakes = ms.list_folder_files(self.parameters)
        for num, earthquake in enumerate(list_of_earthquakes):
            try:
                mti_config = self._load_parameters(earthquake)
            except:
                mti_config = None
            if mti_config is not None:
                files_list = self._get_files_from_config(mti_config)
                parameters = mti_config.to_dict()
                try:
                    self._run_inversion(parameters, inventory, files_list, str(num), save_stream_plot=save_stream_plot)
                except:
                    print("Inversion not possible for earthquake: ", parameters['origin_date'], parameters['latitude'], parameters['longitude'],
            parameters['depth'], parameters['magnitude'])

    def _run_inversion(self, parameters, inventory, files_list, num, save_stream_plot=False):

        # TODO: might be is good idea to include option to remove previuos inversions
        # cleaning working directory

        files = glob.glob(self.working_directory + "/*")
        for f in files:
            os.remove(f)

        local_folder = os.path.join(self.ouput_directory, num)

        if not os.path.exists(self.ouput_directory):
            os.makedirs(self.ouput_directory)

        if not os.path.exists(local_folder):
            os.makedirs(local_folder)
        else:
            files = glob.glob(self.working_directory + "/*")
            for f in files:
                os.remove(f)


        ### Process data ###

        st = MTIManager.default_processing(files_list, parameters['origin_date'], inventory, local_folder, regional=True,
                    remove_response=parameters['signal_processing_pams']['remove_response'],
                    save_stream_plot=save_stream_plot)

        mt = MTIManager(st, inventory, parameters['latitude'], parameters['longitude'],
            parameters['depth'], UTCDateTime(parameters['origin_date']), parameters["inversion_parameters"]["min_dist"]*1000,
                        parameters["inversion_parameters"]["max_dist"]*1000, parameters['magnitude'],
                        parameters['signal_processing_pams']['rms_thresh'], self.working_directory)

        MTIManager.move_files2workdir(green_bin_dir, self.working_directory)
        [st, deltas] = mt.get_stations_index()
        inputs = load_data(outdir=local_folder)
        inputs.set_event_info(lat=parameters['latitude'], lon=parameters['longitude'], depth=parameters['depth'],
                               mag=parameters['magnitude'], t=UTCDateTime(parameters['origin_date']))
        #
        # # Sets the source time function for calculating elementary seismograms inside green folder type, working_directory, t0=0, t1=0
        inputs.set_source_time_function(parameters["inversion_parameters"]["source_type"].lower(), self.working_directory,
                                        t0=2.0, t1=0.5)
        #
        # # Create data structure self.stations
        inputs.read_network_coordinates(os.path.join(self.working_directory, "stations.txt"))

        # edit self.stations_index
        inputs.read_network_coordinates(filename=os.path.join(self.working_directory, "stations.txt"))
        #
        stations = inputs.stations
        stations_index = inputs.stations_index

        # NEW FILTER STATIONS PARTICIPATION BY RMS THRESHOLD
        mt.get_participation()

        inputs.stations, inputs.stations_index = mt.filter_mti_inputTraces(stations, stations_index)

        # read crustal file and writes in green folder, read_crust(source, output='green/crustal.dat')
        inputs.read_crust(parameters["inversion_parameters"]["earth_model_file"],
            output=os.path.join(self.working_directory, "crustal.dat"))

        # writes station.dat in working folder from self.stations
        inputs.write_stations(self.working_directory)
        #
        inputs.data_raw = st
        inputs.create_station_index()
        inputs.data_deltas = deltas
        #
        grid = BayesISOLA.grid(inputs, self.working_directory, location_unc=parameters["inversion_parameters"]["location_unc"],
            depth_unc=parameters["inversion_parameters"]['depth_unc'], time_unc=parameters["inversion_parameters"]['time_unc'],
            step_x=200, step_z=200, max_points=500, circle_shape=False,
            rupture_velocity=parameters["inversion_parameters"]["rupture_velocity"])
        #
        fmax = parameters['signal_processing_pams']["freq_max"]
        fmin = parameters['signal_processing_pams']["freq_min"]
        data = BayesISOLA.process_data(inputs, self.working_directory, grid, threads=self.cpuCount,
                 use_precalculated_Green=False, fmin=fmin,
                                       fmax=fmax, correct_data=False)

        cova = BayesISOLA.covariance_matrix(data)
        cova.covariance_matrix_noise(crosscovariance=parameters["inversion_parameters"]['covariance'],
                                     save_non_inverted=True)
        # deviatoric=True: force isotropic component to be zero
        solution = BayesISOLA.resolve_MT(data, cova, self.working_directory,
                    deviatoric=parameters["inversion_parameters"]["deviatoric"], from_axistra=True)

        #if self.parameters['plot_save']:
        if self.save_plots:
            plot_mti = BayesISOLA.plot(solution, self.working_directory, from_axistra=True)
            plot_mti.html_log(h1='surfQuake MTI')

        del inputs
        del grid
        del data
        del deltas
        del mt
        del st
        del stations
        del stations_index
        del cova
        try:
            if self.save_plots:
                del plot_mti
        except:
            print("coudn't release plotting memory usage")

        gc.collect()






