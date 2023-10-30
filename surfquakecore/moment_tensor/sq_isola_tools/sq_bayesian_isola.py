import os
from obspy import UTCDateTime, read_inventory
from surfquakecore import green_dir
from surfquakecore.moment_tensor.mti_parse import load_mti_configuration
from surfquakecore.moment_tensor.sq_isola_tools import BayesISOLA
from surfquakecore.moment_tensor.sq_isola_tools.mti_utilities import MTIManager
from surfquakecore.utils.obspy_utils import MseedUtil


class bayesian_isola_core:
    def __init__(self, project: dict, metadata_file: str, parameters_folder: str, working_directory: str,
                 ouput_directory: str):

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

        inventory = read_inventory(self.metadata_file)
        ms = MseedUtil()
        files_list = []
        list_of_earthquakes = ms.list_folder_files(self.parameters)
        for num, earthquake in enumerate(list_of_earthquakes):
            try:
                mti_config = self._load_parameters(earthquake)
            except:
                mti_config = None
            if mti_config is not None:
                files_list = self._get_files_from_config(mti_config)
                parameters = mti_config.to_dict()
                self._run_inversion(parameters, inventory, files_list, str(num), save_stream_plot=save_stream_plot)

    def _run_inversion(self, parameters, inventory, files_list, num, save_stream_plot=False):

        # TODO: might be is good idea to include option to remove previuos inversions

        local_folder = os.path.join(self.ouput_directory, num)

        if not os.path.exists(self.ouput_directory):
            os.makedirs(self.ouput_directory)

        if not os.path.exists(local_folder):
            os.makedirs(local_folder)

        ### Process data ###

        st = MTIManager.default_processing(files_list, parameters['origin_date'], inventory, local_folder, regional=True,
                    remove_response=parameters['signal_processing_pams']['remove_response'],
                    save_stream_plot=save_stream_plot)

        mt = MTIManager(st, inventory, parameters['latitude'], parameters['longitude'],
            parameters['depth'], UTCDateTime(parameters['origin_date']), parameters["inversion_parameters"]["min_dist"]*1000,
                        parameters["inversion_parameters"]["max_dist"]*1000, self.working_directory)

        MTIManager.move_files2workdir(green_dir, self.working_directory)
        # [st, deltas] = mt.get_stations_index()
        # inputs = BayesISOLA.load_data(outdir=local_folder)
        # inputs.set_event_info(lat=parameters['latitude'], lon=parameters['longitude'], depth=(parameters['depth'] / 1000),
        #                       mag=parameters['magnitude'], t=UTCDateTime(parameters['origin_Date']))
        #
        # # Sets the source time function for calculating elementary seismograms inside green folder type, working_directory, t0=0, t1=0
        # inputs.set_source_time_function(parameters["inversion_paramters"]["source_type"].lower(), self.working_directory, t0=2.0, t1=0.5)
        #
        # # Create data structure self.stations
        # inputs.read_network_coordinates(os.path.join(self.working_directory, "stations.txt"))

        # edit self.stations_index
        inputs.read_network_coordinates(filename=os.path.join(self.working_directory, "stations.txt"))
        #
        stations = inputs.stations
        stations_index = inputs.stations_index
        #
        # NEW FILTER STATIONS PARTICIPATION BY RMS THRESHOLD
        # mt.get_traces_participation(None, 20, self.parameters['rms_thresh'], magnitude=event_info.mw,
        #                             distance=self.max_distance)
        # inputs.stations, inputs.stations_index = mt.filter_mti_inputTraces(stations, stations_index)
        #
        # # read crustal file and writes in green folder
        # inputs.read_crust(self.parameters['earth_model'], output=os.path.join(self.working_directory_local,
        #                     "crustal.dat"))  # read_crust(source, output='green/crustal.dat')
        #
        # # writes station.dat in working folder from self.stations
        # inputs.write_stations(self.working_directory_local)
        #
        # inputs.data_raw = st
        # inputs.create_station_index()
        # inputs.data_deltas = deltas
        #
        # grid = BayesISOLA.grid(inputs, self.working_directory, location_unc=3000, depth_unc=self.parameters['depth_unc'],
        #         time_unc=self.parameters['time_unc'], step_x=200, step_z=200, max_points=500, circle_shape=False,
        #                        rupture_velocity=self.parameters['rupture_velocity'])
        #
        # data = BayesISOLA.process_data(inputs, self.working_directory_local, grid, threads=self.cpuCount,
        #         use_precalculated_Green=False, fmin=self.parameters["fmin"],fmax=self.parameters["fmax"],
        #                                correct_data=False)
        #
        # cova = BayesISOLA.covariance_matrix(data)
        # cova.covariance_matrix_noise(crosscovariance=self.parameters['covariance'], save_non_inverted=True)
        # #
        # solution = BayesISOLA.resolve_MT(data, cova, self.working_directory_local,
        #             deviatoric=self.parameters["deviatoric"], from_axistra=True)
        #
        # # deviatoric=True: force isotropic component to be zero
        # #
        # if self.parameters['plot_save']:
        #     plot = BayesISOLA.plot(solution, self.working_directory_local, from_axistra=True)
        #     plot.html_log(h1='Example_Test')
        #
        # del inputs
        # del grid
        # del data
        # del plot
        # del mt
        # del self.st
        # del stations
        # del stations_index
        # gc.collect()






