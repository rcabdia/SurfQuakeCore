import gc
import glob
import os
from obspy import UTCDateTime, read_inventory
from surfquakecore.bin import green_bin_dir
from surfquakecore.moment_tensor.mti_parse import load_mti_configuration
from surfquakecore.moment_tensor.sq_isola_tools import BayesISOLA
from surfquakecore.moment_tensor.sq_isola_tools.BayesISOLA._covariance_matrix import covariance_matrix_noise
from surfquakecore.moment_tensor.sq_isola_tools.BayesISOLA.load_data import load_data
from surfquakecore.moment_tensor.sq_isola_tools.mti_utilities import MTIManager
from surfquakecore.moment_tensor.structures import MomentTensorInversionConfig
from surfquakecore.utils.obspy_utils import MseedUtil


class BayesianIsolaCore:
    def __init__(self, project: dict, metadata_file: str, parameters_folder: str, working_directory: str,
                 ouput_directory: str, save_plots=False):
        """

        :param project:
        :param metadata_file:
        :param parameters_folder:
        :param working_directory:
        :param ouput_directory:
        :param save_plots:
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
        return read_inventory(self.metadata_file)

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
            mti_config = load_mti_configuration(earthquake)
            files_list = self._get_files_from_config(mti_config)
            self._run_inversion(mti_config, inventory, files_list, str(num), save_stream_plot=save_stream_plot)
            # try:
            #     self._run_inversion(parameters, inventory, files_list, str(num), save_stream_plot=save_stream_plot)
            # except Exception as e:
            #     print(e)
            #     print("Inversion not possible for earthquake: ", parameters['origin_date'], parameters['latitude'],
            #           parameters['longitude'],
            #           parameters['depth'], parameters['magnitude'])

    def _run_inversion(self, mti_config: MomentTensorInversionConfig, inventory, files_list, num, save_stream_plot=False):

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

        st = MTIManager.default_processing(files_list, mti_config.origin_date,
                                           inventory, output_directory=local_folder, regional=True,
                                           remove_response=mti_config.signal_processing_parameters.remove_response,
                                           save_stream_plot=save_stream_plot)

        mt = MTIManager(st, inventory, mti_config.latitude, mti_config.longitude,
                        mti_config.depth, UTCDateTime(mti_config.origin_date),
                        mti_config.inversion_parameters.min_dist*1000,
                        mti_config.inversion_parameters.max_dist*1000, mti_config.magnitude,
                        mti_config.signal_processing_parameters.rms_thresh, self.working_directory)

        MTIManager.move_files2workdir(green_bin_dir, self.working_directory)
        [st, deltas] = mt.get_stations_index()
        inputs = load_data(outdir=local_folder)
        inputs.set_event_info(lat=mti_config.latitude, lon=mti_config.longitude, depth=mti_config.depth,
                              mag=mti_config.magnitude, t=UTCDateTime(mti_config.origin_date))

        inputs.set_source_time_function(mti_config.inversion_parameters.source_type.lower(), self.working_directory,
                                        t0=mti_config.inversion_parameters.source_duration, t1=0.5)
        #
        # Create data structure self.stations
        # edit self.stations_index
        inputs.read_network_coordinates(filename=os.path.join(self.working_directory, "stations.txt"))
        #
        stations = inputs.stations
        stations_index = inputs.stations_index


        # NEW FILTER STATIONS PARTICIPATION BY RMS THRESHOLD
        mt.get_participation()

        inputs.stations, inputs.stations_index = mt.filter_mti_inputTraces(stations, stations_index)

        # read crustal file and writes in green folder, read_crust(source, output='green/crustal.dat')
        inputs.read_crust(mti_config.inversion_parameters.earth_model_file,
                          output=os.path.join(self.working_directory, "crustal.dat"))

        # writes station.dat in working folder from self.stations
        inputs.write_stations(self.working_directory)
        #
        inputs.data_raw = st
        inputs.create_station_index()
        inputs.data_deltas = deltas
        #
        grid = BayesISOLA.grid(inputs, self.working_directory,
                               location_unc=mti_config.inversion_parameters.location_unc,
                               depth_unc=mti_config.inversion_parameters.depth_unc,
                               time_unc=mti_config.inversion_parameters.time_unc,
                               step_x=200, step_z=200, max_points=500, circle_shape=False,
                               rupture_velocity=mti_config.inversion_parameters.rupture_velocity)
        #
        fmax = mti_config.signal_processing_parameters.freq_max
        fmin = mti_config.signal_processing_parameters.freq_min
        data = BayesISOLA.process_data(inputs, self.working_directory, grid, threads=self.cpuCount,
                                       use_precalculated_Green=False, fmin=fmin,
                                       fmax=fmax, correct_data=False)

        cova = BayesISOLA.covariance_matrix(data)
        covariance_matrix_noise(cova, crosscovariance=mti_config.inversion_parameters.covariance,
                                     save_non_inverted=True)
        # deviatoric=True: force isotropic component to be zero
        solution = BayesISOLA.resolve_MT(data, cova, self.working_directory,
                                         deviatoric=mti_config.inversion_parameters.deviatoric, from_axistra=True)

        #if self.parameters['plot_save']:
        if self.save_plots:
            plot_mti = BayesISOLA.plot(solution, self.working_directory, from_axistra=True)
            # plot_mti.html_log(h1='surfQuake MTI')

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







