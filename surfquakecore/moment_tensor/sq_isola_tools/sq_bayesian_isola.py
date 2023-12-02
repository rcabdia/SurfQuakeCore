import gc
import os
import shutil
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import List, Optional, Union

from obspy import UTCDateTime, read_inventory, Inventory

from surfquakecore.bin import green_bin_dir
from surfquakecore.moment_tensor.mti_parse import load_mti_configurations, load_mti_configuration
from surfquakecore.moment_tensor.sq_isola_tools import BayesISOLA
from surfquakecore.moment_tensor.sq_isola_tools.BayesISOLA._covariance_matrix import covariance_matrix_noise
from surfquakecore.moment_tensor.sq_isola_tools.BayesISOLA.load_data import load_data
from surfquakecore.moment_tensor.sq_isola_tools.mti_utilities import MTIManager
from surfquakecore.moment_tensor.structures import MomentTensorInversionConfig
from surfquakecore.utils.obspy_utils import MseedUtil


def generate_mti_id_output(mti_config: MomentTensorInversionConfig) -> str:
    """
    Create an id for the output results of moment tensor inversion. This is used for generate folder names
    for each MomentTensorInversionConfig

    Args:
        mti_config:

    Returns:

    """
    return f"{mti_config.origin_date.strftime('%d_%m_%Y_%H%M%S')}"


class BayesianIsolaCore:
    def __init__(self, project: dict, inventory_file: str,
                 output_directory: str, save_plots=False):
        """

        :param project:
        :param inventory_file:
        :param output_directory:
        :param save_plots:
        """

        self.working_directory: Optional[str] = None
        self.inventory_file = inventory_file
        self.project = project
        self.output_directory = output_directory
        self.save_plots = save_plots

        # if not os.path.exists(self.working_directory):
        #     os.makedirs(self.working_directory)
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        self._cpu_count = max(1, os.cpu_count() - 1)
        self._inventory: Optional[Inventory] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    def inventory(self) -> Inventory:
        if not self._inventory:
            self._inventory = read_inventory(self.inventory_file)

        return self._inventory

    @property
    def results(self):

        for out_dir in os.listdir(self.output_directory):
            if os.path.isdir(out_dir):
                yield os.path.join(self.output_directory, out_dir, "log.txt")

    @contextmanager
    def _load_work_directory(self):
        """
        Context manager to create and delete working directory.
        Returns:
        """
        try:
            if self.working_directory is None:
                self.working_directory = TemporaryDirectory(ignore_cleanup_errors=True).name
            if not os.path.isdir(self.working_directory):
                os.mkdir(self.working_directory)
            yield self.working_directory
        finally:
            if os.path.isdir(self.working_directory):
                shutil.rmtree(self.working_directory)

    def _get_files_from_config(self, mti_config: MomentTensorInversionConfig):

        files_list = []

        for station in mti_config.stations:
            files_list.extend(
                MseedUtil().get_now_files(
                    project=self.project,
                    date=mti_config.origin_date,
                    stations_list=station.name,
                    channel_list='|'.join(map(str, station.channels))
                )
            )

        return files_list

    def run_inversion(self, mti_config: Union[str, MomentTensorInversionConfig], **kwargs):

        """
        This method should be to loop over config files and run the inversion.
        Previously it is needed to load the project and metadata.

        Args:
            mti_config: Either a directory of .ini files, a .ini file or an instance of MomentTensorInversionConfig
            **kwargs:

        Returns:

        """

        if isinstance(mti_config, str) and os.path.isdir(mti_config):
            _mti_configurations = load_mti_configurations(mti_config)
        elif isinstance(mti_config, str) and os.path.isfile(mti_config):
            _mti_configurations = (load_mti_configuration(mti_config), )
        elif isinstance(mti_config, MomentTensorInversionConfig):
            _mti_configurations = (mti_config, )
        else:
            raise ValueError(f"mti_config {mti_config} is not valid. It must be either a directory "
                             f"with valid .ini files or a MomentTensorInversionConfig instance.")

        save_stream_plot = kwargs.pop('save_plot', True)

        for mti_config in _mti_configurations:
            files_list = self._get_files_from_config(mti_config)
            self._run_inversion(
                mti_config=mti_config,
                files_list=files_list,
                save_stream_plot=save_stream_plot
            )

    def _run_inversion(self, mti_config: MomentTensorInversionConfig, files_list, save_stream_plot=False):

        # TODO: might be is good idea to include option to remove previuos inversions
        # cleaning working directory

        out_dir_key = generate_mti_id_output(mti_config=mti_config)
        local_folder = os.path.join(self.output_directory, out_dir_key)
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)

        st = MTIManager.default_processing(
            files_path=files_list,
            origin_time=mti_config.origin_date,
            inventory=self.inventory,
            output_directory=local_folder,
            regional=True,
            remove_response=mti_config.signal_processing_parameters.remove_response,
            save_stream_plot=save_stream_plot
        )

        with self._load_work_directory() as green_func_dir:

            mt = MTIManager(st, self.inventory, mti_config.latitude, mti_config.longitude,
                            mti_config.depth, UTCDateTime(mti_config.origin_date),
                            mti_config.inversion_parameters.min_dist*1000,
                            mti_config.inversion_parameters.max_dist*1000, mti_config.magnitude,
                            mti_config.signal_processing_parameters.rms_thresh, green_func_dir)

            MTIManager.move_files2workdir(green_bin_dir, green_func_dir)
            [st, deltas] = mt.get_stations_index()
            inputs = load_data(outdir=local_folder)
            inputs.set_event_info(lat=mti_config.latitude, lon=mti_config.longitude, depth=mti_config.depth,
                                  mag=mti_config.magnitude, t=UTCDateTime(mti_config.origin_date))

            inputs.set_source_time_function(mti_config.inversion_parameters.source_type.lower(), green_func_dir,
                                            t0=mti_config.inversion_parameters.source_duration, t1=0.5)
            #
            # Create data structure self.stations
            # edit self.stations_index
            inputs.read_network_coordinates(filename=os.path.join(green_func_dir, "stations.txt"))
            #
            stations = inputs.stations
            stations_index = inputs.stations_index


            # NEW FILTER STATIONS PARTICIPATION BY RMS THRESHOLD
            mt.get_participation()

            inputs.stations, inputs.stations_index = mt.filter_mti_inputTraces(stations, stations_index)

            # read crustal file and writes in green folder, read_crust(source, output='green/crustal.dat')
            inputs.read_crust(mti_config.inversion_parameters.earth_model_file,
                              output=os.path.join(green_func_dir, "crustal.dat"))

            # writes station.dat in working folder from self.stations
            inputs.write_stations(green_func_dir)
            #
            inputs.data_raw = st
            inputs.create_station_index()
            inputs.data_deltas = deltas
            #
            grid = BayesISOLA.grid(inputs, green_func_dir,
                                   location_unc=mti_config.inversion_parameters.location_unc,
                                   depth_unc=mti_config.inversion_parameters.depth_unc,
                                   time_unc=mti_config.inversion_parameters.time_unc,
                                   step_x=200, step_z=200, max_points=500, circle_shape=False,
                                   rupture_velocity=mti_config.inversion_parameters.rupture_velocity)
            #
            fmax = mti_config.signal_processing_parameters.freq_max
            fmin = mti_config.signal_processing_parameters.freq_min
            data = BayesISOLA.process_data(inputs, green_func_dir, grid, threads=self._cpu_count,
                                           use_precalculated_Green=False, fmin=fmin,
                                           fmax=fmax, correct_data=False)

            cova = BayesISOLA.covariance_matrix(data)
            covariance_matrix_noise(cova, crosscovariance=mti_config.inversion_parameters.covariance,
                                    save_non_inverted=True)
            # deviatoric=True: force isotropic component to be zero
            solution = BayesISOLA.resolve_MT(data, cova, green_func_dir,
                                             deviatoric=mti_config.inversion_parameters.deviatoric, from_axistra=True)

            #if self.parameters['plot_save']:
            if self.save_plots:
                plot_mti = BayesISOLA.plot(solution, green_func_dir, from_axistra=True)
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








