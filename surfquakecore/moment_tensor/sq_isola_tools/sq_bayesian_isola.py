import gc
import os
import shutil
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import Optional, Union, Tuple

from obspy import UTCDateTime, read_inventory, Inventory

from surfquakecore.binaries import BINARY_GREEN_DIR
from surfquakecore.moment_tensor.mti_parse import load_mti_configurations, load_mti_configuration, read_isola_result
from surfquakecore.moment_tensor.sq_isola_tools import bayes_isola
from surfquakecore.moment_tensor.sq_isola_tools.bayes_isola import ResolveMt, InversionDataManager
from surfquakecore.moment_tensor.sq_isola_tools.mti_utilities import MTIManager
from surfquakecore.moment_tensor.structures import MomentTensorInversionConfig, MomentTensorResult
from surfquakecore.project.surf_project import SurfProject
from surfquakecore.utils.obspy_utils import MseedUtil
from surfquakecore.utils.system_utils import get_python_major_version


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
    def __init__(self, project: SurfProject, inventory_file: str,
                 output_directory: str, save_plots=False):
        """

        :param project: SurfProject object
        :param inventory_file: File to the metadata file
        :param output_directory: Root path to the output directory where inversion results will be saved
        :param save_plots: if figures summarizing the results for each inversion are desired
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
    def results(self, is_log=False):

        filename = 'log.txt' if is_log else 'inversion.json'

        for out_dir in os.listdir(self.output_directory):
            if os.path.isdir(out_dir):
                yield os.path.join(self.output_directory, out_dir, filename)

    @property
    def inversion_results(self) -> Tuple[MomentTensorResult, ...]:

        return tuple(read_isola_result(result_file) for result_file in self.results)

    @contextmanager
    def _load_work_directory(self):
        """
        Context manager to create and delete working directory.
        Returns:
        """
        temp_dir = None
        try:
            if self.working_directory is None:
                kw = {"ignore_cleanup_errors": True} if get_python_major_version() > 9 else {}
                temp_dir = TemporaryDirectory(**kw)
                self.working_directory = temp_dir.name
                print("Working Directory at temporal Folder ", self.working_directory )
            if not os.path.isdir(self.working_directory):
                os.mkdir(self.working_directory)
            yield self.working_directory
        finally:
            if temp_dir:
                temp_dir.cleanup()

            if os.path.isdir(self.working_directory):
                shutil.rmtree(self.working_directory)

    def _get_files_from_config(self, mti_config: MomentTensorInversionConfig):

        files_list = []

        for station in mti_config.stations:
            files_list.extend(
                self.project.get_now_files(date=mti_config.origin_date, stations_list=station.name,
                                           channel_list='|'.join(map(str, station.channels)),
                                           only_datafiles_list=True)
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
            _mti_configurations = (load_mti_configuration(mti_config),)
        elif isinstance(mti_config, MomentTensorInversionConfig):
            _mti_configurations = (mti_config,)
        else:
            raise ValueError(f"mti_config {mti_config} is not valid. It must be either a directory "
                             f"with valid .ini files or a MomentTensorInversionConfig instance.")

        save_stream_plot = kwargs.pop('save_plot', self.save_plots)

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

            mt = MTIManager(
                stream=st,
                inventory=self.inventory,
                working_directory=green_func_dir,
                mti_config=mti_config,
            )

            mt.copy_to_working_directory(BINARY_GREEN_DIR)
            st, deltas = mt.get_stations_index()

            inputs = InversionDataManager(outdir=local_folder)
            inputs.set_event_info(lat=mti_config.latitude, lon=mti_config.longitude, depth=mti_config.depth_km,
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
            print("Creating Green Functions")
            grid = bayes_isola.grid(inputs, green_func_dir,
                                    location_unc=mti_config.inversion_parameters.location_unc,
                                    depth_unc=mti_config.inversion_parameters.depth_unc,
                                    time_unc=mti_config.inversion_parameters.time_unc,
                                    step_x=200, step_z=200, max_points=500, circle_shape=False,
                                    rupture_velocity=mti_config.inversion_parameters.rupture_velocity)
            #

            # TODO refactor this

            print("Processing Seismic Waveforms")
            data = bayes_isola.process_data(
                data=inputs,
                working_directory=green_func_dir,
                grid=grid,
                threads=self._cpu_count,
                use_precalculated_Green=False,
                fmin=mti_config.signal_processing_parameters.min_freq,
                fmax=mti_config.signal_processing_parameters.max_freq,
                correct_data=False
            )

            cova = bayes_isola.CovarianceMatrix(data)
            cova.covariance_matrix_noise(
                crosscovariance=mti_config.inversion_parameters.covariance,
                save_non_inverted=True
            )
            # deviatoric=True: force isotropic component to be zero
            print("Processing Inversion")
            solution = ResolveMt(
                data=data,
                cova=cova,
                working_directory=green_func_dir,
                deviatoric=mti_config.inversion_parameters.deviatoric,
                from_axistra=True
            )

            inputs.save_inversion_results(mti_config.to_dict())

            # if self.parameters['plot_save']:
            if self.save_plots:
                print("Plotting Solutions")
                try:
                    plot_mti = bayes_isola.plot(solution, green_func_dir, from_axistra=True)
                    plot_mti.html_log(h1='surfQuake MTI')
                except:
                    print("Coudn't complete plotting")

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
