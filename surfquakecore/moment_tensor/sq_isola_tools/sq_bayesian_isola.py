import gc
import os
from obspy import UTCDateTime, Stream
from surfquakecore.moment_tensor.sq_isola_tools import BayesISOLA
# from surfquakecore import green_path
# from surfquakecore.DataProcessing import SeismogramDataAdvanced
# from surfquakecore.Gui.Frames.qt_components import MessageDialog
# from surfquakecore.Utils.obspy_utils import MseedUtil
# from surfquakecore.seismogramInspector.signal_processing_advanced import duration
# from surfquakecore.sq_isola_tools import BayesISOLA
# from surfquakecore.sq_isola_tools.mti_utilities import MTIManager


class bayesian_isola_db:
    def __init__(self, model, entities, metadata: dict, project: dict, parameters: dict, macro: dict):

        """
        ----------
        Parameters
        ----------
        metadata dict: information of stations
        project dict: information of seismogram data files available
        parameters: dictionary containing database entities and all GUI parameters
        """

        self.model = model
        self.entities = entities
        self.metadata = metadata
        self.project = project
        self.parameters = parameters
        self.macro = macro
        self.cpuCount = os.cpu_count() - 1
        self.working_directory_local = None

    def get_now_files(self, date, pick_time, stations_list):
        date = UTCDateTime(date)
        pick_time = UTCDateTime(pick_time)
        selection = [".", stations_list, "."]

        _, self.files_path = MseedUtil.filter_project_keys(self.project, net=selection[0], station=selection[1],
                                                       channel=selection[2])
        start = date - 1300 #half and hour before
        end = pick_time + 2*3600 #end 2 hours after
        files_path = MseedUtil.filter_time(list_files=self.files_path, starttime=start, endtime=end)
        return files_path

    def get_stations_list(self, phase_info):

        pick_times = []
        stations_list = []
        distances = []

        for phase in phase_info:
            if phase.time_weight >= 0.9 and abs(phase.time_residual) <= 3.5:
                stations_list.append(phase.station_code)
                pick_times.append(phase.time)
                distances.append(phase.distance_km)

        station_filter = '|'.join(stations_list)
        max_time = max(pick_times)
        max_distance = max(distances)

        return station_filter, max_time, max_distance

    def run_inversion(self):

        if not os.path.exists(self.parameters['output_directory']):
            os.makedirs(self.parameters['output_directory'])

        for (i, entity) in enumerate(self.entities):
            self.st = []
            if not os.path.exists(os.path.join(self.parameters['output_directory'], str(i))):
                os.makedirs(os.path.join(self.parameters['output_directory'], str(i)))
            # TODO: COULD BE POSSIBLE TO ADD THE POSSIBILITY
            #  TO SAVE THIS DIRECTORY INSIDE CORRESPONDING OUTPUT FOR LATER ANALYSIS
            self.output_directory_local = os.path.join(self.parameters['output_directory'], str(i))
            self.working_directory_local = self.parameters['working_directory']
            event_info = self.model.find_by(latitude=entity[0].latitude, longitude=entity[0].longitude,
                        depth=entity[0].depth, origin_time=entity[0].origin_time)
            phase_info = event_info.phase_info
            origin_time = event_info.origin_time
            print(event_info)
            station_filter, max_time, max_distance = self.get_stations_list(phase_info)
            self.max_distance = max_distance
            files_path = self.get_now_files(origin_time, max_time, station_filter)

            # TODO: TAKE CARE WITH TYPE OF MAGNITUDE
            try:
                self.process_data(files_path, origin_time, entity[0].transformation,  pick_time=max_time,
                                  magnitude=event_info.mw, save_stream_plot=True)
            except:
                self.process_data(files_path, origin_time, entity[0].transformation)

            self.invert(event_info)

    def filter_error_message(self, msg):
        md = MessageDialog(self)
        md.set_info_message(msg)

    def process_data(self, files_path, date, transform, **kwargs):

        all_traces = []
        date = UTCDateTime(date)
        start_time = date
        end_time = date + 240

        pick_time = kwargs.pop('pick_time', None)
        max_distance = kwargs.pop('max_distance', None)
        magnitude = kwargs.pop('magnitude', None)
        save_stream_plot = kwargs.pop('save_stream_plot', True)

        if pick_time is not None:
            pick_time = UTCDateTime(pick_time)
            delta_min = pick_time - date
            if delta_min <= 240:
                start_time = date - 8*60
                end_time = pick_time + 8*80
            else:
                if pick_time != None and magnitude != None:
                    D = duration(self.parameters["max_dist"], magnitude)
                    delta_time = (self.parameters["max_dist"] / 3.5) + D
                    start_time = date - (delta_time/3)
                    end_time = pick_time + delta_time
        else:
            if transform == "SIMPLE" and max_distance <= 600:
                delta_time = 8*60
                start_time = date - (delta_time / 3)
                end_time = date + delta_time
            else:
                delta_time = 1300
                start_time = date - (delta_time / 3)
                end_time = date + delta_time

        for file in files_path:
            sd = SeismogramDataAdvanced(file)
            tr = sd.get_waveform_advanced(self.macro, self.metadata, start_time=start_time, end_time=end_time)
            if len(tr.data) > 0:
                all_traces.append(tr)

        self.st = Stream(traces=all_traces)
        self.st.merge()

        if save_stream_plot:
            outputdir = os.path.join(self.output_directory_local, "stream_raw.png")
            self.st.plot(outfile=outputdir, size=(800, 600))

    def invert(self, event_info):

        mt = MTIManager(self.st, self.metadata, event_info.latitude, event_info.longitude,
            event_info.depth, UTCDateTime(event_info.origin_time), self.parameters["min_dist"]*1000,
                        self.parameters["max_dist"]*1000, self.working_directory_local)

        # TODO: Move binaries depending on your OS SYSTEM to the working Directory
        # TODO: It is needed to remove this folder before another inversion
        # TODO: It is necessary to filter by max number of stations

        MTIManager.move_files2workdir(green_path, self.working_directory_local)
        [st, deltas] = mt.get_stations_index()

        inputs = BayesISOLA.load_data(outdir=self.output_directory_local)
        inputs.set_event_info(lat=event_info.latitude, lon=event_info.longitude, depth=(event_info.depth/1000),
        mag=event_info.mw, t=UTCDateTime(event_info.origin_time))

        # Sets the source time function for calculating elementary seismograms inside green folder type, working_directory, t0=0, t1=0
        inputs.set_source_time_function('triangle', self.working_directory_local, t0=2.0, t1=0.5)

        # Create data structure self.stations
        inputs.read_network_coordinates(os.path.join(self.working_directory_local, "stations.txt"))

        # edit self.stations_index
        inputs.read_network_coordinates(filename=os.path.join(self.working_directory_local, "stations.txt"))

        stations = inputs.stations
        stations_index = inputs.stations_index

        # NEW FILTER STATIONS PARTICIPATION BY RMS THRESHOLD
        mt.get_traces_participation(None, 20, self.parameters['rms_thresh'], magnitude=event_info.mw,
                                    distance=self.max_distance)
        inputs.stations, inputs.stations_index = mt.filter_mti_inputTraces(stations, stations_index)

        # read crustal file and writes in green folder
        inputs.read_crust(self.parameters['earth_model'], output=os.path.join(self.working_directory_local,
                            "crustal.dat"))  # read_crust(source, output='green/crustal.dat')

        # writes station.dat in working folder from self.stations
        inputs.write_stations(self.working_directory_local)

        inputs.data_raw = st
        inputs.create_station_index()
        inputs.data_deltas = deltas

        grid = BayesISOLA.grid(inputs, self.working_directory_local, location_unc=3000, depth_unc=self.parameters['depth_unc'],
                time_unc=self.parameters['time_unc'], step_x=200, step_z=200, max_points=500, circle_shape=False,
                               rupture_velocity=self.parameters['rupture_velocity'])

        data = BayesISOLA.process_data(inputs, self.working_directory_local, grid, threads=self.cpuCount,
                use_precalculated_Green=False, fmin=self.parameters["fmin"],fmax=self.parameters["fmax"],
                                       correct_data=False)

        cova = BayesISOLA.covariance_matrix(data)
        cova.covariance_matrix_noise(crosscovariance=self.parameters['covariance'], save_non_inverted=True)
        #
        solution = BayesISOLA.resolve_MT(data, cova, self.working_directory_local,
                    deviatoric=self.parameters["deviatoric"], from_axistra=True)

        # deviatoric=True: force isotropic component to be zero
        #
        if self.parameters['plot_save']:
            plot = BayesISOLA.plot(solution, self.working_directory_local, from_axistra=True)
            plot.html_log(h1='Example_Test')

        del inputs
        del grid
        del data
        del plot
        del mt
        del self.st
        del stations
        del stations_index
        gc.collect()
