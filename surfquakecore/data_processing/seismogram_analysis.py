from obspy import Stream
import numpy as np
from surfquakecore.Structures.structures import TracerStatsAnalysis, TracerStats
from surfquakecore.data_processing.processing_methods import spectral_derivative, spectral_integration, filter_trace, \
    wiener_filter, add_frequency_domain_noise, whiten, hampel, normalize, wavelet_denoise, safe_downsample, smoothing


class SeismogramData:
    def __init__(self, stream, inventory, realtime=False, **kwargs):
        self.inventory = inventory
        _stream = kwargs.pop('stream', [])

        self.config_keys = None

        if stream:
            self.st = stream

        if realtime:
            self.__tracer = _stream

        else:
            gaps = self.st.get_gaps()

            if len(gaps) > 0:
                self.st.print_gaps()
                self.st.merge(fill_value="interpolate")

            self.__tracer = self.st[0]

        self.stats = TracerStatsAnalysis.from_dict(self.tracer.stats)

    @classmethod
    def from_tracer(cls, tracer):
        sd = cls(None)
        sd.set_tracer(tracer)
        return sd

    @property
    def tracer(self):
        return self.__tracer

    def set_tracer(self, tracer):
        self.__tracer = tracer
        self.stats = TracerStats.from_dict(self.__tracer.stats)



    def run_analysis(self, config, **kwargs):

        """
        This method loop over config dictionary.
        config dictionary contain the primary key with the processing step and the parameters or that key processing

        Previously it is needed to load the project and metadata.

        Args:
            analysis_config: a .yaml file

        Returns:
        """

        start_time = kwargs.get("start_time", self.stats.StartTime)
        end_time = kwargs.get("end_time", self.stats.EndTime)
        # trace_number = kwargs.get("trace_number", 0)
        tr = self.tracer

        tr.trim(starttime=start_time, endtime=end_time)

        for i in range(len(config)):
            _config = config[i]
            _keys = config[i]

            if _config['name'] == 'rmean':
                if _config['method'] in ['linear, simple', 'demean']:
                    tr.detrend(type=_config['method'])
                elif _config['method'] == 'polynomial':
                    tr.detrend(type=_config['method'], order=_config['order'])
                elif _config['method'] == 'spline':
                    tr.detrend(type=_config['method'], order=_config['order'], dspline=_config['dspline'])

            if _config['name'] == 'taper':
                tr.taper(max_percentage=_config['max_percentage'], type=_config['method'],
                         max_length=_config['max_length'], side=_config['side'])

            if _config['name'] == 'normalize':
                if isinstance(_config['norm'], bool):
                    tr.normalize()
                else:
                    tr.normalize(norm=_config['norm'])

            if _config['name'] == 'differentiate':
                if _config['method'] == "spectral":
                    tr = spectral_derivative(tr)
                else:
                    tr.differentiate(method=_config['method'])

            if _config['name'] == 'integrate':
                if _config['method'] == "spectral":
                    tr = spectral_integration(tr)
                else:
                    tr.integrate(method=_config['method'])

            if _config['name'] == 'filter':

                tr = filter_trace(tr, _config['type'], _config['fmin'], _config['fmax'],
                                  zerophase=_config['zerophase'], corners=_config['corners'])

            if _config['name'] == 'wiener_filter':
                tr = wiener_filter(tr, time_window=_config['time_window'],
                                   noise_power=_config['noise_power'])

            if _config['name'] == 'shift':
                shifts = _config['time_shifts']
                shifts = shifts.split(",")
                for i in range(0, len(shifts) - 1):
                    tr.stats.starttime = tr.stats.starttime + shifts[i]

            if _config['name'] == 'remove_response':
                # inventory = read_inventory(_config['inventory'])
                # print(inventory)
                if _config['units'] != "Wood Anderson":
                    # print("Deconvolving")
                    try:
                        tr.remove_response(inventory=self.inventory, pre_filt=_config['pre_filt'],
                                           output=_config['units'], water_level=_config['water_level'])
                    except:
                        print("Coudn't deconvolve", tr.stats)
                        tr.data = np.array([])

                elif _config['units'] == "Wood Anderson":
                    # print("Simulating Wood Anderson Seismograph")
                    if self.inventory is not None:
                        resp = self.inventory.get_response(tr.id, tr.stats.starttime)

                        resp = resp.response_stages[0]
                        paz_wa = {'sensitivity': 2800, 'zeros': [0j], 'gain': 1,
                                  'poles': [-6.2832 - 4.7124j, -6.2832 + 4.7124j]}

                        paz_mine = {'sensitivity': resp.stage_gain * resp.normalization_factor, 'zeros': resp.zeros,
                                    'gain': resp.stage_gain, 'poles': resp.poles}

                        try:
                            tr.simulate(paz_remove=paz_mine, paz_simulate=paz_wa,
                                        water_level=_config['water_level'])
                        except:
                            print("Coudn't deconvolve", tr.stats)
                            tr.data = np.array([])

            if _config['name'] == 'add_noise':
                tr = add_frequency_domain_noise(tr, noise_type=_config['noise_type'], SNR_dB=_config['SNR_dB'])

            if _config['name'] == 'whitening':
                tr = whiten(tr, _config['freq_width'], taper_edge=_config['taper_edge'])

            if _config['name'] == 'remove_spikes':
                # TODO: NEED to be added the CYTHON OPTION FROM ISP
                tr = hampel(tr, _config['window_size'], _config['n'])

            if _config['name'] == 'time_normalization':
                tr = normalize(tr, norm_win=_config['norm_win'], norm_method=_config['method'])

            if _config['name'] == 'wavelet_denoise':
                tr = wavelet_denoise(tr, dwt=_config['dwt'], threshold=_config['threshold'])

            if _config['name'] == 'resample':
                if tr.sats.sampling_rate < _config['sampling_rate']:
                    tr.resample(sampling_rate=_config['sampling_rate'], window='hanning',
                                no_filter=_config['pre_filter'])
                elif tr.sats.sampling_rate > _config['sampling_rate']:
                    tr = safe_downsample(tr, _config['sampling_rate'], pre_filter=_config['pre_filter'])
                else:
                    pass

            if _config['name'] == 'fill_gaps':
                st = Stream(tr)
                st.merge(fill_value=_config['method'])
                tr = st[0]

            if _config['name'] == 'smoothing':
                tr = smoothing(tr, type=_config['method'], k=_config['time_window'], fwhm=_config['FWHM'])

        return tr
