from obspy import Stream, Trace, UTCDateTime
import numpy as np
from surfquakecore.data_processing.processing_methods import spectral_derivative, spectral_integration, filter_trace, \
    wiener_filter, add_frequency_domain_noise, normalize, wavelet_denoise, safe_downsample, smoothing, \
    trace_envelope, whiten_new, trim_trace
from surfquakecore.cython_module.hampel import hampel
from obspy.signal.util import stack
from obspy.signal.cross_correlation import correlate_template
from surfquakecore.data_processing.seismicUtils import SeismicUtils
from surfquakecore.spectral.specrun import TraceSpectrumResult, TraceSpectrogramResult


class SeismogramData:

    @staticmethod
    def run_analysis(tr_input, config, inventory=None, fill_gaps=True):

        """
        Process a trace or 1-trace stream using a config dict.

        Parameters
        ----------
        tr_input : obspy.Trace or obspy.Stream
            The input trace or single-trace stream.
        config : list of dict
            List of processing steps, e.g. from YAML.
        inventory : obspy.Inventory, optional
            For response removal.
        fill_gaps : bool
            Whether to merge gaps using interpolation.

        Returns
        -------
        obspy.Trace
        Processed trace.
        """

        try:
            # --- Ensure input is a valid Stream or Trace ---
            if isinstance(tr_input, Trace):
                st = Stream([tr_input])
            elif isinstance(tr_input, Stream):
                st = tr_input.copy()
            else:
                print("[WARNING] Invalid trace input type, skipping.")
                return None

            # --- Handle gaps and merging ---
            if fill_gaps:
                try:
                    gaps = st.get_gaps()
                    if gaps:
                        print("[INFO] Gaps found, merging trace with interpolation...")
                        st.merge(method=1, fill_value='interpolate')
                    else:
                        st.merge(method=1)
                except Exception as e:
                    print(f"[WARNING] Failed to check/merge gaps: {e}")
                    return None
            else:
                st.merge(method=1)

            if len(st) == 0:
                print("[WARNING] Stream is empty after merging, skipping.")
                return None

            tr = st[0]


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

                if _config['name'] == 'remove_response':
                    if _config['units'] != "Wood Anderson":
                        try:
                            tr.remove_response(inventory=inventory, pre_filt=_config['pre_filt'],
                                               output=_config['units'], water_level=_config['water_level'])
                        except:
                            print("Coudn't deconvolve", tr.stats)
                            tr.data = np.array([])

                    elif _config['units'] == "Wood Anderson":
                        # print("Simulating Wood Anderson Seismograph")
                        if inventory is not None:
                            resp = inventory.get_response(tr.id, tr.stats.starttime)

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
                    #tr = whiten(tr, _config['freq_width'], taper_edge=_config['taper_edge'])
                    tr = whiten_new(tr, _config['freq_width'], taper_edge=_config['taper_edge'])

                if _config['name'] == 'remove_spikes':

                    filtered, outliers, medians, mads, thresholds = (
                        hampel(tr.data, _config['window_size'] * tr.stats.sampling_rate, _config['sigma']))
                    tr.data = filtered

                if _config['name'] == 'time_normalization':
                    if 'norm_win' in _config.keys():
                        tr = normalize(tr, norm_win=_config['norm_win'], norm_method=_config['method'])
                    else:
                        tr = normalize(tr, norm_method=_config['method'])

                if _config['name'] == 'wavelet_denoise':
                    tr = wavelet_denoise(tr, dwt=_config['dwt'], threshold=_config['threshold'])

                if _config['name'] == 'resample':
                    if tr.stats.sampling_rate < _config['sampling_rate']:
                        tr.resample(sampling_rate=_config['sampling_rate'], window='hanning',
                                    no_filter=_config['pre_filter'])
                    elif tr.stats.sampling_rate > _config['sampling_rate']:
                        tr = safe_downsample(tr, _config['sampling_rate'], pre_filter=_config['pre_filter'])
                    else:
                        pass

                if _config['name'] == 'fill_gaps':
                    st = Stream(tr)
                    st.merge(fill_value=_config['method'])
                    tr = st[0]

                if _config['name'] == 'smoothing':
                    tr = smoothing(tr, type=_config['method'], k=_config['time_window'], fwhm=_config['FWHM'])

                if _config['name'] == 'envelope':
                    if _config['method'] == "SMOOTH" and "corner_freq" in _config.keys():
                        tr = trace_envelope(tr, method=_config['method'], corner_freq=_config['corner_freq'])
                    else:
                        tr = trace_envelope(tr, method=_config['method'])

                if _config['name'] == 'cut':
                    tr = trim_trace(tr, _config['mode'], _config['t1'], _config['t2'])

                if _config['name'] == 'spectrum':
                    spec = TraceSpectrumResult(tr)
                    spec.compute_spectrum()
                    spec.to_pickle(folder_path=_config['output_path'])

                if _config['name'] == 'spectrogram':

                    if "overlap" in _config.keys():
                        overlap = _config["overlap"]
                    else:
                        overlap = 50

                    spec = TraceSpectrogramResult(tr)
                    spec.compute_spectrogram(win=_config["win"], overlap_percent=overlap)
                    spec.to_pickle(folder_path=_config['output_path'])


            return tr

        except Exception as e:
            print(f"[ERROR] Trace processing failed: {e}")
            return None

class StreamProcessing:
    """
    Class for applying stream-wide processing steps (e.g., stack, cross-correlation, rotate, shift).
    """

    STREAM_METHODS = {"stack", "cross_correlate", "rotate", "shift", "synch"}

    def __init__(self, stream: Stream, config: list, **kwargs):
        self.stream = stream
        self.config = config
        self.kwargs = kwargs

    def run_stream_processing(self) -> Stream:
        """
        Apply each stream-wide processing method defined in config.
        """

        if self.config is None:

            return self.stream

        else:

            for step in self.config:
                method_name = step.get("name")

                if method_name not in self.STREAM_METHODS:
                    continue  # skip unknown or trace-level methods

                if method_name == "stack":
                    self.stream = self.apply_stack(step)
                elif method_name == "cross_correlate":
                    self.stream = self.apply_cross_correlation(step)
                elif method_name == "rotate":
                    self.stream = self.apply_rotation(step)
                elif method_name == "shift":
                    self.stream = self.apply_shift(step)
                elif method_name == "synch":
                    self.stream = self.apply_synch(step)
                elif method_name == "concat":
                    self.stream = self.apply_concat()

            return self.stream

    def apply_stack(self, step_config):

        if len(self.stream) == 0:
            return self.stream  # Return empty

        method = "linear"
        if step_config["method"] == "pw" or step_config["method"] == "root":
            method = (step_config["method"], step_config["order"])
        elif step_config["method"] == "linear":
            method = step_config["method"]
        # Ensure all traces are the same length
        min_length = min(len(tr.data) for tr in self.stream)
        trimmed_data = np.array([tr.data[:min_length] for tr in self.stream])

        # Apply the stack
        stacked_array = stack(trimmed_data, stack_type=method)

        # Create a new Trace with metadata from the first one
        stacked_trace = Trace(data=stacked_array)
        stacked_trace.stats = self.stream[0].stats.copy()
        stacked_trace.stats.station = "STA"
        stacked_trace.stats.network = "NET"
        stacked_trace.stats.channel = "STK"  # Mark it's a stacked trace

        # Return a new Stream
        return Stream(traces=[stacked_trace])

    def apply_cross_correlation(self, step_config):
        """
        Cross-correlate all traces in stream with respect to a reference trace.

        Returns a new Stream with the correlation functions as Trace objects.
        """
        # --- Configuration ---

        normalize = step_config.get("normalize", 'full')
        mode = step_config.get("mode", 'full')
        reference_idx = step_config.get("reference", 0)

        st = self.stream.copy()

        # --- Check for empty stream ---
        if len(st) == 0:
            print("[WARNING] Empty stream for cross-correlation")
            return Stream()

        # --- Ensure uniform sampling rate ---
        sr = st[0].stats.sampling_rate
        for tr in st:
            if tr.stats.sampling_rate != sr:
                raise ValueError("Inconsistent sampling rates in stream.")

        # --- Trim to common time window ---
        common_start = max(tr.stats.starttime for tr in st)
        common_end = min(tr.stats.endtime for tr in st)
        st.trim(starttime=common_start, endtime=common_end)

        # --- Ensure uniform number of samples ---
        npts = st[0].stats.npts
        for tr in st:
            if tr.stats.npts != npts:
                raise ValueError("Traces do not have same number of samples after trimming.")

        ref_trace = st[reference_idx]
        cc_stream = Stream()

        for i, tr in enumerate(st):
            cc = correlate_template(tr, ref_trace, mode=mode, normalize=normalize, demean=True, method='auto')
            cc_tr = Trace()
            cc_tr.data = cc
            cc_tr.stats.sampling_rate = sr
            cc_tr.stats.network = tr.stats.network
            cc_tr.stats.station = tr.stats.station
            cc_tr.stats.channel = tr.stats.channel[0:1]+"X"+tr.stats.channel[-1]
            cc_tr.stats.correlation_with = ref_trace.id
            cc_tr.stats.original_trace = tr.id
            cc_tr.stats.is_autocorrelation = (i == reference_idx)

            cc_stream.append(cc_tr)

        return Stream(cc_stream)

    def apply_rotation(self, step_config):

        self.stream = SeismicUtils.standardize_to_NE_components(self.stream)

        if "GAC" in step_config["method"]:
            self.stream.rotate(method=step_config["type"])
        else:

            self.stream.rotate(method=step_config["type"], back_azimuth=step_config["angle"],
                               inclination=step_config["inclination"])
        return self.stream

    def apply_shift(self, step_config):
        try:
            time_shifts = step_config.get("time_shifts", [])
            for i, shift_val in enumerate(time_shifts):
                if i < len(self.stream):
                    self.stream[i].stats.starttime += shift_val
            # Now `traces` is updated in-place
        except Exception as e:
            print(f"Error applying time shifts: {e}")
        return self.stream


    def apply_synch(self, step_config):

        if step_config["method"] == "starttime":
            ref_starttime = UTCDateTime("1983-12-23T00:00:00.0")
            for i, tr in enumerate(self.stream):
                self.stream[i].stats.starttime = ref_starttime

        elif step_config["method"] == "MCCC":
            self.stream, _ = SeismicUtils.multichannel(self.stream, resample=False)

        return self.stream

    def apply_concat(self):
        self.stream.merge(method=1, fill_value='interpolate')
        return self.stream

