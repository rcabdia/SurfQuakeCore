from obspy import Stream, Trace, UTCDateTime, Inventory
import numpy as np
from surfquakecore.arrayanalysis.beamrun import TraceBeamResult
from surfquakecore.coincidence_trigger.cf_kurtosis import CFKurtosis
from surfquakecore.data_processing.processing_methods import spectral_derivative, spectral_integration, filter_trace, \
    wiener_filter, add_frequency_domain_noise, normalize, wavelet_denoise, safe_downsample, smoothing, \
    trace_envelope, whiten_new, trim_trace, compute_entropy_trace, compute_snr, downsample_trace, particle_motion, \
    rename_trace
try:
    from surfquakecore.cython_module.hampel import hampel
except:
    print("Hampel no compiled. Install gcc compiler and reinstall surfquake")

from obspy.signal.util import stack
from obspy.signal.cross_correlation import correlate_template
from surfquakecore.data_processing.seismicUtils import SeismicUtils
from surfquakecore.spectral.cwtrun import TraceCWTResult
from surfquakecore.spectral.specrun import TraceSpectrumResult, TraceSpectrogramResult
from typing import Optional


class SeismogramData:

    @staticmethod
    def run_analysis(tr_input, config, inventory=None, fill_gaps=False):

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
                            print("Coudn't Remove Instrument Response, Review Inventory ", tr.id)
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
                    # tr = whiten(tr, _config['freq_width'], taper_edge=_config['taper_edge'])
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
                    spec.compute_spectrum(method=_config['method'])
                    spec.to_pickle(folder_path=_config['output_path'])

                if _config['name'] == 'spectrogram':

                    if "overlap" in _config.keys():
                        overlap = _config["overlap"]
                    else:
                        overlap = 50

                    spec = TraceSpectrogramResult(tr)
                    spec.compute_spectrogram(win=_config["win"], overlap_percent=overlap)
                    spec.to_pickle(folder_path=_config['output_path'])

                if _config['name'] == 'cwt':
                    cwt = TraceCWTResult(tr)

                    if "fmin" in _config.keys():
                        fmin = _config["fmin"]
                    else:
                        fmin = None

                    if "fmax" in _config.keys():
                        fmax = _config["fmax"]
                    else:
                        fmax = None

                    cwt.compute_cwt(wavelet_type=_config['wavelet'], param=_config["param"], fmin=fmin, fmax=fmax, nf=80)
                    cwt.to_pickle(folder_path=_config['output_path'])

                if _config['name'] == 'entropy':

                    if "overlap" in _config.keys():
                        overlap = _config['overlap'] * 1E-2
                    else:
                        overlap = 0.5

                    tr = compute_entropy_trace(tr, win=_config['win'], overlap=overlap)

                if _config['name'] == 'snr':
                    tr = compute_snr(tr, method=_config['method'], sign_win=_config['sign_win'],
                                     noise_win=_config['noise_win'])

                if _config['name'] == 'raw':
                    tr = downsample_trace(tr, factor=_config['factor'], to_int=_config['integers'], scale_target=1000)

                if _config['name'] == 'rename':
                    tr = rename_trace(tr, _config)

            return tr

        except Exception as e:
            print(f"[ERROR] Trace processing failed: {e}")
            return None


class StreamProcessing:
    """
    Class for applying stream-wide processing steps (e.g., stack, cross-correlation, rotate, shift).
    """

    STREAM_METHODS = {"stack", "cross_correlate", "rotate", "shift", "synch", "concat", "beam", "particle_motion",
                      "kurtosis"}

    def __init__(self, stream: Stream, config: list, inventory: Optional[Inventory] = None, **kwargs):
        self.stream = stream
        self.config = config
        self.inventory = inventory
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
                elif method_name == "beam":
                    self.apply_beam(step)
                elif method_name == "particle_motion":
                    self.apply_pm(step)
                elif method_name == "kurtosis":
                    self.apply_kurtosis(step)

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
        Cross-correlate all traces in the stream with respect to a reference trace.

        Parameters:
            step_config (dict):
                - "normalize": normalization type (default: 'full')
                - "mode": correlation mode ('full', 'valid', 'same')
                - "reference": index of the reference trace (default: 0)
                - "strict": if True, all traces must align in time (default: True)

        Returns:
            Stream: New stream with correlation functions as Trace objects.
        """
        normalize = step_config.get("normalize", 'full')
        mode = step_config.get("mode", 'full')
        reference_idx = step_config.get("reference", 0)
        strict = step_config.get("trim", False)

        st = self.stream.copy()

        if len(st) == 0:
            print("[WARNING] Empty stream for cross-correlation")
            return Stream()

        # Ensure uniform sampling rate and trim to common time window and ensure aligned length
        sr = st[0].stats.sampling_rate
        npts = st[reference_idx].stats.npts

        if strict:

            for tr in st:
                if tr.stats.sampling_rate != sr:
                    raise ValueError("Inconsistent sampling rates in stream.")

            common_start = max(tr.stats.starttime for tr in st)
            common_end = min(tr.stats.endtime for tr in st)
            st.trim(starttime=common_start, endtime=common_end, pad=True, fill_value=0)

            for tr in st:
                if tr.stats.npts != npts:
                    raise ValueError("Traces do not have same number of samples after trimming.")
        else:
            # In flexible mode, no trimming
            pass  # Traces may have different start times or lengths

        ref_trace = st[reference_idx]
        cc_stream = Stream()

        for i, tr in enumerate(st):
            try:
                cc = correlate_template(tr, ref_trace, mode=mode, normalize=normalize, demean=True, method='auto')
            except Exception as e:
                print(f"[WARNING] Failed to correlate {tr.id} with {ref_trace.id}: {e}")
                continue

            cc_tr = Trace(data=cc)
            cc_tr.stats.sampling_rate = sr
            cc_tr.stats.network = tr.stats.network
            cc_tr.stats.station = ref_trace.stats.station + "_" + tr.stats.station
            cc_tr.stats.channel = tr.stats.channel[0:1] + "X" + tr.stats.channel[-1]
            cc_tr.stats.correlation_with = ref_trace.id
            cc_tr.stats.original_trace = tr.id
            cc_tr.stats.is_autocorrelation = (i == reference_idx)
            if strict:
                num = len(ref_trace.data)
                if (num % 2) == 0:
                    c = int(np.ceil(num / 2.) + 1)  # even
                else:

                    c = int(np.ceil((num + 1) / 2))  # odd

                cc_tr.stats.mid_point = (cc_tr.stats.starttime + c * cc_tr.stats.sampling_rate).timestamp

            cc_stream.append(cc_tr)

        return cc_stream

    def apply_rotation(self, step_config):

        self.stream = SeismicUtils.standardize_to_NE_components(self.stream)

        if "GAC" in step_config["method"]:
            self.stream.rotate(method=step_config["type"])

        else:

            self.stream.rotate(method=step_config["type"], back_azimuth=step_config["angle"],
                               inclination=step_config["inclination"])
        return self.stream

    def apply_shift(self, step_config):
        """
        Apply time alignment to traces in the stream using manual time shifts,
        pick-based phase alignment, or theoretical arrival time alignment.

        Supported shift strategies (evaluated in priority order):

        1. Theoretical Phase Alignment (`phase_theo`)
            Aligns to the theoretical arrival time stored in
            `trace.stats.geodetic['arrivals']`.

        2. Pick-Based Phase Alignment (`phase`)
            Aligns to the observed pick time in `trace.stats.picks`.

        3. Manual Time Shifts (`time_shifts`)
            Adds a fixed offset (in seconds) to the starttime of each trace.

        Parameters
        ----------
        step_config : dict
            Keys:
                - "phase_theo": str, optional
                    Phase name to align using theoretical arrival.
                - "phase": str, optional
                    Phase name to align using pick.
                - "time_shifts": list of float, optional
                    Per-trace time offsets to apply.

        Returns
        -------
        stream : obspy.Stream
            Stream with aligned traces.

        Notes
        -----
        - If both "phase" and "time_shifts" are provided, phase alignment takes precedence.
        - If no valid shift config is given, stream is returned unchanged.
        - If alignment exceeds data length, trace is skipped.
        """
        try:
            phase_theo = step_config.get("phase_theo")
            phase_pick = step_config.get("phase")
            time_shifts = step_config.get("time_shifts")

            if phase_theo:
                for tr in self.stream:
                    arrivals = tr.stats.get("geodetic", {}).get("arrivals", [])
                    theo_time = None
                    for arr in arrivals:
                        if arr.get("phase") == phase_theo:
                            theo_time = arr.get("time")
                            break
                    if theo_time is not None:
                        shift_amount = UTCDateTime(theo_time) - tr.stats.starttime
                        tr.stats.starttime = UTCDateTime(0)
                        tr.data = tr.data[int(shift_amount / tr.stats.delta):]
                    else:
                        print(f"[WARN] Trace {tr.id} missing theoretical phase '{phase_theo}' — skipped.")

            elif phase_pick:
                for tr in self.stream:
                    picks = getattr(tr.stats, "picks", [])
                    pick_time = None
                    for pick in picks:
                        if pick.get("phase") == phase_pick:
                            pick_time = pick.get("time")
                            break
                    if pick_time is not None:
                        shift_amount = UTCDateTime(pick_time) - tr.stats.starttime
                        tr.stats.starttime = UTCDateTime(0)
                        tr.data = tr.data[int(shift_amount / tr.stats.delta):]
                    else:
                        print(f"[WARN] Trace {tr.id} missing picked phase '{phase_pick}' — skipped.")

            elif time_shifts:
                for i, shift_val in enumerate(time_shifts):
                    if i < len(self.stream):
                        self.stream[i].stats.starttime += shift_val

            else:
                print("[WARN] No valid shift config found (phase, phase_theo, or time_shifts). No action taken.")

        except Exception as e:
            print(f"[ERROR] apply_shift failed: {e}")

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

    def apply_beam(self, step_config):
        bm = TraceBeamResult(stream=self.stream, overlap=step_config["overlap"], fmin=step_config["fmin"],
                             fmax=step_config["fmax"], smax=step_config["smax"], slow_grid=step_config["slow_grid"],
                             inventory=self.inventory, method=step_config["method"])

        bm.compute_beam()
        bm.to_pickle(folder_path=step_config["output_folder"])

    def apply_kurtosis(self, step_config):
        cf_kurt = CFKurtosis(self.stream, step_config["CF_decay_win"], 4,  step_config["fmin"], step_config["fmax"])
        self.stream = cf_kurt.run_kurtosis()


    def apply_pm(self, step_config):

        """
        Run particle motion analysis for current displayed stream.
        Usage: pm
        """

        from collections import defaultdict

        if "output_path" not in step_config.keys():
            write_output = None
        else:
            write_output = step_config["output_path"]

        common_start = max(tr.stats.starttime for tr in self.stream)
        common_end = min(tr.stats.endtime for tr in self.stream)
        self.stream.trim(starttime=common_start, endtime=common_end, pad=True, fill_value=0)

        trace_list = list(self.stream)
        if not trace_list:
            print("[WARN] No traces currently displayed.")
            return

        # Group traces by (net, sta, loc)
        grouped = defaultdict(list)
        for tr in trace_list:
            key = (tr.stats.network, tr.stats.station, tr.stats.location)
            grouped[key].append(tr)

        valid_sets = []
        accepted_combos = [
            ("Z", "N", "E"),
            ("Z", "1", "2"),
            ("Z", "Y", "X")
        ]

        for key, traces in grouped.items():
            comp_map = {tr.stats.channel[-1].upper(): tr for tr in traces}

            for names in accepted_combos:
                if all(c in comp_map for c in names):
                    # Reorder as Z, N, E regardless of naming
                    z, n, e = comp_map[names[0]].copy(), comp_map[names[1]].copy(), comp_map[names[2]].copy()
                    print(f"[INFO] Mapping channels {names} → Z, N, E for station {key[1]}")

                    try:

                        min_len = min(len(z.data), len(n.data), len(e.data))
                        z.data, n.data, e.data = z.data[:min_len], n.data[:min_len], e.data[:min_len]

                        valid_sets.append((z, n, e))
                    except Exception as err:
                        print(f"[WARN] Failed trimming for station {key[1]}: {err}")
                    break  # only process the first valid combo

        if not valid_sets:
            print("[WARN] No valid 3-component sets (ZNE, Z12, ZYX) found.")
            return

        for z, n, e in valid_sets:
            print(f"[INFO] Plotting particle motion for {z.id}")
            particle_motion(z, n, e, save_path=write_output)
