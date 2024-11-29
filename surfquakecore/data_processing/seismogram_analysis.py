from obspy import read, Stream, read_inventory
import copy
import scipy, numpy as np
import math
import pywt

from surfquakecore.Structures.structures import TracerStatsAnalysis, TracerStats


class SeismogramData:

    def __init__(self, file_path,  realtime = False, **kwargs):

        stream = kwargs.pop('stream', [])

        #self.config_file = config_file
        self.config_keys = None
        #self.output = output_path

        if file_path:
            #self.st = read(file_path[0])
            self.st = read(file_path)
        if realtime:
            self.__tracer = stream

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

    def __send_filter_error_callback(self, func, msg):
        if func:
            func(msg)

    def resample_check(self, start_time=None, end_time=None):

        decimator_factor = None
        check = False

        if start_time is not None and end_time is not None:
            start = start_time
            end = end_time
        else:
            start = self.stats.StartTime
            end = self.stats.EndTime

        diff = end - start

        lim1 = 3600 * 6
        lim2 = 3600 * 3

        if diff >= lim1:
            check = True
            decimator_factor = 1
            return [decimator_factor, check]

        if diff >= lim2 and diff < lim1:
            check = True
            decimator_factor = 5

            return [decimator_factor, check]

        else:
            return [decimator_factor, check]

    def run_analysis(self, config, **kwargs):

        """
        This method should be to loop over config files and run the inversion.
        Previously it is needed to load the project and metadata.

        Args:
            analysis_config: a .yaml file

        Returns:

        """

        start_time = kwargs.get("start_time", self.stats.StartTime)
        end_time = kwargs.get("end_time", self.stats.EndTime)
        trace_number = kwargs.get("trace_number", 0)
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
                tr.differentiate()

            if _config['name'] == 'integrate':
                tr.integrate(method=_config['method'])

            if _config['name'] == 'filter':
                options = copy.deepcopy(_config)
                del options['method']
                del options['name']
                tr.filter(type=_config['method'], **options)

            if _config['name'] == 'wiener_filter':
                tr = self.wiener_filter(tr, time_window=_config['time_window'],
                                   noise_power=_config['noise_power'])
            #Borrar shift
            #if _config['name'] == 'shift':
            #    shifts = self.config_file['shift']
            #    i = 0
            #    for i in range(0,len(shifts)):
            #        tr.stats.starttime = tr.stats.starttime + shifts[i]['time']

            if _config['name'] == 'remove_response':
                inventory = read_inventory(_config['inventory'])
                print(inventory)
                if _config['units'] != "Wood Anderson":
                    # print("Deconvolving")
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

            if _config['name'] == 'add_white_noise':
                tr = self.add_white_noise(tr,_config['SNR_dB'])

            if _config['name'] == 'whitening':
                tr = self.whiten(tr, _config['freq_width'], taper_edge = _config['taper_edge'])

            if _config['name'] == 'remove_spikes':
                tr = self.hampel(tr, _config['window_size'], _config['n'])

            if _config['name'] == 'time_normalization':
                tr = self.normalize(tr, norm_win=_config['norm_win'], norm_method=_config['method'])

            if _config['name'] == 'wavelet_denoise':
                tr = self.wavelet_denoise(tr, dwt = _config['dwt'], threshold=_config['threshold'])

            if _config['name'] == 'resample':
                tr.resample(sampling_rate=_config['sampling_rate'],window='hann',no_filter=_config['pre_filter'])

            if _config['name'] == 'fill_gaps':
                st = Stream(tr)
                st.merge(fill_value=_config['method'])
                tr = st[0]

            if _config['name'] == 'smoothing':
                tr = self.smoothing(tr, type=_config['method'], k=_config['time_window'], fwhm=_config['FWHM'])

        return tr

    def get_waveform_advanced(self, parameters, inventory=None, filter_error_callback=None, **kwargs):

        start_time = kwargs.get("start_time", self.stats.StartTime)
        end_time = kwargs.get("end_time", self.stats.EndTime)
        trace_number = kwargs.get("trace_number", 0)
        tr = self.tracer

        tr.trim(starttime=start_time, endtime=end_time)

        # Detrend
        if parameters.rmean is not None:
            if parameters.rmean in ['linear, simple', 'demean']:
                tr.detrend(type=parameters.rmean)
            elif parameters.rmean is 'polynomial':
                tr.detrend(type=parameters.rmean, order=parameters.order)
            elif parameters.rmean is 'spline':
                tr.detrend(type=parameters.rmean, order=parameters.order, dspline=parameters.dspline)

        # Taper

        if parameters.taper is not None:
            tr.taper(type=parameters.taper, max_percentage=0.5)



        return tr

    def wiener_filter(self, tr, time_window, noise_power):
        data = tr.data

        if time_window == 0 and noise_power == 0:

            denoise = scipy.signal.wiener(data, mysize=None, noise=None)
            tr.data = denoise

        elif time_window != 0 and noise_power == 0:

            denoise = scipy.signal.wiener(data, mysize=int(time_window * tr.stats.sampling_rate), noise=None)
            tr.data = denoise

        elif time_window == 0 and noise_power != 0:

            noise = noise_power * np.std(data)
            noise = int(noise)
            denoise = scipy.signal.wiener(data, mysize=None, noise=noise)
            tr.data = denoise

        elif time_window != 0 and noise_power != 0:

            noise = noise_power * np.std(data)
            noise = int(noise)
            denoise = scipy.signal.wiener(data, mysize=int(time_window * tr.stats.sampling_rate), noise=noise)
            tr.data = denoise

        return tr

    def add_white_noise(self, tr, SNR_dB):
        L = len(tr.data)
        SNR = 10 ** (SNR_dB / 10)
        Esym = np.sum(np.abs(tr.data) ** 2) / L
        N0 = Esym / SNR
        noiseSigma = np.sqrt(N0)
        n = noiseSigma * np.random.normal(size=L)
        tr.data = tr.data + n
        return tr

    def whiten(self, tr, freq_width=0.05, taper_edge=True):

        """"
        freq_width: Frequency smoothing windows [Hz] / both sides
        taper_edge: taper with cosine window  the low frequencies

        return: whithened trace (Phase is not modified)
        """""

        fs = tr.stats.sampling_rate
        N = tr.count()
        D = 2 ** math.ceil(math.log2(N))
        freq_res = 1 / (D / fs)
        # N_smooth = int(freq_width / (2 * freq_res))
        N_smooth = int(freq_width / (freq_res))

        if N_smooth % 2 == 0:  # To have a central point
            N_smooth = N_smooth + 1
        else:
            pass

        # avarage_window_width = (2 * N_smooth + 1) #Denominador
        avarage_window_width = (N_smooth + 1)  # Denominador
        half_width = int((N_smooth + 1) / 2)  # midpoint
        half_width_pos = half_width - 1

        # Prefilt
        tr.detrend(type='simple')
        tr.taper(max_percentage=0.05)

        # ready to whiten
        data = tr.data
        data_f = np.fft.rfft(data, D)
        freq = np.fft.rfftfreq(D, 1. / fs)
        N_rfft = len(data_f)
        data_f_whiten = data_f.copy()
        index = np.arange(0, N_rfft - half_width, 1)

        data_f_whiten = self.whiten_aux(data_f, data_f_whiten, index, half_width, avarage_window_width, half_width_pos)

        # Taper (optional) and remove mean diffs in edges of the frequency domain

        wf = (np.cos(np.linspace(np.pi / 2, np.pi, half_width)) ** 2)

        if taper_edge:

            diff_mean = np.abs(np.mean(np.abs(data_f[0:half_width])) - np.mean(np.abs(data_f_whiten[half_width:])) * wf)

        else:

            diff_mean = np.abs(np.mean(np.abs(data_f[0:half_width])) - np.mean(np.abs(data_f_whiten[half_width:])))

        diff_mean2 = np.abs(
            np.mean(np.abs(data_f[(N_rfft - half_width):])) - np.mean(np.abs(data_f_whiten[(N_rfft - half_width):])))

        if taper_edge:

            data_f_whiten[0:half_width] = ((data_f[0:half_width]) / diff_mean) * wf  # First part of spectrum tapered
        else:

            data_f_whiten[0:half_width] = ((data_f[0:half_width]) / diff_mean)

        data_f_whiten[(N_rfft - half_width):] = (data_f[(N_rfft - half_width):]) / diff_mean2  # end of spectrum
        data = np.fft.irfft(data_f_whiten)
        data = data[0:N]
        tr.data = data

        return tr

    def whiten_aux(self,data_f, data_f_whiten, index, half_width, avarage_window_width, half_width_pos):
        return self.__whiten_aux(data_f, data_f_whiten, index, half_width, avarage_window_width, half_width_pos)

    def __whiten_aux(self, data_f, data_f_whiten, index, half_width, avarage_window_width, half_width_pos):
        for j in index:
            den = np.sum(np.abs(data_f[j:j + 2 * half_width])) / avarage_window_width
            data_f_whiten[j + half_width_pos] = data_f[j + half_width_pos] / den
        return data_f_whiten

    def hampel_aux(self, input_series, window_size, size, n_sigmas):
        return self.__hampel_aux(input_series, window_size, size, n_sigmas)

    def hampel(self, tr, window_size, n_sigmas=3):
        """
                Median absolute deviation (MAD) outlier in Time Series
                :param ts: a trace obspy object representing the timeseries
                :param window_size: total window size in seconds
                :param n: threshold, default is 3 (Pearson's rule)
                :return: Returns the corrected timeseries
                """

        size = tr.count()
        input_series = tr.data
        window_size = int(window_size * tr.stats.sampling_rate)
        tr.data = self.hampel_aux(input_series, window_size, size, n_sigmas)

        return tr

    def __hampel_aux(self, input_series, window_size, size, n_sigmas):

        k = 1.4826  # scale factor for Gaussian distribution
        # indices = []
        new_series = input_series.copy()
        # possibly use np.nanmedian
        for i in range((window_size), (size - window_size)):
            x0 = np.median(input_series[(i - window_size):(i + window_size)])
            S0 = k * np.median(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
            if (np.abs(input_series[i] - x0) > n_sigmas * S0):
                new_series[i] = x0
                # indices.append(i)

        return new_series

    def normalize(self, tr, clip_factor=6, clip_weight=10, norm_win=None, norm_method="1bit"):
        if norm_method == 'clipping':
            lim = clip_factor * np.std(tr.data)
            tr.data[tr.data > lim] = lim
            tr.data[tr.data < -lim] = -lim

        elif norm_method == "clipping_iteration":
            lim = clip_factor * np.std(np.abs(tr.data))

            # as long as still values left above the waterlevel, clip_weight
            while tr.data[np.abs(tr.data) > lim] != []:
                tr.data[tr.data > lim] /= clip_weight
                tr.data[tr.data < -lim] /= clip_weight

        elif norm_method == 'time_normalization':
            lwin = int(tr.stats.sampling_rate * norm_win)
            st = 0  # starting point
            N = lwin  # ending point

            while N < tr.stats.npts:
                win = tr.data[st:N]

                w = np.mean(np.abs(win)) / (2. * lwin + 1)

                # weight center of window
                tr.data[st + int(lwin / 2)] /= w

                # shift window
                st += 1
                N += 1

            # taper edges
            # taper = get_window(tr.stats.npts)
            # tr.data *= taper

        elif norm_method == "1bit":
            tr.data = np.sign(tr.data)
            tr.data = np.float32(tr.data)

        return tr

    def wavelet_denoise(self, tr, threshold=0.04, dwt='sym4'):
        # Threshold for filtering
        # Create wavelet object and define parameters
        w = pywt.Wavelet(dwt)
        maxlev = pywt.dwt_max_level(len(tr.data), w.dec_len)
        # Decompose into wavelet components, to the level selected:
        coeffs = pywt.wavedec(tr.data, dwt, level=maxlev)
        # cA = pywt.threshold(cA, threshold*max(cA))
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
        datarec = pywt.waverec(coeffs, dwt)
        tr.data = datarec
        return tr

    def smoothing(self, tr, type='gaussian', k=5, fwhm=0.05):
        # k window size in seconds

        n = len(tr.data)

        if type == 'mean':
            k = int(k * tr.stats.sampling_rate)

            # initialize filtered signal vector
            filtsig = np.zeros(n)
            for i in range(k, n - k - 1):
                # each point is the average of k surrounding points
                # print(i - k,i + k)
                filtsig[i] = np.mean(tr.data[i - k:i + k])

            tr.data = filtsig

        if type == 'gaussian':
            ## create Gaussian kernel
            # full-width half-maximum: the key Gaussian parameter in seconds
            # normalized time vector in seconds
            k = int(k * tr.stats.sampling_rate)
            fwhm = int(fwhm * tr.stats.sampling_rate)
            gtime = np.arange(-k, k)
            # create Gaussian window
            gauswin = np.exp(-(4 * np.log(2) * gtime ** 2) / fwhm ** 2)
            # compute empirical FWHM

            pstPeakHalf = k + np.argmin((gauswin[k:] - .5) ** 2)
            prePeakHalf = np.argmin((gauswin - .5) ** 2)

            # then normalize Gaussian to unit energy
            gauswin = gauswin / np.sum(gauswin)
            # implement the filter
            # initialize filtered signal vector
            filtsigG = copy.deepcopy(tr.data)
            # implement the running mean filter
            for i in range(k + 1, n - k - 1):
                # each point is the weighted average of k surrounding points
                filtsigG[i] = np.sum(tr.data[i - k:i + k] * gauswin)

            tr.data = filtsigG

        if type == 'tkeo':
            # extract needed variables

            emg = tr.data

            # initialize filtered signal
            emgf = copy.deepcopy(emg)

            # the loop version for interpretability
            # for i in range(1, len(emgf) - 1):
            #    emgf[i] = emg[i] ** 2 - emg[i - 1] * emg[i + 1]

            # the vectorized version for speed and elegance

            emgf[1:-1] = emg[1:-1] ** 2 - emg[0:-2] * emg[2:]

            ## convert both signals to zscore

            # find timepoint zero
            # time0 = np.argmin(emgtime ** 2)

            # convert original EMG to z-score from time-zero
            # emgZ = (emg - np.mean(emg[0:time0])) / np.std(emg[0:time0])

            # same for filtered EMG energy
            # emgZf = (emgf - np.mean(emgf[0:time0])) / np.std(emgf[0:time0])
            # tr.data = emgZf
            tr.data = emgf

        return tr
