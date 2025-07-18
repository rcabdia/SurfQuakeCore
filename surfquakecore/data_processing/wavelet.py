import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import scipy
from deprecated import deprecated
from obspy import read, Stream, Trace, UTCDateTime
from obspy.signal.filter import lowpass
from scipy.signal import argrelextrema
from Exceptions.exceptions import InvalidFile
from surfquakecore.Structures.structures import TracerStats
from surfquakecore.utils.obspy_utils import ObspyUtil, MseedUtil


class ConvolveWaveletBase:
    """
    Class to apply wavelet convolution to a mseed file. The bank of atoms is computed at the class initialisation.
        Examples
        --------
        cw = ConvolveWavelet(file_path)
        convolve = cw.ccwt_ba_fast()
    """

    def __init__(self, data, **kwargs):
        """
        Class to apply wavelet convolution to a mseed file.
        The bank of atoms is computed at the class initialisation.
        :param data: Either the mseed file path or an obspy Tracer.
        :keyword kwargs:
        :keyword wmin: Minimum number of cycles. Default = 6.
        :keyword wmax: Maximum number of cycles. Default = 6.
        :keyword tt: Period of the Morlet Wavelet. Default = 2.
        :keyword fmin: Minimum Central frequency (in Hz). Default = 2.
        :keyword fmax: Maximum Central frequency (in Hz). Default = 12.
        :keyword m: Parameter for Paul Wavelet. Default = 30.
        :keyword nf: Number of logarithmically spaced frequencies between fmin and fmax. Default = 20.
        :keyword use_wavelet: Default = Complex Morlet
        :keyword use_rfft: True if it should use rfft instead of fft. Default = True.
        :keyword decimate: True if it should try to decimate the trace. Default = False. The decimation
            factor is equal to q = 0.4*SR/fmax. For SR=200Hz and fmax=40Hz, q=2. This will downsize the
            sample rate to 100 Hz.
        :raise InvalidFile: If file is not a valid mseed.
        :example:
        >>> cw = ConvolveWavelet(data)
        >>> cw.setup_wavelet()
        >>> sc = cw.scalogram_in_dbs
        >>> cf = cw.cf_lowpass()
        """

        if isinstance(data, Trace):
            #self.stats = TracerStats.from_dict(data.stats)
            self.trace: Trace = data
            self.stats = self.trace.stats.copy()

        elif isinstance(data, Stream):
            self.st: Stream = data
            #self.stats = TracerStats.from_dict(self.st[0].stats)
            self._decimate_stream = kwargs.get("decimate_stream", False)
        else:
            if not MseedUtil.is_valid_mseed(data):
                raise InvalidFile("The file: {} is not a valid mseed.".format(data))
            self.trace: Trace = read(data)[0]
            self.stats = ObspyUtil.get_stats(data)

        self._wmin = float(kwargs.get("wmin", 6.))
        self._wmax = float(kwargs.get("wmax", 6.))
        self._tt = float(kwargs.get("tt", 2.))
        self._fmin = float(kwargs.get("fmin", 2.))
        self._fmax = float(kwargs.get("fmax", 12.))
        self._nf = int(kwargs.get("nf", 20))
        self._use_wavelet = kwargs.get("use_wavelet", "Complex Morlet")
        self._m = int(kwargs.get("m", 30))
        self._use_rfft = kwargs.get("use_rfft", False)
        self._decimate = kwargs.get("decimate", False)
        #if self.stats:
            #self._validate_kwargs()
            # print(self.stats)

        self._data = None
        self._npts = 0
        self._tf = None
        self._start_time = self.trace.stats.starttime
        self._end_time = self.trace.stats.endtime
        self._sample_rate = self.trace.stats.sampling_rate

        self._frex = None
        self._n_cycles = None
        self._wtime = None
        self._half_wave = None

    def __repr__(self):
        return "ConvolveWavelet(data={}, wmin={}, wmax={}, tt={}, fmin={}, fmax={}, nf={})".format(
            self.trace, self._wmin, self._wmax, self._tt, self._fmin, self._fmax, self._nf)

    def __eq__(self, other):
        # noinspection PyProtectedMember
        return self.trace == other.trace and self._wmin == other._wmin and self._wmax == other._wmax \
               and self._tt == other._tt and self._fmin == other._fmin and self._fmax == other._fmax \
               and self._nf == other._nf and self._use_rfft == other._use_rfft \
               and self._start_time == other._start_time and self._end_time == other._end_time \
               and self._decimate == other._decimate and self._use_wavelet == other._use_wavelet

    def _validate_data(self):
        if self._data is None:
            raise AttributeError("Data not found. Run setup_wavelet().")

    def _validate_kwargs(self):
        if self._wmax < self._wmin:
            AttributeError("The kwarg wmin can't be bigger than wmax. wmin = {}, wmax = {}".
                           format(self._wmin, self._wmax))

        if self._fmax < self._fmin:
            AttributeError("The kwarg fmin can't be bigger than fmax. fmin = {}, fmax = {}".
                           format(self._fmin, self._fmax))

    @property
    def npts(self):
        return self._npts

    def filter_win(self, freq, index):

        if self._use_wavelet == "Paul":

            num = (2 ** self._m * 1j * np.math.factorial(self._m))
            den = np.sqrt(np.pi * np.math.factorial((2 * self._m)))
            s = (2 * self._m + 1) / (freq * np.pi * 4)
            mu = self._wtime / s

            p = 1 / (1 - 1j * mu) ** (self._m + 1)
            cmw = (num * p) / den

        elif self._use_wavelet == "Mexican Hat":
            m = 2
            sigma = (2 * m + 1) / (4 * np.pi * freq)
            k1 = 2 / ((np.pi ** 0.25) * np.sqrt(2 * sigma))
            k2 = ((self._wtime ** 2) / (sigma ** 2)) - 1
            k3 = np.exp((-1 * self._wtime ** 2) / (2 * sigma ** 2))
            cmw = -1 * k1 * k2 * k3

        else:

            # Create the Morlet wavelet and get its fft
            s = self._n_cycles[index] / (2 * np.pi * freq)
            # Normalize Factor
            normalization = 1 / (np.pi * s ** 2) ** 0.25
            # Complex sine = np.multiply(1j*2*(np.pi)*frex[fi],wtime))
            # Gaussian = np.exp(-1*np.divide(np.power(wtime,2),2*s**2))
            cmw = np.multiply(np.exp(np.multiply(1j * 2 * np.pi * freq, self._wtime)),
                              np.exp(-1 * np.divide(np.power(self._wtime, 2), 2 * s ** 2)))
            cmw = cmw.conjugate()
            # Normalizing. The square root term causes the wavelet to be normalized to have an energy of 1.
            cmw = normalization * cmw
        #if self._use_rfft:
        #   cmw = np.real(cmw)
        #    print("R.Morlet Wavelet")

        return cmw

    def setup_wavelet(self, start_time: UTCDateTime = None, end_time: UTCDateTime = None, **kwargs):
        """
        Recompute the bank of atoms based on the new kwargs and the waveform data range. If start_time or end_time
        is not given then it will read the whole data from the mseed file.
        :param start_time: The start time of the waveform data. If not given default is the start time from the
            mseed header.
        :param end_time: The end time of the waveform data. If not given default is the end time from the
            mseed header.
        :keyword  kwargs:
        :keyword wmin: Minimum number of cycles.
        :keyword wmax: Maximum number of cycles.
        :keyword tt: Central frequency of the Morlet Wavelet.
        :keyword fmin: Minimum frequency (in Hz).
        :keyword fmax: Maximum frequency (in Hz).
        :keyword nf: Number of logarithmically spaced frequencies between fmin and fmax.
        :return:
        """

        self._start_time = start_time if start_time else self.stats.starttime
        self._end_time = end_time if end_time else self.stats.endtime

        self.__setup_wavelet(start_time, end_time, **kwargs)

    def setup_atoms(self, **kwargs):
        """
        Recompute the bank of atoms based on the new kwargs. This method will only recompute the atoms. Use
        :class:`setup_wavelet()` if you want to change the data.
        :keyword  kwargs:
        :keyword wmin: Minimum number of cycles.
        :keyword wmax: Maximum number of cycles.
        :keyword tt: Central frequency of the Morlet Wavelet.
        :keyword fmin: Minimum frequency (in Hz).
        :keyword fmax: Maximum frequency (in Hz).
        :keyword nf: Number of logarithmically spaced frequencies between fmin and fmax.
        :return:
        """
        self._wmin = float(kwargs.get("wmin", self._wmin))
        self._wmax = float(kwargs.get("wmax", self._wmax))
        self._tt = float(kwargs.get("tt", self._tt))
        self._fmin = float(kwargs.get("fmin", self._fmin))
        self._fmax = float(kwargs.get("fmax", self._fmax))
        self._nf = int(kwargs.get("nf", self._nf))
        self._use_rfft = kwargs.get("use_rfft", False)
        self._use_wavelet = kwargs.get("use_wavelet", "Complex Morlet")
        self._validate_kwargs()
        self._tf = None  # Makes tf none to force to recompute tf when calling other methods.
        self._setup_atoms()

    def _setup_atoms(self):
        self._validate_data()

        self._frex = np.logspace(np.log10(self._fmin), np.log10(self._fmax), self._nf, base=10)
        self._n_cycles = np.linspace(self._wmin, self._wmax, self._nf)
        dt = 1 / self._sample_rate
        self._wtime = np.arange(-self._tt, self._tt + dt, dt)  # Kernel of the Mother Morlet Wavelet
        self._half_wave = (len(self._wtime) - 1) / 2

    def __get_data_in_time(self, start_time, end_time):
        tr = self.trace.copy()
        tr.trim(starttime=start_time, endtime=end_time)
        if self._decimate:
            tr = self.decimate_data(tr)
        tr.detrend(type='demean')
        tr.taper(max_percentage=0.05)
        self._npts = tr.stats.npts
        self._sample_rate = tr.stats.sampling_rate
        return tr.data

    def __get_data_in_time_stream(self, start_time, end_time):

        self.st.trim(starttime=start_time, endtime=end_time)
        # if self._decimate_stream:
        #     self.st = self.decimate_data_stream(self.st)

        self.st.detrend(type='demean')
        self.st.taper(max_percentage=0.025)

        self._sample_rate = self.st[0].stats.sampling_rate

        return self.st

    def __get_resample_factor(self):
        rf = int(0.4 * self._sample_rate / self._fmax)
        return rf

    def decimate_data_stream(self, st: Stream, fs_new=10):


        st.resample(fs_new)

        return st

    def decimate_data(self, tr: Trace):
        rf = self.__get_resample_factor()
        if rf > 1:
            data = scipy.signal.decimate(tr.data, rf, ftype='fir', zero_phase=True)
            new_stats = tr.stats
            new_stats["npts"] = len(data)
            new_stats["sampling_rate"] /= rf
            new_stats["delta"] = 1. / new_stats["sampling_rate"]
            return Trace(data, new_stats)

        return tr

    def get_nproc(self):
        total_cpu = multiprocessing.cpu_count()
        nproc = total_cpu - 2 if total_cpu > 3 else total_cpu - 1  # avoid to get all available cores.
        nproc = min(nproc, self._nf)
        nproc = max(nproc, 1)
        return nproc

    @staticmethod
    def __tapper(data, max_percentage=0.05):
        tr = Trace(data)
        tr.taper(max_percentage=max_percentage, type='blackman')
        return tr.data

    def __setup_wavelet(self, start_time: UTCDateTime, end_time: UTCDateTime, **kwargs):
        setup_wavelets_stream = kwargs.get("setup_wavelets_stream", False)
        if setup_wavelets_stream:
            self._data = self.__get_data_in_time_stream(start_time, end_time)

        else:

            self._data = self.__get_data_in_time(start_time, end_time)

        self.setup_atoms(**kwargs)

    def _convolve_atoms(self, parallel: bool):
        # implement at the child.
        pass

    def scalogram_in_dbs(self):
        if self._tf is None:
            self.compute_tf()

        sc = np.abs(self._tf) ** 2
        return 10. * (np.log10(sc / np.max(sc)))

    def scalogram(self):
        if self._tf is None:
            self.compute_tf()

        return self._tf


    def phase(self):

        if self._tf is None:
            self.compute_tf()

        phase = np.unwrap(np.angle(self._tf), axis = 0)
        # Following convencional numerical diferentiation
        # inst_freq = np.abs(np.diff(phase, axis = 1))
        # Following Synchrosqueezing wavelet transform
        #np.diff(self._tf, axis=1) can be differentiated in frequency domain --> np.fft.fft(self._tf, axis=1)*2*np.pi*1j
        a,b = np.shape(self._tf)
        freq = np.fft.fftfreq(len(self._tf[0,:]), d = (1/self._sample_rate))
        freq = np.tile(freq, (a,1))
        derivate_freq = np.fft.ifft(2*np.pi*1j*freq*np.fft.fft(self._tf, axis=1))
        #inst_freq = (np.diff(self._tf, axis=1))/(self._tf[:,1:])
        inst_freq =  derivate_freq / (2*np.pi*1j*(self._tf))
        inst_freq = np.abs(inst_freq)
        ins_freq_hz = (inst_freq*self._sample_rate)/2*np.pi
        return phase, inst_freq, ins_freq_hz

    def get_data_window(self):
        start = int(self._half_wave + 1)
        end = self._npts + int(self._half_wave + 1)
        return start, end

    # def __ccwt_ba_multitread(self):
    #     nproc = self.get_nproc()
    #     nproc = min(nproc, len(self._data))
    #
    #     with ThreadPool(nproc) as pool:
    #         ro = pool.map(self.__cwt_ba, self._data)
    #
    #     cwt = np.array([]).reshape(self._nf, 0)
    #     for index, r in enumerate(ro):
    #         cwt = np.concatenate((cwt, r), axis=1)
    #
    #     return cwt

    def compute_tf(self, parallel=True):
        pass

    def cf(self, tapper=True, parallel=True):
        """
        Characteristic function.
        Compute the mean values of the log10 differences of the convolved waveform with the wavelet from fmin to fmax.
        :param tapper: True for tapper the result. Default=True.
        :param parallel: Either or not it should run cwt in parallel. Default=True.
        :return: Mean values of the log10 difference of the convolved waveform with the wavelet from fmin to fmax.
        """

        if self._tf is None:
            self.compute_tf(parallel=parallel)

        cf = np.mean(np.diff(np.log10(np.abs(self._tf) ** 2)), axis=0, dtype=np.float32)

        if tapper:
            cf = self.__tapper(cf)

        return cf

    def cf_lowpass(self, tapper=True, parallel=True, freq=0.15):
        """
        Characteristic function with lowpass.
        Compute the mean values of the log10 differences of the convolved waveform with the wavelet from fmin to fmax
        with a low pass filter.
        :param tapper: True for tapper the result. Default=True.
        :param parallel: Either or not it should run cwt in parallel. Default=True.
        :param freq: Filter corner frequency. Default=0.15.
        :return: The filtered (lowpass, fmin=0.15) mean values of the log10 difference of the convolved waveform with
            the wavelet from fmin to fmax.
        """

        cf = lowpass(self.cf(tapper, parallel=parallel), freq, df=self._sample_rate, corners=3, zerophase=True)

        return cf

    def charachteristic_function_kurt_stream(self, window_size_seconds=5, parallel=False, stream=True):

        kurt_st = []
        if self._tf is None:
            self.st.normalize()
            self.compute_tf(parallel=parallel, stream=stream)

        for item in self._tf:
            stats = item[1]
            print("Processing Kurtosis", stats.station, stats.channel)
            pow_scalogram = np.abs(item[0])**2
            kurtosis_values, time_vector = self.conventional_kurtosis(pow_scalogram, window_size_seconds=window_size_seconds,
                                                                      sampling_rate=self._sample_rate)

            time_vector_resample = np.linspace(time_vector[0], time_vector[-1], int(time_vector[-1]*self._sample_rate))

            kurtosis_values_resample = np.interp(time_vector_resample, time_vector, kurtosis_values)

            # Create Trace object with the synthetic data
            tr_kurt = Trace(data=kurtosis_values_resample, header=stats)
            kurt_st.append(tr_kurt)

        kurt_st = Stream(kurt_st)

        kurt_st.detrend(type="simple")
        kurt_st.detrend(type='constant')
        # ...and the linear trend...
        kurt_st.detrend(type='linear')
        kurt_st.taper(max_percentage=0.05, type="blackman")
        kurt_st.filter(type='lowpass', freq=0.15, zerophase=True, corners=4)

        kurt_st.detrend(type="simple")
        kurt_st.detrend(type='constant')
        # ...and the linear trend...
        kurt_st.detrend(type='linear')
        kurt_st.taper(max_percentage=0.05,  type="blackman")

        return kurt_st

    def charachteristic_function_kurt(self, window_size_seconds=5, parallel=True):

        if self._tf is None:
            self._data = self._data/np.max(self._data)
            self.compute_tf(parallel=parallel)

        pow_scalogram = np.abs(self._tf)**2
        kurtosis_values, time_vector = self.conventional_kurtosis(pow_scalogram, window_size_seconds=window_size_seconds,
                                                                  sampling_rate=self._sample_rate)

        time_vector_resample = np.linspace(time_vector[0], time_vector[-1], int(time_vector[-1]*self._sample_rate))

        kurtosis_values_resample = np.interp(time_vector_resample, time_vector, kurtosis_values)

        # Create Trace object with the synthetic data
        tr_kurt = Trace(data=kurtosis_values_resample)

        # Set trace metadata
        tr_kurt.stats.station = self.trace.stats.station  # Station name
        tr_kurt.stats.network = self.trace.stats.network # Network code
        tr_kurt.stats.channel = self.trace.stats.channel  # Channel code
        tr_kurt.stats.starttime = self.trace.stats.starttime + time_vector_resample[0] # Set to current time as start time
        tr_kurt.stats.sampling_rate = self.trace.stats.sampling_rate
        tr_kurt.detrend(type="simple")
        tr_kurt.detrend(type='constant')
        # ...and the linear trend...
        tr_kurt.detrend(type='linear')
        tr_kurt.taper(max_percentage=0.05, type="blackman")
        # tr_kurt.filter(type='lowpass', freq=0.15, zerophase=True, corners=4)
        #
        # tr_kurt.detrend(type="simple")
        # tr_kurt.detrend(type='constant')
        # # ...and the linear trend...
        # tr_kurt.detrend(type='linear')
        # tr_kurt.taper(max_percentage=0.05, type="blackman")

        return tr_kurt
    def conventional_kurtosis(self, data, window_size_seconds, sampling_rate):

        n = data.shape[1]
        window_size_samples = int(window_size_seconds * sampling_rate)
        slide = int(sampling_rate/2)

        # Call the Numba-accelerated helper function
        kurtosis_values = self._conventional_kurtosis_helper(data, window_size_samples, slide, n)

        # Create time vector
        time_vector = np.linspace(0, int((n - window_size_samples) / sampling_rate), len(kurtosis_values)) + int(window_size_seconds)
        time_vector = time_vector[0:-1]
        kurtosis_values = np.abs(np.diff(kurtosis_values))
        return kurtosis_values, time_vector

    def _conventional_kurtosis_helper(self, data, window_size_samples, slide, n):
        kurtosis_values = []

        # Loop through the data with the sliding window
        for i in range(0, n - window_size_samples + 1, slide):
            window_data = data[:, i:i + window_size_samples]  # Extract data in window

            # Compute mean for the window
            mean = np.mean(window_data)

            # Compute variance (second central moment)
            variance = np.mean((window_data - mean) ** 2)

            # Compute fourth central moment
            fourth_moment = np.mean((window_data - mean) ** 4)

            # Compute kurtosis (excess kurtosis)
            if variance > 1e-10:  # Ensure variance is not effectively zero
                kurtosis = (fourth_moment / (variance ** 2)) - 3  # Subtract 3 for excess kurtosis

                # Check for extremely large kurtosis and cap it
                if not np.isfinite(kurtosis):  # Handles inf and NaN cases
                    kurtosis = 0.0
                elif abs(kurtosis) > 1e6:  # Prevent unreasonably large values
                    kurtosis = np.sign(kurtosis) * 1e6
            else:
                kurtosis = 1e-2  # Set kurtosis to 0 if variance is effectively zero

            # Append result to list
            kurtosis_values.append(kurtosis)

        return np.array(kurtosis_values)

    def get_time_delay(self):
        """
        Compute the time delay in seconds of the wavelet.
        :return: The time delay of the wavelet in seconds.
        """
        return 0.5 * self._wmin / (2. * np.pi * self._fmin)

    def detect_max_pick_in_time(self, data: np.ndarray):
        """
        Get the time of the maximum pick.
        :param data: The data from ccwt_ba_fast method.
        :return: Return the obspy.UTCDateTime at the maximum pick if detected. If there is no pick
            above the detection limit it returns None.
        """
        filtered_data = np.abs(np.where(np.abs(data) >= self.get_detection_limit(data), data, 0.))
        if filtered_data.sum() != 0.:
            max_index = np.argmax(filtered_data)
            time_s = max_index / self._sample_rate
            return self._start_time + time_s + self.get_time_delay()
        return None

    def detect_picks_in_time(self, data: np.ndarray, sigmas=5.):
        """
        Get the times of the local maximums that are above the detection limit.
        :param data: The data from cf_lowpass method.
        :param sigmas: The detection limit in sigmas.
        :return: Return a list of obspy.UTCDateTime at the local maximums that are
            over the detection limit.
        """
        max_indexes = self.detect_picks(data, sigmas=sigmas)
        delay = self.get_time_delay()
        times_s = max_indexes / self._sample_rate
        events_time = []
        for t in times_s:
            events_time.append(self._start_time + t + delay)
        return events_time

    def detect_picks(self, data, sigmas=5.):
        """
        Get the local maximums that are above the detection limit.
        :param data: The data from cf_lowpass method.
        :param sigmas: The detection limit in sigmas.
        :return: Return the indexes of the local maximums over the detection limit.
        """
        limit = self.get_detection_limit(data, sigmas=sigmas)
        filtered_data = np.where(data >= limit, data, 0.)
        ind = scipy.signal.argrelextrema(filtered_data, np.greater)
        return ind[0]

    @staticmethod
    def get_detection_limit(data: np.ndarray, sigmas=5.):
        """
        Compute the detection limit of ccwt_ba_fast data.
        :param data: The data from cf_lowpass method.
        :param sigmas: The limit for detection. (Default=5 sigmas)
        :return: The detection limit sigmas * sqrt(variance)
        """
        var = np.sqrt(np.var(data))
        return sigmas * var


@deprecated(reason="You should use ConvolveWaveletScipy")
class ConvolveWavelet(ConvolveWaveletBase):
    """
    Class to apply wavelet convolution to a mseed file. The bank of atoms is computed at the class initialisation.
        Examples
        --------
        cw = ConvolveWavelet(file_path)
        convolve = cw.ccwt_ba_fast()
    """

    def __init__(self, data, **kwargs):
        """
        Class to apply wavelet convolution to a mseed file.
        The bank of atoms is computed at the class initialisation.
        :param data: Either the mseed file path or an obspy Tracer.
        :keyword kwargs:
        :keyword wmin: Minimum number of cycles. Default = 6.
        :keyword wmax: Maximum number of cycles. Default = 6.
        :keyword tt: Central frequency of the Morlet Wavelet. Default = 2.
        :keyword fmin: Minimum frequency (in Hz). Default = 2.
        :keyword fmax: Maximum frequency (in Hz). Default = 12.
        :keyword nf: Number of logarithmically spaced frequencies between fmin and fmax. Default = 20.
        :keyword use_rfft: True if it should use rfft instead of fft. Default = True.
        :keyword decimate: True if it should try to decimate the trace. Default = False. The decimation
            factor is equal to q = 0.4*SR/fmax. For SR=200Hz and fmax=40Hz, q=2. This will downsize the
            sample rate to 100 Hz.
        :raise TypeError: If file is not a valid mseed.
        :example:
        >>> cw = ConvolveWavelet(data)
        >>> cw.setup_wavelet()
        >>> cf = cw.cf_lowpass()
        """

        super(ConvolveWavelet, self).__init__(data, **kwargs)

        self.__conv = None  # convolution of ba and data_fft
        self.__n_conv = 0

    def _setup_atoms(self):
        super()._setup_atoms()
        self._convolve_atoms()

    # def __chop_data(self, delta_time, start_time: UTCDateTime, end_time: UTCDateTime):
    #     total_time = (end_time - start_time) / 3600.
    #     n = np.math.ceil(total_time / delta_time)
    #
    #     data_set = []
    #     for h in range(n):
    #         dt = h * 3600 * delta_time
    #         dt2 = (h + 1) * 3600 * delta_time
    #         data = self.__get_data_in_time(start_time + dt, start_time + dt2)
    #         if data is not None:
    #             self._npts = int(self.stats.Sampling_rate * delta_time * 3600) + 1
    #             data = self.__pad_data(data, self._npts)
    #             data_set.append(data)
    #
    #     return data_set

    def _convolve_atoms(self, parallel=False):

        # FFT parameters
        n_kern = len(self._wtime)
        self.__n_conv = 2 ** np.math.ceil(np.math.log2(self._npts + n_kern))

        # loop over frequencies
        array_size = self.__n_conv / 2 + 1 if self._use_rfft else self.__n_conv
        self.__conv = np.zeros((int(self._nf), int(array_size)), dtype=np.complex64)
        # FFT data
        if self._use_rfft:
            data_fft = np.fft.rfft(self._data, n=self.__n_conv)
        else:
            data_fft = np.fft.fft(self._data, n=self.__n_conv)

        for ii, fi in enumerate(self._frex):
            cmw = self.filter_win(fi, ii)
            if self._use_rfft:
                # Calculate the fft of the "atom"
                cmw_fft = np.fft.rfft(cmw, self.__n_conv)
            else:
                cmw_fft = np.fft.fft(cmw, self.__n_conv)

            # convolution of ba and data_fft.
            self.__conv[ii, :] = np.multiply(cmw_fft, data_fft, dtype=np.complex64)

    def __compute_cwt(self, data):
        start = int(self._half_wave + 1)
        end = self._npts + int(self._half_wave + 1)
        if self._use_rfft:
            cwt = np.fft.irfft(data)[start:end]
        else:
            cwt = np.fft.ifft(data, n=self.__n_conv)[start:end]
        # subtract the mean value
        cwt = cwt - np.mean(cwt, dtype=np.float32)
        return cwt

    def __cwt_ba(self, parallel=False):
        """
        Compute the time frequency or scalogram in time and frequency domain.
        :param parallel: True if it should run in parallel. If the computer has only 1 core this will have no effect.
        :return: The time frequency representation of the convolved waveform with the wavelet.
            The type is a np.array.
        """

        n_proc = self.get_nproc()
        if parallel and n_proc > 1:
            # pool = ThreadPool(processes=n_proc)
            # results = [pool.apply_async(self.__compute_cwt, args=(row,)) for row in m]
            # tf = np.array([p.get() for p in results], copy=False, dtype=np.float32)
            # pool.close()
            with ThreadPool(n_proc) as pool:
                tf = np.array(pool.map(self.__compute_cwt, self.__conv), copy=False, dtype=np.float32)

        else:
            tf = np.array([self.__compute_cwt(row) for row in self.__conv], copy=False, dtype=np.float32)

        # release conv from memory.
        self.__conv = None
        del self.__conv

        return tf

    # def __ccwt_ba_multitread(self):
    #     nproc = self.get_nproc()
    #     nproc = min(nproc, len(self._data))
    #
    #     with ThreadPool(nproc) as pool:
    #         ro = pool.map(self.__cwt_ba, self._data)
    #
    #     cwt = np.array([]).reshape(self._nf, 0)
    #     for index, r in enumerate(ro):
    #         cwt = np.concatenate((cwt, r), axis=1)
    #
    #     return cwt

    def compute_tf(self, parallel=True):
        """
        Compute the convolved waveform with the wavelet from fmin to fmax.
        :param parallel: Either or not it should run cwt in parallel. Default=True.
        :return:
        """
        self._validate_data()
        self._tf = self.__cwt_ba(parallel=parallel)


class ConvolveWaveletScipy(ConvolveWaveletBase):
    """
    Class to apply wavelet convolution to a mseed file. The bank of atoms is computed at the class initialisation.
        Examples
        --------
        cw = ConvolveWavelet(file_path)
        convolve = cw.ccwt_ba_fast()
    """

    def __init__(self, data, **kwargs):
        """
        Class to apply wavelet convolution to a mseed file.
        The bank of atoms is computed at the class initialisation.
        :param data: Either the mseed file path or an obspy Tracer.
        :keyword kwargs:
        :keyword wmin: Minimum number of cycles. Default = 6.
        :keyword wmax: Maximum number of cycles. Default = 6.
        :keyword tt: Central frequency of the Morlet Wavelet. Default = 2.
        :keyword fmin: Minimum frequency (in Hz). Default = 2.
        :keyword fmax: Maximum frequency (in Hz). Default = 12.
        :keyword nf: Number of logarithmically spaced frequencies between fmin and fmax. Default = 20.
        :keyword use_rfft: True if it should use rfft instead of fft. Default = True.
        :keyword decimate: True if it should try to decimate the trace. Default = False. The decimation
            factor is equal to q = 0.4*SR/fmax. For SR=200Hz and fmax=40Hz, q=2. This will downsize the
            sample rate to 100 Hz.
        :raise TypeError: If file is not a valid mseed.
        :example:
        >>> cw = ConvolveWavelet(data)
        >>> cw.setup_wavelet()
        >>> cf = cw.cf_lowpass()
        """

        super().__init__(data, **kwargs)

    def __convolve(self, freq: tuple):
        freq, index = freq
        cmw = self.filter_win(freq, index)
        return scipy.signal.oaconvolve(self._data, cmw, mode='same')

    def _convolve_atoms(self, parallel):

        d_type = np.float32 if self._use_rfft else np.complex64

        n_proc = self.get_nproc()
        if parallel and n_proc > 1:
            with ThreadPool(n_proc) as pool:
                tf = np.array(pool.map(self.__convolve, [(fi, i) for i, fi in enumerate(self._frex)]),
                              copy=False, dtype=d_type)
        else:
            tf = np.array([self.__convolve((fi, i)) for i, fi in enumerate(self._frex)], copy=False, dtype=d_type)

        return tf

    def _convolve_atoms_stream(self):
        d_type = np.float32 if self._use_rfft else np.complex64
        tfs = []
        for trace_index in self.st:
            tf = []
            for i, fi in enumerate(self._frex):
                tf.append(self.__convolve_stream((fi, i), trace_index))
            tf = np.array(tf, copy=False, dtype=d_type)
            tfs.append([tf, trace_index.stats])
        return tfs

    def __convolve_stream(self, freq: tuple, trace_index):

        freq, index = freq
        cmw = self.filter_win(freq, index)
        return scipy.signal.oaconvolve(trace_index, cmw, mode='same')

    def compute_tf(self, parallel=True, stream=False):
        """
        Compute the convolved waveform with the wavelet from fmin to fmax.
        :param parallel: Either or not it should run cwt in parallel. Default=True.
        :return:
        """
        self._validate_data()
        if stream:
            self._tf = self._convolve_atoms_stream()
        else:
            self._tf = self._convolve_atoms(parallel)

