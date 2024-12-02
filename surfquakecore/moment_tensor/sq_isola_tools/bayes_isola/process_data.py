import hashlib
import math
import os

import numpy as np

import multiprocessing as mp
from surfquakecore.moment_tensor.sq_isola_tools.bayes_isola.axitra import Axitra_wrapper
from surfquakecore.moment_tensor.sq_isola_tools.bayes_isola.helpers import my_filter, prefilter_data, next_power_of_2, \
	glcm


class process_data:
	"""
    Process data for MT inversion.
    
    :param data: instance with raw data (event and seismograms)
    :type data: :class:`~BayesISOLA.InversionDataManager`
    :param grid: instance with space and time grid
    :type grid: :class:`~BayesISOLA.grid`
    :type s_velocity: float, optional
    :param s_velocity: characteristic S-wave velocity used for calculating number of wave lengths between the source and stations (default 3000 m/s)
    :type threads: integer, optional
    :param threads: number of threads for parallelization (default 2)
    :type invert_displacement: bool, optional
    :param invert_displacement: convert observed and modeled waveforms to displacement prior comparison (if ``True``), otherwise compare it in velocity (default ``False``)
    :type use_precalculated_Green: bool or ``'auto'``, optional
    :param use_precalculated_Green: use Green's functions calculated in the previous run (default ``False``), value ``'auto'`` for check whether precalculated Green's function exists and were calculated on the same grid
    :param correct_data: if ``True``, run :func:`correct_data`
    :type correct_data: bool, optional
    :param set_parameters: if ``True``, run :func:`set_parameters`; in this case setting parameters `fmax` and `fmin` is strongly recommended
	:param fmax: maximal inverted frequency for all stations
	:type fmax: float, optional
	:param fmin: minimal inverted frequency for all stations
	:type fmax: float, optional
	:parameter min_depth: minimal grid point depth in meters
	:type min_depth: float, optional
    :param skip_short_records: if not ``False``, run :func:`skip_short_records` with the value of the parameter
    :type skip_short_records: bool or int, optional
    :param calculate_or_verify_Green: if ``True``, run :func:`calculate_or_verify_Green`
    :type calculate_or_verify_Green: bool, optional
    :param trim_filter_data: if ``True``, run :func:`trim_filter_data`
    :type trim_filter_data: bool, optional
    :param decimate_shift: if ``True``, run :func:`decimate_shift`
    :type decimate_shift: bool, optional

    .. rubric:: _`Variables`

    ``data`` : list of :class:`~obspy.core.stream`
        Prepared data for the inversion. It's filled by function :func:`trim_filter_data`. The list is ordered ascending by epicentral distance of the station.
    ``noise`` : list of :class:`~obspy.core.stream`
        Before-event slice of ``data_raw`` for later noise analysis. Created by :func:`trim_filter_data`.
    ``samprate`` : float
        Sampling rate used in the inversion.
    ``max_samprate`` : float
        Maximal sampling rate of the source data, which can be reached by integer decimation from all input samplings.
    ``t_min`` : float
        Starttime of the inverted time window, in seconds from the origin time.
    ``t_max`` :  float
        Endtime of the inverted time window, in seconds from the origin time.
    ``t_len`` : float
        Length of the inverted time window, in seconds (``t_max``-``t_min``).
    ``npts_elemse`` : integer
        Number of elementary seismogram data points (time series for one component).
    ``npts_slice`` : integer
        Number of data points for one component of one station used in inversion :math:`\mathrm{npts_slice} \le \mathrm{npts_elemse}`.
    ``tl`` : float
        Time window length used in the inversion.
    ``freq`` : integer
        Number of frequencies calculated when creating elementary seismograms.
    ``xl`` : float
        Parameter ``xl`` for `Axitra` code.
    ``npts_exp`` : integer
        :math:`\mathrm{npts_elemse} = 2^\mathrm{npts_exp}`
    ``components`` : integer
        Number of components of all stations used in the inversion. Created by :func:`count_components`.
    ``data_shifts`` : list of lists of :class:`~obspy.core.stream`
        Shifted and trimmed ``data`` ready.
    ``d_shifts`` : list of :class:`~numpy.ndarray`
        The previous one in form of data vectors ready for the inversion.
    ``shifts`` : list of floats
        Shift values in seconds. List is ordered in the same way as the two previous list.
    ``fmin`` : float
        Lower range of bandpass filter for data.
    ``fmax`` : float
        Higher range of bandpass filter for data.
	"""

	def __init__(self, data, working_directory, grid, s_velocity=3000, velocity_ot_the_fastest_wave=8000, velocity_ot_the_slowest_wave = 1000,
				 threads=2, invert_displacement=False, use_precalculated_Green=False, correct_data=True,
				 set_parameters=True, fmax=1., fmin=0., min_depth=1000., skip_short_records=False,
				 calculate_or_verify_Green=True, trim_filter_data=True, decimate_shift=True):
		self.d = data
		self.working_directory = working_directory
		self.grid = grid
		self.s_velocity = s_velocity
		self.velocity_ot_the_fastest_wave = velocity_ot_the_fastest_wave
		self.velocity_ot_the_slowest_wave = velocity_ot_the_slowest_wave
		self.threads = threads
		self.invert_displacement = invert_displacement
		self.use_precalculated_Green = use_precalculated_Green
		self.data = []
		self.noise = []
		self.fmax = 0.
		self.log = data.log
		self.logtext = data.logtext
		self.idx_use = {0:'useZ', 1:'useN', 2:'useE'}
		self.idx_weight = {0:'weightZ', 1:'weightN', 2:'weightE'}
		
		if correct_data:
			self.correct_data()
		if set_parameters:
			self.set_parameters(fmax, fmin, min_depth)
		if not skip_short_records is False:
			self.skip_short_records(noise=True)
		if calculate_or_verify_Green:
			print("Creating Green Functions")
			self.calculate_or_verify_Green()
		if trim_filter_data:
			self.trim_filter_data()
			self.d.noise = self.noise
		if decimate_shift:
			try:
				self.decimate_shift()
			except:
				print("Coudn't decimate shift")

	def __exit__(self, exc_type, exc_value, traceback):
		self.__del__()
		
	def __del__(self):
		del self.data
		del self.noise

	def evalute_noise(self):
		# compare spectrum of the signal and the noise
		pass

	def correct_data(self, water_level=20):
		"""
		Corrects ``self.d.data_raw`` for the effect of instrument. Poles and zeros must be contained in trace stats.

		:param water_level: Water level in dB for deconvolution of instrument response.
		:type water_level: float, optional
		"""
		self.log('Removing instrument response.\n\tWater level: ' + str(water_level))
		for st in self.d.data_raw:
			st.detrend(type='demean')
			# st.filter('highpass', freq=0.01) # DEBUG
			for tr in st:
				# tr2 = tr.copy() # DEBUG
				if getattr(tr.stats, 'response', 0):
					tr.remove_response(output="VEL", water_level=water_level)
				elif getattr(tr.stats, 'paz', 0):
					tr.simulate(paz_remove=tr.stats.paz, water_level=water_level)
				else:
					print(tr.stats)
					raise ('No response in tr.stats for the trace above.')
		# 2DO: add prefiltering etc., this is not the best way for the correction
		# 	see http://docs.obspy.org/packages/autogen/obspy.core.trace.Trace.remove_response.html
		self.d.data_are_corrected = True

	def trim_filter_data(self, noise_slice=True, noise_starttime=None, noise_length=None):
		"""
		Filter ``self.d.data_raw`` using function :func:`prefilter_data`.
		Decimate ``self.d.data_raw`` to common sampling rate ``self.max_samprate``.
		Optionally, copy a time window for the noise analysis.
		Copy a slice to ``self.data``.

		:type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
		:param starttime: Specify the start time of trimmed data
		:type length: float
		:param length: Length in seconds of trimmed data.
		:type noise_slice: bool, optional
		:param noise_slice: If set to ``True``, copy a time window of the length ``lenght`` for later noise analysis. Copied noise is in ``self.noise``.
		:type noise_starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
		:param noise_starttime: Set the starttime of the noise time window. If ``None``, the time window starts in time ``starttime``-``length`` (in other words, it lies just before trimmed data time window).
		:type noise_length: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
		:param noise_length: Length of the noise time window (in seconds).
		"""

		starttime = self.d.event['t'] + self.grid.shift_min + self.t_min
		length = self.t_max - self.t_min + self.grid.shift_max + 10
		endtime = starttime + length
		if noise_slice:
			if not noise_length:
				noise_length = length * 4
			if not noise_starttime:
				noise_starttime = starttime - noise_length
				noise_endtime = starttime
			else:
				noise_endtime = noise_starttime + noise_length
			DECIMATE = int(round(self.max_samprate / self.samprate))

		print("Noise Window", noise_starttime, noise_endtime)
		for st in self.d.data_raw:
			stats = st[0].stats
			field = '_'.join([stats.network, stats.station, "", stats.channel[0:2]])
			if field in self.d.stations_index.keys():
			#field = '_'.join([stats.network, stats.station, stats.location, stats.channel[0:2])
			# fmax = self.d.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])]['fmax']
				fmax = self.d.stations_index['_'.join([stats.network, stats.station, "", stats.channel[0:2]])]['fmax']
				self.data.append(st.copy())
		for st in self.data:
			stats = st[0].stats
			field = '_'.join([stats.network, stats.station, "", stats.channel[0:2]])
			if field in self.d.stations_index.keys():
			# fmin = self.d.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])]['fmin']
			# fmax = self.d.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])]['fmax']
				fmin = self.d.stations_index['_'.join([stats.network, stats.station, "", stats.channel[0:2]])]['fmin']
				fmax = self.d.stations_index['_'.join([stats.network, stats.station, "", stats.channel[0:2]])]['fmax']
			decimate = int(round(st[0].stats.sampling_rate / self.max_samprate))
			if noise_slice:
				self.noise.append(st.slice(noise_starttime, noise_endtime))
				# print self.noise[-1][0].stats.endtime-self.noise[-1][0].stats.starttime, '<', length*1.1 # DEBUG
				# if (len(self.noise[-1])!=3 or (self.noise[-1][0].stats.endtime-self.noise[-1][0].stats.starttime < length*1.1)) and self.d.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])]['use'+stats.channel[2]]:
				if ((
						self.noise[-1][0].stats.endtime - self.noise[-1][0].stats.starttime < length * 1.1)) and \
						self.d.stations_index['_'.join([stats.network, stats.station, "", stats.channel[0:2]])][
							'use' + stats.channel[2]]:
					self.log('Noise slice too short to generate covariance matrix (station ' + st[
						0].stats.station + '). Stopping generating noise slices.')
					noise_slice = False
					self.noise = []
				elif len(self.noise[-1]):
					my_filter(self.noise[-1], fmin / 2, fmax * 2)
					self.noise[-1].decimate(int(decimate * DECIMATE / 2),
											no_filter=True)  # noise has 2-times higher sampling than data
			prefilter_data(st, self.freq / self.tl)
			st.decimate(decimate, no_filter=True)
			st.trim(starttime, endtime)

	# TODO: kontrola, jestli neorezavame mimo puvodni zaznam
	# TODO: Test this new version
	def decimate_shift(self):
		"""
        Generate ``self.data_shifts`` where are multiple copies of ``self.data`` (needed for plotting).
        Decimate ``self.data_shifts`` to sampling rate for inversion ``self.samprate``.
        Generate ``self.d_shifts`` where are multiple vectors :math:`d`, each of them shifted according to ``self.SHIFT_min``, ``self.SHIFT_max``, and ``self.SHIFT_step``
        """
		self.d_shifts = []
		self.data_shifts = []
		self.shifts = []
		starttime = self.d.event['t']  # + self.t_min
		length = self.t_max - self.t_min
		endtime = starttime + length
		decimate = int(round(self.max_samprate / self.samprate))

		for SHIFT in range(self.grid.SHIFT_min, self.grid.SHIFT_max + 1, self.grid.SHIFT_step):
			# data = deepcopy(self.data)
			shift = SHIFT / self.max_samprate
			self.shifts.append(shift)
			data = []
			for st in self.data:
				st2 = st.slice(starttime + shift - self.elemse_start_origin,
							   endtime + shift + 1)  # we add 1 s to be sure, that no index will point outside the range
				st2.trim(starttime + shift - self.elemse_start_origin, endtime + shift + 1, pad=True,
						 fill_value=0.)  # short records are not inverted, but they should by padded because of plotting
				st2.decimate(decimate, no_filter=True)
				stats = st2[0].stats
				stats.location = ''
				stn = self.d.stations_index['_'.join([stats.network, stats.station, "", stats.channel[0:2]])]
				fmin = stn['fmin']
				fmax = stn['fmax']

				my_filter(st2, fmin, fmax)

				# we add 1 s to be sure, that no index will point outside the range
				st2.trim(starttime + shift, endtime + shift + 1)
				data.append(st2)
			self.data_shifts.append(data)

			d_shift = np.empty((self.components * self.npts_slice, 1))

			comp_check_sum = 0
			for r in range(self.d.nr):

				comp_check = 0
				for comp in range(3):
					if self.d.stations[r][
						{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:  # this component has flag 'use in inversion'
						weight = self.d.stations[r][{0: 'weightZ', 1: 'weightN', 2: 'weightE'}[comp]]

						for i in range(self.npts_slice):
							d_shift[comp_check_sum * self.npts_slice + i] = data[r][comp_check].data[i] * weight

						comp_check += 1
						comp_check_sum += 1

			self.d_shifts.append(d_shift)

	# def decimate_shift(self):
	# 	"""
	# 	Generate ``self.data_shifts`` where are multiple copies of ``self.data`` (needed for plotting).
	# 	Decimate ``self.data_shifts`` to sampling rate for inversion ``self.samprate``.
	# 	Filter ``self.data_shifts`` by :func:`my_filter`.
	# 	Generate ``self.d_shifts`` where are multiple vectors :math:`d`, each of them shifted according to ``self.grid.SHIFT_min``, ``self.grid.SHIFT_max``, and ``self.grid.SHIFT_step``
	# 	"""
	# 	self.d_shifts = []
	# 	self.data_shifts = []
	# 	self.shifts = []
	# 	starttime = self.d.event['t']  # + self.t_min
	# 	length = self.t_max - self.t_min
	# 	endtime = starttime + length
	# 	decimate = int(round(self.max_samprate / self.samprate))
	# 	for SHIFT in range(self.grid.SHIFT_min, self.grid.SHIFT_max + 1, self.grid.SHIFT_step):
	# 		# data = deepcopy(self.data)
	# 		shift = SHIFT / self.max_samprate
	# 		self.shifts.append(shift)
	# 		data = []
	# 		for st in self.data:
	# 			st2 = st.slice(starttime + shift - self.elemse_start_origin,
	# 						   endtime + shift + 1)  # we add 1 s to be sure, that no index will point outside the range
	# 			st2.trim(starttime + shift - self.elemse_start_origin, endtime + shift + 1, pad=True,
	# 					 fill_value=0.)  # short records are not inverted, but they should by padded because of plotting
	# 			if self.invert_displacement:
	# 				st2.detrend('linear')
	# 				st2.integrate()
	# 			st2.decimate(decimate, no_filter=True)
	# 			stats = st2[0].stats
	# 			# stn = self.d.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])]
	# 			stn = self.d.stations_index['_'.join([stats.network, stats.station, "", stats.channel[0:2]])]
	# 			fmin = stn['fmin']
	# 			fmax = stn['fmax']
	# 			if stn['accelerograph']:
	# 				st2.integrate()
	# 			my_filter(st2, fmin, fmax) # TODO is this necessary?
	# 			st2.trim(starttime + shift,
	# 					 endtime + shift + 1)  # we add 1 s to be sure, that no index will point outside the range
	# 			data.append(st2)
	# 		self.data_shifts.append(data)
	# 		c = 0
	# 		d_shift = np.empty((self.components * self.npts_slice, 1))
	# 		for r in range(self.d.nr):
	# 			for comp in range(3):
	# 				if self.d.stations[r][
	# 					{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:  # this component has flag 'use in inversion'
	# 					weight = self.d.stations[r][{0: 'weightZ', 1: 'weightN', 2: 'weightE'}[comp]]
	# 					for i in range(self.npts_slice):
	# 						try:
	# 							d_shift[c * self.npts_slice + i] = data[r][comp].data[i] * weight
	# 						except:
	# 							self.log(
	# 								'Index out of range while generating shifted data vectors. Waveform file probably too short.',
	# 								printcopy=True)
	# 							print('values for debugging: ', r, comp, c, self.npts_slice, i, c * self.npts_slice + i,
	# 								  len(d_shift), len(data[r][comp].data), SHIFT)
	# 							raise Exception(
	# 								'Index out of range while generating shifted data vectors. Waveform file probably too short.')
	# 					c += 1
	# 		self.d_shifts.append(d_shift)

	def set_frequencies(self, fmax, fmin=0., wavelengths=5):
		"""
		Sets frequency range for each station according its distance.

		:type fmax: float
		:param fmax: maximal inverted frequency for all stations
		:type fmin: float, optional
		:param fmin: minimal inverted frequency for all stations
		:type wavelengths: float, optional
		:param wavelengths: maximal number of wavelengths between the source and the station; if exceeded, automatically decreases ``fmax``

		The maximal frequency for each station is determined according to the following formula:

		:math:`\min ( f_{max} = \mathrm{wavelengths} \cdot \mathrm{self.s\_velocity} / r, \; fmax )`,

		where `r` is the distance the source and the station.
		"""
		for stn in self.d.stations:
			dist = np.sqrt(stn['dist'] ** 2 + self.d.event['depth'] ** 2)
			stn['fmax'] = min(wavelengths * self.s_velocity / dist, fmax)
			stn['fmin'] = fmin
			self.fmax = max(self.fmax, stn['fmax'])

	def set_working_sampling(self, multiple8=False):
		"""
		Determine maximal working sampling as at least 8-multiple of maximal inverted frequency (``self.fmax``). If needed, increases the value to eneables integer decimation factor.

		:param multiple8: if ``True``, force the decimation factor to be such multiple, that decimation can be done with factor 8 (more times, if needed) and finaly with factor <= 8. The reason for this is decimation pre-filter unstability for higher decimation factor (now not needed).
		:type multiple8: bool, optional
		"""
		# min_sampling = 4 * self.fmax
		min_sampling = 8 * self.fmax  # teoreticky 4*fmax aby fungovala L2 norma????
		inverses = []
		for num in self.d.data_deltas:
			if num == 0:
				raise ValueError("Division by zero is not allowed.")
			inverses.append(int(1 / num))
		SAMPRATE = float(glcm(inverses))
		decimate = SAMPRATE / min_sampling
		if multiple8:
			if decimate > 128:
				decimate = int(decimate / 64) * 64
			elif decimate > 16:
				decimate = int(decimate / 8) * 8
			else:
				decimate = int(decimate)
		else:
			decimate = int(decimate)
		self.max_samprate = SAMPRATE
		# print(min_sampling, SAMPRATE, decimate) # DEBUG
		# print(self.d.data_deltas) # DEBUG
		self.samprate = SAMPRATE / decimate
		self.logtext['samplings'] = samplings_str = ", ".join(
			["{0:5.1f} Hz".format(1. / delta) for delta in self.d.data_deltas])
		self.log(
			'\nSampling frequencies:\n  Data sampling: {0:s}\n  Common sampling: {3:5.1f}\n  Decimation factor: {1:3d} x\n  Sampling used: {2:5.1f} Hz'.format(
				samplings_str, decimate, self.samprate, SAMPRATE))

	def count_components(self, log=True):
		"""
		Counts number of components, which should be used in inversion (e.g. ``self.d.stations[n]['useZ'] = True`` for `Z` component). This is needed for allocation proper size of matrices used in inversion.

		:param log: if true, write into log table of stations and components with information about component usage and weight
		:type log: bool, optional
		"""
		c = 0
		stn = self.d.stations
		for r in range(self.d.nr):
			if stn[r]['useZ']: c += 1
			if stn[r]['useN']: c += 1
			if stn[r]['useE']: c += 1
		self.components = c
		if log:
			out = '\nComponents used in inversion and their weights\nstation     \t   \t Z \t N \t E \tdist\tazimuth\tfmin\tfmax\n            \t   \t   \t   \t   \t(km)    \t(deg)\t(Hz)\t(Hz)\n'
			for r in range(self.d.nr):
				out += '{net:>3s}:{sta:5s} {loc:2s}\t{ch:2s} \t'.format(sta=stn[r]['code'], net=stn[r]['network'],
																		loc=stn[r]['location'],
																		ch=stn[r]['channelcode'])
				for c in range(3):
					if stn[r][self.idx_use[c]]:
						out += '{0:3.1f}\t'.format(stn[r][self.idx_weight[c]])
					else:
						out += '---\t'
				if stn[r]['dist'] > 2000:
					out += '{0:4.0f}    '.format(stn[r]['dist'] / 1e3)
				elif stn[r]['dist'] > 200:
					out += '{0:6.1f}  '.format(stn[r]['dist'] / 1e3)
				else:
					out += '{0:8.3f}'.format(stn[r]['dist'] / 1e3)
				out += '\t{2:3.0f}\t{0:4.2f}\t{1:4.2f}'.format(stn[r]['fmin'], stn[r]['fmax'], stn[r]['az'])
				out += '\n'
			self.logtext['components'] = out
			self.log(out, newline=False)

	def min_time(self, distance, mag=0):
		"""
		Defines the beginning of inversion time window in seconds from location origin time. Save it into ``self.t_min`` (now save 0 -- FIXED OPTION)

		:param distance: station distance in meters
		:type distance: float
		:param mag: magnitude (unused)
		:param v: the first inverted wave-group characteristic velocity in m/s
		:type v: float

		Sets ``self.t_min`` as minimal time of interest (in seconds).
		"""
		# t = distance/v		# FIXED OPTION
		##if t<5:
		##t = 0
		# self.t_min = t
		v = self.velocity_ot_the_fastest_wave
		self.t_min = 0  # FIXED OPTION, because Green's functions with beginning in non-zero time are nou implemented yet

	def max_time(self, distance, mag=0):
		"""
		Defines the end of inversion time window in seconds from location origin time. Calculates it as :math:`\mathrm{distance} / v`.
		Save it into ``self.t_max``.

		:param distance: station distance in meters
		:type distance: float
		:param mag: magnitude (unused)
		:param v: the last inverted wave-group characteristic velocity in m/s
		:type v: float
		"""
		v = self.velocity_ot_the_slowest_wave
		t = distance / v  # FIXED OPTION
		self.t_max = t

	def set_time_window(self):
		"""
		Determines number of samples for inversion (``self.npts_slice``) and for Green's function calculation (``self.npts_elemse`` and ``self.npts_exp``) from ``self.min_time`` and ``self.max_time``.

		:math:`\mathrm{npts\_slice} \le \mathrm{npts\_elemse} = 2^{\mathrm{npts\_exp}} < 2\cdot\mathrm{npts\_slice}`
		"""
		self.min_time(np.sqrt(self.d.stations[0]['dist'] ** 2 + self.grid.depth_min ** 2))
		self.max_time(np.sqrt(self.d.stations[self.d.nr - 1]['dist'] ** 2 + self.grid.depth_max ** 2))
		# self.t_min -= 20 # FIXED OPTION
		self.t_min = round(self.t_min * self.samprate) / self.samprate
		if self.t_min > 0:
			self.t_min = 0.
		self.elemse_start_origin = -self.t_min
		self.t_len = self.t_max - self.t_min
		self.npts_slice = int(math.ceil(self.t_max * self.samprate))
		self.npts_elemse = next_power_of_2(int(math.ceil(self.t_len * self.samprate)))
		if self.npts_elemse < 64:  # FIXED OPTION
			self.npts_exp = 6
			self.npts_elemse = 64
		else:
			self.npts_exp = int(math.log(self.npts_elemse, 2))

	def set_parameters(self, fmax, fmin=0., wavelengths=5, min_depth=1000, log=True):
		"""
		Sets some technical parameters of the inversion.

		Technically, just runs following functions:
			- :func:`set_frequencies`
			- :func:`set_working_sampling`
			- :func:`set_grid`
			- :func:`set_time_grid`
			- :func:`set_time_window`
			- :func:`set_Greens_parameters`
			- :func:`count_components`

		The parameters are parameters of the same name of these functions.
		"""
		self.set_frequencies(fmax, fmin, wavelengths)
		self.set_working_sampling()
		self.grid.set_grid(min_depth=min_depth)  # must be after set_working_sampling
		self.grid.set_time_grid(self.fmax, self.max_samprate)
		self.set_time_window()
		self.set_Greens_parameters()
		self.count_components(log)

	def skip_short_records(self, noise=False):
		"""
		Checks whether all records are long enough for the inversion and skips unsuitable ones.

		:parameter noise: checks also whether the record is long enough for generating the noise slice for the covariance matrix (if the value is ``True``, choose minimal noise length automatically; if it's numerical, take the value as minimal noise length)
		:type noise: bool or float, optional
		"""
		self.log('\nChecking record length:')
		for st in self.d.data_raw:
			for comp in range(3):
				stats = st[comp].stats
				if stats.starttime > self.d.event['t'] + self.t_min + self.grid.shift_min or stats.endtime < \
						self.d.event['t'] + self.t_max + self.grid.shift_max:
					self.log(
						'  ' + stats.station + ' ' + stats.channel + ': record too short, ignoring component in inversion')
					self.d.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])][
						'use' + stats.channel[2]] = False
				if noise:
					if type(noise) in (float, int):
						noise_len = noise
					else:
						noise_len = (
												self.t_max - self.t_min + self.grid.shift_max + 10) * 1.1 - self.grid.shift_min - self.t_min
					# print stats.station, stats.channel, noise_len, '>', self.d.event['t']-stats.starttime # DEBUG
					if stats.starttime > self.d.event['t'] - noise_len:
						self.log(
							'  ' + stats.station + ' ' + stats.channel + ': record too short for noise covariance, ignoring component in inversion')
						self.d.stations_index[
							'_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])][
							'use' + stats.channel[2]] = False

	def calculate_or_verify_Green(self):
		"""
		If ``self.use_precalculated_Green`` is True, verifies whether the pre-calculated Green's functions were calculated on the same grid and with the same parameters (:func:`verify_Greens_headers` and :func:`verify_Greens_parameters`)
		Otherwise calculates Green's function (:func:`write_Greens_parameters` and :func:`calculate_Green`).
		"""

		if not self.use_precalculated_Green:  # calculate Green's functions in all grid points
			self.write_Greens_parameters()
			self.calculate_Green()
		else:  # verify whether the pre-calculated Green's functions are calculated on the same grid and with the same parameters
			differs = False
			if not self.verify_Greens_parameters():
				differs = True
			if not self.verify_Greens_headers():
				differs = True
			if differs:
				if self.use_precalculated_Green == 'auto':
					self.log('Shape or the grid or some parameters changed, calculating Gren\'s functions again...')
					self.write_Greens_parameters()
					self.calculate_Green()
				else:
					raise ValueError(
						'Metadata of pre-calculated Green\'s functions differs from actual calculation. More details are shown above and in the log file.')

	def write_Greens_parameters(self):
		"""
		Writes file grdat.hed - parameters for gr_xyz (Axitra)
		"""
		for model in self.d.models:
			if model:
				# f = 'green/grdat' + '-' + model + '.hed'
				f = os.path.join(self.working_directory, 'grdat') + '-' + model + '.hed'
			else:
				# f = 'green/grdat.hed'
				f = os.path.join(self.working_directory, 'grdat.hed')
			grdat = open(f, 'w')
			grdat.write(
				"&input\nnc=99\nnfreq={freq:d}\ntl={tl:1.2f}\naw=0.5\nnr={nr:d}\nns=1\nxl={xl:1.1f}\nikmax=100000\nuconv=0.1E-06\nfref=1.\n/end\n".format(
					freq=self.freq, tl=self.tl, nr=self.d.models[model],
					xl=self.xl))  # 'nc' is probably ignored in the current version of gr_xyz???
			grdat.close()

	def calculate_Green(self):
		"""
		Runs :func:`Axitra_wrapper` (Green's function calculation) in parallel.
		"""
		grid = self.grid.grid
		logfile = self.d.outdir + '/log_green.txt'
		open(logfile, "w").close()  # erase file contents
		# run `gr_xyz` aand `elemse`
		for model in self.d.models:
			if self.threads > 1:  # parallel
				with mp.Pool(processes=self.threads) as pool:
					results = [pool.apply_async(Axitra_wrapper, args=(
					i, model, grid[i]['x'], grid[i]['y'], grid[i]['z'], self.npts_exp, self.elemse_start_origin,
					self.working_directory, logfile)) for i in range(len(grid))]
					output = [p.get() for p in results]

				for i in range(len(grid)):
					if not output[i]:
						grid[i]['err'] = 1
						grid[i]['VR'] = -10
			else:
				for i in range(len(grid)):
					gp = grid[i]
					Axitra_wrapper(i, model, gp['x'], gp['y'], gp['z'], self.npts_exp, self.elemse_start_origin,
								   self.working_directory, logfile)

	def verify_Greens_parameters(self):
		"""
		Check whetrer parameters in file grdat.hed (probably used in Green's function calculation) are the same as used now.
		If it agrees, return True, otherwise returns False, print error description, and writes it into log.
		"""
		try:
			# grdat = open('green/grdat.hed', 'r')
			grdat = open(os.path.join(self.working_directory, 'grdat.hed'), 'r')
		except:
			readable = False
		else:
			readable = True
		if not readable or grdat.read() != "&input\nnc=99\nnfreq={freq:d}\ntl={tl:1.2f}\naw=0.5\nnr={nr:d}\nns=1\nxl={xl:1.1f}\nikmax=100000\nuconv=0.1E-06\nfref=1.\n/end\n".format(
				freq=self.freq, tl=self.tl, nr=self.d.nr, xl=self.xl):
			desc = 'Pre-calculated Green\'s functions calculated with different parameters (e.g. sampling) than used now, calculate Green\'s functions again.'
			self.log(desc)
			print(desc)
			print(
				"Expected content of green/grdat.hed:\n&input\nnc=99\nnfreq={freq:d}\ntl={tl:1.2f}\naw=0.5\nnr={nr:d}\nns=1\nxl={xl:1.1f}\nikmax=100000\nuconv=0.1E-06\nfref=1.\n/end\n".format(
					freq=self.freq, tl=self.tl, nr=self.d.nr, xl=self.xl))
			return False
		grdat.close()
		return True


	def verify_Greens_headers(self):
		"""
		Checked whether elementary-seismogram-metadata files (created when the Green's functions were calculated) agree with curent grid points positions.
		Used to verify whether pre-calculated Green's functions were calculated on the same grid as used now.
		"""
		path_crustal = os.path.join(self.working_directory, 'crustal.dat')
		path_stations = os.path.join(self.working_directory, 'station.dat')
		sourcetype = os.path.join(self.working_directory, 'soutype.dat')
		elemse_path = os.path.join(self.working_directory, 'elemse')
		md5_crustal = hashlib.md5(open(path_crustal, 'rb').read()).hexdigest()
		md5_station = hashlib.md5(open(path_stations, 'rb').read()).hexdigest()
		txt_soutype = open(sourcetype).read().strip().replace('\n', '_')

		problem = False
		desc = ''
		for g in range(len(self.grid.grid)):
			gp = self.grid.grid[g]
			point_id = str(g).zfill(4)
			try:
				meta = open(elemse_path + point_id + '.txt', 'r')
				lines = meta.readlines()
				meta.close()
			except:
				problem = True
				desc = 'Meta-data file for grid point {0:d} was not found. '.format(g)
			else:
				if len(lines) == 0:
					self.grid.grid[g]['err'] = 1
					self.grid.grid[g]['VR'] = -10
				elif lines[0] != '{0:1.3f} {1:1.3f} {2:1.3f} {3:s} {4:s} {5:s}'.format(gp['x'] / 1e3, gp['y'] / 1e3,
																					   gp['z'] / 1e3, md5_crustal,
																					   md5_station, txt_soutype):
					problem = True
			if problem:
				if not desc:
					l = lines[0].split()
					desc = 'Pre-calculated grid point {0:d} was calculated with different parameters. '.format(g)
					if l[0:3] != '{0:1.3f} {1:1.3f} {2:1.3f}'.format(gp['x'] / 1e3, gp['y'] / 1e3, gp['z'] / 1e3).split():
						desc += 'Its coordinates differs, probably the shape of the grid was changed. '
					if l[3] != md5_crustal:
						desc += 'File green/crustal.dat has different hash, probably crustal model was changed. '
					if l[4] != md5_station:
						desc += 'File green/station.dat has different hash, probably station set was different. '
					if l[5] != txt_soutype:
						desc += 'Source time function (file soutype.txt) was different. '
				self.log(desc)
				print(desc)
				return False
		return True

	def set_Greens_parameters(self):
		"""
		Sets parameters for Green's function calculation:
			- time window length ``self.tl``
			- number of frequencies ``self.freq``
			- spatial periodicity ``self.xl``

		Writes used parameters to the log file.
		"""
		self.tl = self.npts_elemse / self.samprate
		# freq = int(math.ceil(fmax*tl))
		# self.freq = min(int(math.ceil(self.fmax*self.tl))*2, self.npts_elemse/2) # pocitame 2x vic frekvenci, nez pak proleze filtrem, je to pak lepe srovnatelne se signalem, ktery je kauzalne filtrovany
		self.freq = int(self.npts_elemse / 2) + 1
		self.xl = max(np.ceil(self.d.stations[self.d.nr - 1]['dist'] / 1000),
					  100) * 1e3 * 20  # `xl` 20x vetsi nez nejvetsi epicentralni vzdalenost, zaokrouhlena nahoru na kilometry, minimalne 2000 km
		self.log(
			"\nGreen's function calculation:\n  npts: {0:4d}\n  tl: {1:4.2f}\n  freq: {2:4d}\n  npts for inversion: {3:4d}\n  source time function: {4:s}".format(
				self.npts_elemse, self.tl, self.freq, self.npts_slice, self.d.stf_description))
