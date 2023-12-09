"""
Various data manipulation / arithmetic / waveform filtering functions.

"""
import fractions
import math

import numpy as np
from scipy import signal
from scipy.signal import decimate


# TODO  Remove all unnecessary methods here. Lots of bad code
def rename_keys(_dict: dict, prefix: str = '', suffix: str = '') -> dict:
	"""
	Returns a dictionary with keys renamed by adding some prefix and/or suffix
	:param _dict: dictionary, whose keys will be remapped
	:type _dict: dictionary
	:param prefix: new keys starts with
	:type prefix: string, optional
	:param suffix: new keys ends with
	:type suffix: string, optional
	:returns : dictionary with keys renamed
	"""

	return {f"{prefix}{k}{suffix}": v for k, v in _dict.items()}


def next_power_of_2(n):
	"""
	Return next power of 2 greater than or equal to ``n``
	
	:type n: integer
	"""
	return 2**(n-1).bit_length()

def lcmm(b, *args):
	"""
	Returns generelized least common multiple.
	
	:param b,args: numbers to compute least common multiple of them 
	:type b,args: float, which is a multiple of 0.00033333
	:returns: the least multiple of ``a`` and ``b``
	"""
	b = 3/b
	if b - round(b) < 1e6:
		b = round(b)
	for a in args:
		a = 3/a
		if a - round(a) < 1e6:
			a = round(a)
		b = fractions.gcd(a, b)
		#b = math.gcd(a,b)
	return 3/b


def glcm(numbers):
	"""
    Calculate the Generalized Least Common Multiple (GLCM) of a list of integers
    without exponents. This is equivalent to finding the LCM of the numbers.

    Parameters:
    - numbers: A list of integers.

    Returns:
    The LCM of the numbers.
    """
	if not numbers:
		return 1  # LCM of an empty list is 1 by convention.

	glcm_result = numbers[0]
	for num in numbers[1:]:
		glcm_result = math.lcm(glcm_result, num)

	return glcm_result

def my_filter(data, fmin, fmax):
	"""
	Filter used for filtering both elementary and observed seismograms
	"""
	if fmax:
		data.filter('lowpass', freq=fmax)
	if fmin:
		data.filter('highpass', freq=fmin, corners=2)
		data.filter('highpass', freq=fmin, corners=2)

def prefilter_data(st, f):
	"""
	Drop frequencies above Green's function computation high limit using :func:`numpy.fft.fft`.
	
	:param st: stream to be filtered
	:type st: :class:`~obspy.core.stream`
	:param f: the highest frequency which will be kept in st, the frequencies above will be erased
	:type f: float
	"""
	for tr in st:
		npts = tr.stats.npts
		NPTS = next_power_of_2(npts)
		TR = np.fft.fft(tr.data,NPTS)
		df = tr.stats.sampling_rate / NPTS
		flim = int(np.ceil(f/df))
		TR[flim:NPTS-flim+1] = 0+0j
		tr_filt = np.fft.ifft(TR)
		tr.data = np.real(tr_filt[0:npts])

# TODO look at signal.decimate(y, q) from scipy. It seems it gives similar results and I would trust scipy method more.
def decimate(a, n=2):
	"""
	Decimates given sequence.

	:param data: data
	:type data: 1-D array
	:param n: decimation factor
	:type n: integer, optional

	Before decimating, filter out frequencies over Nyquist frequency using :func:`numpy.fft.fft`
	"""
	npts = len(a)
	#NPTS = npts # next_power_of_2(npts)
	amp = np.fft.fft(a, npts)
	idx = int(np.round(npts / n / 2. ))
	amp[idx: npts - idx + 1] = .0 + .0j

	a = np.fft.ifft(amp)
	if npts % (2*n) == 1 or n != 2:  # keep odd length for decimation factor 2
		return a[:npts:n].real
	else:
		return a[1:npts:n].real

# TODO small example to compare the 2 methods
# wave_duration = 3
# sample_rate = 100
# freq = 2
# q = 5
# samples = wave_duration*sample_rate
# x = np.linspace(0, wave_duration, samples, endpoint=False)
# y = np.cos(x*np.pi*freq*2)
# corr = signal.decimate(y, q)
# corr2 = decimate(y, q)
# print(corr)
# print(corr2)
# print(np.abs(corr2 - corr))
