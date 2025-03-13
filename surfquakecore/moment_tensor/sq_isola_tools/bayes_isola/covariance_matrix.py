import numpy as np
from numpy.linalg import LinAlgError

from surfquakecore.moment_tensor.sq_isola_tools.bayes_isola.helpers import decimate


class CovarianceMatrix:
	"""
    Design covariance matrix(es).

    :param data: instance with processed data
    :type data: :class:`~BayesISOLA.process_data`

    .. rubric:: _`Variables`

    ``Cd_inv`` : list of :class:`~numpy.ndarray`
        Inverse of the data covariance matrix :math:`C_D^{-1}` saved block-by-block. Created by :func:`covariance_matrix`.
    ``Cd`` : list of :class:`~numpy.ndarray`
        Data covariance matrix :math:`C_D^{-1}` saved block-by-block. Optionally created by :func:`covariance_matrix`.
    ``LT`` : list of list of :class:`~numpy.ndarray`
        Cholesky decomposition of the data covariance matrix :math:`C_D^{-1}` saved block-by-block with the blocks corresponding to one component of a station. Created by :func:`covariance_matrix`.
    ``LT3`` : list of :class:`~numpy.ndarray`
        Cholesky decomposition of the data covariance matrix :math:`C_D^{-1}` saved block-by-block with the blocks corresponding to all component of a station. Created by :func:`covariance_matrix`.
    ``Cf`` :  list of 3x3 :class:`~numpy.ndarray` of :class:`~numpy.ndarray`
        List of arrays of the data covariance functions.
    ``Cf_len`` : integer
        Length of covariance functions.    
	"""

	def __init__(self, data):
		self.d = data
		self.stations = data.d.stations
		self.log = data.log
		self.Cd_inv = []
		self.Cd = []
		self.LT = []
		self.LT3 = []
		self.Cf = []
		self.Cd_inv_shifts = []
		self.Cd_shifts = []
		self.LT_shifts = []

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		self.close()

	def close(self):
		del self.Cd_inv
		del self.Cd
		del self.LT
		del self.LT3
		del self.Cf
		del self.Cd_inv_shifts
		del self.Cd_shifts
		del self.LT_shifts

	def covariance_matrix_noise(self, crosscovariance=False, save_non_inverted=False, save_covariance_function=False):
		"""
		Creates covariance matrix :math:`C_D` from ``self.d.noise``.

		:type crosscovariance: bool, optional
		:param crosscovariance: Set ``True`` to calculate crosscovariance between components. If ``False``, it assumes that noise at components is not correlated, so non-diagonal blocks are identically zero.
		:type save_non_inverted: bool, optional
		:param save_non_inverted: If ``True``, save also non-inverted matrix, which can be plotted later.
		:type save_covariance_function: bool, optional
		:param save_covariance_function: If ``True``, save also the covariance function matrix, which can be plotted later.
		"""
		self.log('\nCreating covariance matrix')
		if not self.d.noise:
			self.log(
				'No noise slice to generate covariance matrix. Some of records probably too short or noise slices not generated [param noise_slice at func trim_filter_data()]. Exiting...',
				printcopy=True)
			raise ValueError('No noise slice to generate covariance matrix.')
		n = self.d.npts_slice
		self.Cf = []
		for r in range(len(self.stations)):
			sta = self.stations[r]
			idx = []
			if sta['useZ']: idx.append(0)
			if sta['useN']: idx.append(1)
			if sta['useE']: idx.append(2)
			size = len(idx) * n
			C = np.empty((size, size))
			if save_covariance_function:
				self.Cf.append(np.ndarray(shape=(3, 3), dtype=np.ndarray))
			if not crosscovariance:
				for i in idx:
					I = idx.index(i) * n
					for j in idx:
						if i == j and j < len(self.d.noise[r]) and i < len(self.d.noise[r]):
							corr = np.correlate(self.d.noise[r][i].data, self.d.noise[r][i].data, 'full') / len(
								self.d.noise[r][i].data)
							corr = decimate(corr, 2)  # noise has 2-times higher sampling than data
							middle = int((len(corr) - 1) / 2)
							if save_covariance_function:
								self.Cf[-1][i, i] = corr.copy()
								self.Cf_len = len(corr)
							for k in range(n):
								for l in range(k, n):
									C[l + I, k + I] = C[k + I, l + I] = corr[middle + k - l]
						if i != j:
							J = idx.index(j) * n
							C[I:I + n, J:J + n] = 0.
			else:
				for i in idx:
					I = idx.index(i) * n
					for j in idx:
						J = idx.index(j) * n
						if i > j:
							continue
						# index,value,corr = xcorr(self.d.noise[r][i], self.d.noise[r][j], n, True) # there was some problem with small numbers, solved by tr.data *= 1e20
						corr = np.correlate(self.d.noise[r][i].data, self.d.noise[r][j].data, 'full') / len(
							self.d.noise[r][i].data)
						corr = decimate(corr, 2)  # noise has 2-times higher sampling than data
						middle = int((len(corr) - 1) / 2)
						if save_covariance_function:
							self.Cf[-1][i, j] = corr.copy()
							self.Cf_len = len(corr)
						for k in range(n):
							if i == j:
								for l in range(k, n):
									C[l + I, k + I] = C[k + I, l + I] = corr[middle + k - l]
							else:
								for l in range(n):
									C[l + J, k + I] = C[k + I, l + J] = corr[middle + k - l]

							# podle me nesmysl, ale fungovalo lepe nez predchozi
							# C[k+I, l+J] = corr[middle+abs(k-l)]
							# C[l+J, k+I] = corr[middle-abs(k-l)]
			# C = np.diag(np.ones(size)*corr[middle]*10) # DEBUG
			for i in idx:  # add to diagonal 2% of its average
				I = idx.index(i) * n
				C[I:I + n, I:I + n] += np.diag(np.zeros(n) + np.average(C[I:I + n, I:I + n].diagonal()) * 0.02)
			if save_non_inverted:
				self.Cd.append(C)
			if crosscovariance and len(C):
				try:
					C_inv = np.linalg.inv(C)
					self.LT3.append(np.linalg.cholesky(C_inv).T)
					self.Cd_inv.append(C_inv)
				except LinAlgError:
					w, v = np.linalg.eig(C)
					print('Minimal eigenvalue C[{0:1d}]: {1:6.1e}, clipping'.format(r, min(w)))
					w = w.real.clip(
						min=0)  # set non-zero eigenvalues to zero and remove complex part (both are numerical artefacts)
					# mx = max(w)
					# w = w.real.clip(min=w*1e-18) # set non-zero eigenvalues to almost-zero and remove complex part (both are numerical artefacts)
					v = v.real  # remove complex part of eigenvectors
					C = v.dot(np.diag(w)).dot(v.T)
					# C = nearcorr(C)
					C_inv = np.linalg.inv(C)
					w, v = np.linalg.eig(C_inv)  # DEBUG
					if min(w) < 0:
						print('Minimal eigenvalue C_inv: {1:6.1e}, CLIPPING'.format(r, min(w)))
						w = w.real.clip(min=0)  # DEBUG
						v = v.real  # DEBUG
						C_inv = v.dot(np.diag(w)).dot(v.T)
					self.Cd_inv.append(C_inv)
					self.LT3.append(np.diag(np.sqrt(w)).dot(v.T))
					self.LT.append([1, 1, 1])
			elif crosscovariance:  # C is zero-size matrix
				self.Cd_inv.append(C)
				self.LT.append([1, 1, 1])
				self.LT3.append(1)
			else:
				C_inv = np.linalg.inv(C)
				self.Cd_inv.append(C_inv)
				self.LT.append([1, 1, 1])
				for i in idx:
					I = idx.index(i) * n
					try:
						self.LT[-1][i] = np.linalg.cholesky(C_inv[I:I + n, I:I + n]).T
					except LinAlgError:
						# w,v = np.linalg.eig(C[I:I+n, I:I+n])
						# mx = max(w)
						# print ('Minimal eigenvalue C[{0:1d}, {1:d}]: {2:6.1e}, clipping'.format(r,i,min(w)))
						# w = w.real.clip(min=0)
						# v = v.real
						# C[I:I+n, I:I+n] = v.dot(np.diag(w)).dot(v.T)
						# C_inv[I:I+n, I:I+n] = np.linalg.inv(C[I:I+n, I:I+n])
						w, v = np.linalg.eig(C_inv[I:I + n, I:I + n])
						print('Minimal eigenvalue C[{0:1d}, {1:d}]: {2:6.1e}, CLIPPING'.format(r, i, min(w)))
						w = w.real.clip(min=0)
						v = v.real
						self.LT[-1][i] = np.diag(np.sqrt(w)).dot(v.T)
		
