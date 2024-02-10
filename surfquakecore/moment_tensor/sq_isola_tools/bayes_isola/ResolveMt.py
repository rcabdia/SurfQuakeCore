# """
#     Class for the solution of the MT inverse problem.
#
#     :param data: instance of processed data
#     :type data: :class:`~BayesISOLA.process_data`
#     :param cova: instance of a covariance matrix
#     :type cova: :class:`~BayesISOLA.covariance_matrix`
#     :type deviatoric: bool, optional
#     :param deviatoric: if ``False``: invert full moment tensor (6 components); if ``True`` invert deviatoric part of the moment tensor (5 components) (default ``False``)
#     :type decompose: bool, optional
#     :param decompose: performs decomposition of the found moment tensor in each grid point
#     :param run_inversion: run :func:`run_inversion`
#     :type run_inversion: bool, optional
#     :param find_best_grid_point: run :func:`find_best_grid_point`
#     :type find_best_grid_point: bool, optional
#     :param save_seismo: run :func:`save_seismo`
#     :type save_seismo: bool, optional
#     :param VR_of_components: run :func:`VR_of_components`
#     :type VR_of_components: bool, optional
#     :param print_solution: run :func:`print_solution`
#     :type print_solution: bool, optional
#     :param print_fault_planes: run :func:`print_fault_planes` (only if ``decompose`` is ``True``)
#     :type print_fault_planes: bool, optional
#
#     .. rubric:: _`Variables`
#
#     ``centroid`` : Reference to ``grid`` item.
#         The best grid point found by the inversion.
#     ``mt_decomp`` : list
#         Decomposition of the best centroid moment tensor solution calculated by :func:`decompose` or :func:`decompose_mopad`
#     ``max_VR`` : tuple (VR, n)
#         Variance reduction `VR` from `n` components of a subset of the closest stations
# 	"""
from multiprocessing import Pool

import numpy as np
from obspy import UTCDateTime

from surfquakecore.moment_tensor.sq_isola_tools.bayes_isola.inverse_problem import invert
from surfquakecore.moment_tensor.sq_isola_tools.bayes_isola.MT_comps import a2mt, decompose, decompose_mopad
from surfquakecore.moment_tensor.sq_isola_tools.bayes_isola.fileformats import read_elemse
from surfquakecore.moment_tensor.sq_isola_tools.bayes_isola.helpers import my_filter
from surfquakecore.moment_tensor.sq_isola_tools.bayes_isola.inversion_data_manager import InversionDataManager


class ResolveMt:
	"""
	Class for the solution of the MT inverse problem.

	"""

	def __init__(
			self, data: InversionDataManager, cova, working_directory, deviatoric=False, decompose=True,
			run_inversion=True, find_best_grid_point=True,
			save_seismo=False, VR_of_components=False, print_solution=True, print_fault_planes=True,
			from_axistra=True
	):
		"""
		Class for the solution of the MT inverse problem.

		Args:
			data: instance of processed data
			cova: class: instance of a covariance matrix
			working_directory:
			deviatoric: if ``False``: invert full moment tensor (6 components); if ``True`` invert deviatoric part of
				the moment tensor (5 components) (default ``False``)
			decompose: performs decomposition of the found moment tensor in each grid point
			run_inversion: run :func:`run_inversion`
			find_best_grid_point: un :func:`find_best_grid_point`
			save_seismo: run :func:`save_seismo`
			VR_of_components: run :func:`VR_of_components`
			print_solution: run :func:`print_solution`
			print_fault_planes: run :func:`print_fault_planes` (only if ``decompose`` is ``True``)
			from_axistra:
		"""
		self.log = data.log
		self.d = data
		self.inp: InversionDataManager = data.d
		self.cova = cova
		self.working_directory = working_directory
		self.g = data.grid
		self.grid = data.grid.grid
		self.threads = data.threads
		self.event = data.d.event
		self.from_axistra = from_axistra
		self.deviatoric = deviatoric
		self.decompose = decompose
		self.mt_decomp = []
		self.max_VR = ()

		self._max_sum_c = 0.
		self._max_c = 0.
		self._sum_c = 0.
		self._centroid = None
		self._vr_comp = {}

		self.log('Inversion of ' + {1:'deviatoric part of', 0:'full'}[self.deviatoric] + ' moment tensor (' + {1:'5', 0:'6'}[self.deviatoric] + ' components)')

		if run_inversion:
			self.run_inversion()
		if find_best_grid_point:
			self.find_best_grid_point()
		if save_seismo:
			self.save_seismo(self.inp.outdir+'/data_observed', self.inp.outdir+'/data_modeles')
		if VR_of_components:
			self.vr_of_components()
		if print_solution:
			self.print_solution()
		if print_fault_planes and decompose:
			self.print_fault_planes()

	def run_inversion(self):
		"""
		Runs function :func:`invert` in parallel.

		Module :class:`multiprocessing` does not allow running function of the same class in parallel,
		so the function :func:`invert` cannot be method of class :class:`ISOLA` and this wrapper is needed.
		"""
		grid = self.grid
		d_shifts = self.d.d_shifts
		cd_inv = self.cova.Cd_inv
		cd_inv_shifts = self.cova.Cd_inv_shifts

		todo = []
		for i in range(len(grid)):

			point_id = str(i).zfill(4)
			grid[i]['id'] = point_id
			if not grid[i]['err']:
				todo.append(i)

		# create norm_d[shift]
		norm_d = []
		for shift in range(len(d_shifts)):
			d_shift = d_shifts[shift]
			if cd_inv_shifts:  # ACF
				cd_inv = cd_inv_shifts[shift]
			if cd_inv:
				idx = 0
				d_cd_blocks = []
				for C in cd_inv:
					size = len(C)
					d_cd_blocks.append(np.dot(d_shift[idx:idx + size, :].T, C))
					idx += size
				d_cd = np.concatenate(d_cd_blocks, axis=1)
				norm_d.append(np.dot(d_cd, d_shift)[0, 0])
			else:
				norm_d.append(0)
				for i in range(self.d.npts_slice * self.d.components):
					norm_d[-1] += d_shift[i, 0] * d_shift[i, 0]

		if self.threads > 1:
			with Pool(processes=self.threads) as pool:
				results = [
					pool.apply_async(
						invert,
						args=(
							grid[i]['id'], d_shifts, norm_d, cd_inv, cd_inv_shifts, self.inp.nr,
							self.d.components, self.inp.stations, self.d.npts_elemse,
							self.d.npts_slice, self.d.elemse_start_origin,
							self.inp.event['t'], self.d.samprate, self.deviatoric, self.decompose,
							self.d.invert_displacement, self.working_directory, [i, len(todo), self.threads] ,
							self.from_axistra)) for i in todo]

				output = [p.get() for p in results]
		else:
			output = []
			for i in todo:
				res = invert(
					grid[i]['id'], d_shifts, norm_d, cd_inv, cd_inv_shifts, self.inp.nr, self.d.components,
					self.inp.stations, self.d.npts_elemse, self.d.npts_slice, self.d.elemse_start_origin,
					self.inp.event['t'], self.d.samprate, self.deviatoric, self.decompose, self.d.invert_displacement,
					self.working_directory, [i, len(todo), self.threads], from_axistra=self.from_axistra
				)
				output.append(res)

		min_misfit = output[0]['misfit']
		for i in todo:
			grid[i].update(output[todo.index(i)])
			grid[i]['shift_idx'] = grid[i]['shift']
			# grid[i]['shift'] = self.g.shift_min + grid[i]['shift']*self.g.SHIFT_step/self.d.max_samprate
			grid[i]['shift'] = self.d.shifts[grid[i]['shift']]
			min_misfit = min(min_misfit, grid[i]['misfit'])
		for i in todo:
			gp = grid[i]
			gp['sum_c'] = 0
			for idx in gp['shifts']:
				GP = gp['shifts'][idx]
				if gp['det_Ca'] == np.inf:
					GP['c'] = 0
				else:
					GP['c'] = np.sqrt(gp['det_Ca']) * np.exp(-0.5 * (GP['misfit'] - min_misfit))
				gp['sum_c'] += GP['c']
			gp['c'] = gp['shifts'][gp['shift_idx']]['c']

			self._sum_c += gp['sum_c']
			self._max_c = max(self._max_c, gp['c'])
			self._max_sum_c = max(self._max_sum_c, gp['sum_c'])

	def find_best_grid_point(self):
		"""
		Set ``self.centroid`` to a grid point with higher variance reduction -- the best solution of the inverse problem.
		"""
		self._centroid = max(self.grid, key=lambda g: g['VR'])  # best grid point

	def print_solution(self, mt_comp_precision=2):
		"""
		Write into log the best solution ``self.centroid``.

		:param mt_comp_precision: number of decimal digits of moment tensor components (default ``2``)
		:type mt_comp_precision: int, optional
		"""
		C = self._centroid
		t = self.event['t'] + C['shift']
		self.log(
			'\nCentroid location:\n  Centroid time: {t:s}\n  Lat {lat:8.3f}   Lon {lon:8.3f}   Depth {d:5.1f} km'.format(
				t=t.strftime('%Y-%m-%d %H:%M:%S'), lat=C['lat'], lon=C['lon'], d=C['z'] / 1e3))
		self.log(
			'  ({0:5.0f} m to the north and {1:5.0f} m to the east with respect to epicenter)'.format(C['x'], C['y']))

		self.inp.inversion_result.centroid.latitude = C['lat']
		self.inp.inversion_result.centroid.longitude = C['lon']
		self.inp.inversion_result.centroid.depth = C['z'] / 1e3
		self.inp.inversion_result.centroid.time = t.datetime
		self.inp.inversion_result.centroid.origin_shift = C['shift']
		self.inp.inversion_result.centroid.vr = C['VR'] * 100
		self.inp.inversion_result.centroid.cn = C['CN']
		self.inp.inversion_result.centroid.rupture_length = self.inp.rupture_length*1e-3

		if C['edge']:
			self.log('  Warning: the solution lies on the edge of the grid!')
		mt2 = a2mt(C['a'], system='USE')
		c = max(abs(min(mt2)), max(mt2))
		c = 10 ** np.floor(np.log10(c))
		MT2 = mt2 / c
		if C['shift'] >= 0:
			self.log('  time: {0:5.2f} s after origin time\n'.format(C['shift']))
		else:
			self.log('  time: {0:5.2f} s before origin time\n'.format(-C['shift']))
		if C['shift'] in (self.d.shifts[0], self.d.shifts[-1]):
			self.log('  Warning: the solution lies on the edge of the time-grid!')

		self.log('  VR: {0:4.0f} %\n  CN: {1:4.0f}'.format(C['VR'] * 100, C['CN']))

		# self.log('  VR: {0:8.4f} %\n  CN: {1:4.0f}'.format(C['VR']*100, C['CN'])) # DEBUG
		self.log(
			'  MT [ {1:{0}}  {2:{0}}  {3:{0}}  {4:{0}}  {5:{0}}  {6:{0}}]:'.format(mt_comp_precision + 3, 'Mrr', 'Mtt',
																				   'Mpp', 'Mrt', 'Mrp', 'Mtp'))
		self.log(
			'     [{1:{7}.{8}f}  {2:{7}.{8}f}  {3:{7}.{8}f}  {4:{7}.{8}f}  {5:{7}.{8}f}  {6:{7}.{8}f} ] * {0:5.0e}'.format(
				c, *MT2, mt_comp_precision + 3, mt_comp_precision))

		self.inp.inversion_result.centroid.mrr = mt2[0]
		self.inp.inversion_result.centroid.mtt = mt2[1]
		self.inp.inversion_result.centroid.mpp = mt2[2]
		self.inp.inversion_result.centroid.mrt = mt2[3]
		self.inp.inversion_result.centroid.mrp = mt2[4]
		self.inp.inversion_result.centroid.mtp = mt2[5]

	def print_fault_planes(self, precision='3.0', tool=''):
		"""
		Decompose the moment tensor of the best grid point by :func:`decompose` and writes the result to the log.

		:param precision: precision of writing floats, like ``5.1`` for 5 letters width and 1 decimal place (default ``3.0``)
		:type precision: string, optional
		:param tool: tool for the decomposition, `mopad` for :func:`decompose_mopad`, otherwise :func:`decompose` is used
		"""
		mt = a2mt(self._centroid['a'])
		if tool == 'mopad':
			self.mt_decomp = decompose_mopad(mt)
		else:
			self.mt_decomp = decompose(mt)

		# print("result", self.mt_decomp)
		msg = f"\nScalar Moment: M0 = {self.mt_decomp['mom']: 5.2e} Nm (Mw = {self.mt_decomp['Mw']:3.1f})\n"
		msg += f"  DC component: {self.mt_decomp['dc_perc']: .0f} %  "
		msg += f"CLVD component: {self.mt_decomp['clvd_perc']: .0f} % "
		msg += f"  ISOtropic component: {self.mt_decomp['iso_perc']: .0f} %\n"
		msg += f"  Fault plane 1: strike = {self.mt_decomp['s1']: .0f}, dip = {self.mt_decomp['d1']: .0f} , "
		msg += f"slip-rake = {self.mt_decomp['r1']: .0f}\n"
		msg += f"  Fault plane 2: strike = {self.mt_decomp['s2']: .0f}, dip = {self.mt_decomp['d2']: .0f} , "
		msg += f"slip-rake = {self.mt_decomp['r2']: .0f}"
		print(msg)
		self.log(msg)
		self.inp.inversion_result.scalar.mo = self.mt_decomp['mom']
		self.inp.inversion_result.scalar.mw = self.mt_decomp['Mw']
		self.inp.inversion_result.scalar.dc = self.mt_decomp['dc_perc']
		self.inp.inversion_result.scalar.clvd = self.mt_decomp['clvd_perc']
		self.inp.inversion_result.scalar.isotropic_component = self.mt_decomp['iso_perc']
		self.inp.inversion_result.scalar.plane_1_strike = self.mt_decomp['s1']
		self.inp.inversion_result.scalar.plane_1_dip = self.mt_decomp['d1']
		self.inp.inversion_result.scalar.plane_1_slip_rake = self.mt_decomp['r1']
		self.inp.inversion_result.scalar.plane_2_strike = self.mt_decomp['s2']
		self.inp.inversion_result.scalar.plane_2_dip = self.mt_decomp['d2']
		self.inp.inversion_result.scalar.plane_2_slip_rake = self.mt_decomp['r2']


	def vr_of_components(self, n=1):
		"""
		Calculates the variance reduction from each component and the variance reduction from a subset of stations.

		:param n: minimal number of components used
		:type n: integer, optional
		:return: maximal variance reduction from a subset of stations

		Add the variance reduction of each component to ``self.stations`` with keys ``VR_Z``, ``VR_N``, and ``VR_Z``.
		Calculate the variance reduction from a subset of the closest stations (with minimal ``n`` components used) leading to the highest variance reduction and save it to ``self.max_VR``.
		"""
		npts = self.d.npts_slice
		data = self.d.data_shifts[self._centroid['shift_idx']]
		elemse = read_elemse(
			self.inp.nr, self.d.npts_elemse, 'green/elemse' +
			self._centroid['id'] + '.dat', self.inp.stations, self.d.invert_displacement
		)
		for r in range(self.inp.nr):
			for e in range(6):
				my_filter(elemse[r][e], self.inp.stations[r]['fmin'], self.inp.stations[r]['fmax'])
				elemse[r][e].trim(UTCDateTime(0) + self.d.elemse_start_origin)
		# misfit = 0
		# norm_d = 0
		# comps_used = 0
		max_vr = -99
		self._vr_comp = {}
		for sta in range(self.inp.nr):
			synt = {}
			vr_sum = 0.
			for comp in range(3):
				synt[comp] = np.zeros(npts)
				for e in range(6):
					synt[comp] += elemse[sta][e][comp].data[0:npts] * self._centroid['a'][e, 0]
			comps_used = 0
			for comp in range(3):
				if self.cova.Cd_inv and not self.inp.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:
					self.inp.stations[sta][{0: 'VR_Z', 1: 'VR_N', 2: 'VR_E'}[comp]] = None
					continue
				synt = synt[comp]
				d = data[sta][comp][0:npts]
				if self.cova.LT3:
					d = np.zeros(npts)
					synt = np.zeros(npts)
					x1 = -npts
					for COMP in range(3):
						if not self.inp.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[COMP]]:
							continue
						x1 += npts
						x2 = x1 + npts
						y1 = comps_used * npts
						y2 = y1 + npts
						d += np.dot(self.cova.LT3[sta][y1:y2, x1:x2], data[sta][COMP].data[0:npts])
						synt += np.dot(self.cova.LT3[sta][y1:y2, x1:x2], synt[COMP])

				elif self.cova.Cd_inv:
					d = np.dot(self.cova.LT[sta][comp], d)
					synt = np.dot(self.cova.LT[sta][comp], synt)

				else:
					pass
				comps_used += 1
				misfit = np.sum(np.square(d - synt))
				norm_d = np.sum(np.square(d))
				vr = 1 - misfit / norm_d
				self.inp.stations[sta][{0: 'VR_Z', 1: 'VR_N', 2: 'VR_E'}[comp]] = vr
				if self.inp.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:
					misfit += misfit
					norm_d += norm_d
					vr_sum = 1 - misfit / norm_d
					comps_used += 1
			# print sta, comp, VR, VR_sum # DEBUG
			if comps_used >= n:
				if comps_used > 1:
					self._vr_comp[comps_used] = vr_sum
				if vr_sum >= max_vr:
					max_vr = vr_sum
					self.max_VR = (vr_sum, comps_used)
		return max_vr

	def save_seismo(self, file_d, file_synt):
		"""
		Saves observed and simulated seismograms into files.

		:param file_d: filename for observed seismogram
		:type file_d: string
		:param file_synt: filename for synthetic seismogram
		:type file_synt: string

		Uses :func:`numpy.save`.
		"""
		data = self.d.data_shifts[self._centroid['shift_idx']]
		npts = self.d.npts_slice
		elemse = read_elemse(
			self.inp.nr, self.d.npts_elemse, 'green/elemse' + self._centroid['id'] + '.dat',
			self.inp.stations, self.d.invert_displacement)  # nacist elemse
		for r in range(self.inp.nr):
			for e in range(6):
				my_filter(elemse[r][e], self.inp.stations[r]['fmin'], self.inp.stations[r]['fmax'])
				elemse[r][e].trim(UTCDateTime(0) + self.d.elemse_start_origin)
		synt = np.zeros((npts, self.inp.nr * 3))
		d = np.empty((npts, self.inp.nr * 3))
		for r in range(self.inp.nr):
			for comp in range(3):
				for e in range(6):
					synt[:, 3 * r + comp] += elemse[r][e][comp].data[0:npts] * self._centroid['a'][e, 0]
				d[:, 3 * r + comp] = data[r][comp][0:npts]
		np.save(file_d, d)
		np.save(file_synt, synt)
