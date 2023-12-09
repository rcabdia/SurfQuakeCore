import numpy as np
from scipy import signal


def tukeywin(window_length, alpha=0.5):
    '''
	The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2
    that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and
    at \alpha = 0 it becomes a Hann window.
    '''
    # Special cases
    if alpha <= 0:
        return np.ones(window_length) #rectangular window
    elif alpha >= 1:
        return np.hanning(window_length)
 
    # Normal case
    x = np.linspace(0, 1, window_length)
    w = np.ones(x.shape)
 
    # first condition 0 <= x < alpha/2
    first_condition = x<alpha/2
    w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2) ))
 
    # second condition already taken care of
 
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x>=(1 - alpha/2)
    w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2))) 
 
    return w

def running_mean(x, N):
	cumsum = np.cumsum(np.insert(x, 0, 0))
	rm = (cumsum[N:] - cumsum[:-N])/N
	Nzeros = int(np.floor(N/2))
	rm = np.concatenate((np.zeros((Nzeros)), rm), axis=0)
	rm = np.concatenate((rm, np.zeros((Nzeros))), axis=0)
	return rm


# TODO Do we need this??
def covariance_matrix_SACF(self, T = 15.0, taper = 0.0, save_non_inverted=False, save_covariance_function=False):
	"""
	Creates covariance matrix :math:`C_D`.
	
	:type save_non_inverted: bool, optional
	:param save_non_inverted: If ``True``, save also non-inverted matrix, which can be plotted later.
	
	Author: Miroslav Hallo, http://geo.mff.cuni.cz/~hallo/
	"""
	self.log('\nCreating SACF covariance matrix')
	self.log('signal duration {0:5.1f} sec'.format(T))
	self.log('station         \t L1 (sec)')
	n = self.d.npts_slice
	
	# Get components variance
	d_variance = []
	for r in range(len(self.stations)):
		sta = self.stations[r]
		idx = []
		if sta['useZ']: idx.append(0)
		if sta['useN']: idx.append(1)
		if sta['useE']: idx.append(2)
		d_st_tmp = []
		for i in idx:
			ntp = int(np.floor(len(self.d.data_shifts)/2))
			perOfMax = 0.1 # percentage of data variance from maximum amp (* 100%)
			d_st_tmp.append((perOfMax * max(abs(self.d.data_shifts[ntp][r][i][0:n])))**2) # data variance
		if idx:
			d_st_tmp_max = max(d_st_tmp)
			d_variance.append(d_st_tmp_max)
		else:
			d_variance.append(0)
	d_var_max = max(d_variance)
	
	# Build the matrix
	self.Cf = []
	for r in range(len(self.stations)):
		sta = self.stations[r]
		idx = []
		if sta['useZ']: idx.append(0)
		if sta['useN']: idx.append(1)
		if sta['useE']: idx.append(2)
		size = len(idx)*n
		C = np.empty((size, size))
		C = np.zeros((size, size))
		L1 = 2*(sta['dist']/50000)
		self.log('{0:6s}         \t {1:5.2f}'.format(sta['code'],L1))
		if save_covariance_function:
			self.Cf.append(np.ndarray(shape=(3,3), dtype=np.ndarray))
			
		for i in idx:
			I = idx.index(i)*n
			for j in idx:
				if i == j:
					
					# Prepare parameters
					nsampl = self.d.npts_slice
					dt = 1/self.d.samprate
					L1s = round(L1/dt)
					
					if (L1s < 2):
						L1s = 2
						#print('L1 changed to the minimum allowed value: 2 samples')
					
					ntp = int(np.floor(len(self.d.data_shifts)/2))
					f = g = np.array(self.d.data_shifts[ntp][r][i][0:nsampl]) # [time_shift][station][comp][sampl]
					
					#plt.figure(0)
					#plt.plot(f)
					#plt.savefig('test0.png', bbox_inches='tight')
					#plt.clf()
					#plt.close()
					
					# Cross-correlation of signals
					RfgN = np.empty([nsampl,1])
					RfgP = np.empty([nsampl,1])
					for tshift in range(nsampl):
						RfgN[tshift] = sum(   g[0:nsampl-tshift] * f[0+tshift:nsampl]  )*dt # negative
						RfgP[tshift] = sum(   f[0:nsampl-tshift] * g[0+tshift:nsampl]  )*dt # positive
					Rfg = np.concatenate((RfgN[nsampl:0:-1],RfgP[0:nsampl]), axis=0)
					
					#plt.figure(1)
					#plt.plot(Rfg)
					#plt.savefig('test1.png', bbox_inches='tight')
					#plt.clf()
					#plt.close()
					
					# Convolution of triangle function (width 2*L1) and cross-correlation
					b = np.ones(L1s)/L1s
					SACF = Rfg - signal.filtfilt(b, [1], Rfg, padtype = None, axis=0)						
					
					#plt.figure(2)
					#plt.plot(SACF)
					#plt.savefig('test2.png', bbox_inches='tight')
					#plt.clf()
					#plt.close()
	
					# Norm by the effective signal length
					SACF = SACF/T;
					
					# Save SACF function 
					if save_covariance_function:
						self.Cf[-1][i,i] = SACF.copy()
						self.Cf_len = len(SACF)
					
					# Taper of C matrix
					tw = tukeywin(nsampl, alpha=taper)
					tap = np.outer(tw,tw)
					for k in range(nsampl):
						tap[k,k] = tw[k];
					
					# Fill the matrix
					middle = int(np.floor(len(SACF)/2))
					for k in range(nsampl):
						for l in range(k, nsampl):
							C[l+I, k+I] = C[k+I, l+I] = SACF[middle+k-l] * tap[k,l]
							
					#plt.figure(3)
					#mx = max(C.max(), abs(C.min()))
					#cax = plt.matshow(C, vmin=-mx, vmax=mx)	
					#plt.colorbar(cax)
					#plt.savefig('test3.png', bbox_inches='tight')
					#plt.clf()
					#plt.close()
					
				if i != j:
					J = idx.index(j)*n
					C[I:I+n, J:J+n] = 0.
		
		for i in idx:  # add d_var_max to diagonal
			I = idx.index(i)*n
			#C[I:I+n, I:I+n] += np.diag(np.zeros(n)+np.average(C[I:I+n, I:I+n].diagonal())*0.05)
			#C[I:I+n, I:I+n] += np.diag(np.ones(n)*d_var_max) # one norm for all stations (max of all stations and components)
			C[I:I+n, I:I+n] += np.diag(np.ones(n)*d_variance[r]) # norm stations by station (max of all components)
			
		if save_non_inverted:
			self.Cd.append(C)
			
		# Inversion of matrix C
		C_inv = np.linalg.inv(C)
		self.Cd_inv.append(C_inv)
		self.LT.append([1, 1, 1])
		for i in idx:
			I = idx.index(i)*n
			try:
				self.LT[-1][i] = np.linalg.cholesky(C_inv[I:I+n, I:I+n]).T
			except:
				#w,v = np.linalg.eig(C[I:I+n, I:I+n])
				#mx = max(w)
				#print ('Minimal eigenvalue C[{0:1d}, {1:d}]: {2:6.1e}, clipping'.format(r,i,min(w)))
				#w = w.real.clip(min=0)
				#v = v.real
				#C[I:I+n, I:I+n] = v.dot(np.diag(w)).dot(v.T)
				#C_inv[I:I+n, I:I+n] = np.linalg.inv(C[I:I+n, I:I+n])
				w,v = np.linalg.eig(C_inv[I:I+n, I:I+n])
				print('Minimal eigenvalue C[{0:1d}, {1:d}]: {2:6.1e}, CLIPPING'.format(r,i,min(w)))
				w = w.real.clip(min=0)
				v = v.real
				self.LT[-1][i] = np.diag(np.sqrt(w)).dot(v.T)

def covariance_matrix_ACF(self, save_non_inverted=False):
	"""
	Creates covariance matrix :math:`C_D`.
	
	:type save_non_inverted: bool, optional
	:param save_non_inverted: If ``True``, save also non-inverted matrix, which can be plotted later.
	
	Author: Miroslav Hallo, http://geo.mff.cuni.cz/~hallo/
	"""
	self.log('\nCreating ACF covariance matrix')
	self.log('station         \t L1 (sec)')
	n = self.d.npts_slice
	
	for shift in range(len(self.d.d_shifts)):
		d_shift = self.d.d_shifts[shift]
		
		# Get components variance
		d_variance = []
		for r in range(len(self.stations)):
			sta = self.stations[r]
			idx = []
			if sta['useZ']: idx.append(0)
			if sta['useN']: idx.append(1)
			if sta['useE']: idx.append(2)
			for i in idx:
				d_variance.append((max(abs(self.d.data_shifts[shift][r][i][0:n]))/50)**2) # data variance set to 5% of the maximum
		d_var_max = max(d_variance)
		C_shift = []
		C_inv_shift = []
		LT_shift = []
		
		# Build the matrix
		for r in range(len(self.stations)):
			sta = self.stations[r]
			idx = []
			if sta['useZ']: idx.append(0)
			if sta['useN']: idx.append(1)
			if sta['useE']: idx.append(2)
			size = len(idx)*n
			C = np.empty((size, size))
			C = np.zeros((size, size))
			L1 = 2*(sta['dist']/50000)
			if shift == 0:
				self.log('{0:6s}         \t {1:5.2f}'.format(sta['code'],L1))
				
			for i in idx:
				I = idx.index(i)*n
				for j in idx:
					if i == j:
						
						# Prepare parameters
						nsampl = self.d.npts_slice
						dt = 1/self.d.samprate
						L1s = round(L1/dt)
						L12s = 1
						
						if (L1s < 3):
							L1s = 3
							#print('L1 changed to the minimum allowed value: 3 samples')
							
						if (L1s % 2 == 0):
							L1s = L1s - 1
						
						f = g = np.array(self.d.data_shifts[shift][r][i][0:nsampl]) # [time_shift][station][comp][sampl]
						
						# New number of samples
						nsamplN = len(f) + 2*L1s
						
						# Add zeros
						f = np.concatenate((np.zeros((L1s)), f), axis=0)
						f = np.concatenate((f, np.zeros((L1s))), axis=0)
						g = np.concatenate((np.zeros((L1s)), g), axis=0)
						g = np.concatenate((g, np.zeros((L1s))), axis=0)
						
						# Smooth
						fSmooth = running_mean(f, L1s)
						gSmooth = running_mean(g, L12s)
						
						#print(len(f))
						#print(len(fSmooth))
						
						#plt.figure(1)
						#plt.plot(f)
						#plt.savefig('test1.png', bbox_inches='tight')
						#plt.clf()
						#plt.close()
						
						#plt.figure(2)
						#plt.plot(fSmooth)
						#plt.savefig('test2.png', bbox_inches='tight')
						#plt.clf()
						#plt.close()
						
						# Compute ACF for time-lags
						ACF = np.zeros((nsamplN,nsamplN*2))
						for tshift in range(nsamplN*2):      # loop for time-lags (nsamplN+1 is zero time-lag)
							gShift = np.roll(gSmooth,tshift-nsamplN)
							ACF[:,tshift] = running_mean( f*gShift, L1s) - (fSmooth*running_mean(gShift, L1s))
						
						# Fill the covariance matrix by ACF
						for k in range(nsampl):
							for l in range(k, nsampl):
								C[l+I, k+I] = C[k+I, l+I] = ACF[ k+L1s, nsamplN-(l-k) ]
								
						#plt.figure(3)
						#mx = max(C.max(), abs(C.min()))
						#cax = plt.matshow(C, vmin=-mx, vmax=mx)	
						#plt.colorbar(cax)
						#plt.savefig('test3.png', bbox_inches='tight')
						#plt.clf()
						#plt.close()
						
					if i != j:
						J = idx.index(j)*n
						C[I:I+n, J:J+n] = 0.
			
			for i in idx:  # add d_var_max to diagonal
				I = idx.index(i)*n
				#C[I:I+n, I:I+n] += np.diag(np.zeros(n)+np.average(C[I:I+n, I:I+n].diagonal())*0.01)
				C[I:I+n, I:I+n] += np.diag(np.ones(n)*d_var_max)
			
			if save_non_inverted:
				C_shift.append(C)
				#self.Cd.append(C)
				
			# Inversion of matrix C
			C_inv = np.linalg.inv(C)
			#self.Cd_inv.append(C_inv)
			C_inv_shift.append(C_inv)
			#self.LT.append([1, 1, 1])
			LT_shift.append([1, 1, 1])
			for i in idx:
				I = idx.index(i)*n
				try:
					#self.LT[-1][i] = np.linalg.cholesky(C_inv[I:I+n, I:I+n]).T
					LT_shift[-1][i] = np.linalg.cholesky(C_inv[I:I+n, I:I+n]).T
				except:
					#w,v = np.linalg.eig(C[I:I+n, I:I+n])
					#mx = max(w)
					#print ('Minimal eigenvalue C[{0:1d}, {1:d}]: {2:6.1e}, clipping'.format(r,i,min(w)))
					#w = w.real.clip(min=0)
					#v = v.real
					#C[I:I+n, I:I+n] = v.dot(np.diag(w)).dot(v.T)
					#C_inv[I:I+n, I:I+n] = np.linalg.inv(C[I:I+n, I:I+n])
					w,v = np.linalg.eig(C_inv[I:I+n, I:I+n])
					print('Minimal eigenvalue C[{0:1d}, {1:d}]: {2:6.1e}, CLIPPING'.format(r,i,min(w)))
					w = w.real.clip(min=0)
					v = v.real
					#self.LT[-1][i] = np.diag(np.sqrt(w)).dot(v.T)
					LT_shift[-1][i] = np.diag(np.sqrt(w)).dot(v.T)
		if save_non_inverted:
			self.Cd_shifts.append(C_shift)
		self.Cd_inv_shifts.append(C_inv_shift)
		self.LT_shifts.append(LT_shift)

