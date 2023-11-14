import h5py
import numpy as np
import spaudiopy
import matplotlib.pyplot as plt
import scipy.signal

PATH = 'Device_ATFs.h5'
FS = 32000

f = h5py.File(PATH,'r')
fs = int(list(f['SamplingFreq_Hz'])[0][0])
ir = scipy.signal.resample_poly(np.array(f['IR'])[:, :, [0, 1, 2, 4, 5]], FS, fs)
ele = (np.pi/2)-np.array(f['Theta']) # (ndarray) elevation in radians [1 x nDirection]
ele = ele[0, :]
theta = -ele + np.pi/2
azi = np.array(f['Phi']) # (ndarray) azimuth in radians [1 x nDirection]
azi = azi[0, :]
azi[azi > np.pi] = azi[azi > np.pi] - 2 * np.pi

INTERP_ORDER = 25
weights = np.diag(np.sin(theta))

Y_gr = spaudiopy.sph.sh_matrix(INTERP_ORDER, azi, theta, 'real')


e = np.linalg.eigvalsh(Y_gr.T @ weights @ Y_gr)
e2 = np.linalg.eigvalsh(Y_gr.T @ weights @ Y_gr + np.max(e) * 0.1 * np.eye(Y_gr.shape[1])) 
Y_gr_pinv = np.linalg.inv(Y_gr.T @ weights @ Y_gr + np.max(e) * 0.1 * np.eye(Y_gr.shape[1])) @ Y_gr.T @ weights
hoa_array = Y_gr_pinv @ ir
gr = spaudiopy.grids.load_n_design(35)
azigr, zengr, _ = spaudiopy.utils.cart2sph(gr[:, 0], gr[:, 1], gr[:, 2])

TRUNC_ORDER = 25
shmat = spaudiopy.sph.sh_matrix(TRUNC_ORDER, azigr, zengr, 'real')
shmat15 = spaudiopy.sph.sh_matrix(15, azigr, zengr, 'real')
MIC = 1


b, a = scipy.signal.butter(2, (10000, 12000), 'bandpass', fs=FS)
VMIN = -60
VMAX = -30

# plt.subplot(312, projection="mollweide")
# plt.scatter(azigr, np.pi/2 - zengr, s = 10, 
#             c = 10 * np.log10(np.mean(
#             scipy.signal.lfilter(b, a, (shmat[:, :(TRUNC_ORDER+1)**2] @ hoa_array[:, :(TRUNC_ORDER+1)**2, :])[:, :, MIC], axis=0)**2, axis=0)), vmin=VMIN, vmax=VMAX)
# plt.title("resampled")
# plt.colorbar()
# plt.grid(True)

# plt.subplot(313, projection="mollweide")
# plt.scatter(azi, np.pi/2 - theta, s = 10, 
#             c = 10 * np.log10(
#             np.mean(scipy.signal.lfilter(b, a, ir[:, :, MIC], axis=0)**2, axis=0)), vmin=VMIN, vmax=VMAX)
# plt.title("sampled")
# plt.colorbar()
# plt.grid(True)

np.save('Easycom_array_%dHz_o%d_22samps_delay.npy' % (FS, INTERP_ORDER), hoa_array)