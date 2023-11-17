import h5py
import numpy as np
import spaudiopy
import matplotlib.pyplot as plt
import scipy.signal

ARRAY_IR_FILE_PATH = "origin_array_tf_data/Device_ATFs.h5"
FS = 32000

f = h5py.File(ARRAY_IR_FILE_PATH, "r")

## load impulse responses
fs = int(list(f["SamplingFreq_Hz"])[0][0])
# resample to FS
ir = scipy.signal.resample_poly(np.array(f["IR"])[:, :, [0, 1, 2, 4, 5]], FS, fs)
zenith = np.array(f["Theta"])  # zenith angle in radians [1 x nDirection]
zenith = zenith[0, :]
azi = np.array(f["Phi"])  # azimuth in radians [1 x nDirection]
azi = azi[0, :]
azi[azi > np.pi] = azi[azi > np.pi] - 2 * np.pi

INTERP_ORDER = 25
weights = np.diag(np.sin(zenith))

Y_gr = spaudiopy.sph.sh_matrix(INTERP_ORDER, azi, zenith, "real")

# encoding matrix with area weights (sin(zenith)) and regularization by diagonal loading
REGUL_DIAG_LOAD = 0.1
e = np.linalg.eigvalsh(Y_gr.T @ weights @ Y_gr)
Y_gr_pinv = (
    np.linalg.inv(
        Y_gr.T @ weights @ Y_gr + np.max(e) * REGUL_DIAG_LOAD * np.eye(Y_gr.shape[1])
    )
    @ Y_gr.T
    @ weights
)
hoa_array = Y_gr_pinv @ ir

# decode and compare (change to True and adjust frequency range to view)
if False:
    FMIN_BANDPASS = 10000
    FMAX_BANDPASS = 12000
    TRUNC_ORDER = 25
    gr = spaudiopy.grids.load_n_design(56)
    azigr, zengr, _ = spaudiopy.utils.cart2sph(gr[:, 0], gr[:, 1], gr[:, 2])
    shmat = spaudiopy.sph.sh_matrix(TRUNC_ORDER, azigr, zengr, "real")
    MIC = 1

    b, a = scipy.signal.butter(2, (FMIN_BANDPASS, FMAX_BANDPASS), "bandpass", fs=FS)
    VMIN = -60
    VMAX = -30

    plt.subplot(211, projection="mollweide")
    plt.scatter(
        azigr,
        np.pi / 2 - zengr,
        s=10,
        c=10
        * np.log10(
            np.mean(
                scipy.signal.lfilter(
                    b,
                    a,
                    (
                        shmat[:, : (TRUNC_ORDER + 1) ** 2]
                        @ hoa_array[:, : (TRUNC_ORDER + 1) ** 2, :]
                    )[:, :, MIC],
                    axis=0,
                )
                ** 2,
                axis=0,
            )
        ),
        vmin=VMIN,
        vmax=VMAX,
    )
    plt.title("resampled")
    plt.colorbar()
    plt.grid(True)

    plt.subplot(212, projection="mollweide")
    plt.scatter(
        azi,
        np.pi / 2 - zenith,
        s=10,
        c=10
        * np.log10(
            np.mean(scipy.signal.lfilter(b, a, ir[:, :, MIC], axis=0) ** 2, axis=0)
        ),
        vmin=VMIN,
        vmax=VMAX,
    )
    plt.title("sampled")
    plt.colorbar()
    plt.grid(True)
    plt.show()

np.save("Easycom_array_%dHz_o%d_22samps_delay.npy" % (FS, INTERP_ORDER), hoa_array)
