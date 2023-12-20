# %%
import tqdm
import scipy.signal
from spaudiopy.sph import sh_matrix
from scipy.io import loadmat
from ambisonics import sph_hankel2_diff, sh_azi_zen, sph2cart
import numpy as np
import matplotlib.pyplot as plt
import h5py
from joblib import Parallel, delayed

# %% define global parameters
NFFT = 512
FS = 32000
snr_all = np.arange(10, -20.1, -3)

#%% load original easycom array
PATH = 'origin_array_tf_data/Device_ATFs.h5'
f = h5py.File(PATH, 'r')
grid_phi = f['Phi'][0, :]
grid_theta = f['Theta'][0, :]
ir_orig = scipy.signal.resample_poly(
    np.array(f['IR'])[:, :, [0, 1, 2, 4, 5]], FS, list(f['SamplingFreq_Hz'])[0][0])
grid_phi = grid_phi[grid_theta <= np.pi/2]
ir_orig = ir_orig[:, grid_theta <= np.pi/2, :]
grid_theta = grid_theta[grid_theta <= np.pi/2]
# define upper half of the easycom grid as the MUSIC search grid
# and the grid of possible source directions
shmat_grid = sh_matrix(25, grid_phi, grid_theta, 'real')

#%% Load SH represented Easycom array data
MIC_PATH = 'Easycom_array_32000Hz_o25_22samps_delay.npy'
sh_array_easycom = np.load(MIC_PATH)
    
# %% spherical scatterer model


def getRadialTerms(N, Nfft, fs, R):
    """Returns the radial terms of a spherical scatterer in the frequency domain.

    Args:
        N (int): SH order
        fc (ndarray): cut-on frequencies for bandpass regularization filters
        Nfft (int): FFT size
        R (float): array radius

    Returns:
        ndarray: limited radial filters
    """

    f = np.linspace(0, fs/2, int(Nfft/2)+1)
    f[0] = f[1]/4

    #  Parameters, Constants
    c = 343
    k = 2*np.pi*f / c
    hn = np.zeros((len(f), N+1), dtype=complex)
    hankel2_diff = sph_hankel2_diff(k*R, N)
    for n in range(N+1):
        hn[:, n] = 4 * np.pi * (1j)**(n+1) / ((k*R)**2 * hankel2_diff[:, n])

    return hn


def getSphericalMicTF(N, azi, zen, Nfft, fs, R):
    """Returns the sh-domain tranfer functions of microphones on a spherical scatterer"""
    H = getRadialTerms(N, Nfft, fs, R)
    print(np.max(np.abs(H)))
    H_all = []
    for i in range(N+1):
        H_all += [H[:, i]]*(i*2+1)

    H_all = np.stack(H_all, -1)
    sh = sh_azi_zen(N, azi, zen)

    return H_all[:, None, :] * sh[None, :, :]


# %% load array responsed by McCormack et al.
MIC_PATH = 'origin_array_tf_data/HMD_SensorArrayResponses.mat'
mic_space = loadmat(MIC_PATH)
dirs_deg = mic_space['dirs_deg']
fvec_mccormack = mic_space['freqs'][:, 0]
mic_responses = mic_space['micResponses']
azi = dirs_deg[:, 0] / 180 * np.pi
ele = dirs_deg[:, 1] / 180 * np.pi
zenith = np.pi/2 - ele
shmat_mccormack = sh_matrix(25, azi, zenith)
# encode into SH domain
sh_array_mccormack = (np.linalg.pinv(shmat_mccormack) @ mic_responses[..., None])[..., 0]




# %% This is the core function: computes the error of the MUSIC algorithm for a
# grid of directions and corresponding source covariance matrices
def compute_music_error_at_snr(snr, cov_source, cov_isotropic, grid_phi,
                               grid_theta, steer_search):
    snr_no_db = 10**(snr/10)
    cov_source_noisy = cov_source + 1/snr_no_db * cov_isotropic[:, None, :, :]
    s, V = np.linalg.eigh(cov_source_noisy)
    noise_subspace = V[..., :-1] @ np.conj(V[..., :-1].transpose(0, 1, 3, 2))

    music_denom = (np.conj(steer_search[:, :, None, None, :]) @
                   noise_subspace[:, None, :, :, :] @ steer_search[:, :, None, :, None])[..., 0, 0]

    found_ind = np.argmin(music_denom, axis=2)

    
    found_phi = grid_phi[found_ind]
    found_theta = grid_theta[found_ind]
    found_ele = np.pi/2 - found_theta
    grid_ele = np.pi/2 - grid_theta
    
    found_xyz = sph2cart(found_phi, found_theta)
    grid_xyz = sph2cart(grid_phi, grid_theta)

    dot_product = np.clip(np.sum(found_xyz * grid_xyz[:, None, :], 0), -1, 1)
    assert np.all(np.abs(dot_product) <= 1), 'out of range for arccos'
    
    angle_diff = np.arccos(dot_product)



    # # Haversine formula
    # def hav(x):
    #     return (1-np.cos(x)) / 2

    # azi_diff = found_phi - grid_phi
    # zenith_diff = found_ele - grid_ele
    # a = (
    #     hav(zenith_diff)
    #     + np.cos(found_ele)
    #     * np.cos(grid_ele)
    #     * hav(azi_diff) ** 2
    # )
    # angle_diff = 2 * np.arcsin(np.sqrt(a))
    return (angle_diff,)


# %%
angle_diff_over_methods = []
fvec_over_methods = []
ARRAYS = ["SphericalScatterer-5", "SphericalScatterer-7", "Easycom-5-Orig",
          "BEM-5-EncDec-ld0", "BEM-7-EncDec-ld0",
          "Easyc-5-EncDec-ld0.1-cut-ir-search", "SphericalScatterer-7-circ"]

# %%
for array_str in ARRAYS:
    # "default" frequency vector
    fvec = np.arange(NFFT//2 + 1) / NFFT * FS
    if array_str == "SphericalScatterer-5":
        sh_array = getSphericalMicTF(25,
                                     np.concatenate([
                                        np.linspace(-np.pi / 2, np.pi / 2, 5,
                                        endpoint=True)]), 
                                        np.array([np.pi/2]*5), NFFT, FS, 0.08)

    elif array_str == "SphericalScatterer-7":
        sh_array = getSphericalMicTF(25, np.concatenate([np.linspace(-np.pi / 2, np.pi / 2, 5,
                                                                     endpoint=True), np.array([0, np.pi])]), np.array([np.pi/2]*5 + [0, np.pi/2]), NFFT, FS, 0.08)
    elif array_str == "SphericalScatterer-7-circ":
        sh_array = getSphericalMicTF(25, np.concatenate([np.linspace(-np.pi / 2, np.pi / 2, 5,
                                                                     endpoint=True), np.array([-5 * np.pi / 6, 5 * np.pi / 6])]), np.array([np.pi/2]*5 + [np.pi/2, np.pi/2]), NFFT, FS, 0.08)
    
    elif array_str == "Easycom-5-EncDec-ld0.1" or array_str == "Easyc-5-EncDec-ld0.1-cut-ir-search":
        if array_str != "Easyc-5-EncDec-ld0.1-cut-ir-search":
            sh_array = np.fft.rfft(sh_array_easycom, axis=0).transpose(0, 2, 1)
        else:
            sh_array = np.fft.rfft(
                sh_array_easycom[:340, :, :], n=NFFT, axis=0).transpose(0, 2, 1) 
    
    elif array_str == "BEM-7-EncDec-ld0" or array_str == "BEM-5-EncDec-ld0":
        # "default" frequency vector for BEM simulation
        fvec = fvec_mccormack
        if array_str == "BEM-5-EncDec-ld0":
            sh_array = sh_array_mccormack[:, :5, :]
        else:
            sh_array = sh_array_mccormack

    if array_str != "Easycom-5-Orig":
        # when working SH representations, just decode from them
        steer_search = (shmat_grid@sh_array[:, :, :, None]
                        )[..., 0].transpose(0, 2, 1)
        cov_isotropic = sh_array @ sh_array.transpose(0, 2, 1) / (4*np.pi)
        steer_source = (shmat_grid@sh_array[:, :, :, None]
                        )[..., 0].transpose(0, 2, 1)
    
    else:
        # when directly working with impulse responses
        steer_source = np.fft.rfft(ir_orig, axis=0)
        steer_search = steer_source.copy()
        weights = np.diag(np.sin(grid_theta))
        weights /= np.trace(weights)
        cov_isotropic = steer_source.transpose(
            0, 2, 1) @ weights @ np.conj(steer_source)

    cov_source = steer_source[:, :, :, None] @ np.conj(steer_source[:, :, None, :])

    angle_diff_all = Parallel(n_jobs=4)(delayed(compute_music_error_at_snr)(
        snr, cov_source, cov_isotropic, grid_phi, grid_theta, steer_search) for snr in tqdm.tqdm(snr_all))

    angle_diff_over_methods.append(angle_diff_all)
    fvec_over_methods.append(fvec)

# %%
mult = 0.8
errors_fig = plt.figure(figsize=(14.5*mult, 2.8*mult))

for plt_ind, array_ind in enumerate([5, 2, 0, 3, 1,  4, 6]):

    array_str = ARRAYS[array_ind]
    ad = angle_diff_over_methods[array_ind]
    fvec = fvec_over_methods[array_ind]
    median_ad_total = np.array([np.median(m[0], axis=1) for m in ad])
    perc90_ad_total = np.array([np.quantile(m[0], 0.9, axis=1) for m in ad])

    # plt.figure(perc90_errors_fig)
    myplot = plt.subplot(1, 7, plt_ind+1)
    contourplot = plt.contourf(snr_all,
                               fvec,
                               np.clip(180*median_ad_total.T/np.pi, 0, 30),
                               cmap='Reds', levels=3, vmin=0, vmax=30, extend='neither')

    plt.yscale('log')
    if not (plt_ind == 0):
        plt.yticks([])

    # if plt_ind >= 3:
    plt.xlabel('DDR in dB')
    # else:
    plt.xticks([10, 0, -10, -20], ['10', '0', '-10', ''])
    if np.mod(plt_ind, 7) == 0:
        plt.ylabel('frequency in Hz')
    plt.ylim([100, 12000])

    if False:
        cbar = plt.colorbar()
        cbar.set_label('median angular error')

    if array_str == "Easyc-5-EncDec-ld0.1-cut-ir-search":
        title = "Easycom-5Mic-EncDec"
    elif array_str == "Easycom-5-Orig":
        title = "Easycom-5Mic"
    elif array_str == "BEM-5-EncDec-ld0":
        title = "BEM-5Mic"
    elif array_str == "BEM-7-EncDec-ld0":
        title = "BEM-7Mic"
    elif array_str == "SphericalScatterer-5":
        title = "Sphere-5Mic"
    elif array_str == "SphericalScatterer-7":
        title = "Sphere-7Mic"
    elif array_str == "SphericalScatterer-7-circ":
        title = "Sphere-7Mic-circ"

    plt.title(title, fontsize=10)
    plt.tight_layout()
    myplot.invert_xaxis()

errors_fig.subplots_adjust(bottom=0.22, left=0.06,
                           right=0.9, wspace=0.07, hspace=0.29)
cbar_ax = errors_fig.add_axes([0.92, 0.15, 0.008, 0.7])
cbar = errors_fig.colorbar(contourplot, cax=cbar_ax)
cbar.set_label('median angular error\nin degrees')
cbar.ax.set_yticks([24, 16, 8, 0])
cbar.ax.set_yticklabels(['24', '16', '8', '0'])

plt.savefig('figures/music_sim.svg', dpi=300, bbox_inches='tight')
plt.show(block=True)
print('Done')
# %%
