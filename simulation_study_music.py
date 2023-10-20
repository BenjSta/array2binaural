#%%
%matplotlib qt
import h5py
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
import matplotlib.colors
PATH = 'Device_ATFs.h5'
f = h5py.File(PATH,'r')
f_phi = f['Phi']
f_theta = f['Theta']
# #%%
# spear_hoa_d, fs  = soundfile.read('ambix2array_fir.wav')
# spear_hoa_d = spear_hoa_d.reshape(256, spear_hoa_d.shape[0] // 256, 6).transpose(1, 0, 2)
# spear_hoa_d = scipy.signal.resample_poly(np.array(spear_hoa_d)[:, :, [0, 1, 2, 4, 5]], FS, fs)
import matplotlib.pyplot as plt
import numpy as np
from ambisonics import sph_hankel2_diff, sh_azi_zen

from scipy.io import loadmat
from spaudiopy.sph import sh_matrix
import scipy.signal
import tqdm
from joblib import Parallel, delayed
#%%

def getRadialTerms(N, Nfft, fs, R):
    """Returns the radial terms of a spherical scatterer in the frequency domain.
    LS coordinates only necesarry for excursion checks / fc  setting and tuning, when plotIt==True 

    Args:
        N (int): SH order
        fc (ndarray): cut-on frequencies for bandpass regularization filters
        Nfft (int): FFT size
        R (float): array radius
        rLS (float): LS radius
        ls_thetaphi (ndarray): LS coordinates, only necesarry for excursion checks / fc setting
        mixo_idx (ndarray): mixed-order indices
        array_name (str, optional): array name. Defaults to "".
        plotIt (bool, optional): flat for plotting results. Defaults to False.

    Returns:
        ndarray: limited radial filters
    """    

    f=np.linspace(0,fs/2, int(Nfft/2)+1)
    f[0]=f[1]/4

    #  Parameters, Constants
    c = 343
    k = 2*np.pi*f / c
    #rho = 1.2
    
    hn = np.zeros((len(f), N+1), dtype=complex)

    for n in range(N+1):
        hn[:,n] = 4 * np.pi * (1j)**(n+1) /  ((k*R)**2 * sph_hankel2_diff(k*R, N))

    return hn

#%%
def getSphericalMicIR(N, azi, zen, Nfft, fs, R):
    H = getRadialTerms(N, Nfft, fs, R)
    print(np.max(np.abs(H)))
    H_all = []
    for i in range(N+1):
        H_all += [H[:, i]]*(i*2+1)
        
    H_all = np.stack(H_all, -1)
    #print(H_all.shape)
    sh1 = sh_azi_zen(N, azi[0], zen[0])
    #print(sh1[:, 0])
    #norm = np.sum(sh1 * sh1)
    sh = sh_azi_zen(N, azi, zen)
    print(np.linalg.norm(sh1))
    #Hf = H_all[:, None, :] * sh1[None, :, :]

    #print(Hf.shape)
    return H_all[:, None, :] * sh[None, :, :]


#%%
MIC_AMBI_PATH = 'HMD_SensorArrayResponses.mat'
mic_space = loadmat(MIC_AMBI_PATH)
dirs_deg = mic_space['dirs_deg']
freqs =  mic_space['freqs']
mic_responses =  mic_space['micResponses']
azi = dirs_deg[:, 0] / 180 * np.pi
ele = dirs_deg[:, 1] / 180 * np.pi
zenith = np.pi/2 - ele
shmat_leo = sh_matrix(25, azi, zenith)
sh_array = (np.linalg.pinv(shmat_leo) @ mic_responses[..., None])[..., 0]
#sh_array = np.fft.rfft(mic_ambi, axis=0).transpose(0, 2, 1)

#%%
f_phi = f_phi[:, f_theta[0, :] <= np.pi/2]
f_theta = f_theta[:, f_theta[0, :] <= np.pi/2]
#%%
#data = loadmat(path)
NFFT = 512
FS = 32000

snr_all = np.arange(10, -20.1, -3)


#plt.subptitle('90th Percentile Angular Error')


#%%
def compute_music_error_at_snr(snr, cov_source, C, fphi, ftheta, steer_search):
    snr_no_db = 10**(snr/10)
    cov_source_noisy = cov_source + 1/snr_no_db * C[:, None, :, :] 
    s, V = np.linalg.eigh(cov_source_noisy)
    noise_subspace = V[..., :-1] @ np.conj(V[..., :-1].transpose(0, 1, 3, 2))


    music_denom =  (np.conj(steer_search[:, :, None, None, :]) @ \
        noise_subspace[:, None, :, :, :] @ steer_search[:, :, None, :, None])[..., 0, 0]


    found_ind = np.argmin(music_denom, axis=2)

    
    found_phi = fphi[0, :][found_ind]
    found_theta = ftheta[0, :][found_ind]

    # Haversine formula
    azi_diff = found_phi - fphi

    azi_diff = np.mod(azi_diff, 2*np.pi)

    azi_diff[azi_diff > np.pi] = \
        azi_diff[azi_diff > np.pi] - 2 * np.pi
    azi_diff[azi_diff < -np.pi] = \
        azi_diff[azi_diff < -np.pi] + 2 * np.pi

    zenith_diff = found_theta - ftheta
    a = (
        np.sin(zenith_diff / 2) ** 2
        + np.cos(found_theta)
        * np.cos(ftheta)
        * np.sin(azi_diff / 2) ** 2
    )
    angle_diff = 2 * np.arcsin(np.sqrt(a))
    return angle_diff, azi_diff, zenith_diff

#%%
angle_diff_over_methods = []
fvec_over_methods = []
ARRAYS = ["SphericalScatterer-5", "SphericalScatterer-7","Easycom-5-Orig", 
                                       "BEM-5-EncDec-ld0", "BEM-7-EncDec-ld0", 
                                        "Easycom-5-EncDec-ld0.1", "Easyc-5-EncDec-ld0.1-cut-ir-search"]


#%%
for array_str in ARRAYS:
    fvec = np.arange(NFFT//2+ 1) / NFFT * FS
    if array_str == "SphericalScatterer-5":
        sh_array = getSphericalMicIR(25, np.concatenate([np.linspace(-np.pi / 2, np.pi / 2, 5, 
                                            endpoint=True)]), np.array([np.pi/2]*5), NFFT, FS, 0.08)

    elif array_str == "SphericalScatterer-7":
        sh_array = getSphericalMicIR(25, np.concatenate([np.linspace(-np.pi / 2, np.pi / 2, 5, 
                                            endpoint=True), np.array([0, np.pi])]), np.array([np.pi/2]*5 + [0, np.pi/2]), NFFT, FS, 0.08)
    elif array_str == "Easycom-5-EncDec-ld0.1" or  array_str == "Easyc-5-EncDec-ld0.1-cut-ir-search":
        MIC_AMBI_PATH = 'SPEAR_array_32000Hz_o25_22samps_delay.npy'
        mic_ambi = np.load(MIC_AMBI_PATH)
        sh_array = np.fft.rfft(mic_ambi, axis=0).transpose(0, 2, 1)

        if array_str == "Easyc-5-EncDec-ld0.1-cut-ir-search":
            sh_array_cut_steervec = np.fft.rfft(mic_ambi[:340, :, :], n=512, axis=0).transpose(0, 2, 1)
    elif array_str == "BEM-7-EncDec-ld0" or array_str =="BEM-5-EncDec-ld0":
        MIC_AMBI_PATH = 'HMD_SensorArrayResponses.mat'
        mic_space = loadmat(MIC_AMBI_PATH)
        dirs_deg = mic_space['dirs_deg']
        fvec =  mic_space['freqs'][:, 0]
        mic_responses =  mic_space['micResponses']
        azi = dirs_deg[:, 0] / 180 * np.pi
        ele = dirs_deg[:, 1] / 180 * np.pi
        zenith_leo = np.pi/2 - ele
        azi_leo = azi.copy()
        shmat_leo = sh_matrix(25, azi, zenith_leo)

        if array_str == "BEM-5-EncDec-ld0":
            sh_array = (np.linalg.pinv(shmat_leo) @ mic_responses[..., :5, :, None])[..., 0]
        else:
            sh_array = (np.linalg.pinv(shmat_leo) @ mic_responses[..., None])[..., 0]




    if array_str != "Easycom-5-Orig":
        shmat = sh_matrix(25, f_phi, f_theta, 'real')
        steer_search = (shmat@sh_array[:, :, :, None])[..., 0].transpose(0, 2, 1)
        if array_str == "Easyc-5-EncDec-ld0.1-cut-ir-search":
            steer_search = (shmat@sh_array_cut_steervec[:, :, :, None])[..., 0].transpose(0, 2, 1)
        C = sh_array @ sh_array.transpose(0, 2, 1) / (4*np.pi)
        steer_source = (shmat@sh_array[:, :, :, None])[..., 0].transpose(0, 2, 1)
    else:
        f = h5py.File('Device_ATFs.h5', 'r')
        fs = int(list(f['SamplingFreq_Hz'])[0][0])
        #AIR['IR'] = np.array(f['IR']) # (ndarray) [nSample x nDirection x nChan]
        ir_orig = scipy.signal.resample_poly(np.array(f['IR'])[:, :, [0, 1, 2, 4, 5]], FS, fs)
        

        f = h5py.File(PATH,'r')
        ele = (np.pi/2)-np.array(f['Theta']) # (ndarray) elevation in radians [1 x nDirection]
        ele = ele[0, :]

        steer_source = np.fft.rfft(ir_orig[:, ele >= 0, :], axis=0)
        steer_search = steer_source.copy()

        theta = -ele[ele >= 0] + np.pi/2
        azi = np.array(f['Phi']) # (ndarray) azimuth in radians [1 x nDirection]
        azi = azi[0, :]
        azi[azi > np.pi] = azi[azi > np.pi] - 2 * np.pi

        weights = np.diag(np.sin(theta))
        C = steer_source.transpose(0, 2, 1) @ weights @ np.conj(steer_source) / np.trace(weights)


    cov_source = steer_source[:, :, :, None] @ np.conj(steer_source[:, :, None, :])

    
    #np.mean(angle_diff, axis=1), np.mean(np.abs(azi_diff), axis=1), np.mean(np.abs(zenith_diff), axis=1),\
            #np.quantile(angle_diff, 0.9, axis=1), np.quantile(np.abs(azi_diff), 0.9, axis=1), \
            #np.quantile(np.abs(zenith_diff), 0.9, axis=1) 




    angle_diff_all = Parallel(n_jobs=4)(delayed(compute_music_error_at_snr)(
        snr, cov_source, C, f_phi, f_theta, steer_search) for snr in tqdm.tqdm(snr_all))

    angle_diff_over_methods.append(angle_diff_all)
    fvec_over_methods.append(fvec)
    # mean_angle_diff_total = np.array([m[0] for m in angle_diff_all])
    # mean_angle_diff_azi = np.array([m[1] for m in angle_diff_all])
    # mean_angle_diff_zenith = np.array([m[2] for m in angle_diff_all])

    # perc90_angle_diff_total = np.array([m[3] for m in angle_diff_all])
    # perc90_angle_diff_azi = np.array([m[4] for m in angle_diff_all])
    # perc90_angle_diff_zenith = np.array([m[5] for m in angle_diff_all])
    
#%%

#Maybe what you are looking for is svgutils

#import svgutils.compose as sc



mult = 0.8
errors_fig = plt.figure(figsize=(13*mult, 2.8*mult))

for plt_ind, array_ind in enumerate([2, 6, 0, 3, 1,  4]):
    
    array_str = ARRAYS[array_ind]
    ad = angle_diff_over_methods[array_ind]
    fvec = fvec_over_methods[array_ind]
    mean_ad_total = np.array([np.median(m[0], axis=1) for m in ad])
    perc90_ad_total = np.array([np.quantile(m[0], 0.9, axis=1) for m in ad])
    
    # myplot = plt.subplot(2, 6, 2*array_ind+1)
    # plt.contourf(snr_all, 
    #             fvec, 
    #             np.clip(180*mean_ad_total.T/np.pi,0, 50),
    #             normalize=matplotlib.colors.Normalize(vmin=0, vmax=50, clip=True),
    #         cmap='Reds', levels=4, vmin=0, vmax=50, extend='neither')
    # cbar = plt.colorbar()       
    # plt.yscale('log')
    # if array_ind >= 3:
    #     plt.xlabel('SNR in dB')
    # if np.mod(array_ind, 3) == 0:
    #     plt.ylabel('frequency in Hz')
    # plt.ylim([200, 12000])

    # cbar.set_label('mean angular error')
    # plt.title(array_str)
    # plt.tight_layout()
    # myplot.invert_xaxis()


    #plt.figure(perc90_errors_fig)
    myplot = plt.subplot(1, 6, plt_ind+1)
    contourplot = plt.contourf(snr_all, 
                fvec, 
                np.clip(180*mean_ad_total.T/np.pi, 0, 30),
                normalize=matplotlib.colors.Normalize(vmin=0, vmax=30, clip=True),
            cmap='Reds', levels=3, vmin=0, vmax=30, extend='neither')
   
    plt.yscale('log')
    if not (plt_ind == 0):
        plt.yticks([])

    #if plt_ind >= 3:
    plt.xlabel('DDR in dB')
    #else:
    plt.xticks([10, 0, -10, -20], ['10', '0', '-10', ''])
    if np.mod(plt_ind, 6) == 0:
        plt.ylabel('frequency in Hz')
    plt.ylim([100, 12000])

    if False:
        cbar = plt.colorbar()       
        cbar.set_label('median angular error')
        cbar.ax.set_yticks([24, 16, 8, 0])
        cbar.ax.set_yticklabels(['24', '16', '8', '0'])

    if array_str == "Easyc-5-EncDec-ld0.1-cut-ir-search":
        title = "Easycom-5Mic-EncDec"
    elif  array_str == "Easycom-5-Orig":
        title = "Easycom-5Mic"
    elif  array_str == "BEM-5-EncDec-ld0":
        title = "BEM-5Mic"
    elif  array_str == "BEM-7-EncDec-ld0":
        title = "BEM-7Mic"
    elif  array_str == "SphericalScatterer-5":
        title = "Sphere-5Mic"
    elif  array_str == "SphericalScatterer-7":
        title = "Sphere-7Mic"
    
    #title = title.replace('-ld0', "")
    plt.title(title, fontsize=10)
    plt.tight_layout()
    myplot.invert_xaxis()

errors_fig.subplots_adjust(bottom=0.22, left=0.06, right=0.92, wspace=0.07, hspace=0.29)
cbar_ax = errors_fig.add_axes([0.94, 0.15, 0.008, 0.7])
#errors_fig.subplots_adjust(right=0.91, wspace=0.08, hspace=0.24)
cbar = errors_fig.colorbar(contourplot, cax=cbar_ax)
cbar.set_label('median angular error')
cbar.ax.set_yticks([24, 16, 8, 0])
cbar.ax.set_yticklabels(['24', '16', '8', '0'])

plt.show()
# %%
