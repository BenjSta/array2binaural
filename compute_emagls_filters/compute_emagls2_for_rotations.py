# Compute end2end magnitude least squares filters. This code is inspired by https://github.com/thomasdeppisch/eMagLS.

import os
import spaudiopy
import soundfile
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from ambisonics import calculate_rotation_matrix
import scipy.signal as signal
# torch is used for fast computation on GPU (note that automatic differentiation is not used)
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0,'
DEVICE = 'cuda'

# load reference HRIRs from 5th-order symmetric HRTFs
FS = 48000 # the filters are computed at 48kHz for usage with standard DAWs
hrir, fs = soundfile.read('compute_emagls2_for_rotations/irsOrd5.wav')
hrir = signal.resample_poly(hrir, FS, fs, axis=0) * fs / FS
hrir_delay = np.argmax(np.abs(hrir[:, 0]))
nm = np.arange(0, hrir.shape[1])
n = np.floor(np.sqrt(nm))
m = nm - n**2 - n
mult = np.zeros(hrir.shape[1])
mult[m >= 0] = 1
mult[m < 0] = -1
right = hrir * mult[None, :]
left = hrir

# DFT to obtain HRTFs
NFFT = 1536
fvec = np.arange(NFFT//2+1) / NFFT * FS
dl_sh5 = np.fft.rfft(left, NFFT, axis=0)
dr_sh5 = np.fft.rfft(right, NFFT, axis=0)
dl_sh5 *= np.exp(1j * 2 * np.pi * fvec * hrir_delay / FS)[:, None]
dr_sh5 *= np.exp(1j * 2 * np.pi * fvec * hrir_delay / FS)[:, None]

# load ambisonic array impulse responses
ir_sh = np.load('Easycom_32000Hz_o25_22samps_delay.npy')
FS_ARRAY = 32000
if FS_ARRAY != FS:
    ir_sh = FS_ARRAY / FS * signal.resample_poly(ir_sh, FS, FS_ARRAY, axis=0)
mic_delay = int(22 * FS / FS_ARRAY)
optim_grid = spaudiopy.grids.load_n_design(35)
azi, polar, _ = spaudiopy.utils.cart2sph(optim_grid[:, 0], 
                                         optim_grid[:, 1], 
                                         optim_grid[:, 2])

# sh matrices for the grid on which the optimization is carried out
optim_sh5 = spaudiopy.sph.sh_matrix(5, azi, polar, 'real')
optim_sh25 = spaudiopy.sph.sh_matrix(25, azi, polar, 'real')

# decode to obtain space-domain impulse responses ...
optim_irs = optim_sh25[None, :, :] @ ir_sh

# ... and transfer functions / steering vectors
d = np.fft.rfft(optim_irs, NFFT, axis=0)
d = d * np.exp(1j * 2 * np.pi * fvec * mic_delay / FS)[:, None, None]

# define a grid of relative array/listener orientations
yaw = np.linspace(-np.pi, np.pi, 60, endpoint=False)
pitch = np.linspace(-np.pi/3, np.pi/3, 21, endpoint=True)
yaw, pitch = np.meshgrid(yaw, pitch)
yaw = yaw.flatten()
pitch = pitch.flatten()
dir_x_rel = np.cos(yaw) * np.cos(pitch)
dir_y_rel = np.sin (yaw) * np.cos(pitch)
dir_z_rel = -np.sin(pitch)

gridpoints = np.stack([dir_x_rel, dir_y_rel, dir_z_rel], -1)

roll =  np.linspace(-np.pi/4, np.pi/4, 16, endpoint=True)


# and compute a 5th-order rotation matrix
yaw = np.tile(yaw[:, None], (1, roll.shape[0]))
pitch = np.tile(pitch[:, None], (1, roll.shape[0]))
roll = np.tile(roll[None, :], (yaw.shape[0], 1))
rotmat_o5 = calculate_rotation_matrix(5, yaw, pitch, roll)



with torch.no_grad():
    dl_sh5 = torch.from_numpy(dl_sh5.astype('complex64')).to(DEVICE)
    dr_sh5 = torch.from_numpy(dr_sh5.astype('complex64')).to(DEVICE)
    d = torch.from_numpy(d.astype('complex64')).to(DEVICE)
    rotmat_o5 = torch.from_numpy(rotmat_o5.astype('float32')).to(DEVICE)
    optim_sh5 = torch.from_numpy(optim_sh5.astype('float32')).to(DEVICE)

    # decode HRTFs withot rotation
    dl0 = torch.sum(optim_sh5[None, :, :] * dl_sh5[:, None, :], -1)
    dr0 = torch.sum(optim_sh5[None, :, :] * dr_sh5[:, None, :], -1)

    H = torch.stack([dl0, dr0], axis=-1)
    # obtain diffuse field 2x2 covariance matrix of HRTFs
    R = 1/H.shape[1] * torch.sum((H[..., None] @ torch.conj(H[..., None, :])), dim=1)

    # MxM diffuse-field covariance of array
    diff_cov = torch.conj(d.permute(0, 2, 1)) @ d 
    e = torch.linalg.eigvalsh(diff_cov)
    diff_cov = diff_cov + 0.001 * e[:, -1][:, None, None] * \
        torch.eye(diff_cov.shape[-1], device=diff_cov.device)[None, :, :]
    # ... and its pseudo-inverse with diagonal loading
    regInvY = torch.linalg.inv(diff_cov) @ torch.conj(d.permute(0, 2, 1))


    all_mls_l = []
    all_mls_r = []

    err_l_all = []
    err_r_all = []

    f_cut = 2000

    for g1 in tqdm.tqdm(range(rotmat_o5.shape[0])):
        sh5_rotated = rotmat_o5[g1, :, None, :, :] @ optim_sh5[None, :, :, None]
        dl = torch.sum(sh5_rotated * dl_sh5.T[None, None, :, :], -2)
        dr = torch.sum(sh5_rotated * dr_sh5.T[None, None, :, :], -2)

        w_mls_l = torch.zeros((dl.shape[0], fvec.shape[0], 5), dtype=torch.complex64).to(DEVICE)
        w_mls_r = torch.zeros((dr.shape[0], fvec.shape[0], 5), dtype=torch.complex64).to(DEVICE)
    
        for k, freq in enumerate(fvec):
            # least-squares below f_cut
            w_ls_l = (regInvY[None, k, :, :] @ dl[:, :, k, None])[..., 0]
            w_ls_r = (regInvY[None, k, :, :] @ dr[:, :, k, None])[..., 0]
            
            # magnitude-least-squares above
            if freq > f_cut:
                phiMagLsSmaL = torch.angle(d[None, k, :, :] @ w_mls_l[:, k-1, :, None])[..., 0]
                phiMagLsSmaR = torch.angle(d[None, k, :, :] @ w_mls_r[:, k-1, :, None])[..., 0]

                w_mls_l[:, k, :] = (regInvY[None, k, :, :] @ (torch.abs(dl[:, :, k, None]) * torch.exp(1j * phiMagLsSmaL[..., None])))[..., 0]
                w_mls_r[:, k, :] = (regInvY[None, k, :, :] @ (torch.abs(dr[:, :, k, None]) * torch.exp(1j * phiMagLsSmaR[..., None])))[..., 0]
            else:
                w_mls_l[:, k, :] = w_ls_l
                w_mls_r[:, k, :] = w_ls_r

            # compute nothing above the array steering vectors' sampling frequency
            if freq > FS_ARRAY/2:
                break
        
        # covariance of magLS HRTF set
        Hhat = torch.stack([w_mls_l, w_mls_r], axis=-1)
        Hest = torch.stack([torch.sum(w_mls_l[:, :, None, :] * d[None, :, :, :], -1), 
                        torch.sum(w_mls_r[:, :, None, :] * d[None, :, :, :], -1)], axis=-1)            
        Rhat = 1/H.shape[1] * torch.sum((Hest[..., None] @ torch.conj(Hest[..., None, :])), 
                                        dim=2) + 1e-20 * torch.eye(2).to(Hest.device)[None, None, :, :]
        
        # diffuse-field equalization / constraint
        DiffL = Rhat[..., 0, 0]
        DiffLTarget = R[..., 0, 0]
        DiffR = Rhat[..., 1, 1]
        DiffRTarget = R[..., 1, 1]

        w_mls_l = torch.sqrt(torch.abs((DiffLTarget / (DiffL + 1e-20)))[..., None]) * w_mls_l#torch.conj(HCorr[..., 0])
        w_mls_r = torch.sqrt(torch.abs((DiffRTarget / (DiffR + 1e-20)))[..., None]) * w_mls_r#torch.conj(HCorr[..., 1])
    
        all_mls_l.append(w_mls_l.detach().cpu().numpy())
        all_mls_r.append(w_mls_r.detach().cpu().numpy())


all_mls_l = np.stack(all_mls_l, 0)
all_mls_r = np.stack(all_mls_r, 0)



all_mls = torch.stack([torch.from_numpy(all_mls_l), 
                       torch.from_numpy(all_mls_r)], -1).permute(0, 1, 3, 4, 2)

# time domain
all_mls_td = torch.fft.irfft(all_mls, axis=-1)
all_mls_fd_orig = all_mls.detach().cpu().numpy().copy()
all_mls_td = torch.roll(all_mls_td, (NFFT//2,), (-1,))

# cut to central part
FILTLEN = 768
all_mls_td = all_mls_td[:, :, :, :, NFFT//2 - FILTLEN//2 : NFFT//2 + FILTLEN//2]
all_mls_td = all_mls_td.detach().cpu().numpy() * signal.windows.tukey(all_mls_td.shape[-1], 0.1)

# Processing blocksize will be 256, so for a fast convolution with FILTLEN = 768, FFTSIZE=1024 is required 
# transform to frequency-domain for fast convolution in VST plugin
FFTSIZE = 1024
all_mls_fd = np.fft.rfft(all_mls_td, FFTSIZE, -1)
all_mls_fd = np.stack([np.real(all_mls_fd), np.imag(all_mls_fd)], -1).astype('float32')
all_mls_fd = all_mls_fd.reshape(all_mls_td.shape[0], 
                               all_mls_td.shape[1], 
                               all_mls_td.shape[2], 
                               all_mls_td.shape[3], -1)


np.save('compute_emagls2_for_rotations/xyz.npy', gridpoints)
np.save('compute_emagls2_for_rotations/filters.npy', all_mls_fd)
np.save('compute_emagls2_for_rotations/filters_fd.npy', all_mls_fd_orig[..., :513])
np.save('compute_emagls2_for_rotations/roll.npy', roll[0, :])

