from pyfilterbank import gammatone
from pyfilterbank import GammatoneFilterbank
import matplotlib.pyplot as plt
import numpy as np
import spaudiopy
import soundfile
import scipy.signal as signal
import os
import sys
dirpath = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(dirpath)
sys.path.append(parent)
from ambisonics import calculate_rotation_matrix

PLOT_DB_RANGE = 20
MIC_AMBI_PATH = os.path.join(
    parent, 'Easycom_array_32000Hz_o25_22samps_delay.npy')
fs = 32000
array_sh_delay = 22 # samples
array_sh = np.load(MIC_AMBI_PATH)

DENSITY = 1
filtbank = GammatoneFilterbank(fs,
                               startband=-12.5,
                               endband=10,
                               density=DENSITY,
                               bandwidth_factor=1.0)
gfbmat = []
for f in filtbank.freqz(1024):
    gfbmat.append(np.abs(f[0]))
gfbmat = np.stack(gfbmat)
fvec_gfb = gammatone.erbscale_to_hertz(
    np.arange(-12.5, 10, DENSITY) + 15.62144971397049)

# create plot grid
plt.close('all')
fig, ax = plt.subplots(5,
                       8,
                       sharey=False,
                       sharex=True,
                       figsize=(13, 7),
                       gridspec_kw={
                           'wspace': 0.1,
                           'hspace': 0.1
                       },
                       width_ratios=[1, 0.02, 1, 1, 1, 0.02, 1, 1.2],
                       height_ratios=[1, 0.5, 0.02, 1, 0.5])

# dummy axes (used for spacing) should be invisble
ax[0, 1].set_visible(False)
ax[1, 1].set_visible(False)

ax[0, 5].set_visible(False)
ax[1, 5].set_visible(False)

ax[2, 0].set_visible(False)
ax[2, 1].set_visible(False)
ax[2, 2].set_visible(False)
ax[2, 3].set_visible(False)
ax[2, 4].set_visible(False)
ax[2, 5].set_visible(False)
ax[2, 6].set_visible(False)
ax[2, 7].set_visible(False)

ax[2, 1].set_visible(False)
ax[3, 1].set_visible(False)

ax[2, 5].set_visible(False)
ax[3, 5].set_visible(False)

ax[4, 1].set_visible(False)
ax[4, 5].set_visible(False)

# compute resulting ILDs and ITDs for frontal and 90° right-rotated array 
# (= 90° (left) rotation of the soundfield)
for ROTATION, row_ind in zip([0, 90], [0, 3]):
    NFFT = 1024
    fvec = np.arange(NFFT // 2 + 1) / NFFT * fs

    # define a grid of source direction on the equator, spacing: 5°
    source_dir = np.arange(-180, 180, 5)

    # sh coeffs for reference source directions
    Y_norot = spaudiopy.sph.sh_matrix(5, (source_dir) / 180 * np.pi, np.pi / 2,
                                      'real')
    # sh coeffs for rotated source directions
    Y = spaudiopy.sph.sh_matrix(25, (source_dir + ROTATION) / 180 * np.pi,
                                np.pi / 2, 'real')
    
    # impulse responses / steering vectors for (rotated) source directions
    D = []
    for i in range(Y.shape[0]):
        D.append(np.sum(Y[None, i, :, None] * array_sh[:, :, :], -2))
    D = np.stack(D, -2)
    
    Df = np.fft.rfft(D, NFFT, 0)
    steer_vec_sh = np.fft.rfft(array_sh, NFFT, 0)
    diff_cov = 1 / (np.pi * 4) * np.sum(
        steer_vec_sh[..., None] * np.conj(steer_vec_sh[..., None, :]), axis=1)

    rotmat1 = calculate_rotation_matrix(1, np.array([ROTATION]), np.array([0]),
                                        np.array([0]))
    r1i = np.linalg.inv(rotmat1)

    # load HRIR
    hrir, fs_hrirs = soundfile.read(os.path.join(
        parent, 'ku100_magls_sh_hrir/irsOrd5.wav'))
    hrir = signal.resample_poly(hrir, fs, fs_hrirs, axis=0) * fs_hrirs / fs
    hrir_delay = np.argmax(np.abs(hrir[:, 0]))
    nm = np.arange(0, hrir.shape[1])
    n = np.floor(np.sqrt(nm))
    m = nm - n**2 - n
    mult = np.zeros(hrir.shape[1])
    mult[m >= 0] = 1
    mult[m < 0] = -1
    right = hrir * mult[None, :]
    left = hrir
    left_diff = np.sqrt(1 / (np.pi * 4) *
                        np.sum(np.abs(np.fft.rfft(left, NFFT, 0))**2, axis=1))
    right_diff = np.sqrt(
        1 / (np.pi * 4) *
        np.sum(np.abs(np.fft.rfft(right, NFFT, 0))**2, axis=1))
    # decode
    Hl = (Y_norot @ left.T).T
    Hr = (Y_norot @ right.T).T
    Hlf = np.fft.rfft(Hl, n=NFFT, axis=0)
    Hrf = np.fft.rfft(Hr, n=NFFT, axis=0)

    # load first-order HRIRs
    hrir, fs_hrirs = soundfile.read(os.path.join(parent, 'ku100_magls_sh_hrir/irsOrd1.wav'))
    hrir = signal.resample_poly(hrir, fs, fs_hrirs, axis=0) * fs_hrirs / fs
    hrir_delay = np.argmax(np.abs(hrir[:, 0]))
    nm = np.arange(0, hrir.shape[1])
    n = np.floor(np.sqrt(nm))
    m = nm - n**2 - n
    mult = np.zeros(hrir.shape[1])
    mult[m >= 0] = 1
    mult[m < 0] = -1
    right1 = hrir * mult[None, :]
    left1 = hrir
    right1_f = np.fft.rfft(right1, NFFT, 0)
    left1_f = np.fft.rfft(left1, NFFT, 0)

    ##-----BFBR-----
    bfbr_filter = np.load(os.path.join(dirpath, 'filters/bfbr_rot_%d_f.npy') % ROTATION)
    bfbr_directions = np.linspace(0, 360, 5, endpoint=False) / 180 * np.pi
    y_bfbr = spaudiopy.sph.sh_matrix(5, bfbr_directions, np.pi / 2,
                                          'real')
    Hl_bfbr_indiv_f = np.fft.rfft(y_bfbr @ left.T, NFFT, 1).T
    Hr_bfbr_indiv_f = np.fft.rfft(y_bfbr @ right.T, NFFT, 1).T
    aout = ((
        bfbr_filter @ Df.transpose(1, 0, 2)[..., None])[...,
                                                             0]).transpose(
                                                                 1, 0, 2)
    diff_cov_aout = (bfbr_filter[0, ...] @ diff_cov @ np.conj(
        bfbr_filter[0, ...].transpose(0, 2, 1)))
    bfbr_diff_l = np.real(Hl_bfbr_indiv_f[:, None, :] @ diff_cov_aout
                               @ np.conj(Hl_bfbr_indiv_f[:, :, None]))
    bfbr_diff_r = np.real(Hr_bfbr_indiv_f[:, None, :] @ diff_cov_aout
                               @ np.conj(Hr_bfbr_indiv_f[:, :, None]))
    Hl_bfbr_f = np.sum(aout * Hl_bfbr_indiv_f[:, None, :], -1)
    Hr_bfbr_f = np.sum(aout * Hr_bfbr_indiv_f[:, None, :], -1)

    ##-----FOA-----
    foa_encoder_f = np.load(os.path.join(dirpath, 'filters/foa_encoder_f.npy'))
    foa_encoded_f = (foa_encoder_f[:, None, :, :] @ Df[..., None])[..., 0]

    diff_cov_foa = (r1i @ foa_encoder_f @ diff_cov @ np.conj(
        foa_encoder_f.transpose(0, 2, 1)) @ r1i.transpose(0, 2, 1))

    left1_f = np.fft.rfft(left1, 1024, 0)
    right1_f = np.fft.rfft(right1, 1024, 0)

    foa_diff_l = np.real(
        left1_f[:, None, :] @ diff_cov_foa @ np.conj(left1_f[:, :, None]))
    foa_diff_r = np.real(
        right1_f[:, None, :] @ diff_cov_foa @ np.conj(right1_f[:, :, None]))

    backrot = (r1i[None, :, :] @ foa_encoded_f[..., None])[..., 0]

    Hl_foa_f = np.sum(left1_f[:, None, :] * backrot, -1)
    Hr_foa_f = np.sum(right1_f[:, None, :] * backrot, -1)

    ## -----End-to-end MAGLS-----
    # emagls_filter = np.load(os.path.join(parent, 'compute_emagls_filters/filters_fd.npy'))
    # target_xyz = np.array(
    #     [np.cos(ROTATION / 180 * np.pi), -np.sin(ROTATION / 180 * np.pi), 0])
    # target_roll = 0
    # emagls_xyz = np.load(os.path.join(parent, 'compute_emagls_filters/xyz.npy'))
    # emagls_roll = np.load(os.path.join(parent, 'compute_emagls_filters/roll.npy'))

    # xyz_ind = np.argmax(np.mean(emagls_xyz * target_xyz[None, :], axis=-1))
    # roll_ind = np.argmin(np.abs(emagls_roll - target_roll))+

    # rot_filter = emagls_filter[xyz_ind, roll_ind, :, :, :]
    # rot_filter *= np.exp(-1j * 2 * np.pi * fvec * 139 / 48000)
    rot_filter = np.load(os.path.join(parent, 
        'compute_emagls_filters/emagls_32kHz_dft_aligned_ypr_%d_0_0.npy' % ROTATION))

    Hl_emagls_f = np.sum(rot_filter[:, 0, None, :].T * Df, -1)
    Hr_emagls_f = np.sum(rot_filter[:, 1, None, :].T * Df, -1)
    emagls_diff_l = np.real(rot_filter[:, 0, None, :].T @ diff_cov @ np.conj(
        rot_filter[None, :, 0, :].T))
    emagls_diff_r = np.real(rot_filter[:, 1, None, :].T @ diff_cov @ np.conj(
        rot_filter[None, :, 1, :].T))

    ## -----LCMV + End-to-end MAGLS----
    bf_filter = []
    res_filter = []
    for a in source_dir:
        bf_filter.append(
            np.load(os.path.join(dirpath, 'filters/maxdir_azi_%d_rot_%d_f.npy') %
                    (np.mod(a, 360), ROTATION))[0, ...])
        res_filter.append(
            np.load(os.path.join(dirpath, 'filters/resfilt_azi_%d_rot_%d_f.npy') %
                    (np.mod(a, 360), ROTATION))[0, ...])

    bf_filter = np.stack(bf_filter, 1)
    res_filter = np.stack(res_filter, 1)

    bf_out = bf_filter @ Df[..., None]
    res_out = res_filter @ Df[..., None]

    bf_diff_resp = np.real(bf_filter @ diff_cov[:, None, :, :] @ np.conj(
        np.transpose(bf_filter, [0, 1, 3, 2])))
    bf_diff_resp_l = np.real(Hlf * bf_diff_resp[:, :, 0, 0] * np.conj(Hlf))
    bf_diff_resp_r = np.real(Hrf * bf_diff_resp[:, :, 0, 0] * np.conj(Hrf))

    res_diff_cov = res_filter @ diff_cov[:, None, :, :] @ np.conj(
        np.transpose(res_filter, [0, 1, 3, 2]))
    res_diff_l = np.real(
        rot_filter[:, 0, None, None, :].T @ res_diff_cov @ np.conj(
            rot_filter[:, 0, None, None, :].T.transpose(0, 1, 3, 2)))
    res_diff_r = np.real(
        rot_filter[:, 0, None, None, :].T @ res_diff_cov @ np.conj(
            rot_filter[:, 0, None, None, :].T.transpose(0, 1, 3, 2)))

    bf_diff_l = bf_diff_resp_l + res_diff_l[:, :, 0, 0]
    bf_diff_r = bf_diff_resp_r + res_diff_r[:, :, 0, 0]

    bf_out_l = bf_out[:, :, 0, 0] * Hlf
    bf_out_r = bf_out[:, :, 0, 0] * Hrf

    res_out_l = np.sum(rot_filter[:, 0, None, :].T * res_out[..., 0], -1)
    res_out_r = np.sum(rot_filter[:, 1, None, :].T * res_out[..., 0], -1)

    Hl_bf_res_f = bf_out_l + res_out_l
    Hr_bf_res_f = bf_out_r + res_out_r

    ## -----fixed LCMV + End-to-end MAGLS----
    bf_filter = []
    res_filter = []
    bf_filter.append(
        np.load(os.path.join(dirpath, 'filters/maxdir_azi_45_rot_%d_f.npy') %
                (ROTATION))[0, ...])
    res_filter.append(
        np.load(os.path.join(dirpath, 'filters/resfilt_azi_45_rot_%d_f.npy') %
                (ROTATION))[0, ...])
    bf_filter = np.stack(bf_filter, 1)
    res_filter = np.stack(res_filter, 1)

    bf_out = bf_filter @ Df[..., None]
    res_out = res_filter @ Df[..., None]

    bf_out_l = bf_out[:, :, 0, 0] * Hlf
    bf_out_r = bf_out[:, :, 0, 0] * Hrf

    bf_diff_resp = np.real(bf_filter @ diff_cov[:, None, :, :] @ np.conj(
        np.transpose(bf_filter, [0, 1, 3, 2])))
    bf_diff_resp_l = np.real(Hlf * bf_diff_resp[:, :, 0, 0] * np.conj(Hlf))
    bf_diff_resp_r = np.real(Hrf * bf_diff_resp[:, :, 0, 0] * np.conj(Hrf))

    res_diff_cov = res_filter @ diff_cov[:, None, :, :] @ np.conj(
        np.transpose(res_filter, [0, 1, 3, 2]))
    res_diff_l = np.real(
        rot_filter[:, 0, None, None, :].T @ res_diff_cov @ np.conj(
            rot_filter[:, 0, None, None, :].T.transpose(0, 1, 3, 2)))
    res_diff_r = np.real(
        rot_filter[:, 0, None, None, :].T @ res_diff_cov @ np.conj(
            rot_filter[:, 0, None, None, :].T.transpose(0, 1, 3, 2)))

    bf0_diff_l = bf_diff_resp_l + res_diff_l[:, :, 0, 0]
    bf0_diff_r = bf_diff_resp_r + res_diff_r[:, :, 0, 0]

    res_out_l = np.sum(rot_filter[:, 0, None, :].T * res_out[..., 0], -1)
    res_out_r = np.sum(rot_filter[:, 1, None, :].T * res_out[..., 0], -1)

    Hl_bf0_res_f = bf_out_l + res_out_l
    Hr_bf0_res_f = bf_out_r + res_out_r

    ## evaluate ILDs
    ild_ref = (10 * np.log10(np.abs(np.abs(Hlf))**2) -
               10 * np.log10(np.abs(np.abs(Hrf))**2))
    ild_emagls2 = (10 * np.log10(np.abs(np.abs(Hl_emagls_f))**2) -
                   10 * np.log10(np.abs(np.abs(Hr_emagls_f))**2))
    ild_bfbr = (10 * np.log10(np.abs(np.abs(Hl_bfbr_f))**2) -
                     10 * np.log10(np.abs(np.abs(Hr_bfbr_f))**2))
    ild_foa = (10 * np.log10(np.abs(np.abs(Hl_foa_f))**2) -
               10 * np.log10(np.abs(np.abs(Hr_foa_f))**2))
    ild_bf_res = (10 * np.log10(np.abs(np.abs(Hl_bf_res_f))**2) -
                  10 * np.log10(np.abs(np.abs(Hr_bf_res_f))**2))
    ild_bf0_res = (10 * np.log10(np.abs(np.abs(Hl_bf0_res_f))**2) -
                   10 * np.log10(np.abs(np.abs(Hr_bf0_res_f))**2))

    ## evaluate ITDs
    CIRCSHIFT = NFFT//2
    itd_ref = (np.argmax(np.roll(
        np.fft.irfft(gfbmat[:, :, None] * Hlf[None, :, :] *
                     np.conj(gfbmat[:, :, None] * Hrf[None, :, :]),
                     axis=1), CIRCSHIFT, 1),
        axis=1) - CIRCSHIFT) / fs
    itd_emagls2 = (np.argmax(np.roll(
        np.fft.irfft(gfbmat[:, :, None] * Hl_emagls_f[None, :, :] *
                     np.conj(gfbmat[:, :, None] * Hr_emagls_f[None, :, :]),
                     axis=1), CIRCSHIFT, 1),
        axis=1) - CIRCSHIFT) / fs
    itd_bfbr = (np.argmax(np.roll(
        np.fft.irfft(gfbmat[:, :, None] * Hl_bfbr_f[None, :, :] *
                     np.conj(gfbmat[:, :, None] * Hr_bfbr_f[None, :, :]),
                     axis=1), CIRCSHIFT, 1),
        axis=1) - CIRCSHIFT) / fs
    itd_foa = (np.argmax(np.roll(
        np.fft.irfft(gfbmat[:, :, None] * Hl_foa_f[None, :, :] *
                     np.conj(gfbmat[:, :, None] * Hr_foa_f[None, :, :]),
                     axis=1), CIRCSHIFT, 1),
        axis=1) - CIRCSHIFT) / fs
    itd_bf_res = (np.argmax(np.roll(
        np.fft.irfft(gfbmat[:, :, None] * Hl_bf_res_f[None, :, :] *
                     np.conj(gfbmat[:, :, None] * Hr_bf_res_f[None, :, :]),
                     axis=1), CIRCSHIFT, 1),
        axis=1) - CIRCSHIFT) / fs
    itd_bf0_res = (np.argmax(np.roll(
        np.fft.irfft(gfbmat[:, :, None] * Hl_bf0_res_f[None, :, :] *
                     np.conj(gfbmat[:, :, None] * Hr_bf0_res_f[None, :, :]),
                     axis=1), CIRCSHIFT, 1),
        axis=1) - CIRCSHIFT) / fs

    titles = [
        'reference', 'FOA', 'BFBR', 'MLS', 'BF+MLS:\nsteered at source',
        'BF+MLS:\nsteered at 45°'
    ]
    ilds = [
        ild_ref, ild_foa, ild_bfbr, ild_emagls2, ild_bf_res, ild_bf0_res
    ]
    itds = [
        itd_ref, itd_foa, itd_bfbr, itd_emagls2, itd_bf_res, itd_bf0_res
    ]
    indices = [0, 2, 3, 4, 6, 7]

    for ild, ind, title, itd in zip(ilds, indices, titles, itds):
        if row_ind == 0:
            ax[row_ind, ind].set_title(title)

        h = ax[row_ind, ind].pcolormesh(source_dir,
                                        fvec,
                                        ild,
                                        shading='gouraud',
                                        vmin=-PLOT_DB_RANGE,
                                        vmax=PLOT_DB_RANGE,
                                        cmap='RdBu')

        if ind == 0:
            ax[row_ind, ind].set_ylabel('frequency in Hz')

        if row_ind == 3:
            ax[row_ind + 1, ind].set_xlabel('azimuth in °')

        h2 = ax[row_ind + 1, ind].pcolormesh(source_dir,
                                             fvec_gfb,
                                             -itd * 1000,
                                             shading='gouraud',
                                             vmin=-1.3,
                                             vmax=1.3,
                                             cmap='RdBu')

        if ind == 0:
            ax[row_ind + 1, ind].set_ylabel('frequency in Hz')

        ax[row_ind, ind].set_ylim([200, 15000])
        ax[row_ind, ind].set_yscale('log')
        ax[row_ind, ind].set_xticks([90, 0, -90])

        ax[row_ind + 1, ind].set_ylim([100, 1500])
        ax[row_ind + 1, ind].set_yscale('log')
        ax[row_ind + 1, ind].set_xticks([90, 0, -90])

        if row_ind == 0:
            ax[row_ind, ind].title.set_size(10)

        if ind != 0:
            ax[row_ind, ind].set_yticks([])
            ax[row_ind + 1, ind].set_yticks([])

    cbar = fig.colorbar(h)
    cbar.set_label('ILD in dB', rotation=90)

    cbar = fig.colorbar(h2, shrink=1.0, aspect=10)
    cbar.set_label('ITD in ms', rotation=90)

ax[3, 0].invert_xaxis()
ax[0, 7].set_ylabel('forward array orientation',
                    y=0.22,
                    labelpad=60,
                    fontsize=13)
ax[0, 7].yaxis.set_label_position('right')
ax[3, 7].set_ylabel('90° right-rotated array',
                    y=0.22,
                    labelpad=60,
                    fontsize=13)
ax[3, 7].yaxis.set_label_position('right')

plt.savefig(os.path.join(parent, 'figures/ild_itd.png'), dpi=300, bbox_inches='tight')
plt.show(block=True)
print('Done')
