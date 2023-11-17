import numpy as np
import spaudiopy
import scipy.signal as signal
import sys
import os
dirpath = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(dirpath)
sys.path.append(parent)
from ambisonics import calculate_rotation_matrix

MIC_AMBI_PATH = os.path.join(parent, "Easycom_array_32000Hz_o25_22samps_delay.npy")
fs = 32000
array_sh_delay = 22
array_sh = np.load(MIC_AMBI_PATH)

WINLEN = 640
HOPSIZE = 320
FILTLEN = 512
N_FFT = WINLEN + FILTLEN


NFFT_DESIGN = 1024
d = np.fft.rfft(array_sh, NFFT_DESIGN, axis=0)
# first order encoder
fvec = np.arange(d.shape[0]) / NFFT_DESIGN * fs
d = d * np.exp(1j * 2 * np.pi * fvec * array_sh_delay / fs)[:, None, None]

# foa encoder
dfoa = d[:, :4, :].transpose(0, 2, 1)
dcov = np.conj(dfoa[:, :, [0, 1, 3]].transpose(
    0, 2, 1)) @ dfoa[:, :, [0, 1, 3]]
p = np.linalg.eigvalsh(dcov)[:, -1]
fdependent_reg = 0.001 * p
dcov_reg = dcov + (fdependent_reg[:, None, None]) * np.eye(3)[None, :, :]
e = np.linalg.inv(dcov_reg) @ np.conj(dfoa[:, :, [0, 1, 3]].transpose(0, 2, 1))
ewxy = np.zeros((e.shape[0], 4, e.shape[2]), dtype="complex128")
ewxy[:, [0, 1, 3], :] = e
e = ewxy
e = e * 2 / np.linalg.norm(e @ d.transpose(0, 2, 1),
                           "fro", axis=(1, 2))[:, None, None]
foa_enc = e

foa_enc_td = np.fft.irfft(foa_enc, axis=0)

foa_enc_td = np.roll(foa_enc_td, FILTLEN // 2, axis=0)
foa_enc_td = (
    foa_enc_td[:FILTLEN, :, :] *
    signal.windows.tukey(FILTLEN, 0.1)[:, None, None]
)


np.save(os.path.join(dirpath, 'filters/foa_encoder_f.npy'), foa_enc)
np.save(os.path.join(dirpath, 'filters/foa_encoder.npy'), foa_enc_td)


diff_cov_matrix = np.sum(d[..., None] * np.conj(d[..., None, :]), -3)
e = np.linalg.eigvalsh(diff_cov_matrix)
inv_diff_cov_matrix = np.linalg.inv(
    0.001 * e[:, -1, None, None] * np.eye(5)[None, :, :] + diff_cov_matrix
)


for rotation in [0, 90]:
    yaw = np.array([np.pi * rotation / 180])
    pitch = np.array([0])
    roll = np.array([0])

    rotmat1 = calculate_rotation_matrix(1, yaw, pitch, roll)
    rotmat_xyz = rotmat1[..., [3, 1, 2], :]
    rotmat_xyz = rotmat_xyz[..., :, [3, 1, 2]]

    rotmat5 = calculate_rotation_matrix(5, yaw, pitch, roll)
    inv_rotmat1 = np.linalg.inv(rotmat1)
    inv_rotmat5 = np.linalg.inv(rotmat5)

    print("evaluating bfbr steering vectors...")
    directions_bfbr = (
        np.pi
        / 180
        * np.stack([np.linspace(0, 360, 5, endpoint=False), np.array(5 * [90])], axis=0)
    )
    xyz_bfbr = np.stack(
        spaudiopy.utils.sph2cart(
            directions_bfbr[0, :], directions_bfbr[1, :]
        ),
        -1,
    )
    azi, colat, _ = spaudiopy.utils.cart2sph(
        xyz_bfbr[:, 0], xyz_bfbr[:, 1], xyz_bfbr[:, 2]
    )
    Y_norot_bfbr = spaudiopy.sph.sh_matrix(5, azi, colat, "real")
    # Frames x Freq x Sources x SH Components x Microphones

    xyz_frame_bfbr = (rotmat_xyz[:, None, :, :] @ xyz_bfbr[None, :, :, None])[
        ..., 0
    ]
    azi_bfbr, colat_bfbr, _ = spaudiopy.utils.cart2sph(
        xyz_frame_bfbr[..., 0].flatten(),
        xyz_frame_bfbr[..., 1].flatten(),
        xyz_frame_bfbr[..., 2].flatten(),
    )
    Y_bfbr = spaudiopy.sph.sh_matrix(25, azi_bfbr, colat_bfbr, "real")
    Y_bfbr = Y_bfbr.reshape(
        xyz_frame_bfbr.shape[0], xyz_frame_bfbr.shape[1], -1
    )

    D_bfbr = []
    for i in range(Y_bfbr.shape[1]):
        D_bfbr.append(
            np.sum(Y_bfbr[:, None, i, :, None] * d[None, :, :, :], -2)
        )
    D_bfbr = np.stack(D_bfbr, -2)
    # D_bfbr (Steering Vectors): Frames x Freq x Directions x Microphones
    print("Done.")
    # u, s, vh = np.linalg.svd(np.conj(D_bfbr).transpose(0, 1, 3, 2), full_matrices=False)
    W_bfbr = np.conj(
        inv_diff_cov_matrix[None, :, None, :, :]
        @ D_bfbr[..., None]
        / (
            np.conj(D_bfbr[..., None, :])
            @ inv_diff_cov_matrix[None, :, None, :, :]
            @ D_bfbr[..., None]
        )
    )[..., 0]
    W_bfbr_td = np.fft.irfft(W_bfbr, axis=1)
    W_bfbr_td = np.roll(W_bfbr_td, FILTLEN // 2, axis=1)
    W_bfbr_td = (
        W_bfbr_td[:, :FILTLEN, :, :]
        * signal.windows.tukey(FILTLEN, 0.1)[None, :, None, None]
    )

    np.save(os.path.join(dirpath, 'filters/bfbr_rot_%d_f.npy') % rotation, W_bfbr)
    np.save(os.path.join(dirpath, 'filters/bfbr_rot_%d.npy') % rotation, W_bfbr_td)

    for source_dir in np.arange(0, 360, 5):
        Y = spaudiopy.sph.sh_matrix(
            25, (source_dir + rotation) / 180 * np.pi, np.pi / 2, "real"
        )[None, :, :]

        print("evaluating steering vectors...")
        # Frames x Freq x Sources x SH Components x Microphones
        D = []
        for i in range(Y.shape[1]):
            D.append(np.sum(Y[:, None, i, :, None] * d[None, :, :, :], -2))
        D = np.stack(D, -2)
        # D_full (Steering Vectors): Frames x Freq x Sources x Microphones
        print("Done.")

        lcmv_to_invert = (
            np.conj(D) @ inv_diff_cov_matrix[None,
                                             :, :, :] @ D.transpose(0, 1, 3, 2)
        )

        e = np.linalg.eigvalsh(lcmv_to_invert)
        lcmv_to_invert = (
            lcmv_to_invert
            + 0.1
            * e[:, :, -1, None, None]
            * np.eye(lcmv_to_invert.shape[-1])[None, None, :, :]
        )
        sepfilt = np.conj(
            (
                inv_diff_cov_matrix[None, :, :, :]
                @ D.transpose(0, 1, 3, 2)
                @ np.linalg.inv(lcmv_to_invert)
            ).transpose(0, 1, 3, 2)
        )
        sepfilt_td = np.fft.irfft(sepfilt, axis=1)
        sepfilt_td = np.roll(sepfilt_td, FILTLEN // 2, axis=1)
        sepfilt_td = (
            sepfilt_td[:, :FILTLEN, :, :]
            * signal.windows.tukey(FILTLEN, 0.1)[None, :, None, None]
        )
        np.save(
            os.path.join(dirpath, 'filters/maxdir_azi_%d_rot_%d_f.npy')
            % (source_dir, rotation),
            sepfilt,
        )
        np.save(
            os.path.join(dirpath, 'filters/maxdir_azi_%d_rot_%d.npy')
            % (source_dir, rotation),
            sepfilt_td,
        )

        resfilt = (
            np.eye(sepfilt.shape[-1])[None, None, :, :]
            - D.transpose(0, 1, 3, 2) @ sepfilt
        )
        resfilt_td = np.fft.irfft(resfilt, axis=1)
        resfilt_td = np.roll(resfilt_td, FILTLEN // 2, axis=1)
        resfilt_td = (
            resfilt_td[:, :FILTLEN, :, :]
            * signal.windows.tukey(FILTLEN, 0.1)[None, :, None, None]
        )
        np.save(
            os.path.join(dirpath, 'filters/resfilt_azi_%d_rot_%d_f.npy')
            % (source_dir, rotation),
            resfilt,
        )
        np.save(
            os.path.join(dirpath, 'filters/resfilt_azi_%d_rot_%d.npy')
            % (source_dir, rotation),
            resfilt_td,
        )
