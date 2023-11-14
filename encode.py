#%%
import numpy as np
import spaudiopy
import librosa
import soundfile
import scipy.signal as signal
from ambisonics import calculate_rotation_matrix
import os

MIC_AMBI_PATH = 'Easycom_array_32000Hz_o25_22samps_delay.npy'
fs = 32000
array_sh_delay = 22  # samples
array_sh = np.load(MIC_AMBI_PATH)

WINLEN = 640
HOPSIZE = 320
FILTLEN = 512
N_FFT = WINLEN + FILTLEN

def overlap_add(frames, hop_length):
    frames_shape = frames.shape
    frames_channels_shape = frames_shape[:-2]
    framelen = frames.shape[-2]

    y = np.zeros(frames_channels_shape + tuple([(frames.shape[-1] - 1) * hop_length + frames.shape[-2],]))
    
   
    for frame in range(frames.shape[-1]):
        sample = frame * hop_length
        y[..., sample : (sample + framelen)] += frames[..., :framelen, frame]
    return y


NFFT_DESIGN = 1024
d = np.fft.rfft(array_sh, NFFT_DESIGN, axis=0)
# first order encoder
fvec = np.arange(d.shape[0]) / NFFT_DESIGN * fs
d = d * np.exp(1j * 2 * np.pi * fvec * array_sh_delay / fs)[:, None, None]

# foa encoder
dfoa = d[:, :4, :].transpose(0, 2, 1)
dcov = np.conj(dfoa[:, :, [0, 1, 3]].transpose(0, 2, 1)) @ dfoa[:, :,
                                                                [0, 1, 3]]
p = np.linalg.eigvalsh(dcov)[:, -1]
fdependent_reg = 0.001 * p
dcov_reg = dcov + (fdependent_reg[:, None, None]) * np.eye(3)[None, :, :]
e = np.linalg.inv(dcov_reg) @ np.conj(dfoa[:, :, [0, 1, 3]].transpose(0, 2, 1))
ewxy = np.zeros((e.shape[0], 4, e.shape[2]), dtype='complex128')
ewxy[:, [0, 1, 3], :] = e
e = ewxy
e = e * 2 / np.linalg.norm(e @ d.transpose(0, 2, 1), 'fro',
                           axis=(1, 2))[:, None, None]
foa_enc = e

foa_enc_td = np.fft.irfft(foa_enc, axis=0)
foa_enc_td = np.roll(foa_enc_td, FILTLEN//2, axis=0)
foa_enc_td = foa_enc_td[:FILTLEN, :, :] * signal.windows.tukey(FILTLEN, 0.1)[:, None, None]
#%%
diff_cov_matrix = np.sum(d[..., None] * np.conj(d[..., None, :]), -3)
e = np.linalg.eigvalsh(diff_cov_matrix)
inv_diff_cov_matrix = np.linalg.inv(0.001 * e[:, -1, None, None] *
                                    np.eye(5)[None, :, :] + diff_cov_matrix)

#%%


for recording_rot in ['static', 'dynamic']:
    # is the same for every static/dynamic scenario, so use "string quartet anech" here
    yaw = np.load(
        os.path.join(
            'simulate_scenarios_and_mic_signals/rendered_mic',
            'string_quartet_anech' + '_' + recording_rot + '_yaw.npy'))
    pitch = np.zeros_like(yaw)
    roll = np.zeros_like(yaw)

    rotmat1 = calculate_rotation_matrix(1, yaw, pitch, roll)
    rotmat_xyz = rotmat1[..., [3, 1, 2], :]
    rotmat_xyz = rotmat_xyz[..., :, [3, 1, 2]]

    rotmat5 = calculate_rotation_matrix(5, yaw, pitch, roll)
    inv_rotmat1 = np.linalg.inv(rotmat1)
    inv_rotmat5 = np.linalg.inv(rotmat5)

    print('evaluating bfbr steering vectors...')
    directions_bfbr = np.pi / 180 * np.stack([np.linspace(0, 360, 5, endpoint=False), np.array(5 * [90])], axis=0)
    xyz_bfbr = np.stack(
        spaudiopy.utils.sph2cart(directions_bfbr[0, :], directions_bfbr[1, :]),
        -1)
    azi, colat, _ = spaudiopy.utils.cart2sph(xyz_bfbr[:, 0], xyz_bfbr[:, 1],
                                                xyz_bfbr[:, 2])
    Y_norot_bfbr = spaudiopy.sph.sh_matrix(5, azi, colat, 'real')
    # Frames x Freq x Sources x SH Components x Microphones
    
    xyz_frame_bfbr = (rotmat_xyz[:, None, :, :] @ xyz_bfbr[None, :, :, None])[..., 0]
    azi_bfbr, colat_bfbr, _ = spaudiopy.utils.cart2sph(xyz_frame_bfbr[..., 0].flatten(),
                                                xyz_frame_bfbr[..., 1].flatten(),
                                                xyz_frame_bfbr[..., 2].flatten())
    Y_bfbr = spaudiopy.sph.sh_matrix(25, azi_bfbr, colat_bfbr, 'real')
    Y_bfbr = Y_bfbr.reshape(xyz_frame_bfbr.shape[0], xyz_frame_bfbr.shape[1], -1)
    
    D_bfbr = []
    for i in range(Y_bfbr.shape[1]):
        D_bfbr.append(np.sum(Y_bfbr[:, None, i, :, None] * 
                            d[None, :, :, :], -2))
    D_bfbr = np.stack(D_bfbr, -2)
    # D_bfbr (Steering Vectors): Frames x Freq x Directions x Microphones
    print('Done.')
    #u, s, vh = np.linalg.svd(np.conj(D_bfbr).transpose(0, 1, 3, 2), full_matrices=False)
    W_bfbr = np.conj(inv_diff_cov_matrix[None, :, None, :, :] @ D_bfbr[..., None] / 
                          (np.conj(D_bfbr[..., None, :]) @ inv_diff_cov_matrix[None, :, None, :, :] @ D_bfbr[..., None]))[..., 0] #) / np.linalg.norm(D_bfbr, 2, -1, keepdims=True)**2#(1/np.sqrt(D_bfbr.shape[-2]) * u @ vh).transpose(0, 1, 3, 2)
    #W_bfbr_sum = np.sum(W_bfbr, -2)
    #diff_sensit = np.sqrt(W_bfbr[..., None, :] @ diff_cov_matrix[None, ...] @ np.conj(W_bfbr[..., None]))[..., 0, 0]
    #W_bfbr /=  diff_sensit[..., None, None] 
    W_bfbr_td = np.fft.irfft(W_bfbr, axis=1)
    W_bfbr_td = np.roll(W_bfbr_td, FILTLEN//2, axis=1)
    W_bfbr_td = W_bfbr_td[:, :FILTLEN, :, :] * signal.windows.tukey(FILTLEN, 0.1)[None, :, None, None]
    W_bfbr_fd = np.fft.rfft(W_bfbr_td, N_FFT, axis=1)
    


    for scenario in [
            'string_quartet', 'two_speakers_close', 'two_speakers_opposite'
    ]:
        if scenario == 'two_speakers_close':
            directions = np.pi / 180 * np.array([[-30, 90], [0, 90]]).T
            xyz = np.stack(
                spaudiopy.utils.sph2cart(directions[0, :], directions[1, :]),
                -1)

        elif scenario == 'string_quartet':
            directions = np.pi / 180 * np.array([[-90, 90], [-30, 90],
                                                 [30, 90], [90, 90]]).T
            xyz = np.stack(
                spaudiopy.utils.sph2cart(directions[0, :], directions[1, :]),
                -1)

        elif scenario == 'two_speakers_opposite':
            directions = np.pi / 180 * np.array([[-135, 90], [45, 70]]).T
            xyz = np.stack(
                spaudiopy.utils.sph2cart(directions[0, :], directions[1, :]),
                -1)

        xyz_frame = (rotmat_xyz[:, None, :, :] @ xyz[None, :, :, None])[..., 0]
        azi, colat, _ = spaudiopy.utils.cart2sph(xyz_frame[..., 0].flatten(),
                                                 xyz_frame[..., 1].flatten(),
                                                 xyz_frame[..., 2].flatten())
        Y = spaudiopy.sph.sh_matrix(25, azi, colat, 'real')
        Y = Y.reshape(xyz_frame.shape[0], xyz_frame.shape[1], -1)

        print('evaluating steering vectors...')
        # Frames x Freq x Sources x SH Components x Microphones
        D_full = []
        for i in range(Y.shape[1]):
            D_full.append(np.sum(Y[:, None, i, :, None] * 
                                 d[None, :, :, :], -2))
        D_full = np.stack(D_full, -2)
        # D_full (Steering Vectors): Frames x Freq x Sources x Microphones
        print('Done.')
        
        for reverb in ['anech', 'strongrev']:
            filepath = os.path.join(
                'simulate_scenarios_and_mic_signals/rendered_mic', scenario +
                '_' + reverb + '_' + recording_rot + '_mic_20dB_pad.wav')
            mic, fs = soundfile.read(filepath)
            assert fs == 32000

            mic_frames = librosa.util.frame(np.pad(mic, 
                ((HOPSIZE, HOPSIZE), (0,0))).T, 
                frame_length=WINLEN, 
                hop_length=HOPSIZE).T
            window=signal.windows.hann(WINLEN, sym=False)
            windowed_mic_frames = mic_frames * window[None, :, None]

            for discrete_beams in ['none', 'incomplete', 'complete']:
                if discrete_beams == 'incomplete':
                    if scenario == 'speech+noise':
                        xyz_lim = xyz[[0, 2], :]
                        D = D_full[..., [0, 2], :]
                    elif scenario == 'string_quartet':
                        xyz_lim = xyz[[1, 2], :]
                        D = D_full[..., [1, 2], :]
                    elif scenario == 'two_speakers_close':
                        xyz_lim = xyz[:1, :]
                        D = D_full[..., :1, :]
                    elif scenario == 'two_speakers_opposite':
                        xyz_lim = xyz[1:, :]
                        D = D_full[..., 1:, :]
                elif discrete_beams == 'none':
                    xyz_lim = xyz[[], :]
                    D = D_full[..., [], :]
                elif discrete_beams == 'complete':
                    xyz_lim = xyz
                    D = D_full


                azi, colat, _ = spaudiopy.utils.cart2sph(xyz_lim[:, 0], xyz_lim[:, 1],
                                                    xyz_lim[:, 2])
                Y_norot = spaudiopy.sph.sh_matrix(5, azi, colat, 'real')
                
                if discrete_beams != 'none':
                    lcmv_to_invert = (
                        np.conj(D) @ inv_diff_cov_matrix[None, :, :, :]
                        @ D.transpose(0, 1, 3, 2))

                    e = np.linalg.eigvalsh(lcmv_to_invert)
                    lcmv_to_invert = lcmv_to_invert + 0.1 * e[:, :, -1, None,
                                                            None] * np.eye(
                                                                lcmv_to_invert.
                                                                shape[-1]
                                                            )[None, None, :, :]
                    sepfilt = np.conj((inv_diff_cov_matrix[None, :, :, :] @ D.transpose(
                        0, 1, 3, 2) @ np.linalg.inv(lcmv_to_invert)).transpose(
                            0, 1, 3, 2))
                    sepfilt_td = np.fft.irfft(sepfilt, axis=1)
                    sepfilt_td = np.roll(sepfilt_td, FILTLEN//2, axis=1)
                    sepfilt_td = sepfilt_td[:, :FILTLEN, :, :] * signal.windows.tukey(FILTLEN, 0.1)[None, :, None, None]

                    resfilt = np.eye(sepfilt.shape[-1])[None, None, :, :] - D.transpose(0, 1, 3, 2) @ sepfilt
                    resfilt_td = np.fft.irfft(resfilt, axis=1)
                    resfilt_td = np.roll(resfilt_td, FILTLEN//2, axis=1)
                    resfilt_td = resfilt_td[:, :FILTLEN, :, :] * signal.windows.tukey(FILTLEN, 0.1)[None, :, None, None]
                    
                    y = np.sum(signal.fftconvolve(sepfilt_td, windowed_mic_frames[:, :, None, :], axes=1), axis=-1)
                    hoa = np.sum(Y_norot[None, None, :, :] * y[..., None], -2)
                    hoa_td = overlap_add(hoa.T, HOPSIZE)[..., HOPSIZE:HOPSIZE+mic.shape[0]].T
                    amb_mic = np.sum(signal.fftconvolve(resfilt_td, windowed_mic_frames[:, :, None, :], axes=1), axis=-1)
                    soundfile.write(os.path.join(
                        'encoded', scenario + '_' + reverb + '_' + recording_rot +
                        '_' + discrete_beams + '_hoa.wav'),
                                    hoa_td,
                                    fs,
                                    subtype='FLOAT') 
                else:
                    amb_mic = windowed_mic_frames
                    
                amb_mic_td = overlap_add(amb_mic.T, HOPSIZE)[:, HOPSIZE:HOPSIZE + mic.shape[0]].T

                soundfile.write(os.path.join(
                    'encoded', scenario + '_' + reverb + '_' + recording_rot +
                    '_' + discrete_beams + '_amb_mic.wav'),
                                amb_mic_td,
                                fs,
                                subtype='FLOAT')

                for ambient_rendering in [
                        'foa', 'bfbr', 'emagls2'
                ]:
                    if ambient_rendering == 'foa':
                        foa = np.sum(signal.fftconvolve(foa_enc_td[None, :, :, :], amb_mic[..., None, :], axes=1), axis=-1)
                        foa = (inv_rotmat1[:, None, :, :] @ foa[..., None])[..., 0]
                        foa_td = overlap_add(foa.T, HOPSIZE)[:, HOPSIZE:HOPSIZE + mic.shape[0]].T
                        soundfile.write(os.path.join(
                            'encoded', scenario + '_' + reverb + '_' + recording_rot +
                            '_' + discrete_beams + '_' + ambient_rendering + '.wav'),
                                        foa_td,
                                        fs,
                                        subtype='FLOAT')
                    elif ambient_rendering == 'bfbr':
                        y_bfbr = np.sum(signal.fftconvolve(W_bfbr_td, amb_mic[..., None, :], axes=1), axis=-1)
                        enc_bfbr = np.sum(
                            Y_norot_bfbr[None, None, :, :] * y_bfbr[..., None], -2)
                        enc_bfbr_td = overlap_add(enc_bfbr.T, HOPSIZE)[:, HOPSIZE:HOPSIZE + mic.shape[0]].T
                        soundfile.write(os.path.join(
                            'encoded', scenario + '_' + reverb + '_' + recording_rot +
                            '_' + discrete_beams + '_' + ambient_rendering + '.wav'),
                                        enc_bfbr_td,
                                        fs,
                                        subtype='FLOAT')
                    elif ambient_rendering == 'emagls2':
                        pass

            if recording_rot == 'static': 
                ref, fs_ref = soundfile.read(
                    os.path.join('simulate_scenarios_and_mic_signals/audio_o25',
                                scenario + '_' + reverb + '_ref_amb.wav'))
                ref = ref[:, :36]
                if fs_ref != fs:
                    ref = signal.resample_poly(ref, fs, fs_ref)

                soundfile.write(os.path.join(
                    'encoded',
                    scenario + '_' + reverb + '_' + recording_rot + '_ref.wav'),
                                ref[:, :36],
                                fs,
                                subtype='FLOAT')
