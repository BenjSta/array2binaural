
import numpy as np
import soundfile
import scipy.signal as signal
import os
import sys
dirpath = os.path.dirname(os.path.abspath(__file__))
parentpath = os.path.dirname(dirpath)
sys.path.insert(0, parentpath)
from ambisonics import calculate_rotation_matrix
import librosa

MIC_AMBI_PATH = os.path.join(parentpath, 'Easycom_array_32000Hz_o25_22samps_delay.npy')
fs = 32000
array_sh_delay = 22
array_sh = np.load(MIC_AMBI_PATH)


AMB_FILEDIR = os.path.join(dirpath, 'audio_o25')
TARGET_MIC_FILEDIR = os.path.join(dirpath, 'rendered_mic')
ROTATION_HOPSIZE = 320
SENSOR_NOISE = -85

for reverb in ['anech', 'strongrev']:
    for scenario in [
            'two_speakers_close', 'string_quartet', 
            'two_speakers_opposite']:
        dirpath = os.path.join(AMB_FILEDIR, scenario + '_' + reverb + '_ref_amb.wav')
        amb, fs_amb = soundfile.read(dirpath)
        if fs_amb != fs:
            amb = signal.resample_poly(amb, fs, fs_amb)

        for recording_rot in ['static', 'dynamic']:
            ambi_mix_high_order_frames = librosa.util.frame(np.pad(amb, 
                ((ROTATION_HOPSIZE, ROTATION_HOPSIZE), (0,0))), 
                frame_length=2 * ROTATION_HOPSIZE, 
                hop_length=ROTATION_HOPSIZE, axis=0)

            window=signal.windows.hann(2 * ROTATION_HOPSIZE, sym=False)

            ambi_mix_high_order_frames = ambi_mix_high_order_frames * window[None, :, None]
            if recording_rot == 'static':
                yaw = np.zeros(ambi_mix_high_order_frames.shape[0])
            elif recording_rot == 'dynamic':
                yaw = np.linspace(-np.pi/2, np.pi/2, ambi_mix_high_order_frames.shape[0])
            
            pitch = np.zeros_like(yaw)
            roll = np.zeros_like(yaw)

            rotmat = calculate_rotation_matrix(25, yaw, pitch, roll)
            ambi_mix_high_order_rotated = (rotmat[:, None, :, :] @ ambi_mix_high_order_frames[..., None])[..., 0]
            
            ambi_mix_high_order_rotated_full = np.zeros(((ambi_mix_high_order_rotated.shape[0] - 1) * 
                ROTATION_HOPSIZE + ambi_mix_high_order_rotated.shape[1], ambi_mix_high_order_rotated.shape[2]))
            
            for frame in range(ambi_mix_high_order_rotated.shape[0]):
                sample = frame * ROTATION_HOPSIZE
                ambi_mix_high_order_rotated_full[sample : (sample + 2 * ROTATION_HOPSIZE), :] += \
                    ambi_mix_high_order_rotated[frame, :, :]
                        
            array_signals = 0.1 * np.sum(signal.oaconvolve(ambi_mix_high_order_rotated_full[..., None],
                array_sh, axes=0), axis=1)[ROTATION_HOPSIZE + array_sh_delay:
                                                ROTATION_HOPSIZE + 
                                                array_sh_delay+amb.shape[0], :].astype('float32')
            
            noise = np.random.randn(array_signals.shape[0], array_signals.shape[1])
            rms = np.sqrt(np.mean(noise**2))

            array_signals = array_signals + np.clip(10**(SENSOR_NOISE/20) / rms * noise, -1, 1)
            
            assert np.max(np.abs(array_signals)) < 1, 'clipping!'

            soundfile.write(os.path.join(TARGET_MIC_FILEDIR, 
                scenario + '_' + reverb + '_' + recording_rot + '_mic_20dB_pad.wav'), 
                array_signals, fs, subtype='PCM_24')
            
            np.save(os.path.join(TARGET_MIC_FILEDIR, 
                scenario + '_' + reverb + '_' + recording_rot + '_yaw.npy'), yaw)
