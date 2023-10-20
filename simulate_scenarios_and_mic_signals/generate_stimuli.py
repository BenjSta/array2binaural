import numpy as np
import soundfile
import scipy.signal as signal
from utils.room_simulation import simulate_simple_room
from spaudiopy.utils import cart2sph
import spaudiopy


def compose_ambisonic_signal(source_sigs, souce_dirs, diffuse_ambi_signal,
                             order):
    '''
    Puts together an ambisonic signal

    source_sigs: ndarray (SxL)
    source_dirs: ndarray (Sx2)
    diffuse_ambi_signal: diffuse part of the sig, ndarray ((order+1)^2, L)
    order: ambisonics order: int

    returns: ambisonics signal: ndarray ((order+1)^2, L)
    '''

    sh_mat = spaudiopy.sph.sh_matrix(order, souce_dirs[0, :], souce_dirs[1, :],
                                     'real')

    return (source_sigs[:, None, :] @ sh_mat)[:, 0, :] + diffuse_ambi_signal

def generate_scenario(source_signals,
                   out_path_name_prefix,
                   directions,
                   room_dim,
                   rt60,
                   drr_db,
                   receiver_pos,
                   normalise_db,
                   fs,
                   order):
    '''
    generate a trial with binaural and ambisonic stimuli

    source_signals: numpy array (SxL)
    target_directory: where to save
    directions: numpy array (Sx2)
    room_dim: None or numpy array (3,)
    rt60: reverberation time: float (ignored if room_dim is None)
    drr_db: direct-to-reverberant ratio: float (ignored if room_dim is None)
    receiver_pos: receiver position in the room numpy array (3,) 
        (ignored if room_dim is None)
    normalise_db: target peak value for omnidirectional channel float
    fs: sampling rate: float
    order: ambisonics order: int
    '''
    if room_dim is not None:
        # room simulation
        C = 340
        MAXDELAY_DISCRETE = np.max(room_dim) * (ROOM_SIM_MIN_IMAGE_ORDER +
                                                1) / C
        maxdelay_discrete_samples = int(MAXDELAY_DISCRETE * fs)
        directions, delays, amplitudes, is_distinct, diffuse_envelope = \
            simulate_simple_room(rt60, drr_db, room_dim, receiver_pos,
            directions, MAXDELAY_DISCRETE, fs, C)

        all_discrete_signals = []
        all_is_distinct = []
        all_discrete_directions = []
        diffuse_sh = np.zeros((source_signals.shape[0], (order + 1)**2))

        # iterate over all sources
        for i in range(source_signals.shape[1]):
            for j in range(directions[i].shape[0]):
                d = delays[i][j]
                s = amplitudes[i][j] * np.pad(source_signals[:, i],
                                              ((d, 0), ))[:-d]
                all_is_distinct.append(is_distinct[i][j])
                all_discrete_signals.append(s)
                all_discrete_directions.append(directions[i][j, :])

            diffuse_ir = np.random.randn(diffuse_envelope.shape[0],
                                         (order + 1)**2)
            diffuse_ir = diffuse_ir * diffuse_envelope[:, None]
            diffuse_sh += 1 / np.sqrt(4 * np.pi) * np.pad(
                signal.oaconvolve(
                    source_signals[:, i, None], diffuse_ir, axes=0),
                ((maxdelay_discrete_samples, 0),
                 (0, 0)))[:source_signals.shape[0], :]

        # stack all direct sounds and reflections of all sources together
        all_discrete_directions = np.stack(all_discrete_directions, axis=-1)
        all_discrete_directions_azi, all_discrete_directions_zen, _ = \
            cart2sph(all_discrete_directions[0, :],
            all_discrete_directions[1, :], all_discrete_directions[2, :])
        all_discrete_directions = np.stack(
            [all_discrete_directions_azi, all_discrete_directions_zen], axis=0)

        all_discrete_signals = np.stack(all_discrete_signals, axis=-1)
        all_is_distinct = np.stack(all_is_distinct, axis=-1)

    else:
        all_discrete_signals = source_signals
        all_is_distinct = np.stack([True] * all_discrete_signals.shape[1])
        all_discrete_directions = directions
        diffuse_sh = np.zeros((source_signals.shape[0], (order + 1)**2))

    hoa = compose_ambisonic_signal(all_discrete_signals,
                                        all_discrete_directions, diffuse_sh,
                                        order)

    w_max = np.max(np.abs(hoa[:, 0]))
    normalization = 10**(normalise_db / 20) / w_max
    hoa *= normalization

    soundfile.write(out_path_name_prefix + '_ref_amb.wav',
                    hoa,
                    fs,
                    subtype='PCM_24')
  


if __name__ == '__main__':
    FS = 44100
    ROOM_SIM_MIN_IMAGE_ORDER = 2
    AMB_ORDER = 25

    main_room = np.array([7, 6, 3.5])
    listener_pos = np.array([3.5, 4, 1.5])

    for reverb in ['anech', 'strongrev']:

        if reverb == 'anech':
            room = None
            drr = np.nan
            t60 = np.nan
        else:
            room = main_room
            if reverb == 'medrev':
                drr = 0
                t60 = 0.3
            elif reverb == 'strongrev':
                drr = -6
                t60 = 0.6

        for scenario in [
                'two_speakers_opposite', 'two_speakers_close', 'string_quartet',
        ]:
            name = scenario + '_' + reverb
            normalise_db = -12
            

            if scenario == 'string_quartet':
                violin1, _ = soundfile.read(
                    'simulate_scenarios_and_mic_signals/source_audio/gomes_string_quartet/Mov1_Violin1_Haydn_StringQuartet_op76_n1_cut.wav'
                )
                violin2, _ = soundfile.read(
                    'simulate_scenarios_and_mic_signals/source_audio/gomes_string_quartet/Mov1_Violin2_Haydn_StringQuartet_op76_n1_cut.wav'
                )
                viola, _ = soundfile.read(
                    'simulate_scenarios_and_mic_signals/source_audio/gomes_string_quartet/Mov1_Viola_Haydn_StringQuartet_op76_n1_cut.wav'
                )
                cello, _ = soundfile.read(
                    'simulate_scenarios_and_mic_signals/source_audio/gomes_string_quartet/Mov1_Cello_Haydn_StringQuartet_op76_n1_cut.wav'
                )
                signals = np.stack([violin1, violin2, viola, cello], -1)
                directions = np.pi / 180 * np.array([[-90, 90], [-30, 90],
                                                     [30, 90], [90, 90]]).T


            elif scenario == 'two_speakers_close':
                s1, _ = soundfile.read('simulate_scenarios_and_mic_signals/source_audio/ebu_sqam/50_cut.wav')
                s2, _ = soundfile.read('simulate_scenarios_and_mic_signals/source_audio/ebu_sqam/49_cut.wav')
                signals = np.stack([s1, s2], -1)
                directions = np.pi / 180 * np.array([[-30, 90], [0, 90]]).T


            
            elif scenario == 'two_speakers_opposite':
                s1, _ = soundfile.read('simulate_scenarios_and_mic_signals/source_audio/ebu_sqam/52_cut.wav')
                s2, _ = soundfile.read('simulate_scenarios_and_mic_signals/source_audio/ebu_sqam/53_cut.wav')
            
                signals = np.stack([s1, s2], -1)
                directions = np.pi / 180 * np.array([[-135, 90],
                                                     [45, 70]]).T

            generate_scenario(signals, 'simulate_scenarios_and_mic_signals/audio_o25/' + name, directions, room,
                           t60, drr, listener_pos, normalise_db, FS, AMB_ORDER)