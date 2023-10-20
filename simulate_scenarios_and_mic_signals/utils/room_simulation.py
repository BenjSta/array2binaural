import numpy as np
import pyroomacoustics as pra
import itertools


def sph2cart(r, phi, theta):
    x = np.cos(phi) * np.sin(theta) * r
    y = np.sin(phi) * np.sin(theta) * r
    z = np.cos(theta) * r
    return np.stack([x, y, z], -1)


def simulate_simple_room(rt60,
                         drr_db,
                         room_dim,
                         receiver_pos,
                         directions,
                         maxdelay_discrete,
                         fs,
                         c=340):
    '''
    Computes a simple room simulation with frequency independent
    reflections and diffuse reverberation

    rt60: the reverberation time
    drr_db: the direct-to-diffuse ratio
    room_dim: numpy array with shape (3,)
    receiver_pos: numpy array with shape (3,)
    directions: (Sx2) numpy array with directions,
        where S is the number of sources
    maxdelay_discrete: length of the part of the IR,
        that is modeled using discrete reflections
    fs : sampling rate
    c: speed of sound, defaults to 340m/s
    
    returns:
    tuple: (directions, delays, amplitudes, is_distinct, diffuse_predelay,
            diffuse_start variance)
    directions: length-S list of numpy arrays (Ix3), where I is the number of
        images (including direct sound, for each source)
    delays: length-S list of numpy arrays (I,) with 
        integer delays (in samples) of the discrete dirac-deltas in the IR
    amplitudes: length-S list of numpy arrays (I,) with 
        integer delays of discrete dirac-deltas in the IR
    is_distinct: length-S list of booleans, whether a reflection can be heard 
        out as a distinct echo
    diffuse_rms_envelope: the rms envelope of the diffuse part of the IR
        (shifted by maxdelay_discrete * fs)
    '''

    absorption_coeff, _ = pra.inverse_sabine(rt60, room_dim)
    surface = 2 * (room_dim[0] * room_dim[1] + room_dim[0] * room_dim[2] +
                   room_dim[1] * room_dim[2])

    # number of reflections to estimate reverberation gain
    DIFFUSE_VARIANCE_EST_NUM_REFL = 25
    absorbing_surface = absorption_coeff * surface
    distance_of_drr_1 = np.sqrt(absorbing_surface / (24 * np.log(10)))

    # compute radius from drr
    radius = np.sqrt(distance_of_drr_1**2 / 10**(drr_db / 10))
    print('Distance: %.2f' % radius)

    t = np.arange(int(fs * rt60)) / fs
    reverberation_rms = 10**(-60 / 20 * (t - maxdelay_discrete) / rt60)
    diffuse_rms_envelope = reverberation_rms[t > maxdelay_discrete]

    # compute the minimum required image source order for the first
    # maxdelay_discrete part of the impulse response
    # (see pyroomacoustics.inverse_sabine to better understand the next four
    # lines)
    R = []
    for l1, l2 in itertools.combinations(room_dim, 2):
        R.append(l1 * l2 / np.sqrt(l1**2 + l2**2))
    ism_order = int(np.ceil(c * maxdelay_discrete / np.min(R)))

    mag_absorption = 1 - np.sqrt(1 - absorption_coeff)
    room = pra.ShoeBox(room_dim,
                       fs,
                       absorption=mag_absorption,
                       max_order=ism_order)

    src_pos = receiver_pos[None, :] + \
        sph2cart(radius, directions[0, :], directions[1, :])
    for s in range(src_pos.shape[0]):
        room.add_source(src_pos[s, :])

    room.add_microphone(receiver_pos)
    room.image_source_model()

    directions = []
    delays = []
    amplitudes = []
    is_distinct_list = []
    for i in range(src_pos.shape[0]):
        images = room.sources[i].images
        orders = room.sources[i].orders
        direct_sound_index = np.argmax(orders == 0)
        direction_unnormalized = images.T - receiver_pos[None, :]
        dist = np.linalg.norm(direction_unnormalized, axis=-1)
        amplitude = 1 / dist * (1 - mag_absorption)**orders
        delay = np.round(fs * dist / c).astype('int64')
        rel_delay = (delay - delay[direct_sound_index]) / fs

        # echo masking threshold
        #threshold = amplitude[direct_sound_index] * 10**(
        #    (-0.25 * 1e3 * rel_delay) / 20)
        is_distinct = orders == 0#amplitude >= threshold

        # only use reflections that arrive before maxdelay_discrete
        is_distinct_list.append(is_distinct[delay < maxdelay_discrete * fs])
        directions.append((direction_unnormalized /
                           dist[:, None])[delay < maxdelay_discrete * fs, :])
        delays.append(delay[delay < maxdelay_discrete * fs])
        amplitudes.append(amplitude[delay < maxdelay_discrete * fs])

        if i == 0:
            # from the first source, estimate the gain of the late reverberation
            ord = np.argsort(delays[0])
            start = delays[0][ord[DIFFUSE_VARIANCE_EST_NUM_REFL - 1]]
            dur = maxdelay_discrete * fs - start
            a = amplitudes[0][ord[DIFFUSE_VARIANCE_EST_NUM_REFL:]]
            std = np.sqrt(np.sum(a**2) / dur)
            diffuse_rms_envelope *= std

    return (directions, delays, amplitudes, is_distinct_list,
            diffuse_rms_envelope)
