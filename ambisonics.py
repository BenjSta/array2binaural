import numpy as np  
import numpy.matlib as ml
import scipy.special as spec
from scipy.spatial.transform import Rotation


"""
Ambisonics utility functions implemented by Stefan Riedel and Franz Zotter, IEM Graz, 2020.
"""


def sh_xyz(nmax,ux,uy,uz):
    """Returns sh-matrix, evaluated at ux,uy,uz up to order nmax.

    Args:
        nmax (int): max SH order to be evaluated
        ux (ndarray): x coordinates
        uy (ndarray): y coordinates
        uz (ndarray): z coordinates

    Returns:
        ndarray: SH matrix [len(ux) , (nmax+1)**2]
    """

    nmax = int(nmax)
    Y=np.zeros((ux.size,(nmax+1)**2))
    Y[:,0]=np.sqrt(1/(4*np.pi))
    if(nmax == 0):
        return Y
    Y[:,2]=np.sqrt(3/(4*np.pi))*uz.reshape(uz.size)
    for n in range(1,nmax):
        Y[:,(n+1)*(n+2)] = -np.sqrt((2*n+3)/(2*n-1))*n/(n+1) * Y[:,(n-1)*n] +\
                   np.sqrt((2*n+1)*(2*n+3))/(n+1) * uz.flat[:] * Y[:,n*(n+1)]
    
    for i in range(uz.size):
        for n in range(0,nmax):
            for m in range(n+1):
                if m==0:
                    Y[i,(n+1)*(n+2)+(m+1)*np.array((1,-1))] = np.sqrt(2*(2*n+3)*(2*n+1)/((n+m+1)*(n+m+2))) * \
                                Y[i,n*(n+1)] * \
                                np.array((ux.flat[i],uy.flat[i]))
                else:
                    Y[i,(n+1)*(n+2)+(m+1)*np.array((1,-1))] = np.sqrt((2*n+3)*(2*n+1)/((n+m+1)*(n+m+2))) * \
                                Y[i,n*(n+1)+m*np.array((1,-1))].dot(
                                  np.array(((ux.flat[i],uy.flat[i]),(-uy.flat[i],ux.flat[i]))))
                if (m+1<=n-1):
                    Y[i,(n+1)*(n+2)+(m+1)*np.array((1,-1))] += \
                         np.sqrt((2*n+3)*(n-m-1)*(n-m)/((2*n-1)*(n+m+1)*(n+m+2))) * \
                        Y[i,(n-1)*n+(m+1)*np.array((1,-1))]
    return Y

def sh_azi_zen(nmax, azi, zen):
    """Returns sh-matrix, evaluated at azi and zen (in radians) up to order nmax.

    Args:
        nmax (int): max SH order to be evaluated
        azi (ndarray): azimuth angles in radians
        zen (ndarray): zenith (not elevation) angles in radians

    Returns:
        ndarray: SH matrix [len(azi) , (nmax+1)**2]
    """

    xyz = sph2cart(azi, zen)
    return sh_xyz(nmax, xyz[0], xyz[1], xyz[2])

def sph2cart(azi, zen):
    """Convert spherical to cartesian coordinates.

    Args:
        azi (ndarray): azimuth angles in radians
        zen (ndarray): zenith angles in radians

    Returns:
        ndarray: cartesian coordinates array
    """

    if(azi.size != zen.size):
        print("Azimuth and Zenith angles don't match in size!")
    
    ux = np.sin(zen) * np.cos(azi)
    uy = np.sin(zen) * np.sin(azi)
    uz = np.cos(zen)
    
    xyz = np.array([ux,uy,uz])
    return xyz

def SN3D_to_N3D(N):
    """computing renorm weights for adapting format conventions:
    get rid of sqrt(2n+1) in Y
    Args
        N (int): SH order

    Returns:
        ndarray: renorm weights
    """

    [n,m] = sh_indices(N)
    weights =  np.sqrt(2*n +1) / np.sqrt(4*np.pi) 

    return weights

def N3D_to_SN3D(N):
    """computing renorm weights for adapting format conventions:
    get rid of sqrt(2n+1) in Y
    Args
        N (int): SH order

    Returns:
        ndarray: renorm weights
    """

    [n,m] = sh_indices(N)
    weights = np.sqrt(4*np.pi) /  np.sqrt(2*n +1)

    return weights

def sh_n2nm_vec(win):
    """expands from n vector of length N+1 to full nm vector of length (N+1)**2

    Args:
        win (ndarray): order window / weights 

    Returns:
        ndarray: order and degree window / weights
    """

    if(np.size(win) == 1):
        return win

    win = np.squeeze(win)

    N = win.size -1
    nm_win = np.zeros(((N+1)**2), win.dtype)

    for n in range(N+1):
        for m in range(-n,n+1):
            nm_win[n*(n+1) + m] = win[n]

    return nm_win

def sh_indices(nmax):
    """Get order n = [0,1,1,1,...] and degree m = [0,-1,0,1,...] indices up to order nmax.

    Args:
        nmax (int): max SH order up to indicies are returned

    Returns:
        ndarray: order n indices and degree m indices
    """

    k = np.arange(0, (nmax+1)**2)
    n = np.floor(np.sqrt(k))
    m = k - n**2 - n

    return [n,m]

def sh_decoder_mixo(N, mixo_idx, azi, zen):
    """Returns zero-padded pseudo-inverse Y_pinv with non-zero mixo idxs.

    Args:
        N (int): max SH order
        mixo_idx (ndarray): indices of mixed-order harmonics that should be extracted, starting from 0
        azi (ndarray): azimuth angles in radians to be evaluated
        zen (ndarray): zenith angles in radians to be evaluated

    Returns:
        ndarray: pseudo-inverse mixed-order decoder matrix
    """

    Y_ls = sh_azi_zen(N, azi, zen)
    Y_ls_mixo = Y_ls[:, mixo_idx]
    Ypinv_ls_mixo = np.linalg.pinv(Y_ls_mixo)
    Ypinv_ls = np.zeros(Y_ls.transpose().shape)
    Ypinv_ls[mixo_idx,:] = Ypinv_ls_mixo

    return Ypinv_ls

def renormalize(N):
    """computing renorm weights for adapting format conventions:
    get rid of sqrt(2n+1) in Y
    and change sign of sinusoids (1-2deltam)
    ---the 180deg rotation (-1)^m is IKO performance practice with ambiX
    encoder GUI

    Args:
        N (int): SH order

    Returns:
        ndarray: renorm weights
    """

    [n,m] = sh_indices(N)
    one_minus2deltam = np.ones(np.size(m))
    idx_negm = np.where(m<0)
    one_minus2deltam[idx_negm] = -1
    renormalize = (-1)**m * one_minus2deltam / np.sqrt(2*n +1)

    return renormalize

def renormalize_matrix(Y, N):
    """adapting format conventions:
    get rid of sqrt(2n+1) in Y
    and change sign of sinusoids (1-2deltam)
    ---the 180deg rotation (-1)^m is IKO performance practice with ambiX
    encoder GUI

    Args:
        Y (ndarray): matrix to renormnalize
        N (int): SH order

    Returns:
        ndarray: renormalized matrix
    """

    renorm = renormalize(N)

    if(np.shape(Y)[1] == (N+1)**2):
        Y = np.dot(Y, np.diag(1/renorm))
    else:
        Y = np.dot(Y.T, np.diag(1/renorm)).T
    
    return Y

def mixo_weights(mixo_idx):
    """computes corrected mixed-order max-rE weights.

    Args:
        mixo_idx (ndarray): vector of indices of mixed-order SH's, e.g. [0, 1,2,3 , 4,8]

    Returns:
        ndarray: mixed-order max-rE weights
    """

    N = (np.sqrt(mixo_idx[-1] + 1) - 1).astype(int)
    y = sh_azi_zen(N, np.array([0]), np.array([np.pi/2]))
    w = maxre_sph(N)
    win = sh_n2nm_vec(w)
    k = np.arange((N+1)**2)     # linear sh index: 0,1,...,(N+1)**2 - 1
    n = np.floor(np.sqrt(k))    # n index: 0,1,1,1,...,N
    m = k - n * (n+1)           # m index: 0,-1,0,1,...,N

    idx_mixo=np.zeros((N+1)**2)
    idx_mixo[mixo_idx]=1
    cm=np.zeros(N+1)
    c=np.zeros((N+1)**2)

    for mi in range(N+1):
        idx_m = (m==mi).astype(int)
        #y=np.squeeze(y)
        a1 = np.dot(y, np.dot(np.diag(win*idx_mixo*idx_m), y.transpose()))
        a2 = np.dot(y, np.dot(np.diag(win*idx_m), y.transpose()))
        cm[mi] = a2/a1
        idx_abs_m = np.nonzero((np.abs(m)==mi))
        #idx_abs_m = np.nonzero(idx_abs_m)
        c[idx_abs_m] = np.squeeze(a2/a1)

    # apply correction vector c
    win=c*win
    #return only mixed-order entries, and additionally zero-padded version
    zero_pad = np.zeros((N+1)**2)
    zero_pad[mixo_idx] = 1
    win_padded = win * zero_pad
    return [win[mixo_idx] , win_padded]

def maxre_sph(N):
    """max-rE weights (3D / spherical).

    Args:
        N (int): SH order

    Returns:
        ndarray: (N+1) max-rE weights
    """

    thetaE = 137.9 / (N+1.52)
    rE = np.cos(thetaE / 180 * np.pi)
    win = legendre_u(N, rE)
    
    return win

def legendre_u(nmax, costheta):
    """returns the values of the unassociated legendre 
    polynomials up to order nmax at values costheta.
    This is a recursive implementation after Franz Zotter, IEM Graz
    Stefan Riedel, IEM Graz, 2020

    Args:
        nmax (int): max SH order
        costheta (ndarray): evaluation points

    Returns:
        ndarray: evaluated legendre polynomials [len(costheta), nmax+1]
    """

    if isinstance(costheta, (list, tuple, np.ndarray)):
        P = np.zeros((len(costheta),nmax+1))
    else:
        P = np.zeros((1,nmax+1))

    # Zeroth order polynomial is constant
    P[:,0] = 1

    # First oder polynomial is linear
    if nmax > 0:
        P[:,1] = costheta

    for n in range (1,nmax):
        P[:,n+1] = ((2*n+1) * np.multiply(costheta, P[:, n]) - n*P[:, n-1] ) / (n+1)

    return np.squeeze(P)
    # is equal too.. 
    #if P.shape[0] == 1:
       # return np.squeeze(P)
    #else:
       # return P  
    # as np.squeeze does nothing to true multidimensional arrays   

def farfield_extrapolation_filters(N, r0, fs, Nfft):
    """Extrapolate pressure in SH domain at r0 to far-field pressure by means of these filters.

    Args:
        N (int): SH order
        r0 (float): start radius r0 to far-field 
        fs (float): sampling rate
        Nfft (int): FFT size

    Returns:
        ndarray: extrapolation filters [Nfft/2+1, N+1]
    """

    R = r0
    c = 343
    f = np.linspace(0,fs/2, int(Nfft/2+1))
    f[0] = f[1] / 4
    k = 2*np.pi*f / c
    
    hn = np.zeros((len(f), N+1), dtype=complex)
    H = np.zeros((len(f), N+1), dtype=complex)

    for n in range(N+1):
        hn[:,n] = (1 / k) * (1j**(n+1))     # far-field radial tern

    h2 = sph_hankel2(k*R, N) * ml.repmat(np.exp(1j*k*R), N+1, 1).transpose() 
    H = (hn / h2) / R

    H[0, 1:] = 0

    h = np.fft.irfft(H, axis=0)

    return h

def cap_window(alpha,N):
    """the coefficients for a sphere cap window function,
    the sphere cap filter is calculated from:
    ratio between azimuthally ("rectular") sphere cap and dirac delta distribution
    the azimuthally symmetric cap is only an integral over unassociated legendre polynomials between
    1 and cos(alpha/2)

    Args:
        alpha (float): spherical cap opening angle in radians
        N (int): SH order

    Returns:
        ndarray: SH cap coefficients
    """

    alpha=np.squeeze(alpha)
    w=np.zeros((alpha.size,N+1))

    # window computation:
    P=legendre_u(N, np.cos(alpha/2))
    if alpha.size == 1:
        P = np.array([P])

    z1=np.cos(alpha/2)

    w[:,0]=1-P[:,1]
    for n in range(1,N+1):
        w[:,n]=( -z1 * P[:,n] + P[:,n-1] ) / (n+1)
    
    return np.squeeze(w)

def sph_hankel1(x, nmax):
    """evaluates all spherical Hankel functions of
    the first kind (part of the singular solution)
    up to the degree nmax

    Args:
        x (ndarray): evaluation points
        nmax (int): max SH order

    Returns:
        ndarray: [len(x), nmax+1]
    """

    return sph_bessel(x, nmax) + 1j * sph_neumann(x, nmax)

def sph_hankel1_diff(x, nmax):
    """evaluates all derivatives of spherical Hankel functions of
    the first kind (part of the singular solution)
    up to the degree nmax

    Args:
        x (ndarray): evaluation points
        nmax (int): max SH order

    Returns:
        ndarray: [len(x), nmax+1]
    """

    return sph_bessel_diff(x, nmax) + 1j * sph_neumann_diff(x, nmax)

def sph_hankel2(x, nmax):
    """evaluates all spherical Hankel functions of
    the second kind up to the degree nmax

    Args:
        x (ndarray): evaluation points
        nmax (int): max SH order

    Returns:
        ndarray: [len(x), nmax+1]
    """

    return np.conj(sph_hankel1(x, nmax))

def sph_hankel2_diff(x, nmax):
    """evaluates all derivatives of the spherical Hankel functions of
    the second kind up to the degree nmax

    Args:
        x (ndarray): evaluation points
        nmax (int): max SH order

    Returns:
        ndarray: [len(x), nmax+1]            
    """

    return np.conj(sph_hankel1_diff(x, nmax))

def sph_bessel(x, nmax):
    """evaluates all spherical Bessel functions of
    the first kind up to the degree nmax at values x

    Args:
        x (ndarray): evaluation points
        nmax (int): max SH order

    Returns:
        ndarray: [len(x), nmax+1]            
    """

    j_n = np.zeros((x.size, nmax+1))
    for n in range (0, nmax+1):
        j_n[:,n] = spec.spherical_jn(n, x)

    return j_n

def sph_bessel_diff(x, nmax):
    """evaluates all derivatives of spherical Bessel functions of
    the first kind up to the degree nmax at values x

    Args:
        x (ndarray): evaluation points
        nmax (int): max SH order

    Returns:
        ndarray: [len(x), nmax+1]            
    """

    j_n = np.zeros((x.size, nmax+1))
    for n in range (0, nmax+1):
        j_n[:,n] = spec.spherical_jn(n, x, True)

    return j_n

def sph_neumann(x, nmax):
    """evaluates all spherical Neumann functions (Bessel functions of
    the second kind) up to the degree nmax at values x

    Args:
        x (ndarray): evaluation points
        nmax (int): max SH order

    Returns:
        ndarray: [len(x), nmax+1]            
    """

    y_n = np.zeros((x.size, nmax+1))
    for n in range (0, nmax+1):
        y_n[:,n] = spec.spherical_yn(n, x)

    return y_n

def sph_neumann_diff(x, nmax):
    """evaluates all derivatives of spherical Neumann functions (Bessel functions of
    the second kind) up to the degree nmax at values x

    Args:
        x (ndarray): evaluation points
        nmax (int): max SH order

    Returns:
        ndarray: [len(x), nmax+1]            
    """
    
    y_n = np.zeros((x.size, nmax+1))
    for n in range (0, nmax+1):
        y_n[:,n] = spec.spherical_yn(n, x, True)

    return y_n


# rotation matrices (ported by Benjamin Stahl)
def p_func(i, l, a, b, r1, rlm1):
    ri1 = r1[... ,i + 1, 2]
    rim1 = r1[..., i+1, 0]
    ri0 = r1[..., i+1, 1]

    if b == -l:
        return ri1 * rlm1[..., a + l - 1, 0] + rim1 * rlm1[..., a + l - 1, 2 * l - 2]
    elif b == l: 
        return ri1 * rlm1[..., a + l - 1,  2 * l - 2] - rim1 * rlm1[..., a + l - 1, 0]
    else:
        return ri0 * rlm1[..., a + l - 1, b + l - 1]

def u_func(l, m, n, r1, rlm1):
    return p_func(0, l, m, n, r1, rlm1)

def v_func(l, m, n, r1, rlm1):
    if m == 0:
        p0 = p_func(1, l, 1, n, r1, rlm1)
        p1 = p_func(-1, l, -1, n, r1, rlm1)
        return p0 + p1

    elif m > 0:
        p0 = p_func(1, l, m - 1, n, r1, rlm1)
        if m == 1:
            return p0 * np.sqrt(2)
        else:
            return p0 - p_func(-1, l, 1 - m, n, r1, rlm1)
    else:
        p1 = p_func (-1, l, -m - 1, n, r1, rlm1)
        if m == -1:
            return p1 * np.sqrt (2)
        else:
            return p1 + p_func(1, l, m + 1, n, r1, rlm1)

def w_func(l, m, n, r1, rlm1):
    if m > 0:
        p0 = p_func(1, l, m + 1, n, r1, rlm1)
        p1 = p_func (-1, l, -m - 1, n, r1, rlm1)
        return p0 + p1
    elif m < 0:
        p0 = p_func (1, l, m - 1, n, r1, rlm1)
        p1 = p_func (-1, l, 1 - m, n, r1, rlm1)
        return p0 - p1
    return 0



def calculate_rotation_matrix(order, yaw, pitch, roll):
    yaw_shape = yaw.shape
    yaw_flat = yaw.flatten()
    pitch_flat = pitch.flatten()
    roll_flat = roll.flatten()
    rot_mat = Rotation.from_euler('ZYX', np.stack([yaw_flat, pitch_flat, roll_flat], axis=-1)).as_matrix()

    rot_mat = np.reshape(rot_mat, yaw_shape + (3, 3))
    rot_mat_sh = np.zeros(yaw.shape + ((order+1)**2, (order+1)**2))
    rot_mat_sh[..., 0, 0] = 1

    r1 = np.zeros(yaw.shape + (3, 3))
    r1[..., 0, 0] = rot_mat[..., 1, 1]
    r1[..., 0, 1] = rot_mat[..., 1, 2]
    r1[..., 0, 2] = rot_mat[..., 1, 0]
    r1[..., 1, 0] = rot_mat[..., 2, 1]
    r1[..., 1, 1] = rot_mat[..., 2, 2]
    r1[..., 1, 2] = rot_mat[..., 2, 0]
    r1[..., 2, 0] = rot_mat[..., 0, 1]
    r1[..., 2, 1] = rot_mat[..., 0, 2]
    r1[..., 2, 2] = rot_mat[..., 0, 0]


    rot_mat_sh[..., 1:4, 1:4] = r1 

    offset = 4
    rlm1 = r1
    for l in range(2, order+1):   
        #Rl = order_matrices[l]
        rl = np.zeros(yaw.shape + (2*l+1, 2*l+1))
        for m in range(-l, l+1):
            for n in range(-l, l+1):
                d = int(m == 0)
                if abs(n) == l:
                    denom = (2 * l) * (2 * l - 1)
                else:
                    denom = l * l - n * n

                u = np.sqrt ((l * l - m * m) / denom)
                v = np.sqrt ((1.0 + d) * (l + abs (m) - 1.0) * (l + abs (m)) / denom) * (1.0 - 2.0 * d) * 0.5
                w = np.sqrt ((l - abs (m) - 1.0) * (l - abs (m)) / denom) * (1.0 - d) * (-0.5)

                if u != 0:
                    u *= u_func(l, m, n, r1, rlm1)
                if v != 0:
                    v *= v_func(l, m, n, r1, rlm1)
                if w != 0:
                    w *= w_func(l, m, n, r1, rlm1)

                rl[..., m + l, n + l] = u + v + w
        

        rot_mat_sh[..., offset : offset + 2 * l + 1, offset : offset + 2 * l + 1] = rl
        offset += 2 * l + 1
        rlm1 = rl

    return rot_mat_sh