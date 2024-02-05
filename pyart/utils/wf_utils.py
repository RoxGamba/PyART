import numpy as np;
import utils as ut;

def mnfactor(m):
    """
    Factor to account for negative m modes
    """
    return 1 if m == 0 else 2

def mode_to_k(ell,emm):
    return int(ell*(ell-1)/2 + emm-2)

def modes_to_k(modes):
    return [mode_to_k(x[0],x[1]) for x in modes]

def k_to_ell(k):
    LINDEX = [\
    2,2,\
    3,3,3,\
    4,4,4,4,\
    5,5,5,5,5,\
    6,6,6,6,6,6,\
    7,7,7,7,7,7,7,\
    8,8,8,8,8,8,8,8]
    return LINDEX[k]

def k_to_emm(k):
    MINDEX = [\
    1,2,\
    1,2,3,\
    1,2,3,4,\
    1,2,3,4,5,\
    1,2,3,4,5,6,\
    1,2,3,4,5,6,7,\
    1,2,3,4,5,6,7,8];
    return MINDEX[k]  


def compute_hphc(hlm, phi=0, i=0, modes=['1']):
    """
    For aligned spins, assuming usual symmetry between hlm and hl-m
    """
    h = 0+1j*0
    for k in modes:
        ki  = int(k)
        ell = k_to_ell(ki)
        emm = k_to_emm(ki)
        Alm = hlm[k][0]
        plm = hlm[k][1]
        Hp  = Alm*np.exp(-1j*plm)
        Hn  = (-1)**ell*Alm*np.exp( 1j*plm)

        Ylmp = ut.spinsphericalharm(-2, ell, emm, np.pi/2-phi, i)
        Ylmn = ut.spinsphericalharm(-2, ell,-emm, np.pi/2-phi, i)
        h   += Ylmp*Hp + Ylmn*Hn

    hp =  np.real(h)
    hc = -np.imag(h)
    return hp, hc

def taper(t, h, M, alpha, tau, Msuns=1.):
    """
    Taper a waveform using an hyperbolic tangent
    """
    tm = t/(M*Msuns)
    window = 0.5*(1.+np.tanh(tm*alpha-tau))
    return (window*h)

##########################
#   Phasing, from watpy  #
##########################

def align_phase(t, Tf, phi_a_tau, phi_b):
    """
        Align two waveforms in phase by minimizing the chi^2
        
        \chi^2 = \int_0^Tf [\phi_a(t + \tau) - phi_b(t) - \Delta\phi]^2 dt
        
        as a function of \Delta\phi.
        
        * t         : time, must be equally spaced
        * Tf        : final time
        * phi_a_tau : time-shifted first phase evolution
        * phi_b     : second phase evolution
        
        This function returns \Delta\phi.
    """
    dt     = t[1] - t[0]
    weight = np.double((t >= 0) & (t < Tf))
    return np.sum(weight * (phi_a_tau - phi_b) * dt) / np.sum(weight * dt)

def Align(t, Tf, tau_max, t_a, phi_a, t_b, phi_b):
    """
        Align two waveforms in phase by minimizing the chi^2
        
        chi^2 = \sum_{t_i=0}^{t_i < Tf} [phi_a(t_i + tau) - phi_b(t_i) - dphi]^2 dt
        
        as a function of dphi and tau.
        
        * t          : time
        * Tf         : final time
        * tau_max    : maximum time shift
        * t_a, phi_a : first phase evolution
        * t_b, phi_b : second phase evolution
        
        The two waveforms are re-sampled using the given time t
        
        This function returns a tuple (tau_opt, dphi_opt, chi2_opt)
    """
    dt     = t[1] - t[0]
    N      = int(tau_max/dt)
    weight = np.double((t >= 0) & (t < Tf))
    
    res_phi_b = np.interp(t, t_b, phi_b)
    
    tau  = []
    dphi = []
    chi2 = []
    for i in range(-N, N):
        tau.append(i*dt)
        res_phi_a_tau = np.interp(t, t_a + tau[-1], phi_a)
        dphi.append(align_phase(t, Tf, res_phi_a_tau, res_phi_b))
        chi2.append(np.sum(weight*
                              (res_phi_a_tau - res_phi_b - dphi[-1])**2)*dt)
    
    chi2 = np.array(chi2)
    imin = np.argmin(chi2)
    return tau[imin], dphi[imin], chi2[imin]

def remap(h_re, h_im):
    amp   = np.sqrt(h_re**2 + h_im**2)
    phase = np.unwrap(np.angle(h_re - 1j*h_im))   
    return amp, phase

def shift_waveform(h_re, h_im, t_shift_idx, phi_shift):
    h_re = np.roll(h_re, -t_shift_idx)
    h_im = np.roll(h_im, -t_shift_idx)
    A, phi     = remap(h_re, h_im)
    phi        = phi - phi_shift
    return A*np.cos(phi), -A*np.sin(phi)