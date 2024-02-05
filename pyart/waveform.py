# Classed to handle waveforms

#standard imports
import numpy as np; import h5py; import json
from scipy.signal import find_peaks

# other imports
import utils as ut
import wf_utils as wf_ut

class Waveform(object):
    """
    Parent class to handle waveforms
    Children classes will inherit methods & propreties
    """

    def __init__(self):
        self._t      = None
        self._u      = None
        self._f      = None
        self._hp     = None
        self._hc     = None
        self._hlm    = {}
        self._psi4lm = {}
        self._dyn    = {}
        self._kind   = None
        pass
    
    @property
    def t(self):
        return self._t
    @property
    def u(self):
        return self._u
    @property
    def f(self):
        return self._f
    @property
    def hp(self):
        return self._hp
    @property
    def hc(self):
        return self._hc
    @property
    def hlm(self):
        return self._hlm
    @property
    def dyn(self):
        return self._dyn
    @property
    def kind(self):
        return self._kind
    
    # methods
    def extract_hlm(self, ell, emm):
        k = int(ell*(ell-1)/2 + emm-2)
        return self.hlm[str(k)][0], self.hlm[str(k)][1]

    def find_max(
                self, 
                mode   = '1', 
                kind   = 'first-max-after-t',
                umin   = 0,
                height = 0.15
                ):
        
        u   = self.u
        p   = self.hlm[mode][1]
        Alm = self.hlm[mode][0]

        # compute omega
        omg      = np.zeros_like(p)
        omg[1:]  = np.diff(p)/np.diff(u)
        # compute domega
        domg     = np.zeros_like(omg)
        domg[1:] = np.diff(omg)/np.diff(u)
        
        # find peaks
        peaks, _ = find_peaks(Alm, height=height)

        if kind == 'first-max-after-t':
            for i in range(len(peaks)):
                if(u[peaks[i]] > umin):
                    break
        elif kind == 'last-peak':
            i = len(peaks) - 1
        else:
            raise ValueError("`kind' for merger not found")

        u_mrg    = u[peaks[i]]
        A_mrg    = Alm[peaks[i]]
        omg_mrg  = omg[peaks[i]]
        domg_mrg = domg[peaks[i]]

        return u_mrg, A_mrg, omg_mrg, domg_mrg

    def energetics_hlm(self, modes=['1']):
        """
        Compute the (E, J) from the multipoles
        or the dynamics.
        TODO: improve, see catalogs/processwave.py
        """
        u  = self.u
        du = u[1] - u[0]
        
        E_GW_dot = {}
        E_GW     = {}
        J_GW_dot = {}
        J_GW     = {}    

        E_GW_dot_all = np.zeros_like(u)
        E_GW_all     = np.zeros_like(u)
        J_GW_dot_all = np.zeros_like(u)
        J_GW_all     = np.zeros_like(u)

        for mode in modes:
            m      = wf_ut.k_to_emm(int(mode))
            this_h = self.hlm[mode]
            hlm    = this_h[0]*np.exp(-1j*this_h[1])
            hlm_dot= ut.D02(u, hlm)

            # energy and flux in single |mode| 
            E_GW_dot[mode] = wf_ut.mnfactor(m)*1.0/(16.*np.pi) * np.abs(hlm_dot)**2 
            E_GW[mode]     = wf_ut.mnfactor(m)*ut.integrate(E_GW_dot[mode]) * du
            J_GW_dot[mode] = wf_ut.mnfactor(m)*1.0/(16.*np.pi) * m * np.imag(hlm * np.conj(hlm_dot)) 
            J_GW[mode]     = wf_ut.mnfactor(m)*ut.integrate(J_GW_dot[mode]) * du

            E_GW_dot_all += E_GW_dot[mode]
            E_GW_all     += E_GW[mode]
            J_GW_dot_all += J_GW_dot[mode]
            J_GW_all     += J_GW[mode]
        
        return E_GW_all, E_GW_dot_all, J_GW_all, J_GW_dot_all

    def compute_hphc(self, phi=0, i=0, modes=['1']):
        """
        For aligned spins, compute hp and hc
        """
        self._hp, self._hc = wf_ut.compute_hphc(self.hlm, phi, i, modes)
        return self.hp, self.hc
    
    def interpolate_hlm(self, dT):
        """
        Interpolate the hlm dictionary to a grid of uniform dT
        """
        hlm_i = {}
        new_u = np.arange(self.u[0], self.u[-1], dT)
        
        for k in self.hlm.keys():
            hlm_i[k] = [np.interp(new_u, self.u, self.hlm[k][0]),
                        np.interp(new_u, self.u, self.hlm[k][1])]
        
        return new_u, hlm_i