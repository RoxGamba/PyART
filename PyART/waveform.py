# Classed to handle waveforms

#standard imports
import numpy as np;
from scipy.signal import find_peaks
from scipy import integrate

# other imports
from .utils import utils as ut
from .utils import wf_utils as wf_ut

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
    def psi4lm(self):
        return self._psi4lm
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
                mode   = (2,2), 
                kind   = 'last-peak',
                umin   = 0,
                height = 0.15,
                ):
        
        u   = self.u
        p   = self.hlm[mode]['p']
        Alm = self.hlm[mode]['A']

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
            hlm_i[k] = {"A"   : np.interp(new_u, self.u, self.hlm[k]["A"]),
                        "p"   : np.interp(new_u, self.u, self.hlm[k]["p"]),
                        "real": np.interp(new_u, self.u, self.hlm[k]["real"]),
                        "imag": np.interp(new_u, self.u, self.hlm[k]["imag"]),
                        'h'   : np.interp(new_u, self.u, self.hlm[k]['h']), 
                        }
        
        return new_u, hlm_i
    
    def dynamics_from_hlm(self, modes):
        """
        Compute GW energy and angular momentum fluxes from multipolar waveform
        """

        self.dhlm = {}
        for mode in modes:
            self.dhlm[mode]      = {}
            self.dhlm[mode]['h'] = ut.D02(self.t, self.hlm[mode]['h'])

        dynamicsdict = waveform2energetics(
                        self.hlm, self.dhlm, self.t, modes,
                        )

        self._dyn = {**self._dyn, **dynamicsdict}
        pass     


def waveform2energetics(h, doth, t, modes, mnegative=False):
    """
    Compute GW energy and angular momentum from multipolar waveform
    See e.g. https://arxiv.org/abs/0912.1285

    * h[(l,m)]     : multipolar strain 
    * doth[(l,m)]  : time-derivative of multipolar strain
    * t            : time array
    * modes        : (l,m) indexes
    * mnegative    : if True, account for the factor 2 due to m<0 modes 
    """    
    oo16pi  = 1./(16*np.pi)

    lmodes = [lm[0] for lm in modes]
    mmodes = [lm[1] for lm in modes]
    lmin = min(lmodes)

    if lmin < 2:
        raise ValueError("l>2")
    if lmin != 2:
        print("Warning: lmin > 2")
        
    mnfactor = np.ones_like(mmodes)
    if mnegative:
        mnfactor = [1 if m == 0 else 2 for m in mmodes]
    else:
        if all(m >= 0 for m in mmodes):
            print("Warning: m>=0 but not accouting for it!")

    # set up dictionaries
    kys = ['dotE', 'E', 
           'dotJ', 'J', 
           'dotJz', 'dotJy', 'dotJx', 
           'Jz', 'Jy', 'Jx', 
           'dotP', 'P', 'dotPz', 'dotPy', 'dotPx', 'Pz', 'Py', 'Px'
           ]

    # all of the above will be stored in a dictionary
    dictdyn = {}
    for k in kys:
        dictdyn[k] = {}
        dictdyn[k]['total'] = 0.
    
    for k, (l,m) in enumerate(modes):

        fact = mnfactor[k] * oo16pi
        
        # Energy
        dictdyn['dotE'][(l,m)] = fact * np.abs(doth[(l,m)]['h'])**2 

        # Angular momentum
        dictdyn['dotJz'][(l,m)] = fact * m * np.imag(h[(l,m)]['h'] * np.conj(doth[(l,m)]['h']))

        dothlm_1 = doth[(l,m-1)]['h'] if (l,m-1) in doth else 0*h[(l,m)]['h']
        dothlm1  = doth[(l,m+1)]['h'] if (l,m+1) in doth else 0*h[(l,m)]['h']

        dictdyn['dotJy'][(l,m)] = 0.5 * fact * \
                                np.real( h[(l,m)]['h'] * (wf_ut.mc_f(l,m) * np.conj(dothlm1) - wf_ut.mc_f(l,-m) * np.conj(dothlm_1) ))
        dictdyn['dotJx'][(l,m)] = 0.5 * fact * \
                                np.real( h[(l,m)]['h'] * (wf_ut.mc_f(l,m) * np.conj(dothlm1) + wf_ut.mc_f(l,-m) * np.conj(dothlm_1) ))
        dictdyn['dotJ'][(l,m)] = np.sqrt(dictdyn['dotJx'][(l,m)]**2 + 
                                         dictdyn['dotJy'][(l,m)]**2 + 
                                         dictdyn['dotJz'][(l,m)]**2
                                        )

        # Linear momentum
        dothlm1   = doth[(l,m+1)]['h']   if (l,m+1)   in doth else 0*h[(l,m)]['h']
        dothl_1m1 = doth[(l-1,m+1)]['h'] if (l-1,m+1) in doth else 0*h[(l,m)]['h']
        dothl1m1  = doth[(l+1,m+1)]['h'] if (l+1,m+1) in doth else 0*h[(l,m)]['h']
        dotl_1m   = doth[(l-1,m)]['h']   if (l-1,m)   in doth else 0*h[(l,m)]['h']
        dothl1m   = doth[(l+1,m)]['h']   if (l+1,m)   in doth else 0*h[(l,m)]['h']
        
        dotPxiy = 2.0 * fact * doth[(l,m)]['h'] * \
                (wf_ut.mc_a(l,m) * np.conj(dothlm1) + wf_ut.mc_b(l,-m) * np.conj(dothl_1m1) - wf_ut.mc_b(l+1,m+1) * np.conj(dothl1m1))
        dictdyn['dotPy'][(l,m)] = np.imag(dotPxiy)
        dictdyn['dotPx'][(l,m)] = np.real(dotPxiy)
        dictdyn['dotPz'][(l,m)] = fact * np.imag( doth[(l,m)]['h'] * \
                                (wf_ut.mc_c(l,m) * np.conj(doth[(l,m)]['h']) + wf_ut.mc_d(l,m) * np.conj(dotl_1m) + wf_ut.mc_d(l+1,m) * np.conj(dothl1m)) )

        dictdyn['dotP'][(l,m)] = np.sqrt(dictdyn['dotPx'][(l,m)]**2 + 
                                         dictdyn['dotPy'][(l,m)]**2 + 
                                         dictdyn['dotPz'][(l,m)]**2
                                        )
        # Sum up and set dictionary
        kks = ['E', 'J', 'Jz', 'Jy', 'Jx', 'P', 'Pz', 'Py', 'Px']
        for kk in kks:
            dotk = 'dot' + kk
            this_mode    = integrate.cumtrapz(dictdyn[dotk][(l,m)],t,initial=0)
            dictdyn[kk][(l,m)]    = this_mode
            dictdyn[kk]['total'] += this_mode

    return dictdyn