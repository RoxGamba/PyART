"""
Classes to handle waveforms
"""

# standard imports
import numpy as np;
from scipy.signal import find_peaks
from scipy import integrate
import matplotlib.pyplot as plt
import warnings 

# other imports
from .utils import utils         as ut
from .utils import wf_utils      as wf_ut
from .utils import load_nr_utils as nr_ut
#from .analysis.integrate_multipole import Multipole
from .analysis.integrate_wave import IntegrateMultipole


class Waveform(object):
    """
    Parent class to handle waveforms
    Children classes will inherit methods & propreties
    """

    def __init__(self):
        self._t      = None # I would like to kill this
        self._u      = None
        self._t_psi4 = None
        self._f      = None
        self._hp     = None
        self._hc     = None
        self._hlm    = {}
        self._dothlm = {}
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
    def t_psi4(self):
        return self._t_psi4
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
    def dothlm(self):
        return self._dothlm
    @property
    def psi4lm(self):
        return self._psi4lm
    @property
    def dyn(self):
        return self._dyn
    @property
    def kind(self):
        return self._kind
    
    def find_max(
                self, 
                mode   = (2,2), 
                kind   = 'last-peak',
                umin   = 0,
                height = 0.15,
                return_idx = False
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
        peaks, props = find_peaks(Alm, height=height)

        if kind == 'first-max-after-t':
            for i in range(len(peaks)):
                if(u[peaks[i]] > umin):
                    break
        elif kind == 'last-peak':
            i = len(peaks) - 1
            
        elif kind == 'global':
            Alms = props['peak_heights']
            i    = np.argmax(Alms)
        else:
            raise ValueError("`kind' for merger not found")

        u_mrg    = u[peaks[i]]
        A_mrg    = Alm[peaks[i]]
        omg_mrg  = omg[peaks[i]]
        domg_mrg = domg[peaks[i]]

        if return_idx:
            return u_mrg, A_mrg, omg_mrg, domg_mrg, peaks[i]
        else:
            return u_mrg, A_mrg, omg_mrg, domg_mrg
    
    def cut(self, DeltaT, cut_hpc=True, from_the_end=False, allw=False,
                  cut_dothlm=True, cut_psi4lm=True): 
        """
        Cut the waveform (hlm) removing the 
        first DeltaT M 
        (or last if from_the_end=True)
        Store old retarded time in self.u0
        """
        u_from_zero = self.u-self.u[0]
        if u_from_zero[-1]<DeltaT:
            raise RuntimeError('Cutting too much, no points left!')

        if from_the_end:
            i0 = np.where(u_from_zero > u_from_zero[-1]-DeltaT)[0][0]
            tslice = slice(None, i0)
        else:
            i0 = np.where(u_from_zero > DeltaT)[0][0]
            tslice = slice(i0, None)
        
        self.u0 = self.u
        self._u = self.u[tslice]
        if self.t is not None:
            self._t = self.t[tslice]
        
        def cut_all_modes(wave):
            if wave: 
                for k in wave.keys():
                    for sk in wave[k].keys():
                        wave[k][sk] = wave[k][sk][tslice]
            return wave

        # resize hlm
        self._hlm = cut_all_modes(self.hlm)
        if cut_dothlm:
            self._dothlm = cut_all_modes(self.dothlm)
        if cut_psi4lm:
            self._psi4lm = cut_all_modes(self.psi4lm)
            if self.t_psi4 is None:
                self._t_psi4 = self.t_psi4[tslice]

        # resize polarizations 
        if cut_hpc and self.hp is not None:
            self._hp = self.hp[tslice]
        if cut_hpc and self.hc is not None:
            self._hc = self.hc[tslice]

        return

    def compute_hphc(self, phi=0, i=0, modes=[(2,2)]):
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
            h  = self.hlm[k]['h']
            iA = np.interp(new_u, self.u, np.abs(h))
            ip = np.interp(new_u, self.u, -np.unwrap(np.angle(h)))
            ih = iA*np.exp(-1j*ip)
            hlm_i[k] = {"A": iA, "p": ip, "h":ih, "real": ih.real, "imag": ih.imag}
        
        return new_u, hlm_i
    
    def dynamics_from_hlm(self, modes, warning=False):
        """
        Compute GW energy and angular momentum fluxes from multipolar waveform
        """
        
        if not self.dothlm:
            if warning: warnings.warn('Warning: dothlm not found, computing derivatives of hlm')
            for mode in modes:
                self.dothlm[mode]      = {}
                self.dothlm[mode]['h'] = ut.D02(self.t, self.hlm[mode]['h'])

        dynamicsdict = waveform2energetics(
                        self.hlm, self.dothlm, self.t, modes,
                        )

        self._dyn = {**self._dyn, **dynamicsdict}
        pass

    def ej_from_hlm(self, M_adm, J_adm, m1, m2, modes):
        """
        Compute GW energy and angular momentum from multipolar waveform
        """

        self.dynamics_from_hlm(modes)

        E    = self.dyn['E']['total']
        J    = self.dyn['J']['total']

        e    = (M_adm - E)                             # not nu-normalized total energy
        eb   = (M_adm - E - m1 - m2) / (m1*m2/(m1+m2)) # nu-normalized binding energy
        jorb = (J_adm - J) / (m1*m2)

        return eb, e, jorb
    
    def to_frequency(self, taper=True, pad=True):
        """
        Fourier transform the waveform from time to frequency domain
        """
        
        dt = self.u[1] - self.u[0]
        # window
        if taper:
            self._hp = ut.windowing(self.hp, alpha=0.1)
            self._hc = ut.windowing(self.hc, alpha=0.1)
        
        if pad:
            srate    = 1./dt
            seglen   = ut.nextpow2(self.u[-1])
            dN       = int((seglen - len(self.u)*srate)/srate)
            self._u  = np.arange(0., seglen, dt)
            self._t  = np.arange(0., seglen, dt)
            self._hp = ut.zero_pad_before(self.hp, dN, return_column=False)
            self._hc = ut.zero_pad_before(self.hp, dN, return_column=False)
            assert len(self.u) == len(self.hp)

        self._f, self._hp = ut.fft(self.hp, dt)
        self._f, self._hc = ut.fft(self.hc, dt)
        self._domain = 'Frequency'
        pass
    
    def integrate_psi4(self, t_psi4, radius, integr_opts={}, modes=None, M=1):
        """
        Method to integrate psi4 extracted at finite distance
        """
        if 'method'      not in integr_opts: integr_opts['method']      = 'FFI'
        if 'f0'          not in integr_opts: integr_opts['f0']          = 0.007
        if 'deg'         not in integr_opts: integr_opts['deg']         = 0
        if 'poly_int'    not in integr_opts: integr_opts['poly_int']    = None
        if 'extrap_psi4' not in integr_opts: integr_opts['extrap_psi4'] = False
        if 'window'      not in integr_opts: integr_opts['window']      = None
        if 'walpha'      not in integr_opts: integr_opts['walpha']      = 3
         
        if modes is None:
            modes = self.psi4lm.keys()
        dothlm = {}
        hlm    = {}
        t      = t_psi4
        for i, lm in enumerate(modes):
            l, m = lm
            psi4 = self.psi4lm[lm]['h']
            mode = IntegrateMultipole(l, m, t, psi4, 
                                      mass=M, radius=radius, 
                                      integrand='psi4')
            dothlm[lm] = wf_ut.get_multipole_dict(mode.dh)
            hlm[lm]    = wf_ut.get_multipole_dict(mode.h)
        self._u      = mode.u
        self._hlm    = hlm
        self._dothlm = dothlm
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
    for ky in kys:
        dictdyn[ky] = {}
        dictdyn[ky]['total'] = 0.
    
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
            this_mode    = integrate.cumulative_trapezoid(dictdyn[dotk][(l,m)],t,initial=0)
            dictdyn[kk][(l,m)]      = this_mode
            dictdyn[kk]['total']   += this_mode
            dictdyn[dotk]['total'] += dictdyn[dotk][(l,m)]
    
    return dictdyn


####################################
#class WaveIntegrated(Waveform):
#    """
#    Child class to get NR psi4 and integrate to obtain dh and h
#    """
#
#    def __init__(self,
#                 path        = './',
#                 ellmax      = 4,
#                 r_extr      = 1,
#                 M           = 1,
#                 modes       = [(2,2)],
#                 integr_opts = {},
#                 fmt         = 'etk',
#                 fname       = 'mp_psi4_l@L@_m@M@_r100.00.asc',
#                 integrand   = 'psi4',
#                 norm        = None
#                 ) -> None:
#        super().__init__()
#        
#        self.path      = path
#        self.r_extr    = r_extr
#        self.M         = M
#        self.modes     = modes
#        self.fmt       = fmt
#        self.fname     = fname
#        self.integrand = integrand.lower()
#        self.norm      = norm
#
#        if 'method'      not in integr_opts: integr_opts['method']      = 'FFI'
#        if 'f0'          not in integr_opts: integr_opts['f0']          = 0.007
#        if 'deg'         not in integr_opts: integr_opts['deg']         = 0
#        if 'poly_int'    not in integr_opts: integr_opts['poly_int']    = None
#        if 'extrap_psi4' not in integr_opts: integr_opts['extrap_psi4'] = False
#        if 'window'      not in integr_opts: integr_opts['window']      = None
#        
#        self.load_wave()
#        
#        self.normalize_wave(norm=norm)
#
#        self.integrate_wave(integr_opts)
#
#        self.dynamics_from_hlm(self.modes,warning=True)
#        pass
#
#    def load_wave(self):
#         instance = nr_ut.LoadWave(path=self.path,modes=self.modes,resize=False,fmt=self.fmt,fname=self.fname)
#         self._t  = instance.t
#         self._u  = ut.retarded_time(instance.t,self.r_extr,M=self.M)
#         self.wavelm_file = instance.wave
#         pass
#    
#    def normalize_wave(self, norm=None):
#        """
#        Normalize loaded waveform. 
#        For example, norm='factor2_minusodd_minusm0' activate three flags: factor2, minusodd, minusm0
#        """
#        if norm is None:
#            return 
#        
#        opts = {'factor2':False, 'minusodd':False, 'minusm0':False, 'dividebyR':False} 
#        for elem in norm.split('_'):
#            opts[elem] = True
#        for lm in self.modes:
#            l,m = lm
#            factor = 1
#            if opts['factor2']:
#                factor *= 2
#            if opts['minusodd']:
#                factor *= (-1)**(l+m)
#            if opts['minusm0'] and m==0:
#                factor *= -1
#            if opts['dividebyR']:
#                factor *= 1/self.r_extr
#            self.wavelm_file[(l,m)] = self.wavelm_file[(l,m)]*factor
#        pass 
#
#    def integrate_wave(self, integr_opts):
#        for mm in self.modes:
#            l, m = mm
#            psi4 = self.wavelm_file[(l,m)]
#                        
#            mode = Multipole(l, m, self._t, psi4, mass=self.M, radius=self.r_extr, integrand=self.integrand)
#            mode.integrate_wave(integr_opts=integr_opts)
#            
#            self._psi4lm[(l,m)] = wf_ut.get_multipole_dict(mode.psi)
#            self._dothlm[(l,m)] = wf_ut.get_multipole_dict(mode.dh)
#            self._hlm[(l,m)]    = wf_ut.get_multipole_dict(mode.h)
#        pass
#
#
#
