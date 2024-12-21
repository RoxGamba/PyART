import numpy as np;

try:
    import bajes.obs.gw as gwb
except ModuleNotFoundError:
    print("WARNING: bajes not installed.")

from ..waveform import Waveform
from bajes.obs.gw import Waveform as bWf
from bajes.obs.gw import __approx_dict__

# todo: use astropy units or similar
msun_m = 1.476625061404649406193430731479084713e3
mpc_m  = 3.085677581491367278913937957796471611e22
msun_s = 4.925490947000518753850796748739372504e-6

class Waveform_BaJes(Waveform):
    """
    Class to wrap BaJes waveforms
    """

    def __init__(
                    self, 
                    pars=None, 
                    geom_units=False
                ):
        super().__init__()
        self.__set__default_pars__()
        self.pars  = {**self.pars, **pars}
        self._kind = 'BaJes'
        self.__initialize_generator__()
        self.__run__(geom_units=geom_units)

    def __set__default_pars__(self):
        self.pars = {  'approx'      :   'TEOBResumS',
                        'phi_ref'    :   0.,
                        'distance'   :   1.,
                        'time_shift' :   0.,
                        'iota'  :        0.,
                        'lmax'  :        0,
                        'srate' :        4096,
                        'seglen':        16,
                        'f_min' :        20,
                        'eccentricity':  0.,
                        'f_max' :       2048,
                        'tukey' :       0.4/16,
    }

    def __initialize_generator__(self):

        seglen = self.pars['seglen']
        srate  = self.pars['srate']
        freqs  =  np.linspace(0,srate/2,seglen*srate//2+1)

        self.generator = bWf(freqs, 
                             srate= srate, 
                             seglen=seglen, 
                             approx=self.pars['approx'],
                             )
        
        if __approx_dict__[self.pars['approx']]['domain'] == 'time':
            self._domain = 'Time'
            self.dx = 1./srate
        elif __approx_dict__[self.pars['approx']]['domain'] == 'freq':
            self._domain = 'Freq'
            self.dx = 1./seglen
        else:
            raise ValueError('Domain not recognized')
        
        pass

    def __run__(self, geom_units=False):
        
        hp, hc = self.generator.compute_hphc(self.pars)
        dx     = self.dx

        conv_factor = 1
        if geom_units:
            M  = self.pars['mass']
            Dl = self.pars['distance']
            if self.domain == 'Time':
               conv_factor = M*msun_m/(Dl*mpc_m)
               dx          /= (M*msun_s)
            elif self.domain == 'Freq':
               conv_factor = M*M*msun_m*msun_s/(Dl*mpc_m)
               dx          *= (M*msun_s)

        hp, hc = hp/conv_factor, hc/conv_factor
        x      = np.arange(len(hp))*dx

        self._hp = hp
        self._hc = hc
        if self._domain == 'Time':
            self._u = x
        else:
            self._f = x

        return x, hp, hc

           



