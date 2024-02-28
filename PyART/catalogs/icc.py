import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ..simulations import Simulations
from ..waveform    import Waveform, waveform2energetics

from .processwave import Multipole
from ..analysis.scattering_angle import ScatteringAngle
from ..utils.utils import retarded_time 

################################
# Class for a single waveform
################################
class WaveIntegrated(Waveform):
    def __init__(self,
                 path        = './',
                 ellmax      = 4,
                 r_extr      = 100,
                 M           = 1,
                 modes       = [(2,2)],
                 integr_opts = None
                 ) -> None:
        super().__init__()
        
        self.path    = path
        self.r_extr  = r_extr
        self.M       = M
        self.modes   = modes
        
        if integr_opts is None:
            integr_opts['method']      = 'FFI'
            integr_opts['f0']          = 0.007
            integr_opts['deg']         = 0
            integr_opts['poly_int']    = None
            integr_opts['extrap_psi4'] = False
        
        self.load_psi4()
        self.integrate_psi4(integr_opts)
        self.dynamics_from_hlm(self.modes)
        pass

    def psi4_name(self, l, m, r_extr=None):
        if r_extr is None: r_extr = self.r_extr
        return f'mp_psi4_l{l:d}_m{m:d}_r{r_extr:.2f}.asc'
    
    def load_psi4(self):
        d = {}
        # start by getting time
        fname   = os.path.join(self.path,self.psi4_name(l=2,m=2))
        X0      = np.loadtxt(fname)
        t       = X0[:,0]
        self._t = t
        self._u = retarded_time(t,self.r_extr,M=self.M) 
        tlen = len(t)
        resize = False
        for mm in self.modes:
            l, m = mm
            fname = os.path.join(self.path,self.psi4_name(l=l,m=m))
            if os.path.exists(fname):
                # procedure to load also problematic files
                lines = []
                with open(fname, 'r') as file:
                    for line in file.readlines():
                        parts = line.strip().split()
                        if len(parts)==3 and '#' not in line:
                            lines.append(line)
                X = np.array([line.strip().split() for line in lines], dtype=float)
                
                Xlen = len(X[:,0])
                if Xlen!=tlen:
                    minlen = min(Xlen,tlen)
                    resize = True
                t    = X[:,0]/self.M
                psi4 = (X[:,1]+1j*X[:,2])*self.M*self.r_extr
            else:
                raise FileNotFoundError(f'{fname} not found')
            d[(l,m)] = self.get_multipole_dict(psi4)
        if resize:
            self._t = self._t[:minlen]
            self._u = self._u[:minlen]
            for mm in self.modes:
                l,m = mm
                for key, val in d[(l,m)].items():
                    d[(l,m)][key] = val[:minlen]
        self._psi4lm = d
        pass
    
    def integrate_psi4(self, integr_opts):
        method      = integr_opts['method']
        f0          = integr_opts['f0']
        deg         = integr_opts['deg']
        poly_int    = integr_opts['poly_int']
        extrap_psi4 = integr_opts['extrap_psi4']

        for mm in self.modes:
            l, m = mm
            psi4 = self._psi4lm[(l,m)]['h'] #FIXME this h is confusing
            mode = Multipole(l, m, self._t, psi4, mass=self.M, radius=1.0, path=None) #FIXME remove this path
            if method=='FFI':
                mode.fixed_freq_int(fcut=2*f0/max(1,abs(m)),extrap_psi4=extrap_psi4)
            elif method=='TDI':
                mode.time_domain_int(deg=deg,poly_int=poly_int,extrap_psi4=extrap_psi4) 
            else:
                raise RuntimeError('Unknown method: {:s}'.format(integration['method']))
            self._dothlm[(l,m)] = self.get_multipole_dict(mode.dh)
            self._hlm[(l,m)]    = self.get_multipole_dict(mode.h)
        pass 
    
    def get_multipole_dict(self, wave):
        return {'real':wave.real, 'imag':wave.real, 'h':wave,
                'A':np.abs(wave), 'p':-np.unwrap(np.angle(wave))}

    #def compute_energetics(self):
        #self.energetics = waveform2energetics(h=self._hlm, doth=self._dothlm, t=self._t, modes=self.modes, mnegative=False)

################################
# Class for the ICC catalog
################################
class Catalog(object):
    def __init__(self, 
                 basepath    = './',
                 ell_emms    = 'all',
                 ellmax      = 4,
                 nonspinning = False, # load only nonspinning
                 integr_opts = None,
                 load_puncts = True
                 ) -> None:
        simulations       = Simulations(path=basepath, icc_add_meta=True, metadata_ext='.bbh')
        self.catalog_meta = simulations.data
        self.nonspinning  = nonspinning
        self.integr_opts  = integr_opts

        self.ellmax   = ellmax
        self.ell_emms = ell_emms
        if self.ell_emms == 'all': 
            self.modes = [(ell, emm) for ell in range(2,self.ellmax+1) for emm in range(-ell, ell+1)]
        else:
            self.modes = self.ell_emms # TODO: add check on input
        data = []
        for i, meta in enumerate(self.catalog_meta):
            spin_asum = abs(float(meta['initial-bh-spin1z'])) + abs(float(meta['initial-bh-spin2z']))
            if not self.nonspinning or spin_asum<1e-14:
                wave = WaveIntegrated(path=meta['path'], r_extr=meta['r0'], modes=self.modes, M=meta['M'],
                                      integr_opts=integr_opts)
                if load_puncts:
                    tracks = self.load_tracks(path=meta['path'])
                    punct0 = np.column_stack( (tracks['t'], tracks['t'], tracks['x0'], tracks['y0'], tracks['z0']) )
                    punct1 = np.column_stack( (tracks['t'], tracks['t'], tracks['x1'], tracks['y1'], tracks['z1']) )
                    scat_NR = ScatteringAngle(punct0=punct0, punct1=punct1, file_format='GRA', nmin=2, nmax=5, n_extract=4,
                                           r_cutoff_out_low=25, r_cutoff_out_high=None,
                                           r_cutoff_in_low=25, r_cutoff_in_high=100,
                                           verbose=False)
                    scat_info = {'chi':scat_NR.chi, 'chi_fit_err':scat_NR.fit_err}
                else:
                    tracks    = {}
                    scat_info = {}
                
                sim_data            = lambda:0
                sim_data.meta       = meta
                sim_data.wave       = wave
                sim_data.tracks     = tracks
                sim_data.scat_info  = scat_info
                data.append(sim_data)
        self.data = data
        pass
    
    def load_tracks(self, path):
        # FIXME: very specific
        fname = 'puncturetracker-pt_loc..asc'
        X  = np.loadtxt(os.path.join(path,fname))
        t  = X[:, 8]
        x0 = X[:,22]
        y0 = X[:,32]
        x1 = X[:,23]
        y1 = X[:,33]
        x  = x0-x1
        y  = y0-y1
        r  = np.sqrt(x**2+y**2)
        th = -np.unwrap(np.angle(x+1j*y))
        return {'t':t, 'x0':x0, 'x1':x1, 'y0':y0, 'y1':y1, 'r':r, 'th':th, 'z0':0*t, 'z1':0*t}

    def get_simlist(self):
        """
        Get list of simulations' names
        """
        simlist = []
        for meta in self.catalog_meta:
            simlist.append(meta['name'])
        return simlist

    def idx_from_value(self,value,key='name',single_idx=True):
        """ 
        Return idx with metadata[idx][key]=value.
        If single_idx is False, return list of indeces 
        that satisfy the condition
        """
        idx_list = []
        for idx, meta in enumerate(self.catalog_meta):
            if meta[key]==value:
                idx_list.append(idx)
        if len(idx_list)==0: 
            return None
        if single_idx:
            if len (idx_list)>1:
                raise RuntimeError(f'Found more than one index for value={value} and key={key}')
            else:
                return idx_list[0]
        else: 
            return idx_list

    def meta_from_name(self,name):
        return self.catalog_meta[self.idx_from_value(name)]
    
    def wave_from_name(self,name):
        return self.waves[self.idx_from_value(name)]

