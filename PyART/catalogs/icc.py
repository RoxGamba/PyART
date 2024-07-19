import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ..simulations import Simulations #TODO: remove dependence from this class
from ..waveform    import Waveform, WaveIntegrated

from ..analysis.scattering_angle import ScatteringAngle

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

