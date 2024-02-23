import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from simulations import Simulations
from waveform    import Waveform

################################
# Class for a single waveform
################################
class WaveICC(Waveform):
    def __init__(self,
                 path     = './',
                 ellmax   = 4,
                 r_extr   = 100,
                 M        = 1,
                 modes    = [(2,2)]
                 ) -> None:
        super().__init__()
        self.path     = path

        self.r_extr   = r_extr
        self.M        = M
        self.modes    = modes
          
        self.load_psi4()

        pass

    def psi4_name(self, l, m, r_extr=None):
        if r_extr is None: r_extr = self.r_extr
        return f'mp_psi4_l{l:d}_m{m:d}_r{r_extr:.2f}.asc'
    
    def load_psi4(self):
        d = {}
        # start by getting time
        fname = os.path.join(self.path,self.psi4_name(l=2,m=2))
        X0 = np.loadtxt(fname)
        t       = X0[:,0]
        self._t = t
        self._u = t # FIXME
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
                t  = X[:,0]/self.M
                re = self.r_extr*X[:,1]*self.M
                im = self.r_extr*X[:,2]*self.M 
                h = re+1j*im
                A = np.abs(h)
                p = -np.unwrap(np.angle(h))
            else:
                raise FileNotFoundError(f'{fname} not found')
            d[(l,m)] = {'real':re, 'imag':im, 'h':h, 'A':A, 'p':p}
        self._psi4 = d

        if resize:
            self._t = self._t[:minlen]
            self._u = self._u[:minlen]
            for mm in self.modes:
                l,m = mm
                for key, val in d[(l,m)].items():
                    d[(l,m)][key] = val[:minlen]
        pass

################################
# Class for the ICC catalog
################################
class CatalogICC(object):
    def __init__(self, 
                 basepath ='./',
                 ell_emms = 'all',
                 ellmax   = 4
                 ) -> None:
        
        icc_simulations = Simulations(path=basepath, icc_add_meta=True, metadata_ext='.bbh')
        self.catalog_metadata = icc_simulations.data
        
        self.ellmax   = ellmax
        self.ell_emms = ell_emms
        if self.ell_emms == 'all': 
            self.modes = [(ell, emm) for ell in range(2,self.ellmax+1) for emm in range(-ell, ell+1)]
        else:
            self.modes = self.ell_emms # TODO: add check on input
        
        waves = []
        for i, meta in enumerate(self.catalog_metadata):
            wave = WaveICC(path=meta['path'], r_extr=meta['r0'], modes=self.modes, M=meta['M'])
            waves.append(wave)
        self.waves = waves
        pass
    
    def get_simlist(self):
        simlist = []
        for meta in self.catalog_metadata:
            simlist.append(meta['name'])
        return simlist

    def idx_from_value(self,value,key='name',single_idx=True):
        """ 
        Return idx with metadata[idx][key]=value.
        If single_idx is False, return list of indeces 
        that satisfy the condition
        """
        idx_list = []
        for idx, meta in enumerate(self.catalog_metadata):
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
        return self.catalog_metadata[self.idx_from_value(name)]
    
    def wave_from_name(self,name):
        return self.waves[self.idx_from_value(name)]

if __name__ == '__main__':
    
    icc = CatalogICC(basepath='/Users/simonealbanesi/repos/eob_nr_explore/data/scattering')
    
    modes2plot = [ [(2,2), (3,2)], [(4,4), (4,2)] ]
    ylabs = [ [r'$\psi_{4,22}$', r'$\psi_{4,32}$'], [r'$\psi_{4,44}$', r'$\psi_{4,42}$']]
    colors = matplotlib.cm.jet(np.linspace(0,1,num=len(icc.waves)))
    fig, axs = plt.subplots(2,2,figsize=(12,8))
    for k,wave in enumerate(icc.waves):
        for i in range(2):
            for j in range(2):
                axs[i,j].plot(wave._t, wave._psi4[modes2plot[i][j]]['A'], color=colors[k])
    for i in range(2):
        for j in range(2):
            axs[i,j].set_xlim([250, 440])
            axs[i,j].set_ylabel(ylabs[i][j])
    plt.show()
     
