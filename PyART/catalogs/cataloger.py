import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from .sxs import Waveform_SXS
from .rit import Waveform_RIT
from .icc import Waveform_ICC

from ..analysis.opt_ic import Optimizer

class Cataloger(object):
    """
    Class for catalogs
    """
    def __init__(self,
                 path      = './',
                 sim_list  = [],
                 catalog   = 'rit',
                 verbose   = True,
                 json_file = None, # file forh mismatches
                 add_opts  = {},
                 ):
        
        self.path      = path
        self.catalog   = catalog
        self.sim_list  = sim_list
        self.verbose   = verbose
        
        if json_file is None:
            date     = datetime.now()
            fmt_date = date.strftime("%Y%m%d")
            json_file = f'mismatches_{self.catalog}_{fmt_date}.json'            
        self.json_file = json_file

        if len(self.sim_list)<1:
            raise RuntimeError('Empty list of simulations')
          
        self.Waves = []
        for ID in sim_list:
            try: 
                self.Waves.append(self.Wave(ID, add_opts))
            except Exception as e:
                print(f'Issues with {ID:04}: {e}')
        self.nsims = len(self.Waves)

        pass
    
    def Wave(self, ID, add_opts={}, verbose=None):
        if verbose is None: verbose = self.verbose
        if verbose: print(f'Loading {self.catalog} waveform with ID:{ID:04}')
        
        if self.catalog=='sxs':
            wave = Waveform_SXS(path=self.path, ID=ID, **add_opts)
                                #order='Extrapolated_N3.dir', ellmax=7)
        elif self.catalog=='rit':
            wave = Waveform_RIT(path=self.path, ID=ID, **add_opts)
        elif self.catalog=='icc':
            wave = Waveform_ICC(path=self.path, ID=ID, **add_opts)
        else:
            raise ValueError(f'Unknown catalog: {self.catalog}')
        return wave
    
    def plot_waves(self, cmap='rainbow'):
        mycmap = plt.get_cmap(cmap)
        colors = mycmap(np.linspace(0,1,self.nsims))
        plt.figure()
        for i, wave in enumerate(self.Waves):
            if (2,2) in wave.hlm:
                tmrg,Amrg,_,_ = wave.find_max()
                plt.plot(wave.u-tmrg, wave.hlm[(2,2)]['A'], c=colors[i])
        plt.show()
        pass

    def optimize_mismatches(self, optimizer_opts={}, verbose=None):
        if verbose is None: verbose = self.verbose

        # set some options according to class-instance
        optimizer_opts['json_file'] = self.json_file
        optimizer_opts['verbose']   = verbose

        # run optimizer on all the waveforms 
        for wave in self.Waves:
            opt = Optimizer(wave, **optimizer_opts)
        pass
  







