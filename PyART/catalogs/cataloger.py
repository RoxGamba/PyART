import numpy as np
import matplotlib.pyplot as plt
from .sxs import Waveform_SXS
from .rit import Waveform_RIT
from .icc import Waveform_ICC

class Cataloger(object):
    """
    Class for catalogs
    """
    def __init__(self,
                 path     = './',
                 sim_list = [],
                 catalog  = 'rit',
                 add_opts = {},
                 ):
        
        self.path     = path
        self.catalog  = catalog
        self.sim_list = sim_list

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
    
    def Wave(self, ID, add_opts={}):
        if self.catalog=='sxs':
            wave = Waveform_SXS(path=self.path, download=self.download, ID=ID, **add_opts)
                                #order='Extrapolated_N3.dir', ellmax=7)
        elif self.catalog=='rit':
            wave = Waveform_RIT(path=self.path, download=self.download, ID=ID, **add_opts)
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
                if any(wave.hlm[(2,2)]['A']>0.6):
                    print(f'Warning for {wave.metadata["name"]}')
                plt.plot(wave.u-tmrg, wave.hlm[(2,2)]['A'], c=colors[i])
        plt.show()
        pass




