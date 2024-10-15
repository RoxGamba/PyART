import os, json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from .sxs import Waveform_SXS
from .rit import Waveform_RIT
from .icc import Waveform_ICC

from ..analysis.match  import Matcher
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
                 json_file = None, # file for mismatches. If None, use default name 
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
        #self.mm_data   = self.read_mismatches_json() 

        if len(self.sim_list)<1:
            raise RuntimeError('Empty list of simulations')
          
        self.data = {}
        for ID in sim_list:
            try: 
                wave = self.get_Waveform(ID, add_opts)
                name = wave.metadata['name']
                self.data[name] = {'Waveform':wave, 'Optimizer':None}
            except Exception as e:
                print(f'Issues with {ID:04}: {e}')
        self.nsims = len(self.data)
        
        pass
    
    def get_Waveform(self, ID, add_opts={}, verbose=None):
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
        for i, name in enumerate(self.data):
            wave = self.data[name]['Waveform']
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
        for name in self.data:
            self.data[name]['Optimizer'] = Optimizer(self.data[name]['Waveform'], **optimizer_opts)
        pass
  
    def plot_mm_vs_M(self, mass_min=100, mass_max=200, N=20, cmap='rainbow'):
        mycmap = plt.get_cmap(cmap)
        colors = mycmap(np.linspace(0,1,self.nsims))
        masses = np.linspace(mass_min, mass_max, num=N)
        plt.figure()
        for i, name in enumerate(self.data):
            print(f'mm for: {name}')
            mm = masses*0
            mm_settings = self.data[name]['Optimizer'].mm_settings
            eob = self.data[name]['Optimizer'].generate_opt_EOB()
            nr  = self.data[name]['Waveform']
            for j, M in enumerate(masses):
                mm_settings['M'] = M 
                matcher = Matcher(nr, eob, pre_align=False, settings=mm_settings)
                mm[j] = matcher.mismatch
            plt.plot(masses, mm, label=name, c=colors[i])
        plt.yscale('log')
        #plt.legend(ncol=3)
        plt.grid()
        plt.show()
        return


