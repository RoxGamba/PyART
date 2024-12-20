import os, json, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from .sxs import Waveform_SXS
from .rit import Waveform_RIT
from .icc import Waveform_ICC

from ..analysis.match  import Matcher
from ..analysis.opt_ic import Optimizer

matplotlib.rc('text', usetex=True)

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
        plt.figure(figsize=(8,6))
        for i, name in enumerate(self.data):
            wave = self.data[name]['Waveform']
            if (2,2) in wave.hlm:
                tmrg,Amrg,_,_ = wave.find_max()
                plt.plot(wave.u-tmrg, wave.hlm[(2,2)]['A'], c=colors[i])
        plt.show()
        pass
    
    def optimize_mismatches(self, optimizer_opts={}, verbose=None, ranges={'pph0':[1,10]}):
        if verbose is None: verbose = self.verbose
        # set some options according to class-instance
        optimizer_opts['json_file'] = self.json_file
        optimizer_opts['verbose']   = verbose
        # run optimizer on all the waveforms
        subset = self.find_subset(ranges)
        for name in subset:
            self.data[name]['Optimizer'] = Optimizer(self.data[name]['Waveform'], **optimizer_opts)
        pass
    
    def __is_in_valid_range(self, name, ranges):
        """
        Check if a certain waveform is in the specified
        ranges (for example: ranges={'pph0':[1,10]})
        """
        meta  = self.data[name]['Waveform'].metadata
        for key in ranges:
            if key not in meta:
                raise ValueError(f'{key} is not a metadata entry!')
            x = meta[key]
            if x<ranges[key][0] or x>ranges[key][1]:
              return False
        return True
    
    def find_subset(self, ranges):
        subset   = []
        for name in self.data:
            if self.__is_in_valid_range(name, ranges=ranges):
                subset.append(name)
        return subset 
    
    def quantity_from_dataset(self, name, variable):
        meta = self.data[name]['Waveform'].metadata
        if variable=='chiz_eff':
            chi1z = meta['chi1z']
            chi2z = meta['chi2z']
            m1    = meta['m1']
            m2    = meta['m2']
            x     = (m1*chi1z + m2*chi2z)/(m1+m2)
        elif variable=='mm_opt' or variable=='mm0':
            x = self.data[name]['Optimizer'].opt_data[variable]  
        elif variable in meta:
            chi1z = meta['chi1z']
            x = meta[variable]
        else:
            x = None
        return x
    
    def tex_label_from_key(self, key):
        tex = {'chiz_eff': r'$\chi_{\rm eff}$',
               'pph0'    : r'$p_\varphi^0$',
               'E0byM'   : r'$E_0/M$',
               'mm_opt'  : r'$\bar{\cal F}$',
               }
        if key in tex:
            return tex[key]
        else:
            return key
    
    def get_colors_for_subset(self, subset, cmap_var, cmap_name='jet'):
        cmap_min = None
        cmap_max = None
        for i, name in enumerate(subset):
            x = self.quantity_from_dataset(name, cmap_var)
            if x is not None:
                if cmap_min is None or x<cmap_min:
                    cmap_min = x
                if cmap_max is None or x>cmap_max:
                    cmap_max = x
        N = len(subset)
        cmap   = plt.get_cmap(cmap_name)
        colors = cmap(np.linspace(0,1,N))
        crange = [cmap_min, cmap_max] 
        if abs(cmap_min-cmap_max)>1e-12:
            indices = []
            for name in subset:
                x = self.quantity_from_dataset(name, cmap_var)
                idx = round( (N-1)*(x-cmap_min)/(cmap_max-cmap_min) )
                indices.append(idx)
        else:
            indices = [int(N/2) for i in subset]
        out = {'colors':colors, 'indices':indices, 
                'cmap':cmap, 'range':crange}
        return out

    def plot_colorbar(self, xvar='pph0', yvar='mm_opt', cvar='E0byM', 
                            ranges={'pph0':[1,10]}, cmap='jet',
                            hlines=[],yscale=True):
        subset = self.find_subset(ranges=ranges)
        N = len(subset)
        X = np.zeros((N,1))
        Y = np.zeros((N,1))
        C = np.zeros((N,1))
        for i, name in enumerate(subset):
            X[i] = self.quantity_from_dataset(name, xvar) 
            Y[i] = self.quantity_from_dataset(name, yvar) 
            C[i] = self.quantity_from_dataset(name, cvar) 
        fontsize = 20
        plt.figure(figsize=(8,6))
        plt.scatter(X,Y,c=C, cmap=cmap)
        cbar = plt.colorbar()
        plt.xlabel(self.tex_label_from_key(xvar), fontsize=fontsize)
        plt.ylabel(self.tex_label_from_key(yvar), fontsize=fontsize)
        cbar.set_label(self.tex_label_from_key(cvar), fontsize=fontsize)
        
        styles  = ['-', '--', '-.']
        nstyles = len(styles)
        for i, hline in enumerate(hlines):
            plt.axhline(hline, lw=2.0, ls=styles[i%nstyles], c='k')
        if yscale: 
            plt.yscale('log') 
            plt.grid()
        plt.show()
        return         

    def plot_mm_vs_M(self, 
                     mass_min   = 100, 
                     mass_max   = 200, 
                     N          = 20, 
                     cmap       = 'jet', 
                     ranges     = {'pph0':[1,10]},
                     cmap_var   = 'E0byM',
                     hlines     = [],
                     savepng    = True,
                     figname    = None,
                     ):
         
        # select waveforms and get colors
        subset       = self.find_subset(ranges=ranges)
        colors_dict  = self.get_colors_for_subset(subset,cmap_var=cmap_var, cmap_name=cmap)
        colors       = colors_dict['colors']
        cmap_indices = colors_dict['indices']
        cmap_range   = colors_dict['range']

        masses = np.linspace(mass_min, mass_max, num=N)

        fig, ax = plt.subplots(1,1,figsize=(8,6))
        for i, name in enumerate(subset):
            print(f'mm for: {name}')
            mm = masses*0
            mm_settings = self.data[name]['Optimizer'].mm_settings
            eob = self.data[name]['Optimizer'].generate_opt_EOB()
            nr  = self.data[name]['Waveform']
            pph0_nr = nr.metadata['pph0']
            for j, M in enumerate(masses):
                mm_settings['M'] = M 
                matcher = Matcher(nr, eob, pre_align=False, settings=mm_settings)
                mm[j] = matcher.mismatch
            cidx = cmap_indices[i]
            ax.plot(masses, mm, label=name, c=colors[cidx], lw=0.6)
        
        styles  = ['-', '--', '-.']
        nstyles = len(styles)
        for i, hline in enumerate(hlines):
            ax.axhline(hline, lw=2.0, ls=styles[i%nstyles], c='k')

        cnorm = plt.Normalize(*cmap_range)
        sm    = plt.cm.ScalarMappable(norm=cnorm,cmap=colors_dict['cmap'])
        cbar  = plt.colorbar(sm,ax=ax)
        
        ax.set_xlim(mass_min, mass_max)
        
        cbar_label = self.tex_label_from_key(cmap_var)
        cbar.set_label(cbar_label,         fontsize=20)
        ax.set_xlabel(r'$M [M_\odot]$',    fontsize=20)
        ax.set_ylabel(r'${\bar{\cal F}}$', fontsize=20)
        
        plt.yscale('log')
        plt.grid()

        if savepng:
            if figname is None:
                figname = f'mismatches_{self.catalog}_{cmap_var}.png'
            plt.savefig(figname,dpi=200,bbox_inches='tight')
            print(f'Figure saved: {figname}')
        plt.show()
        return


