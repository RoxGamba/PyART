import sys, os, json, matplotlib, time, copy
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from multiprocessing import Process

from ..analysis.match  import Matcher
from ..analysis.opt_ic import Optimizer
from ..models.teob import CreateDict
from ..models.teob import Waveform_EOB, get_pph_lso

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
            date      = datetime.now()
            fmt_date  = date.strftime("%Y%m%d")
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
            from .sxs import Waveform_SXS
            wave = Waveform_SXS(path=self.path, ID=ID, **add_opts)
            
        elif self.catalog=='rit':
            from .rit import Waveform_RIT
            wave = Waveform_RIT(path=self.path, ID=ID, **add_opts)
        
        elif self.catalog=='icc':
            from .icc import Waveform_ICC
            wave = Waveform_ICC(path=self.path, ID=ID, **add_opts)

        elif self.catalog=='core':
            from .core import Waveform_CoRe
            wave = Waveform_CoRe(path=self.path, ID=ID, **add_opts)
        
        elif self.catalog=='grahyp':
            from .gra_hyp import Waveform_GRAHyp
            wave = Waveform_GRAHyp(path=self.path, ID=ID, **add_opts)

        else:
            raise ValueError(f'Unknown catalog: {self.catalog}')
        return wave
    
    def get_model_waveform(self, name, add_opts={}, verbose=None):
        """
        Compute the waveform with the model corresponding to a catalog ID
        """
        if self.data[name]['Optimizer'] is not None:
            optimizer = self.data[name]['Optimizer']
            kys   = optimizer.opt_vars
            x_opt = optimizer.opt_data[kys[0]+'_opt']
            y_opt = optimizer.opt_data[kys[1]+'_opt']
            ICs = {kys[0]:x_opt, kys[1]:y_opt}
            eob = optimizer.generate_EOB(ICs=ICs)
        else:
            # Create a mock optimizer
            params  = self.data[name]['Waveform'].metadata
            cd_args = CreateDict.__code__.co_varnames
            newpars = {ky:val for ky,val in params.items() if ky in cd_args} 
            params  = CreateDict(**newpars)
            # have a slightly lower f0
            params['initial_frequency'] = 0.95*params['initial_frequency']
            eob    = Waveform_EOB(params)
        return eob

    def plot_waves(self, cmap='rainbow', legend=False):
        mycmap = plt.get_cmap(cmap)
        colors = mycmap(np.linspace(0,1,self.nsims))
        plt.figure(figsize=(8,6))
        for i, name in enumerate(self.data):
            wave = self.data[name]['Waveform']
            label = wave.metadata['name']
            if (2,2) in wave.hlm:
                try:
                    tmrg,_,_,_ = wave.find_max()
                    plt.plot(wave.u-tmrg, wave.hlm[(2,2)]['A'], c=colors[i], label=label)
                except:
                    plt.plot(wave.u, wave.hlm[(2,2)]['A'], c=colors[i], label=label)
        if legend:
            plt.legend()
        plt.show()
        pass
    
    def optimize_mismatches_batch(self, batch, optimizer_opts={}, verbose=None):
        if verbose is None: verbose = self.verbose
        optimizer_opts['verbose']   = verbose
        # run optimizer on all the waveforms
        for name in batch:
            # store on JSON
            Optimizer(self.data[name]['Waveform'], **optimizer_opts)
        pass
    
    def process_with_redirect(self, process_id, nproc, opts={}):
        """
        If we are running in parallel, use log files. 
        Otherwise, use stdout
        """
        if nproc>1:
            now_str  = datetime.now().strftime('%Y%m%d_%H-%M-%S')
            log_file = f'{now_str:s}_cataloger_process_{process_id}.log'
            print(f'Logfile #{process_id:d}: {log_file}')
            with open(log_file, 'w') as file:
                sys.stdout = file
                sys.stderr = file
                try:
                    self.optimize_mismatches_batch(**opts)
                finally:
                    sys.stdout = sys.__stdout__ 
                    sys.stderr = sys.__stderr__
        else:
            self.optimize_mismatches_batch(**opts)
        pass

    def optimize_mismatches(self, optimizer_opts={}, verbose=None, ranges={'pph0':[1,10]}, nproc=1):
        optimizer_opts['json_file'] = self.json_file
        optimizer_opts['verbose']   = True
        
        subset  = self.find_subset(ranges)
        nsubset = len(subset)
        if nsubset<nproc:
            print(f'Warning! More processes than configurations, reducing nproc to {nsubset}')
            nproc = nsubset 
        
        if nproc==1 or nsubset==1:
            self.optimize_mismatches_batch(batch=subset, optimizer_opts=optimizer_opts) 
        
        else:
            batches    = []
            batch_size = len(subset) // nproc
            remainder  = len(subset)  % nproc
            start = 0
            for i in range(nproc):
                current_size = batch_size + 1 if i < remainder else batch_size
                batches.append(subset[start:start + current_size])
                start += current_size
            
            for i in range(len(batches)):
                print(batches[i][0], batches[i][-1])
             
            processes = []
            json_tmp_list = []
            print('Redirecting output in logfiles')
            for i in range(nproc):
                # get name for temporary JSON file 
                json_file_tmp = optimizer_opts['json_file']
                json_file_tmp = json_file_tmp.replace('.json', f'_{i:d}.json')
                json_tmp_list.append(json_file_tmp)
                
                # copy json_file (if exists) in temp file. Otherwise temp created by Optimizer  
                if os.path.exists(self.json_file):
                    with open(self.json_file, 'r') as source:
                        with open(json_file_tmp, 'w') as new: 
                            new.write(source.read())

                # create optimizer_opts with temporary JSON file
                optimizer_opts_tmp = copy.deepcopy(optimizer_opts)
                optimizer_opts_tmp['json_file'] = json_file_tmp
                # launch process
                process = Process(target=self.process_with_redirect,
                                  kwargs={'process_id':i,
                                          'nproc':nproc,
                                          'opts':{
                                              'batch':batches[i],
                                              'optimizer_opts':optimizer_opts_tmp, 
                                              'verbose':verbose}
                                              }
                                             )
                processes.append(process)
                process.start()
            
            # wait for processes to finish 
            for process in processes:
                process.join()
                
            # info stored in the temporary JSONs, collect info
            # 1) load original json file if it exists, otherwise load first temp file
            if os.path.exists(self.json_file):
                json_base = self.json_file
            else:
                json_base = json_tmp_list[0]
            with open(json_base, 'r') as file:
                json_data = json.loads(file.read())
            # 2) add info from temporary json, delete temporary json 
            for json_tmp in json_tmp_list:
                with open(json_tmp, 'r') as file:
                    json_data_tmp = json.loads(file.read())
                mismatches = json_data_tmp['mismatches']
                for key in mismatches:
                    if key not in json_data:
                        json_data['mismatches'][key] = mismatches[key]
                os.remove(json_tmp)
            with open(self.json_file, 'w') as file:
                file.write(json.dumps(json_data,indent=2))
            
        # read collated json
        optimizer_opts['json_file'] = self.json_file
        optimizer_opts['verbose']   = False
        optimizer_opts['overwrite'] = False
        for name in subset:
            self.data[name]['Optimizer'] = Optimizer(self.data[name]['Waveform'], **optimizer_opts)
        pass

    def __is_in_valid_range(self, name, ranges,):
        """
        Check if a certain waveform is in the specified
        ranges (for example: ranges={'pph0':[1,10]})
        Ranges can also contain 'check_pph_lso'.
        If check_pph_lso is True, then check also if the pph of 
        the simulation is above the LSO value
        """
        meta  = self.data[name]['Waveform'].metadata
        for key in ranges:
            if key=='check_pph_lso':
                q     = meta['q']
                chi1z = meta['chi1z']
                chi2z = meta['chi2z']
                nu    = meta['nu']
                m1    = q/(1+q)
                m2    = 1/(1+q)
                a0    = m1*chi1z + m2*chi2z 
                pph_lso = get_pph_lso(nu,a0)
                if meta['pph0']<pph_lso:
                    return False
                else:
                    continue
            elif key not in meta:
                raise ValueError(f'{key} is not a valid option in ranges!')
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

    def mm_at_M(self, name, M, mm_settings = None):
        
        eob = self.get_model_waveform(name)
        print(eob.dyn['Pphi'][0])
        print(eob.dyn['E'][0])
        nr  = self.data[name]['Waveform']
        mm_settings['M'] = M 
        matcher   = Matcher(nr, eob, settings=mm_settings)
        return matcher.mismatch
    
    def mm_vs_M(self, 
                     mass_min   = 100, 
                     mass_max   = 200, 
                     N          = 20,
                     cmap       = 'jet', 
                     json_load  = None, # load if not None
                     json_save  = None, # save if not None
                     mm_settings= None, 
                     ranges     = {'pph0':[1,10]},
                     cmap_var   = 'E0byM',
                     hlines     = [],
                     figname    = None, # save if not None
                     ):

        if figname is not None: savepng = True

        # select waveforms and get colors
        subset       = self.find_subset(ranges=ranges)
        colors_dict  = self.get_colors_for_subset(subset,cmap_var=cmap_var, cmap_name=cmap)
        colors       = colors_dict['colors']
        cmap_indices = colors_dict['indices']
        cmap_range   = colors_dict['range']

        masses = np.linspace(mass_min, mass_max, num=N)    
        
        # start setting up the JSON file
        if isinstance(json_load, str) and os.path.exists(json_load):
            with open(json_load, 'r') as file:
                mm_data = json.load(file) 
        
        else:
            mm_data = {}
            mm_data['masses']     = list(masses)
            mm_data['options']    = {}
            mm_data['mismatches'] = {}
            mm_data['options']['mm_settings'] = mm_settings
        
        vrs = ['q', 'chi1x', 'chi1y', 'chi1z', 'chi2x', 'chi2y', 'chi2z', 'E0byM', 'pph0']

        for i, name in enumerate(subset):
            if name in mm_data['mismatches']:
                continue
            
            if mm_settings is None:
                if self.data[name]['Optimizer'] is not None:
                    if self.verbose: print(f'Using settings from {name} optimizer')
                    mm_settings = self.data[name]['Optimizer'].mm_settings
                else:
                    raise ValueError('No mm_settings provided and no Optimizer available')
            print(f'mm for: {name}')
            mm = masses*0
            eob = self.get_model_waveform(name)
            nr  = self.data[name]['Waveform']
                            
            for j, M in enumerate(masses):
                mm_settings['M'] = M 
                matcher   = Matcher(nr, eob, settings=mm_settings)
                mm[j]     = matcher.mismatch
                
            mm_data['mismatches'][name] = {}
            mm_data['mismatches'][name]['mm_vs_M'] = list(mm)
            mm_data['mismatches'][name]['mm_max']  = max(mm)
            mm_data['mismatches'][name]['mm_min']  = min(mm)

            for vr in vrs:
                mm_data['mismatches'][name][vr] = self.quantity_from_dataset(name, vr)

        _, ax = plt.subplots(1,1,figsize=(8,6))
        for i, name in enumerate(subset):
            cidx = cmap_indices[i]
            ax.plot(masses, mm_data['mismatches'][name]['mm_vs_M'], label=name, c=colors[cidx], lw=0.6)
        
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

        if figname is None:
            figname = f'mismatches_{self.catalog}_{cmap_var}.png'
            plt.savefig(figname,dpi=200,bbox_inches='tight')
            print(f'Figure saved: {figname}')
        plt.show()
        
        if isinstance(json_save, str):
            with open(json_save, 'w') as file:
                file.write(json.dumps(mm_data,indent=2))
                print(f'JSON file saved: {json_save}')

        return


