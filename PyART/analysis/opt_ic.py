import os, json, random, time, copy
import numpy as np
from scipy import optimize 

from  .match import Matcher 
from ..models.teob import Waveform_EOB
from ..models.teob import CreateDict
from ..models.teob import PotentialMinimum
from ..utils       import utils as ut

class Optimizer(object):
    """
    Class to compute EOB initial data that minimize mismatch
    with reference waveform
    """
    def __init__(self,
                 ref_Waveform,

                 # option for dual annealing
                 kind_ic      = 'E0pph0', # implemented: e0f0, E0pph0
                 opt_seed     = 190521,
                 opt_maxfun   = 100,

                 # loop on different initial guesses (nested in bound-iters)
                 opt_max_iter = 1,     # max opt iters (i.e. using different initial guesses) if mm_thres is not reached
                 opt_good_mm  = 5e-3,  # interrupt opt-iters if mm is below this threshold

                 # Option for bounds and iterations
                 opt_bounds   = None,  # specify bounds (None or [[x1,x2],[y1,y2]] )
                 eps_initial  = 1e-2,  # initial bound eps, used if some opt bounds are not specified
                 eps_factor   = 2,     # increase-factor for eps at each eps-iter
                 eps_max_iter = 1,     # If true, iterate on eps-bounds
                 eps_bad_mm   = 0.1,   # if after opt_max_iter(s) we are still above this threshold, 
                                       # increase bound-eps (if eps_max_iter>1)
                 
                 # cache
                 use_matcher_cache = False, 

                 # json-output options
                 json_file    = None,  # JSON file with mm (must be consistent with current options). If None, do not print data 
                 overwrite    = False, # overwrite JSON with new mm-computation
                 
                 # other options
                 mm_settings  = None,  # options for Matcher (dictionary)
                 verbose      = True,
                 debug        = False, # debug plot
                 ):
        
        self.ref_Waveform = ref_Waveform
        self.opt_Waveform = None

        self.kind_ic      = kind_ic
        self.opt_seed     = opt_seed
        self.opt_maxfun   = opt_maxfun
        
        self.opt_max_iter = opt_max_iter
        self.opt_good_mm  = opt_good_mm

        self.opt_bounds   = opt_bounds
        self.eps_initial  = eps_initial
        self.eps_factor   = eps_factor
        self.eps_max_iter = eps_max_iter
        self.eps_bad_mm   = eps_bad_mm
        
        self.use_matcher_cache = use_matcher_cache

        self.json_file    = json_file
        self.overwrite    = overwrite
        self.verbose      = verbose
        
        # mismatch settings
        self.mm_settings = Matcher.__default_parameters__(0) 
        if isinstance(mm_settings, dict):
            for k in mm_settings:
                self.mm_settings[k] = mm_settings[k]
        if self.mm_settings['cut_longer'] and self.verbose:
            print("Warning: using the option 'cut_longer' during optimization should be avoided!")
        if not self.mm_settings['cut_second_waveform'] and self.verbose:
            print("Warning: using the option 'cut_second_waveform' during optimization is strongly suggested!")
        
        # set up things according to IC-kind
        if kind_ic=='e0f0':
            e0_nr = self.ref_Waveform.metadata['e0']
            if e0_nr is None or e0_nr>=1:
                if self.verbose: print('Invalid e0 in NR waveform! Overwriting with e0=0.5')
                self.ref_Waveform.metadata['e0'] = 0.5
            ic_keys = ['e0', 'f0']
        elif kind_ic=='E0pph0':
            ic_keys = ['E0byM', 'pph0']
        else:
            raise ValueError('Unknown IC kind: {kind_ic}')
        self.ic_keys = ic_keys
        
        # update bounds
        if self.opt_bounds is None:
            self.opt_bounds = [[None,None],[None,None]]
        self.__update_bounds(eps=eps_initial)
          
        if verbose:
            q    = ref_Waveform.metadata['q']
            chi1 = ref_Waveform.metadata['chi1z'] 
            chi2 = ref_Waveform.metadata['chi2z'] 
            flags_str = ''
            for flag in ref_Waveform.metadata['flags']:
                flags_str += flag + ', '
            flags_str = flags_str[:-2]
            print( '###########################################')
            print(f'###          Running Optimizer          ###')
            print( '###########################################\n')
            print(f'Reference waveform : {ref_Waveform.metadata["name"]}')
            print(f'(q, chi1z, chi2z)  : ({q:.2f}, {chi1:.2f}, {chi2:.2f})')
            print(f'binary type        : {flags_str}')
            print(f'Variables for ICs  : {ic_keys[0]}, {ic_keys[1]}')
            print(' ')

        mm_data  = self.load_or_create_mismatches()
        ref_name = self.ref_Waveform.metadata['name']
        
        opt_data      = None
        self.opt_data = opt_data

        run_optimization = True
        if ref_name in mm_data['mismatches']:
            opt_data = mm_data['mismatches'][ref_name]
            # TODO: commented this part. Seems useless. Double check 
#            opt_data['q']     = self.ref_Waveform.metadata['q']
#            opt_data['chi1x'] = self.ref_Waveform.metadata['chi1x']
#            opt_data['chi1y'] = self.ref_Waveform.metadata['chi1y']
#            opt_data['chi1z'] = self.ref_Waveform.metadata['chi1z']
#            opt_data['chi2x'] = self.ref_Waveform.metadata['chi2x']
#            opt_data['chi2y'] = self.ref_Waveform.metadata['chi2y']
#            opt_data['chi2z'] = self.ref_Waveform.metadata['chi2z']
#            mm_data['mismatches'][ref_name] = opt_data
#            with open(json_file, 'w') as file:
#                file.write(json.dumps(mm_data,indent=2))
            if not overwrite or opt_data['mm_opt']<eps_bad_mm:
                run_optimization = False
            if verbose: 
                print(f'Loading mismatch from {self.json_file}')
                print('Optimal ICs  : {:s}={:.5f}, {:s}:{:.5f}'.format(opt_data['kx'], opt_data['x_opt'],
                                                                  opt_data['ky'], opt_data['y_opt']))
                print('Original mm  : {:.3e}'.format(opt_data['mm0']))
                print('Optimized mm : {:.3e}\n'.format(opt_data['mm_opt']))
        
        if run_optimization:
            random.seed(self.opt_seed)
            dashes    = '-'*45
            asterisks = '*'*45
            
            eps = self.eps_initial
            
            t0 = time.perf_counter()
            # i-loop on different search bounds
            for i in range(1, self.eps_max_iter+1): 
                if self.eps_max_iter>1 and self.verbose:
                    print(f'\n{asterisks}\nSearch bounds (eps) iteration  #{i:d}\n{asterisks}')
                 
                # j-loop on different initial gueses 
                for j in range(1, self.opt_max_iter+1):
                    if self.verbose: print(f'{dashes}\nOptimization iteration #{j:d}\n{dashes}')
                    if i==1 and j==1 and opt_data is None: # if first iter of both loops
                        opt_data = self.optimize_mismatch(use_ref_guess=True)
                    else:
                        opt_data_new = self.optimize_mismatch(use_ref_guess=False)
                        if opt_data_new['mm_opt']<opt_data['mm_opt']:
                            opt_data = opt_data_new
                    # if we reached a nice mismatche, break loop on initial guesses
                    if opt_data['mm_opt']<=self.opt_good_mm:
                        break
                
                if opt_data['mm_opt']<=self.eps_bad_mm:
                    # if the mismatch is good according to eps-standard, then break
                    break
                
                elif i<self.eps_max_iter:
                    # otherwise, increase the bound search (if we are not at the last iter)
                    flat_old_bounds = [item for sublist in self.opt_bounds for item in sublist]
                    
                    self.opt_bounds = [ [None,None], [None,None] ]
                    eps *= self.eps_factor
                    self.__update_bounds(eps=eps)
                    flat_new_bounds = [item for sublist in self.opt_bounds for item in sublist]
                    print('\nIncreasing search bounds: [{:.3f},{:.3f}], [{:.3f},{:.3f}]'.format(*flat_old_bounds))
                    print('                  ----> : [{:.3f},{:.3f}], [{:.3f},{:.3f}]'.format(*flat_new_bounds))
                
                else:
                    mm_opt = opt_data['mm_opt']
                    print( '\n++++++++++++++++++++++++++++++++++++++')
                    print(f'+++  Reached eps_max_iter : {self.eps_max_iter:2d}     +++')
                    print(f'+++  mm_opt : {mm_opt:.2e} > {self.eps_bad_mm:.2e}  +++')
                    print( '++++++++++++++++++++++++++++++++++++++')
                    
            mm_data['mismatches'][ref_name] = opt_data
            
            if verbose:
                print('\n>> Best mismatch found : {:.3e}'.format(opt_data['mm_opt']))
                print(  '>> Total elapsed time  : {:.1f} s\n'.format(time.perf_counter()-t0))

            if json_file is not None: 
                self.save_mismatches(mm_data)
        self.opt_data = opt_data
        self.opt_Waveform = self.generate_EOB(ICs={self.ic_keys[0]:opt_data['x_opt'], 
                                                  self.ic_keys[1]:opt_data['y_opt']})
        
        if debug:
            self.mm_settings['debug'] = True
            self.match_against_ref(self.opt_Waveform)
        pass
    
    def __update_bounds(self, eps=1e-2):
        kx = self.ic_keys[0]
        ky = self.ic_keys[1]
        vx_ref = self.ref_Waveform.metadata[kx]
        vy_ref = self.ref_Waveform.metadata[ky]
        default_bounds = [ [vx_ref*(1-eps), vx_ref*(1+eps)],
                           [vy_ref*(1-eps), vy_ref*(1+eps)] ]
        for i in range(2):
            for j in range(2):
                if self.opt_bounds[i][j] is None:
                    self.opt_bounds[i][j] = default_bounds[i][j]
        pass

    def load_or_create_mismatches(self): 
        """
        Load mismatches data if the options of the
        json file stored is consistent with current ones.
        Otherwise, create a new dictionary (NOT a new json file)
        """
        # convert numpy array to lists to avoid issues with JSON writing/loading
        loc_mm_settings = copy.deepcopy(self.mm_settings)
        for k in loc_mm_settings:
            val = loc_mm_settings[k]
            if isinstance(val, np.ndarray):
                loc_mm_settings[k] = list(val)
        del loc_mm_settings['initial_frequency_mm'] # save this at sim-level
        del loc_mm_settings['final_frequency_mm']
        
        # options to store/read in JSON 
        options = {'opt_maxfun'   : self.opt_maxfun, 
                   'kind_ic'      : self.kind_ic,
                   'mm_settings'  : loc_mm_settings,
                  } 

        # check if file exists
        if self.json_file is not None and os.path.exists(self.json_file):
            # load mismatches dict
            with open(self.json_file, 'r') as file:
                json_data = json.loads(file.read())

            # fix list of list to list of tuples for 'modes' in json-data
            modes_list_of_list = json_data['options']['mm_settings']['modes']
            json_data['options']['mm_settings']['modes'] = [tuple(mode) for mode in modes_list_of_list]
            
            # check that the options are the same: 
            # 1) start by checking everything except mm_settings
            # 2) then check mm_settings
            dicts2check = [ [json_data['options'],                options               ],
                            [json_data['options']['mm_settings'], options['mm_settings']]
                          ]
            names = [ ['json', 'self'], ['mm_set-json', 'mm_set-self']  ]
            list_excluded_keys = [['mm_settings'], ['debug', 'initial_frequency_mm', 'final_frequency_mm']]
            for i in range(len(dicts2check)):
                dict1     = dicts2check[i][0]
                dict2     = dicts2check[i][1]
                name1     = names[i][0]
                name2     = names[i][1]
                excl_keys = list_excluded_keys[i]
                if not ut.are_dictionaries_equal(dict1, dict2, excluded_keys=excl_keys, verbose=True):
                    ut.print_dict_comparison(dict1, dict2, excluded_keys=excl_keys, dict1_name=name1, dict2_name=name2)
                    raise RuntimeError('The options in the json-file are different w.r.t. the currient ones. Exit.')
            
            data = json_data
        else:
            # create mismatches dict
            data = {'options':options, 'mismatches':{}}
        return data
    
    def save_mismatches(self, data, verbose=None, json_file=None, overwrite=None):
        if verbose   is None: verbose   = self.verbose
        if overwrite is None: overwrite = self.overwrite
        if json_file is None: json_file = self.json_file
        if json_file is None: # i.e., if self.json_file is None
            pass 
        sim_name = self.ref_Waveform.metadata['name']
        creating_new_file = True 
        if os.path.exists(json_file) and not overwrite:
            with open(json_file, 'r') as file:
                json_data = json.loads(file.read())
            creating_new_file = False
            if sim_name in json_data['mismatches']:
                print(f'   ---> File {json_file} alreay exists and contains {sim_name}, but overwriting is off.')
                json_file = json_file.replace('.json', '_new.json')
                print(f'   ---> writing on file: {json_file}')
                creating_new_file = True
        with open(json_file, 'w') as file:
            file.write(json.dumps(data,indent=2))
        if verbose: 
            action = 'Created' if creating_new_file else 'Updated'
            print(f'{action} {json_file}\n')
        pass

    def generate_EOB(self, ICs={'f0':None, 'e0':None}):
        ref_meta = self.ref_Waveform.metadata
        # define subset of info to generate EOB waveform
        keys_to_use =  ['M', 'q', 'chi1x', 'chi1y', 'chi1z', 'chi2x', 'chi2y', 'chi2z']
        sub_meta = {key: ref_meta[key] for key in keys_to_use}
         
        # kind-specific input
        def return_IC(key):
            if key not in ICs:
                raise RuntimeError('Specify f0 in ICs!')
            elif ICs[key] is None:
                return ref_meta[key]
            else:
                return ICs[key]
        
        if self.kind_ic=='quasi-circ': # here only for testing
            sub_meta['f0'] = return_IC('f0')
        
        elif self.kind_ic=='e0f0':
            sub_meta['ecc'] = return_IC('e0')
            sub_meta['f0']  = return_IC('f0')

        elif self.kind_ic=='E0pph0':
            sub_meta['H_hyp'] = return_IC('E0byM')
            sub_meta['J_hyp'] = return_IC('pph0')
            sub_meta['r_hyp'] = None # computed in CreateDict

        else: 
            raise ValueError(f'Unknown kind of ICs: {kind}')
       
        sub_meta['ode_tmax'] = 2e+4
        # return generated EOB waveform 
        try:
            pars        = CreateDict(**sub_meta)
            eob_wave    = Waveform_EOB(pars=pars)
            #eob_wave._u = eob_wave.u#-eob_wave.u[0]
        except Exception as e:
            # FIXME: no error msg is a little bit criminal
            #print(f'Error occured in EOB wave generation:\n{e}')
            eob_wave = None
        return eob_wave
    
    def generate_opt_EOB(self, opt_data=None, verbose=None):
        if verbose is  None: verbose  = self.verbose
        if opt_data is None: opt_data = self.opt_data
        if opt_data is None:
            if verbose: print('Optimal ICs not found! Returning None')
            opt_Waveform = None
        else:
            kx = self.ic_keys[0]
            ky = self.ic_keys[1]
            opt_Waveform = self.generate_EOB(ICs={kx:opt_data['x_opt'], 
                                                  ky:opt_data['y_opt']})
        return opt_Waveform

    def match_against_ref(self, eob_Waveform, verbose=None, iter_loop=False, return_matcher=False, cache={}):
        if verbose is None: verbose = self.verbose
        if eob_Waveform is not None:
            matcher = Matcher(self.ref_Waveform, eob_Waveform, pre_align=False,
                              settings=self.mm_settings, cache=cache)
            mm = matcher.mismatch
        else:
            matcher = None
            mm = 1.0
        if verbose and iter_loop:
            self.annealing_counter += 1
            print( '  >> mismatch - iter  : {:.3e} - {:3d}'.format(mm, self.annealing_counter), end='\r')
        if return_matcher:
            return mm, matcher
        else:
            return mm
    
    def __func_to_minimize(self, vxy, verbose=None, cache={}):
        if verbose is None: verbose = self.verbose
        kx = self.ic_keys[0]
        ky = self.ic_keys[1]
        eob_Waveform = self.generate_EOB(ICs={kx:vxy[0], ky:vxy[1]})
        if eob_Waveform is not None:
            mm = self.match_against_ref(eob_Waveform, verbose=self.verbose, iter_loop=True, cache=cache)
        else:
            if self.kind_ic=='E0pph0':
                pph0 = vxy[1]
                ref_meta = self.ref_Waveform.metadata
                q    = ref_meta['q']
                chi1 = ref_meta['chi1z']
                chi2 = ref_meta['chi2z']
                rvec = np.linspace(2,20,num=200)
                Vmin = PotentialMinimum(rvec,pph0,q,chi1,chi2)
                dV   = Vmin-vxy[0]
            else:
                dV = 0
            mm = 1 + dV
        return mm

    def optimize_mismatch(self, use_ref_guess=True, verbose=None):
        if verbose is None: verbose = self.verbose
        kx = self.ic_keys[0]
        ky = self.ic_keys[1]
        vx_ref  = self.ref_Waveform.metadata[kx]
        vy_ref  = self.ref_Waveform.metadata[ky]
         
        bounds = self.opt_bounds
        if vx_ref<bounds[0][0] and vx_ref>bounds[0][1]:
            print('Warning! Reference value for {:s} is outside searching interval: [{:.2e},{:.2e}]'.format(ks,  bounds[0][0],  bounds[0][1]))
        if vy_ref<bounds[1][0] and vy_ref>bounds[1][1]:
            print('Warning! Reference value for {:s} is outside searching interval: [{:.2e},{:.2e}]'.format(ks,  bounds[1][0],  bounds[1][1]))
        
        if use_ref_guess:
            vxy0 = np.array([vx_ref,vy_ref])
        else:
            vx0  = random.uniform(bounds[0][0], bounds[0][1])
            vy0  = random.uniform(bounds[1][0], bounds[1][1])
            vxy0 = np.array([vx0, vy0])
        
        mm0, matcher0 = self.match_against_ref(self.generate_EOB(ICs={kx:vxy0[0], ky:vxy0[1]}),
                                               iter_loop=False, return_matcher=True)
        if verbose:
            print(f'Original  mismatch    : {mm0:.3e}')
            print( 'Optimization interval : {:5s} in [{:.2e}, {:.2e}]'.format(kx, bounds[0][0], bounds[0][1]))
            print( '                      : {:5s} in [{:.2e}, {:.2e}]'.format(ky, bounds[1][0], bounds[1][1]))
            print(f'Initial guess         : {kx:5s} : {vxy0[0]:.15f}')
            print(f'                        {ky:5s} : {vxy0[1]:.15f}')
        
        if self.use_matcher_cache:
            if matcher0 is None:
                if verbose: print('+++ First mm-computation failed! Not using cache +++')
                cache = {}
            else:
                cache = {'h1f':matcher0.h1f, 'M':matcher0.settings['M']}
        else:
            cache = {}

        f = lambda vxy : self.__func_to_minimize(vxy, verbose=verbose, cache=cache)

        self.annealing_counter = 1

        t0_annealing = time.perf_counter()
        opt_result = optimize.dual_annealing(f, maxfun=self.opt_maxfun, 
                                                seed=self.opt_seed, x0=vxy0,
                                                bounds=bounds)
        opt_pars     = opt_result['x']
        x_opt, y_opt = opt_pars[0], opt_pars[1]
        mm_opt = opt_result['fun']
        
        if verbose:
            print( '  >> mismatch - iter  : {:.3e} - {:3d}'.format(mm_opt, self.annealing_counter), end='\r')
            print(f'Optimized mismatch    : {mm_opt:.3e}')
            print(f'Optimal ICs           : {kx:5s} : {x_opt:.15f}')
            print(f'                        {ky:5s} : {y_opt:.15f}')
            print( 'Annealing time        : {:.1f} s'.format(time.perf_counter()-t0_annealing))
        
        opt_eob = self.generate_opt_EOB(opt_data={'x_opt':x_opt, 'y_opt':y_opt})
        if opt_eob is not None:
            r0_eob = opt_eob.dyn['r'][0]
        else:
            r0_eob = None

        # fixing EOB IDs according to cut-option in matcher
#        if opt_eob is not None and self.mm_settings['cut']:
#            print(self.mm_settings)
#            u_ref = self.ref_Waveform.u
#            u_eob = opt_eob.u
#            umrg_ref,_,_,_ = self.ref_Waveform.find_max()#-u_ref[0]
#            umrg_eob,_,_,_ = opt_eob.find_max()#-u_eob[0]
#            
#            tmrg_ref,_,_,_ = self.ref_Waveform.find_max()-u_ref[0]
#            tmrg_eob,_,_,_ = opt_eob.find_max()-u_eob[0]
#            DeltaT = tmrg_eob-tmrg_ref
#            import matplotlib.pyplot as plt # FIXME: only for debug, to remove
#            plt.figure
#            plt.plot(u_ref-umrg_ref, self.ref_Waveform.hlm[(2,2)]['real'], label='rn ref')
#            plt.plot(u_eob-umrg_eob, opt_eob.hlm[(2,2)]['real'], label='eob')
#            
#            if DeltaT>0:
#                opt_eob.cut(DeltaT)
#                plt.plot(opt_eob.u-umrg_eob, opt_eob.hlm[(2,2)]['real'], label='eob', ls='--')
#            plt.legend()
#            plt.show()

        opt_data = { 
                    # store also some attributes, just for convenience 
                    'q'            : self.ref_Waveform.metadata['q'],
                    'chi1x'        : self.ref_Waveform.metadata['chi1x'],
                    'chi1y'        : self.ref_Waveform.metadata['chi1y'],
                    'chi1z'        : self.ref_Waveform.metadata['chi1z'],
                    'chi2x'        : self.ref_Waveform.metadata['chi2x'],
                    'chi2y'        : self.ref_Waveform.metadata['chi2y'],
                    'chi2z'        : self.ref_Waveform.metadata['chi2z'],
                    'initial_frequency_mm' : self.mm_settings['initial_frequency_mm'],
                    'final_frequency_mm'   : self.mm_settings['final_frequency_mm'],
                    'opt_seed'     : self.opt_seed,
                    'opt_max_iter' : self.opt_max_iter,
                    'opt_good_mm'  : self.opt_good_mm,
                    'eps_initial'  : self.eps_initial,
                    'eps_max_iter' : self.eps_max_iter,
                    'eps_bad_mm'   : self.eps_bad_mm,
                    'eps_initial'  : self.eps_initial,
                    'eps_factor'   : self.eps_factor,
                    # optimization results
                    'kx'           : kx, 
                    'ky'           : ky,
                    'x_ref'        : vx_ref, 
                    'y_ref'        : vy_ref,
                    'bounds'       : bounds, 
                    'x0'           : vxy0[0], 
                    'y0'           : vxy0[1], 
                    'mm0'          : mm0,
                    'r0_eob'       : r0_eob,
                    'x_opt'        : x_opt, 
                    'y_opt'        : y_opt,
                    'mm_opt'       : mm_opt, 
                    }
        
        return opt_data




