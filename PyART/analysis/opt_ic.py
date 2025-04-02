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
    with reference waveform.
    """
    def __init__(self,
                 ref_Waveform,

                 # option for dual annealing
                 kind_ic      = 'E0pph0',           # kind of ICs to optimize over
                 vrs          = ['H_hyp', 'j_hyp'], # variables to optimize over, default is ['H_hyp', 'j_hyp']
                 map_function = None,               # function to map variables to EOB parameters
                 use_nqc      = True,
                 r0_eob       = None,               # use a fixed value for r_hyp in EOB model, if None it will be computed by TEOB

                 # model-specific options
                 model_opts   = {},

                 # loop on different initial guesses (nested in bound-iters)
                 opt_max_iter = 1,     # max opt iters (i.e. using different initial guesses) if mm_thres is not reached
                 opt_good_mm  = 5e-3,  # interrupt opt-iters if mm is below this threshold

                 # Option for bounds and iterations on bounds
                 opt_bounds   = None,  # specify bounds (None or {'key1': [x1,x2], 'key2': [y1,None], ...} )
                 bounds_iter  = {},    # options to use when iterating over bounds 
                 
                 # minimizer options
                 minimizer = {'kind': 'dual_annealing'},
                 
                 # cache
                 use_matcher_cache = False, 

                 # json-output options
                 json_file    = None,  # JSON file with mm (must be consistent with current options). If None, do not print data 
                 overwrite    = False, # overwrite JSON with new mm-computation
                 json_save_dyn = False, # save dynamics in JSON 
                 # other options
                 mm_settings  = None,  # options for Matcher (dictionary)
                 verbose      = True,
                 debug        = False, # debug plot
                 ):
        
        self.ref_Waveform = ref_Waveform
        self.opt_Waveform = None

        self.kind_ic      = kind_ic
        self.use_nqc      = use_nqc
        self.r0_eob       = r0_eob
        self.model_opts   = model_opts
        
        self.opt_max_iter = opt_max_iter
        self.opt_good_mm  = opt_good_mm
        self.opt_data     = None

        self.opt_bounds   = opt_bounds
        
        self.use_matcher_cache = use_matcher_cache

        self.json_file     = json_file
        self.json_save_dyn = json_save_dyn
        self.overwrite     = overwrite
        self.verbose       = verbose
        self.debug         = debug

        # mismatch settings
        self.mm_settings = Matcher.__default_parameters__(0) 
        if isinstance(mm_settings, dict):
            for k in mm_settings:
                self.mm_settings[k] = mm_settings[k]
        if self.mm_settings['cut_longer'] and self.verbose:
            print("Warning: using the option 'cut_longer' during optimization should be avoided!")
        if not self.mm_settings['cut_second_waveform'] and self.verbose:
            print("Warning: using the option 'cut_second_waveform' during optimization is strongly suggested!")
                          
        # decide IC vars based on kind_ic
        self.__set_variables(vrs)
        if map_function is not None:
            if self.map_function is None:
                self.map_function = map_function
            else:
                print('Warning: map_function is not None, but kind_ic is not "choose"')
                print('         user-input map_function will be ignored.')            

        if self.opt_bounds is None:
            self.opt_bounds = {var: [None, None] for var in self.opt_vars}
        # update bounds iterator
        self.__bounds_iter_defaults__()
        self.bounds_iter = {**self.bounds_iter, **bounds_iter}
        # update bounds
        self.__update_bounds(eps=self.bounds_iter['eps_initial'])

        # set minimizer
        self.__minimizer__defaults__()
        self.minimizer = {**self.minimizer, **minimizer}
        self.annealing_counter = 0
        if minimizer['kind'] == 'dynesty':
            self.minimize = self.__minimize__dynesty__
        elif minimizer['kind'] == 'dual_annealing':
            self.minimize = self.__minimize_annealing_
        elif minimizer['kind'] == 'differential_evolution':
            self.minimize = self.__minimize_differential_evo_
        else:
            raise ValueError(f'Unknown minimizer kind: {minimizer["kind"]}')

        if verbose:
            q    = ref_Waveform.metadata['q']
            chi1 = ref_Waveform.metadata['chi1z'] 
            chi2 = ref_Waveform.metadata['chi2z'] 
            flags_str = ''
            if 'flags' in ref_Waveform.metadata:
                for flag in ref_Waveform.metadata['flags']:
                    flags_str += flag + ', '
                flags_str = flags_str[:-2]
            print( '###########################################')
            print(f'###          Running Optimizer          ###')
            print( '###########################################\n')
            print(f'Reference waveform : {ref_Waveform.metadata["name"]}')
            print(f'(q, chi1z, chi2z)  : ({q:.2f}, {chi1:.2f}, {chi2:.2f})')
            print(f'binary type        : {flags_str}')
            print(f'Variables for ICs  : {self.opt_vars}')
            print(' ')

        mm_data  = self.load_or_create_mismatches()
        ref_name = self.ref_Waveform.metadata['name']
        
        run_optimization = True
        opt_data = None
        if ref_name in mm_data['mismatches']:
            opt_data = mm_data['mismatches'][ref_name]

            if not overwrite or opt_data['mm_opt']<self.bounds_iter['bad_mm']:
                run_optimization = False
            if verbose: 
                print(f'Loading mismatch from {self.json_file}')
                print('Optimal ICs  :')
                for ky in self.opt_vars:
                    print(f'                {ky:5s} : {opt_data[ky+"_opt"]:.15f}')
                print('Original mm  : {:.3e}'.format(opt_data['mm0']))
                print('Optimized mm : {:.3e}\n'.format(opt_data['mm_opt']))
        
        if run_optimization:
            random.seed(self.minimizer['opt_seed'])
            np.random.seed(self.minimizer['opt_seed'])
            dashes    = '-'*45
            asterisks = '*'*45
            
            eps = copy.copy(self.bounds_iter['eps_initial'])
            
            t0 = time.perf_counter()
            # i-loop on different search bounds
            for i in range(1, self.bounds_iter['max_iter']+1): 
                if self.bounds_iter['max_iter']>1 and self.verbose:
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
                
                if opt_data['mm_opt']<=self.bounds_iter['bad_mm']:
                    # if the mismatch is good according to eps-standard, then break
                    break
                
                elif i<self.bounds_iter['max_iter']:
                    # otherwise, increase the bound search (if we are not at the last iter)
                    flat_old_bounds = [item for key in self.opt_bounds for item in self.opt_bounds[key]]
                    
                    kys = self.opt_vars
                    self.opt_bounds = {kys[0]:[None,None], kys[1]:[None,None]}
                    for ky in eps:
                        eps[ky] *= self.bounds_iter['eps_factors'][ky]
                    self.__update_bounds(eps=eps)
                    flat_new_bounds = [item for key in self.opt_bounds for item in self.opt_bounds[key]]
                    print('\nIncreasing search bounds: [{:.3f},{:.3f}], [{:.3f},{:.3f}]'.format(*flat_old_bounds))
                    print('                  ----> : [{:.3f},{:.3f}], [{:.3f},{:.3f}]'.format(*flat_new_bounds))
                
                else:
                    mm_opt = opt_data['mm_opt']
                    print( '\n++++++++++++++++++++++++++++++++++++++')
                    print(f'+++  Reached eps_max_iter : {self.bounds_iter["max_iter"]:2d}     +++')
                    print(f'+++  mm_opt : {mm_opt:.2e} > {self.bounds_iter["bad_mm"]:.2e}  +++')
                    print( '++++++++++++++++++++++++++++++++++++++')
                    
            mm_data['mismatches'][ref_name] = opt_data
            
            if verbose:
                print('\n>> Best mismatch found : {:.3e}'.format(opt_data['mm_opt']))
                print(  '>> Total elapsed time  : {:.1f} s\n'.format(time.perf_counter()-t0))

            if json_file is not None: 
                self.save_mismatches(mm_data)
        self.opt_data = opt_data
        pass

    def __set_variables(self, vrs):
        """
        Set the variables to optimize over depending on the kind of ICs
        """
        if self.kind_ic == 'choose':
            self.opt_vars     = vrs
            self.map_function = None # Needs to be defined by the user
        elif self.kind_ic == 'e0f0':
            self.opt_vars = ['e0', 'f0']
            self.map_function = lambda x: {'ecc':x['e0'], 'f0':x['f0']} # map to EOB pars
        elif self.kind_ic == 'E0pph0':
            self.opt_vars = ['E0byM', 'pph0']
            self.map_function = lambda x: {'H_hyp':x['E0byM'], 'J_hyp':x['pph0']}
        elif self.kind_ic == 'phi0theta0':
            self.opt_vars = ['phi_ref', 'theta']

            def rotate_in_plane_spins(chiA,chiB,theta=0.):
                """
                Perform a rotation of the in-plane spins by an angle theta
                """
                from scipy.spatial.transform import Rotation

                zaxis    = np.array([0, 0, 1])
                r        = Rotation.from_rotvec(theta*zaxis)
                chiA_rot = r.apply(chiA)
                chiB_rot = r.apply(chiB)
                return chiA_rot, chiB_rot

            def func(vrs):
                theta   = vrs['theta']
                phi_ref = vrs['phi_ref']
                chiA = np.array([vrs['chi1x'], vrs['chi1y'], vrs['chi1z']])
                chiB = np.array([vrs['chi2x'], vrs['chi2y'], vrs['chi2z']])

                # rotate in-plane spin components by theta
                chiA_rot, chiB_rot = rotate_in_plane_spins(chiA,chiB,theta=theta)
                rotated = {
                        'chi1x'  : chiA_rot[0], 'chi1y': chiA_rot[1], 'chi1z': chiA_rot[2],
                        'chi2x'  : chiB_rot[0], 'chi2y': chiB_rot[1], 'chi2z': chiB_rot[2],
                        'phi_ref': phi_ref,
                        }
                return rotated

            self.map_function = func
        else:
            raise ValueError(f'Unknown kind of ICs: {self.kind_ic}')
        pass

    def __update_bounds(self, eps=None):
        """
        Set the bounds for the optimization; if the bounds are not specified,
        set them to the reference value (read from metadata) +/- eps
        """
        if eps is None: eps = self.bounds_iter['eps_initial']
        vls = []
        for ky in self.opt_vars:
            try:
                vls.append(self.ref_Waveform.metadata[ky])
            except KeyError:
                print(f'WARNING: update bounds, {ky} not found in metadata. Setting to 1.')
                vls.append(1)
        
        default_bounds = {ky: [vl*(1-eps[ky]), vl*(1+eps[ky])] for ky,vl in zip(self.opt_vars, vls)}

        for ky in self.opt_vars:
            for j in range(2):
                if self.opt_bounds[ky][j] is None:
                    self.opt_bounds[ky][j] = default_bounds[ky][j]
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
        options = {'minimizer'    : self.minimizer,
                   'kind_ic'      : self.kind_ic,
                   'vars'         : self.opt_vars,
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
        # Set all the intrinsic parameters that are not in ICs
        default_intrinsic =  ['M', 'q', 'chi1x', 'chi1y', 'chi1z', 'chi2x', 
                              'chi2y', 'chi2z', 'LambdaAl2', 'LambdaBl2']
        for ic in ICs: 
            if ic in default_intrinsic: default_intrinsic.remove(ic)
        
        if 'LambdaAl2' not in ref_meta: ref_meta['LambdaAl2'] = 0.0
        if 'LambdaBl2' not in ref_meta: ref_meta['LambdaBl2'] = 0.0

        sub_meta = {key: ref_meta[key] for key in default_intrinsic}
        sub_meta['use_nqc']  = self.use_nqc
        sub_meta['ode_tmax'] = 3e+4

        # map the ICs (and the other intrinsic pars) to the EOB parameters
        mapped_ids = self.map_function({**ICs, **sub_meta})

        if 'H_hyp' in mapped_ids or 'J_hyp' in mapped_ids:
            if self.r0_eob == 'read':
                # start close to the NR value, a little earlier
                mapped_ids['r_hyp'] = ref_meta['r0']*1.1
            else:
                if self.r0_eob is not None:
                    if self.r0_eob < ref_meta['r0']:
                        print(f'Warning: r0_eob={self.r0_eob} is smaller than the NR value r0={ref_meta["r0"]}')
                        print('         Setting r0_eob to NR value')
                        mapped_ids['r_hyp'] = ref_meta['r0']
                    else:
                        mapped_ids['r_hyp'] = self.r0_eob
                mapped_ids['r_hyp']         = self.r0_eob  # if None, it will be computed in the EOB model
        
        # add the mapped ICs to the sub_meta dictionary & additional model options
        # and run
        sub_meta.update(mapped_ids)
        sub_meta.update(self.model_opts)
        try:
            pars        = CreateDict(**sub_meta)
            eob_wave    = Waveform_EOB(pars=pars)
            #eob_wave._u = eob_wave.u#-eob_wave.u[0]
        except Exception as e:
            # FIXME: no error msg is a little bit criminal
            #print(f'Error occured in EOB wave generation:\n{e}')
            eob_wave = None
        return eob_wave
    
    def match_against_ref(self, eob_Waveform, verbose=None, iter_loop=False, return_matcher=False, cache={}, mm_settings=None):
        if verbose     is None: verbose     = self.verbose
        if mm_settings is None: mm_settings = self.mm_settings
        if eob_Waveform is not None:
            try:
                matcher = Matcher(self.ref_Waveform, eob_Waveform,
                                  settings=mm_settings, cache=cache)
                mm = matcher.mismatch
            except Exception as e:
                print('Error while computing match: ', e)
                matcher = None
                mm = 1.0
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
    
    def __func_to_minimize(self, x, kys, verbose=None, cache={}):
        if verbose is None: verbose = self.verbose
        # reassemble the ICs
        vs = {kys[i]: x[i] for i in range(len(kys))}
        eob_Waveform = self.generate_EOB(ICs=vs)
        if eob_Waveform is not None:
            mm = self.match_against_ref(eob_Waveform, verbose=self.verbose, iter_loop=True, cache=cache)
        else:
            if self.kind_ic=='E0pph0':
                pph0 = vs['pph0']
                ref_meta = self.ref_Waveform.metadata
                q    = ref_meta['q']
                chi1 = ref_meta['chi1z']
                chi2 = ref_meta['chi2z']
                rvec = np.linspace(2,20,num=200)
                Vmin = PotentialMinimum(rvec,pph0,q,chi1,chi2)
                dV   = Vmin-vs['E0byM']
            else:
                dV = 0
            mm = 1 + dV
        return mm

    def optimize_mismatch(self, use_ref_guess=True, verbose=None):
        """
        Optimize the mismatch between the reference waveform and the other model waveform.
        """
        if verbose is None: verbose = self.verbose
        kys     = self.opt_vars
        bounds  = self.opt_bounds
        meta    = self.ref_Waveform.metadata

        # Treat variables which are in common with the reference
        kys_ref = [ky for ky in kys if ky in self.ref_Waveform.metadata] # common keys
        vs_ref  = {ky: self.ref_Waveform.metadata[ky] for ky in kys_ref} # reference values
        for ky in kys_ref:
            vv = vs_ref[ky]
            if vv<bounds[ky][0] or vv>bounds[ky][1]:
                print('Warning! Reference value for {:s} is outside searching interval: {:.2e} not in [{:.2e},{:.2e}]'.format(ky,  vv, bounds[ky][0],  bounds[ky][1]))
        if use_ref_guess:
            # use reference values whenever possible
            vs0 = vs_ref
        else:
            # random initial guess
            vs0 = {ky: np.random.uniform(bounds[ky][0], bounds[ky][1]) for ky in kys}
        
        # randomly select the variables which are not in common with the reference
        for ky in kys:
            if ky not in kys_ref:
                vs0[ky] = np.random.uniform(bounds[ky][0], bounds[ky][1])

        # reference match
        mm0, matcher0 = self.match_against_ref(self.generate_EOB(ICs=vs0),
                                                                iter_loop=False, 
                                                                return_matcher=True
                                              )
        if verbose:
            print(f'Original  mismatch    : {mm0:.3e}')
            print( 'Optimization interval :')
            for ky in kys:
                print(f'                        {ky:5s} : [{bounds[ky][0]:.3e},{bounds[ky][1]:.3e}]')
            print(f'Initial guess         :')
            for ky in kys: 
                print(f'                        {ky:5s} : {vs0[ky]:.15f}')
        
        if self.use_matcher_cache:
            if matcher0 is None:
                if verbose: print('+++ First mm-computation failed! Not using cache +++')
                cache = {}
            else:
                cache = {'h1f':matcher0.h1f, 'M':matcher0.settings['M']}
        else:
            cache = {}
        
        # prepare for annealing
        x0           = [vs0[ky] for ky in kys]
        bounds_array = np.array([[bounds[ky][0], bounds[ky][1]] for ky in kys])
        f = lambda x : self.__func_to_minimize(x, kys, verbose=verbose, cache=cache)

        t0_annealing  = time.perf_counter()
        opts, mm_opt  = self.minimize(f, x0, bounds_array, kys)

        if verbose:
            print( '  >> mismatch - iter  : {:.3e} - {:3d}'.format(mm_opt, self.annealing_counter), end='\r')
            print(f'Optimized mismatch    : {mm_opt:.3e}')
            print(f'Optimal ICs           :' )
            for ky in kys:
                print(f'                        {ky:5s} : {opts[ky]:.15f}')
            print( 'Minimization time        : {:.1f} s'.format(time.perf_counter()-t0_annealing))
        
        # generate the eob waveform corresponding to the optimal ICs
        eob_opt = self.generate_EOB(ICs=opts)
        
        if self.debug:
            temp_settings = copy.copy(self.mm_settings)
            temp_settings['debug'] = True
            self.match_against_ref(eob_opt, mm_settings=temp_settings)
        
        opt_data = { 
                    # store also some attributes, just for convenience
                    'M'            : meta['M'],
                    'q'            : meta['q'],
                    'chi1x'        : meta['chi1x'],
                    'chi1y'        : meta['chi1y'],
                    'chi1z'        : meta['chi1z'],
                    'chi2x'        : meta['chi2x'],
                    'chi2y'        : meta['chi2y'],
                    'chi2z'        : meta['chi2z'],
                    'LambdaAl2'    : meta['LambdaAl2'] if 'LambdaAl2' in meta else 0.,
                    'LambdaBl2'    : meta['LambdaBl2'] if 'LambdaBl2' in meta else 0.,
                    'initial_frequency_mm' : self.mm_settings['initial_frequency_mm'],
                    'final_frequency_mm'   : self.mm_settings['final_frequency_mm'],
                    'opt_seed'     : self.minimizer['opt_seed'],
                    'opt_max_iter' : self.opt_max_iter,
                    'opt_good_mm'  : self.opt_good_mm,
                    'bounds_iter'  : self.bounds_iter,
                    # optimization results
                    'bounds'       : bounds, 
                    'mm0'          : mm0,
                    'mm_opt'       : mm_opt, 
                    }
        for ky in kys:
            opt_data[ky] = vs_ref[ky] if ky in vs_ref else None
            opt_data[ky+'_opt'] = opts[ky]
        
        if eob_opt is not None and self.json_save_dyn:
            dyn0 = eob_opt.dyn
            dyn0 = {ky: list(dyn0[ky]) for ky in dyn0.keys()}
        else:
            dyn0 = None
        opt_data['dyn0'] = dyn0
        
        return opt_data
    
    def __bounds_iter_defaults__(self):
        self.bounds_iter = {
                            'eps_initial': {},    # initial epsilon values 
                            'eps_factors': {},    # increase-factor for eps at each eps-iter
                            'max_iter'   : 1,     # If true, iterate on eps-bounds
                            'bad_mm'     : 0.1,   # if after opt_max_iter(s) we are still above this threshold 
        }
        for ky in self.opt_vars:
            self.bounds_iter['eps_initial'][ky] = 1e-2
            self.bounds_iter['eps_factors'][ky] = 2
        pass

    def __minimizer__defaults__(self):
        """
        Set the default minimizer options.
        """
        self.minimizer = { # annealing options
                           'kind': 'dual_annealing', 
                           'opt_maxfun': 1000, 
                           'opt_seed': 190521,

                           # differiantial_evolution options 
                           'opt_workers':1,

                           # dynesty options
                           'nlive'  : 10,
                           'maxiter': 10000,
                           'maxcall': 10000,
                           'print_progress': True,
        }
        pass

    def __minimize_annealing_(self, f, x0, bounds_array, kys):
        """
        Minimize with dual annealing. 
        """
        maxiter = self.minimizer.get('opt_maxfun', 1000)
        seed    = self.minimizer.get('opt_seed', 190521)
        x0      = x0

        opt_result   = optimize.dual_annealing(
                                                f,
                                                maxfun = maxiter, 
                                                seed   = seed, 
                                                x0     = x0,
                                                bounds = bounds_array,
                                            )
        
        opt_pars     = opt_result['x']
        opts         = {kys[i]: opt_pars[i] for i in range(len(kys))}
        mm_opt       = opt_result['fun']
        return opts, mm_opt
    
    def __minimize_differential_evo_(self, f, x0, bounds_array, kys):
        """
        Minimize with differential evolution. 
        """
        maxiter = self.minimizer.get('opt_maxfun', 1000)
        seed    = self.minimizer.get('opt_seed', 190521)
        workers = self.minimizer.get('opt_workers', 1) 
        x0      = x0

        opt_result   = optimize.differential_evolution(
                                                f,
                                                maxiter = maxiter, 
                                                seed    = seed, 
                                                x0      = x0,
                                                workers = workers,
                                                bounds  = bounds_array,
                                            )
        
        opt_pars     = opt_result['x']
        opts         = {kys[i]: opt_pars[i] for i in range(len(kys))}
        mm_opt       = opt_result['fun']
        return opts, mm_opt

    def __minimize__dynesty__(self, f, x0, bounds_array, kys):
        """
        Minimize with dynesty.
        NOTE: largely untested!
        """
        from dynesty import NestedSampler

        # Define the dimensionality of our problem.
        ndim     = len(kys)
        progress = self.minimizer.get('print_progress', True)
        nlive    = self.minimizer.get('nlive'  , 1024)
        maxiter  = self.minimizer.get('maxiter', 10000)
        maxcall  = self.minimizer.get('maxcall', 10000)

        def loglike(x):
            """
            The log-likelihood function.
            We use the mismatch weighted over the min_thrs, squared
            """
            logl = -0.5*(f(x)/0.001)**2
            if np.isnan(logl) or np.isinf(logl):
                return -np.inf
            return logl

        def prior_transform(u):
            """
            Map the unit cube to the parameter space, assuming
            uniform priors on the parameters.
            """
            return [bounds_array[i][0] + u[i] * (bounds_array[i][1] - bounds_array[i][0]) for i in range(ndim)]

        # Define our sampler.
        sampler = NestedSampler(loglike, 
                                prior_transform, 
                                ndim, 
                                nlive=nlive, 
                                sample="rwalk",  # Specify rwalk as the sampling method
                                bound="multi",   # Use a multi-ellipsoid bound    
                                )  
        sampler.run_nested(maxiter=maxiter, maxcall=maxcall,print_progress=progress, dlogz=0.1)

        # return just the maxL point
        maxL   = sampler.results.logl.argmax()
        opts   = {kys[i]: sampler.results.samples[maxL][i] for i in range(len(kys))}
        mm_opt = f(sampler.results.samples[maxL])

        # make the traceplot
        if self.verbose:
            from dynesty import plotting as dyplot
            fig, _ = dyplot.traceplot(sampler.results,
                             show_titles=True,
                             trace_cmap='viridis')
            fig.savefig('traceplot.png')

        return opts, mm_opt


