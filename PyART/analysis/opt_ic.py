import os, json, random
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
                 opt_seed    = 190521,
                 opt_maxfun  = 100,
                 kind_ic     = 'E0pph0', # implemented: e0f0, E0pph0
                 mm_settings = None,
                 verbose     = True,
                 opt_bounds  = [[None,None],[None,None]],
                 debug       = False,
                 json_file   = None,  # JSON file with mm (must be consistent with current options) 
                 overwrite   = False, # overwrite JSON with new mm-computation
                 ):
        
        self.ref_Waveform = ref_Waveform
        self.kind_ic      = kind_ic
        self.verbose      = verbose
        self.opt_bounds   = opt_bounds
        self.opt_seed     = opt_seed
        self.opt_maxfun   = opt_maxfun
        self.json_file    = json_file
        self.overwrite    = overwrite
        self.opt_iter     = 1 # counter
        
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
        
        # update None values in self.opt_bounds
        self.__update_bounds()
        
        # mismatch settings
        self.mm_settings  = Matcher.__default_parameters__(0) 
        if isinstance(mm_settings, dict):
            for k in mm_settings:
                self.mm_settings[k] = mm_settings[k]
        if not self.mm_settings['cut'] and self.verbose:
            print('Warning: using the cut-option during optimization is strongly suggested!')
          
        if verbose:
            print('-'*70)
            print(f'Running Optimizer')
            print(f'Reference waveform : {ref_Waveform.metadata["name"]}')
            print(f'variables of ICs   : {ic_keys[0]}, {ic_keys[1]}')
            print('-'*70)

        mm_data  = self.load_or_create_mismatches()
        ref_name = self.ref_Waveform.metadata['name']
        if not overwrite and ref_name in mm_data['mismatches']:
            if verbose: 
                print(f'Loading mismatch from {self.json_file}')
                opt_data =  mm_data['mismatches'][ref_name]
                print('Optimal ICs  : {:s}={:.5f}, {:s}:{:.5f}'.format(opt_data['kx'], opt_data['x_opt'],
                                                                  opt_data['ky'], opt_data['y_opt']))
                print('Original mm  : {:.3e}'.format(opt_data['mm0']))
                print('Optimized mm : {:.3e}'.format(opt_data['mm_opt']))
        else:
            opt_data = self.optimize_mismatch()
            mm_data['mismatches'][ref_name] = opt_data
            self.save_mismatches(mm_data)

        if debug:
            self.mm_settings['debug'] = True
            opt_Waveform = self.generate_EOB(ICs={self.ic_keys[0]:opt_data['x_opt'], 
                                                  self.ic_keys[1]:opt_data['y_opt']})
            self.match_against_ref(opt_Waveform)
        pass
    
    def __update_bounds(self):
        kx = self.ic_keys[0]
        ky = self.ic_keys[1]
        vx_ref = self.ref_Waveform.metadata[kx]
        vy_ref = self.ref_Waveform.metadata[ky]
        eps = 1e-2
        default_bounds = [ [vx_ref*(1-eps), vx_ref*(1+eps)],
                           [vy_ref*(1-eps), vy_ref*(1+eps)] ]
        for i in range(2):
            for j in range(2):
                if self.opt_bounds[i][j] is None:
                    self.opt_bounds[i][j] = default_bounds[i][j]
        return 
    
    def load_or_create_mismatches(self): 
        """
        Load mismatches data if the options of the
        json file stored is consistent with current ones.
        Otherwise, create a new dictionary (NOT a new json file)
        """
        # convert numpy array to lists to avoid issues with JSON writing/loading
        loc_mm_settings = self.mm_settings
        for k in loc_mm_settings:
            val = loc_mm_settings[k]
            if isinstance(val, np.ndarray):
                loc_mm_settings[k] = list(val)
        # options to store/read in JSON 
        options = {'opt_maxfun'  : self.opt_maxfun, 
                   'kind_ic'     : self.kind_ic,
                   'mm_settings' : loc_mm_settings,
                  } 

        # check if file exists
        if self.json_file is not None and os.path.exists(self.json_file):
            # load mismatches dict
            with open(self.json_file, 'r') as file:
                json_data = json.loads(file.read())

            # fix list of list to list of tuples for 'modes' in json-data
            modes_list_of_list = json_data['options']['mm_settings']['modes']
            json_data['options']['mm_settings']['modes'] = [tuple(mode) for mode in modes_list_of_list]
            # check that the options are the same
            if not ut.are_dictionaries_equal(json_data['options'], options, 
                                             excluded_keys=[], verbose=True):
                print(f'>> Options in {self.json_file}')
                for key, value in options.items():
                    json_value = json_data['options'][key]
                    if isinstance(value, dict):
                        dbool = ut.are_dictionaries_equal(value, json_value, verbose=True,
                                                          excluded_keys=['debug'])
                        if dbool:
                            print(f'>> issues with {key:16s} (dictionary)')
                        else:
                            print(f'>> {key:16s} is dictionary')
                    else:
                        if value      is None: value      = "None"
                        if json_value is None: json_value = "None"
                        if isinstance(json_value, list) or isinstance(value, list):
                            print(f'>> {key:16s} is list')
                        else:
                            print(f">> {key:16s} - json: {json_value:<20}   self: {value:<20}")
                raise RuntimeError('The options in the json-file are different w.r.t. the current ones. Exit.') 
            else:
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
        if os.path.exists(json_file) and not overwrite:
            with open(json_file, 'r') as file:
                json_data = json.loads(file.read())
            if sim_name in json_data['mismatches']:
                print(f'   ---> File {json_file} alreay exists and contains {sim_name},but overwriting is off.')
                json_file = json_file.replace('.json', '_new.json')
                print( '   ---> writing on file: {json_file}') 
        with open(json_file, 'w') as file:
            file.write(json.dumps(data,indent=2))
        if verbose: print(f'Created {json_file}')
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
        
        # return generated EOB waveform 
        try:
            pars        = CreateDict(**sub_meta)
            eob_wave    = Waveform_EOB(pars=pars)
            eob_wave._u = eob_wave.u#-eob_wave.u[0]
        except Exception as e:
            # FIXME: no error msg is a little bit criminal
            #print(f'Error occured in EOB wave generation:\n{e}')
            eob_wave = None
        return eob_wave
    
    def match_against_ref(self, eob_Waveform, verbose=None, iter_loop=False):
        if verbose is None: verbose = self.verbose
        if eob_Waveform is not None:
            matcher = Matcher(self.ref_Waveform, eob_Waveform, pre_align=False,
                              settings=self.mm_settings)
            mm = matcher.mismatch
        else:
            mm = 1.0
        if verbose and iter_loop:
            self.opt_iter += 1
            print( '  >> mismatch - iter  : {:.3e} - {:3d}'.format(mm, self.opt_iter), end='\r')
        return mm
    
    def __func_to_minimize(self, vxy, verbose=None):
        if verbose is None: verbose = self.verbose
        kx = self.ic_keys[0]
        ky = self.ic_keys[1]
        eob_Waveform = self.generate_EOB(ICs={kx:vxy[0], ky:vxy[1]})
        if eob_Waveform is not None:
            mm = self.match_against_ref(eob_Waveform, verbose=self.verbose, iter_loop=True)
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
        random.seed(self.opt_seed)
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
            random.seed(self.opt_seed)
            vx0  = random.uniform(bounds[0][0], bounds[0][1])
            vy0  = random.uniform(bounds[1][0], bounds[1][1])
            vxy0 = np.array([vx0, vy0])
        
        mm0 = self.match_against_ref(self.generate_EOB(ICs={kx:vxy0[0], ky:vxy0[1]}),iter_loop=False)
        if verbose:
            print(f'Original  mismatch    : {mm0:.3e}')
            print( 'Optimization interval : {:5s} in [{:.2e}, {:.2e}]'.format(kx, bounds[0][0], bounds[0][1]))
            print( '                      : {:5s} in [{:.2e}, {:.2e}]'.format(ky, bounds[1][0], bounds[1][1]))
            print(f'Initial guess         : {kx:5s} : {vxy0[0]:.15f}')
            print(f'                        {ky:5s} : {vxy0[1]:.15f}')

        f = lambda vxy : self.__func_to_minimize(vxy, verbose=verbose)
        opt_result = optimize.dual_annealing(f, maxfun=self.opt_maxfun, 
                                                seed=self.opt_seed, x0=vxy0,
                                                bounds=bounds)
        opt_pars     = opt_result['x']
        x_opt, y_opt = opt_pars[0], opt_pars[1]
        mm_opt = opt_result['fun']
        if verbose:
            print( '  >> mismatch - iter  : {:.3e} - {:3d}'.format(mm_opt, self.opt_iter), end='\r')
            print(f'Optimized mismatch    : {mm_opt:.3e}')
            print(f'Optimal ICs           : {kx:5s} : {x_opt:.15f}')
            print(f'                        {ky:5s} : {y_opt:.15f}')
        opt_data = {'kx':kx, 'ky':ky,
                    'x_ref':vx_ref, 'y_ref':vy_ref,
                    'bounds':bounds, 'seed':self.opt_seed,
                    'x0':vxy0[0], 'y0':vxy0[1], 'mm0':mm0,
                    'mm_opt':mm_opt, 'x_opt':x_opt, 'y_opt':y_opt, 
                    }
        return opt_data




