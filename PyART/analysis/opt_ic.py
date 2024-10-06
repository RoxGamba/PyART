import numpy as np
from  .match import Matcher 
from ..models.teob import Waveform_EOB
from ..models.teob import CreateDict
import random
from scipy import optimize 

class Optimizer(object):
    """
    Class to compute EOB initial data that minimize mismatch
    with reference waveform
    """
    def __init__(self,
                 ref_Waveform,
                 kind_ic     = 'e0f0',
                 mm_settings = None,
                 verbose     = True,
                 opt_bounds  = [[None,None],[None,None]]
                 ):
        
        self.ref_Waveform = ref_Waveform
        self.kind_ic      = kind_ic
        self.verbose      = verbose
        self.opt_iter     = 1
        self.opt_seed     = 190521
        self.opt_bounds   = opt_bounds
        self.opt_maxfun   = 100
        
        if kind_ic=='e0f0':
            e0_nr = self.ref_Waveform.metadata['e0']
            if e0_nr is None or e0_nr>=1:
                if self.verbose: print('Invalid e0 in NR waveform! Overwriting with e0=0.5')
                self.ref_Waveform.metadata['e0'] = 0.5

        # mismatch settings
        self.mm_settings  = Matcher.__default_parameters__(0) 
        if isinstance(mm_settings, dict):
            for k in mm_settings:
                self.mm_settings[k] = mm_settings[k]
        if not self.mm_settings['cut'] and self.verbose:
            print('Warning: using the cut-option during optimization is strongly suggested!')

        # only for testing
        mm_opt, e0_opt, f0_opt = self.optimize_mismatch(keys=['e0', 'f0'], verbose=self.verbose)
        self.opt_Waveform = self.generate_EOB(ICs={'e0':e0_opt, 'f0':f0_opt}) 
        self.opt_mismatch = mm_opt
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
        else: 
            raise ValueError(f'Unknown kind of ICs: {kind}')
        
        # return generated EOB waveform 
        pars = CreateDict(**sub_meta)
        eob_wave = Waveform_EOB(pars=pars)
        eob_wave._u = eob_wave.u#-eob_wave.u[0] 
        return eob_wave
    
    def match_against_ref(self, eob_Waveform, verbose=None, iter_loop=False):
        if verbose is None: verbose = self.verbose
        matcher = Matcher(self.ref_Waveform, eob_Waveform, pre_align=False,
                          settings=self.mm_settings)
        if verbose and iter_loop:
            self.opt_iter += 1
            print( '  >> mismatch - iter : {:.3e} - {:3d}'.format(matcher.mismatch, self.opt_iter), end='\r')
        return matcher.mismatch
    
    def __update_bounds(self, keys):
        kx = keys[0]
        ky = keys[1]
        vx_ref = self.ref_Waveform.metadata[kx]
        vy_ref = self.ref_Waveform.metadata[ky]
        default_bounds = [ [vx_ref*0.9, vx_ref*1.1],
                           [vy_ref*0.9, vy_ref*1.1] ]
        for i in range(2):
            for j in range(2):
                if self.opt_bounds[i][j] is None:
                    self.opt_bounds[i][j] = default_bounds[i][j]
        return 

    def optimize_mismatch(self, keys=['e0', 'f0'],verbose=None, use_ref_guess=True):
        if verbose is None: verbose = self.verbose
        kx = keys[0]
        ky = keys[1]
        f = lambda vxy : self.match_against_ref(self.generate_EOB(ICs={kx:vxy[0], ky:vxy[1]}),
                                                verbose=verbose, iter_loop=True)
        random.seed(self.opt_seed)
        vx_ref  = self.ref_Waveform.metadata[kx]
        vy_ref  = self.ref_Waveform.metadata[ky]
         
        self.__update_bounds(keys=keys)
        
        bounds = self.opt_bounds
        if vx_ref<bounds[0][0] and vx_ref>bounds[0][1]:
            print('Warning! Reference value for {:s} is outside searching interbal: [{:.2e},{:.2e}]'.format(ks,  bounds[0][0],  bounds[0][1]))
        if vy_ref<bounds[1][0] and vy_ref>bounds[1][1]:
            print('Warning! Reference value for {:s} is outside searching interbal: [{:.2e},{:.2e}]'.format(ks,  bounds[1][0],  bounds[1][1]))
        
        if use_ref_guess:
            vxy0 = np.array([vx_ref,vy_ref])
        else:
            vx0  = random.uniform(bounds[0][0], bounds[0][1])
            vy0  = random.uniform(bounds[1][0], bounds[1][1])
            vxy0 = np.arrays([vx0, vy0])
        
        if verbose:
            mm0 = self.match_against_ref(self.generate_EOB(ICs={kx:vxy0[0], ky:vxy0[1]}),iter_loop=False)
            print(f'Original  mismatch   : {mm0:.3e}')
            print( 'Optimizing in        : {:s} in [{:.2e},{:.2e}]'.format(kx, bounds[0][0], bounds[0][1]))
            print( '                     : {:s} in [{:.2e},{:.2e}]'.format(ky, bounds[1][0], bounds[1][1]))
            print(f'Initial guess        : {vxy0[0]:.2e}, {vxy0[1]:.2e}')
        opt_result = optimize.dual_annealing(f, maxfun=self.opt_maxfun, 
                                                seed=self.opt_seed, x0=vxy0,
                                                bounds=bounds)
        opt_pars     = opt_result['x']
        x_opt, y_opt = opt_pars[0], opt_pars[1]
        mm_opt = opt_result['fun']
        if verbose:
            print( '  >> mismatch - iter : {:.3e} - {:3d}'.format(mm_opt, self.opt_iter), end='\r')
            print(f'Optimized mismatch   : {mm_opt:.3e}')
            print(f'Optimal ICs          : {kx}:{x_opt:.5f}')
            print(f'                       {ky}:{y_opt:.3e}')
        return mm_opt, x_opt, y_opt
