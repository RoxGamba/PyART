import numpy as np
from  .match import Matcher 
from ..models.teob import Waveform_EOB
from ..models.teob import CreateDict
from ..models.teob import PotentialMinimum
import random
from scipy import optimize 

class Optimizer(object):
    """
    Class to compute EOB initial data that minimize mismatch
    with reference waveform
    """
    def __init__(self,
                 ref_Waveform,
                 kind_ic     = 'e0f0', # implemented: e0f0, E0pph0
                 mm_settings = None,
                 verbose     = True,
                 opt_bounds  = [[None,None],[None,None]],
                 debug       = False,
                 ):
        
        self.ref_Waveform = ref_Waveform
        self.kind_ic      = kind_ic
        self.verbose      = verbose
        self.opt_iter     = 1
        self.opt_seed     = 190521
        self.opt_bounds   = opt_bounds
        self.opt_maxfun   = 100

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
        
        if verbose:
            print('-'*70)
            print(f'Running Optimizer')
            print(f'Reference waveform : {ref_Waveform.metadata["name"]}')
            print(f'variables of ICs   : {ic_keys[0]}, {ic_keys[1]}')
            print('-'*70)

        # mismatch settings
        self.mm_settings  = Matcher.__default_parameters__(0) 
        if isinstance(mm_settings, dict):
            for k in mm_settings:
                self.mm_settings[k] = mm_settings[k]
        if not self.mm_settings['cut'] and self.verbose:
            print('Warning: using the cut-option during optimization is strongly suggested!')

        mm_opt, x_opt, y_opt = self.optimize_mismatch()
        self.opt_Waveform = self.generate_EOB(ICs={ic_keys[0]:x_opt, ic_keys[1]:y_opt}) 
        self.opt_mismatch = mm_opt
        
        if debug:
            self.mm_settings['debug'] = True
            self.match_against_ref(self.opt_Waveform)
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
         
        self.__update_bounds()
        bounds = self.opt_bounds
        if vx_ref<bounds[0][0] and vx_ref>bounds[0][1]:
            print('Warning! Reference value for {:s} is outside searching interbal: [{:.2e},{:.2e}]'.format(ks,  bounds[0][0],  bounds[0][1]))
        if vy_ref<bounds[1][0] and vy_ref>bounds[1][1]:
            print('Warning! Reference value for {:s} is outside searching interbal: [{:.2e},{:.2e}]'.format(ks,  bounds[1][0],  bounds[1][1]))
        
        if use_ref_guess:
            vxy0 = np.array([vx_ref,vy_ref])
        else:
            random.seed(self.opt_seed)
            vx0  = random.uniform(bounds[0][0], bounds[0][1])
            vy0  = random.uniform(bounds[1][0], bounds[1][1])
            vxy0 = np.array([vx0, vy0])
        
        if verbose:
            mm0 = self.match_against_ref(self.generate_EOB(ICs={kx:vxy0[0], ky:vxy0[1]}),iter_loop=False)
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
        return mm_opt, x_opt, y_opt




