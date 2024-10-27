import os, subprocess
import numpy as np
from scipy.optimize import brentq
from scipy.signal   import find_peaks
import matplotlib.pyplot as plt

try:
    import EOBRun_module as EOB
except ModuleNotFoundError:
    print("WARNING: TEOBResumS not installed.")

from ..waveform import Waveform

class Waveform_EOB(Waveform):
    """
    Class to handle EOB waveforms
    """
    def __init__(
                    self, 
                    pars=None, 
                ):
        super().__init__()
        self.pars = pars
        self._kind = 'EOB'
        self._run_py()
        pass 

    def compute_energetics(self):
        pars = self.pars
        q    = pars['q']
        q    = float(q)
        nu   =  q/(1.+q)**2
        dyn  = self.dyn
        E, j = dyn['E'], dyn['Pphi']
        Eb   = (E-1)/nu
        self.Eb = Eb
        self.j  = j
        return Eb, j

    # python wf func
    def _run_py(self):
        if self.pars['domain']:
            f, rhp,ihp,rhc,ihc, hflm, dyn, _ = EOB.EOBRunPy(self.pars)
            self._f   = f
            self._hlm = hflm
            self._dyn = dyn
            self._hp  = rhp-1j*ihp
            self._hc  = rhc-1j*ihc
            self.domain = 'Freq'
        else:
            t, hp,hc, hlm, dyn = EOB.EOBRunPy(self.pars)
            self._u   = t
            hlm_conv  = convert_hlm(hlm)
            self._hlm = hlm_conv
            self._dyn = dyn
            self._hp  = hp
            self._hc  = hc
            self.domain = 'Time'
        return 0

def convert_hlm(hlm):
    from ..utils import wf_utils as wfu
    """
    Convert the hlm dictionary from k to (ell, emm) notation
    """
    hlm_conv = {}
    for key in hlm.keys():
        ell = wfu.k_to_ell(int(key))
        emm = wfu.k_to_emm(int(key))
        A   = hlm[key][0]
        p   = hlm[key][1]
        hlm_conv[(ell, emm)] = {'real': A*np.cos(p), 'imag': -1*A*np.sin(p),
                                'A'   : A, 'p' : p,
                                'h'   : A*np.exp(-1j*p)
                                }
    return hlm_conv

# external function for dict creation
def CreateDict(M=1., q=1, 
               chi1z=0., chi2z=0, 
               chi1x=0., chi2x=0,
               chi1y=0., chi2y=0,
               l1=0, l2=0, 
               iota=0, f0=0.0035, srate=4096., df = 1./128.,
               phi_ref = 0.,
               ecc = 1e-8, r_hyp = 0, H_hyp = 0, J_hyp=0, anomaly = np.pi,
               interp="yes", arg_out="yes", use_geom="yes", 
               use_mode_lm=[1], ode_tmax=1e+6,
               cN3LO=None, a6c=None):
        """
        Create the dictionary of parameters for EOBRunPy
        """
        if H_hyp>0 and J_hyp>0 and r_hyp is None:
            if H_hyp>1: 
                r_hyp = 300
            else:
                #PotentialPlot(H_hyp,J_hyp,q,chi1z,chi2z)
                r_apa = search_apastron(q, chi1z*0, chi2z*0, J_hyp, H_hyp, step_size=0.1)
                if r_apa is None:
                    raise RuntimeError(f'Apastron not found, check initial conditon: E={H_hyp:.5f}, pph={J_hyp:.5f}')
                r_hyp = r_apa - 1e-2 # small tol to avoid numerical issues

        pardic = {
            'M'                  : M,
            'q'                  : q,
            'chi1'               : chi1z,
            'chi2'               : chi2z,
            'chi1z'              : chi1z, 
            'chi2z'              : chi2z,
            'chi1x'              : chi1x,
            'chi2x'              : chi2x,
            'chi1y'              : chi1y,
            'chi2y'              : chi2y,
            'LambdaAl2'          : l1,
            'LambdaBl2'          : l2,
            'distance'           : 1.,
            'initial_frequency'  : f0,
            'use_geometric_units': use_geom,
            'interp_uniform_grid': interp,
            'domain'             : 0,
            'srate_interp'       : srate,
            'inclination'        : iota,
            'output_hpc'         : "no",
            'use_mode_lm'        : use_mode_lm,    # List of modes to use
            'arg_out'            : arg_out,        # output dynamics and hlm in addition to h+, hx
            'ecc'                : ecc,
            'r_hyp'              : r_hyp,
            'H_hyp'              : H_hyp,
            'j_hyp'              : J_hyp,
            'coalescence_angle'  : phi_ref,
            'df'                 : df,
            'anomaly'            : anomaly,
            'spin_flx'           : 'EOB',
            'spin_interp_domain' : 0,
            'ode_tmax'           : ode_tmax,
        }

        if a6c is not None:
            pardic['a6c'] = a6c
        if cN3LO is not None:
            pardic['cN3LO'] = cN3LO
        return pardic

def TEOB_info(input_module,verbose=False):
    module = {}
    module['softlink']      = input_module.__file__
    module['name']          = module['softlink'].split('/')[-1]
    module['real_path']     = os.path.realpath(module['softlink'])
    
    teob_path = module['real_path'].replace(module['name'],'')
    teob_path = teob_path.replace('Python/','')
    module['teob_path'] = teob_path 
    module['commit']    = subprocess.Popen(['git', '--git-dir='+teob_path+'.git',\
                                            'rev-parse', 'HEAD'],stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
    module['branch']    = subprocess.check_output(['git','--git-dir='+teob_path+'.git', \
                                            'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.STDOUT, text=True).strip()
    if verbose:
        for key,value in module.items():
            print(f'{key:10s} : {value}')
    return module 

def SpinHamiltonian(r, pph, q, chi1, chi2, prstar=0.):
    hatH   = EOB.eob_ham_s_py(r, q, pph, prstar, chi1, chi2)
    nu     = q/(1+q)**2
    E0     = nu*hatH[0]
    return E0

def RadialPotential(r,pph,q,chi1,chi2):
    return np.array([SpinHamiltonian(ri, pph, q, chi1, chi2) for ri in r])
    
def bracketing(f,start,end,step_size):
    bracketed_intervals = []
    a  = start
    b  = a + step_size
    fa = f(a)
    while b<=end:
        fb = f(b)
        if fa * fb < 0:
            bracketed_intervals.append([a,b])
            a  = b
            fa = f(a)
        b += step_size
    bracketed_intervals.append([a,end])
    return bracketed_intervals

def PotentialMinimum(rvec,pph,q,chi1,chi2):
    V = RadialPotential(rvec,pph,q,chi1,chi2)
    peaks, _ = find_peaks(-V, height=-1)
    if len(peaks)>0:
        Vmin = V[peaks[0]]
    else:
        Vmin = 1 
    return Vmin

def search_apastron(q, chi1, chi2, pph, E, step_size=0.1):
    def fzero(r):
        V = SpinHamiltonian(r, pph, q, chi1, chi2)
        return E-V
    r_infty = 200
    bracketed_intervals = bracketing(fzero, 2, r_infty, step_size=step_size)
    approx_r_apa = bracketed_intervals[-1][0]
    # debug
    #import matplotlib.pyplot as plt # for debug
    #rvec = np.linspace(2, 200, num=1000)
    #V = RadialPotential(rvec, pph, q, chi1 ,chi2)
    #plt.figure
    #plt.plot(rvec, V)
    #plt.axhline(E)
    #plt.show()
    if len(bracketed_intervals)<3:
        r_apa = None
    else:
        r_apa = brentq(fzero, approx_r_apa-step_size, approx_r_apa+step_size)
    return r_apa

def PotentialPlot(E0,pph0,q,chi1,chi2):
    rvec  = np.linspace(2,100,num=1000)
    V     = RadialPotential(rvec, pph0, q, chi1, chi2) 
    r_apa = search_apastron(q,chi1,chi2,pph0,E0)
    Vmin  = PotentialMinimum(rvec,pph0,q,chi1,chi2)
    plt.figure()
    plt.title(f'E0={E0:.5f}, pph0={pph0:.5f}')
    plt.plot(rvec, V)
    plt.axhline(E0, c='r')
    plt.axvline(r_apa, c='g')
    plt.axhline(Vmin,  c='c')
    plt.show()
    return

#---------------------
# Span parameterspace

if __name__ == '__main__':
    module = TEOB_info(EOB,verbose=True)
    #for key,value in module.items():
    #    print(f'{key:10s} : {value}')
