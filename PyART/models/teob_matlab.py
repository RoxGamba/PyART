import os, subprocess
import numpy as np
from scipy.optimize import brentq
from scipy.signal   import find_peaks
from scipy.io       import loadmat
import matplotlib.pyplot as plt
from string import Template
import h5py

from ..waveform import Waveform
from ..utils import wf_utils as wfu

matlab_base = """
addpath('${code_dir}TEOBRun/');
addpath('${code_dir}TEOBRun/parfiles/');
addpath('${code_dir}PointMass/');

evalc("SetNaming('new')");
evalc("eob_run_spin_eccentric_Pade33(${q}, 'single', ${chi1}, ${chi2}${ecc_s}${omg_ap_s}${c3_s}${a6_s}${nr_s}${outdir_s}${ell_s}${rholm}${H_model}${Tmax}${newlogs}${rho22_SO_resum}${rend}${nqc_amp}${nqc_omg})");
quit;
"""
matlab_base_loud = """
addpath('${code_dir}TEOBRun/');
addpath('${code_dir}TEOBRun/parfiles/');
addpath('${code_dir}PointMass/');

evalc("SetNaming('new')");
eob_run_spin_eccentric_Pade33(${q}, 'single', ${chi1}, ${chi2}${ecc_s}${omg_ap_s}${c3_s}${a6_s}${nr_s}${outdir_s}${ell_s}${rholm}${H_model}${Tmax}${newlogs}${rho22_SO_resum}${rend}${nqc_amp}${nqc_omg});
quit;
"""

matlab_base_hyp = """
addpath('${code_dir}TEOBRun/');
addpath('${code_dir}TEOBRun/parfiles/');
addpath('${code_dir}PointMass/');

evalc("SetNaming('new')");
evalc("eob_run_hyperbolic(${q}, ${E0}, ${L0}, ${chi1}, ${chi2}${Tmax}${r0}${c3_s}${a6_s}${nr_s}${outdir_s}${ell_s}${rholm}${H_model}${newlogs}${rho22_SO_resum}${rend}${nqc_amp}${nqc_omg})");
quit;
"""
matlab_base_hyp_loud = """
addpath('${code_dir}TEOBRun/');
addpath('${code_dir}TEOBRun/parfiles/');
addpath('${code_dir}PointMass/');

evalc("SetNaming('new')");
eob_run_hyperbolic(${q}, ${E0}, ${L0}, ${chi1}, ${chi2}${Tmax}${r0}${c3_s}${a6_s}${nr_s}${outdir_s}${ell_s}${rholm}${H_model}${newlogs}${rho22_SO_resum}${rend}${nqc_amp}${nqc_omg});
quit;
"""

matlab_base_leob = """
addpath('${code_dir}TEOBRun/');
addpath('${code_dir}TEOBRun/parfiles/');
addpath('${code_dir}PointMass/');

evalc("SetNaming('new')");
evalc("eob_run_spin_leob(${q}, ${irf}, ${chi1}, ${chi2}${omg_ap_s}${r0}${Tmax}${c3_s}${a6_s}${nr_s}${outdir_s}${ell_s}${rholm}${newlogs}${rho22_SO_resum}${rend}${ASS_fact}${SSPM}${nqc_amp}${nqc_omg})");
quit;
"""
matlab_base_leob_loud = """
addpath('${code_dir}TEOBRun/');
addpath('${code_dir}TEOBRun/parfiles/');
addpath('${code_dir}PointMass/');

evalc("SetNaming('new')");
eob_run_spin_leob(${q}, ${irf}, ${chi1}, ${chi2}${omg_ap_s}${r0}${Tmax}${c3_s}${a6_s}${nr_s}${outdir_s}${ell_s}${rholm}${newlogs}${rho22_SO_resum}${rend}${ASS_fact}${SSPM}${nqc_amp}${nqc_omg});
quit;
"""

class Waveform_EOBMatlab(Waveform):
    """
    Class to handle EOB waveforms generated with the Matlab code
    """
    def __init__(
                    self, 
                    pars=None,
                    code_dir="/home/danilo.chiaramello/teobresumsmatlab/",
                    data_dir="/data/prometeo/danilo.chiaramello/teob_seob/data/matlab_data/",
                    source="/usr/local/src/R2020b/bin/matlab",
                    run=False,
                    verbose=False,
                    load_insp=False
                ):
        super().__init__()
        self.pars      = pars
        self._kind     = 'EOB'
        self.code_dir  = code_dir
        self.dir       = data_dir
        self.source    = source
        self.run       = run
        self.verbose   = verbose
        self.load_insp = load_insp
        self.hyp       = True if self.pars['H_hyp'] is not None else False
        self.leob      = pars['leob']

        if self.hyp and self.pars['j_hyp'] is None:
            raise ValueError("For hyperbolic orbits, H_hyp and j_hyp must be provided.")
        if self.hyp and not self.load_insp:
            print("WARNING: Inspiral-only waveform is needed for hyperbolic orbits.")

        if self.hyp:
            if self.verbose:
                self.template = Template(matlab_base_hyp_loud)
            else:
                self.template = Template(matlab_base_hyp)
        elif self.leob:
            if self.verbose:
                self.template = Template(matlab_base_leob_loud)
            else:
                self.template = Template(matlab_base_leob)
        else:
            if self.verbose:
                self.template = Template(matlab_base_loud)
            else:
                self.template = Template(matlab_base)

        self.wavefile, self.dynfile = matnames(self.pars['q'], chi1=self.pars['chi1z'], chi2=self.pars['chi2z'],
                                               ecc=self.pars['ecc'], omg0=self.pars['initial_frequency']*2*np.pi,
                                               E0=self.pars['H_hyp'], L0=self.pars['j_hyp'], r0=self.pars['r0'],
                                               nr=self.pars['nr'], leob=self.leob)

        if self.run == True:
            self._run_matlab()
        elif self._check_dir() == 0:
            self._run_matlab()
        self._load_dyn()
        self._load_hlm()
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
    
    def _check_dir(self):
        if os.path.exists(self.dir + self.wavefile) == True and os.path.exists(self.dir + self.dynfile) == True:
            return 1
        else:
            return 0
        
    def _convert_pars(self):
        """
        Build dictionary for template, converting parameters as needed.
        """
        if self.pars['a6c'] is None:
            a6_d = None
        elif isinstance(self.pars['a6c'], float):
            a6_d = self.pars['a6c']
        elif isinstance(self.pars['a6c'], str):
            a6_d = "'{}'".format(self.pars['a6c'])
        else:
            raise ValueError("a6c must be a float or a string.")
        
        if self.pars['cN3LO'] is None:
            c3_d = None
        elif isinstance(self.pars['cN3LO'], float):
            c3_d = self.pars['cN3LO']
        elif isinstance(self.pars['cN3LO'], str):
            c3_d = "'{}'".format(self.pars['cN3LO'])
        else:
            raise ValueError("a6c must be a float or a string.")
        
        if isinstance(self.pars['ode_rend'], str):
            rend_val = f"'{self.pars['ode_rend']}'"
        elif self.pars['ode_rend'] is not None:
            rend_val = f"{self.pars['ode_rend']}"
        else:
            rend_val = "0"
            
        if self.hyp:
            temp_dict = {"code_dir": self.code_dir,
                         "q": self.pars['q'], 
                         "chi1": self.pars['chi1z'], "chi2": self.pars['chi2z'],
                         "E0": self.pars['H_hyp'], "L0": self.pars['j_hyp'],
                         "Tmax": ", 'Tmax', {}".format(self.pars['ode_tmax']),
                         "r0": ", 'r0', {}".format(self.pars['r0']),
                         "c3_s": ", 'cN3LO', {}".format(c3_d) if c3_d is not None else "",
                         "a6_s": ", 'a6', {}".format(a6_d) if a6_d is not None else "",
                         "nr_s": ", 'nrid', '{}'".format(self.pars['nr']) if self.pars['nr'] is not None else "",
                         "outdir_s": ", 'outdir', '{}'".format(self.dir),
                         "ell_s": ", 'l_max', {}".format(self.pars['l_max']),
                         "newlogs": ", 'newlogs', {}".format(self.pars['newlogs']),
                         "rho22_SO_resum": ", 'rho22_SO_resum', {}".format(self.pars['rho22_SO_resum']),
                         "rholm": ", 'rholm', '{}'".format(self.pars['rholm']),
                         "H_model": ", 'H_model', '{}'".format(self.pars['H_model']),
                         "rend": ", 'rend', {}".format(rend_val),
                         "nqc_amp": ", 'nqc_amp_params', {}".format(self.pars['nqc_amp_params']) if self.pars['nqc_amp_params'] is not None else 2,
                         "nqc_omg": ", 'nqc_omg_params', {}".format(self.pars['nqc_omg_params']) if self.pars['nqc_omg_params'] is not None else 2,
                         }
        elif self.leob:
            temp_dict = {"code_dir": self.code_dir,
                         "q": self.pars['q'], 
                         "irf": self.pars['iresum'],
                         "chi1": self.pars['chi1z'], "chi2": self.pars['chi2z'],
                         "omg_ap_s": ", 'omg_ap', {}".format(self.pars['initial_frequency']*2.*np.pi) if self.pars['initial_frequency'] is not None else "",
                         "Tmax": ", 'Tmax', {}".format(self.pars['ode_tmax']),
                         "r0": ", 'r0', {}".format(self.pars['r0']) if self.pars['r0'] is not None else "",
                         "c3_s": ", 'cN3LO', {}".format(c3_d) if c3_d is not None else "",
                         "a6_s": ", 'a6', {}".format(a6_d) if a6_d is not None else "",
                         "nr_s": ", 'nrid', '{}'".format(self.pars['nr']) if self.pars['nr'] is not None else "",
                         "outdir_s": ", 'outdir', '{}'".format(self.dir),
                         "ell_s": ", 'l_max', {}".format(self.pars['l_max']),
                         "newlogs": ", 'newlogs', {}".format(self.pars['newlogs']),
                         "rho22_SO_resum": ", 'rho22_SO_resum', {}".format(self.pars['rho22_SO_resum']),
                         "rholm": ", 'rholm', '{}'".format(self.pars['rholm']),
                         "rend": ", 'rend', {}".format(rend_val),
                         "ASS_fact": ", 'ASS_fact', '{}'".format(self.pars['ASS_fact'] if self.pars['ASS_fact'] is not None else ''),
                         "SSPM": ", 'SSPM', '{}'".format(self.pars['SSPM'] if self.pars['SSPM'] is not None else ''),
                         "nqc_amp": ", 'nqc_amp_params', {}".format(self.pars['nqc_amp_params']) if self.pars['nqc_amp_params'] is not None else 2,
                         "nqc_omg": ", 'nqc_omg_params', {}".format(self.pars['nqc_omg_params']) if self.pars['nqc_omg_params'] is not None else 2,
            }
        else:
            temp_dict = {"code_dir": self.code_dir,
                         "q": self.pars['q'], 
                         "chi1": self.pars['chi1z'], "chi2": self.pars['chi2z'],
                         "ecc_s": ", 'ecc', {}".format(self.pars['ecc']),
                         "omg_ap_s": ", 'omg_ap', {}".format(self.pars['initial_frequency']*2.*np.pi),
                         "c3_s": ", 'cN3LO', {}".format(c3_d) if c3_d is not None else "",
                         "a6_s": ", 'a6', {}".format(a6_d) if a6_d is not None else "",
                         "nr_s": ", 'nrid', '{}'".format(self.pars['nr']) if self.pars['nr'] is not None else "",
                         "outdir_s": ", 'outdir', '{}'".format(self.dir),
                         "ell_s": ", 'l_max', {}".format(self.pars['l_max']),
                         "newlogs": ", 'newlogs', {}".format(self.pars['newlogs']),
                         "rho22_SO_resum": ", 'rho22_SO_resum', {}".format(self.pars['rho22_SO_resum']),
                         "rholm": ", 'rholm', '{}'".format(self.pars['rholm']),
                         "H_model": ", 'H_model', '{}'".format(self.pars['H_model']),
                         "Tmax": ", 'Tmax', {}".format(self.pars['ode_tmax']),
                         "rend": ", 'rend', {}".format(rend_val),
                         "nqc_amp": ", 'nqc_amp_params', {}".format(self.pars['nqc_amp_params']) if self.pars['nqc_amp_params'] is not None else 2,
                         "nqc_omg": ", 'nqc_omg_params', {}".format(self.pars['nqc_omg_params']) if self.pars['nqc_omg_params'] is not None else 2,
                         }
        return temp_dict

    def _run_matlab(self):
        temp_dict = self._convert_pars()
        
        code = self.template.safe_substitute(**temp_dict)

        with open(self.dir + 'runmatlab.m', 'w') as f:
            f.write(code)
        
        out = subprocess.run(["{}".format(self.source),  "-nojvm", "-nodesktop", "-nodisplay", "-nosplash", "-batch", "run('{}')".format(self.dir + 'runmatlab.m')], capture_output=True)
        if self.verbose:
            print(out.stdout.decode('UTF-8'))
            print(out.stderr.decode('UTF-8'))
        return 0
    
    def _load_hlm(self):
        mat = h5py.File(self.dir + self.wavefile)
        s   = mat['s/inspl_mrg_rng/ell/emm']

        self._u = mat[mat[s[1][0]]['t'][1][0]][0]
        self._hlm = {}
        for l in range(1, self.pars['l_max']):
            for m in range(l+2):
                h = np.array([(el if isinstance(el, float) else el[0] + 1j*el[1]) for el in mat[mat[s[l][0]]['psi'][m][0]][0]])
                h *= np.sqrt((l + 3)*(l + 2)*(l + 1)*l)
                self._hlm[(l+1, m)] = {'real': h.real, 'imag': h.imag,
                                       'A':    abs(h), 'p': -np.unwrap(np.angle(h)),
                                       'z':    h}
        self.domain = 'Time'

        self._hp, self._hc = wfu.compute_hphc(self._hlm, modes=list(self._hlm.keys()))

        if self.load_insp:
            s = mat['s/inspl/ell/emm']
            self.hlm_inspl = {}
            for l in range(1, self.pars['l_max']):
                for m in range(l+2):
                    h = np.array([(el if isinstance(el, float) else el[0] + 1j*el[1]) for el in mat[mat[s[l][0]]['psi'][m][0]][0]])
                    h *= np.sqrt((l + 3)*(l + 2)*(l + 1)*l)
                    self.hlm_inspl[(l+1, m)] = {'real': h.real, 'imag': h.imag,
                                           'A':    abs(h), 'p': -np.unwrap(np.angle(h)),
                                           'z':    h}
            
            s = mat['s/inspl_mrg/ell/emm']
            self.hlm_inspl_mrg = {}
            for l in range(1, self.pars['l_max']):
                for m in range(l+2):
                    h = np.array([(el if isinstance(el, float) else el[0] + 1j*el[1]) for el in mat[mat[s[l][0]]['psi'][m][0]][0]])
                    h *= np.sqrt((l + 3)*(l + 2)*(l + 1)*l)
                    self.hlm_inspl_mrg[(l+1, m)] = {'real': h.real, 'imag': h.imag,
                                           'A':    abs(h), 'p': -np.unwrap(np.angle(h)),
                                           'z':    h}
        pass

    def _load_dyn(self):
        mat = loadmat(self.dir + self.dynfile)

        N_pt = len(mat['EOB']['T'][0,0])

        self._dyn = {'t'     : np.array([mat['EOB']['T'][0,0][j][0] for j in range(N_pt)]),
                     'r'     : np.array([mat['EOB']['r'][0,0][j][0] for j in range(N_pt)]),
                     'phi'   : np.array([mat['EOB']['phi'][0,0][j][0] for j in range(N_pt)]),
                     'Pr'    : np.array([mat['EOB']['pr'][0,0][j][0] for j in range(N_pt)]),
                     'Prstar': np.array([mat['EOB']['pr_star'][0,0][j][0] for j in range(N_pt)]),
                     'Pphi'  : np.array([mat['EOB']['pph'][0,0][j][0] for j in range(N_pt)]),
                     'E'     : np.array([mat['EOB']['E'][0,0][j][0] for j in range(N_pt)]),
                     'MOmega': np.array([mat['EOB']['Omg'][0,0][j][0] for j in range(N_pt)]),
                     'A'     : np.array([mat['EOB']['metric'][0,0]['A'][0,0][j][0] for j in range(N_pt)]),
                     'B'     : np.array([mat['EOB']['metric'][0,0]['B'][0,0][j][0] for j in range(N_pt)]),
                     'dA'    : np.array([mat['EOB']['metric'][0,0]['dA'][0,0][j][0] for j in range(N_pt)]),
                     'dB'    : np.array([mat['EOB']['metric'][0,0]['dB'][0,0][j][0] for j in range(N_pt)]),
                     'd2A'   : np.array([mat['EOB']['metric'][0,0]['d2A'][0,0][j][0] for j in range(N_pt)]),
                     'd2B'   : np.array([mat['EOB']['metric'][0,0]['d2B'][0,0][j][0] for j in range(N_pt)]),
                     }
        pass

    def clean_data(self):
        os.remove(self.dir + self.wavefile)
        os.remove(self.dir + self.dynfile)
        pass


def matnames(q, chi1=0, chi2=0, ecc=0, omg0=0.02, E0=None, L0=None, r0=None, nr=None, leob=False):
    """
    Name of dynamics, wave mat files
    """
    if nr is not None:
        return 'Waves_{}.mat'.format(nr), 'Dynam_{}.mat'.format(nr)
    else:
        chi1s = f'{chi1:+.6g}' if abs(chi1) >= 0.0001 else '0'
        chi2s = f'{chi2:+.6g}' if abs(chi2) >= 0.0001 else '0'
        base  = 'q{:.6g}_chi1_{:s}_chi2_{:s}'.format(q, chi1s, chi2s)
        if E0 is not None:
            E0s = '_E0_{:.6g}'.format(E0)
            L0s = '_L0_{:.6g}'.format(L0)
            r0s = '_r0_{:.6g}'.format(r0)
            return 'Waves_' + base + E0s + L0s + r0s + '.mat', 'Dynam_' + base + E0s + L0s + r0s + '.mat'
        elif leob:
            if r0 is not None and r0 > 0:
                ids = '_r0_{:.6g}'.format(r0)
            elif omg0 is not None:
                ids = '_omg0_{:.6g}'.format(omg0)
            else:
                raise ValueError('matnames(): provide one of r0 or omg0 for LEOB.')
            return 'Waves_' + base + '_leob.mat', 'Dynam_' + base + ids + '_leob.mat'
        else:
            eccs  = '' if ecc < 1.e-5 else '_ecc_{:.3g}'.format(ecc)
            omgs  = '_omg0_{:.6g}'.format(omg0)
            return 'Waves_' + base + omgs + eccs + '.mat', 'Dynam_' + base + omgs + eccs + '.mat'


def CreateDict(M=1., q=1, 
               chi1z=0., chi2z=0., 
               chi1x=0., chi2x=0.,
               chi1y=0., chi2y=0.,
               f0=0.0035, ecc=1e-8, 
               l_max=2, ode_tmax=1e+6, ode_rend=None,
               r0=None, H_hyp=None, J_hyp=None,
               cN3LO=None, a6c=None,
               nr=None,
               Hmod="std",
               rho22_SO_resum=0,
               newlogs=1,
               rholm="newlogs",
               use_nqc=True,
               nqc_amp=2, nqc_omg=2,
               leob=False,
               iresum=0,
               ASS_fact='none',
               SSPM=4):
        """
        Create the dictionary of parameters for teobresumsmatlab
        """

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
            'initial_frequency'  : f0,
            'output_hpc'         : "no",
            'l_max'              : l_max,
            'ecc'                : ecc,
            'ode_tmax'           : ode_tmax,
            'a6c'                : a6c,
            'cN3LO'              : cN3LO,
            'nr'                 : nr,
            'H_model'            : Hmod,
            'rho22_SO_resum'     : rho22_SO_resum,
            'newlogs'            : newlogs,
            'rholm'              : rholm,
            'ode_rend'           : ode_rend,
            'r0'                 : r0,
            'H_hyp'              : H_hyp,
            'j_hyp'              : J_hyp,
            'leob'               : leob,
            'iresum'             : iresum,
            'ASS_fact'           : ASS_fact,
            'SSPM'               : SSPM,
            'nqc_amp_params'     : nqc_amp,
            'nqc_omg_params'     : nqc_omg,
        }

        if leob:
            if r0 is not None and f0 is not None:
                print('WARNING: both r0 and f0 provided for LEOB; using r0.')
                pardic['initial_frequency'] = None

        return pardic

if __name__ == '__main__':
    print("This is a PyART class for running the TEOBResumS matlab code.")
