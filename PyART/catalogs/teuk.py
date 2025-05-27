import numpy as np; import os
import h5py; import json
from glob import glob
from ..waveform import  Waveform
from ..utils.wf_utils import get_multipole_dict

class Waveform_Teuk(Waveform):
    """
    Class to handle Teukode test-mass waveforms.
    """
    def __init__(self,
                 path,
                 ellmax  = 5,
                 datafmt = 'npz', # 'npz' or 'teuk_mAll'
                 input_meta = None
                 ):
        super().__init__()
        self.path    = path
        self.ellmax  = ellmax
        self._kind   = 'Teukode'
        self.domain  = 'Time'
        
        if datafmt in ['npz', 'teuk_mAll']:
            self.datafmt = datafmt
        else:
            raise ValueError(f'Unknown data format: {datafmt:s}')

        self.load_metadata()

        if input_meta is not None:
            for ky in input_meta:
                self.metadata[ky] = input_meta[ky]
        
        self.load_dynamics()
        self.load_hlm()
        pass
    
    def load_metadata(self):
        """
        Load metadata from the Teukode parfile.
        """
        # find the parfile
        parfile = glob(os.path.join(self.path, "*.par"))
        if len(parfile) == 0:
            #raise RuntimeError("No parfile found!")
            self.metadata = {}
            return 

        if len(parfile) > 1:
            raise RuntimeError("Multiple parfiles found!")

        with open(parfile[0], 'r') as f:
            lines = [line.strip() for line in f if '=' in line and not line.lstrip().startswith('#')]
        
        mtdt = {}
        for line in lines:
            k, v = line.split('=')
            k = k.replace(' ', '')
            try:
                v_converted = int(v)
            except ValueError:
                try:
                    v_converted = float(v)
                except ValueError:
                    v_converted = v
            mtdt[k] = v_converted

        self.metadata = mtdt
        pass

    def load_dynamics(self):
        """
        Load the dynamics and energy from the RWZ output.
        """
        
        if self.datafmt=='npz':
            fname_dyn  = os.path.join(self.path,'teuk_dyn.npz')
            dyn_loaded = np.load(fname_dyn,allow_pickle=True)
            self._dyn  = dyn_loaded['arr_0'].item()
        
        elif self.datafmt=='teuk_mAll':
            subdirs = os.listdir(self.path)
            for sd in subdirs:
                if sd[:5]=='kerr_':
                    full_dyn_path = os.path.join(self.path,sd)
                    traj = np.loadtxt(os.path.join(full_dyn_path, 'traj.dat'))
                    fpr  = np.loadtxt(os.path.join(full_dyn_path, 'pr.dat'))
                    fprr = np.loadtxt(os.path.join(full_dyn_path, 'pr_star.dat'))
                    fpph = np.loadtxt(os.path.join(full_dyn_path, 'pph.dat'))
                    fH   = np.loadtxt(os.path.join(full_dyn_path, 'H.dat'))
                    dyn = {'t'  :traj[:,0], 'r'  :traj[:,1],
                           'ph' :traj[:,2], 'Omg':traj[:,3], 
                           'pr' : fpr[:,1], 'prr':fprr[:,1], 
                           'pph':fpph[:,1], 'H'  :  fH[:,1],
                           'tp_dynamics':False}
            self._dyn = dyn
        pass

    def load_hlm(self):
        # load data 
        if self.datafmt=='npz':
            fname_hlm  = os.path.join(self.path,'teuk_hlm.npz')
            hlm_loaded = np.load(fname_hlm,allow_pickle=True)
            
            hlm_z   = hlm_loaded['arr_0'].item()
            self._u = hlm_z['t']
            del hlm_z['t']

        elif self.datafmt=='teuk_mAll':
            subdirs = os.listdir(self.path)
            hlm_z = {}
            for sd in subdirs:
                if sd[:9]=='teuk_HH10':
                    mstr = sd.split('_')[-1]
                    m = int(mstr.replace('m',''))
                    for l in range(2,self.ellmax+1):
                        if l<m:
                            continue
                        hname = f'out0d/h_Yl{l:d}m{m:d}_x10.0000.dat'
                        full_path_h = os.path.join(self.path,sd,hname) 
                        X = np.loadtxt(full_path_h)
                        hlm_z[(l,m)] = X[:,1] + 1j*X[:,2]
                        if l==2 and m==2:
                            self._u = X[:,0]
        
        # convert to PyART hlm-dict
        hlm = {}
        for lm in hlm_z:
            if lm[0]<=self.ellmax:
                hlm[lm] = get_multipole_dict(hlm_z[lm])
        self._hlm = hlm

        shift   = self._compute_waveform_shift()
        self._u = self.u+shift
        pass

    def _compute_waveform_shift(self):
        
        rBL = self.dyn['r'][0]
        tBL = self.dyn['t'][0]
        
        M = self.metadata['kerr_mbh']
        S = self.metadata['grid_xmax']
        a = self.metadata['kerr_abh']

        if abs(M - a) < 1e-13:
            rs = rBL - (2 * M * (M + (-rBL + M) * np.log(rBL - M))) / (rBL - M)
        else:
            sqrtma = np.sqrt(abs(M * M - a * a))
            rplus = M + sqrtma
            rmins = M - sqrtma
            oodr = 2 * M / (rplus - rmins)
            tmp = (rplus * np.log(rBL - rplus) - rmins * np.log(rBL - rmins)) * oodr
            rs = rBL + tmp
        tki = tBL - rBL + rs
        R   = S * rBL / (S + rBL)
        traj_shift = -tki - 4 * M * np.log(1. - R / S) + R * R / (S - R)
        
        rho = S
        shift = -rho -4*M*np.log( (S*rho+2*M*rho-2*M*S)/S )+2*M*np.log(2*M)-traj_shift
        return shift 



