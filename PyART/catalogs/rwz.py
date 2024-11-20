import numpy as np; import os
import h5py; import json
from glob import glob
from ..waveform import  Waveform
#from .cat_utils import check_metadata

class Waveform_RWZ(Waveform):
    """
    Class to handle RWZ test-mass waveforms.
    https://arxiv.org/abs/1107.5402
    https://bitbucket.org/BBHLab/rwzhyp/src/master/
    """
    def __init__(self,
                 path,
                 r_ext,
                 cut_N=None, 
                 cut_U=None,
                 ellmax=4,
                 load_m0=False
                 ):
        super().__init__()
        self.path   = path
        self.r_ext  = r_ext
        self.ellmax = ellmax
        self.cut_N  = cut_N
        self.cut_U  = cut_U
        self._kind  = 'RWZ'
        self.domain = 'Time'

        self.check_cut_consistency()

        self.load_metadata()
        self.load_hlm(load_m0=load_m0)
        pass
    
    def load_metadata(self):
        """
        Load metadata from the RWZ parfile.
        """
        # find the parfile
        parfile = glob(os.path.join(self.path, "*.par"))
        if len(parfile) == 0:
            raise RuntimeError("No parfile found!")
        if len(parfile) > 1:
            raise RuntimeError("Multiple parfiles found!")

        with open(parfile[0], 'r') as f:
            lines = f.readlines()
        
        mtdt = {}
        for line in lines:
            k, v = line.split('=')
            k = k.strip("'").strip(" ")
            v = v.strip("'").strip(" ").replace(".d0", "")
            mtdt[k] = v

        self.metadata = mtdt
        pass

    def load_dynamics(self):
        """
        Load the dynamics and energy from the RWZ output.
        """

        dynf = os.path.join(self.path, "trajectory.dat")
        Ef   = os.path.join(self.path, "energy.dat")

        with open(dynf, 'r') as f:
            t, phi, r, pph, pr, phidot, rdot, pphdot, r_star, pr_star = np.loadtxt(f, skiprows=1, unpack=True)
        
        with open(Ef, 'r') as f:
            _, _, _, E = np.loadtxt(f, skiprows=1, unpack=True)
        
        self._dyn = {'t': t, 'phi': phi, 
                         'r': r, 'pph': pph, 
                         'pr': pr, 'phidot': phidot, 
                         'rdot': rdot, 'pphdot': pphdot, 
                         'r_star': r_star, 'pr_star': pr_star,
                         'E': E}

        pass

    def load_hlm(self, ellmax=None, load_m0=False):
        if ellmax==None: ellmax = self.ellmax
    
        if not hasattr(self, 'metadata'):
            raise RuntimeError("Load metadata before loading the hlm!")
        
        nu  = self.metadata['nu']

        from itertools import product
        modes = [(l, m) for l, m in product(range(2, ellmax+1), range(-ellmax, ellmax+1)) 
                 if (m!=0 or load_m0) and l >= np.abs(m)
                 ]

        # cut the waveform to the desired time interval
        f22 = os.path.join(self.path,f"/Psi_l2_m2_r{self.r_ext}.dat")
        with open(f22, 'r') as f:
            data = np.loadtxt(f)
        tmp_t = np.array(data[:, 1])

        self.check_cut_consistency()    
        if self.cut_N is None: self.cut_N = np.argwhere(tmp_t>=self.cut_U)[0][0] 
        if self.cut_U is None: self.cut_U = tmp_t[self.cut_N]   

        self._u  = tmp_t[self.cut_N:]
        self._t  = self._u # FIXME: should we use another time?   

        dict_hlm = {}
        for (l, m) in modes:
            filename = os.path.join(self.path,f"/Psi_l{l}_m{m}_r{self.r_ext}.dat")
            with open(filename, 'r') as f:
                data = np.loadtxt(f)
            
            psitoh = np.sqrt((l+2)*(l+1)*l*(l-1))
            
            t    = np.array(data[:, 1])
            Reh  = np.array(data[:, 2])/nu*psitoh
            Imh  = np.array(data[:, 3])/nu*psitoh
            Ch   = Reh + 1j*Imh
            Amp  = np.sqrt(Reh**2 + Imh**2)
            phi  = -np.unwrap(np.arctan2(Imh, Reh))

            dict_hlm[(l, m)] = {'real': Reh, 'imag': Imh, 
                           'A': Amp, 'p': phi,
                           'z':Ch}
        
        self.hlm = dict_hlm
        pass
