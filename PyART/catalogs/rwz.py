import numpy as np
import os
import h5py
import json
from glob import glob
from ..waveform import Waveform
from ..utils.wf_utils import get_multipole_dict

class Waveform_RWZ(Waveform):
    """
    Class to handle RWZ test-mass waveforms.
    https://arxiv.org/abs/1107.5402
    https://bitbucket.org/BBHLab/rwzhyp/src/master/
    """

    def __init__(self, 
                 path, 
                 r_ext          = None,
                 ellmax         = 4, 
                 mmin           = 1,
                 cut_u          = None,
                 par_rel_path   = None, 
                 parfile_tokens = []):

        super().__init__()
        
        if not os.path.exists(path):
            raise RuntimeError(f'Path not found: {path}')

        self.path         = path
        self.ellmax       = ellmax
        self.mmin         = mmin
        self.cut_u        = cut_u
        self._kind        = "RWZ"
        self.domain       = "Time"
        self.par_rel_path = par_rel_path
             
        self.load_metadata(tokens=parfile_tokens)
        self.load_dynamics()
        self.load_hlm(r_ext=r_ext)
        
        pass

    def load_metadata(self, tokens=[]):
        """
        Load metadata from the RWZ parfile.
        """
        # find the parfile
        pattern = '*' + ''.join(x+'*' for x in tokens) + '.par'
        if self.par_rel_path is None:
            parfile = glob(os.path.join(self.path, pattern))
        else:
            parfile = glob(os.path.join(self.path, self.par_rel_path, pattern))
        
        if len(parfile) == 0:
            raise FileNotFoundError("No parfile found!")
        if len(parfile) > 1:
            raise RuntimeError("Multiple parfiles found!")

        with open(parfile[0], "r") as f:
            lines = f.readlines()
        
        mtdt = {}
        for line in lines:
            k, v = self._clean_parfile_line(line)
            mtdt[k] = v

        self.metadata = mtdt
        pass
    
    def _eventual_int_float_conv(self, s):
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s
        
    def _clean_parfile_line(self,line):
        k, v = line.split("=")
        to_rm_list = ["'", '.d0', 'd0', '\n']
        
        k = k.strip(' ').strip('\t').strip(' ')
        if '[it]' in k:
            k = k.replace('[it]','').strip(' ') + '_it'
        if '[pt]' in k:
            k = k.replace('[pt]','').strip(' ') + '_pt'
        
        v = v.strip(' ').strip('\t').replace(' ', '')
        for to_rm in to_rm_list:
            k = k.replace(to_rm, '') 
            v = v.replace(to_rm, '')
        if '.false.' in v:
            v = False
        elif '.true.' in v:
            v = True
        v = self._eventual_int_float_conv(v)    
        return k,v 

    def load_dynamics(self):
        """
        Load the dynamics and energy from the RWZ output.
        """

        dynf = os.path.join(self.path, "trajectory.dat")
        Ef = os.path.join(self.path, "energy.dat")

        with open(dynf, "r") as f:
            t, phi, r, pph, pr, phidot, rdot, pphdot, r_star, pr_star = np.loadtxt(
                f, skiprows=1, unpack=True
            )

        with open(Ef, "r") as f:
            _, _, _, E = np.loadtxt(f, skiprows=1, unpack=True)

        self._dyn = {
            "t": t,
            "phi": phi,
            "r": r,
            "pph": pph,
            "pr": pr,
            "phidot": phidot,
            "rdot": rdot,
            "pphdot": pphdot,
            "r_star": r_star,
            "pr_star": pr_star,
            "E": E,
        }

        pass

    def load_hlm(self, r_ext=None, ellmax=None):
        
        if not hasattr(self, 'metadata'):
            raise RuntimeError('Load metadata before loading the hlm!')
        
        if r_ext is None:
            r_ext = f'r0{self.metadata["jmax"]:d}'
        elif isinstance(r_ext, int):
            r_ext = f'r0{r_ext}'
        elif isinstance(r_ext, str):
            r_ext = r_ext # 'scri' or something like 'r05001'
        else:
            raise ValueError(f'Unknown r_ext: {r_ext}')
        
        if ellmax == None:
            ellmax = self.ellmax

        modes = []
        for l in range(2,ellmax+1):
            for m in range(self.mmin,l+1):
                modes.append( (l,m) ) 

        # cut the waveform to the desired time interval
        f22 = os.path.join(self.path, f'Psi_l2_m2_{r_ext}.dat')
        with open(f22, "r") as f:
            data = np.loadtxt(f)
        tmp_t = np.array(data[:, 1])
        
        if self.cut_u is not None:
            n0 = np.argwhere(tmp_t>=self.cut_u)[0][0]
        else:
            n0 = 0

        self._u = tmp_t[n0:]
        self._t = self._u  # FIXME: should we use another time?

        dict_hlm = {}
        nu = self.metadata['nu']
        for l, m in modes:
            filename = os.path.join(self.path, f'Psi_l{l}_m{m}_{r_ext}.dat')
            with open(filename, "r") as f:
                data = np.loadtxt(f)

            sqrtL = np.sqrt((l + 2) * (l + 1) * l * (l - 1))  
            #t = np.array(data[cut_N:,1])
            h = np.array((data[n0:,2] + 1j*data[n0:,3])) * sqrtL
            
            dict_hlm[(l, m)] = get_multipole_dict(h)

        self._hlm = dict_hlm
        pass


