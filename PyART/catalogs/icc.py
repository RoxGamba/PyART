import os, json, re
import numpy as np
import matplotlib.pyplot as plt

from ..analysis.scattering_angle import ScatteringAngle
from ..waveform  import Waveform 
from ..utils     import os_utils  as os_ut
from ..utils     import wf_utils  as wf_ut
from ..utils     import cat_utils as cat_ut
from ..utils     import utils     as ut

class Waveform_ICC(Waveform):
    """
    Class to handle ICC waveforms
    The path to be specified is the one that contains the catalog
    (catalog not yet public, this might be updated at some point)
    """
    def __init__(self,
                 path            = './',
                 ID              = '0001',
                 ellmax          = 8,
                 tmax_after_peak = 200.,  # cut psi4lm after tmax+this val 
                 integrate       = False,
                 integr_opts     = {},    # options to integrate Psi4
                 load_puncts     = False,
                 nu_rescale      = False,
                 ):
        super().__init__()
        
        if isinstance(ID, int):
            ID = f'{ID:04}'
        self.path            = path
        self.ID              = ID
        self.sim_path        = os.path.join(self.path, 'ICC_BBH_'+self.ID)
        self.domain          = 'Time'
        self._kind           = 'ICC'
        self.ellmax          = ellmax
        self.tmax_after_peak = tmax_after_peak
        self.nu_rescale      = nu_rescale

        # define self.metadata and self.ometadata (original)
        self.load_metadata()

        # load puncts
        if load_puncts:
            self.load_puncts()
        else:
            self.puncts = None
        
        # define self.psi4lm, self.t_psi4 and self.r_extr
        self.load_psi4lm(tmax_after_peak=self.tmax_after_peak)
        
        if integrate:
            # get hlm and dhlm by psi4-integration
            self.integr_opts = self.integrate_psi4lm(t_psi4=self.t_psi4, 
                                                   radius=self.r_extr, 
                                                   integr_opts=integr_opts, 
                                                   M=1, modes=[(2,2)])
            i0     = np.where(self.u>=0)[0][0] 
            DeltaT = self.u[i0]-self.u[0]
            self.cut(DeltaT, cut_dothlm=True, cut_psi4lm=False)
        else:
            self.integr_opts = {}
            self.load_hlm()
        
        if not self.dothlm:
            self.compute_dothlm(factor=self.metadata['nu'], only_warn=True)
        pass

    def load_metadata(self):
        """
        Load metadata (store in JSON)
        """
        meta_file = os.path.join(self.sim_path,'metadata.json')
        with open(meta_file, 'r') as file:
            ometa = json.load(file)
        q  = ometa['q']
        nu = q/(1+q)**2
        M  = 1
        M1 = q*M/(1+q)
        M2 =   M/(1+q)
        S1 = np.array([0, 0, ometa['chi1']*M1*M1])
        S2 = np.array([0, 0, ometa['chi2']*M2*M2])

        Jz = ometa['J0']
        Lz = Jz - S1[2] - S2[2]
        meta = {'name'       : ometa['name'],
                'ref_time'   : 0,
                'm1'         : M1,
                'm2'         : M2,
                'M'          : M,
                'q'          : q,
                'nu'         : nu,
                'S1'         : S1,
                'S2'         : S2,
                'chi1x'      : 0,
                'chi1y'      : 0,
                'chi1z'      : ometa['chi1'],
                'chi2x'      : 0,
                'chi2y'      : 0,
                'chi2z'      : ometa['chi2'],
                'LambdaAl2'  : 0.,
                'LambdaBl2'  : 0.,
                'r0'         : ometa['D'],
                'e0'         : ometa['ecc'],
                'E0'         : ometa['E0'],
                'Jz0'        : ometa['J0'],
                'P0'         : np.array(ometa['P0']),
                'J0'         : np.array([0,0,Jz]),
                'pph0'       : Lz/(nu*M*M),
                'E0byM'      : ometa['E0'],
                'pos1'       : np.array([+ometa['D']/2,0,0]),
                'pos2'       : np.array([-ometa['D']/2,0,0]),
                'f0v'        : None,
                'f0'         : None,
                'Mf'         : None,
                'af'         : None,
                'afv'        : None,
                'scat_angle' : ometa['scat_angle'],
                }
        
        meta['flags'] = cat_ut.get_flags(meta)
        cat_ut.check_metadata(meta)

        self.ometadata = ometa
        self.metadata  = meta
        return
    
    def load_puncts(self, fname='puncturetracker-pt_loc..asc'):
        full_name = os.path.join(self.sim_path,fname)
        if os.path.exists(full_name):
            X  = np.loadtxt(full_name)
            t  = X[:, 8]
            x0 = X[:,22]
            y0 = X[:,32]
            x1 = X[:,23]
            y1 = X[:,33]
            x  = x0-x1
            y  = y0-y1
            r  = np.sqrt(x**2+y**2)
            th = -np.unwrap(np.angle(x+1j*y))
            zeros = 0*t
            pdict = {'t':t,   'r' :r,  'th':th,
                     'x0':x0, 'y0':y0, 'z0':zeros,
                     'x1':x1, 'y1':y1, 'z1':zeros}
        else:
            print("Warning: no punctures' tracks found!")
            pdict = None
        self.puncts = pdict
        return
     
    def load_multipole_txtfile(self, fname_token, raise_error=True, ellmax=None):
        if ellmax is None: ellmax = self.ellmax
        files = os_ut.find_fnames_with_token(self.sim_path, fname_token) 
        if len(files)<1:
            msg = f'No {fname_token}-files found in {self.sim_path}'
            if raise_error:
                raise RuntimeError(msg)
            else:
                print(f'Warning! {msg}. Returning empty vars')
                return {}, None, []
        tmp = ut.safe_loadtxt(files[0])
        t   = tmp[:,0]
        n   = len(t)
        zeros = np.zeros((n,1)) 
        mydict = {}
        for l in range(ellmax+1):
            for m in range(l+1):
                mydict[(l,m)] = {'real':zeros, 'imag':zeros, 'A':zeros,
                                 'p':zeros, 'z':zeros}
        for f in files:
            l   = ut.extract_value_from_str(f, 'l')
            m   = ut.extract_value_from_str(f, 'm')
            X   = ut.safe_loadtxt(f)
            flm = -(X[:,1]+1j*X[:,2])
            if self.nu_rescale:
                flm /= self.metadata['nu']
            mydict[(l,m)] = wf_ut.get_multipole_dict(flm)
        
        return mydict, t, files 

    def load_psi4lm(self, tmax_after_peak=200):
        dict_psi4, t, files = self.load_multipole_txtfile('mp_psi4', raise_error=True)
        self._t_psi4 = t 
        self._psi4lm = dict_psi4
        self.r_extr  = ut.extract_value_from_str(files[0], 'r')
        
        if tmax_after_peak is not None:
            try:
                tmax_psi4,_,_,_ = self.find_max(wave='psi4lm', height=1e-04)
                DeltaT_end = self.t_psi4[-1]-(tmax_psi4+tmax_after_peak)
            except Exception as e:
                print(f'Error while searching max of psi4 time: {e}')
                DeltaT_end = 0
            if DeltaT_end>0:
                self.cut(DeltaT_end, from_the_end=True, cut_psi4lm=True)
        pass
    
    def load_hlm(self):
        """
        Load hlm and compute dothlm
        """
        dict_hlm, u, _ = self.load_multipole_txtfile('mp_strain', raise_error=False)
        self._u   = u
        self._hlm = dict_hlm
        if self.u is not None:
            self._u = self.u-self.u[0]
        pass



