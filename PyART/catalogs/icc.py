import os, json, re
import numpy as np
import matplotlib.pyplot as plt

from ..analysis.scattering_angle import ScatteringAngle
from ..waveform    import Waveform 
from ..utils       import os_utils  as os_ut
from ..utils       import wf_utils  as wf_ut
from ..utils       import cat_utils as cat_ut
from ..utils.utils import D1

class Waveform_ICC(Waveform):
    """
    Class to handle ICC waveforms
    The path to be specified is the one that contains the catalog
    (catalog not yet public, this might be updated at some point)
    """
    def __init__(self,
                 path        = './',
                 ID          = '0001',
                 ellmax      = 8,
                 integrate   = False,
                 integr_opts = {},  # options to integrate Psi4
                 load_puncts = False,
                 ):
        super().__init__()
        
        if isinstance(ID, int):
            ID = f'{ID:04}'
        self.path        = path
        self.ID          = ID
        self.sim_path    = os.path.join(self.path, 'ICC_BBH_'+self.ID)
        self.domain      = 'Time'
        self._kind       = 'ICC'
        self.ellmax      = ellmax
        
        # define self.metadata and self.ometadata (original)
        self.load_meta()

        # load puncts
        if load_puncts:
            self.puncts = self.load_puncts()
        else:
            self.puncts = None
        
        if self.puncts is not None:
            if self.puncts['r'][-1]>3:
                puncts = self.puncts
                punct0 = np.column_stack( (puncts['t'], puncts['t'], puncts['x0'], puncts['y0'], puncts['z0']) )
                punct1 = np.column_stack( (puncts['t'], puncts['t'], puncts['x1'], puncts['y1'], puncts['z1']) )
                scat   = ScatteringAngle(punct0=punct0, punct1=punct1, file_format='GRA', 
                                         nmin=2, nmax=5, n_extract=4,
                                         r_cutoff_out_low=25, r_cutoff_out_high=None,
                                         r_cutoff_in_low=25, r_cutoff_in_high=100,
                                         verbose=False)
                self.metadata['scat_angle'] = scat.chi

        # define self.psi4lm, self.t_psi4 and self.r_extr
        self.load_psi4()
        
        if integrate:
            # get hlm and dhlm by psi4-integration
            self.integr_opts = self.integrate_psi4(t_psi4=self.t_psi4, 
                                                   radius=self.r_extr, 
                                                   integr_opts=integr_opts, 
                                                   M=1, modes=[(2,2)])
            i0     = np.where(self.u>=0)[0][0] 
            DeltaT = self.u[i0]-self.u[0]
            self.cut(DeltaT)
            try:
                tmrg, _, _, _ = self.find_max()
                DeltaT_end = self.u[-1]-(tmrg+150)
            except Exception as e:
                print(f'Error while searching merger time: {e}')
                DeltaT_end = 0
            if DeltaT_end>0:
                # leave only 150 M after merge
                self.cut(DeltaT_end, from_the_end=True)
        else:
            self.integr_opts = {}
            self.load_hlm_compute_dothlm()
        
        pass

    def load_meta(self):
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
                'r0'         : ometa['D'],
                'e0'         : ometa['ecc'],
                'E0'         : ometa['E0'],
                'Jz0'        : ometa['J0'],
                'P0'         : np.array(ometa['P0']),
                'J0'         : np.array([0,0,ometa['J0']]),
                'pph0'       : ometa['J0']/(nu*M*M),
                'E0byM'      : ometa['E0'],
                'pos1'       : np.array([0,0,+ometa['D']/2]),
                'pos2'       : np.array([0,0,-ometa['D']/2]),
                'f0v'        : None,
                'f0'         : None,
                'Mf'         : None,
                'af'         : None,
                'afv'        : None,
                'scat_angle' : None, # eventually update while loading punctures
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
            pdict = None
        return pdict

    def extract_value(self, string, key):
        # FIXME: move in utils
        pattern = rf"{key}(\d+(\.\d+|p\d+)?|\d+)"
        match = re.search(pattern, string)
        if match:
            value = match.group(1)
            value = value.replace('p', '.')
            if '.' in value:
                return float(value)
            else:
                return int(value)
        return None
     
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
        
        def load_removing_nans(fname):
            X     = np.loadtxt(fname)
            ncols = len(X[0,:])  
            X     = X[~np.isnan(X).any(axis=1)]
            return X.reshape(-1, ncols)
        
        tmp   = load_removing_nans(files[0])
        t     = tmp[:,0]
        n     = len(t)
        zeros = np.zeros((n,1)) 
        mydict = {}
        for l in range(ellmax+1):
            for m in range(l+1):
                mydict[(l,m)] = {'real':zeros, 'imag':zeros, 'A':zeros,
                                 'p':zeros, 'h':zeros}
        for f in files:
            l   = self.extract_value(f, 'l')
            m   = self.extract_value(f, 'm')
            X   = load_removing_nans(f)
            flm = X[:,1]+1j*X[:,2]
            mydict[(l,m)] = wf_ut.get_multipole_dict(flm)
        return mydict, t, files 

    def load_psi4(self):
        dict_psi4, t, files = self.load_multipole_txtfile('mp_psi4', raise_error=True)
        self._t_psi4 = t 
        self._psi4lm = dict_psi4
        self.r_extr  = self.extract_value(files[0], 'r')
        pass
    
    def load_hlm_compute_dothlm(self):
        """
        Load hlm and compute dothlm
        """
        dict_hlm, u, _ = self.load_multipole_txtfile('mp_strain', raise_error=False)
        self._u   = u
        self._hlm = dict_hlm
        if self.u is not None:
            self._u = self.u-self.u[0]
        
        # compute dothlm
        dothlm = {}
        for k in self.hlm:
            hlm  = self.hlm[k]['h']
            dhlm = D1(hlm, self.u, 4)
            dothlm[k] = wf_ut.get_multipole_dict(dhlm)
        self._dothlm = dothlm 
        pass

################################
# Class for the ICC catalog
################################
#class Catalog(object):



