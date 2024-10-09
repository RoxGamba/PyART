import os, json, re
import numpy as np
import matplotlib.pyplot as plt

from ..waveform    import Waveform, WaveIntegrated
#from ..analysis.scattering_angle import ScatteringAngle
from ..utils import os_utils as os_ut
from ..utils import wf_utils as wf_ut

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
                 integr_opts = {'f0':0.001},  # options to integrate Psi4
                 #load_puncts = True,
                 ):
        super().__init__()
        
        if isinstance(ID, int):
            ID = f'{ID:04}'
        self.path        = path
        self.ID          = ID
        #self.integr_opts = integr_opts
        self.sim_path    = os.path.join(self.path, 'ICC_BBH_'+self.ID)
        self.domain      = 'Time'
        self._kind       = 'ICC'
        self.ellmax      = ellmax

        # define self.metadata and self.ometadata (original)
        self.load_meta()
        
        # define self.psi4lm, self.t_psi4 and self.r_extr
        self.load_psi4()
        
        if integrate:
            # get hlm and dhlm by psi4-integration
            self.integrate_psi4(t_psi4=self.t_psi4, radius=self.r_extr, 
                                integr_opts=integr_opts, M=1, modes=[(2,2)])
            i0     = np.where(self.u>=0)[0][0] 
            DeltaT = self.u[i0]-self.u[0]
            self.cut(DeltaT)
            tmrg, _, _, _ = self.find_max()
            DeltaT_end = self.u[-1]-(tmrg+150)
            if DeltaT_end>0:
                # leave only 150 M after merge
                self.cut(DeltaT_end, from_the_end=True)
        else:
            self.load_hlm()
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
        meta = {'name'     : ometa['name'],
                'ref_time' : 0,
                'm1'       : M1,
                'm2'       : M2,
                'M'        : M,
                'q'        : q,
                'nu'       : nu,
                'chi1x'    : 0,
                'chi1y'    : 0,
                'chi1z'    : ometa['chi1'],
                'chi2x'    : 0,
                'chi2y'    : 0,
                'chi2z'    : ometa['chi2'],
                'r0'       : ometa['D'],
                'e0'       : ometa['ecc'],
                'E0'       : ometa['E0'],
                'Jz0'      : ometa['J0'],
                'P0'       : np.array(ometa['P0']),
                'J0'       : np.array([0,0,ometa['J0']]),
                'pph0'     : ometa['J0']/(nu*M*M),
                'E0byM'    : ometa['E0'],
                'pos1'     : np.array([0,0,+ometa['D']/2]),
                'pos2'     : np.array([0,0,-ometa['D']/2]),
                'f0v'      : None,
                'f0'       : None,
                'Mf'       : None,
                'af'       : None,
                'afv'      : None,
                }
        self.ometadata = ometa
        self.metadata  = meta
        return
    
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

        tmp   = np.loadtxt(files[0])
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
            X   = np.loadtxt(f)
            flm = X[:,1]+1j*X[:,2]
            mydict[(l,m)] = wf_ut.get_multipole_dict(flm)
        
        return mydict, t, files 

    def load_psi4(self):
        dict_psi4, t, files = self.load_multipole_txtfile('mp_psi4', raise_error=True)
        self._t_psi4 = t 
        self._psi4lm = dict_psi4
        self.r_extr  = self.extract_value(files[0], 'r')
        pass
    
    def load_hlm(self):
        dict_hlm, u, _ = self.load_multipole_txtfile('mp_strain', raise_error=False)
        self._u   = u
        self._hlm = dict_hlm
        if self.u is not None:
            self._u = self.u-self.u[0]
        pass

################################
# Class for the ICC catalog
################################
#class Catalog(object): #FIXME: old code, to FIX!
#    def __init__(self, 
#                 path        = './',
#                 ID          = '0001',
#                 ell_emms    = 'all',
#                 ellmax      = 4,
#                 nonspinning = False, # load only nonspinning
#                 integr_opts = None,
#                 load_puncts = True
#                 ) -> None:
#        simulations       = Simulations(path=basepath, icc_add_meta=True, metadata_ext='.bbh')
#        self.catalog_meta = simulations.data
#        self.nonspinning  = nonspinning
#        self.integr_opts  = integr_opts
#
#        self.ellmax   = ellmax
#        self.ell_emms = ell_emms
#        if self.ell_emms == 'all': 
#            self.modes = [(ell, emm) for ell in range(2,self.ellmax+1) for emm in range(-ell, ell+1)]
#        else:
#            self.modes = self.ell_emms # TODO: add check on input
#        data = []
#        for i, meta in enumerate(self.catalog_meta):
#            spin_asum = abs(float(meta['initial-bh-spin1z'])) + abs(float(meta['initial-bh-spin2z']))
#            if not self.nonspinning or spin_asum<1e-14:
#                wave = WaveIntegrated(path=meta['path'], r_extr=meta['r0'], modes=self.modes, M=meta['M'],
#                                      integr_opts=integr_opts)
#                if load_puncts:
#                    tracks = self.load_tracks(path=meta['path'])
#                    punct0 = np.column_stack( (tracks['t'], tracks['t'], tracks['x0'], tracks['y0'], tracks['z0']) )
#                    punct1 = np.column_stack( (tracks['t'], tracks['t'], tracks['x1'], tracks['y1'], tracks['z1']) )
#                    scat_NR = ScatteringAngle(punct0=punct0, punct1=punct1, file_format='GRA', nmin=2, nmax=5, n_extract=4,
#                                           r_cutoff_out_low=25, r_cutoff_out_high=None,
#                                           r_cutoff_in_low=25, r_cutoff_in_high=100,
#                                           verbose=False)
#                    scat_info = {'chi':scat_NR.chi, 'chi_fit_err':scat_NR.fit_err}
#                else:
#                    tracks    = {}
#                    scat_info = {}
#                
#                sim_data            = lambda:0
#                sim_data.meta       = meta
#                sim_data.wave       = wave
#                sim_data.tracks     = tracks
#                sim_data.scat_info  = scat_info
#                data.append(sim_data)
#        self.data = data
#        pass
#    
#    def load_tracks(self, path):
#        # FIXME: very specific
#        fname = 'puncturetracker-pt_loc..asc'
#        X  = np.loadtxt(os.path.join(path,fname))
#        t  = X[:, 8]
#        x0 = X[:,22]
#        y0 = X[:,32]
#        x1 = X[:,23]
#        y1 = X[:,33]
#        x  = x0-x1
#        y  = y0-y1
#        r  = np.sqrt(x**2+y**2)
#        th = -np.unwrap(np.angle(x+1j*y))
#        return {'t':t, 'x0':x0, 'x1':x1, 'y0':y0, 'y1':y1, 'r':r, 'th':th, 'z0':0*t, 'z1':0*t}
#
#    def get_simlist(self):
#        """
#        Get list of simulations' names
#        """
#        simlist = []
#        for meta in self.catalog_meta:
#            simlist.append(meta['name'])
#        return simlist
#
#    def idx_from_value(self,value,key='name',single_idx=True):
#        """ 
#        Return idx with metadata[idx][key]=value.
#        If single_idx is False, return list of indeces 
#        that satisfy the condition
#        """
#        idx_list = []
#        for idx, meta in enumerate(self.catalog_meta):
#            if meta[key]==value:
#                idx_list.append(idx)
#        if len(idx_list)==0: 
#            return None
#        if single_idx:
#            if len (idx_list)>1:
#                raise RuntimeError(f'Found more than one index for value={value} and key={key}')
#            else:
#                return idx_list[0]
#        else: 
#            return idx_list
#
#    def meta_from_name(self,name):
#        return self.catalog_meta[self.idx_from_value(name)]
#    
#    def wave_from_name(self,name):
#        return self.waves[self.idx_from_value(name)]
#

