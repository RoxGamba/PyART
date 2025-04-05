import numpy as np
import os, json 

from ..analysis.scattering_angle import ScatteringAngle
from ..waveform import  Waveform
from ..utils    import wf_utils  as wf_ut
from ..utils    import cat_utils  as cat_ut

class Waveform_GRAHyp(Waveform):
    """
    Class to handle GRAthena++ wafeforms 
    from the hyperbolic dataset. Assume data
    store in grathena_bbhhyper/runs/data/DSET/.
    Temporary, should be merged in Waveform_GRA
    once that the format is fixed
    """
    def __init__(self,
                 path        = './',
                 ID          = '0001',
                 load_puncts = False,
                 nu_rescale  = False,
                 ):
        super().__init__()
        
        if isinstance(ID, int):
            ID = f'{ID:04}'

        self.path        = path
        self.ID          = ID
        self.sim_path    = os.path.join(self.path, 'BBH_GRAHYP_'+self.ID)
        self.domain      = 'Time'
        self._kind       = 'GRAHYP'
        self.nu_rescale  = nu_rescale

        # define self.metadata and self.ometadata (original)
        self.load_metadata()
        
        if load_puncts:
            self.load_puncts()
        else:
            self.puncts = None
            
        # define self.psi4lm, self.t_psi4
        self.load_psi4lm()
        
        # define self.hlm and self.u
        self.load_hlm()

        # get hlm and dhlm by psi4-integration
        self.compute_dothlm(factor=self.metadata['nu'])
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
        M1 = ometa['M1_ADM']
        M2 = ometa['M2_ADM']
        M  = M1 + M2
        S1 = np.array(ometa['par_S_plus'])
        S2 = np.array(ometa['par_S_minus'])
        
        chi1 = S1/(M1**2)
        chi2 = S2/(M2**2)
        
        Lz = ometa['L_ADM']
        Jz = Lz + S1[2] + S2[2]
        D  = ometa['initial_separation']
        offset_x = ometa['center_offset'][0]
        meta = {'name'       : ometa['name'],
                'ref_time'   : 0,
                'm1'         : M1,
                'm2'         : M2,
                'M'          : M,
                'q'          : q,
                'nu'         : nu,
                'S1'         : S1,
                'S2'         : S2,
                'chi1x'      : chi1[0],
                'chi1y'      : chi1[1],
                'chi1z'      : chi1[2],
                'chi2x'      : chi2[0],
                'chi2y'      : chi2[1],
                'chi2z'      : chi2[2],
                'LambdaAl2'  : 0.,
                'LambdaBl2'  : 0.,
                'r0'         : ometa['initial_separation'],
                'e0'         : None,
                'E0'         : ometa['M_ADM'],
                'Jz0'        : Jz/(M**2),
                'P0'         : np.array(ometa['par_P_plus']),
                'J0'         : np.array([0,0,Jz]),
                'pph0'       : Lz/(nu*M*M),
                'E0byM'      : ometa['M_ADM']/M,
                'pos1'       : np.array([+D/2+offset_x,0,0]),
                'pos2'       : np.array([-D/2+offset_x,0,0]),
                'f0v'        : None,
                'f0'         : None,
                'Mf'         : None,
                'af'         : None,
                'afv'        : None,
                'scat_angle' : ometa['scat_angle']['chi'],
                }
        
        meta['flags'] = cat_ut.get_flags(meta)
        cat_ut.check_metadata(meta)

        self.ometadata = ometa
        self.metadata  = meta
        return
    
    def load_puncts(self, fnames=['puncture_0.txt', 'puncture_1.txt']):
        full_name_0 = os.path.join(self.sim_path,fnames[0])
        full_name_1 = os.path.join(self.sim_path,fnames[1])
        if os.path.exists(full_name_0) and os.path.exists(full_name_1):
            X0 = np.loadtxt(full_name_0)
            X1 = np.loadtxt(full_name_1)
            t  = X0[:,1]
            x0 = X0[:,2]
            y0 = X0[:,3]
            z0 = X0[:,4]
            x1 = X1[:,2]
            y1 = X1[:,3]
            z1 = X1[:,4]
            x  = x0-x1
            y  = y0-y1
            r  = np.sqrt(x**2+y**2) # in plane
            th = -np.unwrap(np.angle(x+1j*y))
            pdict = {'t':t,   'r' :r,  'th':th,
                     'x0':x0, 'y0':y0, 'z0':z0,
                     'x1':x1, 'y1':y1, 'z1':z1}
        else:
            print("Warning: no punctures' tracks found!")
            pdict = None
        self.puncts = pdict
        return

    def load_psi4lm(self):
        """
        Load only (2,2) mode at the moment
        """
        r_extr = self.ometadata['r_extr']
        fname = f'psi4_l2m2_r{r_extr:.2f}.txt'
        X = np.loadtxt(os.path.join(self.sim_path,fname))
        self._t_psi4 = X[:,0]
        
        self._psi4lm = {}
        psi4_l2m2 = X[:,1] + 1j*X[:,2]
        myzeros = np.zeros_like(psi4_l2m2)
        ellmax = 5 # fill with zeros
        for l in range(2,ellmax+1):
            for m in range(0,l+1):
                if l==2 and m==2:
                    self._psi4lm[(l,m)] = wf_ut.get_multipole_dict(psi4_l2m2)
                else:
                    self._psi4lm[(l,m)] = {'z':myzeros, 'A':myzeros,
                                           'p':myzeros, 'real':myzeros,
                                           'imag':myzeros}
        pass
    
    def load_hlm(self):
        sim_dir = os.path.join(self.path,'BBH_GRAHYP_'+self.ID)
        self._hlm = {}
        for l in range(2,6):
            for m in range(0,l+1):
                fname = f'h_l{l:d}_m{m:d}.txt'
                X = np.loadtxt(os.path.join(sim_dir,fname))
                z = X[:,1]+1j*X[:,2]
                if self.nu_rescale:
                    z /= self.metadata['nu']
                self._hlm[(l,m)] = wf_ut.get_multipole_dict(z)
        self._u = X[:,0]
        return









