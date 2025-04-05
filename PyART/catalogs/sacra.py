import numpy as np
import matplotlib.pyplot as plt
import h5py 
import os, json, requests, time
import pathlib

from ..waveform import Waveform
from ..utils    import cat_utils, wf_utils, utils
from ..analysis.match import condition_td_waveform
from pycbc.types.timeseries import TimeSeries

class Waveform_SACRA(Waveform):
    """
    This class is used to load the SACRA data and store it 
    in a convenient way. Some relevant info is stored in the 
    file sacra.info, while the waveform are downloaded from
    Tullio.
    """
    def __init__(self,
                 ID         = None,   # SACRA do not have "official" ID, but we consider 
                                   # the line of the sacra.info file. If provided, search
                                   # waveform according to this value
                 name       = None,   # alternative to ID
                 path       = './',   # path with SACRA BHNS data (assume same structure as in 
                                      # /data/numrel/SACRA/BHNS)
                 nu_rescale = False,
                 cut_final  = 150     # cut waveform after tmrg+150 (if None, do not cut) 
                 ) -> None:
        
        super().__init__()
        
        sacra_info = self.load_tex_info()
        if ID is not None:
            info = sacra_info[int(ID)-1] # IDs start from 1
            name = info['name']
        elif name is not None:
            for i, sim_info in enumerate(sacra_info):
                if sim_info['name']==name:
                    info = sim_info
                    ID   = i+1
                    break
            if ID is None:
                raise ValueError(f'Name not found: {name}')
        else:
            raise ValueError('Specify the ID or the name')
        
        self.name       = name
        self.ID         = ID
        self.info       = info
        self.path       = path        
        self.nu_rescale = nu_rescale  
        self.cut_final  = cut_final
        self.domain     = 'Time'

        self.load_metadata()
        self.load_hlm()
        
        if self.cut_final is not None:
            t_mrg, _, _, _, _ = self.find_max(kind='global', return_idx=True)
            DeltaT = self.u[-1] - (t_mrg+self.cut_final)
            if DeltaT>0:
                self.cut(DeltaT, from_the_end=True)
    pass

    def load_tex_info(self):
        """
        Load info from TeX table stored in sacra.info,
        taken from Zappa+:1903.11622
        """
        script_path = pathlib.Path(__file__).parent.resolve()
        fname = os.path.join(script_path,'sacra.info')
        with open(fname,'r') as file:
            lines = [line.rstrip() for line in file]
        header = lines[0].replace('#','')
        header = header.replace(' ', '')
        lines  = lines[1:]
        info   = []
        keys   = header.split(',')
        for line in lines:
            line = line.replace('\\', '')
            if len(line)<1:
                continue
            parts = line.split('&')
            sim_dict = {}
            for i, ky in enumerate(keys):
                if i>1:
                    try:
                        sim_dict[ky] = float(parts[i])
                    except Exception as e:
                        #print(f'Error: {e}')
                        sim_dict[ky] = None
                else:
                    sim_dict[ky] = parts[i].replace(' ', '')
            info.append(sim_dict)
        return info
    
    def load_metadata(self):
        q  = self.info['q']
        nu = q/(1+q)**2
        M  = self.info['M']
        m2 = self.info['M_NS'] 
        m1 = M - m2 
        S1 = np.array([0.,0.,self.info['a_BH']*m1*m1])
        S2 = np.array([0.,0.,0.])
        MOmega0 = self.info['MOmega0']
        if MOmega0 is None:
            f0 = None 
        else:
            f0 = MOmega0/np.pi
        metadata = {'name'      : self.info['name'],
                    'ref_time'  : None,
                    'm1'        : m1,
                    'm2'        : m2,
                    'M'         : M,
                    'q'         : q,
                    'nu'        : nu,
                    'S1'        : S1,
                    'S2'        : S2,
                    'chi1x'     : 0.,
                    'chi1y'     : 0.,
                    'chi1z'     : self.info['a_BH'],
                    'chi2x'     : 0.,
                    'chi2y'     : 0.,
                    'chi2z'     : 0.,
                    'LambdaAl2' : 0.,
                    'LambdaBl2' : self.info['Lambda'],
                    'r0'        : None,
                    'e0'        : None,
                    'E0'        : None,
                    'E0byM'     : None,
                    'P0'        : None,
                    'Jz0'       : None,
                    'J0'        : None,
                    'pph0'      : None,
                    'pos1'      : None,
                    'pos2'      : None,
                    'f0v'       : np.array([0.,0.,f0]),
                    'f0'        : f0,
                    'Mf'        : self.info['Xf'],
                    'af'        : self.info['af'],
                    'afv'       : np.array([0.,0.,self.info['af']]),
                    'scat_angle': None,
                    }
        metadata['flags'] = cat_utils.get_flags(metadata)
        cat_utils.check_metadata(metadata, raise_err=True)
        self.metadata = metadata
        pass

    def load_hlm(self):
        subdirs = ['gwa-5', 'gwa0', 'gwa25', 'gwa5', 'gwa75']
        tmp = self.name.replace('-','')+'.d'
        
        fname_eos = tmp.split('Q')[0].replace('.','')
        fname_q   = f"Q{self.metadata['q']:.0f}"
        fname_M   = f"M{self.metadata['m2']:.2f}".replace('.','')
        fname_a   = f"a{self.metadata['chi1z']:.2f}".replace('0.','')

        fname2p = fname_eos+'_'+fname_q+'_'+fname_M+'_'+fname_a+'.d'
        fname2p = fname2p.replace('_M120_', '_M12_')
        fname2p = fname2p.replace('_a00.d', '.d')
        fname2p = fname2p.replace('0.d', '.d')
        
        fname4p = fname_eos+fname_q+fname_a+'.d'
        fname4p = fname4p.replace('a50.d', 'a5.d')
        fname4p = fname4p.replace('a00.d', 'a0.d')
        X = None
        piecewise = None
        for subdir in subdirs:
            full_path = os.path.join(self.path, subdir, fname2p)
            if os.path.exists(full_path):
                X = np.loadtxt(full_path)
                piecewise = 2
                break
        
        if X is None: # if fname not found in 2-piecewise polytrops, search in 4-piecewise
            full_path = os.path.join(self.path, 'gw4pwp', fname4p)
            if os.path.exists(full_path):
                piecewise = 4
                X = np.loadtxt(full_path)
         
        if X is None: # if still None, we cry
            raise FileNotFoundError(f'No {fname2p:20s} or {fname4p:20s} in {self.path}')
        
        if piecewise is not None:
            u  = X[:,0]
            hp = X[:,1]
            hc = X[:,2]
            h  = hp+1j*hc
        if piecewise==2: # see readme.txt in /data/numrel/SACRA/BHNS 
            h /= utils.spinsphericalharm(-2, 2, 2, 0, 0)
        
        if self.nu_rescale:
            h /= self.metadata['nu']
        
        my_zeros = np.zeros_like(u)
        for l in range(2,6):
            for m in range(0,l+1):
                self._hlm[(l,m)] = {'real':my_zeros, 'imag':my_zeros, 
                                    'z':my_zeros, 'A':my_zeros, 'p':my_zeros}
        self._u = u   
        self._hlm[(2,2)] = wf_utils.get_multipole_dict(h)
        
        if self.metadata['f0'] is None:
            f0 = self.get_MOmega0_from_FFT(h,u)
            print(f'Updating f0 using estimate from FT: {f0:.5f}')
            self.metadata['f0']  = f0
            self.metadata['f0v'] = np.array([0.,0.,f0])
        pass 

    def get_MOmega0_from_FFT(self,h,u):
        condition_settings = {'tlen':len(h)*4,
                              'pad_end_frac':0.5,
                              'taper':'sigmoid',
                              'taper_start':0.1,
                              'taper_end'  : 0.05,
                              'taper_alpha':0.02,
                              'taper_alpha_end':0.02,
                              'M':1.,
                              }
        dt = u[1]-u[0]
        hT = TimeSeries(h.real,dt)
        hw = condition_td_waveform(hT, condition_settings)
        hf = hw.to_frequencyseries()
        f  = hf.get_sample_frequencies()
        imaxFT = np.argmax(hf)
        # 1.3 is a tolerance, should not be there in principle 
        f0_FT  = f[imaxFT]/1.3 
        return f0_FT



