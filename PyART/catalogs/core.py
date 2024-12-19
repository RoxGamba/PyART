import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import h5py 
import glob
import os

from ..waveform import  Waveform

## Conversion dictionary
conversion_dict_floats = {
    'database_key': 'name',
    'simulation_name': 'alternative_names',
    'id_mass': 'M',
    'id_rest_mass': 'Mb',
    'id_mass_ratio': 'q',
    'id_ADM_mass': 'E0',
    'id_ADM_angularmomentum': 'J0',
    'id_eos': 'EOS',
    'id_kappa2T': 'k2T',
    'id_eccentricity': 'ecc',
}

conversion_dict_vectors = {
    'id_spin_starA': 'S1',
    'id_spin_starB': 'S2',
    'id_Lambdaell_starA': 'Lambda_ell_A',
    'id_Lambdaell_starB': 'Lambda_ell_B',
}

def vector_string_to_array(vstr):
    return np.array([float(v) for v in vstr.split(',')])

class Waveform_CoRe(Waveform):

    def __init__(self,
                 path='../dat/CoRe/',
                 ID      = 'BAM:0001',
                 run     = 'R01',
                 kind     = 'h5',
                 mtdt_path=None,
                 ell_emms='all',
                 download=False,
                 nu_rescale=False,
                 )->None:

        super().__init__()
        self.ID = ID.replace(':','_')
        self.run = run
        self.ell_emms = ell_emms
        self.core_data_path = os.path.join(path,ID)
        self.metadata = None

        if os.path.exists(self.core_data_path) == False:
            if download:
                print("The path ", self.core_data_path, " does not exist.")
                print("Downloading the simulation from the CoRe database.")
                self.download_simulation(ID=self.ID, path=path)
            else:
                print("Use download=True to download the simulation from the CoRe database.")
                raise FileNotFoundError(f"The path {self.core_data_path} does not exist.")
        
        self.simpath = self.core_data_path
        self.runpath = os.path.join(self.simpath, run)
        # read metadata
        if mtdt_path is None:
            mtdt_path = os.path.join(self.simpath,'metadata_main.txt')
        self.metadata = self.load_metadata(mtdt_path)

        # read data
        # self.load_hlm(self.runpath, kind)

        pass

    def download_simulation(self, ID='BAM_0001', path='.',protocol='https',verbose=False):
        pre = {'ssh': 'git@', 'https': 'https://'}
        sep = {'ssh': ':'   , 'https': '/'}
        server = 'core-gitlfs.tpi.uni-jena.de'
        gitbase = 'core_database'
        if protocol not in pre.keys():
            raise NameError("Protocol not supported!")
        git_repo = '{}{}{}{}/{}.git'.format(pre[protocol],server,
                                        sep[protocol],gitbase,ID)
        print('git-clone {} ...'.format(git_repo))
        os.system('git clone '+git_repo)
        self.core_data_path = os.path.join(path,ID)
        # pull with lfs
        os.system('cd {}; git lfs pull'.format(self.core_data_path))

    def load_metadata(self, mtdt_path):
        metadata = {}
        with open(mtdt_path, "r") as f:
            lines = [l for l in f.readlines() if l.strip()] # rm empty
            for line in lines:
                if line[0]=="#": continue
                line               = line.rstrip("\n")
                key, val           = line.split("= ")
                key                = key.strip()
                if key in conversion_dict_floats.keys():
                    metadata[conversion_dict_floats[key]] = val.strip()
                elif key in conversion_dict_vectors.keys():
                    metadata[conversion_dict_vectors[key]] = vector_string_to_array(val)
                else:
                    metadata[key] = val.strip()

        return metadata
    
    def read_h(self, basepath, kind):
        if kind == 'txt':
            self.read_h_txt(basepath)
        elif kind == 'h5':
            self.read_h_h5(basepath)
        else:
            raise NameError('kind not recognized')
        
    def read_h_h5(self, basepath):
        self.dfile = os.path.join(basepath,'data.h5')
        dset = {}
        with h5py.File(self.dfile, 'r') as fn:
            for g in fn.keys():
                dset[g] = {}
                for f in fn[g].keys():
                    dset[g][f] = fn[g][f][()]
        try:
            uM   = dset['rh_22'][:,0]
            RehM = dset['rh_22'][:,1]
            ImhM = dset['rh_22'][:,2]
            Momg = dset['rh_22'][:,5]
            aM   = dset['rh_22'][:,6]
            phi  = dset['rh_22'][:,7]
        except:
            uM   = dset['rh_22'][:,0]
            RehM = dset['rh_22'][:,1]
            ImhM = dset['rh_22'][:,2]
            Momg = dset['rh_22'][:,3]
            aM   = dset['rh_22'][:,4]
            phi  = dset['rh_22'][:,5]

        # TODO: add option to get all modes available or get specific mode

        
    def read_h_txt(self, basepath):
        # TODO: modify in case one has the txt files already
        # find all modes under basepath
        modes = glob.glob(basepath+'/Rh_l*.txt')

        r = []
        # find extraction radii
        for m in modes:
            r.append(m.split('/')[-1].split('_')[-1].split('.')[0])
        # find largest extraction radius
        r = list(set(r))
        imaxr = np.argmax([int(rr[1:]) for rr in r])

        # read modes
        d = {}
        for m in modes:
            
            this_r = m.split('/')[-1].split('_')[-1].split('.')[0]
            if this_r != r[imaxr]: continue
            
            ell = int(m.split('/')[-1].split('_')[1][1:])
            emm = int(m.split('/')[-1].split('_')[2][1:].split('.')[0])
            if self.ell_emms != 'all':
                if (ell, emm) not in self.ell_emms: continue
            d[(ell, emm)] = {}

            # u/M:0 Reh/M:1 Imh/M:2 Momega:3 A/M:4 phi:5 t:6
            u, re, im, Momg, A, phi, t = np.loadtxt(m, unpack=True, skiprows=3)
            d[(ell, emm)] ={
                'A': A,              'p': phi,
                't': t,              'real': re,             'imag': im
            }
        self._t = t
        self._u = u
        self._hlm = d
        pass