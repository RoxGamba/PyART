import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import h5py 
import glob
import os

from ..waveform import  Waveform

class CoRe(Waveform):

    def __init__(self,
                 path='../dat/CoRe/',
                 ID      = 'BAM:0001',
                 run     = 'R01',
                 kind     = 'h5',
                 mtdt_path='../dat/CoRe/metadata.txt',
                 ell_emms='all',
                 download=False
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
            
        self.runpath = os.path.join(self.core_data_path,self.run)
        # read metadata
        self.metadata = self.read_metadata(mtdt_path)

        # read data
        self.read_h(self.core_data_path, kind)

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
        print('done!')


    def read_metadata(self, mtdt_path):
        metadata = {}
        with open(mtdt_path, "r") as f:
            lines = [l for l in f.readlines() if l.strip()] # rm empty
            for line in lines:
                if line[0]=="#": continue
                line               = line.rstrip("\n")
                key, val           = line.split("= ")
                key                = key.strip()
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
        self.dfile = os.path.join(self.runpath,'data.h5')
        dset = {}
        with h5py.File(self.dfile, 'r') as fn:
            for g in fn.keys():
                dset[g] = {}
                for f in fn[g].keys():
                    dset[g][f] = fn[g][f][()]
        try:
            uM = dset[:,0]
            RehM = dset[:,1]
            ImhM = dset[:,2]
            Momg = dset[:,5]
            aM   = dset[:,6]
            phi  = dset[:,7]
        except:
            uM = dset[:,0]
            RehM = dset[:,1]
            ImhM = dset[:,2]
            Momg = dset[:,3]
            aM   = dset[:,4]
            phi  = dset[:,5]

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

