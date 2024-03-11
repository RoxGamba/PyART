# Functions/classes used to load NR data like multipoles, punctures, horizon etc.

import os, sys, numpy as np
import matplotlib.pyplot as plt 

from .wf_utils import get_multipole_dict

class LoadPsi4(object):
    def __init__(self, path='./', fmt='etk', modes=[(2,2)], fname='wave.txt', resize=False, M=1, R=1):
        """
        Load psi4 from file
        
        Parameters:
          path   : path where file(s) are stored
          fmt    : format, like gra, etk etc.
          modes  : modes to load
          fname  : name of the file to load. If you have a file for each multipole,
                   then use the wildcard @L@ and @M@, e.g. 'psi4_l@L@_@M@.asc'
          resize : if True, handle files with different lenghts (not safe)
          M      : rest mass
          R      : extraction radius
        """
        #TODO: - norm_kind: normalize waveform according to fmt
         
        self.path  = path
        self.fmt       = fmt
        self.modes     = modes
        self.fname     = fname        
        self.M         = M 
        self.R         = R
        self.resize    = False

        # load data
        d = {}
        if '@L@' in self.fname and '@M@' in self.fname:
            previous_t = None
            for mm in modes:
                l, m = mm
                fname    = self.psi4_lm_name(l,m)
                X        = self.load_file(fname, safe=True)
                re,im,t  = self.reim_from_multipole(X,fmt=self.fmt)
                if previous_t is not None:
                    if len(t)!=len(previous_t) and not self.resize:
                        raise RuntimeError('Found times with different lenght')
                else:
                    previous_t = t
                psi4 = (re+1j*im)*self.M*self.R
                d[(l,m)] = get_multipole_dict(psi4)
            if self.resize:
                t, d = self.resize(t,d)

        else:
            print('To implement..')
        self.t    = t        
        self.psi4 = d
        return 
    
    def psi4_lm_name(self, l, m):
        out = self.fname.replace('@L@', f'{l:d}')
        return out.replace('@M@', f'{m:d}')
    
    def load_file(self,fname,safe=False):
        fullname = os.path.join(self.path,fname)
        if not safe:
            return np.loadtxt(fullname)
        else:
            lines = []
            with open(fullname, 'r') as file:
                for line in file.readlines():
                    parts = line.strip().split()
                    if len(parts)==3 and '#' not in line:
                        lines.append(line)
            return np.array([line.strip().split() for line in lines], dtype=float)
    
    def file_indices(self, fmt, l=None, m=None):
        if fmt=='etk':
            indices_dict = {'t':0, 're':1, 'im':2}
        elif fmt=='gra':
            print('more complicated...')
        else:
            raise RuntimeError(f'Unknown format: {fmt}')
        return indices_dict 

    def reim_from_multipole(self, X, fmt=None):
        if fmt is None: fmt = self.fmt
        indices_dict = self.file_indices(fmt=fmt)
        t  = X[:,indices_dict['t'] ]
        re = X[:,indices_dict['re']]
        im = X[:,indices_dict['im']]
        return re,im,t
    
    def resize(self,d,t):
        # find min lenght
        minlen = len(t)
        for mm in self.modes:
            len_d = len(d[mm])
            if len_d<minlen:
                minlen = len_d
        # resize t and psi4
        t = t[:minlen]
        for mm in self.modes:
            for key, val in d[mm].items():
                d[mm][key] = val[:minlen]
        return t,d 

if __name__=="__main__":
    x = LoadPsi4(path='/Users/simonealbanesi/repos/eob_nr_explore/data/scattering/BBH_hyp_D100_b10p3', 
                 fmt='etk', fname='mp_psi4_l@L@_m@M@_r100.00.asc', lmmax=4)
    print(x.psi4[(2,2)])
    plt.figure
    plt.plot(x.t, x.psi4[(2,2)]['real'])
    plt.show()


