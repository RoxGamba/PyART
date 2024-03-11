# Functions/classes used to load NR data like multipoles, punctures, horizon etc.

import os, sys, numpy as np
import matplotlib.pyplot as plt 

class LoadPsi4(object):
    def __init__(self, basepath='./', fmt='etk', mneg=True, lmmax=5, fname='wave.txt', resize=False, M=1, R=1):
        """
        Load psi4 from file
        
        Parameters:
          basepath : path where file(s) are stored
          fmt      : format, like gra, etk etc.
          mneg     : load m<0 modes (default is True)
          lmmax    : load multipoles up to this values
          fname    : name of the file to load. If you have a file for each multipole,
                     then use the wildcard @L@ and @M@, e.g. 'psi4_l@L@_@M@.asc'
          resize   : if True, handle files with different lenghts (not safe)
          M        : rest mass
          R        : extraction radius
        """
        #TODO: - norm_kind: normalize waveform according to fmt
         
        self.basepath  = basepath
        self.fmt       = fmt
        self.mneg      = mneg
        self.lmmax     = lmmax
        self.fname     = fname        
        self.M         = M 
        self.R         = R
        self.resize    = False
        # Get modes to load 
        modes = []
        for l in range(2,self.lmmax+1):
            mmin = -l if self.mneg else 0
            for m in range(mmin, l+1):
                modes.append((l,m))
        self.modes = modes 

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
                d[(l,m)] = self.get_multipole_dict(psi4)
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
        fullname = os.path.join(self.basepath,fname)
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
    
    def reim_from_multipole(self, X, fmt=None):
        if fmt is None: fmt = self.fmt
        if fmt=='etk':
            t  = X[:,0]
            re = X[:,1]
            im = X[:,2]
        else:
            raise RuntimeError(f'Unknown format: {fmt}')
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

    def get_multipole_dict(self, wave):
        return {'real':wave.real, 'imag':wave.real, 'h':wave,
                'A':np.abs(wave), 'p':-np.unwrap(np.angle(wave))}

if __name__=="__main__":
    x = LoadPsi4(basepath='/Users/simonealbanesi/repos/eob_nr_explore/data/scattering/BBH_hyp_D100_b10p3', 
                 fmt='etk', fname='mp_psi4_l@L@_m@M@_r100.00.asc', lmmax=4)
    print(x.psi4[(2,2)])
    plt.figure
    plt.plot(x.t, x.psi4[(2,2)]['real'])
    plt.show()


