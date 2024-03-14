# Functions/classes used to load NR data like multipoles, punctures, horizon etc.

import os, sys, numpy as np
import matplotlib.pyplot as plt 

class LoadWave(object):
    def __init__(self, path='./', fmt='etk', modes=[(2,2)], fname='wave.txt', resize=False):
        """
        Load wave from file
        
        Parameters:
          path   : path where file(s) are stored
          fmt    : format, like gra, etk etc.
          modes  : modes to load
          fname  : name of the file to load. If you have a file for each multipole,
                   then use the wildcard @L@ and @M@, e.g. 'psi4_l@L@_@M@.asc'
          resize : if True, handle files with different lenghts (not safe)
        """
        #TODO: - norm_kind: normalize waveform according to fmt
         
        self.path      = path
        self.fmt       = fmt
        self.modes     = modes
        self.fname     = fname        
        self.resize    = False

        # load data
        indices_dict = self.get_indices_dict()
        wave = {}
        if '@L@' in self.fname and '@M@' in self.fname:
            previous_t = None
            for mm in modes:
                l, m  = mm
                fname = self.wave_lm_name(l,m)
                Xlm   = self.load_file(fname, safe=True)
                t     = Xlm[:,indices_dict[(l,m)]['t'] ] 
                re    = Xlm[:,indices_dict[(l,m)]['re']] 
                im    = Xlm[:,indices_dict[(l,m)]['im']] 
                if previous_t is not None:
                    if len(t)!=len(previous_t) and not self.resize:
                        raise RuntimeError('Found times with different lenght')
                else:
                    previous_t = t
                wave[(l,m)] = re+1j*im

        else:
            X = self.load_file(fname, safe=False)
            t = X[:,indices_dict[(2,2)]['t']]
            for mm in self.modes:
                l,m = mm
                re  = X[:,indices_dict[mm]['re']]
                im  = X[:,indices_dict[mm]['im']]
                wave[(l,m)] = re+1j*im
            
        if self.resize:
            t, wave = self.resize(t,wave)
        
        self.t    = t        
        self.wave = wave
        return 
    
    def wave_lm_name(self, l, m):
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
    
    def get_indices_dict(self):
        fmt = self.fmt
        indices_dict = {}
        if fmt=='etk':
            for mm in self.modes:
                indices_dict[mm] = {'t':0, 're':1, 'im':2}
        
        elif fmt=='gra':
            # get col indices up to l=10
            col_indices = {}
            c = 0
            cstart = 2
            for l in range(2, 11):
                for m in range(-l, l+1):
                    col_indices[(l,m)] = (cstart+c, cstart+c+1)
                    c += 2
            # now store the ones that we need
            for mm in self.modes:
                re_idx = col_indices[mm][0]
                im_idx = col_indices[mm][1]
                indices_dict[mm] = {'t':1, 're':re_idx, 'im':im_idx} 
        
        else:
            raise RuntimeError(f'Unknown format: {fmt}')
        
        return indices_dict 
    
    def resize(self,t,wave):
        # find min lenght
        minlen = len(t)
        for mm in self.modes:
            len_wave = len(wave[mm])
            if len_wave<minlen:
                minlen = len_wave
        # resize t and wave
        t = t[:minlen]
        for mm in self.modes:
            wave[mm] = wave[mm][:minlen]
        return t,wave

if __name__=="__main__":
    x = LoadWave(path='/Users/simonealbanesi/repos/eob_nr_explore/data/scattering/BBH_hyp_D100_b10p3', 
                 fmt='etk', fname='mp_psi4_l@L@_m@M@_r100.00.asc', modes=[(2,2)])
    plt.figure
    plt.plot(x.t, x.wave[(2,2)].real)
    plt.show()


