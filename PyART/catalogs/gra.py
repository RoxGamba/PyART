import numpy as np; import os
from ..waveform import  Waveform
import glob as glob

class Waveform_GRA(Waveform):
    """
    Class to handle GRAthena++ waveforms.
    This is still under development, depending on the
    final format of the data, and as such still quite rough.
    For now, it assumes that the data is in the format
    of Alireza's simulations.
    """

    def __init__(
            self,
            path,
            cut_N  = None,
            cut_U  = None,
            modes = [(2,2)]
        ):

        super().__init__()
        self.path  = path
        self.cut_N = cut_N
        self.cut_U = cut_U
        self.modes = modes
        pass

    def load_metadata(self):
        """
        Load the metadata
        """
        pass

    def load_hlm(self):
        """
        Load the data, assume the structure to be:
        # 1:tortoise_time 2:code_time 3:real 4:imag 5:phi 6:omega(=dphi/dt)[geom] 7:|amplitude| 8:freq[Hz]
        """
        
        hlms_files = glob.glob(self.path + '/*.txt')
        if len(hlms_files) == 0:
            raise FileNotFoundError('No files found in the given path: {}'.format(self.path))
        
        # load the time from one mode
        tmp_u,_,_,_,_,_,_,_ = np.loadtxt(hlms_files[0], unpack=True, skiprows=2)

        self.check_cut_consistency()
        if self.cut_N is None: self.cut_N = np.argwhere(tmp_u>=self.cut_U)[0][0] 
        if self.cut_U is None: self.cut_U = tmp_u[self.cut_N]

        self._u  = tmp_u[self.cut_N:]
        self._t  = self._u # FIXME: should we use another time? 

        dict_hlm = {}

        # Note: this is not nu-rescaled!
        for this_file in hlms_files:
            name = this_file.split('/')[-1]
            ell  = int(name.split('_')[1][1:])
            emm  = int(name.split('_')[2][1:])

            u,_,re,im,plm,_,Alm,_ = np.loadtxt(this_file, unpack=True, skiprows=2)
            key = (ell,emm)
            Alm = Alm[self.cut_N:]; plm = plm[self.cut_N:]
            re  = re[self.cut_N:];  im  = im[self.cut_N:]
            h   = re + 1j*im
            dict_hlm[key] = {'real': re,  'imag': im,
                             'A'   : Alm, 'p' : plm, 
                             'h'   : h
                            }

        self._hlm = dict_hlm
        pass

    def dump_h5(self):
        """
        Dump the data to an hdf5 file
        """
        pass

    def check_cut_consistency(self):
        if self.cut_N is not None and self.cut_U is not None:
            raise RuntimeError('Conflict between cut_N and cut_U!\n'
                               'When initializing, only one between cut_N and cut_U should be given in input.\n'
                               'The other one is temporarly set to None and (consistently) updated in self.load_hlm()')
        elif self.cut_N is None and self.cut_U is None:
            self.cut_N = 0
        pass

    def get_indices_dict(self):
        """
        Get the indices of the various cols in the data
        """
        # get col indices up to l=10
        indices_dict = {}
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
        
        return indices_dict 

    def load_psi4lm(self, path=None, fname=None):
        """
        Load the psi4lm modes. For now, assume that the data is in the format
        of Athena's output.
        """
        if path is None: path = self.path
        psi4lm_files = glob.glob(path + '/*.txt')
        if len(psi4lm_files) == 0:
            raise FileNotFoundError('No files found in the given path: {}'.format(path))
        
        fullname = os.path.join(path,fname)
        indices_dict = self.get_indices_dict()
        X = np.loadtxt(fullname)

        # load and store the time 
        # todo: use a different time array for psi4?
        t = X[:,indices_dict[(2,2)]['t']]
        self._t = t
        self._u = t

        dict_psi4lm = {}
        for mm in self.modes:
            re  = X[:,indices_dict[mm]['re']]
            im  = X[:,indices_dict[mm]['im']]
            Amp = np.sqrt(re**2 + im**2)
            phi = np.arctan2(im, re)
            dict_psi4lm[mm] = {'real': re, 'imag': im,
                                'A'  : Amp, 'p'   : phi
                                }
        self._psi4lm = dict_psi4lm
        pass

