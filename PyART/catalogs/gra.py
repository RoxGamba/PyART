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
        ):

        super().__init__()
        self.path  = path
        self.cut_N = cut_N
        self.cut_U = cut_U
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

    