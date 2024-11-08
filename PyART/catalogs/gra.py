import numpy as np; import os; import h5py
from ..waveform import  Waveform
import glob as glob; import json

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
            ellmax = 8,
            ext    = 'ext',
            r_ext  = None,
            cut_N  = None,
            cut_U  = None,
            mtdt_path = None,
            rescale = False,
            modes = [(2,2)]
        ):

        super().__init__()
        self.path   = path
        self.cut_N  = cut_N
        self.cut_U  = cut_U
        self.modes  = modes
        self.ellmax = ellmax
        self.extrap = ext
        self.domain = 'Time'
        self.r_ext  = r_ext
        self.rescale = rescale
        # comment out the following for the moment
        # self.load_metadata(mtdt_path)
        self.load_hlm(extrap=ext, ellmax=ellmax, r_ext=r_ext)
        pass

    def load_metadata(self, path):
        """
        Load the metadata, if path is None assume 
        that they are in the same dir as the .h5 files
        """
        if path is None: path = self.path
        ometa = json.load(open(path, 'r'))

        m1 = float(ometa['initial-mass1'])
        m2 = float(ometa['initial-mass2'])
        M    = m1 + m2
        q    = m1/m2
        nu   = q/(1+q)**2
        hS1  = ometa['initial-dimensionless-spin1'].strip('"').split(','); hS1 = np.array([float(hS1[i]) for i in range(3)])
        hS2  = ometa['initial-dimensionless-spin2'].strip('"').split(','); hS2 = np.array([float(hS2[i]) for i in range(3)])
        pos1 = ometa['initial-position1'].strip('"').split(','); pos1 = np.array([float(pos1[i]) for i in range(3)])
        pos2 = ometa['initial-position2'].strip('"').split(','); pos2 = np.array([float(pos2[i]) for i in range(3)])
        r0   = ometa['initial-separation']
        P0   = ometa['initial-ADM-linear-momentum'].strip('"').split(','); P0 = np.array([float(P0[i]) for i in range(3)])
        L0   = ometa['initial-ADM-angular-momentum'].strip('"').split(','); L0 = np.array([float(L0[i]) for i in range(3)])

        metadata = {'name'     : ometa['simulation-name'],
                    'ref_time' : 0.,
                    # masses and spins 
                    'm1'       : m1,
                    'm2'       : m2,
                    'M'        : M,
                    'q'        : q,
                    'nu'       : nu,
                    'S1'       : hS1*m1*m1, # [M2]
                    'S2'       : hS2*m2*m2,
                    'chi1x'    : hS1[0],  # dimensionless
                    'chi1y'    : hS1[1],
                    'chi1z'    : hS1[2],
                    'chi2x'    : hS2[0],  # dimensionless
                    'chi2y'    : hS2[1],
                    'chi2z'    : hS2[2],
                    # positions
                    'pos1'     : pos1,
                    'pos2'     : pos2,
                    'r0'       : r0,
                    'e0'       : None,
                    # frequencies
                    'f0v'      : None,
                    'f0'       : float(ometa['initial-orbital-frequency'])/np.pi,
                    # ADM quantities (INITIAL, not REF)
                    'E0'       : float(ometa['initial-ADM-energy']),
                    'P0'       : P0,
                    'J0'       : L0,
                    'Jz0'      : L0[2],
                    'E0byM'    : float(ometa['initial-ADM-energy'])/M,
                    'pph0'     : None,
                    # remnant
                    'Mf'       : None,
                    'afv'      : None,
                    'af'       : None,
                   }
        
        self.metadata = metadata 
        pass

    def load_hlm(self, extrap='ext', ellmax=None, load_m0=False, r_ext=None):
        """
        Load the data from the h5 file
        """
        if ellmax==None: ellmax=self.ellmax
        if r_ext==None: r_ext='100.00'

        if extrap == 'ext':
            h5_file = os.path.join(self.path, 'rh_Asymptotic_GeometricUnits.h5')
        elif extrap == 'CCE':
            h5_file = os.path.join(self.path, 'rh_CCE_GeometricUnits_radii.h5')
        elif extrap == 'finite':
            h5_file = os.path.join(self.path, 'rh_FiniteRadii_GeometricUnits.h5')
        else:
            raise ValueError('extrap should be either "ext", "CCE" or "finite"')
        
        if not os.path.isfile(h5_file):
            raise FileNotFoundError('No file found in the given path: {}'.format(h5_file))
        
        nr    = h5py.File(h5_file, 'r')
        if r_ext not in nr.keys():
            raise ValueError('r_ext not found in the h5 file. Available values are: {}'.format(nr.keys()))
        tmp_u = nr[r_ext]['Y_l2_m2.dat'][:,0]

        self.check_cut_consistency()
        if self.cut_N is None: self.cut_N = np.argwhere(tmp_u>=self.cut_U)[0][0] 
        if self.cut_U is None: self.cut_U = tmp_u[self.cut_N]

        self._u  = tmp_u[self.cut_N:]
        self._t  = self._u

        from itertools import product
        modes = [(l, m) for l, m in product(range(2, ellmax+1), range(-ellmax, ellmax+1)) if (m!=0 or load_m0) and l >= np.abs(m)]

        dict_hlm = {}

        for mode in modes:
            l    = mode[0]; m = mode[1]
            mode = "Y_l" + str(l) + "_m" + str(m) + ".dat"
            hlm  = nr[r_ext][mode]
            h    = (hlm[:, 1] + 1j * hlm[:, 2])
            if self.rescale:
                h /= self.metadata['nu']
            # amp and phase
            Alm = abs(h)[self.cut_N:]
            plm = -np.unwrap(np.angle(h))[self.cut_N:]
            # save in dictionary
            key = (l, m)
            dict_hlm[key] =  {'real': Alm*np.cos(plm), 'imag': Alm*np.sin(plm),
                              'A'   : Alm, 'p' : plm, 
                              'z'   : h[self.cut_N:]
                              }
        self._hlm = dict_hlm
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

    def load_psi4lm(self, path=None, fname=None, ellmax=None, r_ext=None, extrap='ext',  load_m0=False):
        if ellmax==None: ellmax=self.ellmax
        
        if r_ext==None: r_ext='100.00'
        
        if extrap == 'ext':
            h5_file = os.path.join(self.path, 'rPsi4_Asymptotic_GeometricUnits.h5')
        elif extrap == 'CCE':
            h5_file = os.path.join(self.path, 'rPsi4_CCE_GeometricUnits.h5')
        elif extrap == 'finite':
            h5_file = os.path.join(self.path, 'rPsi4_FiniteRadii_GeometricUnits.h5')
        else:
            raise ValueError('extrap should be either "ext", "CCE" or "finite"')

        if not os.path.isfile(h5_file):
            raise FileNotFoundError('No file found in the given path: {}'.format(h5_file))
        
        nr    = h5py.File(h5_file, 'r')
        if r_ext not in nr.keys():
            raise ValueError('r_ext not found in the h5 file. Available values are: {}'.format(nr.keys()))
        tmp_u = nr[r_ext]['Y_l2_m2.dat'][:,0]

        # self.check_cut_consistency()
        if self.cut_N is None: self.cut_N = np.argwhere(tmp_u>=self.cut_U)[0][0] 
        if self.cut_U is None: self.cut_U = tmp_u[self.cut_N]

        self._u  = tmp_u[self.cut_N:]
        self._t  = self._u

        from itertools import product
        modes = [(l, m) for l, m in product(range(2, ellmax+1), range(-ellmax, ellmax+1)) if (m!=0 or load_m0) and l >= np.abs(m)]

        dict_psi4lm = {}
        for mode in modes:
            l    = mode[0]; m = mode[1]
            mode = "Y_l" + str(l) + "_m" + str(m) + ".dat"
            psi4lm  = nr[r_ext][mode]
            psi4 = (psi4lm[:,1] + 1j*psi4lm[:,2])
            if self.rescale:
                psi4 /= self.metadata['nu']
            Alm = abs(psi4)[self.cut_N:]
            plm = -np.unwrap(np.angle(psi4))[self.cut_N:]
            key = (l,m)
            dict_psi4lm[key] = {'real'  : Alm * np.cos(plm),
                                 'imag' : Alm * np.sin(plm),
                                 'A'    : Alm, 
                                 'p'    : plm,
                                 'z'    : psi4[self.cut_N:]
                                }

        self._psi4lm = dict_psi4lm
        pass 
