import numpy as np; import os
import h5py; import json
from ..waveform import  Waveform
from ..utils import cat_utils as cat_ut

class Waveform_SXS(Waveform):
    """
    Class to handle SXS waveforms
    Assumes that the data is in the directory specified py `path`,
    and that all simulations are stored in folders like SXS_BBH_XXXX,
    each containing the various `LevY` folders.
    e.g., the current default is 
        ../dat/SXS_BBH_XXXX/LevY/
    """
    def __init__(
                    self,
                    path          = '../dat/SXS/',
                    ID            = '0001',
                    order         = 2,
                    level         = None,
                    cut_N         = None,
                    cut_U         = None,
                    ellmax        = 8,
                    load          = ['hlm','metadata'],
                    download      = False,
                    downloads     = ['hlm','metadata'],
                    load_m0       = False,
                    nu_rescale    = False,
                    src           = 'BBH',
                    ignore_deprecation = False,
                    basename      = None, # if None, use default according to src
                ):
        super().__init__()
        if isinstance(ID, int):
            ID = f'{ID:04}'
            
        self.ID            = ID
        self.sxs_data_path = os.path.join(path,'SXS_'+src+'_'+ID)
        self.order         = order
        self.level         = level
        self.cut_N         = cut_N
        self.cut_U         = cut_U
        self.ellmax        = ellmax
        self._kind         = 'SXS'
        self.src           = src
        self.nr            = None
        self.domain        = 'Time'
        self.nu_rescale    = nu_rescale
        
        if basename is None:
            if src=='BHNS'and int(self.ID)<=7:
                basename = 'rhOverM_Asymptotic_GeometricUnits.h5'
            elif (src=='BHNS' and int(self.ID)>7) or src=='BBH':
                basename = 'rhOverM_Asymptotic_GeometricUnits_CoM.h5'
            else:
                raise ValueError('basename is None, but unknown src!')
        self.basename = basename

        if level is not None and isinstance(level, int):
            levpath = f'{self.sxs_data_path}/Lev{level}'
        else:
            levpath = self.sxs_data_path

        self.check_cut_consistency()
        if os.path.exists(levpath) == False:
            if download:
                print("The path ", self.sxs_data_path, " does not exist.")
                print("Downloading the simulation from the SXS catalog.")
                self.download_simulation(ID=ID, path=path, downloads=downloads, level=level, 
                                         ignore_deprecation=ignore_deprecation,
                                         extrapolation_order=order)
            else:
                print("Use download=True to download the simulation from the SXS catalog.")
                raise FileNotFoundError(f"The path {self.sxs_data_path} does not exist.")
        
        if isinstance(self.level, int):
            fname = self.get_lev_fname(basename=self.basename)
            if os.path.exists(fname):
                self.nr = h5py.File(fname)
            else:
                raise FileNotFoundError(f'SXS path found, but the requested level ({self.level:d}) is not available!')

        elif self.level is None:
            ref_lv_max = 7
            ref_lv_min = 1
            if 'hlm' not in load:
                self.basename='metadata.json'
            for lvn in range(ref_lv_max,ref_lv_min-1,-1):
                fname = self.get_lev_fname(level=lvn,basename=self.basename)
                if os.path.exists(fname):
                    if 'hlm' in load: self.nr    = h5py.File(fname)
                    self.level = lvn 
                    break
                elif lvn==ref_lv_min:
                    raise RuntimeError(f'No data for ref-levels:[{ref_lv_min:d},{ref_lv_max:d}] found')
        
        else:
            raise RuntimeError(f'Invalid input for level: {self.level}')
        
        if 'metadata' in load: self.load_metadata()
        if 'hlm'      in load: self.load_hlm(load_m0=load_m0)
        if 'horizons' in load: self.load_horizon()
        if 'psi4lm'   in load: self.load_psi4lm(load_m0=load_m0)
        pass
    
    def check_cut_consistency(self):
        if self.cut_N is not None and self.cut_U is not None:
            raise RuntimeError('Conflict between cut_N and cut_U!\n'
                               'When initializing, only one between cut_N and cut_U should be given in input.\n'
                               'The other one is temporarly set to None and (consistently) updated in self.load_hlm()')
        elif self.cut_N is None and self.cut_U is None:
            self.cut_N = 0
        pass

    def get_lev_fname(self,level=None,basename=None):
        """
        Return file-name in a SXS-path with specified level,
        e.g. /my/sxs/path/Lev4/my_basename
        If basename is None, then return only /my/sxs/path/Lev4
        """
        
        if not isinstance(basename, str):
            raise ValueError('basename must be a string!')
        if level is None: level = self.level

        tojoin = f'Lev{level:d}/{basename}'
        return os.path.join(self.sxs_data_path, tojoin)
    
    def download_simulation(self, ID='0001', path=None, downloads=['hlm','metadata'],level=None, ignore_deprecation=False,
                            extrapolation_order=None):
        """
        Download the simulation from the SXS catalog; requires the sxs module
        """

        import sxs as sxsmod

        if path is not None:
            print("Setting the download (cache) directory to ", path)
            os.environ['SXSCACHEDIR'] = path

        nm = 'SXS:'+self.src+':'+ID
        if level is not None:
            nm_ld = nm+ f'/Lev{level}/'
        else:
            nm_ld = nm
        
        sxs_sim = sxsmod.load(nm_ld,
                              extrapolation_order = extrapolation_order, 
                              ignore_deprecation = ignore_deprecation)

        if 'hlm'      in downloads: _     = sxs_sim.h 
        if 'metadata' in downloads: mtdt  = sxs_sim.metadata
        if 'horizons' in downloads: _     = sxs_sim.horizons
        if 'psi4lm'   in downloads: _     = sxs_sim.psi4
        
        sxs_dir = 'SXS_'+self.src+'_'+ID
        
        # find folder(s) corresponding to the name, mkdir the new one
        flds = [f for f in os.listdir(os.environ['SXSCACHEDIR']) if nm in f]
        if not os.path.exists(os.path.join(path,sxs_dir)):
            os.mkdir(os.path.join(path,sxs_dir))

        # move the files in the folders to the new folder and find lev
        lev = None
        for fld in flds:
            for fls in os.listdir(os.path.join(os.environ['SXSCACHEDIR'],fld)):
                try:
                    # identify the Lev of the file
                    if lev is None:
                        lev   = fls.split(':')[0] if 'Lev' in fls else None
                    this_lev  = lev + '_'
                    this_file = fls.split(':')[1:][0]
                except:
                    this_lev  = ''
                    this_file = fls
                # move each file, renaming
                print(os.path.join(os.environ['SXSCACHEDIR'],fld,fls),'-->',os.path.join(path,sxs_dir,this_lev+this_file))
                os.rename(os.path.join(os.environ['SXSCACHEDIR'],fld,fls), os.path.join(path,sxs_dir,this_lev+this_file))

        # prepare to dump the modes in the old format, for retrocompatibility

        # create the related Lev directory
        lev_dir = os.path.join(path,sxs_dir,lev)
        if not os.path.exists(lev_dir):
            os.mkdir(lev_dir)
        if 'hlm' in downloads:
            import h5py
            from itertools import product

            extp = f'Extrapolated_N{extrapolation_order}.dir'
            to_h5file = {
               extp : {}
            }
            wav    = sxs_sim.h
            ellmax = 8#wav.ellmax
            modes  = [(l, m) for l, m in product(range(2, ellmax+1), range(-ellmax, ellmax+1)) if l >= np.abs(m)]
            for mode in modes:
                mode_string = 'Y_l' + str(mode[0]) + '_m' + str(mode[1]) + '.dat'
                to_h5file[extp][mode_string] = wav[mode_string]

            # create the h5 file
            h5file = h5py.File(os.path.join(path,sxs_dir,lev,f'rhOverM_Asymptotic_GeometricUnits_CoM.h5'), 'w')    
            save_dict_to_h5(h5file, to_h5file)
            h5file.close()

        if 'psi4lm' in downloads:
            import h5py
            from itertools import product

            extp = f'Extrapolated_N{extrapolation_order}.dir'
            to_h5file = {
               extp : {}
            }
            wav    = sxs_sim.psi4
            ellmax = 8#wav.ellmax
            modes  = [(l, m) for l, m in product(range(2, ellmax+1), range(-ellmax, ellmax+1)) if l >= np.abs(m)]
            for mode in modes:
                mode_string = 'Y_l' + str(mode[0]) + '_m' + str(mode[1]) + '.dat'
                to_h5file[extp][mode_string] = wav[mode_string]

            # create the h5 file
            h5file = h5py.File(os.path.join(path,sxs_dir,lev,f'rMPsi4_Asymptotic_GeometricUnits_CoM.h5'), 'w')    
            save_dict_to_h5(h5file, to_h5file)
            h5file.close()

        if 'horizons' in downloads:
            import h5py
            to_h5file = {}
            hrz = sxs_sim.horizons
            # fill the dictionary
            for kk in ['AhA.dir','AhB.dir']:
                to_h5file[kk] = {}
                for key in ['CoordCenterInertial.dat', 'ChristodoulouMass.dat',  'DimensionfulInertialSpinMag.dat']:
                    to_h5file[kk][key] = hrz[f'{kk}/{key}']
            
            # create the h5 file
            h5file = h5py.File(os.path.join(path,sxs_dir,lev,f'Horizons.h5'), 'w')
            save_dict_to_h5(h5file, to_h5file)
            h5file.close()

        # dump the metadata in a json
        if 'metadata' in downloads:
            with open(os.path.join(path,sxs_dir,lev,f'metadata.json'), 'w') as file:
                json.dump(mtdt, file, indent=2)
                file.close()

        # delete the empty folder
        os.rmdir(os.path.join(os.environ['SXSCACHEDIR'],fld))

        pass
    
    def load_metadata(self):
        with open(self.get_lev_fname(basename="metadata.json"), 'r') as file:
            ometa = json.load(file) # original_metadata
            file.close()
        self.ometadata = ometa # store also original metadata, for completeness
        
        # TODO : 1) check if these quantities are mass-rescaled or not
        #        2) here we are using initial quantities, not ref. The
        #           reason is that ADM integrals are not given at ref time

        def is_valid(key, vtype=None):
            if key not in ometa:
                return False
            if isinstance(ometa[key], str):
                return False
            if vtype is not None: # check var-type if provided
                return isinstance(ometa[key], vtype)
            return True

        M1 = ometa['reference_mass1']
        if is_valid('reference_mass2', vtype=float):
            M2 = ometa['reference_mass2']
        else:
            print('+++ Warning +++ reference_mass2 not found or invalid! Usinig initial masses')
            M1 = ometa['initial_mass1']
            M2 = ometa['initial_mass2']

        q  = M2/M1
        if q<1:
            q = 1/q
        nu = q/(1+q)**2
        M  = M1 + M2

        def read_spin_variable(spin_idx):
            attempts = ['reference_dimensionless_spin', 
                        'initial_dimensionless_spin',
                        'reference_spin']
            for attempt in attempts:
                key = attempt+str(spin_idx)
                if is_valid(key):
                    hS = np.array(ometa[key])
                    if attempt=='reference_':
                        if spin_idx==1:
                            hS = hS/M1**2
                        elif spin_idx==2:
                            hS = hS/M2**2
                    break
            return hS, attempt
        
        hS1, skey1 = read_spin_variable(1)
        hS2, skey2 = read_spin_variable(2)
        
        if not skey1==skey2:
            print(f'Warning: using different spin-entries! {skey1} and {skey2}')

        pos1 = np.array(ometa['reference_position1'])
        pos2 = np.array(ometa['reference_position2'])
        r0   = np.linalg.norm(pos1-pos2)
        
        try:
            Mf = float(ometa['remnant_mass']) 
            if is_valid('remnant_dimensionless_spin'):
                afv = np.array(ometa['remnant_dimensionless_spin'])
            elif is_valid('remnant_spin'):
                afv = np.array(ometa['remnant_spin'])/Mf**2
            else:
                raise ValueError("Unknown key for remnant's spin or invalid value")
            afz = afv[2] 
        except Exception as e:
            print(f'Failed in reading remnant properties: {e}')
            Mf  = None
            afv = None
            afz = None

        if isinstance(ometa['alternative_names'], list):
            name = ometa['alternative_names'][1]
        else:
            name = ometa['alternative_names']
        
        ecc = ometa['reference_eccentricity']
        if isinstance(ecc, str): 
            if '<' in ecc and 'e+00': # there are things like '<1.7e+00' in meta
                ecc = None
            else:
                ecc = 1e-5
        else:
            ecc = float(ecc)
        
        # Read ADM quantities. If not available (e.g. BHNS:0008 or BHNS:0009)
        # set to None
        if 'initial_ADM_angular_momentum' in ometa:
            J0   = np.array(ometa['initial_ADM_angular_momentum'])
            J0z  = J0[2]
            Lz   = J0 - hS1*M1*M1 - hS2*M2*M2
            pph0 = Lz[2]/(M*M*nu) 
        else:
            print('Warning! No angular momentum found')
            J0   = None
            J0z  = None
            Lz   = None
            pph0 = None
        
        if 'initial_ADM_energy' in ometa:
            E0    = ometa['initial_ADM_energy']
            E0byM = E0/M
        else:
            E0    = None
            E0byM = None
        
        if 'initial_ADM_linear_momentum' in ometa: 
            P0v = np.array(ometa['initial_ADM_linear_momentum'])
        else:
            P0v = None

        orb_freq_ometa = ometa['reference_orbital_frequency']
        if isinstance(orb_freq_ometa,float):
            f0 = orb_freq_ometa/np.pi
        elif len(orb_freq_ometa)==3:
            f0 = orb_freq_ometa[2]/np.pi
        else:
            raise RuntimeError('Unknown format for reference_orbital_frequency')
        
        # Set Lambda(s)
        if self.src=='BBH':
            LambdaAl2 = 0.
            LambdaBl2 = 0.
        elif self.src=='BHNS':
            LambdaAl2 = 0.
            if name=='SXS:BHNS:0001':
                LambdaBl2 = 526.
            elif name=='SXS:BHNS:0003':
                LambdaBl2 = 607.
            elif int(self.ID)<=9:
                LambdaBl2 = 791.
            else:
                raise RuntimeError(f'Unknown LambdaBl2!')
            if M2>M1:
                raise RuntimeError(f'BHNS: M2>M1 but Lambda1 (LambdaAl2) is zero!')
        else:
            raise RuntimeError(f'Unknown source: {self.src}')
      
        metadata = {'name'       : name, # i.e. store as name 'SXS:BBH:ID'
                    'ref_time'   : ometa['reference_time'],
                    # masses and spins 
                    'm1'         : M1,
                    'm2'         : M2,
                    'M'          : M,
                    'q'          : q,
                    'nu'         : nu,
                    'S1'         : hS1*M1*M1, # [M2]
                    'S2'         : hS2*M2*M2,
                    'chi1x'      : hS1[0],  # dimensionless
                    'chi1y'      : hS1[1],
                    'chi1z'      : hS1[2],
                    'chi2x'      : hS2[0],  # dimensionless
                    'chi2y'      : hS2[1],
                    'chi2z'      : hS2[2],
                    'LambdaAl2'  : LambdaAl2,
                    'LambdaBl2'  : LambdaBl2,
                    # positions
                    'pos1'       : pos1,
                    'pos2'       : pos2,
                    'r0'         : r0,
                    'e0'         : ecc,
                    # frequencies
                    'f0v'        : np.array(ometa['reference_orbital_frequency'])/np.pi,
                    'f0'         : f0, 
                    # ADM quantities (INITIAL, not REF)
                    'E0'         : E0,
                    'P0'         : P0v,
                    'J0'         : J0,
                    'Jz0'        : J0z,
                    'E0byM'      : E0byM, 
                    'pph0'       : pph0,
                    # remnant
                    'Mf'         : Mf,
                    'afv'        : afv,
                    'af'         : afz,
                    'scat_angle' : None,
                   }
        metadata['flags'] = cat_ut.get_flags(metadata)
        # check that all the required quantities are given 
        cat_ut.check_metadata(metadata, raise_err=True)
        # then store as attribute
        self.metadata = metadata 
        pass

    def load_horizon(self):
        horizon = h5py.File(self.get_lev_fname(basename="Horizons.h5"))
        
        mA   = horizon['AhA.dir']['ChristodoulouMass.dat']
        mB   = horizon['AhB.dir']['ChristodoulouMass.dat']
        chiA = horizon["AhA.dir/DimensionfulInertialSpinMag.dat"]
        chiB = horizon["AhB.dir/DimensionfulInertialSpinMag.dat"]
        xA   = horizon["AhA.dir/CoordCenterInertial.dat"]
        xB   = horizon["AhB.dir/CoordCenterInertial.dat"]

        self._dyn['t']     = chiA[:, 0]
        self._dyn['m1']    = mA[:, 1]
        self._dyn['m2']    = mB[:, 1]
        self._dyn['chi1']  = chiA[:, 1]
        self._dyn['chi2']  = chiB[:, 1]
        self._dyn['x1']    = xA[:, 1:]
        self._dyn['x2']    = xB[:, 1:]

        pass

    def compute_spins_at_tref(self, tref):
        """
        Compute the parallel and perpendicular components of the spins w.r.t L 
        at a reference time tref

        Parameters
        ----------
        tref : float
            Reference time
        
        Returns
        -------
        chi1_L, chi1_perp, chi2_L, chi2_perp : float
            The parallel and perpendicular components of the spins at tref
        """
        d = self.dyn

        # find the index of the reference time
        idx = np.argmin(np.abs(d['t']-tref))
        chi1_ref = d['chi1'][idx][1:]
        chi2_ref = d['chi2'][idx][1:]
        x1_ref   = d['x1'][idx][1:]       
        x2_ref   = d['x2'][idx][1:]
        
        # time derivative of x1 and x2
        x1_dot    = np.transpose([np.gradient(d['x1'][:,i], d['t']) for i in range(1,4)])
        x2_dot    = np.transpose([np.gradient(d['x2'][:,i], d['t']) for i in range(1,4)])
        x         = x1_ref - x2_ref
        
        x_dot     = [x1_dot[idx][i] - x2_dot[idx][i] for i in range(3)]
        L_hat_ref = np.cross(x, x_dot)/np.linalg.norm(np.cross(x, x_dot))

        # compute the spins projected on L_hat_ref
        chi1_L = np.dot(chi1_ref, L_hat_ref)
        chi2_L = np.dot(chi2_ref, L_hat_ref)
        chi1_perp = np.linalg.norm(chi1_ref - chi1_L*L_hat_ref)
        chi2_perp = np.linalg.norm(chi2_ref - chi2_L*L_hat_ref)
        return chi1_L, chi1_perp, chi2_L, chi2_perp

    def load_hlm(self, ellmax=None, load_m0=False):
        if ellmax==None: ellmax=self.ellmax
        order   = f'Extrapolated_N{self.order}.dir'
        
        if not hasattr(self, 'metadata'):
            raise RuntimeError('Load metadata before loading hlm!')

        from itertools import product

        modes = [(l, m) for l, m in product(range(2, ellmax+1), range(-ellmax, ellmax+1)) if (m!=0 or load_m0) and l >= np.abs(m)]

        
        tmp_u = self.nr[order]['Y_l2_m2.dat'][:, 0]
        
        self.check_cut_consistency()
        if self.cut_N is None: self.cut_N = np.argwhere(tmp_u>=self.cut_U)[0][0] 
        if self.cut_U is None: self.cut_U = tmp_u[self.cut_N]
        
        self._u  = tmp_u[self.cut_N:]
        self._t  = self._u # FIXME: should we use another time? 

        dict_hlm = {}
        for mode in modes:
            l    = mode[0]; m = mode[1]
            mode = "Y_l" + str(l) + "_m" + str(m) + ".dat"
            hlm  = self.nr[order][mode]
            h    = (hlm[:, 1] + 1j * hlm[:, 2])
            if self.nu_rescale:
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
        all_keys = self._hlm.keys()
        pass

    def load_psi4lm(self, ellmax=None, load_m0=False):
        
        psi4_basename = self.basename.replace('rhOverM', 'rMPsi4')
        fname = self.get_lev_fname(level=self.level,basename=psi4_basename)
        if os.path.exists(fname):
            self.nr_psi = h5py.File(fname)

        if ellmax==None: ellmax=self.ellmax
        order   = f"Extrapolated_N{self.order}.dir"

        if not hasattr(self, 'metadata'):
            raise RuntimeError('Load metadata before loading hlm!')

        from itertools import product

        modes = [(l, m) for l, m in product(range(2, ellmax+1), range(-ellmax, ellmax+1)) if (m!=0 or load_m0) and l >= np.abs(m)]
       
        tmp_u = self.nr_psi[order]['Y_l2_m2.dat'][:, 0]
        
        if self.cut_N is None: self.cut_N = np.argwhere(tmp_u>=self.cut_U)[0][0] 
        if self.cut_U is None: self.cut_U = tmp_u[self.cut_N]
        
        self._t_psi4  = self._u # FIXME: should we use another time? 

        dict_psi4lm = {}
        for mode in modes:
            l    = mode[0]; m = mode[1]
            mode = "Y_l" + str(l) + "_m" + str(m) + ".dat"
            psi4lm  = self.nr_psi[order][mode]
            psi4    = (psi4lm[:, 1] + 1j * psi4lm[:, 2])
            if self.nu_rescale:
                psi4 /= self.metadata['nu']
            # amp and phase
            Alm = abs(psi4)[self.cut_N:]
            plm = -np.unwrap(np.angle(psi4))[self.cut_N:]
            # save in dictionary
            key = (l, m)
            dict_psi4lm[key] = { 'real': Alm*np.cos(plm), 
                                 'imag': Alm*np.sin(plm),
                                 'A'   : Alm, 
                                 'p'   : plm, 
                                 'z'   : psi4[self.cut_N:]
                              }
        self._psi4lm = dict_psi4lm
        pass

    def compute_psi4_from_hlm(self):
        """
        Compute the psi4lm by taking two time derivatives
        of the hlm modes
        """
        dict_psi4lm = {}
        t = self.u
        for ky in self.hlm.keys():
            h = self.hlm[ky]['z']
            ddh             = np.gradient(np.gradient(h, t), t)
            dict_psi4lm[ky] = {'A': abs(ddh), 'p': -np.unwrap(np.angle(ddh)),
                               'real': ddh.real, 'imag': ddh.imag,
                               'z': ddh}
        self._psi4lm = dict_psi4lm



def save_dict_to_h5(h5file, dictionary, path="/"):
    """
    Recursively save a nested dictionary to an HDF5 file.
    """
    for key, value in dictionary.items():
        key_path = f"{path}/{key}" if path != "/" else key
        
        if isinstance(value, dict):
            group = h5file.create_group(key_path)  # Create a group for nested dicts
            save_dict_to_h5(group, value)  # Recurse
        elif isinstance(value, (int, float, str, bytes)):
            h5file.attrs[key] = value  # Store scalars and strings as attributes
        else:
            h5file.create_dataset(key_path, data=value)  # Store arrays