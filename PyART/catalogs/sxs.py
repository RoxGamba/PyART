import numpy as np; import os
import h5py; import json
from ..waveform import  Waveform
from ..utils.wf_utils import compute_hphc

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
                    path     = '../dat/SXS/',
                    ID       = '0001',
                    order    = "Extrapolated_N2.dir",
                    level    = None,
                    cut_N    = None,
                    cut_U    = None,
                    ellmax   = 8,
                    download = False
                ):
        super().__init__()
        self.ID            = ID
        self.sxs_data_path = os.path.join(path,'SXS_BBH_'+ID)
        self.order         = order
        self.level         = level
        self.cut_N         = cut_N 
        self.cut_U         = cut_U
        self.ellmax        = ellmax
        self._kind         = 'SXS'
        self.nr            = None
        self.domain        = 'Time'

        self.check_cut_consistency()

        if os.path.exists(self.sxs_data_path) == False:
            if download:
                print("The path ", self.sxs_data_path, " does not exist.")
                print("Downloading the simulation from the SXS catalog.")
                self.download_simulation(ID=ID, path=path)
            else:
                print("Use download=True to download the simulation from the SXS catalog.")
                raise FileNotFoundError(f"The path {self.sxs_data_path} does not exist.")
        
        if isinstance(self.level, int):
            fname = self.get_lev_fname(basename="rhOverM_Asymptotic_GeometricUnits_CoM.h5")
            if os.path.exists(fname):
                self.nr = h5py.File(fname)
            else:
                raise FileNotFoundError('SXS path found, but the requested level ({self.level:d}) is not available!')

        elif self.level is None:
            ref_lv_max = 6
            ref_lv_min = 1
            for lvn in range(ref_lv_max,ref_lv_min,-1):
                fname = self.get_lev_fname(level=lvn,basename="rhOverM_Asymptotic_GeometricUnits_CoM.h5")
                if os.path.exists(fname):
                    self.nr    = h5py.File(fname)
                    self.level = lvn 
                    break
                elif lvn==ref_lv_min:
                    raise RuntimeError('No data for ref-levels:[{ref_lv_min:d},{ref_lv_max:d}] found')
        
        else:
            raise RuntimeError(f'Invalid input for level: {self.level}')
        
        self.load_hlm()
        self.load_metadata()
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
        if level is None: level = self.level
        tojoin = [f'Lev{level:d}']
        if isinstance(basename, str): tojoin.append(basename)
        return os.path.join(self.sxs_data_path, *tojoin)
    
    def download_simulation(self, ID='0001', src='BBH',path=None):
        """
        Download the simulation from the SXS catalog; requires the sxs module
        """
        import sxs

        if path is not None:
            print("Setting the download (cache) directory to ", path)
            os.environ['SXSCACHEDIR'] = path

        nm = 'SXS:'+src+':'+ID
        _  = sxs.load(nm+'/Lev/'+"metadata.json")
        _  = sxs.load(nm+'/Lev/'+"rhOverM_Asymptotic_GeometricUnits_CoM.h5")
        _  = sxs.load(nm+'/Lev/'+"Horizons.h5")
        
        # find folder(s) corresponding to the name, mkdir the new one
        flds = [f for f in os.listdir(os.environ['SXSCACHEDIR']) if nm in f]
        if not os.path.exists(os.path.join(path,'SXS_BBH_'+ID)):
            os.mkdir(os.path.join(path,'SXS_BBH_'+ID))

        # move the files in the folders to the new folder
        for fld in flds:
            for lev in os.listdir(os.path.join(os.environ['SXSCACHEDIR'],fld)):
                try: 
                    # move each Lev folder
                    print(os.path.join(os.environ['SXSCACHEDIR'],fld,lev),'-->',os.path.join(path,'SXS_BBH_'+ID,lev))
                    os.rename(os.path.join(os.environ['SXSCACHEDIR'],fld,lev), os.path.join(path,'SXS_BBH_'+ID,lev))
                except Exception:
                    # Lev already exists, move the files
                    for file in os.listdir(os.path.join(os.environ['SXSCACHEDIR'],fld,lev)):
                        print(os.path.join(os.environ['SXSCACHEDIR'],fld,lev,file),'-->',os.path.join(path,'SXS_BBH_'+ID,lev,file))
                        os.rename(os.path.join(os.environ['SXSCACHEDIR'],fld,lev,file), os.path.join(path,'SXS_BBH_'+ID,lev,file))
                    os.rmdir(os.path.join(os.environ['SXSCACHEDIR'],fld,lev))

            # delete the empty folder
            os.rmdir(os.path.join(os.environ['SXSCACHEDIR'],fld))

        pass
    
    def load_metadata(self):
        with open(self.get_lev_fname(basename="metadata.json"), 'r') as file:
            metadata = json.load(file)
            file.close()
        self.metadata = metadata
        pass

    def load_horizon(self):
        horizon = h5py.File(self.get_lev_fname(basename="Horizons.h5"))
    
        chiA = horizon["AhA.dir/chiInertial.dat"]
        chiB = horizon["AhB.dir/chiInertial.dat"]
        xA   = horizon["AhA.dir/CoordCenterInertial.dat"]
        xB   = horizon["AhB.dir/CoordCenterInertial.dat"]

        self._dyn['t']     = chiA[:,0]
        self._dyn['chi1']  = chiA
        self._dyn['chi2']  = chiB
        self._dyn['x1']    = xA
        self._dyn['x2']    = xB

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

    def load_hlm(self, ellmax=None):
        if ellmax==None: ellmax=self.ellmax
        order   = self.order

        from itertools import product
        modes = [(l, m) for l, m in product(range(2, ellmax+1), range(-ellmax, ellmax+1)) if m!=0 and l >= np.abs(m)]
       
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
            h    = hlm[:, 1] + 1j * hlm[:, 2]
            # amp and phase
            Alm = abs(h)[self.cut_N:]
            plm = np.unwrap(np.angle(h))[self.cut_N:]
            # save in dictionary
            key = (l, m)
            dict_hlm[key] =  {'real': Alm*np.cos(plm), 'imag': Alm*np.sin(plm),
                              'A'   : Alm, 'p' : plm, 
                              'h'   : h[self.cut_N:]
                              }
        self._hlm = dict_hlm
        all_keys = self._hlm.keys()
        pass


