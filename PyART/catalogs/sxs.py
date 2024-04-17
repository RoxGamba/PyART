import numpy as np
import h5py; import json
from ..waveform import  Waveform

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
                    path   ='../dat/SXS/SXS',
                    ID     ='0001',
                    order  ="Extrapolated_N2.dir",
                    level = None,
                    cut_N = 300
                ):
        super().__init__()
        self.ID            = ID
        self.sxs_data_path = path+'_BBH_'+ID
        self.order         = order
        self.level         = level
        self.cut           = cut_N
        self._kind         = 'SXS'
        
        if level == None:
            # Default behavior: load only the highest level
            for lv in ['/Lev6','/Lev5','/Lev4', '/Lev3', '/Lev2', '/Lev1']:
                try:
                    self.nr = h5py.File(self.sxs_data_path+lv+"/rhOverM_Asymptotic_GeometricUnits_CoM.h5")
                    break
                except Exception: 
                    continue
        else:
            self.nr = h5py.File(self.sxs_data_path+level+"/rhOverM_Asymptotic_GeometricUnits_CoM.h5")

        self.load_hlm()
        self.load_metadata()
        #self.compute_hphc()
        pass

    def load_metadata(self):
        if self.level == None:
            # Default behavior: load only the highest level
            for lv in ['/Lev6','/Lev5','/Lev4', '/Lev3', '/Lev2', '/Lev1']:
                try:
                    with open(self.sxs_data_path +lv+"/metadata.json", 'r') as file:
                        metadata = json.load(file)
                        file.close()
                    self.metadata = metadata
                    break
                except Exception:
                    continue
        else:
            with open(self.sxs_data_path +lv+"/metadata.json", 'r') as file:
                metadata = json.load(file)
                file.close()
            self.metadata = metadata

        pass

    def load_hlm(self):
        order   = self.order
        modes   = [[l,m] for l in range(2,9) for m in range(1,l+1)]
        self._u  = self.nr[order]['Y_l2_m2.dat'][:, 0][self.cut:]
        dict_hlm = {}
        for mode in modes:
            l    = mode[0]; m = mode[1]
            mode = "Y_l" + str(l) + "_m" + str(m) + ".dat"
            hlm  = self.nr[order][mode]
            h    = hlm[:, 1] + 1j * hlm[:, 2]
            # amp and phase
            Alm = abs(h)[self.cut:]
            plm = np.unwrap(np.angle(h))[self.cut:]
            # save in dictionary
            key = (l, m)
            dict_hlm[key] = [Alm, plm]
        self._hlm = dict_hlm
        pass
