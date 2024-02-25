import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import h5py 
import glob

from ..waveform import  Waveform

# This class is used to load the RIT data and store it in a convenient way
class RIT(Waveform):

    def __init__(self,
                 basepath = '../dat/RIT/',
                 psi_path = None,
                 h_path   = None,
                 mtdt_path= None,
                 ell_emms = 'all'
                 ) -> None:
        
        super().__init__()

        self.t_psi  = None
        self.t_h    = None
        self.ell_emms = ell_emms
        self.metadata      = None
        self.metadata_psi4 = None

        # metadata available
        if mtdt_path is not None:
            self.mtdt_path = basepath+mtdt_path
            self.metadata  = self.load_metadata(self.mtdt_path)

        # psi4 available
        if psi_path is not None:
            self.psi_path      = basepath+psi_path
            self.mtdt_psi4     = basepath+psi_path+'Metadata'
            self.metadata_psi4 = self.load_metadata(self.mtdt_psi4)
            self.load_psi4()
        
        # strain available
        if h_path is not None:
            self.h_file   = h5py.File(basepath+h_path, 'r')
            self.load_h()

        pass

    def load_psi4(self):

        files = glob.glob(self.psi_path + '*.asc')
        d = {}

        if self.ell_emms == 'all': 
            modes = [(ell, emm) for ell in range(2,6) for emm in range(-ell, ell+1)]
        else:
            modes = self.ell_emms

        for ff in files:
            ell = int((ff.split('/')[-1]).split('_')[1][1:])
            emm = int((ff.split('/')[-1]).split('_')[2][1:])
            if (ell, emm) not in modes:
                continue
            # print(f"Loading {ff}")
            try:
                t,re,im,A,p = np.loadtxt(ff, unpack=True, skiprows=4, usecols=(0,1,2,3,4) )
            except Exception:
                t,re,im,A,p,o = np.loadtxt(ff, unpack=True, skiprows=4, usecols=(0,1,2,3,4))
            d[(ell,emm)] = {'real':re, 'imag':im, 'A':A, 'p':p, 'h': A*np.exp(-1j*p)}
        
        self._psi4lm = d
        self.t_psi  = t
        pass

    def load_h(self):

        d  = {}
        f  = self.h_file
        th = f['NRTimes'][:]
        if self.ell_emms == 'all': 
            modes = [(ell, emm) for ell in range(2,6) for emm in range(-ell, ell+1)]
        else:
            modes = self.ell_emms

        for mm in modes:
            ell, emm = mm
            try:
                A   =  f[f'amp_l{ell}_m{emm}']['Y'][:]
                A_u =  f[f'amp_l{ell}_m{emm}']['X'][:]
                p   = -f[f'phase_l{ell}_m{emm}']['Y'][:]
                p_u =  f[f'phase_l{ell}_m{emm}']['X'][:]
                # interp to common time array
                A   = self.__interp_qnt__(A_u, A, th)
                p   = self.__interp_qnt__(p_u, p, th)
                d[(ell, emm)] = {'real' : A*np.cos(p), 'imag': -A*np.sin(p), 'A':A, 'p':p, 'h': A*np.exp(-1j*p)}

            except KeyError:
                pass
                    
        self._hlm = d
        self.t_h  = th.astype(np.float64)
        self._t   = self.t_h
        self._u   = self.t_h
        pass

    def load_metadata(self, path):

        nm     = path
        metadata = {}
        with open(nm, 'r') as f:

            lines = [l for l in f.readlines() if l.strip()] # rm empty

            for line in lines[1:]:
                if line[0]=="#": continue
                line               = line.rstrip("\n")
                #line = line.split("#", 1)[0]
                key, val           = line.split("= ")
                key                = key.strip()
                metadata[key] = val

        return metadata 


    def compute_initial_data(self):
        """
        Compute initial data (J0, S0, L0) from the metadata
        """

        if self.metadata is not None:
            mtdt = self.metadata
        elif self.metadata_psi4 is not None:
            mtdt = self.metadata_psi4
        elif (self.metadata is None and self.metadata_psi4 is None):
            print("No metadata loaded")
            raise FileNotFoundError("No metadata read. Please load metadata first.")        

        try:
            chi1x = float(mtdt['initial-bh-chi1x']);  chi2x = float(mtdt['initial-bh-chi2x'])
            chi1y = float(mtdt['initial-bh-chi1y']);  chi2y = float(mtdt['initial-bh-chi2y'])
        except KeyError:
            chi1x = 0.;  chi2x = 0.
            chi1y = 0.;  chi2y = 0.
        chi1z = float(mtdt['initial-bh-chi1z']);  chi2z = float(mtdt['initial-bh-chi2z'])
        chi1  = np.array([chi1x, chi1y, chi1z])
        chi2  = np.array([chi2x, chi2y, chi2z])

        # masses
        m1 = float(mtdt['initial-mass1']);  m2 = float(mtdt['initial-mass2'])
        M  = m1 + m2
        X1 = m1/M; X2 = m2/M

        # Spin vectors
        S1 = chi1*m1**2
        S2 = chi2*m2**2

        # Ang momentum
        Jx = float(mtdt['initial-ADM-angular-momentum-x'])
        Jy = float(mtdt['initial-ADM-angular-momentum-y'])
        Jz = float(mtdt['initial-ADM-angular-momentum-z'])
        J  = np.array([Jx, Jy, Jz])

        # Orb ang momentum
        L = J - S1 - S2

        self._dyn['id'] = {
                            'm1':m1, 'm2':m2, 'M':M, 'X1':X1, 'X2':X2, 
                            'S1':S1, 'S2':S2, 'L0':L,
                            'J0':J
                        }
        pass

    def compute_dynamics(self):
        # compute dynamics
        pass
    
    def __interp_qnt__(self, x, y, x_new):

        f  = interpolate.interp1d(x, y)
        yn = f(x_new)

        return yn

if __name__ == '__main__':

    psi_path= 'Psi4/ExtrapPsi4_RIT-eBBH-1634-n100-ecc/'
    h_path  = 'Strain/ExtrapStrain_RIT-eBBH-1634-n100.h5'

    mtdt_path = None
    r = RIT(psi_path=psi_path, h_path=h_path, mtdt_path=mtdt_path)

    r.compute_initial_data()
    print(r.dyn['id'])

    # plot h22
    plt.plot(r.t_h, r.hlm[(2,2)]['real'])
    plt.plot(r.t_h, r.hlm[(2,2)]['A'])
    plt.show()

    # plot psi4
    for k in r.psi4lm.keys():
        plt.plot(r.t_psi, r.psi4lm[k]['A'], label=k)
    plt.legend(ncol=3)
    plt.show()

    # symmetry between +m and -m
    plt.plot(r.t_h, r.hlm[(2,2)]['A'])
    plt.plot(r.t_h, r.hlm[(2,2)]['A']-r.hlm[(2,-2)]['A'])
    plt.show()

    # hierarchy of modes
    for k in r.hlm.keys():
        plt.plot(r.t_h, r.hlm[k]['A'], label=k)
    plt.legend(ncol=3)
    plt.show()
