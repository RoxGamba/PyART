import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import h5py, wget, glob, os 

from ..waveform import  Waveform

try:
    from mayawaves.coalescence import Coalescence
except Exception:
    raise ModuleNotFound("Need mayawaves to handle the MAYA catalog")

# This class is used to load the RIT data and store it in a convenient way
class MAYA(Waveform):

    def __init__(self,
                 basepath = '/data/numrel/MAYA/',
                 id = None,
                 ell_emms = 'all'
                 ) -> None:
        
        super().__init__()
        h5file = id.upper()+'.h5'
        path = os.path.join(basepath, h5file)
        # Download file in Maya Format from repository
        if not glob.glob(path):
            print(f'Downloading NR file {h5file}...')
            wget.download('https://cgpstorage.ph.utexas.edu/maya_format/'+h5file, out=path)
        self.ell_emms = ell_emms
        self.h_file   = h5py.File(path, 'r')
        self.coalescence = Coalescence(path)

        self.load_h()
        self.metadata = self.load_metadata()

        pass

    def load_h(self):

        d  = {}
        if self.ell_emms == 'all': 
            modes = [(ell, emm) for ell in range(2,6) for emm in range(-ell, ell+1)]
        else:
            modes = self.ell_emms
        
        self.coalescence.set_radiation_frame(center_of_mass_corrected=True)

        for mm in modes:
            ell, emm = mm
            try:
                t, re, im = self.coalescence.strain_for_mode(l=ell, m=emm)
                A         = np.sqrt(re**2 + im**2)
                p         = np.unwrap(np.angle(re+1j*im))
            except KeyError:
                pass

            d[(ell,emm)] = {'real':re, 'imag':im, 'A':A, 'p':p}    
            
        self._hlm = d
        self._t   = t
        self._u   = t
        pass

    def load_metadata(self):

        parfile  = dict(self.h_file['parfile'].attrs.items())
        f        = parfile['par_content']
        metadata = {}

        lines = [l for l in f.split('\n') if l.strip()] # rm empty
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
        else:
            print("No metadata loaded")
            raise FileNotFoundError("No metadata read. Please load metadata first.")        

        try:
            chi1x = float(mtdt['twopunctures::par_s_plus[0]']);  chi2x = float(mtdt['twopunctures::par_s_minus[0]'])
            chi1y = float(mtdt['twopunctures::par_s_plus[1]']);  chi2y = float(mtdt['twopunctures::par_s_minus[1]'])
        except KeyError:
            chi1x = 0.;  chi2x = 0.
            chi1y = 0.;  chi2y = 0.
        chi1z = float(mtdt['twopunctures::par_s_plus[2]']);  chi2z = float(mtdt['twopunctures::par_s_minus[2]'])
        chi1  = np.array([chi1x, chi1y, chi1z])
        chi2  = np.array([chi2x, chi2y, chi2z])

        # masses
        m1 = float(mtdt['TwoPunctures::target_M_plus']);  m2 = float(mtdt['TwoPunctures::target_M_minus'])
        M  = m1 + m2
        X1 = m1/M; X2 = m2/M

        # Spin vectors
        S1 = chi1*m1**2
        S2 = chi2*m2**2

        # Orbital momenta
        Pp = [float(mtdt['twopunctures::par_P_plus['+str(i)+']']) for i in range(3)]
        Pm = [float(mtdt['twopunctures::par_P_minus['+str(i)+']']) for i in range(3)]
        
        # position (TODO: correct)
        Rp = [float(mtdt['TwoPunctures::par_b']), 0, 0]
        Rm = [-float(mtdt['TwoPunctures::par_b']), 0, 0]

        # Ang momentum
        J  = np.cross(Rp,Pp)+np.cross(Rm,Pm)+S1+S2

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
    r = MAYA(id='MAYA1056')
    r.compute_initial_data()
    print(r.dyn['id'])

    # plot h22
    plt.plot(r.t, r.hlm[(2,2)]['real'])
    plt.plot(r.t, r.hlm[(2,2)]['A'])
    plt.savefig('test.png')
