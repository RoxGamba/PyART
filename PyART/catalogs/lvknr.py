
from ..waveform import  Waveform
try:
    import lalsimulation as lalsim
    import lal
except ImportError:
    raise ImportError("WARNING: LALSimulation and/or LAL not installed.")

import os; import numpy as np;
import h5py

# TODO: read them from utils
Msun_m = 1.4766250614046494e3
Msun_s = 4.9254910255435759e-6

class Waveform_LVKNR(Waveform):
    """
    Class to handle waveforms from the LVCNR catalog
    """

    def __init__(
            self,
            path   =None,
            ID     = None,
            ellmax = 8,
            load_m0 = True,
            download = False,
            nu_rescale = False,
            **kwargs
        ):
    
        super().__init__()
        if path is None:
            raise ValueError("Please provide a path to the waveform file")
        
        self.ID = ID
        self.data_path = os.path.join(path, ID)
        self.ellmax    = ellmax
        self.load_m0   = load_m0
        self.nu_rescale= nu_rescale

        if os.path.exists(self.data_path) == False:
            if download == True:
                print("The path ", self.sxs_data_path, " does not exist.")
                print("Downloading the simulation from the LVCNR catalog.")
                self.download_simulation(ID=ID, path=path)
            else:
                print("Use download=True to download the simulation from the LVCNR catalog.")
                raise FileNotFoundError(f"The path {self.data_path} does not exist.")
        
        # load the data
        self.data = h5py.File(self.data_path, 'r')
        
        self.load_metadata()
        self.load_hlm()

    def load_metadata(self):
        """
        Load (some) metadata from the simulation"
        """

        f = self.data
        M = 1 # set M = 1, always
        self.metadata = {}

        Mt    = f.attrs['mass1'] + f.attrs['mass2']
        m1    = f.attrs['mass1']*M/Mt
        m2    = f.attrs['mass2']*M/Mt
        q     = m1/m2
        nu    = q/(1+q)**2
        flow  = f.attrs['f_lower_at_1MSUN']
        fref  = flow
        spins = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(fref, M, self.data_path)
        c1x, c1y, c1z = spins[0], spins[1], spins[2]
        c2x, c2y, c2z = spins[3], spins[4], spins[5]

        # overwrite some of the metadata
        self.metadata['m1']    = m1
        self.metadata['m2']    = m2
        self.metadata['M']     = m1+m2
        self.metadata['q']     = m1/m2
        self.metadata['nu']    = nu
        self.metadata['chi1x'] = c1x; self.metadata['chi1y'] = c1y;  self.metadata['chi1z'] = c1z
        self.metadata['chi2x'] = c2x; self.metadata['chi2y'] = c2y;  self.metadata['chi2z'] = c2z
        self.metadata['initial_frequency'] = flow*Msun_s
        self.metadata['eccentricity']   = f.attrs['eccentricity']

        # also add SI quantities, useful for loading the waveform
        # (because of the INSANE way LALSimulation functions are written...)

        self.metadata['m1SI']   = m1 * lal.MSUN_SI
        self.metadata['m2SI']   = m2 * lal.MSUN_SI
        self.metadata['frefSI'] = flow

        self.metadata['original'] = {}
        for key in f.attrs.keys():
            self.metadata['original'][key] = f.attrs[key]
        pass

    def load_hlm(self):
        """
        Load the waveform modes
        """

        ellmax   = self.ellmax
        load_m0  = self.load_m0
        metadata = self.metadata
        dT       = 1./262144. # hardcoded to 1/262144 for now
        self._hlm = {}

        # add the modes to the LALdictionary
        from itertools import product
        modes = [(l, m) for l, m in product(range(2, ellmax+1), range(-ellmax, ellmax+1)) if (m!=0 or load_m0) and l >= np.abs(m)]
        modearr = lalsim.SimInspiralCreateModeArray()
        for mode in modes:
            lalsim.SimInspiralModeArrayActivateMode(modearr, mode[0], mode[1])
        
        _,hlms = lalsim.SimInspiralNRWaveformGetHlms(dT,
                                            metadata['m1SI'],
                                            metadata['m2SI'],
                                            1e6*lal.PC_SI,
                                            metadata['frefSI'],
                                            metadata['frefSI'],
                                            metadata['chi1x'],
                                            metadata['chi1y'],
                                            metadata['chi1z'],
                                            metadata['chi2x'],
                                            metadata['chi2y'],
                                            metadata['chi2z'],
                                            self.data_path,
                                            modearr
                                            )
        
        # store the modes
        for i in range(len(modes)):
            l, m = hlms.l, hlms.m
            this_mode =  hlms.mode.data.data
            A = np.abs(this_mode)
            p = np.unwrap(np.angle(this_mode))

            # scale the amplitude
            A *= 1e6*lal.PC_SI/Msun_m
            if not self.nu_rescale:
                A /= metadata['nu']

            self._hlm[(l,m)] = {'real':A*np.cos(p), 'imag':A*np.sin(p),
                                'A':A,   'p':p,
                                'z':this_mode
                                }
            hlms = hlms.next

        # get the time array, transfored to physical units
        self._u = np.array(range(0, len(self._hlm[(2,2)]['real'])))*(dT/Msun_s)
        self._t = self.u

        pass

    def download_simulation(self, ID, path):
        """
        Download the simulation from the LVCNR catalog
        """

        # download the simulation
        pass

if __name__ == '__main__':
    
    # test the class
    ID = 'SXS_BBH_0007_Res5'
    path = 'home/rox/Documents/projects/git_repos/lvcnr-lfs/SXS/'

    w = Waveform_LVKNR(path=path, ID=ID)
    