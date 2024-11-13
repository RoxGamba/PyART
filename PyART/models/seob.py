import os, subprocess
import numpy as np
try:
    import pyseobnr.generate_waveform as SEOB
except ModuleNotFoundError:
    print("WARNING: pyseobnr not installed.")
import lal

from ..waveform import Waveform
from ..utils import wf_utils as wfu

class Waveform_SEOB(Waveform):
    """
    Class for SEOBNRv5HM waveforms
    Uses pySEOBNR package
    """
    def __init__(
                    self,
                    pars=None,
                ):
        super().__init__()
        self.pars  = pars
        self._kind = 'SEOB'
        self.check_pars()
        self.SEOB  = SEOB.GenerateWaveform(self.pars)
        self._run()
        pass

    def _run(self):
        # This gives time, modes in physical units
        t, hlm_seob = self.SEOB.generate_td_modes()
        nu          = self.pars["q"]/(1. + self.pars["q"])**2
        if self.pars["use_geometric_units"] == "yes":
            M         = self.pars["mass1"] + self.pars["mass2"]
            fac       = -1 * M * lal.MRSUN_SI / (self.pars["distance"] * lal.PC_SI * 1.e6)
            t         = t/(M * lal.MTSUN_SI)
            for mode in hlm_seob.keys():
                # Also rescale by nu
                hlm_seob[mode] = hlm_seob[mode]/fac/nu

        self._u   = t
        hlm       = convert_hlm(hlm_seob)
        self._hlm = hlm

        self._hp, self._hc = wfu.compute_hphc(hlm, modes=list(hlm.keys()))
        self._dyn = {'t'     : self.SEOB.model.dynamics[:,0],
                     'r'     : self.SEOB.model.dynamics[:,1],
                     'phi'   : self.SEOB.model.dynamics[:,2],
                     'Pr'    : self.SEOB.model.dynamics[:,3],
                     'Pphi'  : self.SEOB.model.dynamics[:,4],
                     'E'     : self.SEOB.model.dynamics[:,5]*nu,
                     'MOmega': self.SEOB.model.dynamics[:,6],
                     }
        self.domain = 'Time'
        return 0
    
    def check_pars(self):
        """
        Check and adjust pars dictionary
        """
        if "q" not in self.pars:
            if "mass1" not in self.pars or "mass2" not in self.pars:
                raise ValueError("SEOB: Neither mass ratio nor individual masses given in input.")
            self.pars["q"] = self.pars["mass1"]/self.pars["mass2"]
        
        if ("mass1" not in self.pars or "mass2" not in self.pars) and self.pars["use_geometric_units"] == "no":
            raise ValueError("SEOB: If not using geom. units, need individual masses.")
        
        if self.pars["use_geometric_units"] == "yes":
            # If using geometric units, set total mass to 100
            self.pars["mass1"] = 100.*self.pars["q"]/(1. + self.pars["q"])
            self.pars["mass2"] = 100./(1. + self.pars["q"])

            # Convert initial frequency to physical units, save geometric in dict
            self.pars["f22_start_geom"] = self.pars["f22_start"]
            self.pars["f22_start"]      = self.pars["f22_start"]/(self.pars["mass1"] + self.pars["mass2"])/lal.MTSUN_SI
        
        # If x, y spin components given, check that approximant is SEOBNRv5->P<-HM
        if max([np.abs(self.pars["spin1x"]), np.abs(self.pars["spin1y"]), 
                np.abs(self.pars["spin2x"]), np.abs(self.pars["spin2y"])]) > 1.e-10:
            if self.pars["approximant"] != "SEOBNRv5PHM":
                print("Switching to SEOBNRv5PHM for non-aligned spins.")
                self.pars["approximant"] = "SEOBNRv5PHM"

    def compute_energetics(self):
        """
        Compute binding energy and angular momentum from dynamics.
        """
        pars    = self.pars
        q       = pars['q']
        q       = float(q)
        nu      = q/(1.+q)**2
        dyn     = self.dyn
        E, j    = dyn['E'], dyn['Pphi']
        Eb      = (E-1)/nu
        self.Eb = Eb
        self.j  = j
        return Eb, j
    
def convert_hlm(hlm):
    """
    Convert the hlm dictionary from SEOB to PyART notation
    """
    hlm_conv = {}
    for key in hlm.keys():
        A   = np.abs(hlm[key])
        p   = -np.unwrap(np.angle(hlm[key]))
        hlm_conv[key] = {'real': A*np.cos(p), 'imag': -1*A*np.sin(p),
                                'A'   : A, 'p' : p,
                                'h'   : A*np.exp(-1j*p)
                                }
    return hlm_conv

def CreateDict(M=1., q=1, 
               chi1z=0., chi2z=0, 
               chi1x=0., chi2x=0,
               chi1y=0., chi2y=0,
               dist=100.,
               iota=0, f0=0.0035, df=1./128., dt=1./2048,
               phi_ref = 0.,
               use_geom="yes", 
               approx="SEOBNRv5HM",
               use_mode_lm=[(2,2)]):
    """
    Create the dictionary of parameters for pyseobnr->GenerateWaveform
    """
    pardic = {
                "q"                   : q,
                "mass1"               : M*q/(1. + q),
                "mass2"               : M/(1. + q),
                "spin1x"              : chi1x,
                "spin1y"              : chi1y,
                "spin1z"              : chi1z,
                "spin2x"              : chi2x,
                "spin2y"              : chi2y,
                "spin2z"              : chi2z,
                "distance"            : dist,
                "inclination"         : iota,
                "phi_ref"             : phi_ref,
                "f22_start"           : f0,
                "deltaF"              : df,
                "deltaT"              : dt,
                "ModeArray"           : use_mode_lm,
                "use_geometric_units" : use_geom,
                "approximant"         : approx
    }
    return pardic