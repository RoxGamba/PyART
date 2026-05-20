import numpy as np
from ..waveform import Waveform
from PyART.utils.wf_utils import get_multipole_dict

try:
    from phenomxpy import IMRPhenomT,IMRPhenomTHM,IMRPhenomTP,IMRPhenomTPHM
except ImportError:
    raise ImportError("WARNING: phenomxpy not installed.")


class Waveform_IMRPhenomT(Waveform):
    """
    Interface for IMRPhenomT via phenomxpy
    """

    def __init__(self, pars=None, approx="IMRPhenomTHM"):
        super().__init__()
        self.pars = pars
        self.approx = approx
        self._run()
        pass

    # ---------------------------
    # Approximant selector
    # ---------------------------
    def _get_approximant(self):
        if self.approx == "IMRPhenomT":
            return IMRPhenomT(**self.pars)

        elif self.approx == "IMRPhenomTHM":
            return IMRPhenomTHM(**self.pars)

        elif self.approx == "IMRPhenomTP":
            return IMRPhenomTP(**self.pars)

        elif self.approx == "IMRPhenomTPHM":
            return IMRPhenomTPHM(**self.pars)

        else:
            raise ValueError("Unknown approximant")

    # ---------------------------
    # Main dispatcher
    # ---------------------------
    def _run(self):
        phenom = self._get_approximant()

        # Compute THM polarizations
        hp, hc, t = phenom.compute_polarizations(times=None)
        
        self._u  = t
        self._hp = np.array(hp)
        self._hc = np.array(hc)
        
        # Compute THM individual modes
        if self.approx[-2:]=='HM':
            if 'PhenomTP' in self.approx:
                hlm_phen, tlm = phenom.compute_CPmodes(times=None)
            else:
                hlm_phen, tlm = phenom.compute_hlms(times=None)

            hlm = {}
            for ky in hlm_phen: # keys: '22', '21', etc.
                l = int(ky[0])
                m = int(ky[1:])
                z = hlm_phen[ky]
                hlm[(l,m)] = get_multipole_dict(z)

            self._hlm = hlm
        pass

def CreateDict(
    q=1.,
    chi1z=0.,
    chi2z=0.,
    chi1x=0.,
    chi2x=0.,
    chi1y=0.,
    chi2y=0.,
    f0=0.0035,
    dt=0.5,
):
    """
    Create the dictionary of parameters for IMRPhenomT (geom units)
    
    Parameters
    ----------
    q : float
        Mass ratio m1/m2 >= 1. Default is 1.
    chi1z : float
        Dimensionless spin of the primary along the orbital angular momentum.
        Default is 0.0.
    chi2z : float
        Dimensionless spin of the secondary along the orbital angular momentum.
        Default is 0.0.
    chi1x : float
        Dimensionless spin of the primary in the orbital plane (x-component).
        Default is 0.0.
    chi2x : float
        Dimensionless spin of the secondary in the orbital plane (x-component).
        Default is 0.0.
    chi1y : float
        Dimensionless spin of the primary in the orbital plane (y-component).
        Default is 0.0.
    chi2y : float
        Dimensionless spin of the secondary in the orbital plane (y-component).
        Default is 0.0.
    f0 : float
        Starting frequency of the (2,2) mode in geometric units. Default is 0.0035.
    dt : float
        Time step in seconds. Default is 0.5
    """
    pars = {
        "eta"     : q/(1+q)**2, 
        "s1"      : [chi1x,chi1y,chi1z],
        "s2"      : [chi2x,chi2y,chi2z],
        "f_min"   : f0,
        "delta_t" : dt,
    }
    return pars


