from ..waveform import Waveform
from phenomxpy.utils import MasstoSecond, SecondtoMass, AmpNRtoSI
import numpy as np
from PyART.utils.wf_utils import get_multipole_dict

try:
    from phenomxpy import IMRPhenomT,IMRPhenomTHM,IMRPhenomTP,IMRPhenomTPHM
except ImportError:
    raise ImportError("WARNING: phenomxpy not installed.")


class Waveform_IMRPhenomT(Waveform):
    """
    Interface for IMRPhenomT via phenomxpy
    """

    def __init__(self, pars=None, approx="IMRPhenomTHM", kind="TD"):
        super().__init__()
        self.pars = pars
        self.approx = approx
        self._kind = kind
        self._run()
        pass

    # ---------------------------
    # Parameter conversion
    # ---------------------------
    def _build_params(self):
        """
        Assume that pars of self contains:
            - q                   : mass ratio
            - M                   : total mass
            - chi1x, chi1y, chi1z : spin components of body 1
            - chi2x, chi2y, chi2z : spin components of body 2
            - distance            : distance to source
            - inclination         : inclination
            - coalescence_angle   : reference phase
            - initial frequency   : initial frequency
            - srate_interp        : sampling rate
            - use_mode_lm         : list of modes to use for hpc
        """
        # Read in from pars the relevant parameters
        pp = self.pars
        q = pp["q"]
        c1x, c1y, c1z = pp["chi1x"], pp["chi1y"], pp["chi1z"]
        c2x, c2y, c2z = pp["chi2x"], pp["chi2y"], pp["chi2z"]
        flow = pp["initial_frequency"]

        nu = q / (1.0 + q)**2

        params = dict(
            eta = nu,
            s1 = [c1x,c1y,c1z],
            s2 = [c2x,c2y,c2z],   
            f_min = flow,    
            delta_t = 0.5,
        )

        return params

    # ---------------------------
    # Approximant selector
    # ---------------------------
    def _get_approximant(self, params):
        if self.approx == "IMRPhenomT":
            return IMRPhenomT(**params)

        elif self.approx == "IMRPhenomTHM":
            return IMRPhenomTHM(**params)

        elif self.approx == "IMRPhenomTP":
            return IMRPhenomTP(**params)

        elif self.approx == "IMRPhenomTPHM":
            return IMRPhenomTPHM(**params)

        else:
            raise ValueError("Unknown approximant")

    # ---------------------------
    # Main dispatcher
    # ---------------------------
    def _run(self):
        params = self._build_params()
        phen = self._get_approximant(params)
        
        if self.kind == "THM":
            self._run_THM(phen)

        elif self.kind == "TPHM":
            self._run_TPHM(phen)

        elif self.kind == "FD":
            self._run_FD(phen)

        else:
            raise ValueError("kind must be THM, TPHM or FD")

    # ---------------------------
    # Time domain
    # ---------------------------
    def _run_THM(self, phen):

        # Compute THM polarizations
        hp, hc, t = phen.compute_polarizations(times=None)
        
        # Compute THM individual modes
        hlms, t = phen.compute_hlms(times=None)

        hlms_pyart = {}
        for ky in hlms:
            l = int(ky[0])
            m = int(ky[1:])
            z = hlms[ky]
            hlms_pyart[(l,m)] = get_multipole_dict(z)


        self._u = t
        self._hp = np.array(hp)
        self._hc = np.array(hc)

        self.hlms_pyart = hlms_pyart

        pass

    def _run_TPHM(self, phen):

        # Compute TPHM polarizations
        hp, hc, t = phen.compute_polarizations(times=None)
        
        # Compute TPHM individual modes
        hlms, t = phen.compute_CPmodes(times=None)

        hlms_pyart = {}
        for ky in hlms:
            l = int(ky[0])
            m = int(ky[1:])
            z = hlms[ky]
            hlms_pyart[(l,m)] = get_multipole_dict(z)


        self._u = t
        self._hp = np.array(hp)
        self._hc = np.array(hc)

        self.hlms_pyart = hlms_pyart

        pass

    # ---------------------------
    # Frequency domain
    # ---------------------------
    def _run_FD(self, phen):
        # Creation time array
        #srate = self.pars["srate_interp"]
        #duration = self.pars.get("duration", 1.0)

        #t = MasstoSecond(phen.set_time_array(), phen.pWF.total_mass)

        # Compute FD polarizations
        hpf, hcf, f = phen.compute_fd_polarizations(times=None)

        #Calcolo HMs?

        self._f = np.array(f)
        self._hp = np.array(hpf)
        self._hc = np.array(hcf)