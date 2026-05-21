import numpy as np
from ..waveform import Waveform
from PyART.utils.wf_utils import get_multipole_dict

try:
    from phenomxpy import IMRPhenomT, IMRPhenomTHM, IMRPhenomTP, IMRPhenomTPHM
except ImportError:
    raise ImportError("WARNING: phenomxpy not installed.")


class Waveform_IMRPhenomT(Waveform):
    """
    Interface for IMRPhenomT via phenomxpy
    """

    def __init__(self, pars=None, approx="IMRPhenomTHM"):
        """
        Pars as in Waveform_EOB class, the mapping is done internally
        """
        super().__init__()
        self.pars = pars
        self.approx = approx
        self._domain = "Time"
        self._run()
        pass

    def _phenom_params(self):
        pp = self.pars

        for chi in ["chi1", "chi2"]:
            for w in ["x", "y", "z"]:
                pp.setdefault(f"{chi}{w}", 0)

        params = {
            "eta": pp["q"] / (1 + pp["q"]) ** 2,
            "s1": [pp["chi1x"], pp["chi1y"], pp["chi1z"]],
            "s2": [pp["chi2x"], pp["chi2y"], pp["chi2z"]],
            "f_min": pp["initial_frequency"],
            "delta_t": pp["dt"] if "dt" in pp else 0.5,
        }
        return params

    def _get_approximant(self):
        params = self._phenom_params()
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

    def _run(self):
        phenom = self._get_approximant()

        # Compute THM polarizations
        hp, hc, t = phenom.compute_polarizations(times=None)

        self._u = t
        self._hp = np.array(hp)
        self._hc = np.array(hc)

        # Compute THM individual modes
        if self.approx[-2:] == "HM":
            if "PhenomTP" in self.approx:
                hlm_phen, tlm = phenom.compute_CPmodes(times=None)
            else:
                hlm_phen, tlm = phenom.compute_hlms(times=None)

            hlm = {}
            for ky in hlm_phen:  # keys: '22', '21', etc.
                l = int(ky[0])
                m = int(ky[1:])
                z = hlm_phen[ky]
                hlm[(l, m)] = get_multipole_dict(z)

            self._hlm = hlm
        pass
