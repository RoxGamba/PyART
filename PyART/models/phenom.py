import numpy as np
import logging
from ..waveform import Waveform
from PyART.utils.wf_utils import get_multipole_dict, k_to_ell, k_to_emm

try:
    from phenomxpy import IMRPhenomT, IMRPhenomTHM, IMRPhenomTP, IMRPhenomTPHM
except ImportError:
    raise ImportError("WARNING: phenomxpy not installed.")

class Waveform_IMRPhenomT(Waveform):
    """
    Interface for IMRPhenomT via phenomxpy
    """

    def __init__(
        self, 
        pars=None, 
        approx="IMRPhenomTHM", 
        reference_frame="CP", 
        skip_hphc=False,
    ):
        """
        Pars as in Waveform_EOB class, the mapping is done internally
        """
        super().__init__()
        self.pars = pars
        self.approx = approx
        self._domain = "Time"
        self.reference_frame = reference_frame
        self._run(skip_hphc=skip_hphc)
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
        }
        if "f_max" in pp:
            if "dt" in pp:
                logging.warning("If f_max is provided, dt is ignored")
            params["f_max"] = pp["f_max"]
            params["delta_t"] = 0.5 / pp["f_max"]
        else:
            params["delta_t"] = pp["dt"] if "dt" in pp else 0.5

        # modes selection:
        if "mode_array" in pp:
            mode_array = pp["mode_array"]
        elif "use_mode_lm" in pp:
            mode_array = []
            for k in pp["use_mode_lm"]:
                mode_array.append((k_to_ell(k),k_to_emm(k)))
        else:
            mode_array = [(2,2)]
        params["mode_array"] = mode_array
            
        return params

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

    def _run(self, skip_hphc=False):
        params = self._phenom_params()
        phenom = self._get_approximant(params)

        # Compute THM polarizations
        if skip_hphc:
            if "HM" not in self.approx:
                raise RuntimeError("Skipping hp,hc computation but no HMs available!")
        else:
            hp, hc, t = phenom.compute_polarizations(times=None)
            self._u = t
            self._hp = np.array(hp)
            self._hc = np.array(hc)

        # Compute THM individual modes
        if self.approx[-2:] == "HM":
            if "PhenomTP" in self.approx:
                if self.reference_frame == "CP":  # co-precessing
                    hlm_phen, tlm = phenom.compute_CPmodes(times=None)
                elif self.reference_frame == "J":  # J-frame
                    hlm_phen, tlm = phenom.compute_Jmodes(times=None)
                elif self.reference_frame == "L0":  # L0-frame
                    hlm_phen, tlm = phenom.compute_L0modes(times=None)
                else:
                    raise ValueError(f"Unknown reference frame: {self.reference_frame}")
            else:
                hlm_phen, tlm = phenom.compute_hlms(times=None)
            
            hlm = {}
            for ky in params["mode_array"]:
                # looping on mode_array since phenom automatically removes 
                # odd-m modes for q=1., but we prefer to have zero arrays
                l = ky[0]
                m = ky[1]
                kys = f'{l:d}{m:d}'
                if kys in hlm_phen:
                    z = hlm_phen[kys]
                else:
                    z = np.zeros_like(tlm)
                hlm[(l,m)] = get_multipole_dict(z)

            if skip_hphc:
                self._u = tlm
            self._hlm = hlm
        pass
