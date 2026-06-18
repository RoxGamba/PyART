import os, subprocess
import logging
import numpy as np
#from PyART.utils.wf_utils import get_multipole_dict, k_to_ell, k_to_emm

try:
    import pyseobnr.generate_waveform as SEOB
except ModuleNotFoundError:
    logging.warning("pyseobnr not installed.")

from ..waveform import Waveform
from ..utils import wf_utils as wfu
from ..utils import utils as ut


class Waveform_SEOB(Waveform):
    """
    Class for SEOBNRv5HM waveforms
    Uses pySEOBNR package
    """

    def __init__(
        self,
        pars=None,
        approx="SEOBNRv5HM",
    ):
        """
        Initialize the Waveform_SEOB class.

        Parameters
        ----------
        pars : dict
            Dictionary of parameters for the SEOBNR waveform generation.
        """
        super().__init__()
        if pars is None:
            raise RuntimeError("No input parameters given for SEOB!")
        self.pars = pars
        self.approx = approx
        self._kind = "SEOB"

        # do not store as attribute to avoid duplicates
        params = self._SEOB_params()
        self.check_pars(params)
        self.SEOB = SEOB.GenerateWaveform(params)
        self._run(params)
        pass

    def check_pars(self, params):
        """
        Check and adjust pars dictionary
        """

        if "q" not in params and "M" not in params:
            if "mass1" not in params or "mass2" not in params:
                raise ValueError(
                    "SEOB: Neither mass ratio nor individual masses given in input."
                )
            params["q"] = self.pars["mass1"] / self.pars["mass2"]
            params["M"] = self.pars["mass1"] + self.pars["mass2"]

        if ("mass1" not in params or "mass2" not in params) and params[
            "use_geometric_units"
        ] == "no":
            raise ValueError("SEOB: If not using geom. units, need individual masses.")

        if params["use_geometric_units"] == "yes":
            # If using geometric units
            params["mass1"] = params["M"] * params["q"] / (1.0 + params["q"])
            params["mass2"] = params["M"] / (1.0 + params["q"])

            # Convert initial frequency to physical units, save geometric in dict
            params["f22_start"] = (
                params["f22_start_geom"] / params["M"] / ut.consts["Msun"]
            )

        # If x, y spin components given and large enough, check that approximant is SEOBNRv5PHM
        if (
            max(
                [
                    np.abs(params["spin1x"]),
                    np.abs(params["spin1y"]),
                    np.abs(params["spin2x"]),
                    np.abs(params["spin2y"]),
                ]
            )
            >= 1.0e-4
        ):
            if ("eccentricity" in params and params["eccentricity"] != 0.0) or (
                "rel_anomaly" in params and params["rel_anomaly"] != 0.0
            ):
                logging.error(
                    "Non-aligned spins not supported with orbital eccentricity for SEOBNRv5."
                )
            else:
                if params["approximant"] != "SEOBNRv5PHM":
                    logging.info(
                        "In-plane spin components are non-zero; switching to SEOBNRv5PHM."
                    )
                    params["approximant"] = "SEOBNRv5PHM"
        else:
            if ("eccentricity" in self.pars and self.pars["eccentricity"] != 0.0) or (
                "rel_anomaly" in self.pars and self.pars["rel_anomaly"] != 0.0
            ):
                if self.pars["approximant"] != "SEOBNRv5EHM":
                    logging.info("Switching to SEOBNRv5EHM for eccentric waveform.")
                    self.pars["approximant"] = "SEOBNRv5EHM"
                for spin_comp in ["spin1x", "spin1y", "spin2x", "spin2y"]:
                    if self.pars[spin_comp] != 0.0:
                        # logging.warning(f"Setting {spin_comp} to 0 for eccentric waveform.")
                        self.pars[spin_comp] = 0.0

    def _SEOB_params(self):
        pp = self.pars

        for i in ["1", "2"]:
            for w in ["x", "y", "z"]:
                pp[f"spin{i}{w}"] = self.pars[f"chi{i}{w}"]

        params = {
            "M": pp["M"],
            "q": pp["q"],
            "use_geometric_units": pp["use_geometric_units"],
            "f22_start_geom": pp["initial_frequency"],
            "spin1x": pp["spin1x"],
            "spin1y": pp["spin1y"],
            "spin1z": pp["spin1z"],
            "spin2x": pp["spin2x"],
            "spin2y": pp["spin2y"],
            "spin2z": pp["spin2z"],
            "distance": pp["distance"],
        }

        if ("eccentricity" in pp) and ("rel_anomaly" in pp):
            params["eccentricity"] = pp["eccentricity"]
            params["rel_anomaly"] = pp["rel_anomaly"]

        # modes selection:
        if "mode_array" in pp:
            mode_array = pp["mode_array"]
        elif "use_mode_lm" in pp:
            mode_array = []
            for k in pp["use_mode_lm"]:
                mode_array.append((wfu.k_to_ell(k), wfu.k_to_emm(k)))
        else:
            mode_array = [(2, 2)]
        params["mode_array"] = mode_array

        if "dt" in pp:
            params["deltaT"] = pp["dt"]
        else:
            params["deltaT"] = 1 / 2048

        # Add rest of parameter in pp to params
        for key in pp.keys():
            if key not in params:
                params[key] = pp[key]

        return params

    def _run(self, params):
        """
        Run the SEOB waveform generation and store the results in the class attributes.
        This method generates the waveform modes, computes the plus and cross polarizations,
        and extracts the dynamics.
        """
        # This gives time, modes in physical units
        t, hlm_seob = self.SEOB.generate_td_modes()
        nu = params["q"] / (1.0 + params["q"]) ** 2
        if params["use_geometric_units"] == "yes":
            M = params["M"]
            fac = (
                -1
                * M
                * ut.consts["Msun"]
                * ut.consts["c_SI"]
                / (params["distance"] * ut.consts["pc_SI"] * 1.0e6)
            )
            t = t / (M * ut.consts["Msun"])
            for mode in hlm_seob.keys():
                # Also rescale by nu
                hlm_seob[mode] = hlm_seob[mode] / fac / nu

        self._u = t
        # hlm = convert_hlm(hlm_seob)
        hlm = {}
        for ky in hlm_seob:
            hlm[ky] = wfu.get_multipole_dict(hlm_seob[ky])

        self._hlm = hlm

        self._hp, self._hc = wfu.compute_hphc(hlm, modes=list(hlm.keys()))
        if self.approx == "SEOBNRv5EHM":
            idx_H = 8
            idx_MOmg = 9
        else:
            idx_H = 5
            idx_MOmg = 6

        self._dyn = {
            "t": self.SEOB.model.dynamics[:, 0],
            "r": self.SEOB.model.dynamics[:, 1],
            "phi": self.SEOB.model.dynamics[:, 2],
            "Pr": self.SEOB.model.dynamics[:, 3],
            "Pphi": self.SEOB.model.dynamics[:, 4],
            "E": self.SEOB.model.dynamics[:, idx_H] * nu,
            "MOmega": self.SEOB.model.dynamics[:, idx_MOmg],
        }
        self._domain = "Time"
        return 0

    def compute_energetics(self, params):
        """
        Compute binding energy and angular momentum from dynamics.
        """
        pars = params
        q = pars["q"]
        q = float(q)
        nu = q / (1.0 + q) ** 2
        dyn = self.dyn
        E, j = dyn["E"], dyn["Pphi"]
        Eb = (E - 1) / nu
        self.Eb = Eb
        self.j = j
        return Eb, j


def CreateDict(
    M=1.0,
    q=1,
    chi1z=0.0,
    chi2z=0,
    chi1x=0.0,
    chi2x=0,
    chi1y=0.0,
    chi2y=0,
    dist=100.0,
    iota=0,
    f0=0.0035,
    df=1.0 / 128.0,
    dt=1.0 / 2048,
    phi_ref=0.0,
    ecc=0.0,
    anomaly=0.0,
    use_geom="yes",
    approx="SEOBNRv5HM",
    use_mode_lm=[(2, 2)],
    lmax_nyquist=1,
):
    """
    Create the dictionary of parameters for pyseobnr->GenerateWaveform

    Parameters
    ----------
    M : float
        Total mass in solar masses. Default is 1.0.
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
    dist : float
        Distance to the source in Mpc. Default is 100.0.
    iota : float
        Inclination angle in radians. Default is 0 (face-on).
    f0 : float
        Starting frequency of the (2,2) mode in geometric units. Default is 0.0035.
    df : float
        Frequency resolution in Hz. Default is 1/128.
    dt : float
        Time step in seconds. Default is 1/2048.
    phi_ref : float
        Reference phase at f0 in radians. Default is 0.0.
    ecc : float
        Initial eccentricity at f0. Default is 0.0.
    anomaly : float
        Relativistic anomaly at f0 in radians. Default is pi.
    use_geom : str
        Whether to use geometric units ('yes' or 'no'). Default is 'yes'.
    approx : str
        Approximant to use ('SEOBNRv5HM', 'SEOBNRv5PHM', 'SEOBNRv5EHM').
        Default is 'SEOBNRv5HM'.
    use_mode_lm : list of tuple
        List of (l, m) tuples specifying which modes to use. Default is [(2, 2)].
    lmax_nyquist : int
        Maximum l mode for Nyquist frequency check. Default is 1 (i.e. no check).
    """
    pardic = {
        "q": q,
        "mass1": M * q / (1.0 + q),
        "mass2": M / (1.0 + q),
        "spin1x": chi1x,
        "spin1y": chi1y,
        "spin1z": chi1z,
        "spin2x": chi2x,
        "spin2y": chi2y,
        "spin2z": chi2z,
        "distance": dist,
        "inclination": iota,
        "phi_ref": phi_ref,
        "f22_start": f0,
        "eccentricity": ecc,
        "rel_anomaly": anomaly,
        "deltaF": df,
        "deltaT": dt,
        "ModeArray": use_mode_lm,
        "use_geometric_units": use_geom,
        "approximant": approx,
        "lmax_nyquist": lmax_nyquist,
    }
    return pardic
