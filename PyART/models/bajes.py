import numpy as np

try:
    import bajes.obs.gw as gwb
except ModuleNotFoundError:
    print("WARNING: bajes not installed.")

from ..waveform import Waveform
from bajes.obs.gw.approx.nrpm import NRPM
from bajes.obs.gw.utils import lambda_2_kappa
from bajes.obs.gw import __approx_dict__

# todo: use astropy units or similar
msun_m = 1.476625061404649406193430731479084713e3
mpc_m = 3.085677581491367278913937957796471611e22
msun_s = 4.925490947000518753850796748739372504e-6

conversion_dict_from_bajes = {
    "mtot": "M",
    "s1z": "chi1",
    "s2z": "chi2",
    "s1z": "chi1z",
    "s2z": "chi2z",
    "s1x": "chi1x",
    "s2x": "chi2x",
    "s1y": "chi1y",
    "s2y": "chi2y",
    "lambda1": "LambdaAl2",
    "lambda2": "LambdaBl2",
    "phi_ref": "coa_phase",
    "iota": "inclination",
    "f_min": "initial_frequency",
}

conversion_dict_to_bajes = {k: v for v, k in conversion_dict_from_bajes.items()}


class Waveform_NRPM(Waveform):
    """
    Class to attach NRPM to waveforms
    """

    def __init__(self, pars=None, waveform=None, geom_units=False):
        super().__init__()
        self.__set__default_pars__()
        self.pars = {**self.pars, **pars}
        self.pars_bajes = self.__convert_pars__()
        self.wf = waveform
        self._kind = "BaJes"
        self.domain = "Time"
        self.__attach__()

    def __set__default_pars__(self):
        self.pars = {
            "approx": "TEOBResumS",
            "phi_ref": 0.0,
            "distance": 1.0,
            "time_shift": 0.0,
            "iota": 0.0,
            "lmax": 0,
            "srate": 4096,
            "seglen": 16,
            "f_min": 20,
            "eccentricity": 0.0,
            "f_max": 2048,
            "tukey": 0.4 / 16,
        }

    def __convert_pars__(self):
        pars = {}
        for key in self.pars.keys():
            if key in conversion_dict_to_bajes.keys():
                pars[conversion_dict_to_bajes[key]] = self.pars[key]
            else:
                pars[key] = self.pars[key]
        return pars

    def __NRPM__(self, phi_last):

        params = self.pars
        seglen = params["seglen"]
        srate = params["srate"]
        Mtot = params["mtot"]
        q = params["q"]

        kappa2T = lambda_2_kappa(
            params["mtot"] / (1.0 + 1.0 / params["q"]),
            params["mtot"] / (1.0 + params["q"]),
            params["lambda1"],
            params["lambda2"],
        )

        _, _, h22_nrpm = NRPM(
            srate,
            seglen,
            Mtot,
            q,
            kappa2T,
            1.0,
            0.0,
            phi_last,
            f_merg=None,
            alpha=None,
            beta=None,
            phi_kick=params["NRPM_phi_pm"],
            recal=params["recal"],
            output=True,
        )

        # remove initial zeros, this will start at seglen/2
        h22_nrpm = h22_nrpm[len(h22_nrpm) // 2 :]
        h22_nrpm /= msun_m * Mtot
        return h22_nrpm

    def __attach__(self):

        h22_inspiral = self.wf.hlm[(2, 2)]["A"] * np.exp(-1j * self.wf.hlm[(2, 2)]["p"])
        phi_mrg = self.wf.hlm[(2, 2)]["p"][np.argmax(np.abs(h22_inspiral))]
        h22_nrpm = self.__NRPM__(phi_mrg)

        # remove tail before merger and rescale
        amp_fact = np.max(abs(h22_inspiral)) / np.max(abs(h22_nrpm))
        h22_nrpm *= amp_fact

        # cut
        idx_max_nrpm = np.argmax(np.abs(h22_nrpm))
        h22_nrpm = h22_nrpm[idx_max_nrpm:]

        idx_max_insp = np.argmax(np.abs(h22_inspiral))
        h22_inspiral = h22_inspiral[:idx_max_insp]

        # time arrays
        t_nrpm = np.arange(len(h22_nrpm)) / self.pars["srate"]
        t_nrpm /= self.pars["mtot"] * msun_s
        t_inspiral = self.wf.u[:idx_max_insp]

        # build h
        h = np.append(h22_inspiral, h22_nrpm)
        t = np.append(t_inspiral, t_nrpm)

        # interpolate everything
        dt = t[1] - t[0]
        tnew = np.arange(t[0], t[-1] + dt, dt) + dt
        h = np.interp(tnew, t, h)

        self._u = tnew
        self._t = tnew
        self._hlm[(2, 2)] = {
            "A": np.abs(h),
            "p": -np.unwrap(np.angle(h)),
            "z": h,
            "real": np.real(h),
            "imag": np.imag(h),
        }
