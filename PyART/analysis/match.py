"""
Stuff for mismatches, still need to port the parallelization,
the precessing case and debug/test the code
"""

import copy, time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar, dual_annealing
from ..utils import utils as ut
from ..utils import wf_utils as wf_ut

try:
    import lal
except ModuleNotFoundError:
    print("WARNING: lal not installed.")

# PyCBC imports
from pycbc.filter import (
    sigmasq,
    matched_filter_core,
    overlap_cplx,
    optimized_match,
    matched_filter,
    compute_max_snr_over_sky_loc_stat_no_phase,
)
from pycbc.types.timeseries import TimeSeries, FrequencySeries
from pycbc.psd import aLIGOZeroDetHighPower, sensitivity_curve_lisa_semi_analytical
from pycbc.psd.read import from_txt


class Matcher(object):
    """
    Class to compute the match between two waveforms.
    """

    def __init__(
        self,
        WaveForm1,
        WaveForm2,
        settings=None,
        cache={},  # {} or {'h1f':h1f, 'h2f':h2f, 'M':M} (can specify also only h1f or h2f)
    ) -> None:

        self.cache = cache
        self.settings = self.__default_parameters__()
        if settings:
            self.settings.update(settings)
        self.modes = self.settings.get("modes", [])
        del settings

        # Choose the appropriate mismatch function
        if self.settings["kind"] == "single-mode":
            self.match_f = self._compute_mm_single_mode
        elif (
            self.settings["kind"].lower() == "hm"
            or "prec" in self.settings["kind"].lower()
        ):
            self.match_f = self._compute_mm_skymax
        elif self.settings["kind"].lower() == "hm-overlap":
            if self.settings["modes-or-pol"] == "modes":
                raise ValueError("Only 'pol' is allowed for 'hm-overlap' kind")
            self.match_f = self._compute_overlap_skymax
        else:
            raise ValueError(f"Kind '{self.settings['kind']}' not recognized")

        if self.settings["cut_longer"] and self.settings["cut_second_waveform"]:
            raise RuntimeError(
                "The options 'cut_longer' and 'cut_second_waveform' cannot be used together!"
            )

        if self.settings["cut_second_waveform"] or self.settings["cut_longer"]:
            tmrg1, _, _, _ = WaveForm1.find_max() - WaveForm1.u[0]
            tmrg2, _, _, _ = WaveForm2.find_max() - WaveForm2.u[0]
            DeltaT = tmrg2 - tmrg1
            if DeltaT > 0:
                WaveForm2.cut(DeltaT)
            elif DeltaT < 0 and self.settings["cut_longer"]:
                WaveForm1.cut(-DeltaT)

        if self.settings["pre_align"]:
            # pre-align
            umrg1, _, _, _ = WaveForm1.find_max()
            umrg2, _, _, _ = WaveForm2.find_max()

            # shift time of second waveform
            shift = umrg1 - umrg2
            WaveForm2._u = WaveForm2._u + shift

            # fix WaveForm2.hlm[(2,2)] with (approximate) phase difference
            t0 = max(WaveForm1.u[0], WaveForm2.u[0]) + self.settings["pre_align_shift"]
            i01 = np.where(WaveForm1.u >= t0)[0][0]
            i02 = np.where(WaveForm2.u >= t0)[0][0]
            dphi22 = WaveForm1.hlm[(2, 2)]["p"][i01] - WaveForm2.hlm[(2, 2)]["p"][i02]

            for lm in self.settings["modes"]:

                h = WaveForm2.hlm[lm]["z"] * np.exp(-1j * dphi22 / 2 * lm[1])
                WaveForm2._hlm[lm] = wf_ut.get_multipole_dict(h)

        if self.settings["f0_from_merger"]:
            if self.settings["kind"] != "single-mode" or len(self.modes) > 1:
                raise ValueError(
                    "The option 'f0_from_merger' can be used only with a single mode"
                )
            mode = self.modes[0]
            u_mrg, _, _, _, i_mrg = WaveForm1.find_max(return_idx=True)
            p1 = WaveForm1.hlm[mode]["p"]
            Omg1 = np.abs(np.gradient(p1, WaveForm1.u))
            Omg10_postmrg = Omg1[i_mrg]
            f0_postmrg = Omg10_postmrg / (self.settings["M"] * ut.Msun * 2 * np.pi)
            self.settings["initial_frequency_mm"] = f0_postmrg

        # Get local objects with TimeSeries
        wf1 = self._wave2locobj(WaveForm1)
        wf2 = self._wave2locobj(WaveForm2)

        # Determine time length for resizing
        self.settings["tlen"] = self._find_tlen(
            wf1, wf2, resize_factor=self.settings["resize_factor"]
        )
        # Compute and store the mismatch
        mm, out = self.match_f(wf1, wf2, self.settings)
        self.mismatch = 1 - mm
        self.match_out = out
        if "h1f" in out and "h2f" in out:
            self.h1f = out["h1f"]
            self.h2f = out["h2f"]
        else:  # when using skymax
            self.h1f = None
            self.h2f = None
        pass

    def _wave2locobj(self, WaveForm, isgeom=True):
        """
        Extract useful information from WaveForm class
        (see PyART/waveform.py) and store in a lambda-obj
        that will be used in this Matcher class
        """
        if not hasattr(WaveForm, "hp"):
            raise RuntimeError("hp not found! Compute it before calling Matcher")

        wf = lambda: None
        wf.domain = WaveForm.domain
        wf.f = None  # Assume TD waveform at the moment
        wf.hlm = WaveForm.hlm
        wf.t = WaveForm.u
        wf.compute_hphc = WaveForm.compute_hphc

        if self.settings["modes-or-pol"] == "pol":
            # Get updated time and hp/hc-TimeSeries
            # compute hphc, eventually again. WaveForm might have been cut/pre-aligned
            if WaveForm.hp is None:
                coa_phase = self.settings["pol_phase"]
                incl = self.settings["pol_incl"]
                WaveForm.compute_hphc(coa_phase, incl, modes=self.modes)
            wf.hp, wf.hc, wf.u = self._mass_rescaled_TimeSeries(
                WaveForm.u, WaveForm.hp, WaveForm.hc, isgeom=isgeom
            )

        # also update the modes in a TimeSeries
        wf.modes = {}

        # for k in WaveForm.hlm.keys():
        wf.mrg_idx = None
        for k in self.settings["modes"]:
            re = WaveForm.hlm[k]["real"]
            im = WaveForm.hlm[k]["imag"]
            re_lm, im_lm, wf.u = self._mass_rescaled_TimeSeries(
                WaveForm.u, re, im, isgeom=isgeom
            )
            wf.modes[k] = {"real": re_lm, "imag": im_lm}
            if k[0] == 2 and k[1] == 2:
                umrg, _, _, _ = WaveForm.find_max()
                C = wf.u[0]
                D = wf.u[-1]
                A = WaveForm.u[0]
                B = WaveForm.u[-1]
                umrg_TS = (D - C) * (umrg - A) / (B - A) + C
                wf.mrg_idx = np.where(wf.u >= umrg_TS)[0][0]

        return wf

    def _mass_rescaled_TimeSeries(self, u, hp, hc, isgeom=True, kind="cubic"):
        """
        Rescale waveforms with the mass used in settings
        and return TimeSeries.
        If the waveform is not in geom-units, the simply
        return the TimeSeries
        """
        # TODO : test isgeom==False
        dT = self.settings["dt"]
        if isgeom:
            M = self.settings["M"]
            dT_resc = dT / (M * ut.Msun)
            new_u = np.arange(u[0], u[-1], dT_resc)
            hp = ut.spline(u, hp, new_u, kind=kind)
            hc = ut.spline(u, hc, new_u, kind=kind)
        return TimeSeries(hp, dT), TimeSeries(hc, dT), new_u

    def _find_tlen(self, wf1, wf2, resize_factor=2):
        """
        Given two local-waveform objects (see wave2locobj()),
        return the time-length to use in TD-waveform
        conditioning (before match computation)
        """
        dT = self.settings["dt"]
        if self.settings["modes-or-pol"] == "pol":
            h1 = TimeSeries(wf1.hp, dT)
            h2 = TimeSeries(wf2.hp, dT)
        else:
            mode_keys = list(wf1.modes.keys())
            h1 = TimeSeries(wf1.modes[mode_keys[0]]["real"], dT)
            h2 = TimeSeries(wf2.modes[mode_keys[0]]["real"], dT)
        LM = max(len(h1), len(h2))
        tl = (LM - 1) * dT
        tN = ut.nextpow2(resize_factor * tl)
        tlen = int(tN / dT)
        if tlen < LM:
            print(tlen, LM)
        return tlen if tlen > LM else LM

    def __default_parameters__(self):
        """
        Default parameters for the mismatch calculation
        """
        return {
            "kind": "single-mode",
            "modes-or-pol": "modes",
            "modes": [(2, 2)],
            "pre_align": True,
            "pre_align_shift": 0.0,
            "initial_frequency_mm": 20.0,
            "final_frequency_mm": 2048.0,
            "psd": "aLIGOZeroDetHighPower",
            "dt": 1.0 / 4096,
            "M": 100.0,
            "iota": 0.0,
            "coa_phase": np.linspace(0, 2 * np.pi, 1),
            "eff_pols": np.linspace(0, np.pi, 1),
            "pad_end_frac": 0.5,  # fraction of pad after the signal
            "taper": "sigmoid",  # None, 'sigmoid', or 'tukey'
            "taper_start": 0.10,  # parameter for sigmoid or tukey window
            "taper_end": None,  # parameter for sigmoid or tukey window
            "taper_alpha": 0.5,  # alpha parameter for sigmoid or tukey (will be M-normalized)
            "taper_alpha_end": None,  # if specified, use this alpha for the end (only with sigmoid)
            "resize_factor": 4,
            "debug": False,
            "geom": True,
            "cut_longer": False,  # cut longer waveform
            "cut_second_waveform": False,  # cut waveform2 if longer than waveform1
            "f0_from_merger": False,  # Use frequency at merger as initial_frequency_mm
            # (computed from WaveForm1)
        }

    def _get_psd(self, flen, df, fmin):
        """
        Get the PSD for the mismatch calculation
        """
        if self.settings["psd"] == "aLIGOZeroDetHighPower":
            psd = aLIGOZeroDetHighPower(flen, df, fmin)
        elif self.settings["psd"] == "LISA":
            psd = sensitivity_curve_lisa_semi_analytical(flen, df, fmin)
        elif self.settings["psd"] == "flat":
            psd = FrequencySeries(np.ones(flen), delta_f=df)
        elif self.settings["psd"] == "txt":
            psd = from_txt(
                filename=self.settings["asd_file"],
                length=flen,
                delta_f=df,
                low_freq_cutoff=fmin,
                is_asd_file=True,
            )
        else:
            raise ValueError("psd not recognized")
        return psd

    def _get_single_mode_nc(self, wf, settings):
        if settings["modes-or-pol"] == "pol":
            h_nc = wf.hp
        elif settings["modes-or-pol"] == "modes":
            if len(settings["modes"]) > 1:
                raise ValueError("Only one mode is allowed in this function")
            h_nc = wf.modes[settings["modes"][0]]["real"]
        return h_nc

    def _compute_mm_single_mode(self, wf1, wf2, settings):
        """
        Compute the mismatch between two waveforms with only a single mode.
        Use either h+ (modes-or-pol = 'pol') or the mode itself (modes-or-pol = 'modes')
        This is true for non-precessing systems with a single (ell, |m|)
        """

        h1_nc = self._get_single_mode_nc(wf1, settings)
        h2_nc = self._get_single_mode_nc(wf2, settings)

        Mref = settings["M"]

        if "h1f" in self.cache and np.isclose(Mref, self.cache["M"], atol=1e-14):
            h1f = self.cache["h1f"]
        elif wf1.domain == "Time":
            h1, tap_times_w1 = condition_td_waveform(
                h1_nc, settings, mrg_idx=wf1.mrg_idx, return_tap_times=True
            )
            h1f = h1.to_frequencyseries()
        else:
            h1f = h1_nc

        if "h2f" in self.cache and np.isclose(Mref, self.cache["M"], atol=1e-14):
            h2f = self.cache["h2f"]
        elif wf2.domain == "Time":
            h2, tap_times_w2 = condition_td_waveform(
                h2_nc, settings, mrg_idx=wf2.mrg_idx, return_tap_times=True
            )
            h2f = h2.to_frequencyseries()
        else:
            h2f = h2_nc

        if isinstance(settings["initial_frequency_mm"], str):
            if "fAM" in settings["initial_frequency_mm"]:
                fAM = h1f.sample_frequencies[np.argmax(abs(h1f))]
                fms = settings["initial_frequency_mm"]
                if fms.split("fAM")[1] == "":
                    settings["initial_frequency_mm"] = eval(fms.split("fAM")[0]) * fAM
                else:
                    settings["initial_frequency_mm"] = max(
                        eval(fms.split("fAM")[0]) * fAM, eval(fms.split("fAM")[1])
                    )

        if isinstance(settings["final_frequency_mm"], str):
            if "A" in settings["final_frequency_mm"]:
                thr = float(settings["final_frequency_mm"].split("A")[1])
                j_f = np.where(abs(h1f) / (Mref * lal.MTSUN_SI) > thr)[0][0] - 1
                settings["final_frequency_mm"] = h1f.sample_frequencies[j_f]

        assert len(h1f) == len(h2f)
        df = 1.0 / h1f.duration
        flen = len(h1f) // 2 + 1
        psd = self._get_psd(flen, df, settings["initial_frequency_mm"])

        m, j_shift, ph_shift = optimized_match(
            h1f,
            h2f,
            psd=psd,
            low_frequency_cutoff=settings["initial_frequency_mm"],
            high_frequency_cutoff=settings["final_frequency_mm"],
            return_phase=True,
        )

        if (
            settings["debug"]
            and wf1.domain == "Time"
            and wf2.domain == "Time"
            and not self.cache
        ):
            self._debug_plot_waveforms(
                h1_nc,
                h2_nc,
                h1,
                h2,
                psd,
                settings,
                tap_times_w1=tap_times_w1,
                tap_times_w2=tap_times_w2,
                mm=1 - m,
            )

        out = {
            "h1f": h1f,
            "h2f": h2f,
            "j_shift": j_shift * h2.delta_t,
            "ph_shift": ph_shift,
        }
        return m, out

    def _debug_plot_waveforms(
        self,
        h1_nc,
        h2_nc,
        h1,
        h2,
        psd,
        settings,
        tap_times_w1=None,
        tap_times_w2=None,
        six_panels=False,
        mm=None,
    ):
        """
        Plot waveforms and PSD for debugging.
        """

        hf1 = h1.to_frequencyseries()
        f1 = hf1.get_sample_frequencies()
        hf2 = h2.to_frequencyseries()
        f2 = hf2.get_sample_frequencies()
        Af1 = np.abs(hf1)
        Af2 = np.abs(hf2)

        if six_panels:
            figm = 2
            fign = 3
            figsize = (15, 7)
            FT_panels = [3, 6]
        else:
            figm = 2
            fign = 2
            figsize = (10, 7)
            FT_panels = [3, 4]

        plt.figure(figsize=figsize)

        plt.subplot(figm, fign, 1)
        plt.title("Real part of waveforms before conditioning")
        plt.plot(
            h1_nc.sample_times,
            h1_nc,
            label="h1 unconditioned",
            color="blue",
            linestyle="-",
        )
        plt.plot(
            h2_nc.sample_times,
            h2_nc,
            label="h2 unconditioned",
            color="green",
            linestyle="--",
        )
        plt.legend()

        plt.subplot(figm, fign, 2)
        plt.title("Real part of waveforms after conditioning")
        plt.plot(h1.sample_times, h1, label="h1 conditioned", color="blue")
        plt.plot(h2.sample_times, h2, label="h2 conditioned", color="green", ls="--")
        if tap_times_w1 is not None:
            t1 = tap_times_w1["t1"]
            t2 = tap_times_w1["t2"]
            if t1 is not None:
                plt.axvline(t1, color="blue", ls="-")
            if t2 is not None:
                plt.axvline(t2, color="blue", ls="-")
        if tap_times_w2 is not None:
            t1 = tap_times_w2["t1"]
            t2 = tap_times_w2["t2"]
            if t1 is not None:
                plt.axvline(t1, color="green", ls="--")
            if t2 is not None:
                plt.axvline(t2, color="green", ls="--")
        plt.legend()

        if six_panels:
            plt.subplot(figm, fign, 4)
            plt.title("PSD used for match")
            plt.loglog(
                psd.sample_frequencies,
                np.sqrt(psd.data * psd.sample_frequencies),
                label="PSD",
                color="black",
            )
            plt.legend()

            plt.subplot(figm, fign, 5)
            plt.title("Overlap integrand between h1 and h2")
            freq = np.linspace(
                settings["initial_frequency_mm"],
                settings["final_frequency_mm"],
                len(h1),
            )
            plt.plot(freq, hf1.data * np.conjugate(hf2.data), color="red")

        for i in FT_panels:
            plt.subplot(figm, fign, i)
            plt.title("Fourier transforms (abs value)")
            plt.plot(f1, Af1, c="blue", label="FT h1")
            plt.plot(f2, Af2, c="green", label="FT h2")
            plt.axvline(settings["initial_frequency_mm"], lw=0.8, c="r")
            plt.axvline(settings["final_frequency_mm"], lw=0.8, c="r")
            plt.grid()
            plt.legend()
            if i == FT_panels[0]:
                plt.yscale("log")
                plt.xscale("log")
        if mm is not None:
            plt.subplot(figm, fign, 1)
            plt.title(f"mismatch: {mm:.3e}")
        plt.tight_layout()
        if "save" not in settings.keys():
            plt.show()
        else:
            print("Saving to ", settings["save"])
            plt.savefig(f"{settings['save']}", dpi=100, bbox_inches="tight")

    def _compute_overlap_skymax(self, wf1, wf2, settings):
        """
        Same as compute_mm_skymax, but without numerical
        optimization of the orbital phase. Uses directly
        the polarizations of the waveforms, instead of the modes.
        This has to be specified for a **single** value of
        effective polarization.
        """

        k = settings["eff_pols"]
        # target
        sp, sx = wf1.hp, wf1.hc
        sp = condition_td_waveform(sp, settings)
        sx = condition_td_waveform(sx, settings)
        spf = sp.to_frequencyseries()
        sxf = sx.to_frequencyseries()
        s = np.cos(k) * spf + np.sin(k) * sxf

        # model
        hp, hc = wf2.hp, wf2.hc
        hp = condition_td_waveform(hp, settings)
        hc = condition_td_waveform(hc, settings)
        hpf = hp.to_frequencyseries()
        hcf = hc.to_frequencyseries()

        psd = self._get_psd(len(spf), spf.delta_f, settings["initial_frequency_mm"])

        return (
            sky_and_time_maxed_overlap(
                s,
                hpf,
                hcf,
                psd,
                self.settings["initial_frequency_mm"],
                self.settings["final_frequency_mm"],
                kind=self.settings["kind"],
            ),
            {},
        )

    def _compute_mm_skymax(self, wf1, wf2, settings):
        """
        Compute the match between two waveforms with higher modes.
        Use wf1 as a fixed target, and decompose wf2 in modes to find the
        best orbital phase.
        """

        iota = settings["iota"]
        mms = []

        for coa_phase in settings["coa_phase"]:
            sp, sx = wf1.compute_hphc(coa_phase, iota, modes=self.modes)
            sp, sx, _ = self._mass_rescaled_TimeSeries(
                wf1.t, sp, sx, isgeom=settings["geom"]
            )
            sp = condition_td_waveform(sp, settings)
            sx = condition_td_waveform(sx, settings)
            spf = sp.to_frequencyseries()
            sxf = sx.to_frequencyseries()
            psd = self._get_psd(len(spf), spf.delta_f, settings["initial_frequency_mm"])

            for k in settings["eff_pols"]:
                s = np.cos(k) * spf + np.sin(k) * sxf

                mm = self.skymax_match(
                    s,
                    wf2,
                    iota,
                    psd,
                    self.modes,
                    dT=settings["dt"],
                    fmin_mm=settings["initial_frequency_mm"],
                    fmax=settings["final_frequency_mm"],
                )
                mms.append(mm)
        out = {}  # FIXME: just for consistency with compute_mm_single_mode
        return np.average(mm), out

    def skymax_match(
        self, s, wf, inc, psd, modes, dT=1.0 / 4096, fmin_mm=20.0, fmax=2048.0
    ):

        def to_minimize_dphi(x):
            hp, hc = wf.compute_hphc(x, inc, modes=modes)
            hp, hc, _ = self._mass_rescaled_TimeSeries(
                wf.t, hp, hc, isgeom=self.settings["geom"]
            )
            hps = condition_td_waveform(hp, self.settings)
            hxs = condition_td_waveform(hc, self.settings)
            # To FD
            hpf = hps.to_frequencyseries()
            hcf = hxs.to_frequencyseries()
            return 1.0 - sky_and_time_maxed_overlap(
                s,
                hpf,
                hcf,
                psd,
                self.settings["initial_frequency_mm"],
                self.settings["final_frequency_mm"],
                kind=self.settings["kind"],
            )

        res_ms = minimize_scalar(
            to_minimize_dphi,
            method="bounded",
            bounds=(0, 2.0 * np.pi),
            options={"xatol": 1e-15},
        )
        res = to_minimize_dphi(res_ms.x)
        # print('minimize_scalar:', res)
        if res > 1e-2:
            # try also with (more expensive) dual annealing
            # and choose the minimum
            _, res_da = dual_annealing_wrap(
                to_minimize_dphi, [(0.0, 2.0 * np.pi)], maxfun=100
            )
            res = min(res, res_da)

        return 1.0 - res


### other functions, not just code related to the class
def condition_td_waveform(h_in, settings, return_tap_times=False, mrg_idx=None):
    """
    Condition the waveforms before computing the mismatch.
    h is already a TimeSeries
    """
    h = copy.copy(h_in)  # shallow-copy
    hlen = len(h)
    tlen = settings["tlen"]
    ndiff = tlen - hlen
    if settings["pad_end_frac"] > 1:
        raise ValueError("'pad_end_frac' can't be greater than 1.0!")
    npad_after = int(ndiff * settings["pad_end_frac"])
    npad_before = ndiff - npad_after
    h.resize(hlen + npad_after)
    h_numpy = np.pad(h, (npad_before, 0), mode="constant")
    h = TimeSeries(h_numpy, delta_t=h.delta_t)
    if isinstance(settings["taper"], str):
        tap1 = settings["taper_start"]
        tap2 = settings["taper_end"]
        if mrg_idx is None:
            wlen = hlen
        else:
            wlen = mrg_idx
        t1 = npad_before + wlen * tap1 if (tap1 is not None and tap1 > 0) else None
        t2 = (
            npad_before + hlen * (1 - tap2) if (tap2 is not None and tap2 > 0) else None
        )
        t = np.linspace(0, tlen - 1, num=tlen)
        alpha_M = settings["taper_alpha"] / settings["M"]
        if settings["taper_alpha_end"] is not None:
            alpha_end_M = settings["taper_alpha_end"] / settings["M"]
        else:
            alpha_end_M = alpha_M
        h = ut.taper_waveform(
            t,
            h,
            t1=t1,
            t2=t2,
            alpha=alpha_M,
            alpha_end=alpha_end_M,
            kind=settings["taper"],
        )
    else:
        t1 = None
        t2 = None

    if return_tap_times:
        rescaled_t1 = t1 / tlen * h.sample_times[-1] if t1 is not None else None
        rescaled_t2 = t2 / tlen * h.sample_times[-1] if t2 is not None else None
        tap_times = {"t1": rescaled_t1, "t2": rescaled_t2}
        return h, tap_times
    else:
        return h


def dual_annealing_wrap(func, bounds, maxfun=2000):
    result = dual_annealing(
        func, bounds, maxfun=maxfun
    )  # , local_search_options={"method": "Nelder-Mead"})
    opt_pars, opt_val = result["x"], result["fun"]
    return opt_pars, opt_val


def sky_and_time_maxed_overlap(s, hp, hc, psd, low_freq, high_freq, kind="hm"):
    """
    Compute the sky and time maximized overlap between a signal and a template.
    See https://arxiv.org/pdf/1709.09181 and Eq. 10 of https://arxiv.org/pdf/2207.01654
    """
    signal_norm = s / np.sqrt(
        sigmasq(
            s, psd=psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq
        )
    )
    hplus_norm = hp / np.sqrt(
        sigmasq(
            hp, psd=psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq
        )
    )
    hcross_norm = hc / np.sqrt(
        sigmasq(
            hc, psd=psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq
        )
    )
    hphc_corr = np.real(
        overlap_cplx(
            hplus_norm,
            hcross_norm,
            psd=psd,
            low_frequency_cutoff=low_freq,
            high_frequency_cutoff=high_freq,
            normalized=False,
        )
    )
    Iplus = matched_filter(
        hplus_norm,
        signal_norm,
        psd=psd,
        sigmasq=1.0,
        low_frequency_cutoff=low_freq,
        high_frequency_cutoff=high_freq,
    )
    Icross = matched_filter(
        hcross_norm,
        signal_norm,
        psd=psd,
        sigmasq=1.0,
        low_frequency_cutoff=low_freq,
        high_frequency_cutoff=high_freq,
    )

    if "hm" in kind.lower():
        det_stat = compute_max_snr_over_sky_loc_stat_no_phase(
            Iplus,
            Icross,
            hphccorr=hphc_corr,
            hpnorm=1.0,
            hcnorm=1.0,
            thresh=0.1,
            analyse_slice=slice(0, len(Iplus.data)),
        )
    elif kind.lower() == "precessing":
        from pycbc.filter import compute_u_val_for_sky_loc_stat

        det_stat = compute_u_val_for_sky_loc_stat(
            Iplus,
            Icross,
            hphccorr=hphc_corr,
            hpnorm=1.0,
            hcnorm=1.0,
            thresh=0.1,
            analyse_slice=slice(0, len(Iplus.data)),
        )
    else:
        raise ValueError("Currently only HM and precessing is supported!")

    i = np.argmax(det_stat.data)
    o = det_stat[i]
    if o > 1.0:
        o = 1.0

    return o


def time_maxed_overlap(s, hp, hc, psd, low_freq, high_freq, max_pol=True):
    """
    Assume s is + only.
    We allow for a polarization shift, i.e. a **global** change of sign in the waveform.
    TODO: check if this is implemented correctly, see Sec. VD of https://arxiv.org/abs/1812.07865
    """

    ss = sigmasq(
        s, psd=psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq
    )
    hphp = sigmasq(
        hp, psd=psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq
    )
    hp /= np.sqrt(hphp)

    rhop, _, nrm = matched_filter_core(
        hp, s, psd=psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq
    )
    rhop *= nrm

    # allow for different polarization conventions
    # potential global - sign change
    os = []
    for ang in [np.pi / 2, 0.0]:
        re_rhop = np.real(rhop.data * np.exp(2 * 1j * ang))
        num = re_rhop
        o = max(num) / np.sqrt(ss)
        os.append(o)

    # maximize over allowed polarization
    if max_pol:
        o = max(os)
    else:
        o = os[0]

    if o > 1.0:
        o = 1.0
    return o
