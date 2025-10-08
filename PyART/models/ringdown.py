import logging
import numpy as np
import ringdown_fits as rdf
import ringdown_fits_noncirc as ncf


class TEOBRingdown(object):
    def __init__(
        self,
        t,
        t_ref=0.0,
        parameters=None,
        modes=[(2, 2)],
        use_fits=True,
        noncircular=False,
        kind="aligned-spins-equal-mass",
        databases=["RIT"],
    ) -> None:
        """
        Class to generate ringdown waveforms using fits to the TEOBResumS model
        parameters must be a dictionary containing either the intrinsic parameters
        of the source or the ringdown parameters directly.

        Each mode depends on the following parameters:
        - four amplitude paramters: a1, a2, a3, a4
        - four phase parameters:    p1, p2, p3, p4
        - the real and imaginary parts of the fundamental QNM frequency: alpha1, omega1
        - a time shift  w.r.t. the peak of the 22 mode (Delta_T)
        - a phase shift w.r.t. the (2,2) mode (Delta_phi)

        From the way that the template is constructed, not all these parameters are independent.
        Given the peak Amplitude, frequency and the properties of the final black hole (mass and spin),
        the only free parameters are a3, p3, p4, Delta_T and Delta_phi. The other parameters are computed from these.
        """

        self.params = parameters.copy()  # make a copy to avoid overwriting the original
        self.modes = modes
        self.rd_params = {}
        self.hlm = {}

        # set up the ringdown parameters
        for mode in modes:
            self.rd_params[mode] = {}

        # use fits if requested
        # the user must provide the intrinsic parameters
        # within the parameters dictionary
        if use_fits:
            self._intrinsic_to_rdpars(modes)
            if noncircular:
                self._noncircular_corrections(modes, kind=kind, databases=databases)

        # overwrite with user-provided parameters
        self._overwrite_rdpars(self.params)

        # compute the QNM-related parameters
        # these are separated from the other fits
        # because they depend on the final BH properties,
        # which may be overwritten by the user
        for mode in modes:
            ell, emm = mode[0], mode[1]
            self._eval_fits_qnm(self.rd_params["af"], ell, emm)

        # set other parameters
        self._set_rd_params(modes)

        # compute the modes
        self.compute_modes(modes, t_ref, t)

        pass

    def _intrinsic_to_rdpars(self, modes):
        """
        Compute the ringdown parameters from the intrinsic ones
        """

        # unpack parameters
        q = self.params["q"]
        chiA = self.params["chiA"]
        chiB = self.params["chiB"]

        # compute useful qnts from intrinsic parameters
        nu = q / (1.0 + q) ** 2
        X1 = q / (1.0 + q)
        X2 = 1.0 / (1.0 + q)
        X12 = X1 - X2
        a0 = X1 * chiA[2] + X2 * chiB[2]
        a12 = X1 * chiA[2] - X2 * chiB[2]
        S_hat = 0.5 * (a0 + X12 * a12)
        S_bar = 0.5 * (X12 * a0 + a12)

        # eval fits
        for mode in modes:
            ell, emm = mode[0], mode[1]
            self._eval_fits(
                nu, X1, X2, X12, a0, a12, S_hat, S_bar, chiA[2], chiB[2], ell, emm
            )

        pass

    def _set_rd_params(self, modes):
        # Compute the non-independent parameters
        # of the template

        Mbhf = self.rd_params["Mbhf"]

        for mode in modes:

            alph1 = self.rd_params[mode]["alpha1"]
            alph21 = self.rd_params[mode]["alpha21"]
            Apk = self.rd_params[mode]["Apk"]
            Omgpk = self.rd_params[mode]["Omgpk"]
            c3A = self.rd_params[mode]["a3"]
            c3p = self.rd_params[mode]["p3"]
            c4p = self.rd_params[mode]["p4"]
            omg1 = self.rd_params[mode]["omega1"]
            dOmg = rdf._dOmega(omg1, Mbhf, Omgpk)

            c2A = 0.5 * alph21
            c1A = Apk * alph1 * np.cosh(c3A) * np.cosh(c3A) / c2A
            c4A = Apk - c1A * np.tanh(c3A)
            c2p = alph21
            c1p = dOmg * (1.0 + c3p + c4p) / (c2p * (c3p + 2 * c4p))

            self.rd_params[mode]["dOmega"] = dOmg
            self.rd_params[mode]["a1"] = c1A
            self.rd_params[mode]["a2"] = c2A
            self.rd_params[mode]["a4"] = c4A
            self.rd_params[mode]["p1"] = c1p
            self.rd_params[mode]["p2"] = c2p

    def _rd_template(self, x, ell, emm):
        # TEOBResumS ringdown template

        mode = (ell, emm)

        a1 = self.rd_params[mode]["a1"]
        a2 = self.rd_params[mode]["a2"]
        a3 = self.rd_params[mode]["a3"]
        a4 = self.rd_params[mode]["a4"]
        b1 = self.rd_params[mode]["p1"]
        b2 = self.rd_params[mode]["p2"]
        b3 = self.rd_params[mode]["p3"]
        b4 = self.rd_params[mode]["p4"]
        sigmar = self.rd_params[mode]["alpha1"]
        sigmai = self.rd_params[mode]["omega1"]

        amp = a1 * np.tanh(a2 * x + a3) + a4
        phase = -b1 * np.log(
            (1.0 + b3 * np.exp(-b2 * x) + b4 * np.exp(-2.0 * b2 * x)) / (1.0 + b3 + b4)
        )

        A = amp * np.exp(-sigmar * x)
        # amplitude
        phase = -(phase - sigmai * x)
        # phase, minus sign in front by convention

        return A, phase

    def _eval_fits(
        self, nu, X1, X2, X12, a0, a12, S_hat, S_bar, chi1, chi2, ell, emm
    ) -> None:
        # Evaluate the fits for each mode

        # ringdown coeffs
        self.rd_params[(ell, emm)]["a3"] = rdf._c3_A(nu, X12, S_hat, a12, ell, emm)
        self.rd_params[(ell, emm)]["p3"] = rdf._c3_phi(nu, X12, S_hat, ell, emm)
        self.rd_params[(ell, emm)]["p4"] = rdf._c4_phi(nu, X12, S_hat, ell, emm)

        # peak quantities
        omgpk = rdf._omega_peak(nu, X12, S_hat, a0, ell, emm)
        self.rd_params[(ell, emm)]["Omgpk"] = omgpk
        self.rd_params[(ell, emm)]["Apk"] = rdf._amplitude_peak(
            nu, X12, S_hat, a12, S_bar, a0, omgpk, ell, emm
        )
        self.rd_params[(ell, emm)]["DeltaT"] = rdf._DeltaT(nu, X12, S_hat, a0, ell, emm)

        # final BH
        self.rd_params["af"] = rdf._JimenezFortezaRemnantSpin(nu, X1, X2, chi1, chi2)
        self.rd_params["Mbhf"] = rdf._JimenezFortezaRemnantMass(
            nu, X1, X2, chi1, chi2, 1.0
        )
        return

    def _eval_fits_qnm(self, af, ell, emm):
        # QNM-related
        self.rd_params[(ell, emm)]["omega1"] = rdf._omega1(af, ell, emm)
        self.rd_params[(ell, emm)]["alpha21"] = rdf._alpha21(af, ell, emm)
        self.rd_params[(ell, emm)]["alpha1"] = rdf._alpha1(af, ell, emm)
        pass

    def _noncircular_corrections(
        self, modes, kind="non-spinning-equal-mass", databases=["RIT", "SXS", "ET"]
    ):
        # Correct fits to account for non-quasicircularity
        # From Carullo+23, https://arxiv.org/pdf/2309.07228.pdf
        # Assumes that self.params contains Heff_til, b_massless_EOB, Jmrg_til, chieff
        # as defined in the paper above

        for key in ["chieff", "Heff_til", "b_massless_EOB", "Jmrg_til"]:
            assert key in self.params.keys(), f"{key} not found in input parameters"

        convert = {"Mbhf": "Mf", "af": "af", "Apk": "A_peak22", "Omgpk": "omega_peak22"}

        for mode in modes:
            if mode != (2, 2):
                continue
            for key in convert.keys():

                # find the independent variables necessary for the fit
                gc_fits_param = convert[key]
                _, fqs_string = ncf.select_fitting_quantities(kind, gc_fits_param)
                fqs = fqs_string.split("-")
                fqs_dict = {k: np.array([self.params[k]]) for k in fqs}

                # compute correction
                res = ncf.eval_fit(
                    gc_fits_param, fqs_dict, fqs_string, kind, databases=databases
                )
                if key in ["Mbhf", "af"]:
                    self.rd_params[key + "_qc"] = self.rd_params[key]
                    self.rd_params[key] *= res
                    self.rd_params[key + "_nc"] = self.rd_params[key]
                else:
                    # save vals explicitly, so that they can be accessed even if overwritten
                    self.rd_params[mode][key + "_qc"] = self.rd_params[mode][key]
                    self.rd_params[mode][key] *= res
                    self.rd_params[mode][key + "_nc"] = self.rd_params[mode][key]
        pass

    def _overwrite_rdpars(self, input_par):
        # overwrite the ringdown parameters with user-provided ones
        # if they exist
        for key in ["Mbhf", "af"]:
            if key in input_par.keys():
                self.rd_params[key] = input_par[key]

        for mode in self.modes:
            if mode in input_par.keys():
                for key in self.rd_params[mode].keys():
                    if key in input_par[mode].keys():
                        self.rd_params[mode][key] = input_par[mode][key]
        pass

    def compute_modes(self, modes, t_ref, t):
        """
        Assume that the max of 22 is at t_ref (contained in t)
        """

        Mbhf = self.rd_params["Mbhf"]

        for mode in modes:
            self.hlm[mode] = {}
            for ky in ["A", "phi"]:
                self.hlm[mode][ky] = np.zeros_like(t)

            this_dt = self.rd_params[mode]["DeltaT"]
            start_time = (t_ref + this_dt) / Mbhf
            mask = t > start_time
            x = (t[mask] - start_time) / Mbhf
            A, p = self._rd_template(x, mode[0], mode[1])
            self.hlm[mode]["A"][mask] = A
            self.hlm[mode]["phi"][mask] = p

        pass


if __name__ == "__main__":

    # test
    params = {"q": 1.4, "chiA": [0.0, 0.0, 0.0], "chiB": [0.0, 0.0, 0.0]}
    t = np.linspace(-100, 400, 10000)

    RD = TEOBRingdown(t, parameters=params, use_fits=True, modes=[(2, 1), (2, 2)])
    logging.info(RD.rd_params)

    import matplotlib.pyplot as plt

    plt.plot(t, RD.hlm[(2, 2)]["A"] * np.cos(RD.hlm[(2, 2)]["phi"]))
    plt.plot(t, RD.hlm[(2, 1)]["A"] * np.cos(RD.hlm[(2, 1)]["phi"]))
    plt.show()
    pass
