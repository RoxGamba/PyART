#!/usr/bin/python
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from scipy import integrate

from ..utils.utils import D1, safe_sigmoid


class IntegrateMultipole(object):
    """
    Class for integrating gravitational wave multipole data using either
    fixed-frequency (FFI) or time-domain (TDI) integration methods.
    Parameters
    ----------
    l : int
        Spherical harmonic index l.
    m : int
        Spherical harmonic index m.
    t : array_like
        Time array.
    data : array_like
        Input data array (psi4 or news).
    mass : float
        Mass parameter (typically ADM mass).
    radius : float
        Extraction radius.
    integrand : str, optional
        Type of integrand ("psi4" or "news"), default "psi4".
    method : str, optional
        Integration method ("FFI" or "TDI"), default "FFI".
    f0 : float, optional
        Low-frequency cutoff for FFI, default 0.001.
    deg : int, optional
        Degree of polynomial for drift removal in TDI, default 0.
    poly_int : tuple or None, optional
        Interval for polynomial drift removal, default None.
    extrap_psi4 : bool, optional
        If True, apply extrapolation to psi4, default False.
    window : list or None, optional
        Windowing parameters [t1, t2], default None.
    patch_opts : dict, optional
        Options for patching psi4, default {}.
    walpha : float, optional
        Window sharpness parameter, default 3.
    Attributes
    ----------
    l, m, mass, radius, integrand, method, f0, deg, poly_int, extrap_psi4,
    window, walpha, patch_opts : see Parameters
    t : array_like
        Time array.
    psi4 : array_like
        Psi4 data (after windowing/patching/extrapolation if applicable).
    dh : array_like
        First integral of psi4 (news).
    h : array_like
        Second integral of psi4 (strain).
    u : array_like
        Retarded time.
    integr_opts : dict
        Dictionary of integration options.
    Methods
    -------
    areal_radius()
        Compute areal radius from extraction radius and mass.
    retarded_time()
        Compute retarded time using areal radius and mass.
    patch_psi4(psi4, t0, t1, t2, debug=False)
        Patch psi4 before t1 and after t2 using power-laws of (t-t0).
    extrapolate_psi4()
        Apply extrapolation to psi4 using mass and radius.
    apply_window(signal, window=[10, -10], walpha=3)
        Apply window function to signal.
    freq_interval(signal)
        Compute frequency interval for FFI.
    fixed_freq_int(signal, steps=1)
        Perform fixed-frequency integration.
    remove_time_drift(signal)
        Remove polynomial drift from time-domain integration.
    time_domain_int(signal, steps=1)
        Perform time-domain integration with polynomial correction.
    integrate(signal, steps=1)
        Integrate signal using selected method (FFI or TDI).
    """

    def __init__(
        self,
        l,
        m,
        t,
        data,
        mass,
        radius,
        integrand="psi4",
        method="FFI",
        f0=0.001,
        deg=0,
        poly_int=None,
        extrap_psi4=False,
        window=None,
        patch_opts={},
        walpha=3,
    ):
        """
        Initialize the wave integration analysis.
        Parameters
        ----------
        l : int
            Spherical harmonic index l.
        m : int
            Spherical harmonic index m.
        t : array_like
            Time array.
        data : array_like
            Input data array, typically psi4 or news.
        mass : float
            Mass parameter for the system.
        radius : float
            Extraction radius.
        integrand : str, optional
            Type of integrand to use ('psi4' or 'news'). Default is 'psi4'.
        method : str, optional
            Integration method to use. Default is 'FFI'.
        f0 : float, optional
            Low-frequency cutoff for integration. Default is 0.001.
        deg : int, optional
            Degree for polynomial integration. Default is 0.
        poly_int : object, optional
            Polynomial integration object. Default is None.
        extrap_psi4 : bool, optional
            If True, extrapolate psi4 data. Default is False.
        window : str or None, optional
            Window function to apply to data. Default is None.
        patch_opts : dict, optional
            Options for patching psi4 data. Default is empty dict.
        walpha : float, optional
            Alpha parameter for window function. Default is 3.
        Raises
        ------
        RuntimeError
            If an unknown integrand is specified.
        Notes
        -----
        Stores integration options in `self.integr_opts`.
        """

        self.l = l
        self.m = m
        self.mass = mass
        self.radius = radius
        self.integrand = integrand
        self.method = method
        self.f0 = f0
        if f0 is not None:
            self.fcut = 2 * self.f0 / max(1, abs(self.m))
        else:
            self.fcut = None
        self.deg = deg
        self.poly_int = poly_int
        self.extrap_psi4 = extrap_psi4
        self.window = window
        self.walpha = walpha
        self.patch_opts = patch_opts

        self.t = t

        if self.patch_opts:
            data = self.patch_psi4(data, **patch_opts)

        if self.window is not None:
            data = self.apply_window(data, window=window, walpha=walpha)

        if self.integrand == "psi4":
            self.psi4 = radius * data
            if self.extrap_psi4:
                self.extrapolate_psi4()
            self.dh, self.h = self.integrate(self.psi4, steps=2)

        elif self.integrand == "news":
            self.dh = radius * data
            self.psi4 = D1(self.dh, self.t, 4)
            self.h = self.integrate(self.dh, steps=1)

        else:
            raise RuntimeError(
                f"Unknown integrand: {self.integrand}, use 'psi4' or 'news'"
            )

        self.u = self.retarded_time()

        # wrap integration options in a dict and store in attribute
        tostore = [
            "integrand",
            "method",
            "extrap_psi4",
            "f0",
            "deg",
            "poly_int",
            "window",
            "walpha",
        ]
        self.integr_opts = {}
        for k in tostore:
            self.integr_opts[k] = getattr(self, k)
        pass

    def areal_radius(self):
        """
        Calculate the areal radius of the object.
        The areal radius is defined as:
            r_areal = r * (1 + M / (2 * r)) ** 2
        where `r` is the radius and `M` is the mass.
        Returns
        -------
        float
            The areal radius.
        """
        r = self.radius
        M = self.mass
        return r * (1 + M / (2 * r)) ** 2

    def retarded_time(self):
        """
        Computes the retarded time for the current object.
        If the radius is non-positive,
        returns the current time `t` directly.
        Returns
        -------
        float
            The retarded time value.
        """

        if self.radius <= 0.0:
            return self.t
        M = self.mass
        R = self.areal_radius()
        rstar = R + 2 * M * np.log(R / (2 * M) - 1)
        return self.t - rstar

    def patch_psi4(self, psi4, t0, t1, t2, debug=False):
        """
        Patch psi4 before t1 and after t2 using power-laws of (t-t0).
        To be tested on high energy simulations.

        Parameters
        ----------
        psi4 : array_like
            Complex waveform data to be patched.
        t0 : float
            Reference time for the power-law extrapolation.
        t1 : float
            Time before which the waveform is patched using a (t-t0)^-3 and (t-t0)^-4 power-law.
        t2 : float
            Time after which the waveform is patched using a (t-t0)^-3 power-law.
        debug : bool, optional
            If True, plot the patched waveform and intermediate results for debugging.
        Returns
        -------
        new_psi4 : ndarray
            The patched complex waveform.
        """

        def return_patch34(t, patch):
            W0 = patch["W0"]
            W1 = patch["W1"]
            dT = patch["dT"]
            t0 = patch["t0"]
            c4 = -3 * dT**4 * (W0 + dT / 3 * W1)
            c3 = dT**3 * W0 - c4 / dT
            y = c3 / (t - t0) ** 3 + c4 / (t - t0) ** 4
            return y

        def return_patch3(t, patch):
            c3 = patch["dT"] ** 3 * patch["W0"]
            y = c3 / (t - patch["t0"]) ** 3
            return y

        t = self.t

        # update t1 to be a grid point
        idx1 = np.where(t > t1)[0][0]
        t1 = t[idx1]
        # same for t2
        idx2 = np.where(t > t2)[0][0]
        t2 = t[idx2]

        A = np.abs(psi4)
        dA = D1(A, t, 4)

        patch1 = {"W0": A[idx1], "W1": dA[idx1], "dT": t1 - t0, "t0": t0}

        x1 = np.linspace(0, t1, num=idx1 + 1)
        y1 = return_patch34(x1, patch1)

        patch2 = {"W0": A[idx2], "W1": dA[idx2], "dT": t2 - t0, "t0": t0}
        x2 = np.linspace(t2, t[-1], num=len(t) - idx2)
        y2 = return_patch3(x2, patch2)

        new_psi4_amp = np.concatenate((y1, A[idx1 + 1 : idx2], y2))
        phase = -np.unwrap(np.angle(psi4))
        new_psi4 = new_psi4_amp * np.exp(-1j * phase)

        # x1_ext = np.linspace(-200, t1, num=1000)
        # y1_ext = return_patch34(x1_ext, patch1)
        # x1_inf = -1e+10
        # y1_inf = return_patch34(x1_inf, patch1)
        # print(y1_inf)

        if debug:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.subplot(2, 2, 1)
            plt.plot(t, np.abs(psi4))
            plt.plot(x1, y1, c=[0, 0.8, 0])
            plt.plot(x2, y2, c=[0.8, 0, 0])
            plt.plot(t, new_psi4_amp, ls="--")
            plt.title(f"{self.l:d},{self.m:d}")
            plt.subplot(2, 2, 2)
            plt.plot(t, phase)
            plt.subplot(2, 2, 3)
            plt.plot(t, psi4.real)
            plt.plot(t, new_psi4.real, "--")
            plt.subplot(2, 2, 4)
            plt.plot(t, psi4.imag)
            plt.plot(t, new_psi4.imag, "--")
            plt.show()

        return new_psi4

    def extrapolate_psi4(self):
        """
        Extrapolates the psi4 waveform to infinity using the formula
        from https://arxiv.org/pdf/1008.4360
        The extrapolated psi4 is stored in self.psi4.
        Returns
        -------
        None
        """

        r = self.radius
        M = self.mass
        R = self.areal_radius()

        psi0 = self.psi4 / r * R  # self.psi4 = radius * data, see __init__
        dh0 = self.integrate(psi0)

        l = self.l
        A = 1 - 2 * M / R
        self.psi4 = A * (psi0 - (l - 1) * (l + 2) * dh0 / (2 * R))
        return

    def apply_window(self, signal, window=[10, -10], walpha=3):
        """
        Applies a smooth window to the input signal using sigmoid functions.
        Parameters
        ----------
        signal : np.ndarray
            The input signal array to be windowed.
        window : list or tuple, optional
            Window specification as [start, end]. The start and end values define the window region.
            Default is [10, -10].
        walpha : float, optional
            The sharpness parameter for the sigmoid window edges. Default is 3.
        Returns
        -------
        np.ndarray
            The windowed signal.
        Raises
        ------
        RuntimeError
            If the window specification is invalid.
        Notes
        -----
        The window is applied smoothly using sigmoid functions to avoid sharp edges.
        """

        t = self.t
        # apply window
        clip_val = np.log(1e20)
        # FIXME : this if-statement is error-prone, to improve
        if window[0] >= 0 and window[1] <= 0:
            w_t1 = window[0]
            w_t2 = t[-1] + window[1]
            sig1 = safe_sigmoid(t - w_t1, alpha=walpha, clip=clip_val)
            sig2 = safe_sigmoid(w_t2 - t, alpha=walpha, clip=clip_val)
            signal *= sig1
            signal *= sig2
        elif window[1] > window[0]:
            sig = safe_sigmoid(
                window[0] - t, alpha=walpha, clip=clip_val
            ) + safe_sigmoid(t - window[1], alpha=walpha, clip=clip_val)
            signal *= sig
        else:
            raise RuntimeError("Invalid window option:: [{:f} {:f}]".format(*window))

        return signal

    def freq_interval(self, signal):
        """
        Computes the frequency array for the given signal, limiting frequencies to the cutoff value.
        Parameters
        ----------
        signal : ndarray
            Input signal array for which the frequency interval is computed.
        Returns
        -------
        f : ndarray
            Frequency array with values limited to +/- self.fcut.
        """

        dt = np.diff(self.t)[0]
        f = fftfreq(signal.shape[0], dt)
        idx_p = np.logical_and(f >= 0, f < self.fcut)
        idx_m = np.logical_and(f < 0, f > -self.fcut)
        f[idx_p] = self.fcut
        f[idx_m] = -self.fcut
        return f

    def fixed_freq_int(self, signal, steps=1):
        """
        Fixed frequency integration
        steps is the number of integrations performed
        Parameters
        ----------
        signal : array_like
            Input signal to be integrated.
        steps : int, optional
            Number of integration steps to perform (default is 1).
        Returns
        -------
        integrals : list of ndarray
            List containing the integrated signal(s) after each step.
        """
        f = self.freq_interval(signal)
        factor = -1j / (2 * np.pi * f)
        fft_signal = fft(signal)
        cumulative = fft_signal
        integrals = []
        for i in range(steps):
            cumulative *= factor
            integrals.append(ifft(cumulative))

        return integrals

    def remove_time_drift(self, signal):
        """
        Removes polynomial time drift from the input signal.
        Fits a polynomial of degree `self.deg` to the signal over the specified interval
        (`self.poly_int`) or the entire time array (`self.t`) if no interval is set.
        Subtracts the fitted polynomial from the signal to remove drift.
        Parameters
        ----------
        signal : np.ndarray
            Input signal array to be corrected for time drift.
        Returns
        -------
        out : np.ndarray
            Signal with polynomial time drift removed.
        Raises
        ------
        RuntimeError
            If the polynomial interval ends after the simulation's end time.
        """

        out = signal
        if self.deg >= 0:
            if self.poly_int is None:
                t_tofit = self.t
                signal_tofit = signal
            else:
                if self.poly_int[1] > self.t[-1]:
                    raise RuntimeError(
                        "Polynomial interval ends after simulation's end (t : [{:.2f}, {:.2f}] M)".format(
                            self.t[0], self.t[-1]
                        )
                    )
                mask = np.logical_and(
                    self.t >= self.poly_int[0], self.t <= self.poly_int[1]
                )
                t_tofit = self.t[mask]
                signal_tofit = signal[mask]
            p = np.polyfit(t_tofit, signal_tofit, self.deg)
            out -= np.polyval(p, self.t)
        return out

    def time_domain_int(self, signal, steps=1):
        """
        Time domain integration with polynomial correction
        The polynomial is obtained fitting the whole signal if poly_int is none,
        otherwise consider only the interval specified; see remove_time_drift
        Parameters
        ----------
        signal : array_like
            The input signal to be integrated.
        steps : int, optional
            Number of integration steps to perform (default is 1).
        Returns
        -------
        integrals : list of ndarray
            List containing the integrated signal(s) after polynomial drift correction
            for each step.
        """
        integrals = []
        f = signal
        for i in range(steps):
            integral = integrate.cumulative_trapezoid(f, self.t, initial=0)
            f = self.remove_time_drift(integral)
            integrals.append(f)
        return integrals

    def integrate(self, signal, steps=1):
        """
        Integrate the input signal using the specified method (FFI or TDI).
        Parameters
        ----------
        signal : array_like
            The input signal to be integrated.
        steps : int, optional
            Number of integration steps to perform (default is 1).
        Returns
        -------
        ndarray or list of ndarray
            The integrated signal if `steps` is 1, otherwise a list of integrated signals.
        Raises
        ------
        RuntimeError
            If an unknown integration method is specified.
        """
        if self.method == "FFI":
            int_list = self.fixed_freq_int(signal, steps=steps)
        elif self.method == "TDI":
            int_list = self.time_domain_int(signal, steps=steps)
        else:
            raise RuntimeError("Unknown method: {:s}".format(self.method))
        if steps == 1:
            return int_list[0]
        else:
            return int_list
