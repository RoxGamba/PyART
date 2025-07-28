#!/usr/bin/python
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from scipy import integrate

from ..utils.utils import D1, safe_sigmoid


class IntegrateMultipole(object):
    """
    Class for integrating multipole
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
        r = self.radius
        M = self.mass
        return r * (1 + M / (2 * r)) ** 2

    def retarded_time(self):
        if self.radius <= 0.0:
            return self.t
        M = self.mass
        R = self.areal_radius()
        rstar = R + 2 * M * np.log(R / (2 * M) - 1)
        return self.t - rstar

    def patch_psi4(self, psi4, t0, t1, t2, debug=False):
        """
        Patch psi4 before t1 and after t2 using
        power-laws of (t-t0).
        To be tested on high energy simulations
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
        Remove drift in TD integration using a fit.
        If poly_int is specified, then fit only that part of the signal.
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
        """
        integrals = []
        f = signal
        for i in range(steps):
            integral = integrate.cumulative_trapezoid(f, self.t, initial=0)
            f = self.remove_time_drift(integral)
            integrals.append(f)
        return integrals

    def integrate(self, signal, steps=1):
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
