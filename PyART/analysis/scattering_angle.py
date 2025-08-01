import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
from ..utils import utils
from .hypfit import fit_quadratic, quadratic_to_canonical, plot_hypfit

matplotlib.rc("text", usetex=True)


class ScatteringAngle:
    """
    Class to compute the scattering of a NR scattering simulation.
    The error associated to the angle is linked to the uncertanties
    of the extrapolation at infinity
    """

    def __init__(self, **kwargs):

        self.puncts = None
        self.use_single_punct = None  # None, 0, or 1
        self.verbose = True

        # options used for u-poly extrapolation
        self.nmin = 2
        self.nmax = 10
        self.n_extract = None
        self.r_cutoff_in_low = 25
        self.r_cutoff_in_high = 80
        self.r_cutoff_out_low = 25
        self.r_cutoff_out_high = None

        # use hypfit instead
        self.hypfit = False

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown option: {key}")

        if self.n_extract is None:
            self.n_extract = self.nmax  # safe if SVD is used

        nmin = self.nmin
        nmax = self.nmax
        n_extract = self.n_extract
        self.nfits = nmax - nmin + 1

        self.to_commonformat()
        self.compute_chi()
        pass

    def get_xy(self):
        if self.use_single_punct is None:
            x = self.x1 - self.x2
            y = self.y1 - self.y2
        elif self.use_single_punct == 0:
            x = self.x1
            y = self.y1
        elif self.use_single_punct == 1:
            x = self.x2
            y = self.y2
        else:
            raise ValueError(
                f"Uknown value for use_single_punct: {self.use_single_punct}"
            )
        return x, y

    def to_commonformat(self):
        nr_keys = ["t", "x0", "y0", "x1", "y1"]
        eob_keys = ["t", "r", "phi"]

        if all(key in self.puncts for key in nr_keys):
            t = self.puncts["t"]
            self.x1 = self.puncts["x0"]
            self.y1 = self.puncts["y0"]
            self.x2 = self.puncts["x1"]
            self.y2 = self.puncts["y1"]
            x, y = self.get_xy()
            r = np.sqrt(x * x + y * y)
            th = np.unwrap(np.arctan(y / x) * 2) / 2

        elif all(key in self.puncts for key in eob_keys):
            self.x1 = None
            self.y1 = None
            self.x2 = None
            self.y2 = None
            t = self.puncts["t"]
            r = self.puncts["r"]
            th = self.puncts["phi"]
            x = r * np.cos(th)
            y = r * np.sin(th)

        else:
            raise RuntimeError(
                "Invalid keys in puncts. Use t,r,phi for EOB, or t,x0,y0,x1,y1 for NR."
            )

        self.t = t
        self.x = x
        self.y = y
        self.r = r
        self.th = th
        return t, x, y, r, th

    def compute_chi(self, verbose=None):
        if verbose is None:
            verbose = self.verbose

        nmin = self.nmin
        nmax = self.nmax
        n_extract = self.n_extract

        t = self.t
        r = self.r
        th = self.th

        fit_in = utils.upoly_fits(
            r,
            th,
            nmin=nmin,
            nmax=nmax,
            n_extract=n_extract,
            direction="in",
            r_cutoff_low=self.r_cutoff_in_low,
            r_cutoff_high=self.r_cutoff_in_high,
        )
        b_in = fit_in["coeffs"]
        mask_in = fit_in["mask"]

        fit_out = utils.upoly_fits(
            r,
            th,
            nmin=nmin,
            nmax=nmax,
            n_extract=n_extract,
            direction="out",
            r_cutoff_low=self.r_cutoff_out_low,
            r_cutoff_high=self.r_cutoff_out_high,
        )
        b_out = fit_out["coeffs"]
        mask_out = fit_out["mask"]

        self.fit_orders = fit_in["fit_orders"]
        self.chi_array = np.zeros((nmax - nmin + 1,))

        th_inf_in_vec = np.zeros_like(self.chi_array)
        th_inf_out_vec = np.zeros_like(self.chi_array)
        for n in self.fit_orders:
            chi_tmp, th_inf_in_tmp, th_inf_out_tmp = self.compute_chi_from_fit(
                b_in, b_out, n
            )
            self.chi_array[n - nmin] = chi_tmp
            th_inf_in_vec[n - nmin] = th_inf_in_tmp
            th_inf_out_vec[n - nmin] = th_inf_out_tmp
            # if n_extract is not None and n==n_extract:
            if n == n_extract:
                chi = chi_tmp
                th_inf_in = th_inf_in_tmp
                th_inf_out = th_inf_out_tmp
        # if n_extract is None:
        # chi        = np.mean(self.chi_array)
        # th_inf_in  = np.mean(th_inf_in_vec)
        # th_inf_out = np.mean(th_inf_out_vec)
        fit_err_in = (max(b_in[-1, :]) - min(b_in[-1, :])) * 180 / np.pi
        fit_err_out = (max(b_out[-1, :]) - min(b_out[-1, :])) * 180 / np.pi
        fit_err = np.sqrt(fit_err_in**2 + fit_err_out**2)

        if verbose:
            print("fit-orders       : {:d} - {:d}".format(nmin, nmax))
            print(
                "r in  fit        : [{:.2f}, {:.2f}]".format(
                    self.r_cutoff_in_low, self.r_cutoff_in_high
                )
            )
            if self.r_cutoff_out_high is None:
                r_out_fit = r[-1]
            else:
                r_out_fit = self.r_cutoff_out_high
            print(
                "r out fit        : [{:.2f}, {:.2f}]".format(
                    self.r_cutoff_out_low, r_out_fit
                )
            )
            print("theta inf in     : {:8.4f} +- {:6.4f}".format(th_inf_in, fit_err_in))
            print(
                "theta inf out    : {:8.4f} +- {:6.4f}".format(th_inf_out, fit_err_out)
            )
            print("scattering angle : {:8.4f} +- {:6.4f}".format(chi, fit_err))

        self.t_in = t[mask_in]
        self.r_in = r[mask_in]
        self.u_in = 1 / self.r_in
        self.th_in = th[mask_in]

        self.t_out = t[mask_out]
        self.r_out = r[mask_out]
        self.u_out = 1 / self.r_out
        self.th_out = th[mask_out]

        if self.hypfit:
            self.chi_hypfit, _ = self.compute_chi_hypfit()
        else:
            self.chi_hypfit = None

        min_r = min(self.r)
        if self.r_cutoff_in_low < min_r:
            print(
                "+++ Warning +++\nmin(r)={:.2f}>r_cutoff_in_low={:.2f}".format(
                    min_r, self.r_cutoff_in_low
                )
            )
        if self.r_cutoff_out_low < min_r:
            print(
                "+++ Warning +++\nmin(r)={:.2f}>r_cutoff_out_low={:.2f}".format(
                    min_r, self.r_cutoff_out_low
                )
            )

        self.p_in = fit_in["polynomials"]
        self.b_in = b_in
        self.p_out = fit_out["polynomials"]
        self.b_out = b_out

        self.chi = chi
        self.fit_err = fit_err

        return

    def compute_chi_from_fit(self, b_in, b_out, n):
        th_inf_in = b_in[-1, n - self.nmin] / np.pi * 180
        th_inf_out = b_out[-1, n - self.nmin] / np.pi * 180
        chi = th_inf_out - th_inf_in - 180
        return chi, th_inf_in, th_inf_out

    def compute_chi_hypfit(self, verbose=None):
        if verbose is None:
            verbose = self.verbose
        angles = np.zeros((2, 2))
        th_start = None
        th_end = None
        fits = {}
        for i in range(2):
            if i == 0:
                th = self.th_in
                r = self.r_in
                th_start = th[0]
                fit_key = "in"
            else:
                th = self.th_out
                r = self.r_out
                th_end = th[-1]
                fit_key = "out"
            x = r * np.cos(th)
            y = r * np.sin(th)
            ABCDF = fit_quadratic(x, y)
            canonical = quadratic_to_canonical(ABCDF)

            fits[fit_key] = {
                "x": x,
                "y": y,
                "r": r,
                "th": th,
                "canonical": canonical,
                "ABCDF": ABCDF,
            }

            A = ABCDF[0]
            B = ABCDF[1]
            C = ABCDF[2]
            sqrt_delta = np.sqrt(B * B - A * C)
            m1 = A / (-B + sqrt_delta)  # angular coeff of asympt
            m2 = A / (-B - sqrt_delta)
            angles[i, :] = np.arctan(np.array([m1, m2])) / 2 / np.pi * 360
        chi_deg = angles[1, 0] - angles[0, 1]
        chi_rad = chi_deg / 180 * np.pi

        n_pi = np.floor((th_end - th_start - chi_rad) / np.pi)
        chi_deg += 180 * n_pi
        if verbose:
            print("chi hypfit       : {:.4f}".format(chi_deg))
        return chi_deg, fits

    def plot_hypfit(self, swap_ab_list=[True, True]):
        _, fits = self.compute_chi_hypfit()
        for i, key in enumerate(fits):
            fit = fits[key]
            plot_hypfit(
                fit["x"],
                fit["y"],
                fit["canonical"],
                swap_ab=swap_ab_list[i],
                rlim=max(fit["r"]),
            )
        pass

    def save_plot(self, show=True, save=False, figname="plot.png"):
        if save:
            plt.savefig(figname, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()
        return

    def plot_summary(self, show=True, save=False, figname=None):
        t = self.t
        r = self.r
        x = self.x
        y = self.y
        th = self.th
        fig, axs = plt.subplots(2, 2, figsize=(9, 7))
        if "x1" in self.puncts:
            axs[0, 0].plot(self.x1, self.y1)
            axs[0, 0].plot(self.x2, self.y2)
            axs[0, 0].set_ylabel(r"$y$", fontsize=15)
            axs[0, 0].set_xlabel(r"$x$", fontsize=15)
        axs[0, 1].plot(x, y)
        axs[1, 0].plot(t, r, c="k")
        axs[1, 0].plot(self.t_in, self.r_in, c=[0, 1, 1], linestyle=":", lw=2.0)
        axs[1, 0].plot(self.t_out, self.r_out, c=[0, 1, 0], linestyle=":", lw=2.0)
        axs[1, 0].hlines(
            self.r_cutoff_in_low, t[0], t[-1], color="r", linestyle="-", lw=0.8
        )
        axs[1, 0].hlines(
            self.r_cutoff_in_high, t[0], t[-1], color="m", linestyle="-", lw=0.8
        )
        axs[1, 0].hlines(
            self.r_cutoff_out_low, t[0], t[-1], color="b", linestyle="--", lw=0.8
        )
        axs[1, 0].hlines(
            self.r_cutoff_out_high, t[0], t[-1], color="g", linestyle="--", lw=0.8
        )
        axs[1, 1].plot(t, th, "k")
        axs[1, 1].plot(self.t_in, self.th_in, c=[0, 1, 1], linestyle=":", lw=2.0)
        axs[1, 1].plot(self.t_out, self.th_out, c=[0, 1, 0], linestyle=":", lw=2.0)

        axs[0, 1].set_xlabel(r"$x_+ - x_-$", fontsize=15)
        axs[0, 1].set_ylabel(r"$y_+ - y_-$", fontsize=15)
        axs[1, 0].set_xlabel(r"$t$", fontsize=15)
        axs[1, 0].set_ylabel(r"$r$", fontsize=15)
        axs[1, 1].set_xlabel(r"$t$", fontsize=15)
        axs[1, 1].set_ylabel(r"$\theta$", fontsize=15)
        if figname is None:
            figname = "plot_summary.png"
        self.save_plot(show=show, save=save, figname=figname)
        return

    def plot_fit_diffs(self, xvar="r", show=True, save=False, figname=None):
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, self.nfits))
        _, axs = plt.subplots(2, 2, figsize=(12, 8))
        for i in range(2):
            if i == 0:
                lab = "in"
                u = self.u_in
                r = self.r_in
                p = self.p_in
                th = self.th_in
            else:
                lab = "out"
                u = self.u_out
                r = self.r_out
                p = self.p_out
                th = self.th_out

            if xvar == "r":
                x = r
                xlabel = r"$r$"
            elif xvar == "u":
                x = u
                xlabel = r"$u$"
            else:
                raise RuntimeError("xvar={:s} is not a valid option".format(xvar))

            axs[0, i].plot(x, th, "k", label="track")
            for j in range(0, self.nfits):
                if j % 2 == 0:
                    linestyle = "--"
                else:
                    linestyle = ":"
                axs[0, i].plot(
                    x,
                    p[:, j],
                    color=colors[j],
                    linestyle=linestyle,
                    label="fit n" + str(self.nmin + j),
                )
                axs[1, i].plot(x, np.abs(th - p[:, j]), color=colors[j])
                axs[1, i].set_yscale("log")
            axs[0, i].legend()
            axs[0, i].set_title("fit-" + lab)
            axs[0, i].set_xlim([x[0], x[-1]])
            axs[1, i].set_xlim([x[0], x[-1]])
            axs[1, i].set_xlabel(xlabel, fontsize=18)
            axs[0, i].set_ylabel(
                r"$\theta_{\rm LAB }$".replace("LAB", lab), fontsize=18
            )
            axs[1, i].set_ylabel(
                r"$|\theta_{\rm LAB }-\theta^{\rm fit}_{\rm LAB }|$".replace(
                    "LAB", lab
                ),
                fontsize=18,
            )
            axs[1, i].grid()
        if figname is None:
            figname = "plot_diffs_" + xvar + ".png"
        self.save_plot(show=show, save=save, figname=figname)
        return

    def plot_fit_extrapolation(self, xvar="u", show=True, save=False, figname=False):
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, self.nfits))
        _, axs = plt.subplots(2, 1, figsize=(10, 8))
        u_extr = np.linspace(1e-4, 0.04, num=100)
        r_extr = 1 / u_extr
        if xvar == "r":
            x_extr = r_extr
            x_in = self.r_in
            x_out = self.r_out
            xlabel = r"$r$"
        elif xvar == "u":
            x_extr = u_extr
            x_in = self.u_in
            x_out = self.u_out
            xlabel = r"$u$"
        else:
            raise RuntimeError("xvar={:s} is not a valid option".format(xvar))
        axs[0].plot(x_in, self.th_in, "k", label="track", lw=2)
        axs[1].plot(x_out, self.th_out, "k", label="track", lw=2)
        for i in range(0, self.nfits):
            p_in = np.polyval(self.b_in[:, i], u_extr)
            p_out = np.polyval(self.b_out[:, i], u_extr)
            axs[0].plot(
                x_extr, p_in, label="fit n" + str(self.nmin + i), lw=1, color=colors[i]
            )
            axs[1].plot(
                x_extr, p_out, label="fit n" + str(self.nmin + i), lw=1, color=colors[i]
            )
        axs[0].legend()
        axs[0].set_ylabel("p-in")
        axs[0].set_xlabel(xlabel)
        axs[1].legend()
        axs[1].set_ylabel("p-out")
        axs[1].set_xlabel(xlabel)
        if figname is None:
            figname = "plot_fit_extrapolation_" + xvar + ".png"
        self.save_plot(show=show, save=save, figname=figname)
        return

    def plot_fit_chi(self, show=True, save=False, figname=False):
        _, ax = plt.subplots(1, 1, figsize=(10, 8))
        plt.scatter(self.fit_orders, self.chi_array)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
        plt.ylabel(r"$\chi$", fontsize=18)
        plt.xlabel("poly order", fontsize=18)
        plt.grid()
        if figname is None:
            figname = "plot_fit_chi.png"
        self.save_plot(show=show, save=save, figname=figname)
        return


def ComputeChiFrom2Sims(
    path_hres=None,
    path_lres=None,
    puncts_hres=None,
    puncts_lres=None,
    verbose=False,
    vverbose=False,
    **kwargs,
):
    # NOTE:
    print("Warning! ComputeChiFrom2Sims: old code, not properly tested!")

    if vverbose:
        verbose = True
    scat_lres = ScatteringAngle(
        path=path_lres, puncts=puncts_lres, verbose=vverbose, **kwargs
    )
    scat_hres = ScatteringAngle(
        path=path_hres, puncts=puncts_hres, verbose=vverbose, **kwargs
    )

    chi_hres = scat_hres.chi
    fit_err_hres = scat_hres.fit_err
    chi_lres = scat_lres.chi
    fit_err_lres = scat_lres.fit_err

    chi = chi_hres
    # fit_err = np.sqrt(fit_err_hres**2 + fit_err_lres**2)
    fit_err = fit_err_hres
    res_err = np.abs(chi_hres - chi_lres)
    err = np.sqrt(fit_err**2 + res_err**2)

    if verbose:
        print("fit-orders       : {:d} - {:d}".format(scat_hres.nmin, scat_hres.nmax))
        print("fit error        : {:6.4f}".format(fit_err))
        print("resolution error : {:6.4f}".format(res_err))
        print("scattering angle : {:8.4f} +- {:6.4f}\n".format(chi, err))

    out = {}
    out["scat_lres"] = scat_lres
    out["scat_hres"] = scat_hres
    out["chi"] = chi
    out["err"] = err
    out["fit_err"] = fit_err
    out["res_err"] = res_err
    return out
