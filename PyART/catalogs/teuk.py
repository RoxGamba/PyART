import logging
import numpy as np
import scipy as sp
import os
import json
from glob import glob
from ..waveform import Waveform
from ..utils.wf_utils import get_multipole_dict
from ..analytic import pn_horizon_fluxes as pn


class Waveform_Teuk(Waveform):
    """
    Class to handle Teukode test-mass waveforms.
    Maintain "old" (deprecated) methods, structure from the
    horizon flux project to handle various m values and resolutions
    """

    def __init__(
        self, path, ellmax=5, datafmt="npz", input_meta=None, load=["metadata"]
    ):

        super().__init__()
        self.path = path
        self.ellmax = ellmax
        self._kind = "Teukode"
        self.domain = "Time"
        self.datafmt = datafmt
        self.input_meta = input_meta

        if "metadata" in load:
            self.load_metadata()

        try:
            self.__read_sims__()
        except FileNotFoundError:
            logging.warning("Either no simulations or old format")

        if "dynamics" in load:
            self.load_dynamics()
        if "hlm" in load:
            self.load_hlm()
        if "horizon" in load:
            self.load_horizon()

    def __read_sims__(self):
        """
        Find available simulations (resolution, cfl, m values) in tk folder
        Assume that structure of the folder is
        teuk_a0.XXXX_r0YYY_NXxNY_*mM, e.g.:
        teuk_a0.0000_r0100.000_3601x161_proc4x2_m2
        """
        self.sims = {}
        globstr = os.path.join(self.path, f"teuk_a{self.a:.4f}_r0*_*x*_m*/")
        sims = glob(globstr)
        if len(sims) == 0:
            raise FileNotFoundError(
                "Waveform_Teuk.__read_sims__(): no simulations found."
            )
        nx_max = 0
        for sim in sims:
            sub = sim[len(self.path) :]
            nx, ny = (int(sub.split("x")[kk].split("_")[kk - 1]) for kk in range(2))
            cfl = 2.0 if "cfl" not in sub else float(sub.split("cfl")[1].split("/")[0])
            m = int(sub.split("_m")[1].split("_")[0].split("/")[0])
            if (nx, ny, cfl) not in self.sims.keys():
                self.sims[(nx, ny, cfl)] = []
            self.sims[(nx, ny, cfl)].append(m)
            if nx > nx_max:
                nx_max = nx
        self.sims[(nx, ny, cfl)].sort()
        ny_max = max([sim[1] for sim in self.sims.keys() if sim[0] == nx_max])
        self.max_res = (nx_max, ny_max)

    def load_metadata(self):
        if self.input_meta is not None:
            mtdt, a, mu = self.__load_metadata_from_parfile(self.input_meta)
        else:
            mtdt, a, mu = self.__load_metadata_from_jsons()

        self.metadata = mtdt
        self.a = a
        self.mu = mu

    def __load_metadata_from_parfile(self, metadata_file):
        """
        Load metadata from the Teukode parfile.
        """
        # find the parfile
        parfile = os.path.join(self.path, metadata_file)
        with open(parfile[0], "r") as f:
            lines = [
                line.strip()
                for line in f
                if "=" in line and not line.lstrip().startswith("#")
            ]

        mtdt = {}
        for line in lines:
            k, v = line.split("=")
            k = k.replace(" ", "")
            try:
                v_converted = int(v)
            except ValueError:
                try:
                    v_converted = float(v)
                except ValueError:
                    v_converted = v
            mtdt[k] = v_converted

        # get also the spin of the BH
        a = mtdt["kerr_abh"]
        if "mu" in mtdt:
            mu = mtdt["mu"]
        else:
            mu = 1e-3  # fallback to default if not present
            logging.warning(
                "Parameter 'mu' not found in parfile; using default value mu=1e-3. Please check your metadata file."
            )
        return mtdt, a, mu

    def __load_metadata_from_jsons(self):
        """
        Load the metadata from two json files:
        `kos_pars.json` and `teuk_pars.json`
        """
        mtdt = {}
        for name in ["kos", "teuk"]:
            name_path = os.path.join(self.path, f"{name}_pars.json")
            with open(name_path, "r") as f:
                data = json.load(f)
            for k, v in data.items():
                mtdt[k] = v
            f.close()

        a = mtdt["a"]
        mu = mtdt["mu"]

        return mtdt, a, mu

    def __load_dynamics_npz(self):
        """
        Load the dynamics from npz output
        """
        dyn = {}
        if self.datafmt == "npz":
            fname_dyn = os.path.join(self.path, "teuk_dyn.npz")
            dyn_loaded = np.load(fname_dyn, allow_pickle=True)
            dyn = dyn_loaded["arr_0"].item()
        return dyn

    def __load_dynamics_mAll(self):
        subdirs = os.listdir(self.path)
        dyn = {}
        for sd in subdirs:
            if sd[:5] == "kerr_":
                full_dyn_path = os.path.join(self.path, sd)
                for qt in ["pr.dat", "pr_star.dat", "pph.dat", "H.dat"]:
                    qt_name = qt.split(".")[0]
                    this_file = np.loadtxt(os.path.join(full_dyn_path, qt))
                    dyn[qt_name] = this_file[:, 1]

                # trajectory
                traj = np.loadtxt(os.path.join(full_dyn_path, "traj.dat"))
                dyn = {
                    "t": traj[:, 0],
                    "r": traj[:, 1],
                    "ph": traj[:, 2],
                    "Omg": traj[:, 3],
                    "tp_dynamics": False,
                }
        return dyn

    def __load_dynamics_from_path(self):
        """
        Assume folder structured for the horizon flux project
        Load the particle dynamics
        """
        dyn = {}
        for var in ["r", "ph", "pr", "pph", "H", "prstar", "Omg"]:
            dynf = os.path.join(self.path, f"{var}.txt")
            try:
                t, dyn[var] = np.loadtxt(dynf, unpack=True)
            except ValueError:
                logging.warning(f"Error loading {var}.txt")
            if "t" not in dyn.keys():
                dyn["t"] = t
        return dyn

    def load_dynamics(self):
        """
        Load and set the dynamics based on the data format
        For now, support old methods as well, deprecate them in the
        future
        """
        if self.datafmt == "npz":
            dyn = self.__load_dynamics_npz()
        elif self.datafmt == "teuk_mAll":
            dyn = self.__load_dynamics_mAll()
        elif self.datafmt == "path":
            dyn = self.__load_dynamics_from_path()
        else:
            raise ValueError("Data format for dynamics unknown")
        self._dyn = dyn

    def __load_hlm_npz(self):
        """
        Load modes for npz format
        """
        fname_hlm = os.path.join(self.path, "teuk_hlm.npz")
        hlm_loaded = np.load(fname_hlm, allow_pickle=True)
        hlm_z = hlm_loaded["arr_0"].item()
        u = hlm_z["t"]
        del hlm_z["t"]
        return u, hlm_z

    def __load_hlm_mAll(self):
        """
        Load modes for mAll format
        """
        subdirs = os.listdir(self.path)
        hlm_z = {}
        for sd in subdirs:
            if sd[:9] == "teuk_HH10":
                mstr = sd.split("_")[-1]
                m = int(mstr.replace("m", ""))
                for ell in range(2, self.ellmax + 1):
                    if ell < m:
                        continue
                    hname = f"out0d/h_Yl{ell:d}m{m:d}_x10.0000.dat"
                    full_path_h = os.path.join(self.path, sd, hname)
                    X = np.loadtxt(full_path_h)
                    hlm_z[(ell, m)] = X[:, 1] + 1j * X[:, 2]
                    if ell == 2 and m == 2:
                        u = X[:, 0]
        return u, hlm_z

    def __load_hlm_path(self):
        """
        Load modes for path.
        Use the highest resolution by default
        """
        raise NotImplementedError("Loading hlm from path not implemented yet.")

    def load_hlm(self):
        # load data
        if self.datafmt == "npz":
            u, hlm_z = self.__load_hlm_npz()
        elif self.datafmt == "teuk_mAll":
            u, hlm_z = self.__load_hlm_mAll()
        elif self.datafmt == "path":
            u, hlm_z = self.__load_hlm_path()
        else:
            raise ValueError("Data format for hlm unknown")

        self._u = u
        # convert to PyART hlm-dict
        hlm = {}
        for lm in hlm_z:
            if lm[0] <= self.ellmax:
                hlm[lm] = get_multipole_dict(hlm_z[lm])
        self._hlm = hlm
        shift = self._compute_waveform_shift()
        self._u = self.u + shift

    def load_horizon(self, m=None, nx=None, ny=None, cfl=2.0):
        """
        Integrate dJ/dt and dm/dt.
        Resolution taken as highest available if not specified.
        Summing over all specified values, or all available if none in input.
        Skipping negative ms if the opposite is available.
        """
        if nx is None:
            nx, ny = self.max_res
            logging.warning(
                f"load_horizon: assuming highest resolution, (nx, ny) = ({nx}, {ny})"
            )
        if m is None:
            emm = self.sims[(nx, ny, cfl)]
        else:
            emm = [m]
        cfls = "" if cfl == 2.0 else f"_cfl{cfl:.2g}"

        res = {}
        for mv in emm:
            if mv < 0 and -mv in emm:
                continue
            data_dir = os.path.join(
                self.path,
                f"teuk_a{self.a:.4f}_r0{self.dyn['r'][0]:.3f}_{nx}x{ny}_proc4x2_m{mv}{cfls}/out1d",
            )

            # Multiply modes other than m = 0 by 2 to account for negative m values
            by2 = 1.0 if mv == 0 else 2.0

            for kind in ["E", "J"]:
                fname = f"d{kind}dt_hrz_td_lsum_m{mv}_Poisson.dat"
                if not os.path.exists(os.path.join(data_dir, fname)):
                    logging.warning(
                        f"load_horizon: no {kind} flux file found for m = {mv}, (nx, ny, cfl) = ({nx}, {ny}, {cfl})"
                    )
                    continue
                t, dy = np.loadtxt(os.path.join(data_dir, fname), unpack=True)
                if "t" not in res.keys():
                    res["t"] = t
                else:
                    if len(t) != len(res["t"]):
                        logging.warning(
                            f"load_horizon: time arrays do not match for m = {mv}, (nx, ny, cfl) = ({nx}, {ny}, {cfl}). Interpolating, but check consistency!"
                        )
                    dy = np.interp(res["t"], t, dy)
                    t = res["t"]
                y = sp.integrate.cumulative_trapezoid(dy, t, initial=0)
                if mv not in res.keys():
                    res[mv] = {}
                res[mv][kind] = {"dy": dy, "y": y}
                if "sum" not in res.keys():
                    res["sum"] = {}
                if kind not in res["sum"].keys():
                    res["sum"][kind] = {
                        key: res[mv][kind][key] * by2 for key in res[mv][kind].keys()
                    }
                else:
                    res["sum"][kind]["dy"] += res[mv][kind]["dy"] * by2
                    res["sum"][kind]["y"] += res[mv][kind]["y"] * by2

        chi = self.a + res["sum"]["J"]["y"] / (1 + res["sum"]["E"]["y"]) ** 2
        res["chi"] = chi

        self.hflx = res
        return res

    def _compute_waveform_shift(self):

        rBL = self.dyn["r"][0]
        tBL = self.dyn["t"][0]

        try:
            M = self.metadata["kerr_mbh"]
        except KeyError:
            M = 1.0
        try:
            S = self.metadata["grid_xmax"]
        except KeyError:
            S = 10.0
        a = self.a

        if abs(M - a) < 1e-13:
            rs = rBL - (2 * M * (M + (-rBL + M) * np.log(rBL - M))) / (rBL - M)
        else:
            sqrtma = np.sqrt(abs(M * M - a * a))
            rplus = M + sqrtma
            rmins = M - sqrtma
            oodr = 2 * M / (rplus - rmins)
            tmp = (rplus * np.log(rBL - rplus) - rmins * np.log(rBL - rmins)) * oodr
            rs = rBL + tmp
        tki = tBL - rBL + rs
        R = S * rBL / (S + rBL)
        traj_shift = -tki - 4 * M * np.log(1.0 - R / S) + R * R / (S - R)

        rho = S
        shift = (
            -rho
            - 4 * M * np.log((S * rho + 2 * M * rho - 2 * M * S) / S)
            + 2 * M * np.log(2 * M)
            - traj_shift
        )
        return shift

    def pn_hflx(self, kind="E", coords="EOB", order="NNLO_mod_fact"):
        """
        Compute horizon fluxes from PN expressions.
        Output normalized by nu^2 to agree with numerical flux.
        TODO: promote this to Waveform method rather than Teuk?
        """
        if self.dyn is None:
            self.load_dyn()

        if "rdot" not in self.dyn.keys():
            self.dyn["rdot"] = np.gradient(self.dyn["r"], self.dyn["t"])

        if coords != "EOB":
            raise NotImplementedError(
                "Harmonic coordinate fluxes are not implemented yet."
            )

        if "QC" in order:
            x = self.dyn["Omg"] ** (1.0 / 3)
            return pn.mj1_dot_QC_NNLO(1.0, self.mu, x, self.dyn["r"], lo=order[-1])[
                int(kind == "J")
            ]
        else:
            if kind == "J":
                return (
                    pn.j1_dot_spin_eob(
                        1.0,
                        self.mu,
                        self.dyn["r"],
                        self.dyn["prstar"],
                        self.dyn["pph"],
                        self.a,
                        0.0,
                        order,
                        omg=self.dyn["Omg"],
                    )
                    / self.mu**2
                )
            elif kind == "E":
                return (
                    pn.m1_dot_spin_eob(
                        1.0,
                        self.mu,
                        self.dyn["r"],
                        self.dyn["prstar"],
                        self.dyn["pph"],
                        self.a,
                        0.0,
                        order,
                        omg=self.dyn["Omg"],
                        rdot=self.dyn["rdot"],
                    )
                    / self.mu**2
                )
            else:
                raise ValueError("Flux kind not recognized; only E and J accepted.")
