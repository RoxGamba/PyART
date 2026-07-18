import logging
import numpy as np
import os
import sys
import h5py
import json
from ..waveform import Waveform
from ..utils import cat_utils as cat_ut
from ..utils.utils import LoggerWriter
from ..utils.wf_utils import get_multipole_dict


class Waveform_SXS(Waveform):
    """
    Class to handle SXS waveforms
    Assumes that the data is in the directory specified py `path`,
    and that all simulations are stored in folders like SXS_BBH_XXXX,
    each containing the various `LevY` folders.
    e.g., the current default is
        ../dat/SXS_BBH_XXXX/LevY/
    """

    def __init__(
        self,
        path=r"../dat/SXS/",
        ID="0001",
        order=2,
        level=None,
        cut_N=None,
        cut_U=None,
        ellmax=8,
        load=["hlm", "metadata"],
        download=False,
        downloads=["hlm", "metadata"],
        load_m0=False,
        nu_rescale=False,
        src="BBH",
        ignore_deprecation=False,
        basename=None,  # if None, use default according to src
    ):
        """
        Initialize the Waveform_SXS class.

        Parameters
        ----------
        path : str, optional
            Path where the SXS data is stored. Default is "../dat/SXS/".
        ID : str or int, optional
            ID of the SXS simulation to load. Default is "0001".
        order : int, optional
            Extrapolation order to use. Default is 2.
        level : int or None, optional
            Numerical resolution level to use. If None, the highest available
            level will be used. Default is None.
        cut_N : int or None, optional
            Number of initial data points to cut from the waveform. If None,
            no initial cut is applied. Default is None.
        cut_U : float or None, optional
            Initial retarded time to start the waveform. If None, no initial cut
            is applied. Default is None.
        ellmax : int, optional
            Maximum ell value to load. Default is 8.
        load : list of str, optional
            Options to load. Can include "hlm", "metadata", "horizons", "psi4lm".
            Default is ["hlm", "metadata"].
        download : bool, optional
            If True, download the simulation from the SXS catalog if not found
            locally. Default is False.
        downloads : list of str, optional
            Options to download if `download` is True. Can include "hlm",
            "metadata", "horizons", "psi4lm". Default is ["hlm", "metadata"].
        load_m0 : bool, optional
            If True, load the m=0 modes as well. Default is False.
        nu_rescale : bool, optional
            If True, rescale the waveform by the symmetric mass ratio nu.
            Default is False.
        src : str, optional
            Source type. Can be "BBH" or "BHNS". Default is "BBH".
        ignore_deprecation : bool, optional
            If True, ignore deprecation warnings when downloading. Default is False.
        basename : str or None, optional
            Base name of the h5 file to load. If None, use default according to `src`.
            Default is None.
        """
        super().__init__()
        if isinstance(ID, int):
            ID = f"{ID:04}"

        self.ID = ID
        sxs_folder = f"SXS_{src}_{ID}"
        if os.path.basename(path) == sxs_folder:
            self.sxs_data_path = path
        else:
            self.sxs_data_path = os.path.join(path, sxs_folder)
        self.order = order
        self.level = level
        self.cut_N = cut_N
        self.cut_U = cut_U
        self.ellmax = ellmax
        self._kind = "SXS"
        self.src = src
        self.nr = None
        self._domain = "Time"
        self.nu_rescale = nu_rescale

        if basename is None:
            if src == "BHNS" and int(self.ID) <= 7:
                basename = "rhOverM_Asymptotic_GeometricUnits.h5"
            elif (src == "BHNS" and int(self.ID) > 7) or src == "BBH":
                basename = "rhOverM_Asymptotic_GeometricUnits_CoM.h5"
            else:
                raise ValueError("basename is None, but unknown src!")
        self.basename = basename

        if isinstance(self.level, int):
            levpath = f"{self.sxs_data_path}/Lev{self.level}"
        else:
            levpath = self.sxs_data_path
            if os.path.exists(levpath):
                lev_dirs = [
                    d
                    for d in os.listdir(levpath)
                    if os.path.isdir(os.path.join(levpath, d)) and d.startswith("Lev")
                ]
                if not lev_dirs:
                    levpath = None
            else:
                levpath = None
        self.check_cut_consistency()

        needs_download = levpath is None or not os.path.exists(levpath)
        if not needs_download:
            # if files are already  downloaded, additionally check that N-order
            # is there as well. Note: if we enter here, levpath is not None
            # and the path exists, and thus lev_dirs not empty
            order_group = f"Extrapolated_N{self.order}.dir"
            if self.level is None:
                level = int(lev_dirs[-1].replace("Lev", ""))
            else:
                level = self.level
            fname = self.get_lev_fname(basename=self.basename, level=level)
            with h5py.File(fname, "r") as f:
                needs_download = order_group not in f
            if needs_download:
                logging.info(
                    f"{levpath} found, but not the requested N={self.order} order. Download needed."
                )

        if needs_download:
            if download:
                logging.info(
                    f"The path {self.sxs_data_path} does not exist, contains no 'Lev*'"
                    + "directory, or does not contain the requested order."
                )
                logging.info("Downloading the simulation from the SXS catalog.")
                self.download_simulation(
                    ID=self.ID,
                    path=path,
                    downloads=downloads,
                    level=self.level,
                    ignore_deprecation=ignore_deprecation,
                    extrapolation_order=order,
                )
            else:
                logging.warning(
                    "Use download=True to download the simulation from the SXS catalog."
                )
                raise FileNotFoundError(
                    f"The path {self.sxs_data_path} does not exist or contains no 'Lev*' directory."
                )

        if isinstance(self.level, int):
            fname = self.get_lev_fname(basename=self.basename)
            if os.path.exists(fname):
                self.nr = h5py.File(fname)
            else:
                raise FileNotFoundError(
                    f"SXS path found, but the requested level ({self.level:d}) is not available!"
                )

        elif self.level is None:
            ref_lv_max = 7
            ref_lv_min = 1
            if "hlm" not in load:
                self.basename = "metadata.json"
            for lvn in range(ref_lv_max, ref_lv_min - 1, -1):
                fname = self.get_lev_fname(level=lvn, basename=self.basename)
                if os.path.exists(fname):
                    if "hlm" in load:
                        self.nr = h5py.File(fname)
                    self.level = lvn
                    break
                elif lvn == ref_lv_min:
                    raise RuntimeError(
                        f"No data for ref-levels:[{ref_lv_min:d},{ref_lv_max:d}] found"
                    )

        else:
            raise RuntimeError(f"Invalid input for level: {self.level}")

        if "metadata" in load:
            self.load_metadata()
        if "hlm" in load:
            self.load_hlm(load_m0=load_m0)
        if "horizons" in load:
            self.load_horizon()
        if "psi4lm" in load:
            self.load_psi4lm(load_m0=load_m0)

        if self.nr is not None:
            self.nr.close()
        pass

    def check_cut_consistency(self):
        """
        Check consistency between cut_N and cut_U.
        If both are None, set cut_N=0.
        If both are given, raise an error.
        If one is given, set the other one accordingly.
        """

        if self.cut_N is not None and self.cut_U is not None:
            raise RuntimeError(
                "Conflict between cut_N and cut_U!\n"
                "When initializing, only one between cut_N and cut_U should be given in input.\n"
                "The other one is temporarly set to None and (consistently) updated in self.load_hlm()"
            )
        elif self.cut_N is None and self.cut_U is None:
            self.cut_N = 0
        pass

    def get_lev_fname(self, level=None, basename=None):
        """
        Return file-name in a SXS-path with specified level,
        e.g. /my/sxs/path/Lev4/my_basename
        If basename is None, then return only /my/sxs/path/Lev4

        Parameters
        ----------
        level : int or None
            Level to use. If None, use self.level.
        basename : str or None
            Base name of the h5 file to load. If None, use self.basename.

        Returns
        -------
        fname : str
            Full path to the requested file.
        """

        if not isinstance(basename, str):
            raise ValueError("basename must be a string!")
        if level is None:
            level = self.level

        tojoin = f"Lev{level:d}/{basename}"
        return os.path.join(self.sxs_data_path, tojoin)

    def download_simulation(
        self,
        ID="0001",
        path=None,
        downloads=["hlm", "metadata"],
        level=None,
        ignore_deprecation=False,
        extrapolation_order=None,
    ):
        """
        Download the simulation from the SXS catalog; requires the sxs module
        Note that for backwards compatibility we transform the new (2025 onwards)
        format of the data to the old one, which is human-readable and easier to handle.

        Parameters
        ----------
        ID : str or int, optional
            ID of the SXS simulation to download. Default is "0001".
        path : str, optional
            Path where to store the downloaded data. Default is None, which
            uses the current value of self.sxs_data_path. If the directory
            does not exist, it will be created.
        downloads : list of str, optional
            Options to download. Can include "hlm", "metadata", "horizons", "psi4lm".
            Default is ["hlm", "metadata"].
        level : int or None, optional
            Numerical resolution level to download. If None, the highest available
            level will be downloaded. Default is None.
        ignore_deprecation : bool, optional
            If True, ignore deprecation warnings when downloading. Default is False.
        extrapolation_order : int or None, optional
            Extrapolation order to use. If None, use self.order. Default is None.
        """
        import h5py
        from itertools import product
        import sxs as sxsmod
        import shutil

        if path is None:
            raise ValueError(
                "download_simulation needs a path: it is both where the data is "
                "written and the cache directory handed to the sxs module."
            )

        # The sxs module reads SXSCACHEDIR to decide where to download. Keep the
        # directory in a local variable too, and use that below: reading the
        # environment back would couple this call to whatever a previous one
        # left there, and rmtree is run against it.
        cache_dir = path
        logging.info(f"Setting the download (cache) directory to {cache_dir}")
        os.environ["SXSCACHEDIR"] = cache_dir

        # Define the simulation ID and load it
        name = f"SXS:{self.src}:{ID}"
        if level is not None:
            name_level = f"{name}/Lev{level}"
        else:
            name_level = name

        # based on the logging level, redirect stdout to the logger. This is
        # because the sxs module prints a lot of information to stdout.
        # try/finally: an exception in between must not leave stdout redirected
        # for good.
        original_stdout = sys.stdout
        sys.stdout = LoggerWriter(logging.getLogger(__name__))
        try:
            sxs_sim = sxsmod.load(
                name_level,
                extrapolation_order=extrapolation_order,
                extrapolation=f"N{extrapolation_order}",
                ignore_deprecation=ignore_deprecation,
                progress=True,
            )
            logging.info(f"Loaded SXS simulation {name_level}.")

            # Set Level if not already set
            self.level = self.level or int(
                sxs_sim.Lev.replace("Lev", "")
            )  # Guarantees int(self.level)

            out_dir = self.get_lev_fname(level=self.level, basename="")
            os.makedirs(out_dir, exist_ok=True)

            # Save hlm data if requested
            if "hlm" in downloads:
                wav = sxs_sim.h
                extp = f"Extrapolated_N{extrapolation_order}.dir"
                to_h5file = {extp: {}}
                ellmax = 8  # wav.ellmax
                modes = [
                    (l, m)
                    for l, m in product(
                        range(2, ellmax + 1), range(-ellmax, ellmax + 1)
                    )
                    if l >= np.abs(m)
                ]
                for ell, m in modes:
                    try:
                        idx = wav.index(ell, m)
                        mode_string = f"Y_l{ell}_m{m}.dat"
                        data = np.column_stack(
                            (wav.time, wav[:, idx].real, wav[:, idx].imag)
                        )
                        to_h5file[extp][mode_string] = data
                    except ValueError:
                        logging.warning(
                            f"Mode Y_l{ell}_m{m} not found in the waveform data! Skipping."
                        )
                        continue
                # create/update the h5 file
                filename = os.path.join(
                    out_dir, f"rhOverM_Asymptotic_GeometricUnits_CoM.h5"
                )
                with h5py.File(filename, "a") as h5file:
                    if extp in h5file:
                        logging.info(f"{extp} already present, skipping.")
                    else:
                        save_dict_to_h5(h5file, {extp: to_h5file[extp]})
                logging.info("Saved hlm data.")

            # Save psi4lm data if requested
            if "psi4lm" in downloads:
                wav = sxs_sim.psi4
                extp = f"Extrapolated_N{extrapolation_order}.dir"
                to_h5file = {extp: {}}
                ellmax = 8  # wav.ellmax
                modes = [
                    (l, m)
                    for l, m in product(
                        range(2, ellmax + 1), range(-ellmax, ellmax + 1)
                    )
                    if l >= np.abs(m)
                ]
                for mode in modes:
                    mode_string = "Y_l" + str(mode[0]) + "_m" + str(mode[1]) + ".dat"
                    if mode_string in wav:
                        to_h5file[extp][mode_string] = wav[mode_string]

                # create/update the h5 file
                filename = os.path.join(
                    out_dir, f"rMPsi4_Asymptotic_GeometricUnits_CoM.h5"
                )
                with h5py.File(filename, "a") as h5file:
                    if extp in h5file:
                        logging.info(f"{extp} already present, skipping.")
                    else:
                        save_dict_to_h5(h5file, {extp: to_h5file[extp]})
                logging.info("Saved psi4lm data.")

            # Save horizons if requested
            if "horizons" in downloads:
                hrz = sxs_sim.horizons
                to_h5file = {}
                for object in ["AhA.dir", "AhB.dir", "AhC.dir"]:
                    to_h5file[object] = {}
                    for key in [
                        "CoordCenterInertial.dat",
                        "ChristodoulouMass.dat",
                        "DimensionfulInertialSpinMag.dat",
                        "chiInertial.dat",
                    ]:
                        try:
                            to_h5file[object][key] = hrz[f"{object}/{key}"]
                        except KeyError:
                            logging.warning(
                                f"{object}/{key} not found in horizons data! Skipping."
                            )
                            continue
                # create the h5 file
                h5file = h5py.File(os.path.join(out_dir, f"Horizons.h5"), "w")
                save_dict_to_h5(h5file, to_h5file)
                h5file.close()
                logging.info("Saved horizons data.")

            # Save metadata if requested
            if "metadata" in downloads:
                import json

                with open(os.path.join(out_dir, "metadata.json"), "w") as file:
                    json.dump(sxs_sim.metadata, file, indent=2)
                logging.info("Saved metadata.")

            # find old SXS download folders and remove them. Only the
            # colon-named ones the sxs module creates, and only in the directory
            # this call actually downloaded into.
            flds = [f for f in os.listdir(cache_dir) if ID in f]
            for fld in flds:
                if ":" in fld:
                    shutil.rmtree(os.path.join(cache_dir, fld))
        finally:
            # Restore stdout
            sys.stdout = original_stdout

        pass

    def load_metadata(self):
        """
        Load the sxs metadata from the metadata.json file.
        Transform it to the PyART format, but also store the original
        metadata as self.ometadata for completeness.
        """
        with open(self.get_lev_fname(basename="metadata.json"), "r") as file:
            ometa = json.load(file)  # original_metadata
            file.close()
        self.ometadata = ometa  # store also original metadata, for completeness

        # TODO : 1) check if these quantities are mass-rescaled or not
        #        2) here we are using initial quantities, not ref. The
        #           reason is that ADM integrals are not given at ref time

        def is_valid(key, vtype=None):
            if key not in ometa:
                return False
            if isinstance(ometa[key], str):
                return False
            if vtype is not None:  # check var-type if provided
                return isinstance(ometa[key], vtype)
            return True

        M1 = ometa["reference_mass1"]
        if is_valid("reference_mass2", vtype=float):
            M2 = ometa["reference_mass2"]
        else:
            logging.warning(
                "reference_mass2 not found or invalid! Using initial masses"
            )
            M1 = ometa["initial_mass1"]
            M2 = ometa["initial_mass2"]

        q = M2 / M1
        if q < 1:
            q = 1 / q
        nu = q / (1 + q) ** 2
        M = M1 + M2

        def read_spin_variable(spin_idx):
            attempts = [
                "reference_dimensionless_spin",
                "initial_dimensionless_spin",
                "reference_spin",
            ]
            for attempt in attempts:
                key = attempt + str(spin_idx)
                if is_valid(key):
                    hS = np.array(ometa[key])
                    # 'reference_spin' is dimensionful, unlike the two
                    # '*_dimensionless_spin' entries: normalize it by the mass
                    # squared, as done for the remnant spin below
                    if attempt == "reference_spin":
                        if spin_idx == 1:
                            hS = hS / M1**2
                        elif spin_idx == 2:
                            hS = hS / M2**2
                    return hS, attempt
            raise KeyError(
                f"No valid spin entry found for body {spin_idx}, tried: {attempts}"
            )

        hS1, skey1 = read_spin_variable(1)
        hS2, skey2 = read_spin_variable(2)

        if not skey1 == skey2:
            logging.warning(f"using different spin-entries! {skey1} and {skey2}")

        pos1 = np.array(ometa["reference_position1"])
        pos2 = np.array(ometa["reference_position2"])
        r0 = np.linalg.norm(pos1 - pos2)

        try:
            Mf = float(ometa["remnant_mass"])
            if is_valid("remnant_dimensionless_spin"):
                afv = np.array(ometa["remnant_dimensionless_spin"])
            elif is_valid("remnant_spin"):
                afv = np.array(ometa["remnant_spin"]) / Mf**2
            else:
                raise ValueError("Unknown key for remnant's spin or invalid value")
            afz = afv[2]
        except Exception as e:
            logging.warning(f"Failed in reading remnant properties: {e}")
            Mf = None
            afv = None
            afz = None

        alt_names = ometa["alternative_names"]
        if isinstance(alt_names, list):
            if len(alt_names) > 1:
                name = alt_names[1]
            else:
                name = alt_names[0]
        else:
            name = alt_names

        ecc = ometa["reference_eccentricity"]
        if isinstance(ecc, str):
            # there are things like '<1.7e+00' in meta: an upper bound of order
            # unity carries no information, while a small one means circular
            if "<" in ecc and "e+00" in ecc:
                ecc = None
            else:
                ecc = 1e-5
        else:
            ecc = float(ecc)

        # Read ADM quantities. If not available (e.g. BHNS:0008 or BHNS:0009)
        # set to None
        if "initial_ADM_angular_momentum" in ometa:
            J0 = np.array(ometa["initial_ADM_angular_momentum"])
            J0z = J0[2]
            Lz = J0 - hS1 * M1 * M1 - hS2 * M2 * M2
            pph0 = Lz[2] / (M * M * nu)
        else:
            logging.warning("No angular momentum found")
            J0 = None
            J0z = None
            Lz = None
            pph0 = None

        if "initial_ADM_energy" in ometa:
            E0 = ometa["initial_ADM_energy"]
            E0byM = E0 / M
        else:
            E0 = None
            E0byM = None

        if "initial_ADM_linear_momentum" in ometa:
            P0v = np.array(ometa["initial_ADM_linear_momentum"])
        else:
            P0v = None

        orb_freq_ometa = ometa["reference_orbital_frequency"]
        if isinstance(orb_freq_ometa, float):
            f0 = orb_freq_ometa / np.pi
        elif len(orb_freq_ometa) == 3:
            f0 = orb_freq_ometa[2] / np.pi
        else:
            raise RuntimeError("Unknown format for reference_orbital_frequency")

        # Set Lambda(s)
        if self.src == "BBH":
            LambdaAl2 = 0.0
            LambdaBl2 = 0.0
        elif self.src == "BHNS":
            LambdaAl2 = 0.0
            if name == "SXS:BHNS:0001":
                LambdaBl2 = 526.0
            elif name == "SXS:BHNS:0003":
                LambdaBl2 = 607.0
            elif int(self.ID) <= 9:
                LambdaBl2 = 791.0
            else:
                raise RuntimeError(f"Unknown LambdaBl2!")
            if M2 > M1:
                raise RuntimeError(f"BHNS: M2>M1 but Lambda1 (LambdaAl2) is zero!")
        else:
            raise RuntimeError(f"Unknown source: {self.src}")

        metadata = {
            "name": name,  # i.e. store as name 'SXS:BBH:ID'
            "ref_time": ometa["reference_time"],
            # masses and spins
            "m1": M1,
            "m2": M2,
            "M": M,
            "q": q,
            "nu": nu,
            "S1": hS1 * M1 * M1,  # [M2]
            "S2": hS2 * M2 * M2,
            "chi1x": hS1[0],  # dimensionless
            "chi1y": hS1[1],
            "chi1z": hS1[2],
            "chi2x": hS2[0],  # dimensionless
            "chi2y": hS2[1],
            "chi2z": hS2[2],
            "LambdaAl2": LambdaAl2,
            "LambdaBl2": LambdaBl2,
            # positions
            "pos1": pos1,
            "pos2": pos2,
            "r0": r0,
            "e0": ecc,
            # frequencies
            "f0v": np.array(ometa["reference_orbital_frequency"]) / np.pi,
            "f0": f0,
            # ADM quantities (INITIAL, not REF)
            "E0": E0,
            "P0": P0v,
            "J0": J0,
            "Jz0": J0z,
            "E0byM": E0byM,
            "pph0": pph0,
            # remnant
            "Mf": Mf,
            "afv": afv,
            "af": afz,
            "scat_angle": None,
        }
        metadata["flags"] = cat_ut.get_flags(metadata)
        # check that all the required quantities are given
        cat_ut.check_metadata(metadata, raise_err=True)
        # then store as attribute
        self.metadata = metadata
        pass

    def load_horizon(self):
        """
        Load the horizon data from Horizons.h5 file.
        Store the data in self._dyn dictionary.

        The datasets are stored with the time in the first column:
        ChristodoulouMass.dat and DimensionfulInertialSpinMag.dat are (N,2),
        while CoordCenterInertial.dat and chiInertial.dat are (N,4). Here the
        time is stored once, in dyn["t"], and the vectors are stored without
        it, i.e. with shape (N,3).

        AhA/AhB are the two individual horizons; AhC is the common horizon,
        which only forms at merger and therefore lives on its own, shorter time
        array (dyn["t_remnant"]) and is absent for runs that do not merge.
        """
        horizon = h5py.File(self.get_lev_fname(basename="Horizons.h5"))

        def read_horizon(obj):
            """
            Read one apparent horizon, dropping the leading time column from
            the vectors. Datasets that the download skipped are returned as
            None rather than raising.
            """
            grp = horizon[obj]

            def dset(name, vector=False):
                if name not in grp:
                    logging.warning(f"{obj}/{name} not found in horizons data!")
                    return None
                return grp[name][:, 1:] if vector else grp[name][:, 1]

            return {
                "t": grp["ChristodoulouMass.dat"][:, 0],
                "m": dset("ChristodoulouMass.dat"),
                # chiInertial is the dimensionless spin vector, which is what
                # dyn["chi"] is meant to hold. DimensionfulInertialSpinMag is
                # |S|: dimensionful (chi = S/m^2) and a magnitude, so it has no
                # direction to project on L.
                "chi": dset("chiInertial.dat", vector=True),
                "S_mag": dset("DimensionfulInertialSpinMag.dat"),
                "x": dset("CoordCenterInertial.dat", vector=True),
            }

        A = read_horizon("AhA.dir")
        B = read_horizon("AhB.dir")

        self._dyn["t"] = A["t"]
        self._dyn["m1"] = A["m"]
        self._dyn["m2"] = B["m"]
        self._dyn["chi1"] = A["chi"]
        self._dyn["chi2"] = B["chi"]
        self._dyn["S1_mag"] = A["S_mag"]
        self._dyn["S2_mag"] = B["S_mag"]
        self._dyn["x1"] = A["x"]
        self._dyn["x2"] = B["x"]

        if "AhC.dir" in horizon:
            C = read_horizon("AhC.dir")
            self._dyn["t_remnant"] = C["t"]
            self._dyn["m_remnant"] = C["m"]
            self._dyn["chi_remnant"] = C["chi"]
            self._dyn["S_remnant_mag"] = C["S_mag"]
            self._dyn["x_remnant"] = C["x"]
        else:
            logging.info(
                "No common horizon (AhC.dir) in the horizons data: "
                "remnant quantities not loaded."
            )

        pass

    def compute_spins_at_tref(self, tref):
        """
        Compute the parallel and perpendicular components of the spins w.r.t L
        at a reference time tref. This requires the horizon data to be loaded.

        Parameters
        ----------
        tref : float
            Reference time

        Returns
        -------
        chi1_L, chi1_perp, chi2_L, chi2_perp : float
            The parallel and perpendicular components of the spins at tref
        """
        d = self.dyn

        # find the index of the reference time. The dyn vectors carry no time
        # column, so they are indexed by time only.
        idx = np.argmin(np.abs(d["t"] - tref))
        chi1_ref = d["chi1"][idx]
        chi2_ref = d["chi2"][idx]
        x1_ref = d["x1"][idx]
        x2_ref = d["x2"][idx]

        # time derivative of x1 and x2
        x1_dot = np.transpose([np.gradient(d["x1"][:, i], d["t"]) for i in range(3)])
        x2_dot = np.transpose([np.gradient(d["x2"][:, i], d["t"]) for i in range(3)])
        x = x1_ref - x2_ref

        x_dot = x1_dot[idx] - x2_dot[idx]
        L_hat_ref = np.cross(x, x_dot) / np.linalg.norm(np.cross(x, x_dot))

        # compute the spins projected on L_hat_ref
        chi1_L = np.dot(chi1_ref, L_hat_ref)
        chi2_L = np.dot(chi2_ref, L_hat_ref)
        chi1_perp = np.linalg.norm(chi1_ref - chi1_L * L_hat_ref)
        chi2_perp = np.linalg.norm(chi2_ref - chi2_L * L_hat_ref)
        return chi1_L, chi1_perp, chi2_L, chi2_perp

    def load_hlm(self, ellmax=None, load_m0=False):
        """
        Load the hlm modes from the h5 file. Store the data in self._hlm dictionary.

        Parameters
        ----------
        ellmax : int or None
            Maximum ell value to load. If None, use self.ellmax.
        load_m0 : bool, optional
            If True, load the m=0 modes as well. Default is False.
        """
        if ellmax == None:
            ellmax = self.ellmax
        order = f"Extrapolated_N{self.order}.dir"

        if not hasattr(self, "metadata"):
            raise RuntimeError("Load metadata before loading hlm!")

        from itertools import product

        modes = [
            (l, m)
            for l, m in product(range(2, ellmax + 1), range(-ellmax, ellmax + 1))
            if (m != 0 or load_m0) and l >= np.abs(m)
        ]

        tmp_u = self.nr[order]["Y_l2_m2.dat"][:, 0]
        # self.check_cut_consistency()
        if self.cut_N is None:
            self.cut_N = np.argwhere(tmp_u >= self.cut_U)[0][0]
        if self.cut_U is None:
            self.cut_U = tmp_u[self.cut_N]

        self._u = tmp_u[self.cut_N :]
        self._t = self._u  # FIXME: should we use another time?

        dict_hlm = {}
        for mode in modes:
            l = mode[0]
            m = mode[1]
            mode = "Y_l" + str(l) + "_m" + str(m) + ".dat"
            hlm = self.nr[order][mode]
            h = hlm[:, 1] + 1j * hlm[:, 2]
            if self.nu_rescale:
                h /= self.metadata["nu"]
            # Build the mode dict with the shared helper, so that the sign
            # conventions match every other producer. It is applied to the whole
            # mode and the junk is cut afterwards: the phase must be unwrapped
            # before the cut, or it would be offset by a multiple of 2pi.
            key = (l, m)
            dict_hlm[key] = {
                ky: val[self.cut_N :] for ky, val in get_multipole_dict(h).items()
            }
        self._hlm = dict_hlm
        pass

    def load_psi4lm(self, ellmax=None, load_m0=False):
        """
        Load the psi4lm modes from the h5 file. Store the data in self._psi4lm dictionary.

        Parameters
        ----------
        ellmax : int or None
            Maximum ell value to load. If None, use self.ellmax.
        load_m0 : bool, optional
            If True, load the m=0 modes as well. Default is False.
        """
        psi4_basename = self.basename.replace("rhOverM", "rMPsi4")
        fname = self.get_lev_fname(level=self.level, basename=psi4_basename)
        if not os.path.exists(fname):
            raise FileNotFoundError(f"psi4 file not found: {fname}")
        self.nr_psi = h5py.File(fname)

        if ellmax == None:
            ellmax = self.ellmax
        order = f"Extrapolated_N{self.order}.dir"

        if not hasattr(self, "metadata"):
            raise RuntimeError("Load metadata before loading hlm!")

        from itertools import product

        modes = [
            (l, m)
            for l, m in product(range(2, ellmax + 1), range(-ellmax, ellmax + 1))
            if (m != 0 or load_m0) and l >= np.abs(m)
        ]

        tmp_u = self.nr_psi[order]["Y_l2_m2.dat"][:, 0]

        if self.cut_N is None:
            self.cut_N = np.argwhere(tmp_u >= self.cut_U)[0][0]
        if self.cut_U is None:
            self.cut_U = tmp_u[self.cut_N]

        if self._u is None:
            raise RuntimeError(
                "psi4 times are taken from the hlm time array, but hlm was "
                "never loaded: add 'hlm' to the load list."
            )
        self._t_psi4 = self._u  # FIXME: should we use another time?

        dict_psi4lm = {}
        for mode in modes:
            l = mode[0]
            m = mode[1]
            mode = "Y_l" + str(l) + "_m" + str(m) + ".dat"
            psi4lm = self.nr_psi[order][mode]
            psi4 = psi4lm[:, 1] + 1j * psi4lm[:, 2]
            if self.nu_rescale:
                psi4 /= self.metadata["nu"]
            # see load_hlm: shared helper first, junk cut afterwards
            key = (l, m)
            dict_psi4lm[key] = {
                ky: val[self.cut_N :] for ky, val in get_multipole_dict(psi4).items()
            }
        self._psi4lm = dict_psi4lm
        pass

    def compute_psi4_from_hlm(self):
        """
        Compute the psi4lm by taking two time derivatives
        of the hlm modes
        """
        dict_psi4lm = {}
        t = self.u
        for ky in self.hlm.keys():
            h = self.hlm[ky]["z"]
            ddh = np.gradient(np.gradient(h, t), t)
            dict_psi4lm[ky] = {
                "A": abs(ddh),
                "p": -np.unwrap(np.angle(ddh)),
                "real": ddh.real,
                "imag": ddh.imag,
                "z": ddh,
            }
        self._psi4lm = dict_psi4lm

    def to_lvk(self, modes="all"):
        """
        Convert the data to LVK format, output an
        SXS_BBH_XXXX_ResY.h5 file

        Wrapper function to the `convert_sxs_to_lvc.py` from
        https://github.com/sxs-collaboration/catalog_tools/tree/master

        Parameters
        ----------
        modes : str or list of tuple, optional
            Modes to convert. If "all", convert all modes up to self.ellmax.
            If a list of tuples, convert only the specified modes.
            Default is "all".
        """
        from ..utils import convert_sxs_to_lvc as conv

        logging.info("Converting SXS data to LVK format...")
        # Path to Horizons file
        horizons_file = os.path.join(
            self.sxs_data_path, f"Lev{self.level}", "Horizons.h5"
        )

        if not os.path.isfile(horizons_file):
            logging.info(f"Horizons file not found: {horizons_file}. Downloading it...")
            self.download_simulation(
                ID=self.ID,
                path=self.sxs_data_path,
                downloads=["horizons"],
                level=self.level,
            )

        conv.convert_simulation(
            f"{self.sxs_data_path}/Lev{self.level}",
            self.level,
            modes,
            self.sxs_data_path,
            None,
        )
        pass


def save_dict_to_h5(h5group, dictionary):
    """
    Recursively save a nested dictionary to an HDF5 group or file.

    Parameters
    ----------
    h5group : h5py.Group or h5py.File
        The HDF5 group or file where the dictionary will be saved.
    dictionary : dict
        The nested dictionary to save.
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            subgroup = h5group.create_group(key)
            save_dict_to_h5(subgroup, value)
        else:
            try:
                h5group.create_dataset(key, data=value)
            except TypeError:
                # Handle scalar values that cannot be turned into datasets
                h5group.attrs[key] = value
