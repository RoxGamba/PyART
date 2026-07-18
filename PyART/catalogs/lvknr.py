from ..waveform import Waveform
from ..utils.wf_utils import get_multipole_dict

try:
    import lalsimulation as lalsim
    import lal
except ImportError:
    raise ImportError("WARNING: LALSimulation and/or LAL not installed.")

import os
import logging
import subprocess
import numpy as np
import h5py

logger = logging.getLogger(__name__)

# TODO: read them from utils
Msun_m = 1.4766250614046494e3
Msun_s = 4.9254910255435759e-6


class Waveform_LVKNR(Waveform):
    """
    Class to handle waveforms from the LVCNR catalog
    """

    def __init__(
        self,
        lvcnr_path=None,
        catalog="SXS",
        ID=None,
        ellmax=8,
        load_m0=True,
        download=False,
        nu_rescale=False,
        **kwargs,
    ):

        super().__init__()

        if lvcnr_path is None:
            raise FileNotFoundError("Please provide a path to the LVCNR catalog.")
        if ID is None:
            raise FileNotFoundError("Please provide a simulation ID.")
        if catalog not in ["SXS", "BAM", "Cardiff-UIB", "GeorgiaTech", "RIT"]:
            raise NameError(f"{catalog} catalog not supported.")

        self.ID = ID
        self.lvcnr_path = lvcnr_path
        self.catalog = catalog
        self.data_path = os.path.join(lvcnr_path, catalog, ID)
        self.ellmax = ellmax
        self.load_m0 = load_m0
        self.nu_rescale = nu_rescale

        if os.path.exists(self.data_path) == False:
            raise FileNotFoundError(f"The simulation {ID} does not exist.")

        if self._is_git_lfs_pointer(self.data_path):
            if download == True:
                logger.info(f"The path {self.lvcnr_path} does not exist.")
                logger.info("Downloading the simulation from the LVCNR catalog.")
                self.download_simulation(ID=ID)
            else:
                logger.warning(
                    "Use download=True to download the simulation from the LVCNR catalog."
                )
                raise FileNotFoundError(f"The path {self.data_path} does not exist.")

        self.load_metadata()
        self.load_hlm()

    def load_metadata(self):
        """
        Load (some) metadata from the simulation"
        """
        # opened locally rather than stored on self: an open h5py handle would
        # make the waveform un-deep-copyable, which the Matcher needs, and
        # load_hlm (called right after) does not need it, working instead
        # through self.data_path via lalsimulation.
        with h5py.File(self.data_path, "r") as f:
            self._load_metadata_from_file(f)

    def _load_metadata_from_file(self, f):
        M = 1  # set M = 1, always
        self.metadata = {}

        Mt = f.attrs["mass1"] + f.attrs["mass2"]
        m1 = f.attrs["mass1"] * M / Mt
        m2 = f.attrs["mass2"] * M / Mt
        q = m1 / m2
        nu = q / (1 + q) ** 2
        flow = f.attrs["f_lower_at_1MSUN"]
        fref = flow
        spins = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(
            fref, M, self.data_path
        )
        c1x, c1y, c1z = spins[0], spins[1], spins[2]
        c2x, c2y, c2z = spins[3], spins[4], spins[5]

        # overwrite some of the metadata
        self.metadata["m1"] = m1
        self.metadata["m2"] = m2
        self.metadata["M"] = m1 + m2
        self.metadata["q"] = m1 / m2
        self.metadata["nu"] = nu
        self.metadata["chi1x"] = c1x
        self.metadata["chi1y"] = c1y
        self.metadata["chi1z"] = c1z
        self.metadata["chi2x"] = c2x
        self.metadata["chi2y"] = c2y
        self.metadata["chi2z"] = c2z
        self.metadata["initial_frequency"] = flow * Msun_s
        self.metadata["eccentricity"] = f.attrs["eccentricity"]

        self.metadata["m1SI"] = m1 * lal.MSUN_SI
        self.metadata["m2SI"] = m2 * lal.MSUN_SI
        self.metadata["frefSI"] = flow

        self.metadata["original"] = {}
        for key in f.attrs.keys():
            self.metadata["original"][key] = f.attrs[key]
        pass

    def load_hlm(self):
        """
        Load the waveform modes
        """

        ellmax = self.ellmax
        load_m0 = self.load_m0
        metadata = self.metadata
        dT = 1.0 / 262144.0  # hardcoded to 1/262144 for now
        self._hlm = {}

        # add the modes to the LALdictionary
        from itertools import product

        modes = [
            (l, m)
            for l, m in product(range(2, ellmax + 1), range(-ellmax, ellmax + 1))
            if (m != 0 or load_m0) and l >= np.abs(m)
        ]
        modearr = lalsim.SimInspiralCreateModeArray()
        for mode in modes:
            lalsim.SimInspiralModeArrayActivateMode(modearr, mode[0], mode[1])

        _, hlms = lalsim.SimInspiralNRWaveformGetHlms(
            dT,
            metadata["m1SI"],
            metadata["m2SI"],
            1e6 * lal.PC_SI,
            metadata["frefSI"],
            metadata["frefSI"],
            metadata["chi1x"],
            metadata["chi1y"],
            metadata["chi1z"],
            metadata["chi2x"],
            metadata["chi2y"],
            metadata["chi2z"],
            self.data_path,
            modearr,
        )

        # store the modes
        for i in range(len(modes)):
            l, m = hlms.l, hlms.m
            this_mode = hlms.mode.data.data

            # scale the amplitude (a positive real factor, so it does not
            # affect the phase); scale this_mode itself, not just A, so that
            # z stays consistent with real/imag/A (get_multipole_dict derives
            # everything from z, and previously z was left unscaled while
            # A/real/imag were scaled, a mismatch of its own).
            scale = 1e6 * lal.PC_SI / Msun_m
            if not self.nu_rescale:
                scale /= metadata["nu"]
            this_mode = this_mode * scale

            self._hlm[(l, m)] = get_multipole_dict(this_mode)
            hlms = hlms.next

        # get the time array, transfored to physical units
        self._u = np.array(range(0, len(self._hlm[(2, 2)]["real"]))) * (dT / Msun_s)
        self._t = self.u

        pass

    def download_simulation(self, ID):
        """
        Download the simulation from the LVCNR catalog.
        """
        subprocess.run(
            ["git", "lfs", "pull", "--include", ID],
            cwd=os.path.join(self.lvcnr_path, self.catalog),
            check=True,
        )
        pass

    def _is_git_lfs_pointer(self, path):
        """
        Check if a file is still a Git LFS pointer (not downloaded)
        """

        try:
            with open(path, "r") as f:
                first_line = f.readline().strip()
                return first_line.startswith("version https://git-lfs.github.com")
        except UnicodeDecodeError:
            return False


if __name__ == "__main__":

    # test the class
    ID = "SXS_BBH_0007_Res5"
    path = "home/rox/Documents/projects/git_repos/lvcnr-lfs/SXS/"

    w = Waveform_LVKNR(path=path, ID=ID)
