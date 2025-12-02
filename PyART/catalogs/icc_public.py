import os, json, re, requests, h5py, logging, numpy
from pathlib import Path
import urllib3
from ..waveform import Waveform
from itertools import product

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def fetch_and_save_egrav_json(output_path="egrav_data.json"):
    """
    Download the JSON catalog of public ICC waveforms and save it locally.
    Parameters
    ----------
    output_path : str
        Path to save the JSON file.
    Returns
    -------
    data : list of dict
        List of entries in the catalog.
    """
    url = "https://egrav.icc.ub.edu/site/list-grid"
    response = requests.get(url, verify=False)
    response.raise_for_status()

    data = response.json()
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logging.info(f"Saved {len(data)} entries to {output_path}")
    return data


class Waveform_ICC(Waveform):
    """
    Class to download and handle public ICC waveforms.
    """

    def __init__(
        self,
        path="./",
        ID="0001",
        download=False,
        load=["hlm", "metadata"],
        ellmax=8,
        nu_rescale=False,
        extraction="extrap",  # extrapolated to r = ∞
    ):
        """
        Initialize the ICC waveform class.

        Parameters
        ----------
        path : str
            Base path to store/load the waveform data.
        ID : str or int
            Simulation ID, e.g., '0001' or 1.
        download : bool
            If True, download the data if not present.
        load : list of str
            List of data to load: 'hlm', 'metadata'.
        ellmax : int
            Maximum ell mode to load.
        nu_rescale : bool
            If True, rescale the waveform by the symmetric mass ratio nu.
        extraction : str or float
            Extraction radius in M, or 'extrap' for extrapolated to infinity.
        """
        super().__init__()
        if isinstance(ID, int):
            ID = str(ID).zfill(4)
        json_catalog = "egrav_data.json"

        if not os.path.exists(json_catalog):
            fetch_and_save_egrav_json()

        self.ID = ID
        self.path = path
        self.sim_path = os.path.join(self.path, "ICCUB-" + self.ID)
        self.ellmax = ellmax
        self.nu_rescale = nu_rescale
        self.download = download
        if extraction != "extrap":
            # Ensure it's formatted as a string like '100.00'
            extraction = f"{float(extraction):.2f}"

        self.extraction = extraction
        if self.download:
            if os.path.exists(self.sim_path):
                logging.info(
                    f"Directory {self.sim_path} already exists. Skipping download."
                )
            else:
                self.download_iccub_entry(json_path=json_catalog)

        if "metadata" in load:
            self.load_metadata()
        if "hlm" in load:
            self.load_hlm()

    def download_iccub_entry(self, json_path="egrav_data.json"):
        """
        Download the ICCUB simulation files for the given ID.
        Parameters
        ----------
        json_path : str
            Path to the local JSON catalog file.
        """
        uid = int(self.ID)
        if not os.path.exists(json_path):
            logging.info(f"JSON catalog {json_path} not found. Fetching...")
            fetch_and_save_egrav_json(json_path)

        with open(json_path, "r") as f:
            entries = json.load(f)

        # Match by UID (robust)
        match = next((e for e in entries if str(e["uid"]) == str(uid)), None)
        if not match:
            logging.warning(f"UID {uid} not found.")
            return

        outdir = Path(f"{self.sim_path}")
        outdir.mkdir(exist_ok=True)

        for key in ["metadata", "partfile", "h5"]:
            url = match.get(key)
            if not url or "fileId=" not in url:
                logging.warning(f"Skipping invalid {key} link for UID {uid}")
                continue

            file_id = url.split("fileId=")[-1]
            download_url = f"https://dataverse.csuc.cat/api/access/datafile/{file_id}"

            # HEAD request to extract filename (lighter)
            head = requests.head(download_url, allow_redirects=True, verify=False)
            if head.status_code != 200:
                logging.error(f"Failed HEAD for {key} (HTTP {head.status_code})")
                continue

            content_disposition = head.headers.get("Content-Disposition", "")
            filename = (
                content_disposition.split("filename=")[-1].strip("\"'")
                if "filename=" in content_disposition
                else f"{file_id}_{key}.bin"
            )

            outpath = outdir / filename
            if outpath.exists():
                logging.info(f"Already exists: {outpath}")
                continue

            logging.info(f"Downloading {key} for UID {uid} ...")
            r = requests.get(download_url, allow_redirects=True, verify=False)
            if r.status_code != 200:
                logging.error(f"Failed to download {key} (HTTP {r.status_code})")
                continue

            with open(outpath, "wb") as f:
                f.write(r.content)

    def load_hlm(self, load_m0=False):
        """
        Load hlm data from ICCUB simulation.

        Parameters
        ----------
        load_m0 : bool
            If True, load the m=0 modes as well.
        """
        modes = [
            (l, m)
            for l, m in product(
                range(2, self.ellmax + 1), range(-self.ellmax, self.ellmax + 1)
            )
            if (m != 0 or load_m0) and l >= numpy.abs(m)
        ]
        hlm = {}
        waveform_file = os.path.join(self.sim_path, f"{self.ID}_wf.h5")
        with h5py.File(waveform_file, "r") as f:
            self._u = f[f"t_r{self.extraction}"][:]
            for l, m in modes:
                name = f"h_l{l}_m{m}_r{self.extraction}"
                h = f[name][:]
                if self.nu_rescale:
                    h /= self.metadata["nu"]
                # Amplitude and phase
                amp, phase = abs(h), -numpy.unwrap(numpy.angle(h))
                hlm[(l, m)] = {
                    "real": amp * numpy.cos(phase),
                    "imag": amp * numpy.sin(phase),
                    "A": amp,
                    "p": phase,
                    "z": h,
                }
        self._hlm = hlm
        pass

    def load_psi4lm(self, load_m0=False):
        """
        Load psi4lm data from ICCUB simulation.

        Parameters
        ----------
        load_m0 : bool
            If True, load the m=0 modes as well.
        """
        modes = [
            (l, m)
            for l, m in product(
                range(2, self.ellmax + 1), range(-self.ellmax, self.ellmax + 1)
            )
            if (m != 0 or load_m0) and l >= numpy.abs(m)
        ]
        psi4lm = {}
        waveform_file = os.path.join(self.sim_path, f"{self.ID}_wf.h5")
        with h5py.File(waveform_file, "r") as f:
            self.u = f[f"t_r{self.extraction}"][:]
            for l, m in modes:
                name = f"psi4_l{l}_m{m}_r{self.extraction}"
                psi4 = f[name][:]
                if self.nu_rescale:
                    psi4 = psi4 / self.metadata["nu"]
                # amplitude and phase
                amp, phase = abs(psi4), -numpy.unwrap(numpy.angle(psi4))
                psi4lm[(l, m)] = {
                    "real": amp * numpy.cos(phase),
                    "imag": amp * numpy.sin(phase),
                    "A": amp,
                    "p": phase,
                    "z": psi4,
                }
        self._psi4lm = psi4lm
        pass

    def compute_psi4lm_from_hlm(self):
        """
        Compute psi4lm from hlm data.
        Requires that hlm data is already loaded.
        """
        psi4lm = {}
        t = self._u
        for key in self.hlm.keys():
            h = self.hlm[key]["z"]
            ddht = numpy.gradient(numpy.gradient(h, t), t)
            psi4lm[key] = {
                "A": abs(ddht),
                "phi": -numpy.unwrap(numpy.angle(ddht)),
                "z": ddht,
                "real": ddht.real,
                "imag": ddht.imag,
            }
        self._psi4lm = psi4lm
        pass

    def load_metadata(self):
        """
        Load metadata from ICCUB simulation.

        Returns
        -------
        meta : dict
            Dictionary with metadata information.
        """
        metadata_file = os.path.join(self.sim_path, f"{self.ID}_metadata.json")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file {metadata_file} not found.")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        q, M = metadata["q"], metadata["M_i"]
        nu = q / (1 + q) ** 2
        M1, M2 = metadata["M1_i"], metadata["M2_i"]
        meta = {
            "name": self.ID,
            "m1": M1,
            "m2": M2,
            "M": M,
            "q": q,
            "nu": nu,
            "chi1x": metadata["s1x"],
            "chi1y": metadata["s1y"],
            "chi1z": metadata["s1z"],
            "chi2x": metadata["s2x"],
            "chi2y": metadata["s2y"],
            "chi2z": metadata["s2z"],
            "S1": numpy.array([metadata["s1x"], metadata["s1y"], metadata["s1z"]])
            * M1**2,
            "S2": numpy.array([metadata["s2x"], metadata["s2y"], metadata["s2z"]])
            * M2**2,
            "e0": metadata["ecc"],
            "Mf": metadata["M_f"],
            "J_f": metadata["J_f"],
            "pos1": numpy.array([0, 0, +metadata["D"] / 2]),
            "pos2": numpy.array([0, 0, -metadata["D"] / 2]),
        }
        self.metadata = meta
        pass
