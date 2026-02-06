import numpy as np
import os
import h5py
from ..waveform import Waveform
import json
import logging
import re
import time

# libraries for downloading
try:
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError as e:
    raise ImportError(
        "To use the GRA catalog, please install the required "
        "dependencies: requests, beautifulsoup4, urllib3"
    ) from e


class Waveform_GRA(Waveform):
    """
    Class to handle GRAthena++ waveforms.
    This is still under development, depending on the
    final format of the data, and as such still quite rough.
    For now, it assumes that the data is in the format
    of Alireza's simulations.
    """

    def __init__(
        self,
        ID="0001",
        path="../dat/GRA",
        ellmax=8,
        ext="ext",
        res="128",
        r_ext=None,
        cut_N=None,
        cut_U=None,
        nu_rescale=False,
        modes=[(2, 2)],
        download=False,
        downloads=["hlm", "metadata"],
    ):

        super().__init__()
        # Normalize ID to a 4-digit zero-padded string for consistency
        if isinstance(ID, int):
            ID = f"{ID:04d}"
        elif isinstance(ID, str) and ID.isdigit() and len(ID) < 4:
            ID = ID.zfill(4)
        self.ID = ID
        self.path = path
        self.cut_N = cut_N
        self.cut_U = cut_U
        self.modes = modes
        self.ellmax = ellmax
        self.extrap = ext
        self.domain = "Time"
        self.r_ext = r_ext
        self.nu_rescale = nu_rescale
        self.res = res
        # comment out the following for the moment

        if download:
            self.download_simulation(ID=ID, path=path, downloads=downloads, res=res)

        self.load_metadata()
        self.load_hlm(extrap=ext, ellmax=ellmax, r_ext=r_ext)
        pass

    def download_simulation(
        self,
        ID="0001",
        path=None,
        downloads=["hlm", "metadata"],
        res=None,
    ):
        """
        Automatically download and unpack a GRAthena++
        simulation from scholarsphere.
        """

        if path is None:
            path = self.path

        session = make_session()

        logging.info("Fetching catalog...")
        id_map = get_id_to_item_url(session)

        if ID not in id_map:
            raise RuntimeError(f"ID {ID} not found in catalog")

        item_url = id_map[ID]

        soup = get_item_soup(session, item_url)

        if "hlm" in downloads:
            logging.info("Downloading hlm data...")
            if res is None:
                res = "128"
                self.res = res
                logging.warning("No resolution specified, defaulting to res=128")

            filename, tar_url = find_tar_for_resolution(soup, res)
            logging.info(f"Found .tar: {filename}")
            logging.info(f"Downloading from: {tar_url}")
            download_safe(session, tar_url, filename)
            # untar, execute via os.system for the moment
            extract_path = os.path.join(path, f"GRA_BHBH_{ID}")
            os.makedirs(extract_path, exist_ok=True)
            logging.info(f"Extracting to: {extract_path}")
            os.system(f"tar -xf {filename} -C {extract_path}")
            os.remove(filename)

        if "metadata" in downloads:
            logging.info("Downloading metadata...")
            filename, meta_url = find_metadata_file(soup)
            logging.info(f"Found metadata file: {filename}")
            logging.info(f"Downloading from: {meta_url}")
            download_safe(session, meta_url, filename)
            # move to correct location
            extract_path = os.path.join(path, f"GRA_BHBH_{ID}", "metadata.json")
            os.makedirs(os.path.dirname(extract_path), exist_ok=True)
            os.rename(filename, extract_path)

        # Be polite to the server
        time.sleep(3)

    def load_metadata(self):
        """
        Load the metadata from the json file and store it in self.metadata
        """
        path = os.path.join(self.path, f"GRA_BHBH_{self.ID}", "metadata.json")
        ometa = json.load(open(path, "r"))

        m1 = float(ometa["initial-mass1"])
        m2 = float(ometa["initial-mass2"])
        M = m1 + m2
        q = m1 / m2
        nu = q / (1 + q) ** 2
        hS1 = ometa["initial-dimensionless-spin1"].strip('"').split(",")
        hS1 = np.array([float(hS1[i]) for i in range(3)])
        hS2 = ometa["initial-dimensionless-spin2"].strip('"').split(",")
        hS2 = np.array([float(hS2[i]) for i in range(3)])
        pos1 = ometa["initial-position1"].strip('"').split(",")
        pos1 = np.array([float(pos1[i]) for i in range(3)])
        pos2 = ometa["initial-position2"].strip('"').split(",")
        pos2 = np.array([float(pos2[i]) for i in range(3)])
        r0 = ometa["initial-separation"]
        P0 = ometa["initial-ADM-linear-momentum"].strip('"').split(",")
        P0 = np.array([float(P0[i]) for i in range(3)])
        L0 = ometa["initial-ADM-angular-momentum"].strip('"').split(",")
        L0 = np.array([float(L0[i]) for i in range(3)])

        metadata = {
            "name": ometa["simulation-name"],
            "ref_time": 0.0,
            # masses and spins
            "m1": m1,
            "m2": m2,
            "M": M,
            "q": q,
            "nu": nu,
            "S1": hS1 * m1 * m1,  # [M2]
            "S2": hS2 * m2 * m2,
            "chi1x": hS1[0],  # dimensionless
            "chi1y": hS1[1],
            "chi1z": hS1[2],
            "chi2x": hS2[0],  # dimensionless
            "chi2y": hS2[1],
            "chi2z": hS2[2],
            "LambdaAl2": 0.0,
            "LambdaBl2": 0.0,
            # positions
            "pos1": pos1,
            "pos2": pos2,
            "r0": r0,
            "e0": None,
            # frequencies
            "f0v": None,
            "f0": float(ometa["initial-orbital-frequency"]) / np.pi,
            # ADM quantities (INITIAL, not REF)
            "E0": float(ometa["initial-ADM-energy"]),
            "P0": P0,
            "J0": L0,
            "Jz0": L0[2],
            "E0byM": float(ometa["initial-ADM-energy"]) / M,
            "pph0": None,
            # remnant
            "Mf": None,
            "afv": None,
            "af": None,
        }

        self.metadata = metadata
        pass

    def load_hlm(self, extrap="ext", ellmax=None, load_m0=False, r_ext=None):
        """
        Load the data from the h5 file
        """
        if ellmax == None:
            ellmax = self.ellmax
        if r_ext == None:
            r_ext = "100.00"

        if extrap == "ext":
            h5_file = os.path.join(
                self.path,
                f"GRA_BHBH_{self.ID}",
                self.res,
                "rh_Asymptotic_GeometricUnits.h5",
            )
        elif extrap == "CCE":
            h5_file = os.path.join(
                self.path, f"GRA_BHBH_{self.ID}", self.res, "rh_CCE_GeometricUnits.h5"
            )
        elif extrap == "finite":
            h5_file = os.path.join(
                self.path,
                f"GRA_BHBH_{self.ID}",
                self.res,
                "rh_FiniteRadii_GeometricUnits.h5",
            )
        else:
            raise ValueError('extrap should be either "ext", "CCE" or "finite"')

        if not os.path.isfile(h5_file):
            raise FileNotFoundError(
                "No file found in the given path: {}".format(h5_file)
            )

        nr = h5py.File(h5_file, "r")
        if r_ext not in nr.keys():
            raise ValueError(
                "r_ext not found in the h5 file. Available values are: {}".format(
                    nr.keys()
                )
            )
        tmp_u = nr[r_ext]["Y_l2_m2.dat"][:, 0]

        self.check_cut_consistency()
        if self.cut_N is None:
            self.cut_N = np.argwhere(tmp_u >= self.cut_U)[0][0]
        if self.cut_U is None:
            self.cut_U = tmp_u[self.cut_N]

        self._u = tmp_u[self.cut_N :]
        self._t = self._u

        from itertools import product

        modes = [
            (l, m)
            for l, m in product(range(2, ellmax + 1), range(-ellmax, ellmax + 1))
            if (m != 0 or load_m0) and l >= np.abs(m)
        ]

        dict_hlm = {}

        for mode in modes:
            l = mode[0]
            m = mode[1]
            mode = "Y_l" + str(l) + "_m" + str(m) + ".dat"
            hlm = nr[r_ext][mode]
            h = hlm[:, 1] + 1j * hlm[:, 2]
            if self.nu_rescale:
                h /= self.metadata["nu"]
            # amp and phase
            Alm = abs(h)[self.cut_N :]
            plm = -np.unwrap(np.angle(h))[self.cut_N :]
            # save in dictionary
            key = (l, m)
            dict_hlm[key] = {
                "real": Alm * np.cos(plm),
                "imag": Alm * np.sin(plm),
                "A": Alm,
                "p": plm,
                "z": h[self.cut_N :],
            }
        self._hlm = dict_hlm
        pass

    def check_cut_consistency(self):
        if self.cut_N is not None and self.cut_U is not None:
            raise RuntimeError(
                "Conflict between cut_N and cut_U!\n"
                "When initializing, only one between cut_N and cut_U should be given in input.\n"
                "The other one is temporarly set to None and (consistently) updated in self.load_hlm()"
            )
        elif self.cut_N is None and self.cut_U is None:
            self.cut_N = 0
        pass

    def get_indices_dict(self):
        """
        Get the indices of the various cols in the data
        """
        # get col indices up to l=10
        indices_dict = {}
        col_indices = {}
        c = 0
        cstart = 2
        for l in range(2, 11):
            for m in range(-l, l + 1):
                col_indices[(l, m)] = (cstart + c, cstart + c + 1)
                c += 2
        # now store the ones that we need
        for mm in self.modes:
            re_idx = col_indices[mm][0]
            im_idx = col_indices[mm][1]
            indices_dict[mm] = {"t": 1, "re": re_idx, "im": im_idx}

        return indices_dict

    def load_psi4lm(
        self,
        ellmax=None,
        r_ext=None,
        extrap="ext",
        load_m0=False,
    ):
        """
        Load the data from the h5 file, but for psi4 instead of h.
        """
        if ellmax == None:
            ellmax = self.ellmax

        if r_ext == None:
            r_ext = "100.00"

        if extrap == "ext":
            h5_file = os.path.join(
                self.path,
                f"GRA_BHBH_{self.ID}",
                self.res,
                "rPsi4_Asymptotic_GeometricUnits.h5",
            )
        elif extrap == "CCE":
            h5_file = os.path.join(
                self.path,
                f"GRA_BHBH_{self.ID}",
                self.res,
                "rPsi4_CCE_GeometricUnits.h5",
            )
        elif extrap == "finite":
            h5_file = os.path.join(
                self.path,
                f"GRA_BHBH_{self.ID}",
                self.res,
                "rPsi4_FiniteRadii_GeometricUnits.h5",
            )
        else:
            raise ValueError('extrap should be either "ext", "CCE" or "finite"')

        if not os.path.isfile(h5_file):
            raise FileNotFoundError(
                "No file found in the given path: {}".format(h5_file)
            )

        nr = h5py.File(h5_file, "r")
        if r_ext not in nr.keys():
            raise ValueError(
                "r_ext not found in the h5 file. Available values are: {}".format(
                    nr.keys()
                )
            )
        tmp_u = nr[r_ext]["Y_l2_m2.dat"][:, 0]

        # self.check_cut_consistency()
        if self.cut_N is None:
            self.cut_N = np.argwhere(tmp_u >= self.cut_U)[0][0]
        if self.cut_U is None:
            self.cut_U = tmp_u[self.cut_N]

        self._u = tmp_u[self.cut_N :]
        self._t = self._u

        from itertools import product

        modes = [
            (l, m)
            for l, m in product(range(2, ellmax + 1), range(-ellmax, ellmax + 1))
            if (m != 0 or load_m0) and l >= np.abs(m)
        ]

        dict_psi4lm = {}
        for mode in modes:
            l = mode[0]
            m = mode[1]
            mode = "Y_l" + str(l) + "_m" + str(m) + ".dat"
            psi4lm = nr[r_ext][mode]
            psi4 = psi4lm[:, 1] + 1j * psi4lm[:, 2]
            if self.nu_rescale:
                psi4 /= self.metadata["nu"]
            Alm = abs(psi4)[self.cut_N :]
            plm = -np.unwrap(np.angle(psi4))[self.cut_N :]
            key = (l, m)
            dict_psi4lm[key] = {
                "real": Alm * np.cos(plm),
                "imag": Alm * np.sin(plm),
                "A": Alm,
                "p": plm,
                "z": psi4[self.cut_N :],
            }

        self._psi4lm = dict_psi4lm
        pass


# ----------------------------------------------------------------------
# Functions needed to download data from GRAthena++
# ----------------------------------------------------------------------

CATALOG_URL = (
    "https://scholarsphere.psu.edu/resources/610744ac-80b9-4689-8119-320dfd2e2b9a"
)
BASE_URL = "https://scholarsphere.psu.edu"


def make_session():
    session = requests.Session()

    retries = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0 Safari/537.36"
            ),
            "Accept": "*/*",
            "Accept-Encoding": "identity",  # avoids chunked/gzip resets
            "Connection": "keep-alive",
            "Referer": "https://scholarsphere.psu.edu/",
        }
    )

    return session


def get_id_to_item_url(session):
    r = session.get(CATALOG_URL, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    id_map = {}

    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        m = re.search(r"GRAthena:BHBH:(\d{4})", text)
        if m:
            id_map[m.group(1)] = urljoin(BASE_URL, a["href"])

    if not id_map:
        raise RuntimeError("No GRAthena IDs found on catalog page")

    return id_map


def get_item_soup(session, item_url):
    r = session.get(item_url, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def find_tar_for_resolution(item_soup, resolution):
    resolution = resolution.lower()

    for a in item_soup.find_all("a", href=True):
        href = a["href"].lower()
        text = a.get_text(strip=True).lower()
        if (
            "/downloads/" in href
            and text.endswith(".tar")
            and resolution in (href + text)
        ):
            filename = os.path.basename(href)
            return filename, urljoin(BASE_URL, a["href"])

    raise RuntimeError(f"No .tar found for resolution '{resolution}'")


def find_metadata_file(item_soup):
    for a in item_soup.find_all("a", href=True):
        href = a["href"].lower()
        text = a.get_text(strip=True).lower()
        if "/downloads/" in href and text.endswith(".json"):
            filename = os.path.basename(href)
            return filename, urljoin(BASE_URL, a["href"])

    raise RuntimeError(f"No metadata.json file found")


def download_safe(session, url, filename, chunk_size=1024 * 1024):
    tmp_file = filename + ".part"
    downloaded = 0

    if os.path.exists(tmp_file):
        downloaded = os.path.getsize(tmp_file)
        logging.info(f"Resuming download from byte {downloaded}")

    headers = {}
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"

    with session.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()

        mode = "ab" if downloaded > 0 else "wb"
        with open(tmp_file, mode) as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

    os.rename(tmp_file, filename)
    logging.info(f"Download completed")
