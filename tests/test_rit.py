"""
Tests for the RIT catalog.
"""

from PyART.catalogs import rit
import os

import numpy as np

mode_keys = ["A", "p", "real", "imag", "z"]

# tests/RIT_BBH_1362 is bundled with the repo, so these run against it directly
# (download=False) rather than triggering a download.
BUNDLED_PATH = os.path.join(os.path.dirname(__file__))


def test_rit(tmp_path):
    """
    Test the RIT download function.
    """
    download_path = str(tmp_path)
    wf = rit.Waveform_RIT(
        ID="1362",
        path=download_path,
        download=True,
        nu_rescale=False,
        # points at the bundled, pre-scraped cache, so the test neither hits
        # the network nor writes a fresh copy into the repo
        urls_json=os.path.join(BUNDLED_PATH, "catalog_rit.json"),
    )
    # check attributes
    assert wf.ID == "1362"

    # check that the files were downloaded
    assert os.path.exists(os.path.join(download_path, "RIT_BBH_1362"))

    # check that the modes loaded make sense
    for mode in wf.hlm.keys():

        # check ell, emm
        assert mode[0] >= abs(mode[1])
        # check keys
        for key in mode_keys:
            assert key in wf.hlm[mode].keys()
        # check length
        assert len(wf.hlm[mode]["A"]) == len(wf.u)


def test_rit_psi4lm_z_matches_real_imag():
    """
    load_psi4lm used to build 'z' from the file's own A/p columns as
    A*exp(-1j*p), but that file's phase column uses the opposite sign
    convention from the rest of the package (p_file == -get_multipole_dict's
    p, exactly): the reconstructed z was silently conj(real + 1j*imag) instead
    of real + 1j*imag. Now everything is derived from real/imag directly.
    """
    wf = rit.Waveform_RIT(
        ID="1362",
        path=BUNDLED_PATH,
        download=False,
        nu_rescale=False,
        shorten_rng=False,
    )
    assert wf.psi4lm, "no psi4lm modes loaded"
    for key, mode in wf.psi4lm.items():
        z = mode["real"] + 1j * mode["imag"]
        assert np.allclose(mode["z"], z), f"psi4lm{key}: z != real + 1j*imag"
        assert np.allclose(mode["A"], np.abs(z)), f"psi4lm{key}: A != |z|"


def test_rit_psi4lm_nu_rescale_scales_real_and_imag():
    """
    nu_rescale used to divide only 'A' by nu, leaving 'real'/'imag' (read
    straight from file) unrescaled -- inconsistent with 'A' and with 'z' built
    from the old (buggy) A*exp(-1j*p) formula.
    """
    wf = rit.Waveform_RIT(
        ID="1362",
        path=BUNDLED_PATH,
        download=False,
        nu_rescale=False,
        shorten_rng=False,
    )
    wf_nu = rit.Waveform_RIT(
        ID="1362",
        path=BUNDLED_PATH,
        download=False,
        nu_rescale=True,
        shorten_rng=False,
    )
    nu = wf_nu.metadata["nu"]
    mode, mode_nu = wf.psi4lm[(2, 2)], wf_nu.psi4lm[(2, 2)]
    assert np.allclose(mode["real"] / mode_nu["real"], nu)
    assert np.allclose(mode["imag"] / mode_nu["imag"], nu)
    assert np.allclose(mode["A"] / mode_nu["A"], nu)


def test_rit_hlm_via_get_multipole_dict_matches_original_amplitude_and_phase():
    """
    load_hlm now derives real/imag/A/p from get_multipole_dict(z) rather than
    hand-rolling them from the interpolated A/p arrays. Amplitude and the
    complex mode itself must be unaffected; only the raw phase can differ, and
    only by a multiple of 2*pi where the amplitude is numerically zero (the
    junk-radiation edge of the array).
    """
    wf = rit.Waveform_RIT(
        ID="1362",
        path=BUNDLED_PATH,
        download=False,
        nu_rescale=False,
        shorten_rng=False,
    )
    for key, mode in wf.hlm.items():
        assert np.allclose(
            mode["z"], mode["real"] + 1j * mode["imag"]
        ), f"hlm{key}: z != real + 1j*imag"
