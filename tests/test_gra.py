"""
Tests for the SXS catalog.
"""

from PyART.catalogs import gra
import os

mode_keys = ["A", "p", "real", "imag", "z"]


def test_gra():
    """
    Test the SXS download function.
    """
    wf = gra.Waveform_GRA(
        ID="0001",
        path="./",
        download=True,
        res="128",
        downloads=["hlm", "metadata"],
    )
    # check attributes
    assert wf.ID == "0001"

    # check that the files were downloaded
    assert os.path.exists("GRA_BHBH_0001")
    assert os.path.exists(f"GRA_BHBH_0001/metadata.json")
    assert os.path.exists(f"GRA_BHBH_0001/128/rh_CCE_GeometricUnits.h5")
    # check that the modes loaded make sense
    for mode in wf.hlm.keys():

        # check ell, emm
        assert mode[0] >= abs(mode[1])
        # check keys
        for key in mode_keys:
            assert key in wf.hlm[mode].keys()
        # check length
        assert len(wf.hlm[mode]["A"]) == len(wf.u)


if __name__ == "__main__":
    test_gra()
