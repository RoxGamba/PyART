"""
Tests for the SXS catalog.
"""

from PyART.catalogs import sxs
import os

mode_keys = ["A", "p", "real", "imag", "z"]


def test_sxs():
    """
    Test the SXS download function.
    """
    wf = sxs.Waveform_SXS(
        ID="0180",
        path="./",
        download=True,
        downloads=["hlm", "metadata", "horizons"],
        load=["hlm", "metadata", "horizons"],
        ignore_deprecation=True,
        level=4,
        order=2,
        nu_rescale=False,
    )
    # check attributes
    assert wf.ID == "0180"
    assert wf.level == 4

    # check that the files were downloaded
    assert os.path.exists("SXS_BBH_0180")
    assert os.path.exists(
        f"SXS_BBH_0180/Lev{wf.level}/rhOverM_Asymptotic_GeometricUnits_CoM.h5"
    )
    assert os.path.exists(f"SXS_BBH_0180/Lev{wf.level}/metadata.json")
    assert os.path.exists(f"SXS_BBH_0180/Lev{wf.level}/Horizons.h5")

    # check that the old folder was removed
    cache_dir = os.environ.get("SXSCACHEDIR")
    assert cache_dir, "SXSCACHEDIR environment variable is not set."
    flds = os.listdir(cache_dir)
    for fld in flds:
        if fld.startswith("SXS:BBH:0180"):
            assert False, f"Old folder {fld} still exists."

    # check that the modes loaded make sense
    for mode in wf.hlm.keys():

        # check ell, emm
        assert mode[0] >= abs(mode[1])
        # check keys
        for key in mode_keys:
            assert key in wf.hlm[mode].keys()
        # check length
        assert len(wf.hlm[mode]["A"]) == len(wf.u)

    # check that conversion to LVKNR works
    wf.to_lvk(modes=[(2, 2)])
