"""
Tests for the GRA catalog.
"""

import json
import logging
from pathlib import Path
import os

import numpy as np
import pytest

from PyART.catalogs import gra

logging.basicConfig(level=logging.INFO)

mode_keys = ["A", "p", "real", "imag", "z"]


def test_gra_live_metadata_download(tmp_path):
    """
    Test the live GRA metadata download path without pulling waveform archives.
    """
    path = Path(tmp_path)
    downloader = object.__new__(gra.Waveform_GRA)
    downloader.ID = "0001"
    downloader.path = str(path)
    downloader.res = "128"

    gra.Waveform_GRA.download_simulation(
        downloader,
        ID="0001",
        path=str(path),
        downloads=["metadata"],
        res="128",
    )

    metadata_path = path / "GRA_BHBH_0001" / "metadata.json"
    assert metadata_path.exists()

    raw_metadata = json.loads(metadata_path.read_text())
    assert raw_metadata["simulation-name"] == "GRAthena:BHBH:0001"

    gra.Waveform_GRA.load_metadata(downloader)
    assert downloader.metadata["name"] == "GRAthena:BHBH:0001"
    assert downloader.metadata["q"] > 0


@pytest.mark.slow
def test_gra(tmp_path):
    """
    Test the hlm and metadata download
    """
    path = Path(tmp_path)

    wf = gra.Waveform_GRA(
        ID="0001",
        path=str(path),
        download=True,
        res="128",
        downloads=["hlm", "metadata"],
        ext="ext",
        ellmax=2,
    )

    assert os.path.exists(path / "GRA_BHBH_0001" / "metadata.json")
    assert os.path.exists(
        path / "GRA_BHBH_0001" / "128" / "rh_Asymptotic_GeometricUnits.h5"
    )

    # check that the modes loaded make sense
    for mode in wf.hlm.keys():
        # check ell, emm
        assert mode[0] >= abs(mode[1])
        # check keys
        for key in mode_keys:
            assert key in wf.hlm[mode].keys()
        # check length
        assert len(wf.hlm[mode]["A"]) == len(wf.u)

        # 'imag' used to be built as A*sin(p), which is -Im(z) under this
        # package's p = -angle(z) convention (the same bug #10 had for SXS):
        # real + 1j*imag must reconstruct z exactly.
        m = wf.hlm[mode]
        assert np.allclose(m["real"] + 1j * m["imag"], m["z"])

    # p = -unwrap(angle(z)) grows positive over an inspiral for m>0: checked
    # on (2,2) only, since this system (q=1, non-spinning) suppresses every
    # other m>0 mode to numerical noise by symmetry, where phase carries no
    # physical meaning.
    if (2, 2) in wf.hlm:
        assert np.mean(wf.hlm[(2, 2)]["p"]) > 0


if __name__ == "__main__":
    test_gra(tmp_path=Path("/tmp/gra_test"))
