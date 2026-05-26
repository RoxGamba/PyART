import numpy as np
from PyART.models.teob import CreateDict
from PyART.models.phenom import Waveform_IMRPhenomT


def test_generation():
    approximants = ["IMRPhenomT", "IMRPhenomTHM", "IMRPhenomTP", "IMRPhenomTPHM"]

    DL = 10  # Mpc
    M = 30  # Msun
    reference_frames = ["CP", "J", "L0"]
    pars = CreateDict(q=1.0, M=M, f0=0.003)
    pars["distance"] = DL
    for approx in approximants:
        for rf in reference_frames:
            if approx == "IMRPhenomTPHM" and rf != "CP":
                continue
            wvf = Waveform_IMRPhenomT(pars=pars, approx=approx, reference_frame=rf)
            assert wvf.hp is not None
            assert wvf.hc is not None
            if "HM" in approx:
                assert (2, 2) in wvf.hlm
            hp0 = np.array(wvf.hp)
            wvf.to_SI(M, DL)
            wvf.to_geom(M, DL)
            assert np.all(wvf.hp - hp0) < 1e-14
    pass
