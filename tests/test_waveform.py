"""
General tests for the waveform class in PyART
"""

from PyART import waveform
import copy
import numpy as np


def test_waveform():

    # Create a mock waveform
    wf = waveform.Waveform()

    # check that it has the right attributes
    for attr in ["hlm", "u", "t", "f", "hp", "hc", "dothlm", "psi4lm", "dyn", "kind"]:
        assert hasattr(wf, attr), f"Waveform object does not have attribute {attr}"

    # fill the waveform with some mock data
    re = np.array([1, 2, 3])
    im = np.array([4, 5, 6])
    h_dict = {
        "z": re + 1j * im,
        "A": np.sqrt(re**2 + im**2),
        "p": np.arctan2(im, re),
        "real": re,
        "imag": im,
    }
    wf._hlm[(2, 2)] = h_dict
    wf._psi4lm[(2, 2)] = copy.deepcopy(h_dict)
    wf._dothlm[(2, 2)] = copy.deepcopy(h_dict)
    wf._u = np.array([0, 1, 2])

    # check that multiplication and division by a factor works
    wf2 = wf * 2
    for var in ["hlm", "dothlm", "psi4lm"]:
        assert np.all(wf2.__getattribute__(var)[(2, 2)]["real"] == 2 * re)

    wfo2 = wf / 2
    for var in ["hlm", "dothlm", "psi4lm"]:
        assert np.all(wfo2.__getattribute__(var)[(2, 2)]["real"] == 0.5 * re)
    pass
