"""
General tests for the waveform class in PyART
"""

from PyART import waveform

import numpy as np

def test_waveform():

    # Create a mock waveform
    wf = waveform.Waveform()

    # check that it has the right attributes
    for attr in ['hlm', 'u', 't', 'f', 'hp', 'hc', 'dothlm', 'psi4lm', 'dyn', 'kind']:
        assert hasattr(wf, attr), f"Waveform object does not have attribute {attr}"
    
    # check methods
    # ...

    pass
