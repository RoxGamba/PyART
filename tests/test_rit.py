"""
Tests for the RIT catalog.
"""

from PyART.catalogs import rit
import os

mode_keys = ['A', 'p', 'real', 'imag', 'z']

def test_rit():
    """
    Test the SXS download function.
    """
    wf = rit.Waveform_RIT(  
                        ID        = '1362', 
                        path      = './', download=True,
                        nu_rescale= False,
                        urls_json='./catalog_rit.json'
                        )
    # check attributes
    assert wf.ID == '1362'

    # check that the files were downloaded
    assert os.path.exists('RIT_BBH_1362')

    # check that the modes loaded make sense
    for mode in wf.hlm.keys():
        
        # check ell, emm
        assert mode[0] >= abs(mode[1])
        # check keys
        for key in mode_keys:
            assert key in wf.hlm[mode].keys()
        # check length
        assert len(wf.hlm[mode]['A']) == len(wf.u)