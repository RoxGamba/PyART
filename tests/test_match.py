"""
Test the Matcher class with a NR waveform
"""

import numpy as np
import matplotlib.pyplot as plt
from PyART.catalogs import sxs
from PyART.analysis.match import Matcher

sxs_id = '0180'
nr   = sxs.Waveform_SXS(ID=sxs_id, download=True, ignore_deprecation=True)
nr_2 = sxs.Waveform_SXS(ID=sxs_id, download=False, ignore_deprecation=True)
nr.cut(300)
nr_2.cut(300)
M = 100
fmin    = 5
fmax    = 2048
srate   = 8192



def test_self_match_ell_emms():
    """
    Test that the self-match of the single modes
    is 1.0 within numerical accuracy.
    """

    settings={
            'kind':'single-mode',
            'initial_frequency_mm':fmin,
            'final_frequency_mm':fmax,
            'tlen':len(nr.u),
            'dt':1/srate,
            'M':M,
            'resize_factor':4,
            'modes-or-pol':'modes',
            'pad_end_frac':0.5,
            'taper_alpha':0.2,
            'taper_start':0.05,
            'taper':'sigmoid',
            'debug':False
            }

    for mode in nr.hlm.keys():
        these_settings = settings.copy()
        these_settings['modes'] = [mode]
        m = Matcher(nr, nr_2, settings=settings)
        match = 1 - m.mismatch
        assert np.isclose(match, 1.0, atol=1e-7)

def test_self_match_pol():
    """
    Test that the self-match of the two polarizations
    is 1.0 within numerical accuracy.
    """

    settings={
            'kind':'hm',
            'initial_frequency_mm':fmin,
            'final_frequency_mm':fmax,
            'tlen':len(nr.u),
            'dt':1/srate,
            'M':M,
            'resize_factor':4,
            'pad_end_frac':0.5,
            'taper_alpha':0.2,
            'taper_start':0.05,
            'taper':'sigmoid',
            'debug':True,
            }

    def test_self_match_pol_helper(cp, ep):
        """
        Helper function to test self-match for given
        coalescence phase and effective polarization.
        """

        settings_cp = settings.copy()
        settings_cp['coa_phase'] = [cp]
        settings_cp['eff_pols'] = [ep]

        m = Matcher(nr, nr_2, settings=settings_cp)
        match = 1 - m.mismatch
        assert np.isclose(match, 1.0, atol=1e-7)
        print(f'cp: {cp:.2f}, ep: {ep:.2f}, match: {match}')

    for cp in [0, np.pi/4, np.pi/2]:
        for ep in [0, np.pi/4, np.pi/2]:
            test_self_match_pol_helper(cp, ep)

    pass

