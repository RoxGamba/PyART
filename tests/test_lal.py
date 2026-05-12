import numpy as np
from PyART.models.teob import CreateDict
from PyART.models.lal import Waveform_LAL

M = 10
q = 2
nu = q / (1 + q) ** 2
chi1z = 0.5
chi2z = -0.8
f0 = 0.003

def mode_to_k(ell, emm):
    return int(ell * (ell - 1) / 2 + emm - 2)


def modes_to_k(modes):
    return [mode_to_k(x[0], x[1]) for x in modes]

modes   = [(2,2),(2,1),(3,3),(4,4)]
modes_k = modes_to_k(modes)
eobpars = CreateDict(
    M=M, q=q, chi1z=chi1z, chi2z=chi2z, f0=f0, srate=8192, use_mode_lm=modes_k
)

def test_generation():
    app_kinds = {'IMRPhenomXPHM':'FD',
                 'IMRPhenomTHM' :'TD'}
    for approx in app_kinds:
        wvf = Waveform_LAL(pars=eobpars, approx=approx, kind=app_kinds[approx])
        assert wvf.hp is not None
        assert wvf.hc is not None

        assert wvf.units=='SI'
        hp_SI_0 = np.copy(wvf.hp)
        wvf.to_geom()
        
        assert wvf.units=='geom'
        wvf.to_SI()
        assert wvf.units=='SI'
        
        diff = wvf.hp-hp_SI_0
        assert np.all(np.abs(diff)<1e-14)
        
        wvf.to_geom()
        wvf.get_hlm(fmaxSI=4096, lmax=5)
        assert wvf.hlm is not None
        assert (2,2) in wvf.hlm
    pass     

