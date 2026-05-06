import numpy as np
from PyART.models.teob import CreateDict
from PyART.models.lal  import Waveform_LAL

M     = 10 
q     = 2 
nu    = q/(1+q)**2
chi1z = 0.5
chi2z = -0.8
f0    = 0.003 

def mode_to_k(ell, emm):
    return int(ell * (ell - 1) / 2 + emm - 2)
def modes_to_k(modes):
    return [mode_to_k(x[0], x[1]) for x in modes]

modes   = [(2,2),(2,1),(3,3),(3,2),(4,4)]
modes_k = modes_to_k(modes)
eobpars = CreateDict(M     = M, 
                     q     = q,
                     chi1z = chi1z,
                     chi2z = chi2z,
                     f0    = f0,
                     srate = 8192,
                     use_mode_lm = modes_k)

def test_generation_FD():
    lalFpc = Waveform_LAL(pars=eobpars, approx='IMRPhenomXPHM', kind='FD', get_hlm=False)
    assert lalFpc.hp is not None
    assert lalFpc.hc is not None

    assert lalFpc.units=='SI'
    lalFpc.to_geom()
    assert lalFpc.units=='geom'
    
    lalFlm = Waveform_LAL(pars=eobpars, approx='IMRPhenomXHM', kind='FD', get_hlm=True)
    lalFlm.to_geom()
    assert lalFlm.hlm is not None
    
    return lalFpc, lalFlm  

def test_generation_TD():
    lalTpc = Waveform_LAL(pars=eobpars, approx='IMRPhenomTHM',  kind='TD', get_hlm=False)
    lalTpc.to_geom()
    assert lalTpc.hp is not None
    assert lalTpc.hc is not None
    
    lalTlm = Waveform_LAL(pars=eobpars, approx='IMRPhenomTHM',  kind='TD', get_hlm=True)
    lalTlm.to_geom()
    assert lalTlm.hlm is not None

    return lalTpc, lalTlm  

if __name__=='__main__':
    
    lalFpc, lalFlm = test_generation_FD()
    lalTpc, lalTlm = test_generation_TD()

    import matplotlib.pyplot as plt
    from PyART.models.teob import Waveform_EOB
    
    plt.figure
    plt.plot(lalFpc.f, np.abs(lalFpc.hp+1j*lalFpc.hc))
    for lm in [(2,2),(2,1),(3,3),(4,4)]:
        plt.plot(lalFlm.f, lalFlm.hlm[lm]['A'])
    plt.yscale('log')
    plt.show()

    tmax,_,_,_ = lalTlm.find_max()
    eob = Waveform_EOB(eobpars)
    plt.figure
    plt.plot(lalTlm.u-tmax, lalTlm.hlm[(2,2)]['A'], c='b', label='IMRPhenomTHM')
    plt.plot(eob.u, eob.hlm[(2,2)]['A']*nu, ls='--', c='r', label='TEOB')
    plt.ylabel(r'$A_{22}$', fontsize=16)
    plt.legend()
    plt.show()



