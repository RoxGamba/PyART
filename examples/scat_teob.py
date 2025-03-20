import PyART.models.teob as teob
from PyART.analysis.scattering_angle import ScatteringAngle
import matplotlib.pyplot as plt 
from PyART.utils.utils import D1
import numpy as np

configurations = [
    {'b': 9.678, 'chi1':0., 'chi2':0., 'E0':1.0226, 'p':0.11456439, 'eob':None, 'scat':None},
    {'b':10.000, 'chi1':0., 'chi2':0., 'E0':1.0226, 'p':0.11456439, 'eob':None, 'scat':None},
    {'b':11.000, 'chi1':0., 'chi2':0., 'E0':1.0226, 'p':0.11456439, 'eob':None, 'scat':None},
    {'b':12.000, 'chi1':0., 'chi2':0., 'E0':1.0226, 'p':0.11456439, 'eob':None, 'scat':None},
    {'b':13.000, 'chi1':0., 'chi2':0., 'E0':1.0226, 'p':0.11456439, 'eob':None, 'scat':None},
]             
n_conf = len(configurations)

r0 = 100
nu = 0.25

# generate EOB data and compute scattering angles
cutoff_min = 25
for i in range(n_conf):
    conf = configurations[i]
    J0 = conf['p']*conf['b']/nu
    eobpars = teob.CreateDict(r_hyp=r0, H_hyp=conf['E0'], J_hyp=J0, q=1, 
                             chi1z=conf['chi1'], chi2z=conf['chi2'])
    eob = teob.Waveform_EOB(pars=eobpars)
    conf['eob']  = eob
    E_final = eob.dyn['E'][-1]
    print(f"E0, pph0         : {conf['E0']:.5f}, {J0:.5f}")
    print(f"Final energy     : {E_final:.5f}")
    
    if E_final>1:
        scat = ScatteringAngle(puncts=eob.dyn, nmin=2, nmax=10, n_extract=None,
                               hypfit=True, hypfit_plot=False,
                               r_cutoff_in_low=cutoff_min,  r_cutoff_in_high=eob.dyn['r'][0],
                               r_cutoff_out_low=cutoff_min, r_cutoff_out_high=None, verbose=True)
        conf['scat'] = scat
    else:
        print('Capture!')

    print(' ')
del eob, scat

print('Plotting...')

plt.figure(figsize=(10,7))
for conf in configurations:
    
    eob  = conf['eob']
    scat = conf['scat']
    
    if scat is None:
        continue
    plt.subplot(2,2,1)
    plt.plot(eob.u, eob.hlm[(2,2)]['real'])
    plt.xlim([-100, 100]) 

    dh   = D1(eob.hlm[(2,2)]['z'], eob.u, 4)
    psi4 = D1(dh, eob.u, 4)

    b      = conf['b']
    chi_BH = conf['chi1']
    plt.subplot(2,2,2)
    plt.plot(eob.u, -psi4.real, label=f'{b:.3f}, {chi_BH:3}')
    plt.xlim([-100, 100]) 
    plt.legend()

    x = eob.dyn['r']*np.cos(eob.dyn['phi'])
    y = eob.dyn['r']*np.sin(eob.dyn['phi'])

    r = np.sqrt( x**2 + y**2 )
    plt.subplot(2,2,3)
    plt.plot(eob.dyn['t'],r)

    plt.subplot(2,2,4)
    plt.plot(x, y, label=f'{scat.chi:.2f}')
    plt.legend()

plt.show()




