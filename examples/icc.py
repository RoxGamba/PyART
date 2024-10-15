import argparse, matplotlib
import matplotlib.pyplot as plt
from PyART.catalogs.icc import Waveform_ICC
from PyART.analysis.integrate_multipole import Multipole
from PyART.analysis.integrate_wave      import IntegrateMultipole
import numpy as np

matplotlib.rc('text', usetex=True)

path = '/Users/simonealbanesi/data/simulations_icc/ICCsims/catalog'
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--id', default=1,     type=int,     help='Simulatoion ID (-1 to show all captures)')
parser.add_argument('--f0',       default=0.002, type=float,   help='FFI f0')
parser.add_argument('--integration_test', action='store_true', help='Show integration test for new integrator')
args = parser.parse_args()

if args.id < 0:
    IDs = list(range(1,224))
else:
    IDs = [args.id]

for ID in IDs:
    icc = Waveform_ICC(path=path, ID=ID)

    integr_opts = {'f0':args.f0, 'extrap_psi4':True, 'method':'FFI', 'window':[20,-20]}
    l = 2
    m = 2
    M = 1

    if args.integration_test:
        t    = icc.t_psi4
        psi4 = icc.psi4lm[(2,2)]['h'] 
        dh   = icc.dothlm[(2,2)]['h'] 

        #t         = icc.u
        #signal    = dh    # psi4
        #integrand = 'news' # 'psi4'
        signal = psi4
        integrand = 'psi4'
        mode_new = IntegrateMultipole(l, m, t, signal, **integr_opts,
                                      mass=M, radius=icc.r_extr, integrand=integrand)
        #integr_opts['extrap_psi4'] = False
        mode_old = Multipole(l, m, t, signal, mass=M, radius=icc.r_extr, integrand=integrand)
        mode_old.hlm, mode_old.dothlm = mode_old.integrate_wave(integr_opts=integr_opts)

        plt.figure(figsize=(10,6))
        plt.subplot(3,2,1)
        plt.plot(t, mode_old.hlm.real, c='r', ls='-',  label='old')
        plt.plot(t, mode_new.h.real,   c='b', ls='--', label='new')
        plt.legend()
        plt.subplot(3,2,2)
        plt.plot(t, np.abs(mode_old.hlm - mode_new.h), c='g', label='diff')
        plt.yscale('log')
        plt.legend()
        plt.subplot(3,2,3)
        plt.plot(t, mode_old.dothlm.real, c='r', ls='-')
        plt.plot(t, mode_new.dh.real,     c='b', ls='--')
        plt.subplot(3,2,4)
        plt.plot(t, np.abs(mode_old.dothlm - mode_new.dh), c='g')
        plt.subplot(3,2,5)
        plt.plot(t, mode_old.psi.real,  c='r', ls='-')
        plt.plot(t, mode_new.psi4.real, c='b', ls='--')
        plt.subplot(3,2,6)
        plt.plot(t, np.abs(mode_old.psi - mode_new.psi4), c='g')
        plt.show()

    iicc = Waveform_ICC(path=path, ID=ID, integrate=True, 
                        integr_opts=integr_opts, load_puncts=True)
    #print( icc.integr_opts)
    #print(iicc.integr_opts)

    for k in icc.metadata:
        print(f'{k:10s} : {icc.metadata[k]}')

    try:
      iicc_tmrg, _, _, _ = iicc.find_max(height=0.3)
      icc_tmrg, _, _, _  = icc.find_max(height=0.3)
      
    except:
      iicc_tmrg = 0
      icc_tmrg  = 0      

    waves = ['hlm', 'dothlm', 'psi4lm']
    ylabs = [r'$h_{22}$', r'$\dot{h}_{22}$', r'$\psi_4^{22}$']
   
    npanels = len(waves)
    plt.figure(figsize=(9,6))
    plt.subplot(npanels,1,1)
    plt.title(f'ID:{ID:04}', fontsize=20)
    for i, wave_name in enumerate(waves):
        plt.subplot(npanels,1,i+1)
        if wave_name=='psi4lm':
            iicc_t = iicc.t_psi4
        else:
            iicc_t = iicc.u - iicc_tmrg
        icc_wave = getattr(icc,wave_name)
        if icc_wave:
            if wave_name=='psi4lm':
                icc_t  =  icc.t_psi4
            else:
                icc_t  =  icc.u  - icc_tmrg
            plt.plot(icc_t, icc_wave[(2,2)]['A'],    lw=0.5, c='r', label='stored')
            plt.plot(icc_t, icc_wave[(2,2)]['real'], lw=1.0, c='r', label=None)
            plt.plot(icc_t, icc_wave[(2,2)]['imag'], lw=1.0, c='r', ls=':', label=None)
        iicc_wave = getattr(iicc, wave_name)
        plt.plot(iicc_t, iicc_wave[(2,2)]['A'],    lw=0.5, c='b', label='integrated')
        plt.plot(iicc_t, iicc_wave[(2,2)]['real'], lw=1.0, c='b', label=None)
        plt.plot(iicc_t, iicc_wave[(2,2)]['imag'], lw=1.0, c='b', ls=':', label=None)
        plt.ylabel(ylabs[i], fontsize=20)
        if i==len(waves):
            plt.xlabel(r'$u$', fontsize=20)
        plt.legend()
    plt.show()

    if iicc.metadata['scat_angle'] is not None:
        plt.figure()
        plt.plot(iicc.puncts['x0'], iicc.puncts['y0'])
        plt.plot(iicc.puncts['x1'], iicc.puncts['y1'])
        plt.xlabel(r'$x$', fontsize=20)
        plt.ylabel(r'$y$', fontsize=20)
        plt.show()


