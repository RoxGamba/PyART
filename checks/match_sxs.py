import sys,os,subprocess,matplotlib
import numpy as np
import matplotlib.pyplot as plt
from PyART.catalogs       import sxs
from PyART.models         import teob
from PyART.analysis.match import Matcher
from PyART.utils import utils as ut 

matplotlib.rc('text', usetex=True)

repo_path = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], \
                             stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')

sxs_id   = '0180'
sxs_path = os.path.join(repo_path, 'checks/local_sxs/')

# load (or download) SXS data 
nr = sxs.Waveform_SXS(path=sxs_path, download=True, ID=sxs_id, order="Extrapolated_N3.dir", ellmax=7)
nr.compute_hphc()

# read SXS-meta
q     = nr.metadata['q']
M     = nr.metadata['M']
chi1z = nr.metadata['chi1z']
chi2z = nr.metadata['chi2z']
f0    = nr.metadata['f0']

# compute corresponding EOB waveform
Tmax    = 1e+4
srate   = 8192
eobpars = { 
            'M'                  : M,
            'q'                  : q,
            'chi1'               : chi1z,
            'chi2'               : chi2z,
            'LambdaAl2'          : 0.,
            'LambdaBl2'          : 0.,
            'distance'           : 1.,
            'initial_frequency'  : f0/np.pi,
            'use_geometric_units': "yes",
            'interp_uniform_grid': "yes",
            'domain'             : 0,
            'srate_interp'       : srate,
            'inclination'        : 0.,
            'output_hpc'         : "no",
            'use_mode_lm'        : [1],     # List of modes to use
            'arg_out'            : "yes",   # output dynamics and hlm in addition to h+, hx
            'ecc'                : 1e-8,
            'r_hyp'              : 0.,
            'H_hyp'              : 0.,
            'j_hyp'              : 0.,
            'coalescence_angle'  : 0.,
            'df'                 : 1./128.,
            'anomaly'            : np.pi,
            'spin_flx'           : 'EOB',
            'spin_interp_domain' : 0
            }
eob = teob.Waveform_EOB(pars=eobpars)
eob.compute_hphc()

# compute (2,2) mismatches for different masses
masses = np.linspace(10, 200, num=20)
mm = masses*0.
for i, M in enumerate(masses):
    matcher = Matcher(nr, eob, settings={'kind':'single-mode', 'tlen':len(nr.u), 
                      'dt':1/srate, 'M':M, 'resize_factor':16, 'modes-or-pol':'modes', 'modes':[(2,2)] })
    mm[i] = matcher.mismatch
    print(f'Mass, mm: {M:7.2f},  {mm[i]:.3e}')

plt.figure(figsize=(9,6))
plt.plot(masses, mm)
plt.yscale('log')
plt.xlabel(r'$M_\odot$', fontsize=25)
plt.ylabel(r'$\bar{\cal{F}}$', fontsize=25)
plt.grid()
plt.show()

