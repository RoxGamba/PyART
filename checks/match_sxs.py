import sys,os,argparse,subprocess,matplotlib
import numpy as np
import matplotlib.pyplot as plt
from PyART.catalogs       import sxs
from PyART.models         import teob
from PyART.analysis.match import Matcher
from PyART.utils import utils as ut 

matplotlib.rc('text', usetex=True)


parser = argparse.ArgumentParser()

parser.add_argument('-id', '--sxs_id', default=180,              help="SXS ID")
parser.add_argument('--mass_min',       type=float, default=10,  help="Minimum mass (Msun)") 
parser.add_argument('--mass_max',       type=float, default=100, help="Minimum mass (Msun)") 
parser.add_argument('-n', '--mass_num', type=int,   default=10,  help="Number of masses")
parser.add_argument('--f1',             type=float, default=20,  help="Initial freq for mm")
parser.add_argument('--f2',             type=float, default=2048,help="Final freq for mm")
parser.add_argument('--taper_alpha',    type=float, default=0.01,help="Taper alpha")
parser.add_argument('--taper_start',    type=float, default=0.05,help="Taper start")
parser.add_argument('--taper_end',      type=float, default=0.00,help="Taper end")
parser.add_argument('-d', '--debug', action='store_true',        help="Show debug plots")
parser.add_argument('--no_plot', action='store_true',            help="Avoid mm-plot")
args = parser.parse_args()


repo_path = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], \
                             stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
sxs_id   = f'{args.sxs_id:04}' # e.g.0180
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

print('-----------------------------')
print(f'Mismatch for SXS:{sxs_id}')
print('-----------------------------')
print(f'q     : {q:.5f}')
print(f'M     : {M:.5f}')
print(f'chi1z : {chi1z:.5f}')
print(f'chi2z : {chi2z:.5f}')
print(f'f0    : {f0:.5f}')
print('-----------------------------')

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
masses = np.linspace(args.mass_min, args.mass_max, num=args.mass_num)
mm = masses*0.
for i, M in enumerate(masses):
    matcher = Matcher(nr, eob, pre_align=True,
                      settings={'kind':'single-mode', 
                                'initial_frequency_mm':args.f1,
                                'final_frequency_mm':args.f2,
                                'tlen':len(nr.u), 
                                'dt':1/srate, 
                                'M':M, 
                                'resize_factor':16, 
                                'modes-or-pol':'modes', 
                                'modes':[(2,2)],
                                'taper_alpha':args.taper_alpha,
                                'taper_start':args.taper_start,
                                'taper_end':args.taper_end,
                                'debug':args.debug,
                                }
                     )
    mm[i] = matcher.mismatch
    print(f'Mass, mm: {M:7.2f},  {mm[i]:.3e}')

if not args.no_plot and args.mass_num>1:
    plt.figure(figsize=(9,6))
    plt.plot(masses, mm)
    plt.yscale('log')
    plt.xlabel(r'$M_\odot$', fontsize=25)
    plt.ylabel(r'$\bar{\cal{F}}$', fontsize=25)
    plt.ylim(1e-4, 1e-1)
    plt.grid()
    plt.show()

