import os, argparse, subprocess, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from PyART.analysis.opt_ic import Optimizer
from PyART.catalogs.sxs    import Waveform_SXS
from PyART.catalogs.rit    import Waveform_RIT
from PyART.catalogs.icc    import Waveform_ICC
from PyART.analysis.match  import Matcher

matplotlib.rc('text', usetex=True)

repo_path = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], \
                             stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
sxs_path = os.path.join(repo_path, 'examples/local_sxs/')
#rit_path = os.path.join(repo_path, 'examples/local_rit/')
rit_path = '/Users/simonealbanesi/repos/eob_generic_catalogs/data/rit/' 
icc_path = '/Users/simonealbanesi/data/simulations_icc/ICCsims/catalog'


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--catalog', required=True, type=str, 
                                 choices=['rit','sxs', 'icc'],   help='Catalog')
parser.add_argument('-i', '--id', default=1,         type=int,   help='Simulatoion ID. If not specified, download hard-coded ID list')
parser.add_argument('--maxfun',       default=100,   type=int,   help='Maxfun in dual annealing')
parser.add_argument('--opt_max_iter', default=1,     type=int,   help='Max opt iter')
parser.add_argument('--opt_good_mm',  default=5e-3,  type=float, help='opt_good_mm from opt_ic')
parser.add_argument('--eps_max_iter', default=1,     type=int,   help='ep_max_iter from opt_ic')
parser.add_argument('--eps_bad_mm',   default=0.1,   type=float, help='eps_bad_mm from opt_ic')
parser.add_argument('-d', '--download', action='store_true',     help='Eventually download data')
parser.add_argument('--kind_ic', choices=['e0f0', 'E0pph0'], 
                                 default='E0pph0',               help='ICs type')
parser.add_argument('--debug_plot',   action='store_true',       help='Show debug plot')
parser.add_argument('--mm_vs_M',      action='store_true',       help='Show plot mm vs M')
parser.add_argument('--json_file',    default=None,              help='JSON file for mismatches')
parser.add_argument('--overwrite',    action='store_true',       help='Overwrite option (json)')
parser.add_argument('--taper_alpha',    type=float, default=1.00, help="Taper alpha")
parser.add_argument('--taper_start',    type=float, default=0.05, help="Taper start")
args = parser.parse_args()

mm_settings = {'cut_second_waveform':True, 'initial_frequency_mm':20, 'M':100, 'final_frequency_mm':1024,
               'taper_alpha':args.taper_alpha, 'taper_start':args.taper_start}
mm_settings['initial_frequency_mm'] = 10

if args.kind_ic=='e0f0':
    bounds = [[0,0.7], [None, None]]
else:
    bounds = [[None,None], [None,None]]

if args.catalog=='rit':
    ebbh = Waveform_RIT(path=rit_path, download=args.download, ID=args.id, nu_rescale=True)

elif args.catalog=='sxs':
    ebbh = Waveform_SXS(path=sxs_path, download=args.download, ID=args.id, order="Extrapolated_N3.dir", ellmax=7,  nu_rescale=True)

elif args.catalog=='icc':
    ebbh = Waveform_ICC(path=icc_path, ID=args.id, integrate=True, nu_rescale=True, 
                        integr_opts={'f0':0.002, 'extrap_psi4':True, 'method':'FFI', 'window':[20,-20]})
else:
    raise ValueError(f'Unknown catalog: {args.catalog}')

print('Metadata:')
for k in ebbh.metadata:
    print(f'{k:10s} : {ebbh.metadata[k]}')
print('\n\n')

if args.catalog=='sxs':
    print('Removing (200 M) junk for SXS')
    ebbh.cut(200)

opt = Optimizer(ebbh, kind_ic=args.kind_ic, mm_settings=mm_settings,
                      opt_maxfun=args.maxfun,
                      opt_max_iter = args.opt_max_iter,
                      opt_good_mm  = args.opt_good_mm,
                      opt_bounds   = bounds, 
                      eps_max_iter = args.eps_max_iter, 
                      eps_bad_mm   = args.eps_bad_mm,
                      debug=args.debug_plot,
                      json_file=args.json_file, overwrite=args.overwrite)

if args.mm_vs_M:
    masses = np.linspace(20, 200, num=19)
    mm = masses*0.
    for i, M in enumerate(masses):
        mm_settings['M'] = M
        matcher = Matcher(ebbh, opt.opt_Waveform, settings=mm_settings)
        mm[i] = matcher.mismatch
        print(f'mass:{M:8.2f}, mm: {mm[i]:.3e}')
    plt.figure(figsize=(9,6))
    plt.plot(masses, mm)
    plt.yscale('log')
    plt.xlabel(r'$M_\odot$', fontsize=25)
    plt.ylabel(r'$\bar{\cal{F}}$', fontsize=25)
    plt.ylim(1e-4, 1e-1)
    plt.grid()
    plt.show()

    #mm_settings['debug'] = True
    #matcher = Matcher(ebbh, opt.opt_Waveform, settings=mm_settings)
    #print(f'Double-check, Matcher  : {matcher.mismatch:.3e}')
    #print(f'              Optimizer: {opt.opt_mismatch:.3e}')
