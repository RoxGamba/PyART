import os, argparse, subprocess, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from PyART.analysis.opt_ic import Optimizer
from PyART.catalogs.sxs    import Waveform_SXS
from PyART.catalogs.rit    import Waveform_RIT
from PyART.analysis.match  import Matcher

matplotlib.rc('text', usetex=True)

repo_path = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], \
                             stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
sxs_path = os.path.join(repo_path, 'examples/local_sxs/')
rit_path = os.path.join(repo_path, 'examples/local_rit/')

mm_settings = {'cut':True, 'initial_frequency_mm':20, 'M':100}

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--catalog', required=True, type=str, 
                                 choices=['rit','sxs'],       help='Catalog')
parser.add_argument('-i', '--id', default=None,  type=int,    help='Simulatoion ID. If not specified, download hard-coded ID list')
parser.add_argument('-d', '--download', action='store_true',  help='Eventually download data')
parser.add_argument('--mm_vs_M',      action='store_true',    help='Show plot mm vs M plot')

args = parser.parse_args()

if args.catalog=='rit':
    ebbh = Waveform_RIT(path=rit_path, download=args.download, ID=args.id)
    Optimizer(ebbh, kind_ic='e0f0',   mm_settings=mm_settings, opt_bounds=[[0,0.7], [None, None]], debug=True)
    opt = Optimizer(ebbh, kind_ic='E0pph0', mm_settings=mm_settings, opt_bounds=[[0.9,1], [3.8,4.4]],    debug=True)

elif args.catalog=='sxs':
    ebbh = Waveform_SXS(path=sxs_path, download=args.download, ID=args.id, order="Extrapolated_N3.dir", ellmax=7)
    Optimizer(ebbh, kind_ic='e0f0',   mm_settings=mm_settings, opt_bounds=[[0,0.7], [None, None]])
    opt = Optimizer(ebbh, kind_ic='E0pph0', mm_settings=mm_settings, opt_bounds=[[0.9,1], [3.8,4.4]])

else:
    raise ValueError(f'Unknown catalog: {args.catalog}')

if args.mm_vs_M:
    masses = np.linspace(20, 200, num=19)
    mm = masses*0.
    for i, M in enumerate(masses):
        mm_settings['M'] = M
        matcher = Matcher(sxs_ebbh, opt.opt_Waveform, settings=mm_settings)
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
