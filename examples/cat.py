import sys,os, subprocess
from PyART.catalogs.cataloger import Cataloger

repo_path = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], \
                             stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
path = os.path.join(repo_path, 'examples/local_data/rit/')

optimizer_opts = {'mm_settings'  : {'cut_second_waveform':True, 'final_frequency_mm':1024}, 
                  'overwrite'    : False,
                  'json_save_dyn': False,
                  'minimizer'    : {'kind':'dual_annealing', 'opt_maxfun':200},
                  'opt_max_iter' : 1,#6,
                  'opt_good_mm'  : 1e-4,#5e-3,
                  'bounds_iter'  : {'eps_initial': {'E0byM':1e-3, 'pph0':1e-2},
                                    'eps_factors': {'E0byM':2,    'pph0':2},
                                    'bad_mm'     : 1e-2,
                                    'max_iter'   : 3
                                   }
                  }
sim_list = list(range(1096,1111))
cat = Cataloger(path=path, catalog='rit', sim_list=sim_list, 
                           json_file='./examples/mismatches_rit_cat.json', 
                           add_opts={'download':True, 'nu_rescale':True})
#cat.plot_waves()
hlines = [5e-2,1e-2,1e-3]
cat.optimize_mismatches(optimizer_opts=optimizer_opts, nproc=4)
#cat.plot_colorbar(xvar='pph0', yvar='mm_opt', hlines=hlines, cvar='E0byM')
#cat.plot_colorbar(xvar='pph0', yvar='mm_opt', hlines=hlines, cvar='chiz_eff')
#cat.plot_colorbar(xvar='pph0', yvar='mm_opt', hlines=hlines, cvar='E0byM',    ranges={'pph0':[3,10]})
#cat.plot_colorbar(xvar='pph0', yvar='mm_opt', hlines=hlines, cvar='chiz_eff', ranges={'pph0':[3,10]})
cat.mm_vs_M(N=10, ranges={'pph0':[1,10]}, savepng=False, cmap_var='pph0', hlines=[5e-2, 1e-2, 1e-3], mass_min=20)
