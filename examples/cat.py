import sys,os, subprocess
from PyART.catalogs.cataloger import Cataloger

path = '/Users/simonealbanesi/repos/eob_generic_catalogs/data/rit'
optimizer_opts = {'mm_settings'  : {'cut_second_waveform':True, 'final_frequency_mm':1024}, 
                  'overwrite'    : False,
                  'opt_maxfun'   : 200,
                  'opt_max_iter' : 6,
                  'opt_good_mm'  : 5e-3,
                  'eps_max_iter' : 3,
                  'eps_factor'   : 2,
                  'eps_bad_mm'   : 0.05,
                  'lock_file'    : 'cat_lock_file.txt'
                  }
sim_list = list(range(1096,1111))
cat = Cataloger(path=path, catalog='rit', sim_list=sim_list, 
                           json_file='./examples/mismatches_rit.json', 
                           add_opts={'download':True})
#cat.plot_waves()
hlines = [5e-2,1e-2,1e-3]
cat.optimize_mismatches(optimizer_opts=optimizer_opts, nproc=4)
cat.plot_colorbar(xvar='pph0', yvar='mm_opt', hlines=hlines, cvar='E0byM')
cat.plot_colorbar(xvar='pph0', yvar='mm_opt', hlines=hlines, cvar='chiz_eff')
#cat.plot_colorbar(xvar='pph0', yvar='mm_opt', hlines=hlines, cvar='E0byM',    ranges={'pph0':[3,10]})
#cat.plot_colorbar(xvar='pph0', yvar='mm_opt', hlines=hlines, cvar='chiz_eff', ranges={'pph0':[3,10]})
cat.plot_mm_vs_M(N=10, ranges={'pph0':[1,10]}, savepng=False, cmap_var='pph0', hlines=[5e-2, 1e-2, 1e-3])
