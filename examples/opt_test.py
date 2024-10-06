import os, subprocess
from PyART.analysis.opt_ic import Optimizer
from PyART.catalogs.sxs    import Waveform_SXS
from PyART.analysis.match  import Matcher

repo_path = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], \
                             stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
datapath = os.path.join(repo_path, 'examples/local_sxs/')

# ecc case
ID   = 1361
ebbh = Waveform_SXS(path=datapath, download=True, ID=ID, order="Extrapolated_N3.dir", ellmax=7)

mm_settings = {'cut':True, 'initial_frequency_mm':20}
opt  = Optimizer(ebbh, kind_ic='e0f0', mm_settings=mm_settings, opt_bounds=[[0,0.7], [None, None]])

mm_settings['debug'] = True
matcher = Matcher(ebbh, opt.opt_Waveform, settings=mm_settings)
print(f'Double-check, Matcher  : {matcher.mismatch:.3e}')
print(f'              Optimizer: {opt.opt_mismatch:.3e}')
