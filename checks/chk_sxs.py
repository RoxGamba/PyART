"""
Perform some sanity checks on the class
for handling SXS waveforms
"""

import sys, os, subprocess
import matplotlib.pyplot as plt
from PyART.catalogs.sxs import Waveform_SXS

repo_path = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], \
                             stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')

def runcmd(cmd,workdir,out=None):
    """
    Execute cmd in workdir
    """
    base = os.getcwd()
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    os.system(cmd)
    os.chdir(base)
    return

clean_local = False
dashes      = '-'*50
cwd         = os.getcwd()
sxs_id      = '0180'
test_path   = os.path.join(repo_path, 'checks/local_sxs/')

#try: # Try to load waveform not available with download False
#    Waveform_SXS(path=test_path, ID=sxs_id, download=False)
#except Exception as e:
#    print('Error generated with download=False:')
#    print(e)

sim1 = Waveform_SXS(path=test_path, ID=sxs_id, download=True, ellmax=7, cut_U=300) # from u=300
sim2 = Waveform_SXS(path=test_path, ID=sxs_id, download=True, ellmax=7, cut_N=500) # from N=500

# plot
plt.figure()
plt.plot(sim1.t, sim1.hlm[(2,2)]['A'])
plt.plot(sim2.t, sim2.hlm[(2,2)]['A'], '--')
plt.grid()
plt.show()

if clean_local:
    runcmd('rm -rv '+test_path, cwd)





