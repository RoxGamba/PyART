"""
Perform some sanity checks on the class
for handling SXS waveforms
"""

import sys, os, subprocess
import matplotlib.pyplot as plt
import numpy as np
from PyART.catalogs.sxs import Waveform_SXS
import PyART.utils.utils as ut 

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

for key, value in sim1.metadata.items():
    print(f'{key:15s} : {value}')

# interpolate
u, hlm = sim1.interpolate_hlm(0.5)
h22   = hlm[(2,2)]['h']
phi22 = np.unwrap(np.angle(hlm[(2,2)]['h']))
omg22 = ut.D1(phi22, u, 4)

# plot
fig, axs = plt.subplots(2,2, figsize=(9,6))
axs[0,0].plot(sim1.u, sim1.hlm[(2,2)]['A'], label='cut_U=300')
axs[0,0].plot(sim2.u, sim2.hlm[(2,2)]['A'], label='cut_N=500', ls='--')
axs[0,1].plot(sim1.u, sim1.hlm[(2,2)]['real'], label='original')
axs[0,1].plot(u, h22.real, label='interp', ls='--')
axs[1,0].plot(u, phi22, label='interp')
axs[1,1].plot(u, omg22, label='interp')
axs[1,1].set_ylim([0,1])
for i in range(2):
    for j in range(2):
        axs[i,j].grid()
        axs[i,j].legend()
        axs[i,j].set_xlim(8e+3, u[-1])
plt.show()

if clean_local:
    runcmd('rm -rv '+test_path, cwd)





