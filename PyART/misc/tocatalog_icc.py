"""
1) Spread metadata in each folder and assign also an id to each configuration
2) Store in dirs like ICC_BBH_????

SA: 07/31/2024
"""
import os, json
import numpy as np
from PyART.analysis.scattering_angle import ScatteringAngle

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

def load_puncts(sim_path, fname='puncturetracker-pt_loc..asc'):
    full_name = os.path.join(sim_path,fname)
    if os.path.exists(full_name):
        X  = np.loadtxt(full_name)
        t  = X[:, 8]
        x0 = X[:,22]
        y0 = X[:,32]
        x1 = X[:,23]
        y1 = X[:,33]
        x  = x0-x1
        y  = y0-y1
        r  = np.sqrt(x**2+y**2)
        th = -np.unwrap(np.angle(x+1j*y))
        zeros = 0*t
        pdict = {'t':t,   'r' :r,  'th':th,
                 'x0':x0, 'y0':y0, 'z0':zeros,
                 'x1':x1, 'y1':y1, 'z1':zeros}
    else:
        pdict = None
    return pdict

def parse_nr_metadata(file_path):
    metadata = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if '=' in line:
                key, value = line.split('=')
                key = key.strip()
                value = value.strip()
                try:
                    if '.' in value or 'e' in value.lower():
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    value = None
                metadata[key] = value
    return metadata

def count_subdirs(path):
    subdirs = []
    for entry in os.listdir(path):
        if entry.startswith('BBH'): 
            subdirs.append(entry)
    return subdirs

def get_info_from_name(name, field):
    val = None
    for elem in name.split('_'):
        if field in elem:
            theta_str = elem.replace(field,'')
            sign = 1.0 
            if theta_str[0]=='m': 
                sign = -1.0
                theta_str = theta_str[1:]
            elif theta_str[0]=='p':
                theta_str = theta_str[1:]
            val = sign*float(theta_str.replace('p', '.'))
            break
    return val

json_file = 'ICC_catalogue_optimized_metadata.json'
with open(json_file, 'r') as file:
    data = json.loads(file.read())

new_dir = 'catalog'
os.makedirs(new_dir, exist_ok=True)

sims_qeq1_nospin = count_subdirs('qeq1_nospin')
sims_qeq1_spin   = count_subdirs('qeq1_spin')
#sims_qdf1_nospin = count_subdirs('qdf1_nospin') 
sims_scat        = count_subdirs('scattering_ICC')

nsims_qeq1_nospin = len(sims_qeq1_nospin)
nsims_qeq1_spin   = len(sims_qeq1_spin)
#nsims_qdf1_nospin = len(sims_qdf1_nospin)
nsims_scat        = len(sims_scat)

#all_sims = sims_qeq1_nospin + sims_qeq1_spin + sims_qdf1_nospin + sims_scat
all_sims = sims_qeq1_nospin + sims_qeq1_spin + sims_scat

# load mergers, which have a global json with meta
for i, sim in enumerate(all_sims):
    ID = i+1
    meta          = {} 
    meta['name']  = f'ICC_BBH_{ID:04}' 
    meta['id']    = f'{ID:04}'
    meta['theta'] = get_info_from_name(sim, 'th')
    meta['b']     = get_info_from_name(sim, 'b')
    
    if sim in data:
        # info to store
        keys2store = ['q', 'chi1', 'chi2', 'E0', 'J0', 'ecc', 'D']
        for k in keys2store:
            meta[k] = data[sim][k]
        q    = meta['q']
        chi1 = meta['chi1']
        chi2 = meta['chi2']
        
        bool_q1     = abs(q-1)<1e-5
        bool_nospin = abs(chi1)<1e-10 and abs(chi2)<1e-10
        if not bool_q1:
            raise RuntimeError('q>1 are inaccurate!')
            datadir = 'qdf1_nospin'
        elif bool_nospin:
            datadir = 'qeq1_nospin'
        else:
            datadir = 'qeq1_spin'
        
        py = meta['J0']/meta['D']
        p  = py/np.sin(meta['theta']*np.pi/180)
        px = -np.sqrt(p**2 - py**2)
        meta['P0'] = [px,py,0]
        datasim = os.path.join(datadir, sim)
        meta['scat_angle']    = None
        meta['original_name'] = datasim

    else: # scatterings
        datadir = 'scattering_ICC' 
        datasim = os.path.join(datadir, sim)
        if not os.path.exists(datasim):
            raise RuntimeError(f'Something wrong carissimo: {datasim}')
        sim_path = os.path.join(datadir,sim)
        fname    = os.path.join(sim_path, 'TwoPunctures.bbh')
        bbh_meta = parse_nr_metadata(fname)        
        M1 = bbh_meta['initial-bh-puncture-adm-mass1']
        M2 = bbh_meta['initial-bh-puncture-adm-mass2']
        D  = abs(bbh_meta['initial-bh-position1x'] - bbh_meta['initial-bh-position2x'])
        M            = M1+M2
        meta['q']    = M1/M2
        meta['chi1'] = bbh_meta['initial-bh-spin1z']/M1**2
        meta['chi2'] = bbh_meta['initial-bh-spin2z']/M2**2
        meta['E0']   = bbh_meta['initial-ADM-energy']/M
        meta['J0']   = bbh_meta['initial-ADM-angular-momentumz']/M**2
        meta['ecc']  = None
        meta['D']    = D/M
        px = bbh_meta['initial-bh-momentum1x']/M
        py = bbh_meta['initial-bh-momentum1y']/M
        meta['P0'] = [px,py,0]
        
        puncts = load_puncts(sim_path)
        if puncts is not None and puncts['r'][-1]>25:
            scat   = ScatteringAngle(puncts=puncts,
                                 nmin=2, nmax=5, n_extract=4,
                                 r_cutoff_out_low=25, r_cutoff_out_high=None,
                                 r_cutoff_in_low=25, r_cutoff_in_high=100,
                                 verbose=False)
            meta['scat_angle'] = scat.chi
        else:
            meta['scat_angle'] = None

        meta['original_name'] = sim_path

    fname = os.path.join(datasim, 'metadata.json')
    with open(fname, 'w') as file:
         file.write(json.dumps(meta, indent=2))
    print(f'#{ID:04} created file: {fname}')
    
    new_sim_dir = os.path.join(new_dir, meta['name'])
    os.makedirs(new_sim_dir, exist_ok=True)
    cmd = f'cp -v {datasim}/* {new_sim_dir}'
    runcmd(cmd, workdir=os.getcwd())
    print(' ')

