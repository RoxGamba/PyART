"""
Catalog utils
"""

import warnings 

# required keys 
KEYS = ['name', 
        'ref_time',       # reference time 
        'm1', 'm2', 'M', 
        'q', 'nu',  
        'S1', 'S2',       # spins (3-vector)
        'chi1x', 'chi2x', # dimensionless spin x-components 
        'chi1y', 'chi2y', # dimensionless spin y-components 
        'chi1z', 'chi2z', # dimensionless spin z-components
        'LambdaAl2', 
        'LambdaBl2',
        'r0',             # distance
        'e0',             # eccentricity 
        'E0', 'Jz0',      # ADM scalars 
        'P0', 'J0',       # ADM 3-vectors
        'pph0',           # Lz/(mu*M)
        'E0byM',          # E0/M 
        'pos1', 'pos2',   # position vectors
        'f0v',            # orbital freq (3-vector)
        'f0',             # orbital freq (float))
        'Mf', 'af',       # remnant properties, scalars (af dimensionless, float)
        'afv',            # af (3-vector)
        'scat_angle',     # scattering angle
        'flags',          # flags to categorize waveform (e.g. 'nonspinning', 'eccentric')
        ]

def errwarn(msg, raise_err=True):
    if raise_err:
        raise RuntimeError(msg)
    else:
        warnings.warn(msg)
    pass

def check_metadata(meta, raise_err=True):
    """
    Check that given metadata satisfies PyART-requirements 
    """
    # check that all required keys are in the dict
    for rkey in KEYS:
        if not rkey in meta:
            errwarn(f"Key '{rkey}' is required but not found in metadata!", raise_err=raise_err)
    for key in meta:
        if not key in KEYS:
            errwarn(f"Unknown key '{key}'! If you think this is correct, updated cat_utils.py", raise_err=raise_err)
    pass

def get_flags(meta, thres_spin=1e-5, thres_q=1e-3, thres_e0=1e-3):
    flags = []
    spins = [meta['chi1x'], meta['chi1y'], meta['chi1z'],
             meta['chi2x'], meta['chi2y'], meta['chi2z']]
    if all(abs(spins[i]) < thres_spin for i in range(6)):
        flags.append('nonspinning')
    elif all(abs(spins[i]) < thres_spin for i in [0, 1, 3, 4]):
        flags.append('spin-aligned')
    else:
        flags.append('spin-precessing')
    
    if abs(meta['q']-1)<thres_q:
        flags.append('equal-mass')
    else:
        flags.append('unequal-mass')
    
    if meta['E0byM']>=1:
        flags.append('hyperbolic')
    elif meta['e0'] is None:
        flags.append('elliptic')
    elif meta['e0']>thres_e0:
        flags.append('elliptic')
    else:
        flags.append('quasi-circular')

    return flags
        



