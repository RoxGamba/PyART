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
        'afv']            # af (3-vector)

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
            errwarn(f"Unknown key '{key}'! If you think this is correct, updated catalog_utils.py", raise_err=raise_err)
    pass






