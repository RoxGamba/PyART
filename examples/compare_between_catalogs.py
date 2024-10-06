import numpy
import pandas as pd
import json
import re
from PyART.catalogs import sxs, gra
from PyART.analysis import match
from PyART.utils import utils, wf_utils, Msun
from rich.progress import track

# Function to convert string values with comparison operators to floats
def parse_eccentricity(value):
    if isinstance(value, str):
        if '<' in value:
            return float(value.replace('<', ''))
        else:
            return numpy.nan
    return value

# Load the data from JSON file
with open('catalog.json', 'r') as f:
    data = json.load(f)

# Create a DataFrame
df = pd.DataFrame(data)

# Apply the function to the 'reference_eccentricity' column
df['reference_eccentricity'] = df['reference_eccentricity'].apply(parse_eccentricity)

# Filter the DataFrame
chosen = df[df['object_types'] == 'BHBH']
chosen = chosen[(chosen['reference_mass_ratio'] < 4.1) &
                (chosen['reference_chi_eff'] < 1e-3) &
                (chosen['reference_chi_eff'] > 0) &
                (chosen['reference_chi1_perp'] < 1e-3) &
                (chosen['reference_chi2_perp'] < 1e-3) &
                (chosen['reference_eccentricity'] < 1e-3) &
                (chosen['reference_eccentricity'] > 1e-9) &
                (chosen['reference_mass_ratio'] < 4.0001) &
                (chosen['reference_mass_ratio'] > 3.9999)]

chosen.dropna(axis=1, how='all', inplace=True)
chosen.reset_index(inplace=True)

chosen = chosen[['reference_eccentricity', 'reference_mass_ratio', 
                 'reference_dimensionless_spin1', 'reference_dimensionless_spin2', 'name']]

# Extract numbers from strings
def extract_numbers(s):
    pattern = r'\d+'
    numbers = re.findall(pattern, s)
    return ''.join(numbers)

# Initialize GRA and SXS
gra_id = 'q3/384'
sxs_id = [extract_numbers(name) for name in chosen['name'].to_list()]

# Define parameters
mode_array = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3), (4, 2), (5, 5)]
gra_q = 3
extraction_method = 'CCE'

# Load GRA Simulations
print('Loading GRA Simulations')
wf_gra = gra.Waveform_GRA(path=f'/ligo/home/ligo.org/rossella.gamba/ali_data/{gra_id}', 
                          cut_U=100, ext=extraction_method, r_ext='50.00', q=gra_q)
wf_gra.compute_hphc(modes=mode_array)
omg22 = numpy.diff(wf_gra.hlm[(2, 2)]['p']) / numpy.diff(wf_gra.u)
f_mm = omg22[0] / (4 * numpy.pi) * max(m[1] for m in mode_array)

# Initialize mismatch array
mismatch_array = {}
base_sxs = './'

for id in sxs_id:
    mismatch_array[id] = {'total_mass': [], 'mismatch': [], 'f0s': []}
    print(f'Loading SXS Simulations for ID {id}')
    wf_sxs = sxs.Waveform_SXS(base_sxs, id, cut_N=300, download=True)
    wf_sxs.compute_hphc(modes=mode_array)

    for total_mass in track(numpy.linspace(30, 300, 10)):
        Mtch = match.Matcher(wf_sxs, wf_gra,
                             settings={'kind': 'hm',
                                       'M': total_mass,
                                       'initial_frequency_mm': 1.5 * f_mm / (total_mass * Msun),
                                       'final_frequency_mm': 1024.,
                                       'dt': 1. / 8192.,
                                       'iota': numpy.pi / 3.,
                                       'coa_phase': numpy.linspace(0, 2 * numpy.pi, 4),
                                       'eff_pols': numpy.linspace(0, numpy.pi, 4),
                                       'modes': mode_array,
                                       'resize_factor': 1,
                                       'taper_start': 0.05,
                                       'psd': 'txt',
                                       'asd_file': '/ligo/home/ligo.org/koustav.chandra/projects/XG/gwforge/GWForge/ifo/noise_curves/CE40-asd.txt',
                                       'debug': True})
        mismatch_array[id]['total_mass'].append(total_mass)
        mismatch_array[id]['mismatch'].append(Mtch.mismatch)
        mismatch_array[id]['f0s'].append(f_mm * 1.5 / (total_mass * Msun))
        print(f'M={total_mass} Msun, mismatch={Mtch.mismatch}, f0={f_mm * 1.5 / (total_mass * Msun)}')

# Save the mismatch array to a JSON file
with open('q3_mismatch_data.json', 'w') as f:
    json.dump(mismatch_array, f, indent=4)

print('Mismatch data saved to q3_mismatch_data.json')
