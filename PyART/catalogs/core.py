import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import h5py 
import glob
import os

from ..waveform import Waveform
from ..utils    import os_utils

## Conversion dictionary
conversion_dict_floats = {
    'database_key': 'name',
    'simulation_name': 'alternative_names',
    'id_mass': 'M',
    'id_rest_mass': 'Mb',
    'id_mass_ratio': 'q',
    'id_ADM_mass': 'E0',
    'id_ADM_angularmomentum': 'J0',
    'id_eos': 'EOS',
    'id_kappa2T': 'k2T',
    'id_eccentricity': 'ecc',
    'id_gw_frequency_Momega22': 'f0',
}

conversion_dict_vectors = {
    'id_spin_starA': 'S1',
    'id_spin_starB': 'S2',
    'id_Lambdaell_starA': 'Lambda_ell_A',
    'id_Lambdaell_starB': 'Lambda_ell_B',
}

def vector_string_to_array(vstr):
    return np.array([float(v) for v in vstr.split(',')])

class Waveform_CoRe(Waveform):

    def __init__(self,
                 path='../dat/CoRe/',
                 ID      = '0001',
                 run     =  None, # if None, select highest resolution
                 code    = 'BAM',
                 kind    = 'h5',
                 mtdt_path=None,
                 ell_emms='all',
                 download=False,
                 cut_at_mrg=False,
                 cut_junk  = None,
                 nu_rescale=False,
                 )->None:

        super().__init__()
        ID                  = code+f'_{ID:04}'
        self.ID             = ID
        self.run            = run
        self.ell_emms       = ell_emms
        self.core_data_path = os.path.join(path,ID)
        self.metadata       = None
        self.domain         = 'Time'
        self.nu_rescale     = nu_rescale

        if os.path.exists(self.core_data_path) == False:
            if download:
                print("The path ", self.core_data_path, " does not exist.")
                print("Downloading the simulation from the CoRe database.")
                self.download_simulation(ID=self.ID, path=path)
            else:
                print("Use download=True to download the simulation from the CoRe database.")
                raise FileNotFoundError(f"The path {self.core_data_path} does not exist.")
        
        self.simpath = self.core_data_path
        
        if self.run is None:
            # select the highest resolution (smaller grid_spacing_min)
            subdirs = os_utils.find_dirs_with_token(self.core_data_path, 'R0') # assume less than 10 runs
            most_accurate_run = None
            dx_min = 1e+6
            for subdir in subdirs:
                meta_Rfile = os.path.join(subdir,'metadata.txt')
                dx = None
                with open(meta_Rfile, 'r') as file:
                    for line in file:
                        if 'grid_spacing_min' in line:
                            _, value_str = line.strip().split('=')
                            break
                try:
                    dx = float(value_str)
                except Exception as e:
                    print(f'Error while reading grid_spacing_min from {meta_Rfile}: {e}')
                if dx is None:
                    continue
                if dx<dx_min:
                    dx_min = dx
                    most_accurate_run = subdir
            self.run     = most_accurate_run.replace(self.core_data_path+'/', '')
            self.runpath = most_accurate_run #os.path.join(self.simpath, self.run)
        else:
            self.runpath = os.path.join(self.simpath, self.run)
            
        # read metadata
        if mtdt_path is None:
            mtdt_path = os.path.join(self.simpath,'metadata_main.txt')
        self.metadata = self.load_metadata(mtdt_path)

        # read data
        self.load_hlm(kind = kind)
        
        # remove postmerger
        if cut_at_mrg:
            self.cut_at_mrg()
        
        if cut_junk:
            self.cut(cut_junk)

        pass

    def download_simulation(self, ID='BAM_0001', path='.',protocol='https',verbose=False):
        pre = {'ssh': 'git@', 'https': 'https://'}
        sep = {'ssh': ':'   , 'https': '/'}
        server = 'core-gitlfs.tpi.uni-jena.de'
        gitbase = 'core_database'
        if protocol not in pre.keys():
            raise NameError("Protocol not supported!")
        git_repo = '{}{}{}{}/{}.git'.format(pre[protocol],server,
                                        sep[protocol],gitbase,ID)
        print('git-clone {} ...'.format(git_repo))
        os.system('git clone '+git_repo)
        self.core_data_path = os.path.join(path,ID)
        os.system(f'mv {ID} {self.core_data_path}')
        # pull with lfs
        os.system('cd {}; git lfs pull'.format(self.core_data_path))

    def cut_at_mrg(self):
        """
        Find the global peak of the 22 and cut the waveform at this time + 10 M.
        Assuming that this is the merger time. For some wfs with postmerger this
        might not be true!
        """
        # find the peak of the 22 mode
        t_mrg, _, _, _, idx_cut = self.find_max(kind='global', return_idx=True)

        # cut all modes at the same index

        # Let's cut at mergr + 10 M     
        DeltaT = self.u[-1] - (t_mrg + 10)
        self.cut(DeltaT, from_the_end=True)
        pass

    def load_metadata(self, mtdt_path):
        metadata = {}
        with open(mtdt_path, "r") as f:
            lines = [l for l in f.readlines() if l.strip()] # rm empty
            for line in lines:
                if line[0]=="#": continue
                line               = line.rstrip("\n")
                key, val           = line.split("= ")
                key                = key.strip()
                if key in conversion_dict_floats.keys():
                    try:
                        metadata[conversion_dict_floats[key]] = float(val.strip())
                    except ValueError:
                        if key=='id_eccentricity':
                            print('Invalid id_eccentricity! Setting ecc=0.')
                            metadata[conversion_dict_floats[key]] = 0.
                        else:
                            metadata[conversion_dict_floats[key]] = val.strip()
                elif key in conversion_dict_vectors.keys():
                    metadata[conversion_dict_vectors[key]] = vector_string_to_array(val)
                else:
                    metadata[key] = val.strip()
        
        q  = float(metadata['q'])
        nu = q/(1+q)**2
        metadata['nu']    = nu
        metadata['J0']    = float(metadata['J0'])
        metadata['pph0']  = metadata['J0']/(nu*metadata['M']**2)
        metadata['E0byM'] = float(metadata['E0'])/metadata['M']
        metadata['chi1x'] = metadata['S1'][0]
        metadata['chi1y'] = metadata['S1'][1]
        metadata['chi1z'] = metadata['S1'][2]
        metadata['chi2x'] = metadata['S2'][0]
        metadata['chi2y'] = metadata['S2'][1]
        metadata['chi2z'] = metadata['S2'][2]
        metadata['LambdaAl2'] = metadata['Lambda_ell_A'][0]
        metadata['LambdaBl2'] = metadata['Lambda_ell_B'][0]
        metadata['f0'] = metadata['f0']/(2*np.pi)
        
        if metadata['LambdaAl2']>0 and metadata['LambdaBl2']>0:
            kind = 'BNS'
        elif metadata['LambdaAl2']<1 and metadata['LambdaBl2']<1:
            kind = 'BBH' # should not be present in CoRe
        else:
            kind = 'BHNS'
        metadata['flags'] = [kind] 
            
        return metadata
    
    def load_hlm(self, kind = 'h5'):
        if kind == 'txt':
            self.read_h_txt(self.runpath)
        elif kind == 'h5':
            self.read_h_h5(self.runpath)
        else:
            raise NameError('kind not recognized')
        
    def read_h_h5(self, basepath):
        """
        Read modes from the h5 file.
        Extract both the modes at ifinite radius
        and extrapolate to infinity using a K=1 polynomial
        """
        self.dfile = os.path.join(basepath,'data.h5')
        dset = {}
        nu   = self.metadata['nu']
        with h5py.File(self.dfile, 'r') as fn:
            for g in fn.keys():
                dset[g] = {}
                for f in fn[g].keys():
                    dset[g][f] = fn[g][f][()]
        
        self.hlm_fr = {}

        # identify all available modes in the keys
        for ky in dset.keys():
            if 'rh' in ky:
                ellemm   = ky.split('_')[-1]
                ell, emm = int(ellemm[0]), int(ellemm[1])
                self.hlm_fr[(ell, emm)] = {}
                
                # load all extraction radii
                A_xtp, p_xtp, r_xtp, t_xtp = [], [], [], []

                for rext in dset[ky].keys():
                    data = dset[ky][rext]
                    rext = float(rext.split('.')[0].split('_')[-1][1:])
                    r_xtp.append(rext)

                    try:
                        uM   = data[:,0]
                        RehM = data[:,1]
                        ImhM = data[:,2]
                        Momg = data[:,5]
                        aM   = data[:,6]
                        phi  = data[:,7]

                    except:
                        uM   = data[:,0]
                        RehM = data[:,1]
                        ImhM = data[:,2]
                        Momg = data[:,3]
                        aM   = data[:,4]
                        phi  = data[:,5]

                    self.hlm_fr[(ell, emm)][rext]= {'u': uM, 
                                                     'real': RehM, 
                                                     'imag': ImhM, 
                                                     'A': aM, 'p': phi}       
                    if self.nu_rescale: aM = aM/nu
                    A_xtp.append(aM)
                    p_xtp.append(phi)
                    t_xtp.append(uM)

                # extrapolate to infinite extraction radius
                # extrap phase and amp separately

                # common time grid
                t0   = max([min(tt) for tt in t_xtp])
                tf   = min([max(tt) for tt in t_xtp])
                dt   = min([np.diff(tt).min() for tt in t_xtp])
                tnew = np.linspace(t0, tf, int((tf-t0)/dt))

                res = []
                for yy in [A_xtp, p_xtp]:
                    # interpolate all on the same time grid
                    ynew = []
                    for i,y in enumerate(yy):
                        f = interpolate.interp1d(t_xtp[i], y)
                        ynew.append(f(tnew))
                    
                    res.append(radius_extrap_polynomial(ynew, r_xtp, 1))

                A_fre, p_fre = res
                # reconstruct complex waveform
                z_fre = A_fre*np.exp(-1j*p_fre)
                self.hlm[(ell, emm)] = {'A': A_fre, 'p': p_fre, 'z': z_fre, 
                                        'real': A_fre*np.cos(p_fre), 'imag': -A_fre*np.sin(p_fre)
                                        }
        self._u = tnew
        self._t = tnew
        
    def read_h_txt(self, basepath):
        # TODO: modify in case one has the txt files already
        # find all modes under basepath
        modes = glob.glob(basepath+'/Rh_l*.txt')

        r = []
        # find extraction radii
        for m in modes:
            r.append(m.split('/')[-1].split('_')[-1].split('.')[0])
        # find largest extraction radius
        r = list(set(r))
        imaxr = np.argmax([int(rr[1:]) for rr in r])

        # read modes
        d = {}
        for m in modes:
            
            this_r = m.split('/')[-1].split('_')[-1].split('.')[0]
            if this_r != r[imaxr]: continue
            
            ell = int(m.split('/')[-1].split('_')[1][1:])
            emm = int(m.split('/')[-1].split('_')[2][1:].split('.')[0])
            if self.ell_emms != 'all':
                if (ell, emm) not in self.ell_emms: continue
            d[(ell, emm)] = {}

            # u/M:0 Reh/M:1 Imh/M:2 Momega:3 A/M:4 phi:5 t:6
            u, re, im, Momg, A, phi, t = np.loadtxt(m, unpack=True, skiprows=3)
            d[(ell, emm)] ={
                'A': A,              'p': phi,
                't': t,              'real': re,             'imag': im
            }
        self._t = t
        self._u = u
        self._hlm = d
        pass

# stolen from watpy
def radius_extrap_polynomial(ys, rs, K):
    """
    Given different datasets yi, i=1...N, collected as
             ys = [y0, y1, y2, ... , yN]
    and array containing extraction radii
             rs = [r0, r1, r2, ... , rN],
    compute the asymptotic value of y as r goes to infinity from an Kth
    order polynomial in 1/r, e.g.

        yi = y_infty + \\sum_i=k^K ci / ri^k,

    where y_infty and the K coefficients ci are determined through a least
    squares polynomial fit from the above data.

    ys ... collection of data sets yi which all are of the same length,
           e.g. all sampled on the same grid u.
    rs ... extraction radii of the data samples yi
    K  ... maximum polynomial order of 1/r polynomial
    """
    import scipy as sp
    N = len(ys)
    if N != len(rs):
        raise ValueError("Mismatch in number of data sets ys and radii rs encountered!")
    L = len(ys[0])
    for i in range(1,N):
        if len(ys[i]) != L:
            raise ValueError("Inhomogenuous data set encountered! Check if all ys are sampled " *
                             "on the same grid")

    yinfty = np.zeros(L)
    # implementation relies on example given at 
    # https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.linalg.lstsq.html#scipy.linalg.lstsq
    M = np.array(rs)[:, np.newaxis]**(-np.array(range(K+1))) # inverse powers of rs
    for i in range(L):
        ys_i = [ ys[k][i] for k in range(N) ] # gather data for common radius
        p, *_ = sp.linalg.lstsq(M, ys_i)
        yinfty[i] = p[0] # zeroth coefficient equals value at r -> infty
        
    return yinfty
