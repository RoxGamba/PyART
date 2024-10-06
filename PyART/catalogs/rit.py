import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import h5py 
import glob
import os, json, requests, time
from bs4 import BeautifulSoup

from ..waveform import Waveform
from ..utils    import os_utils as ou
from .cat_utils import check_metadata

# This class is used to load the RIT data and store it in a convenient way
class Waveform_RIT(Waveform):

    def __init__(self,
                 path     = '../dat/RIT/',
                 ID       = '0001',
                 download = False,
                 basepath = None, # deprecated
                 psi_path = None, # deprecated
                 h_path   = None, # deprecated
                 mtdt_path= None, # deprecated
                 psi_load = True,
                 h_load   = True,
                 mtdt_load= True,
                 ell_emms = 'all',
                 ) -> None:
        
        super().__init__()

        self.t_psi4 = None
        self.t_h    = None
        self.ell_emms = ell_emms
        self.metadata      = None
        self.metadata_psi4 = None

        if isinstance(ID, int):
            ID = f'{ID:04}'
        self.ID = ID
        
        if basepath is not None:
            print("+++ Warning: 'basepath' is deprecated! Use 'path'")
            path = basepath

        sim_path = os.path.join(path, f'RIT_BBH_{ID}')
        if not os.path.exists(sim_path):
            if download:
                print(f"The path {sim_path} does not exist.")
                print("Downloading the simulation from the RIT catalog.")
                self.download_data(ID=ID, path=sim_path)
            else:
                print("Use download=True to download the simulation from the SXS catalog.")
                raise FileNotFoundError(f"The path {sim_path} does not exist.")
        self.sim_path = sim_path

        # metadata available
        if mtdt_path is not None:
            print('+++ Warning! Specifying mtdt_path in input is deprecated +++')
            self.mtdt_path = os.path.join(path,mtdt_path)
        else:
            self.mtdt_path = ou.find_fnames_with_token(self.sim_path, 'Metadata.txt')[0]

        if mtdt_load:
            self.metadata, self.ometadata = self.load_metadata(self.mtdt_path)

        # psi4 available
        if psi_path is not None:
            print('+++ Warning! Specifying psi_path in input deprecated +++')
            self.psi_path  = os.path.join(path,psi_path)
            self.mtdt_psi4 = os.path.join(path,psi_path,'Metadata')
        else:
            self.psi_path  = ou.find_dirs_with_token(self.sim_path, 'ExtrapPsi4')[0]
            self.mtdt_psi4 = os.path.join(path, self.psi_path, 'Metadata')
        
        if psi_load:
            _, self.metadata_psi4 = self.load_metadata(self.mtdt_psi4)
            self.load_psi4()
        
        # strain available
        if h_path is not None:
            print('+++ Warning! Specifying h_path in input is deprecated +++') 
            h_path = os.path.join(path,h_path)
        else:
            h_path = ou.find_fnames_with_token(self.sim_path, 'ExtrapStrain')[0]

        if h_load:
            self.h_file   = h5py.File(h_path, 'r')
            self.load_h()
        
        pass
    
    def download_data(self, ID, path='./', urls_json=None, dump_urls=True):

        def get_id_from_url(url,sep='-'):
            parts = url.split(sep)
            for i, part in enumerate(parts):
                if 'BBH' in part:
                    # 'cleaning' needed for Metadata*.txt fnames
                    next_part = parts[i+1]
                    next_part_cleaned = next_part.split('-')
                    return int(next_part_cleaned[0])
            return None
        
        file_path = os.path.dirname(__file__) # this is in the build/lib
        repo_path = file_path.split('PyART/')[0]
        script_path = os.path.join(repo_path, 'PyART/PyART/catalogs/')
        if urls_json is None: # use default name
           urls_json = os.path.join(script_path,'rit_urls.json')
        elif not isinstance(urls_json, str): 
            raise RuntimeError('Invalid value for urls_json: {urls_json}')

        if os.path.exists(urls_json):
            print(f'Loading urls from {urls_json}')
            with open(urls_json, 'r') as file:
                urls_dict = json.load(file)
        else:
            catalog_url = 'https://ccrgpages.rit.edu/~RITCatalog/'
            print('JSON file with RIT urls not found, fetching and parsing catalog webpage:', catalog_url)
            # fetch and parse catalog webpage
            response     = requests.get(catalog_url)
            html_content = response.text
            soup         = BeautifulSoup(html_content, 'html.parser')
            urls_dict = {}
            # loop on hyper references 
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'Data/' in href or 'Metadata/' in href:
                    # get ID from filename
                    if 'Metadata/' in href:
                        sep = ':'
                    else:
                        sep = '-'
                    rit_id_int = get_id_from_url(href, sep=sep)
                    key = f'{rit_id_int:04}'
                    # full url
                    sim_url = os.path.join(catalog_url, href)
                    # add to dictionary
                    if key not in urls_dict:
                        urls_dict[key] = [sim_url]
                    else:
                        urls_dict[key].append(sim_url)
            if dump_urls:
                with open(urls_json, 'w') as json_file:
                    json.dump(urls_dict, json_file, indent=4)
                print('Created JSON file with RIT urls:', urls_json)

        print('-'*50, f'\nDownloading RIT:BBH:{ID}\n', '-'*50, sep='')
        tstart = time.perf_counter()
        # ensure that the ID corresponds to an existing simulation
        if not ID in urls_dict:
            raise ValueError(f'No data found for ID:{ID}')
        # if everything fine, creat simulation-dir and download data
        os.makedirs(path, exist_ok=True)
        for href in urls_dict[ID]:
            ou.runcmd('wget '+href, workdir=path)
            if 'tar.gz' in href: # if compressed, untar
                elems = href.split('/')
                fname = elems[-1]
                ou.runcmd('tar -vxzf '+fname, workdir=path)
                ou.runcmd('rm -rv '+fname,    workdir=path) # remove compressed archive
                
                # check if the ExtrapPsi4* dir is in the correct level
                subdirs_ExtrapPsi4 = ou.find_dirs_with_subdirs(path, 'ExtrapPsi4')
                subdir = subdirs_ExtrapPsi4[0]
                if ou.is_subdir(path, subdir): # if wron level, move to upper one
                    tomove = os.path.join(subdir, 'ExtrapPsi4*')
                    ou.runcmd(f'mv -v {tomove} .', workdir=path)
                    ou.runcmd(f'rmdir {subdir}',   workdir=path)
        print('>> Elapsed time: {:.3f} s\n'.format(time.perf_counter()-tstart))
                     
        pass

    def load_psi4(self):

        files = glob.glob(os.path.join(self.psi_path, '*.asc'))
        d = {}
        
        if self.ell_emms == 'all': 
            modes = [(ell, emm) for ell in range(2,6) for emm in range(-ell, ell+1)]
        else:
            modes = self.ell_emms
        
        for ff in files:
            ell = int((ff.split('/')[-1]).split('_')[1][1:])
            emm = int((ff.split('/')[-1]).split('_')[2][1:])
            if (ell, emm) not in modes:
                continue
            try:
                t,re,im,A,p = np.loadtxt(ff, unpack=True, skiprows=4, usecols=(0,1,2,3,4) )
            except Exception:
                t,re,im,A,p,o = np.loadtxt(ff, unpack=True, skiprows=4, usecols=(0,1,2,3,4))
            d[(ell,emm)] = {'real':re, 'imag':im, 'A':A, 'p':p, 'h': A*np.exp(-1j*p)}
        
        self._psi4lm = d
        self.t_psi4 = t
        pass

    def load_h(self):

        d  = {}
        f  = self.h_file
        th = f['NRTimes'][:]
        if self.ell_emms == 'all': 
            modes = [(ell, emm) for ell in range(2,6) for emm in range(-ell, ell+1)]
        else:
            modes = self.ell_emms

        for mm in modes:
            ell, emm = mm
            try:
                A   =  f[f'amp_l{ell}_m{emm}']['Y'][:]
                A_u =  f[f'amp_l{ell}_m{emm}']['X'][:]
                p   = -f[f'phase_l{ell}_m{emm}']['Y'][:]
                p_u =  f[f'phase_l{ell}_m{emm}']['X'][:]
                # interp to common time array
                A   = self.__interp_qnt__(A_u, A, th)
                p   = self.__interp_qnt__(p_u, p, th)
                d[(ell, emm)] = {'real' : A*np.cos(p), 'imag': -A*np.sin(p), 'A':A, 'p':p, 'h': A*np.exp(-1j*p)}

            except KeyError:
                pass
                    
        self._hlm = d
        self.t_h  = th.astype(np.float64)
        self._t   = self.t_h
        self._u   = self.t_h
        pass

    def load_metadata(self, path):
        """
        load metadata for RIT catalog.
        Meta with original keys are stored in ometadata,
        metadata instead contain useful info with standard
        naming (see e.g. SXS metadata)
        """
        nm    = path
        ometa = {}
        with open(nm, 'r') as f:

            lines = [l for l in f.readlines() if l.strip()] # rm empty

            for line in lines[1:]:
                if line[0]=="#": continue
                line               = line.rstrip("\n")
                #line = line.split("#", 1)[0]
                key, val           = line.split("= ")
                key                = key.strip()
                ometa[key] = val
            
            kind = 'initial' # initial or relaxed (but no relaxed-separation in meta)
            if kind=='initial':
                ref_time = 0
            elif kind=='relaxed':
                ref_time = float(ometa['relaxed-time']) 
            M1   = float(ometa[f'{kind}-mass1'])
            M2   = float(ometa[f'{kind}-mass2'])
            q    = M2/M1
            
            def return_val_or_default(key,mydict,default=0.0):
                if key in mydict:
                    return float(mydict[key])
                else:
                    return default

            chi1x = return_val_or_default(f'{kind}-chi1x',ometa)
            chi1y = return_val_or_default(f'{kind}-chi1y',ometa)
            chi1z = return_val_or_default(f'{kind}-chi1z',ometa)
            chi2x = return_val_or_default(f'{kind}-chi2x',ometa)
            chi2y = return_val_or_default(f'{kind}-chi2y',ometa)
            chi2z = return_val_or_default(f'{kind}-chi2z',ometa)
            
            D  = float(ometa['initial-separation']) 
            f0 = float(ometa['freq-start-22'])/2 # FIXME

            af = float(ometa['final-chi'])
            meta = {'name'     : ometa['catalog-tag'], # i.e. store as name 'RIT:eBBH:1110'
                    'ref_time' : ref_time,
                    # masses and spins 
                    'm1'       : M1,
                    'm2'       : M2,
                    'M'        : M1+M2,
                    'q'        : M1/M2,
                    'nu'       : q/(1+q)**2,
                    'S1'       : np.array([chi1x,chi1y,chi1z])*M1*M1,
                    'S2'       : np.array([chi2x,chi2y,chi2z])*M2*M2,
                    'chi1x'    : chi1x,  # dimensionless
                    'chi1y'    : chi1y,
                    'chi1z'    : chi1z,
                    'chi2x'    : chi2x,  # dimensionless
                    'chi2y'    : chi2y,
                    'chi2z'    : chi2z,
                    # positions
                    'pos1'     : np.array([0,0,-D/2]),
                    'pos2'     : np.array([0,0, D/2]),
                    'r0'       : D, 
                    'e0'       : float(ometa['eccentricity']),
                     # frequencies
                    'f0v'      : np.array([0,0,f0]), # FIXME: only for spin-aligned
                    'f0'       : f0,
                    # ADM quantities (INITIAL, not REF)
                    'E0'       : float(ometa['initial-ADM-energy']),
                    'P0'       : None,
                    'J0'       : np.array([float(ometa['initial-ADM-angular-momentum-x']),
                                           float(ometa['initial-ADM-angular-momentum-y']),
                                           float(ometa['initial-ADM-angular-momentum-z'])]),
                    'Jz0'      : float(ometa['initial-ADM-angular-momentum-z']),
                    # remnant
                    'Mf'       : float(ometa['final-mass']),
                    'afv'      : np.array([0,0,af]),
                    'af'       : af,
                    }

        # check that all the required quantities are given 
        check_metadata(meta,raise_err=True) 
        
        return meta, ometa


    def compute_initial_data(self):
        """
        Compute initial data (J0, S0, L0) from the metadata
        """

        if self.metadata is not None:
            mtdt = self.metadata
        elif self.metadata_psi4 is not None:
            mtdt = self.metadata_psi4
        elif (self.metadata is None and self.metadata_psi4 is None):
            print("No metadata loaded")
            raise FileNotFoundError("No metadata read. Please load metadata first.")        

        try:
            chi1x = float(mtdt['initial-bh-chi1x']);  chi2x = float(mtdt['initial-bh-chi2x'])
            chi1y = float(mtdt['initial-bh-chi1y']);  chi2y = float(mtdt['initial-bh-chi2y'])
        except KeyError:
            chi1x = 0.;  chi2x = 0.
            chi1y = 0.;  chi2y = 0.
        chi1z = float(mtdt['initial-bh-chi1z']);  chi2z = float(mtdt['initial-bh-chi2z'])
        chi1  = np.array([chi1x, chi1y, chi1z])
        chi2  = np.array([chi2x, chi2y, chi2z])

        # masses
        m1 = float(mtdt['initial-mass1']);  m2 = float(mtdt['initial-mass2'])
        M  = m1 + m2
        X1 = m1/M; X2 = m2/M

        # Spin vectors
        S1 = chi1*m1**2
        S2 = chi2*m2**2

        # Ang momentum
        Jx = float(mtdt['initial-ADM-angular-momentum-x'])
        Jy = float(mtdt['initial-ADM-angular-momentum-y'])
        Jz = float(mtdt['initial-ADM-angular-momentum-z'])
        J  = np.array([Jx, Jy, Jz])

        E  = float(mtdt['initial-ADM-energy'])

        # Orb ang momentum
        L = J - S1 - S2

        self._dyn['id'] = {
                            'm1':m1, 'm2':m2, 'M':M, 'X1':X1, 'X2':X2, 
                            'S1':S1, 'S2':S2, 'L0':L,
                            'J0':J,
                            'E0':E
                        }
        pass

    def compute_dynamics(self):
        # compute dynamics
        pass
    
    def __interp_qnt__(self, x, y, x_new):

        f  = interpolate.interp1d(x, y)
        yn = f(x_new)

        return yn
    

class Catalog(object):
    def __init__(self, 
                 path    = './',
                 ell_emms    = 'all',
                 ellmax      = 4,
                 load_data   = False, # do not load wfs, just metadata
                 nonspinning = False, # load only nonspinning
                 integr_opts = None,
                 load_puncts = False,
                 verbose     = False
                 ) -> None:
        
        self.nonspinning  = nonspinning
        self.integr_opts  = integr_opts
        self.ellmax       = ellmax
        self.ell_emms     = ell_emms
        if self.ell_emms == 'all': 
            self.modes = [(ell, emm) for ell in range(2,self.ellmax+1) for emm in range(-ell, ell+1)]
        else:
            self.modes = self.ell_emms # TODO: add check on input

        self.data = []
        self.catalog_meta = []

        if integr_opts:
            raise NotImplementedError("Integration options not implemented yet")
        if load_puncts:
            raise NotImplementedError("Punctures not implemented yet")
        

        # load all simulations in path
        self.load_simulations_in_path(path, ell_emms, nonspinning, verbose=verbose)

    def load_simulations_in_path(self, path, ell_emms,
                                 nonspinning = False,
                                 eccentric   = True,
                                 verbose     = False
                                 ):
        """
        Load all simulations in path into self.data and all metadata into self.catalog_meta
        """
        import glob;
        h5s  = glob.glob(path + "Data/*h5")
        data = []

        for f in h5s:
            this_id   =  f.split('/')[-1].split('_')[1].split('-')[2]
            this_n    =  f.split('/')[-1].split('_')[1].split('-')[3].split('.')[0]
            if verbose:
                print('Processing:', this_id, this_n)
            if eccentric:
                h_path    = 'Data/ExtrapStrain_RIT-eBBH-'+this_id+'-'+this_n+'.h5'
                mtdt_path = 'Metadata/RIT:eBBH:'+this_id+'-'+this_n+'-ecc_Metadata.txt'
            else:
                raise NotImplementedError("Non eccentric not implemented yet")
            wave  = Waveform_RIT(h_path=h_path, path=path, mtdt_path=mtdt_path, ell_emms=ell_emms)
            mtdt  = wave.metadata
            chi1z = float(mtdt['initial-bh-chi1z']);  chi2z = float(mtdt['initial-bh-chi2z'])

            # checks: nonspinning
            try:
                chi1x = float(mtdt['initial-bh-chi1x']);  chi2x = float(mtdt['initial-bh-chi2x'])
                chi1y = float(mtdt['initial-bh-chi1y']);  chi2y = float(mtdt['initial-bh-chi2y'])
            except KeyError:
                chi1x = 0.;  chi2x = 0.
                chi1y = 0.;  chi2y = 0.
            chi1z = float(mtdt['initial-bh-chi1z']);  chi2z = float(mtdt['initial-bh-chi2z'])
            chi1  = np.array([chi1x, chi1y, chi1z]);
            chi2  = np.array([chi2x, chi2y, chi2z])
            if nonspinning:
                if np.linalg.norm(chi1)+np.linalg.norm(chi2)>1e-3:
                    continue

            # check: eccentric
            if eccentric and float(wave.metadata['eccentricity']) < 1e-2:
                continue

            sim_data            = lambda:0
            sim_data.meta       = wave.metadata
            sim_data.wave       = wave
            sim_data.tracks     = None
            sim_data.scat_info  = None
            self.catalog_meta.append(wave.metadata)
            data.append(sim_data)

        self.data = data

    def idx_from_value(self,value,key='name',single_idx=True):
        """ 
        Return idx with metadata[idx][key]=value.
        If single_idx is False, return list of indeces 
        that satisfy the condition
        """
        idx_list = []
        for idx, meta in enumerate(self.catalog_meta):
            if meta[key]==value:
                idx_list.append(idx)
        if len(idx_list)==0: 
            return None
        if single_idx:
            if len (idx_list)>1:
                raise RuntimeError(f'Found more than one index for value={value} and key={key}')
            else:
                return idx_list[0]
        else: 
            return idx_list

if __name__ == '__main__':

    psi_path= 'Psi4/ExtrapPsi4_RIT-eBBH-1634-n100-ecc/'
    h_path  = 'Strain/ExtrapStrain_RIT-eBBH-1634-n100.h5'

    mtdt_path = None
    r = Waveform_RIT(psi_path=psi_path, h_path=h_path, mtdt_path=mtdt_path)

    r.compute_initial_data()
    print(r.dyn['id'])

    # plot h22
    plt.plot(r.t_h, r.hlm[(2,2)]['real'])
    plt.plot(r.t_h, r.hlm[(2,2)]['A'])
    plt.show()

    # plot psi4
    for k in r.psi4lm.keys():
        plt.plot(r.t_psi4, r.psi4lm[k]['A'], label=k)
    plt.legend(ncol=3)
    plt.show()

    # symmetry between +m and -m
    plt.plot(r.t_h, r.hlm[(2,2)]['A'])
    plt.plot(r.t_h, r.hlm[(2,2)]['A']-r.hlm[(2,-2)]['A'])
    plt.show()

    # hierarchy of modes
    for k in r.hlm.keys():
        plt.plot(r.t_h, r.hlm[k]['A'], label=k)
    plt.legend(ncol=3)
    plt.show()
