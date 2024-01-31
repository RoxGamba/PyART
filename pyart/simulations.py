# Tools to manage metadata of NR simulation from various catalogs
# Loads metadata files and store a large list of dictionaries, one per simulation
#
# SB 2023.05.24 (started)

import sys, os, re, datetime
import glob, itertools
import json, csv
import numpy as np; import math
import subprocess

# ---------------------
# Basic unit conversion
# --------------------- 

ufact= {
    'Msun_sec': 4.925794970773135e-06,
    'Msun_meter': 1.476625061404649406193430731479084713e3,
}

class geom:
    """
    Geometric units + Msun = 1
    """
    grav_constant       = 1.0
    light_speed         = 1.0
    solar_mass          = 1.0
    MeV                 = 1.0
    
class cgs:
    """
    CGS units
    """
    grav_constant       = 6.673e-8
    light_speed         = 29979245800.0
    solar_mass          = 1.988409902147041637325262574352366540e33
    MeV                 = 1.1604505e10

class metric:
    """
    Standard SI units
    """
    grav_constant       = 6.673e-11
    light_speed         = 299792458.0
    solar_mass          = 1.988409902147041637325262574352366540e30
    MeV                 = 1.1604505e10


# --------------
# bash interface
# --------------

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

# ------------------
# ADM to EOB trafo
# ------------------

#TODO: test this
def adm_to_eob(q_vec, p_vec, nu):
    """
    Convert q_ADM, p_ADM to EOB, 2PN trafo.
    (E18) and (E19) of https://arxiv.org/pdf/1210.2834.pdf

    q_vec: ADM puncture relative distance, x_p - x_m, cartesian
    p_vec: ADM puncture relative linear momentum, p_p, cartesian, nu-normalized
    nu   : symmetric mass ratio
    """

    # shorthands
    q_vec   = np.array(q_vec)
    p_vec   = np.array(p_vec)

    q2      = np.dot(q_vec, q_vec) # x, y
    q       = np.sqrt(q2)
    q3      = q*q2
    p2      = np.dot(p_vec, p_vec)
    p       = np.sqrt(p2)
    p3      = p*p2
    p4      = p*p3
    qdotp   = np.dot(q_vec, p_vec)
    qdotp2  = qdotp*qdotp
    nu2     = nu*nu

    # coefficients for coordinates
    cQ_1PN_q = -nu/2*p2 + 1/q*(1 + nu/2)
    cQ_1PN_p = -qdotp*nu
    cQ_2PN_q = nu/8*(1 - nu)*p4 + nu/4*(5 - nu/2)*p2/q + nu*(1 + nu/8)*qdotp2/q3 + 1/4*(1 - 7*nu + nu2)/q2
    cQ_2PN_p = qdotp*(nu/2*(1 + nu)*p2 + 3/2*nu*(1 - nu/2)/q)
    # coefficients for momenta
    cP_1PN_q = qdotp/q3*(1 + nu/2)
    cP_1PN_p = nu/2*p2 - 1/q*(1 + nu/2)
    cP_2PN_q = qdotp/q3*(nu/8*(10 - nu)*p2 + 3/8*nu*(8 + 3*nu)*qdotp2/q2 + 1/4*(-2 - 18*nu + nu2)/q)
    cP_2PN_p = nu/8*(-1 + 3*nu)*p4 - 3/4*nu*(3 + nu/2)*p2/q  - nu/8*(16 + 5*nu)*qdotp2/q3 + 1/4*(3 + 11*nu)/q2
    
    # Put all together
    Q_vec = q_vec + cQ_1PN_q*q_vec + cQ_1PN_p*p_vec + cQ_2PN_q*q_vec + cQ_2PN_p*p_vec
    P_vec = p_vec + cP_1PN_q*q_vec + cP_1PN_p*p_vec + cP_2PN_q*q_vec + cP_2PN_p*p_vec

    return Q_vec, P_vec

def polar_to_cartesian(q_p, p_p):
    """
    q_p : [r , phi ]
    p_p : [pr, pphi]
    """
    # unpack
    r,phi   = q_p[0],q_p[1]
    pr,pphi = p_p[0],p_p[1]

    # coordinates
    x   = r*np.cos(phi)
    y   = r*np.sin(phi)
    q_c = np.array([x,y])

    #momenta
    px = -y/r**2 * pphi + x/r * pr
    py =  x/r**2 * pphi + y/r * pr

    p_c = np.array([px, py])
    
    return q_c, p_c

def cartesian_to_polar(q_c, p_c):
    """
    q_c : [x , y ]
    p_c : [px, py]
    """
    # unpack
    x, y   = q_c[0],q_c[1]
    px, py = p_c[0],p_c[1]

    # coordinates
    r    = np.sqrt(x**2+y**2)
    phi  = math.atan2(y, x) if x >= 0 else math.atan2(y, x) + math.pi
    q_p  = np.array([r, phi])
    
    #momenta
    pphi =  x*py - y*px
    p    =  np.sqrt(py**2 + px**2)
    arg  = p**2 - (pphi/r)**2
    if(abs(arg)) < 1e-15:
        pr = 0
    else: 
        pr   = -np.sqrt(p**2 - (pphi/r)**2)
    p_p  =  np.array([pr, pphi])

    return q_p, p_p

# ---------------------
# Simulation class 
# --------------------- 

COMMENTS_IGNORE = ['#']
METADATA_EXTENSION = ['.txt','.json']
KEYMAP = {'m1': ['m1','M1_i','m_plus','initial-mass1','initial_mass1'],
          'm2': ['m2','M2_i','m_minus','initial-mass2','initial_mass2'],
          'q' : ['q','mass-ratio','initial_mass_ratio'],
          'chi1z': ['chi1z','s1z','initial-bh-chi1z','initial_dimensionless_spin1'],
          'chi2z': ['chi2z','s2z','initial-bh-chi2z','initial_dimensionless_spin2'],
          'e': ['e','eccentricity'],
          'Mf': ['Mf','M_f','final-mass','remnant_mass'],
          'af': ['af','a_f','final-chi','remnant_dimensionless_spin'],
          'Jf': ['Jf','J_f'], # = af/Mf**2
          'E0': ['E0','E_adm','initial-ADM-energy','initial_ADM_energy'],
          'J0': ['J0','J_adm','initial-ADM-angular-momentum-z', 'initial_ADM_angular_momentum'],
          'r0': ['r0','initial-separation','initial_separation'],
          'f0': ['f0','initial_orbital_frequency'],
          'fgw0': ['fgw0','freq-start-22'],
}

class Simulations():
    """
    Class representing simulation data

    path : path to the metadata 
           search is recursive from this level
    
    metadata_ext : extension(s) of metadata files 
    
    comments_ignore : list of characters to ignore

    keymap : dictionary to map key names
    """
    def __init__(self, path ='./', 
                 metadata_ext = METADATA_EXTENSION,
                 comments_ignore = COMMENTS_IGNORE, 
                 keymap = KEYMAP,
                 sxs_unpack = True, 
                 rit_ids_unpack = False,
        ):
        self.path = path
        self.files = list(itertools.chain.from_iterable(
            [glob.glob(path+'/**/*'+x,recursive=True) for x in metadata_ext]))
        self.comments_ignore = comments_ignore

        self.data = []

        if rit_ids_unpack:
            # read the big ID file only once
            d_ics = self.rit_tex_read_punctures_id(path+'/rit_id_table.tex')

        for f in self.files:
            basename,ext = os.path.splitext(f)
            catalog = basename.split('/')[1]
            if ext == '.txt':
                d = self.read_txt(f)
            elif ext == '.json':
                d = self.read_json(f)
            else:
                print('Skip file {}, do not know how to read data from {}'.format(f,ext))
                continue
            if not d: continue # dict is empty, do not store

            if sxs_unpack and catalog == 'SXS':
                # We read the large json files:
                # extract each simulation and append!
                self.sxs_json_extract_append(d,catalog,f)
            else:
                d['CATALOG'] = catalog
                d['METADATA-FILE'] = f
                self.data.append(d)

            if rit_ids_unpack and catalog == 'RIT':
                self.rit_extract_punctures_id(d, d_ics)

            print('Read file {} from catalog {}'.format(f,catalog))

        self.parse_metadata(keymap=keymap)

    def info(self):
        catalog = [d['CATALOG'] for d in self.data]
        print('CATALOG : DATASETS')
        for c in set(catalog):
            print('{} : {}'.format(c,catalog.count(c)))
            
    def read_txt(self,fname):
        """
        Read metadata from a txt file 
        """
        data = {}
        if os.path.isfile(fname):        
            with open(fname,'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.replace(" ", "")
                    if line[0] in self.comments_ignore:
                        continue
                    line = line.rstrip('\n')
                    kv = line.split('=')
                    if len(kv)>1:
                        data[kv[0]] = kv[1]
        return data

    def read_json(self,fname):
        """
        Read metadata from a json file 
        """
        with open(fname) as f:
            return json.load(f)

    def sxs_json_extract_append(self,db,catalog,f):
        """
        Special method to extract single-simulation metadata 
        from sxs_catalog.json and append those
        """
        if 'resolutions' not in f:
            for s in db['simulations'].keys():
                d = {}
                d['CATALOG'] = catalog
                d['METADATA-FILE'] = f
                d.update(db['simulations'][s])
                self.data.append(d)
        else:
            # Add the resolutions, if possible
            snames = [s['alternative_names'] for s in self.data 
                      if 'initial_ADM_angular_momentum' in s.keys()]
            for s in db.keys():
                try:
                    self.data[snames.index(s)]['resolutions'] = db[s]
                except:
                    pass

    def rit_tex_read_punctures_id(self, fname):
        """
        Special method to read punctures ID in ADM coordinates
        from `rit_id_table.txt`

        Headers:
        Run, x_1/m, x_2/m, P_r/m, P_t/m, m^p_1/m, m^p_2/m, |S_1/m^2|, |S_2/m^2|, m^H_1/m, m^H_2/m, M_{\rm ADM}/m,|a_1/m_1^H|,|a_2/m_2^H|
        or
        Run, x_1/m, x_2/m, P_r/m, P_t/m, m^p_1/m, m^p_2/m, m^H_1/m, m^H_2/m, M_{\rm ADM}/m, e, N_{orb}
        """
        
        # read the ID file
        d = {}
        if os.path.isfile(fname):
            with open(fname, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line[:3] != 'RIT':
                        continue
                    line = line.rstrip('\n').rstrip('\\')
                    ids  = line.split('&')
                    if (len(ids) != 14) and (len(ids) !=12):
                        continue
                    tag = ids[0].rstrip()
                    ics = self.cast_to_float(ids[1:], ReturnLast=False)
                    d[tag] = ics
        
        print('Read RIT punctures ID from {}'.format(fname))
        return d

    def rit_extract_punctures_id(self, db, id_db):
        """
        Special method to extract punctures ID in ADM coordinates
        from id_db and add it to simulation
        """
        tag = db['catalog-tag']
        if tag in id_db.keys():
            ics = id_db[tag]
            q = float(db['relaxed-mass-ratio-1-over-2'])
            nu= q/(1+q)**2
            db['x_p'] = [ics[0],0]
            db['x_m'] = [ics[1],0]
            db['P_p'] = [ics[2]/nu, ics[3]/nu]
            db['P_m'] = [-ics[2]/nu, -ics[3]/nu]
        else:
            print("\t No puncture ID avalilable for {}".format(tag))

    def parse_metadata(self,keymap=KEYMAP):
        """
        Parse metadata dictionary 
        For basic quantities:
        * add keys with uniform names across catalogs 
        * make sure necessary data are present
        """
        for d in self.data:
            # add uniform keys 
            for k,v in  d.copy().items():
                for mk,mv in keymap.items():
                    if k in mv: d[mk] = d[k]
        # cast to float basic quantities & add data
        for d in self.data:
            for mk in keymap.keys():
                if mk in d.keys(): 
                    d[mk] = self.cast_to_float(d[mk])
            # mass ratio (q>1)
            if 'q' not in d.keys() and 'm1' in d.keys() and 'm2' in d.keys():
                d['q'] = d['m1']/d['m2']
                if d['q']<1: d['q'] = 1.0/d['q'] 
            # symm mass ratio
            if 'q' in d.keys():
                d['nu'] = self.q_to_nu(d['q'])
            elif 'm1' in d.keys() and 'm2' in d.keys():
                d['nu'] = (d['m1']*d['m2'])/(d['m1']+d['m2'])**2
            # ...
        #
            
    def cast_to_float(self,x,ReturnLast=True):
        if type(x) is list:
            try:
                x = [float(e) for e in x] 
                if ReturnLast: return x[-1]
                else: return x
            except: return []
        else:
            try: return float(x)
            except: return []

    def remove_empty_vals(self):
        return [d for d in self.data if d]

    def search(self,key,val,bound=0.):
        """
        Return list of simulations matching
        a particular key-value pair
        """
        s = []
        if bound>0.:
            s = [self.data[i] for i, item in enumerate(self.data) if abs(item[key] - val)<bound] 
        else:
            s = [self.data[i] for i, item in enumerate(self.data) if item[key] == val]
        return s

    def q_to_nu(self,q):
        return q/(1+q)**2

    #TODO EOB-ADM trafo
    def ADM_to_EOB(self, nu, q_adm, p_adm, polar=False):
        """
        Transform the initial conditions q_adm, p_adm to EOB coordinates.
        """
        if polar:
            q_adm_c, p_adm_c = polar_to_cartesian(q_adm, p_adm)
        else:
            q_adm_c, p_adm_c = q_adm, p_adm
        
        q_eob_c, p_eob_c = adm_to_eob(q_adm_c, p_adm_c, nu)
        q_eob_p, p_eob_p = cartesian_to_polar(q_eob_c, p_eob_c)
        return q_eob_p, p_eob_p

    def www_get_RIT(self,simulation):
        """
        Return command to get RIT data from www

        RIT catalog offers 3 type of files, 
        names are based on a few metadata entries

        Example: 

        catalog-tag                             = RIT:BBH:0001
        resolution-tag                          = n100
        id-tag                                  = id3

        'Metadata/RIT:BBH:0001-n100-id3_Metadata.txt'
        'Data/ExtrapPsi4_RIT-BBH-0001-n100-id3.tar.gz
        'Data/ExtrapStrain_RIT-BBH-0001-n100.h5'

        Note: Currently relies on the system call 'wget'.
        There python packages (see below), but would need installation.
        What is available within anaconda?
        """
        WWW = 'https://ccrgpages.rit.edu/~RITCatalog/'
        GET = 'wget --no-check-certificate ' # w/ white space
        ctag = simulation['catalog-tag'].strip() 
        rtag = simulation['resolution-tag'] .strip()
        itag = simulation['id-tag'].strip()
        ctag_ = ctag.replace(":",'-')
        DAT = ['Metadata/{}-{}-{}_Metadata.txt'.format(ctag,rtag,itag),
               'Data/ExtrapPsi4_{}-{}-{}.tar.gz'.format(ctag_,rtag,itag),
               'Data/ExtrapStrain_{}-{}.h5'.format(ctag_,rtag)]
        return [GET+WWW+d for d in DAT]
        #return [WWW+d for d in DAT] # urls only

    def www_get_SXS(self,simulation):
        """
        Return command to get SXS data from www

        TODO
        SXS metadata contains a 'url' key to Zenodo
        can use wget, curl or the various pyton-based packages:
        https://pypi.org/project/requests/
        https://pypi.org/project/zenodo-get/
        """

    def www_get(self,data=None,outdir='Data/',dryrun=False):
        """
        Get data corresponding to the input metadata
        Websites and www get are set based on the CATALOG key
        Outdir is pre-pended by path and CATALOG 
        """
        #import wget # other option
        cmd = {'RIT': self.www_get_RIT,
               #'SXS': self.www_get_SXS,
        }
        if data is None: data = self.data
        for s in data:
            if s['CATALOG'] not in cmd.keys():
                print('Skip {} : do not know how to get the data'.format())
                continue
            for c in cmd[s['CATALOG']](s):
                if dryrun: 
                    print(c)
                else:
                    path_outdir = '{}/{}/{}/'.format(self.path,s['CATALOG'],outdir)
                    runcmd(c,path_outdir)
                    #wget.download( <url> ,out=path_output)            

if __name__ == '__main__':


    #s = Simulations()
    #s.info()
    
    #s = Simulations('./SXS',sxs_unpack=True)
    #s.info()

    # s = Simulations('./RIT')
    # s.info()

    # for k,v in s.data[0].items():
    #     print(k,v)
    
    # example ADM to EOB for RIT
    s = Simulations('./RIT',rit_ids_unpack=True)
    nu = s.data[0]['nu']
    xp = s.data[0]['x_p']
    pp = s.data[0]['P_p'] # already nu-normalized
    q_eob, p_eob = s.ADM_to_EOB(nu, -2*np.array(xp), np.array(pp), polar=False)
    print(q_eob, p_eob)

    # for k,v in s.data[0].items():
    #     print(k,v)

    # s1 = s.search('m1',0.5)
    # s2 = s.search('m1',0.5,0.1)
    # s3 = s.search('m1',77)
    # print(s1)
    # print(s2)
    # print(s3)

    #s.www_get(data=[s.data[0]])
