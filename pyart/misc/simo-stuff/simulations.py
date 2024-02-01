# Simulations list and info
import sys, os, re, json, subprocess
import numpy as np

# ---------------------
# Simulation class 
# --------------------- 

# Paths
repo_path = subprocess.Popen(['git','rev-parse','--show-toplevel'],stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')+'/'
DATAPATH = os.path.join(repo_path,'data/')                     # not used yet
SIMJSON  = os.path.join(repo_path,'runs/simulations.json')     # Simulations json
#PARPATH  = os.path.join(repo_path,'data/gauss_2021/')          # directory with GRA-parfiles
TP_PATH  = './'                                                # directory whit the TP softlink
TP_DUMMY = os.path.join(repo_path,'py/dummies/TP.dummy')

# Name of dataset
DSET = "GAUSS_2021"

class Sim(object):
    """
    Class to manage metadata of a set (dset) of simulations
    Assumes all the data are stored in a JSON file with 
    at least one dataset and key-values pairs
    """
    def __init__(self, jsonfile=SIMJSON, datapath=DATAPATH, dset=DSET, TP_path=TP_PATH, TP_dummy=TP_DUMMY): 
        with open(jsonfile) as f:
            self.data = json.load(f, strict=False)
        self.dset         = dset
        self.TP_path      = TP_path
        self.TP_dummy     = TP_dummy
        self.datapath     = datapath

        # entries of the json files: 
        #  - parfiles_keys are the ones that can be read from parfile
        #  - ordered_entries are all the (ordered) keys in the json file
        #  - params that are stored as arrays in the json file
        self.parfile_keys =  ['par_b','center_offset', 'git_rev_GRA', 'git_rev_TP',
                              'give_bare_mass', 'par_m_plus','par_m_minus', 
                              'target_M_plus', 'target_M_minus',
                              'par_P_plus', 'par_P_minus', 'par_S_plus', 'par_S_minus',
                              'npoints_A', 'npoints_B', 'npoints_phi',
                              'extraction_radii','extraction_nlev','lmax','cfl_number', 'numlevel',
                              'nx1','x1min', 'x1max', 'ix1_bc', 'ox1_bc',
                              'nx2','x2min', 'x2max', 'ix2_bc', 'ox2_bc',
                              'nx3','x3min', 'x3max', 'ix3_bc', 'ox3_bc', 'z4c_amr_method']

        self.ordered_entries = ['name', 'git_rev_GRA', 'git_rev_TP', 'parfile', 'q', 'nu', 'chi1z', 'chi2z', 'M_ADM', 'J_ADM', 
                                'M1_ADM', 'M2_ADM', 'par_m_plus', 'par_m_minus', 
                                'give_bare_mass', 'target_M_plus', 'target_M_minus', 
                                'par_b', 'par_P_plus', 'par_P_minus', 'par_S_plus', 'par_S_minus', 
                                'center_offset', 'npoints_A', 'npoints_B', 'npoints_phi', 'cfl_number', 
                                'extraction_radii', 'extraction_nlev', 'lmax', 'numlevel', 'delta_xp',
                                'nx1','x1min', 'x1max', 'ix1_bc', 'ox1_bc',
                                'nx2','x2min', 'x2max', 'ix2_bc', 'ox2_bc',
                                'nx3','x3min', 'x3max', 'ix3_bc', 'ox3_bc',
                                'z4c_amr_method', 'bitant', 'ahf', 'cce']
 
        self.vector_keys = ['par_P_plus', 'par_P_minus', 'par_S_plus', 
                            'par_S_minus', 'center_offset', 'extraction_radii']
        return

    def show(self, key="name",dset=None):
        """
        Print all simulation by key
        """
        if dset is None: dset = self.dset
        for s in self.data[dset]:
            print(s[key])
    
    def simlist(self,key="name",dset=None): 
        if dset is None: dset = self.dset
        sim_list = []
        for s in self.data[dset]:
            sim_list.append(s[key])
        return sim_list
    
    def print_simlist(self, key="name",dset=None):
        if dset is None: dset = self.dset
        all_sim_dset = self.simlist(key=key,dset=dset)
        longest_name_len = len(max(all_sim_dset, key=len))
        nstr = str(longest_name_len)
        line_template = '#{:3d}: {:'+nstr+'s}  #{:3d}: {:'+nstr+'s}'
        nsims = len(all_sim_dset)
        print(f'Printing the {nsims:d} simulations of {dset:s}')
        for i in range(0,nsims,2):
            if i<nsims-1:
                print(line_template.format(i, all_sim_dset[i], i+1, all_sim_dset[i+1]))
            else:
                print(f'#{i:3d}: {all_sim_dset[i]:s}')
        print(' ')
    
    def subset(self, pars, float_tol=1e-5, dset=None):
        """
        Return subset (list of names) of current dset
        pars: dict with keys and values
        Criteria for floats: sim[key]=value +- float_tol
        """
        if dset is None: dset = self.dset
        names_list = []
        keys = pars.keys()
        for s in self.data[dset]:
            bool_condition = True
            for key in pars.keys():
                value = pars[key]
                bool_condition = bool_condition and (\
                                 (isinstance(s[key], float) and abs(s[key]-value)<float_tol) or \
                                 (isinstance(s[key],   int) and s[key]==value) or \
                                 (isinstance(s[key],   str) and s[key]==value)  )
            if bool_condition:
                names_list.append(s['name'])
        return names_list 

    def addkv(self, key,val,dset=None):
        """
        Add a pair key-value to every simulation of the dset
        (use one value for all simulations)
        """
        if dset is None: dset = self.dset
        for i,s in enumerate(self.data[dset]):
            if key not in s:
                self.data[dset][i][key] = val
            else:
                raise ValueError('key {} exists'.format(key))

    def delkv(self, key, dset=None):
        """
        Delete a pair key-value to every simulation of the dset
        """
        if dset is None: dset = self.dset
        for s in self.data[dset]:
            if key in s:
                sim_idx = self.getsim_idx(s['name'])
                del self.data[dset][sim_idx][key]
            else:
                raise ValueError('key {} does not exists'.format(key))

    def getkv(self,key="name",dset=None):
        """
        Return list of key-value pairs from simulations
        """
        if dset is None: dset = self.dset
        out = []
        for s in self.data[dset]:
            out.append(s[key])
        return out

    def getsim(self,val,key="name",dset=None):
        """
        Return all simulations matching the key-value pair
        """
        if dset is None: dset = self.dset
        return next((s for s in self.data[dset] if s[key] == val), None)
    
    def write_json(self,fname):
        """
        Write data to JSON
        """
        with open(fname, 'w') as f:
            f.write(json.dumps(self.data,indent=2))
            
    def read_json(self,fname=None):
        """
        Read data from JSON
        """
        if fname is None : fname = SIMJSON 
        with open(fname, 'r') as f:
            data = json.loads(f.read())
        return data

    def getsim_idx(self,sim_name,dset=None):
        """
        Return idx of a simulation given the name
        """
        sim_idx = None
        if dset is None: dset = self.dset
        for i,s in enumerate(self.data[dset]):
          if s['name']==sim_name:
            sim_idx = i
            break
        if sim_idx is None: raise ValueError('Simulation {} not found'.format(sim_name))
        return sim_idx

    def order_data_keys(self,sim_name,dset=None):
        """
        Order the JSON entries
        """
        if dset is None: dset = self.dset
        sim_idx = self.getsim_idx(sim_name,dset=dset)
        dict_tmp = self.data[dset][sim_idx] 
        dict_new = {}
        n_ordered_entries = len(self.ordered_entries)
        n_dict_keys= len(dict_tmp.keys())
        if n_ordered_entries<n_dict_keys:
            #for k in self.ordered_entries:
            #    print('key: {:20} - is in sim_dict? {}'.format(k, k in dict_tmp))
            #print(n_ordered_entries)
            #print(self.ordered_entries)
            #print(n_dict_keys)
            #print(dict_tmp.keys())
            raise RuntimeError('check entries')
        for key in self.ordered_entries:
            if key in dict_tmp.keys():
                dict_new[key] = dict_tmp[key]
            else:
                dict_new[key] = None
        self.data[dset][sim_idx] = dict_new
        return
    
    def get_parfile_groups(self,sim_name,dset=None,datapath=None,parfile='parfile.par'):
        if dset is None: dset = self.dset
        if datapath is None: datapath = self.datapath
        sim_idx = self.getsim_idx(sim_name, dset=dset)
        parfile_fname = os.path.join(datapath, self.dset.lower(), self.data[dset][sim_idx]['name'], parfile)
        with open(parfile_fname, 'r') as file:
            parfile_lines = file.readlines()
        groups = []
        for line in parfile_lines:
            if line[0]=='<' and '>' in line:
                group = line.translate(str.maketrans({'>':'', '<':'', '\n':''}))
                groups.append(group)
        return groups

    def read_parfile_value(self,sim_name,parameter,dset=None,datapath=None,parfile='parfile.par'):
        """
        Read a value from parfile
        """
        if dset is None: dset = self.dset
        if datapath is None: datapath = self.datapath
        # load parfile
        sim_idx = self.getsim_idx(sim_name, dset=dset)
        parfile_fname = os.path.join(datapath, self.dset.lower(), self.data[dset][sim_idx]['name'], parfile)
        with open(parfile_fname, 'r') as file:
            parfile_lines = file.readlines()

        # remove commented lines
        parfile_lines = [string for string in parfile_lines if not string.startswith('#')]

        # search parameter
        parameter_found = False
        group = None
        
        # some parameters are renamed w.r.t. parfile
        if parameter=='z4c_amr_method':
            parameter = 'method'
        
        for line in parfile_lines:
            if line[0]=='<' and '>' in line:
                group = line.translate(str.maketrans({'>':'', '<':'', '\n':''}))
            if parameter in line:
                parameter_found = True
                break
        if not parameter_found:
            return None, None
        
        # read value of the parameter
        string_value = line.split("=")[-1].replace(' ', '')
        if re.match(r'^[-+]?\d+\.\d*$', string_value) or re.match(r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$', string_value):
            value     = float(string_value)
            value_int = int(value)
            if abs(value-value_int)<1e-14:
              value = value_int
        else:
            value = string_value.replace('\n', '')
            # check if bool
            lvalue = value.lower()
            if lvalue.replace(' ','')=="true":
                value = True
            elif lvalue.replace(' ','')=="false":
                value = False
        return value, group    
    
    def update_sim_key_from_parfile(self,sim_name,key,dset=None,verbose=False,datapath=None,parfile='parfile.par'):
        """
        Update a specified key of a simulation according to the stored parfile 
        """
        if dset     is None: dset     = self.dset
        if datapath is None: datapath = self.datapath
        sim_idx = self.getsim_idx(sim_name,dset=dset)
        value = None
        if key in self.vector_keys:
            if key=='extraction_radii':
                entry_template = 'extraction_radius_@n@'
                idx0 = 0
            else:
                entry_template = key+'@n@'
                idx0 = 1
            vector = []
            continue_to_read = True
            i = idx0
            while continue_to_read:
                parameter = entry_template.replace('@n@', str(i))
                par_val, _ = self.read_parfile_value(sim_name,parameter,dset=dset,datapath=datapath,parfile=parfile)
                if par_val is not None:
                    vector.append(par_val)
                    i += 1
                else:
                    continue_to_read = False
            value = vector
        else:
            value, _ = self.read_parfile_value(sim_name,key,dset=dset,datapath=datapath,parfile=parfile)
        if verbose:
            print('key: {:20s} val: '.format(key), value)
        self.data[dset][sim_idx][key] = value
        return
    
    def update_sim_from_parfile(self,sim_name,dset=None,datapath=None,verbose=False,parfile='parfile.par'):
        """
        Update all the keys of a given simulation
        that can be updated according to parfile's info
        """
        if dset     is None: dset     = self.dset
        if datapath is None: datapath = self.datapath
        sim_dict = self.getsim(sim_name)
        sim_dict['parfile'] = parfile
        for key in self.parfile_keys:
            self.update_sim_key_from_parfile(sim_name,key,dset=dset,verbose=verbose,datapath=datapath,parfile=parfile)
        if sim_dict['z4c_amr_method'] is None:
            sim_dict['z4c_amr_method'] = 'Linf_box_in_box'
        # determine cce and bitant
        if sim_dict['x3min']==0 and sim_dict['ix3_bc']=='reflecting':
            sim_dict['bitant'] = True
        else:
            sim_dict['bitant'] = False
        groups = self.get_parfile_groups(sim_name,dset=dset,datapath=datapath,parfile=parfile)
        pars_to_check_in_groups = ['cce', 'ahf']
        for p in pars_to_check_in_groups:
            if p in groups:
                sim_dict[p] = True
            else:
                sim_dict[p] = False

        # compute some useful quantities
        sim_dict['delta_xp'] = (sim_dict['x1max']-sim_dict['x1min'])/sim_dict['nx1']/2**(sim_dict['numlevel']-1)

        sim_idx = self.getsim_idx(sim_name)
        self.data[dset][sim_idx] = sim_dict
        self.order_data_keys(sim_name,dset=dset)
        return
    
    def add_sim(self,sim_name,dset=None,verbose=False,parfile=None,datapath=None):
        if dset     is None: dset     = self.dset
        if datapath is None: datapath = self.datapath
        sim_list = self.simlist(key="name",dset=dset)
        if sim_name in sim_list:
            print('{} already exists!'.format(sim_name))
            return
        new_dict = {'name': sim_name}
        for key in self.parfile_keys:
            if key in self.vector_keys:
                new_dict[key] = [0]
            else:
                new_dict[key] = 0
        self.data[dset].append(new_dict)
        if parfile is not None:
            self.update_sim_from_parfile(sim_name,dset=dset,verbose=verbose,datapath=datapath,parfile=parfile) # parpath=parpath
        return 
    
    def TP_create_parfile(self,sim_name,dset=None,TP_dummy=None,outdir='.'):
        if dset is None: dset = self.dset
        if TP_dummy is None: TP_dummy = self.TP_dummy
        parfile_lines = []
        with open(TP_dummy, 'r') as file:
            for line in file:        
                parfile_lines.append(line)
        sim_idx  = self.getsim_idx(sim_name, dset=dset)
        sim_dict = self.data[dset][sim_idx]
        for line_idx, line in enumerate(parfile_lines):
            if '@' in line:
                par_name = line.split('@')[1]
                if par_name[-1] in ['1','2','3']:
                    key = par_name[0:-1]
                    vec_idx = int(par_name[-1])-1
                    val2write = sim_dict[key][vec_idx]
                else:
                    val2write = sim_dict[par_name]
                # convert bool to int (as needed in TP parfiles)
                if val2write==True:
                    val2write = 1
                elif val2write==False:
                    val2write = 0
                #if par_name=='npoints_A':
                #    val2write = 8
                #if par_name=='npoints_B':
                #    val2write = 8
                #if par_name=='npoints_phi':
                #    val2write = 4
                parfile_lines[line_idx] = line.replace('@'+par_name+'@', str(val2write) ) 
        if outdir[-1]=='/':
            outdir += '/'
        parfile = outdir + 'TP_' + sim_name + '.par' 
        with open(parfile, 'w') as file:
            for line in parfile_lines:
                file.write(line)
        return parfile

    def TP_update_ADM(self,sim_name,dset=None,TP_path=None,TP_dummy=None,outdir=None,verbose=True,search_ID_in_other_sims=True):
        if dset is None: dset = self.dset
        if TP_path is None: TP_path = self.TP_path 
        if TP_dummy is None: TP_dummy = self.TP_dummy
        if outdir is None:
            if self.datapath is not None: 
                outdir = os.path.join(self.datapath, self.dset.lower(), sim_name)
            else:
                outdir = './'
        parfile = self.TP_create_parfile(sim_name,dset=dset,outdir=outdir)
        run_TP = True
        sim_idx = self.getsim_idx(sim_name,dset=dset)
        if search_ID_in_other_sims:
            sim_dict = self.getsim(sim_name) 
            def are_dict_values_equal(dict1, dict2, key_list):
                for key in key_list:
                    if key in dict1 and key in dict2:
                        if dict1[key] != dict2[key]:
                            return False
                    else:
                        return False
                return True
            keys_to_check = ['par_m_plus', 'par_m_minus', 'give_bare_mass', 'target_M_plus', 'target_M_minus', 
                             'par_b', 'par_P_plus', 'par_P_minus', 'par_S_plus', 'par_S_minus', 'center_offset',
                             'npoints_A', 'npoints_B', 'npoints_phi', 'git_rev_TP']
            keys_ADM = ['q', 'nu', 'chi1z', 'chi2z', 'M_ADM', 'J_ADM', 'M1_ADM', 'M2_ADM']
            #print(f'{sim_name}, other sims with same ID:')
            for another_sim in self.data[dset]:
                another_sim_name = another_sim['name']
                if another_sim_name==sim_name:
                    continue
                another_sim_dict = self.getsim(another_sim_name)
                if are_dict_values_equal(sim_dict, another_sim_dict, keys_to_check) and another_sim_dict['M_ADM'] is not None:
                    print('{:s} and {:s} have same ID.\nUsing ADM quantities of '
                          'the latter also for the former'.format(sim_name, another_sim_name) )
                    for k in keys_ADM:
                        sim_dict[k] = another_sim_dict[k]
                    run_TP = False
                    break
        if run_TP: 
            outfile = parfile.replace('.par', '.txt')
            TP_fullname = TP_path+'/TwoPuncturesRun.x'
            if not os.path.exists(TP_fullname):
                raise RuntimeError('TwoPuncturesRun.x not found in {}'.format(TP_path))
            cmd = TP_fullname+' '+parfile+' > '+outfile
            if verbose:
                print('Created parfile', parfile)
                print('Running: ', cmd)
            runcmd(cmd,TP_path)
            with open(outfile, 'r') as file:
                for line in file:        
                    if 'Puncture 1 ADM mass is' in line:
                        M1 = float(line.split(' ')[-1])
                    elif 'Puncture 2 ADM mass is' in line:
                        M2 = float(line.split(' ')[-1])
                    elif 'The total ADM mass is' in line:
                        M_ADM = float(line.split(' ')[-1])
                        break
            par_b      = self.data[dset][sim_idx]['par_b']
            par_P_plus = self.data[dset][sim_idx]['par_P_plus']
            J_ADM      = par_b*2*par_P_plus[1]
            self.data[dset][sim_idx]['M1_ADM'] = M1
            self.data[dset][sim_idx]['M2_ADM'] = M2
            self.data[dset][sim_idx]['M_ADM']  = M_ADM
            self.data[dset][sim_idx]['J_ADM']  = J_ADM
            q = compute_rounded_q(self.data[dset][sim_idx])
            self.data[dset][sim_idx]['q']     = q
            self.data[dset][sim_idx]['nu']    = sim_dict['q']/(1+sim_dict['q'])**2
            self.data[dset][sim_idx]['chi1z'] = sim_dict['par_S_plus'][2]/sim_dict['M1_ADM']**2
            self.data[dset][sim_idx]['chi2z'] = sim_dict['par_S_minus'][2]/sim_dict['M2_ADM']**2
            #rmcmd = 'rm -r TP_'+sim_name+'*'
        
        if verbose:
            #rmcmd = rmcmd.replace('rm -r', 'rm -rv')
            print('q     : ', self.data[dset][sim_idx]['q'])
            print('nu    : ', self.data[dset][sim_idx]['nu'])
            print('chi1z : ', self.data[dset][sim_idx]['chi1z'])
            print('chi2z : ', self.data[dset][sim_idx]['chi2z'])
            print('M1_ADM: ', self.data[dset][sim_idx]['M1_ADM'])
            print('M2_ADM: ', self.data[dset][sim_idx]['M2_ADM'])
            print('M_ADM : ', self.data[dset][sim_idx]['M_ADM'])
            print('J_ADM : ', self.data[dset][sim_idx]['J_ADM'])
        #runcmd(rmcmd, TP_path)
        return

def compute_rounded_q(sim, round_threshold=1e-8):
    M1_ADM = sim['M1_ADM']
    M2_ADM = sim['M2_ADM']
    q_tmp = M1_ADM/M2_ADM
    if q_tmp<1:
        q_tmp = 1/q_tmp
    q_int = round(q_tmp/round_threshold)
    return q_int*round_threshold

# ---------------------
# Basic unit conversion
# --------------------- 
ufact = {
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

    
if __name__ == "__main__":

    sim = Sim()
    print("{:20s} :   {:20s} {:20s}".format('name', 'TP version', 'GRA version'))
    print("-"*66)
    for s in sim.read_json()[DSET]:
        print("{:20s} :   {:20s} {:20s}".format(s['name'], s['git_rev_TP'], s['git_rev_GRA']) )



