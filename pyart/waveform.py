# Classed to handle waveforms

#standard imports
import numpy as np; import h5py; import json
from scipy.signal import find_peaks

# other imports
import utils as ut
import wf_utils as wf_ut

try:
    import EOBRun_module as EOB
except ModuleNotFoundError:
    print("WARNING: TEOBResumS not installed.")

try:
    import lalsimulation as lalsim
    import lal  
except ModuleNotFoundError:
    print("WARNING: LALSimulation and/or LAL not installed.")

class Waveform(object):
    """
    Parent class to handle waveforms
    Children classes will inherit methods & propreties
    """
    def __init__(self):
        self._u   = None
        self._f   = None
        self._hp  = None
        self._hc  = None
        self._hlm = {}
        self._dyn = {}
        self._kind = None
        pass

    @property
    def u(self):
        return self._u
    @property
    def f(self):
        return self._f
    @property
    def hp(self):
        return self._hp
    @property
    def hc(self):
        return self._hc
    @property
    def hlm(self):
        return self._hlm
    @property
    def dyn(self):
        return self._dyn
    @property
    def kind(self):
        return self._kind
    
    # methods
    def extract_hlm(self, ell, emm):
        k = int(ell*(ell-1)/2 + emm-2)
        return self.hlm[str(k)][0], self.hlm[str(k)][1]

    def find_max(self, mode = '1', umin=0):
        u   = self.u
        p   = self.hlm[mode][1]
        A22 = self.hlm[mode][0]

        # compute omega
        omg      = np.zeros_like(p)
        omg[1:]  = np.diff(p)/np.diff(u)
        # compute domega
        domg     = np.zeros_like(omg)
        domg[1:] = np.diff(omg)/np.diff(u)
        
        # find first peak (merger) after umin
        peaks, _ = find_peaks(A22, height=0.15)

        for i in range(len(peaks)):
            if(u[peaks[i]] > umin):
                break

        u_mrg    = u[peaks[i]]
        A_mrg    = A22[peaks[i]]
        omg_mrg  = omg[peaks[i]]
        domg_mrg = domg[peaks[i]]

        return u_mrg, A_mrg, omg_mrg, domg_mrg

    def energetics_hlm(self, modes=['1']):
        """
        Compute the (E, J) from the multipoles
        or the dynamics.
        """
        u  = self.u
        du = u[1] - u[0]
        
        E_GW_dot = {}
        E_GW     = {}
        J_GW_dot = {}
        J_GW     = {}    

        E_GW_dot_all = np.zeros_like(u)
        E_GW_all     = np.zeros_like(u)
        J_GW_dot_all = np.zeros_like(u)
        J_GW_all     = np.zeros_like(u)

        for mode in modes:
            m      = wf_ut.k_to_emm(int(mode))
            this_h = self.hlm[mode]
            hlm    = this_h[0]*np.exp(-1j*this_h[1])
            hlm_dot= ut.D02(u, hlm)

            # energy and flux in single |mode| 
            E_GW_dot[mode] = wf_ut.mnfactor(m)*1.0/(16.*np.pi) * np.abs(hlm_dot)**2 
            E_GW[mode]     = wf_ut.mnfactor(m)*ut.integrate(E_GW_dot[mode]) * du
            J_GW_dot[mode] = wf_ut.mnfactor(m)*1.0/(16.*np.pi) * m * np.imag(hlm * np.conj(hlm_dot)) 
            J_GW[mode]     = wf_ut.mnfactor(m)*ut.integrate(J_GW_dot[mode]) * du

            E_GW_dot_all += E_GW_dot[mode]
            E_GW_all     += E_GW[mode]
            J_GW_dot_all += J_GW_dot[mode]
            J_GW_all     += J_GW[mode]
        
        return E_GW_all, E_GW_dot_all, J_GW_all, J_GW_dot_all

    def compute_hphc(self, phi=0, i=0, modes=['1']):
        """
        For aligned spins, compute hp and hc
        """
        self._hp, self._hc = wf_ut.compute_hphc(self.hlm, phi, i, modes)
        return self.hp, self.hc
    
    def interpolate_hlm(self, dT):
        """
        Interpolate the hlm dictionary to a grid of uniform dT
        """
        hlm_i = {}
        new_u = np.arange(self.u[0], self.u[-1], dT)
        
        for k in self.hlm.keys():
            hlm_i[k] = [np.interp(new_u, self.u, self.hlm[k][0]),
                        np.interp(new_u, self.u, self.hlm[k][1])]
        
        return new_u, hlm_i  

class Waveform_LAL(Waveform):
    """
    Class to handle LAL waveforms (TD, FD or from the lvcnr catalog)
    TODO: extract modes from waveforms
    """
    def __init__(
                    self, 
                    pars=None, 
                    approx='IMRPhenomXPHM', 
                    kind='FD',
                ):
        
        super().__init__()
        self.pars   = pars
        self.approx = approx
        self._kind  = kind
        self._run_lal()
        pass
        
    def _eob_to_lal_dict(self):
        """
        Assume that pars contains:
            - q                   : mass ratio
            - M                   : total mass
            - chi1x, chi1y, chi1z : spin components of body 1
            - chi2x, chi2y, chi2z : spin components of body 2
            - distance            : distance to source
            - inclination         : inclination
            - coalescence_angle   : reference phase
            - initial frequency   : initial frequency
            - srate_interp        : sampling rate
            - use_mode_lm         : list of modes to use for hpc
        Can be generated via the usual CreateDict function
        """
        # create empty LALdict
        pp     = self.pars
        params = lal.CreateDict()
        modearr= lalsim.SimInspiralCreateModeArray()
        modes  = [(wf_ut.k_to_ell(k), wf_ut.k_to_emm(k)) for k in pp['use_mode_lm']]
        for mode in modes:
            lalsim.SimInspiralModeArrayActivateMode(modearr, mode[0], mode[1])
        lalsim.SimInspiralWaveformParamsInsertModeArray(params, modearr)

        # read in from pars
        q    = pp['q']
        M    = pp['M']
        c1x, c1y, c1z = pp['chi1x'], pp['chi1y'], pp['chi1z']
        c2x, c2y, c2z = pp['chi2x'], pp['chi2y'], pp['chi2z']
        DL   = pp['distance']*1e6*lal.PC_SI
        iota = pp['inclination']
        phir = pp['coalescence_angle']
        dT   = 1./pp['srate_interp']
        flow = pp['initial_frequency']
        df   = pp['df']
        srate= pp['srate_interp'] 

        # Compute masses
        m1 = M*q/(1.+q)
        m2 = M/(1.+q)
        m1SI = m1*lal.MSUN_SI
        m2SI = m2*lal.MSUN_SI
        return params, m1SI, m2SI, c1x, c1y, c1z, c2x, c2y, c2z, DL, iota, phir, dT, flow, srate, df

    def _run_lal(self):
        if self.kind == 'TD':
            self._run_lal_TD()
        elif self.kind == 'FD':
            self._run_lal_FD()
        elif self.kind == 'lvcnr':
            self._load_lal_lvcnr()
        else:
            raise ValueError("kind must be TD, FD or lvcnr")
        return 0

    def _run_lal_TD(self):
        params, m1SI, m2SI, c1x, c1y, c1z, c2x, c2y, c2z, DL, iota, phir, dT, flow, _, _ = self._eob_to_lal_dict()
        app      = lalsim.GetApproximantFromString(self.approx)
        hp, hc   = lalsim.SimInspiralTD(m1SI,m2SI,c1x,c1y,c1z,c2x,c2y,c2z,DL,iota,phir,0.,0.,0.,dT,flow,flow,params,app)
        t        = np.array(range(0, len(hp.data.data)))*hp.deltaT
        
        self._u  = t
        self._hp = hp.data.data
        self._hc = hc.data.data
        pass

    def _run_lal_FD(self):
        params, m1SI, m2SI, c1x, c1y, c1z, c2x, c2y, c2z, DL, iota, phir, _, flow, srate, df = self._eob_to_lal_dict()
        app      = lalsim.GetApproximantFromString(self.approx)
        hpf, hcf = lalsim.SimInspiralFD(m1SI,m2SI,c1x,c1y,c1z,c2x,c2y,c2z,DL,iota,phir,0.,0.,0.,df,flow,srate/2,flow,params,app)
        f        = np.array(range(0, len(hpf.data.data)))*hpf.deltaF
        
        self._f  = f
        self._hp = hpf.data.data
        self._hc = hcf.data.data  
        pass 

    def _load_lal_lvcnr(self):
        """
        Assume that pars contains:
            - sim_name          : path to simulation h5 file
            - M                 : target total mass
            - distance          : target distance
            - inclination       : inclination
            - coalescence_angle : coalescence angle
            - srate_interp      : target sampling rate
            - use_mode_lm       : list of modes to use for hpc
        """

        # create empty LALdict
        fname  = self.pars['sim_name']
        M      = self.pars['M']
        DL     = self.pars['distance']
        iota   = self.pars['inclination']
        phi_ref= self.pars['coalescence_angle']
        dT     = 1./self.pars['srate_interp']

        modes  = [(wf_ut.k_to_ell(k), wf_ut.k_to_emm(k)) for k in self.pars['use_mode_lm']]

        # read h5
        f = h5py.File(fname, 'r')

        # Modes (default (2,2) only)
        params = lal.CreateDict()
        modearr= lalsim.SimInspiralCreateModeArray()
        for mode in modes:
            lalsim.SimInspiralModeArrayActivateMode(modearr, mode[0], mode[1])
        lalsim.SimInspiralWaveformParamsInsertModeArray(params, modearr)
        lalsim.SimInspiralWaveformParamsInsertNumRelData(params, fname)

        Mt = f.attrs['mass1'] + f.attrs['mass2']
        m1 = f.attrs['mass1']*M/Mt
        m2 = f.attrs['mass2']*M/Mt
        m1SI = m1 * lal.MSUN_SI
        m2SI = m2 * lal.MSUN_SI
        DLmpc= DL*1e6*lal.PC_SI #assuming DL given in in Mpc

        flow = f.attrs['f_lower_at_1MSUN']/M
        fref = flow

        spins = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(fref, M, fname)
        c1x, c1y, c1z = spins[0], spins[1], spins[2]
        c2x, c2y, c2z = spins[3], spins[4], spins[5]

        # overwrite a bunch of parameters
        self.pars['m1']    = m1
        self.pars['m2']    = m2
        self.pars['M']     = m1+m2
        self.pars['q']     = m1/m2
        self.pars['chi1x'] = c1x; self.pars['chi1y'] = c1y;  self.pars['chi1z'] = c1z
        self.pars['chi2x'] = c2x; self.pars['chi2y'] = c2y;  self.pars['chi2z'] = c2z
        self.pars['initial_frequency'] = flow 
        
        hp, hc   = lalsim.SimInspiralChooseTDWaveform(m1SI,m2SI,c1x,c1y,c1z,c2x,c2y,c2z,DLmpc,iota,phi_ref,0.,0.,0.,dT,flow,fref,params, lalsim.NR_hdf5)
        t        = np.array(range(0, len(hp.data.data)))*hp.deltaT
        self._u  = t
        self._hp = hp.data.data
        self._hc = hc.data.data
        pass

class Waveform_SXS(Waveform):
    """
    Class to handle SXS waveforms
    Assumes that the data is in the directory specified py `path`,
    and that all simulations are stored in folders like SXS_BBH_XXXX,
    each containing the various `LevY` folders.
    e.g., the current default is 
        ../dat/SXS_BBH_XXXX/LevY/
    """
    def __init__(
                    self,
                    path   ='../dat/SXS/SXS',
                    ID     ='0001',
                    order  ="Extrapolated_N2.dir",
                    level = None,
                    cut_N = 300
                ):
        super().__init__()
        self.ID            = ID
        self.sxs_data_path = path+'_BBH_'+ID
        self.order         = order
        self.level         = level
        self.cut           = cut_N
        self._kind         = 'SXS'

        if level == None:
            # Default behavior: load only the highest level
            for lv in ['/Lev6','/Lev5','/Lev4', '/Lev3', '/Lev2', '/Lev1']:
                try:
                    self.nr = h5py.File(self.sxs_data_path+lv+"/rhOverM_Asymptotic_GeometricUnits_CoM.h5")
                    break
                except Exception: 
                    continue
        else:
            self.nr = h5py.File(self.sxs_data_path+level+"/rhOverM_Asymptotic_GeometricUnits_CoM.h5")

        self.load_hlm()
        self.load_metadata()
        self.compute_hphc()
        pass

    def load_metadata(self):
        if self.level == None:
            # Default behavior: load only the highest level
            for lv in ['/Lev6','/Lev5','/Lev4', '/Lev3', '/Lev2', '/Lev1']:
                try:
                    with open(self.sxs_data_path +lv+"/metadata.json", 'r') as file:
                        metadata = json.load(file)
                        file.close()
                    self.metadata = metadata
                    break
                except Exception:
                    continue
        else:
            with open(self.sxs_data_path +lv+"/metadata.json", 'r') as file:
                metadata = json.load(file)
                file.close()
            self.metadata = metadata

        pass

    def load_hlm(self):
        order   = self.order
        modes   = [[l,m] for l in range(2,9) for m in range(1,l+1)]
        self._u  = self.nr[order]['Y_l2_m2.dat'][:, 0][self.cut:]
        dict_hlm = {}
        for mode in modes:
            l    = mode[0]; m = mode[1]
            mode = "Y_l" + str(l) + "_m" + str(m) + ".dat"
            hlm  = self.nr[order][mode]
            h    = hlm[:, 1] + 1j * hlm[:, 2]
            # amp and phase
            Alm = abs(h)[self.cut:]
            plm = np.unwrap(np.angle(h))[self.cut:]
            # save in dictionary
            key = str(wf_ut.mode_to_k(l, m))
            dict_hlm[key] = [Alm, plm]
        self._hlm = dict_hlm
        pass

class Waveform_CoRe(Waveform):
    """
    Cass to handle CoRe database waveforms
    Assumes that they are placed in ../dat/CoRe/BAM_XXXX
    TODO: Richardson extrap
    TODO: extrapolation to infinity
    """
    def __init__(
            self,
            path  ='../dat/CoRe',
            ID    ='0095',
            res   = 'R01'
            ):
        
        super().__init__()
        self.ID   = ID
        self.path = path+'/BAM_'+ID+'/'
        self.res  = res
        self.load_metadata(self.path)
        self.load_hlm(res)
        pass

    def load_hlm(self, res, r_ext=None):
        """
        # u/M:0 Reh/M:1 Imh/M:2 Redh:3 Imdh:4 Momega:5 A/M:6 phi:7 t:8
        """
        nr = self.path+res+'/data.h5' 
        modes   = [[l,m] for l in range(2,9) for m in range(1,l+1)]
        with h5py.File(nr, "r") as f:
            for mode in modes:
                ell = mode[0]; emm = mode[1]
                ky = 'rh_'+str(ell)+str(emm)

                if ky not in f.keys(): continue
                
                self.radii_ext = list(f[ky].keys())
                if r_ext == None: r_ext    = list(f[ky].keys())[-1]
                R = (r_ext.split("_")[-1])[1:-3]
                if R == 'Inf.': 
                    print(f"{ell}{emm} extracted at infty")
                    R  = 1.
                else          : R = float(R)            
                data = np.array(f[ky][r_ext])

                # extract from file
                try:
                    u,_,Alm,philm,t = data[:,0], data[:,5],data[:,6],data[:,7],data[:,8]
                except Exception:
                    u,_,Alm,philm,t = data[:,0], data[:,3],data[:,4],data[:,5],data[:,6]
                if max(Alm) > 1: Alm = Alm/R
                
                k   = wf_ut.mode_to_k(ell, emm)
                self._hlm[str(k)] = [Alm, philm]

        self._u   = u
        pass
    
    def load_metadata(self, dir):
        metadata = {}
        mtdt = dir+'metadata_main.txt'
        f    = open(mtdt, 'r')
        lines= f.readlines()
        for line in lines:
                if line.startswith("#"): continue
                if line.startswith(" "): continue
                line = line.strip("\n")
                line = line.split("=")
                key = line[0].strip(" ")
                val = line[-1].strip(" ")
                metadata[key] = val
        f.close()
        self.metadata = metadata
        pass

    def cut_waveform(self, umax=None):
        """
        Cut the waveform at umax
        """
        if umax is None: umax = self.find_max()[0]
        idx = np.where(self.u <= umax)[0]
        self._u = self.u[idx]
        for k in self.hlm.keys():
            self._hlm[k][0] = self.hlm[k][0][idx]
            self._hlm[k][1] = self.hlm[k][1][idx]
        return 0

class Waveform_EOB(Waveform):
    """
    Class to handle EOB waveforms
    """
    def __init__(
                    self, 
                    pars=None, 
                ):
        super().__init__()
        self.pars = pars
        self._kind = 'EOB'
        self._run_py()
        pass 

    def compute_energetics(self):
        pars = self.pars
        q    = pars['q']
        q    = float(q)
        nu   =  q/(1.+q)**2
        dyn  = self.dyn
        E, j = dyn['E'], dyn['Pphi']
        Eb   = (E-1)/nu
        self.Eb = Eb
        self.j  = j
        return Eb, j

    # python wf func
    def _run_py(self):
        if self.pars['domain']:
            f, rhp,ihp,rhc,ihc, hflm, dyn, _ = EOB.EOBRunPy(self.pars)
            self._f   = f
            self._hlm = hflm
            self._dyn = dyn
            self._hp  = rhp-1j*ihp
            self._hc  = rhc-1j*ihc
        else:
            t, hp,hc, hlm, dyn = EOB.EOBRunPy(self.pars)
            self._u   = t
            self._hlm = hlm
            self._dyn = dyn
            self._hp  = hp
            self._hc  = hc
        return 0
    
# external function for dict creation
def CreateDict(M=1., q=1, 
               chi1z=0., chi2z=0, 
               l1=0, l2=0, 
               iota=0, f0=0.0035, srate=4096., df = 1./128.,
               phi_ref = 0.,
               ecc = 1e-8, r_hyp = 0, H_hyp = 0, J_hyp=0, anomaly = np.pi,
               interp="yes", arg_out="yes", use_geom="yes", 
               cN3LO=None, a6c=None):
        """
        Create the dictionary of parameters for EOBRunPy
        """
        pardic = {
            'M'                  : M,
            'q'                  : q,
            'chi1'               : chi1z,
            'chi2'               : chi2z,
            'LambdaAl2'          : l1,
            'LambdaBl2'          : l2,
            'distance'           : 1.,
            'initial_frequency'  : f0,
            'use_geometric_units': use_geom,
            'interp_uniform_grid': interp,
            'domain'             : 0,
            'srate_interp'       : srate,
            'inclination'        : iota,
            'output_hpc'         : "no",
            'use_mode_lm'        : [0,1,2,3,4,5,6,7,8,13],    # List of modes to use
            'arg_out'            : arg_out,                   # output dynamics and hlm in addition to h+, hx
            'ecc'                : ecc,
            'r_hyp'              : r_hyp,
            'H_hyp'              : H_hyp,
            'j_hyp'              : J_hyp,
            'coalescence_angle'  : np.pi/2. - phi_ref,
            'ecc_freq'           : 2,
            'df'                 : df,
            'anomaly'            : anomaly,
            'model'              : "Giotto"
        }

        # We are not interested in precessing stuff for now (I think?)
        pardic['chi1x'] = pardic['chi1y'] = 0.
        pardic['chi2x'] = pardic['chi2y'] = 0.
        pardic['chi1z'] = chi1z; pardic['chi2z'] = chi2z

        if a6c is not None:
            pardic['a6c'] = a6c
        if cN3LO is not None:
            pardic['cN3LO'] = cN3LO

        return pardic
