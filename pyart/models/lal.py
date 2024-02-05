try:
    import lalsimulation as lalsim
    import lal  
except ImportError:
    raise ImportError("WARNING: LALSimulation and/or LAL not installed.")
    

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