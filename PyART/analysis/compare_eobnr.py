# Code based on the Rossella's scripts in eob_hyperbolic
import os, sys, time, subprocess, argparse, json, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.fftpack import fft, fftfreq, ifft
from scipy.signal import find_peaks

# GW-related imports
from pycbc.filter.matchedfilter import optimized_match
from pycbc.types.timeseries import TimeSeries
from pycbc.psd.analytical   import aLIGOZeroDetHighPower

# local 
repo_path = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], \
                             stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
sys.path.insert(0, os.path.join(repo_path, 'py/teob'))
sys.path.insert(0, os.path.join(repo_path, 'py/sims'))
sys.path.insert(0, os.path.join(repo_path, 'py/others'))
import EOBRun_module, utils, coordschange as cc
import scattering_angle as scat
from process_data import ProcessData
from TEOB_info    import TEOB_info
from simulations  import Sim as SimClass

matplotlib.rc('text', usetex=True)

class CompareEOBNR(object):
    def __init__(self, **kwargs):
        # default options
        self.dset             = 'GAUSS_2023'
        self.integral         = 'FFI'
        self.FFI_f0           = 0.01
        self.TDI_degree       = 1
        self.TDI_poly_int     = None
        self.skip_psi4_extrap = False
        self.verbose          = False
        self.M_min            = 100.
        self.M_max            = 300.
        self.M_ref            = 250.       # reference mass at which the guess E0, J0 are specified.
        self.f_min            = 11.        # Minimum frequency at which the mismatch is computed.
        self.f_max            = 512.       # Maximum frequency at which the mismatch is computed.
        self.sample_rate      = 8192
        self.opt_maxiter      = 1000       # Maximum number of iterations when optimising initial conditions through dual annealing
        self.opt_seed         = 190521     # Random seed used when optimising initial conditions through dual annealing
        self.epsilon_e        = 0.0001     # Semi-interval in which to optimise the E0 value guessed from NR
        self.epsilon_j        = 0.020      # Semi-interval in which to optimise the J0 value guessed from NR
        self.peaks_penalty    = 0.2        # factor in front of the peak-penalty during optimization 
        self.resize_factor    = 16         # Factor to resize the waveforms
        self.Tmax_after_mrg   = 150        # cut NR signal after tmrg+150/M
        self.json_file        = None
        self.lmmax            = 5          # lm-max value used for the energetics (TODO: plot to implement)

        # update default options
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f'Unknown option: {key}')
        
        self.dT = 1.0/self.sample_rate

        if self.verbose:
            for attribute in dir(self):
                if not attribute.startswith('__') and not callable(getattr(self, attribute)):
                    value = getattr(self,attribute)
                    print(f'{attribute:20s} : {value}')
            print(' ')
        
        options = {}
        for attribute in dir(self):
            if not attribute.startswith('__') and not callable(getattr(self, attribute)):
                options[attribute] = getattr(self, attribute)
        self.options = options
        
        self.eobcommonpars = {
            'LambdaAl2'          : 0.,
            'LambdaBl2'          : 0.,
            'nqc'                : "manual",
            'nqc_coefs_hlm'      : "none",
            'nqc_coefs_flx'      : "none",
            #'ode_tmax'           : 2e+5,
            'inclination'        : 0.,
            'distance'           : 1.,
            'domain'             : 0,                 #Set 1 for FD. Default = 0
            'interp_uniform_grid': "yes",             #interpolate mode by mode on a uniform grid. Default = 0 (no interpolation)
            'use_mode_lm'        : [1],               #List of modes to use/output through EOBRunPy
            'output_lm'          : [1],               #List of modes to print on file
            'arg_out'            : "yes",             #Output hlm/hflm. Default = 0
            'output_dynamics'    : "no",              #output of the dynamics
            'output_multipoles'  : "no", 
            'output_hpc'         : "no", 
            }
        
        eob_info = TEOB_info(EOBRun_module) 
        branch   = eob_info['branch']
        commit   = eob_info['commit']
        softlink = eob_info['softlink']
        utils.vprint(f'>> TEOB version:\n   branch   : {branch}\n   commit   : {commit}\n   softlink : {softlink}\n', verbose=self.verbose)
        self.options['teob_branch'] = branch
        self.options['teob_commit'] = commit
        
        self.mm_data = self.load_or_create_mm_data()
        return
    
    def dictionaries_are_equal(self, dict1, dict2, excluded_keys):
        if set(dict1.keys()) - set(excluded_keys) != set(dict2.keys()) - set(excluded_keys):
            return False
        for key in set(dict1.keys()) - set(excluded_keys):
            if dict1[key] != dict2[key]:
                return False
        return True

    def load_or_create_mm_data(self):
        json_file = self.json_file
        if json_file is not None and os.path.exists(json_file):
            with open(json_file, 'r') as file:
                mm_data = json.loads(file.read())    
            # now check that the option in the json file are the same used here
            if not self.dictionaries_are_equal(mm_data['options'], self.options, ['verbose', 'json_file']):
                print(f'Options in {json_file}')
                for key, value in self.options.items():
                    json_value = mm_data['options'][key]
                    if value is None:
                        value = "None"
                    if json_value is None:
                        json_value = "None"
                    print(f"{key:16s} - json: {json_value:<20}   self: {value:<20}")
                raise RuntimeError('The options in the json-file are different w.r.t. the current ones. Exit...')
        else:
            Sim = SimClass(dset=self.dset)
            mm_data = {}
            mm_data['options'] = self.options
            mm_data[self.dset] = {}
            for sim_name in Sim.simlist():
                sim_data = Sim.getsim(sim_name)
                M = sim_data['M1_ADM']+sim_data['M2_ADM']
                mm_data[sim_name]           = {}
                mm_data[sim_name]['E0']     = sim_data['M_ADM']/M
                mm_data[sim_name]['J0']     = sim_data['J_ADM']/M**2
                mm_data[sim_name]['E0_opt'] = None
                mm_data[sim_name]['J0_opt'] = None
                mm_data[sim_name]['mm0']    = None
                mm_data[sim_name]['mm_opt'] = None
        return mm_data
    
    def write_json(self,json_file=None):
        if json_file is None: json_file = self.json_file
        with open(json_file, 'w') as f:
            f.write(json.dumps(self.mm_data,indent=2))
    
    def get_EOB_radius(self, sim, PN_order=2):
        meta  = sim['metadata']
        qa    = np.array([meta['par_b']*2, 0])/sim['M']
        pa    = np.array(meta['par_P_plus'][:-1])/sim['M']/meta['nu']
        qe, _ = cc.Adm2Eob(qa, pa, meta['nu'], PN_order)
        return np.linalg.norm(qe)

    def EOB_h22_from_IC_and_M(self, E0, J0, M, sim, r0=None, dT=None, geo_units=False, complex_TimeSeries=False):
        # Generate EOB wf
        meta  = sim['metadata']
        nu    = meta['nu']
        if r0 is None:
            r0 = self.get_EOB_radius(sim)
        m1    = meta['M1_ADM']/sim['M']*M
        m2    = meta['M2_ADM']/sim['M']*M
        chi1z = meta['par_S_plus'][2]/(sim['M']**2)
        chi2z = meta['par_S_minus'][2]/(sim['M']**2)
        
        pars          = self.eobcommonpars
        pars['M']     = m1+m2
        pars['q']     = m2/m1
        pars['chi1']  = chi1z
        pars['chi2']  = chi2z
        pars['r0']    = r0
        pars['r_hyp'] = r0
        pars['j_hyp'] = J0/nu
        pars['H_hyp'] = E0
        
        if dT is None: dT = self.dT
        pars['srate_interp'] = 1/dT
        if geo_units:
            pars['use_geometric_units'] = 'yes'
        else:
            pars['use_geometric_units'] = 'no'

        t, _, _, hlm, dyn = EOBRun_module.EOBRunPy(pars)
        A_22 = hlm['1'][0]*meta['nu']
        p_22 = hlm['1'][1]
        if complex_TimeSeries:
            h22_eob = A_22*np.exp(-1j*p_22)
        else:
            h22_eob = A_22*np.cos(p_22)
        return t, hlm, TimeSeries(h22_eob, dT), dyn

    def rescale_NR_waveform(self, sim, M, complex_TimeSeries=False):
        t       = sim['u'][(2,2)]/sim['M']
        hlm     = sim['h'][(2,2)]/sim['M']
        dT      = self.dT
        dT_resc = dT/(M*utils.Msuns)
        #tend    = t[-1] -self.rm_last_DeltaT/sim['M']
        tmrg, _ = utils.find_Amax(t, np.abs(hlm))
        tend    = tmrg+self.Tmax_after_mrg
        t_new   = np.arange(t[0], tend, dT_resc)
        _, _, A_nr, p_nr = utils.interpolate_hlm(t, hlm, t_new)
        if complex_TimeSeries:
            h22_nr = A_nr*np.exp(-1j*p_nr)
        else:
            h22_nr = A_nr*np.cos(p_nr)
        return TimeSeries(h22_nr, dT)

    def resize_waveforms(self, h_eob, h_nr):
        dT   = self.dT
        LM   = max(len(h_eob), len(h_nr))
        tl   = (LM-1)*dT
        tN   = utils.nextpow2(self.resize_factor*tl)
        tlen = int(tN/dT)
        h_nr.resize(tlen)
        h_eob.resize(tlen)
        return h_eob, h_nr, tlen
    
    def nr_psi4_peaks(self, sim, height=0.02, time_distance=10.0):
        psi4_amp = np.abs( sim['psi4'][(2,2)] )*sim['M'] 
        dT = (sim['t'][1]-sim['t'][0])/sim['M']
        nr_peaks,_ = find_peaks(psi4_amp, height=height, distance=round(time_distance/dT))
        return nr_peaks

    def compute_mismatch(self, E0J0, M, sim, h_nr=None, h_eob=None):
        E0, J0    = E0J0[0], E0J0[1]
        if h_nr  is None: h_nr  = self.rescale_NR_waveform(sim, M)
        if h_eob is None: h_eob = _,_,h_eob,_ = self.EOB_h22_from_IC_and_M(E0, J0, M, sim)
        h_eob, h_nr, t_len = self.resize_waveforms(h_eob, h_nr)
        delta_f  = 1.0 / h_nr.duration
        f_len    = int(t_len/2 + 1)
        ligo_psd = aLIGOZeroDetHighPower(f_len, delta_f, self.f_min)
        mismatch = 1. - optimized_match(h_nr, h_eob, psd=ligo_psd, 
                                        low_frequency_cutoff=self.f_min,
                                        high_frequency_cutoff=self.f_max)[0]
        return mismatch
    
    def compute_mismatch_wpenalty(self, E0J0, M, sim, peaks_penalty=None, nr_peaks_len=None):
        if self.verbose:
            self.opt_iter += 1 
            print('   mm-iteration               : {:4d}'.format(self.opt_iter), end='\r')
        if peaks_penalty is None: peaks_penalty = self.peaks_penalty
        E0, J0    = E0J0[0], E0J0[1]
        h_nr      = self.rescale_NR_waveform(sim, M)
        _,_,h_eob, dyn_eob = self.EOB_h22_from_IC_and_M(E0, J0, M, sim)
        out = self.compute_mismatch(E0J0, M, sim, h_nr=h_nr, h_eob=h_eob) 
        if peaks_penalty>0.0:
            OmgOrb = dyn_eob['MOmega_orb']
            eob_peaks, _ = find_peaks(OmgOrb)
            if nr_peaks_len is None:
                nr_peaks = self.nr_psi4_peaks(sim)
                nr_peaks_len = len(nr_peaks)
            out += peaks_penalty*( (len(eob_peaks)-nr_peaks_len)**2 )
        return out
    
    def calculate_optimal_initial_conditions(self, sim, verbose=False, fix_energy=False):
        tstart = time.perf_counter()        
        E0_guess = sim['metadata']['M_ADM']/sim['M']
        J0_guess = sim['metadata']['J_ADM']/sim['M']**2
        mm_from_guess = self.compute_mismatch([E0_guess, J0_guess], self.M_ref, sim)
        if verbose:
            sim_name = sim['metadata']['name']
            print(f'   simulation name            : {sim_name}')
            print(f'   First guesses for (E0, J0) : ({E0_guess:.6f},{J0_guess:.6f})')
            print(f'   Mismatch                   : {mm_from_guess:.4f}')
        eps_e       = self.epsilon_e
        eps_j       = self.epsilon_j
        E_low, E_up = E0_guess*(1-eps_e), E0_guess*(1+eps_e)
        J_low, J_up = J0_guess*(1-eps_j), J0_guess*(1+eps_j)
        
        nr_peaks_len = len(self.nr_psi4_peaks(sim))
        f_to_min = lambda E0J0, M, sim: self.compute_mismatch_wpenalty(E0J0, M, sim, \
                                             peaks_penalty=self.peaks_penalty, nr_peaks_len=nr_peaks_len)
         
        self.opt_iter = 0 
        if fix_energy:
            compute_mismatch_fixed = lambda J0: f_to_min([J0, E0_guess], self.M_ref, sim)
            opt_result = optimize.minimize(compute_mismatch_fixed, x0=J0_guess, bounds=[(J_low, J_up)])
            j0_opt = opt_result.x[0]
            e0_opt = E0_guess
        else:
            opt_result = optimize.dual_annealing(f_to_min, bounds=[[E_low, E_up], [J_low, J_up]], \
                                                 args=(self.M_ref, sim), maxfun=self.opt_maxiter,  \
                                                 seed=self.opt_seed)
            opt_pars = opt_result['x']
            e0_opt, j0_opt = opt_pars[0], opt_pars[1]
        if verbose:
            print('   mm-iteration               : {:4d}'.format(self.opt_iter), end='')
            if self.opt_iter>=self.opt_maxiter:
                print(' (maxiter reached)', end='')
            print(' ')
        self.opt_iter = 0 
        
        mismatch = self.compute_mismatch([e0_opt, j0_opt], self.M_ref, sim) 
        tend = time.perf_counter()-tstart
        if verbose:
            print(f'   New estimates for (E0, J0) : ({e0_opt:.6f},{j0_opt:.6f})')
            print(f'   New mismatch               : {mismatch:.4f}')
            print(f'   Elapsed time               : {tend:.2f} s\n')
        return e0_opt, j0_opt, mismatch, mm_from_guess
    
    def mismatch_loop(self,simulations, force_computation=False):
        utils.vprint( '>> Loop on simulations:\n',verbose=self.verbose)
        #sim_class = SimClass(dset=self.dset)
        self.all_process_data = {}
        for sim_name in simulations: #sim_class.simlist():
            if sim_name not in self.all_process_data:
                try:
                    pdata = ProcessData(dset=self.dset, sims=[sim_name], verbose=False, 
                            integral=self.integral, FFI_f0=self.FFI_f0, TDI_degree=self.TDI_degree, TDI_poly_int=self.TDI_poly_int,
                            extrap_psi4=(not self.skip_psi4_extrap), lmmax=self.lmmax)
                    sim = pdata.data[sim_name]
                    self.all_process_data[sim_name] = sim
                except Exception as e:
                    print(f'   Issues with {sim_name}:\n   {e}') 
                    raise
            
            if self.mm_data[sim_name]['mm_opt'] is None or force_computation:
                E0_opt, J0_opt, mm_opt, mm0 = self.calculate_optimal_initial_conditions(sim, verbose=self.verbose)
                opt_results = {}
                opt_results['E0']      = sim['metadata']['M_ADM']/sim['M']
                opt_results['J0']      = sim['metadata']['J_ADM']/sim['M']**2
                opt_results['E0_opt']  = E0_opt
                opt_results['J0_opt']  = J0_opt
                opt_results['mm0']     = mm0
                opt_results['mm_opt']  = mm_opt
                self.mm_data[sim_name] = opt_results

            elif self.verbose:
                E0     = self.mm_data[sim_name]['E0']
                J0     = self.mm_data[sim_name]['J0']
                E0_opt = self.mm_data[sim_name]['E0_opt']
                J0_opt = self.mm_data[sim_name]['J0_opt']
                mm0    = self.mm_data[sim_name]['mm0']
                mm_opt = self.mm_data[sim_name]['mm_opt']
                print(f'   Reading data from {self.json_file}')
                print(f'   simulation name            : {sim_name}')
                print(f'   First guesses for (E0, J0) : ({E0:.6f},{J0:.6f})')
                print(f'   Mismatch                   : {mm0:.4f}')
                print(f'   New estimates for (E0, J0) : ({E0_opt:.6f},{J0_opt:.6f})')
                print(f'   New mismatch               : {mm_opt:.4f}\n')

            if self.json_file is not None:
                self.write_json(self.json_file)
        return

    def return_shift_waveforms(self, t1, signal1, t2, signal2, align_method='chi2'):
        if align_method=='cross-corr':
            signal1 = signal1.real
            signal2 = signal2.real
            signal2_interp = np.interp(t1, t2, signal2)
            cross_corr = np.correlate(signal1, signal2_interp, mode='full')
            idx = np.argmax(cross_corr) - len(signal1) + 1
            shift = t1[idx]
        
        elif align_method=='chi-square':
            #FIXME: this does not work, why? 
            def chi_square_loss(time_shift, t1, signal1, t2, signal2):
                shifted_signal2 = np.interp(t1, t2 - time_shift, signal2, left=np.nan, right=np.nan)
                return np.nanmean((signal1 - shifted_signal2)**2)
            result = optimize.minimize(chi_square_loss, 0.0, args=(t1, signal1.real, t2, signal2.real))
            shift = result.x[0]
        
        elif align_method=='no':
            shift = 0
        
        else:
            raise ValueError(f'Unknown align-method: {align_method}')
        return shift 
    
    def return_MH_after_relax(self, sim, t_relax=30):
        hs0_tmp = sim['horizon_s0']  # 1:iter 2:time 3:mass 4:Sx 5:Sy 6:Sz 7:S 8:area 9:hrms 10:hmean 11:meanradius
        hs1_tmp = sim['horizon_s1'] 
        valid_mask = ~np.isnan(hs0_tmp) & ~np.isnan(hs1_tmp)
        hs0 = np.where(valid_mask, hs0_tmp, 0)
        hs1 = np.where(valid_mask, hs1_tmp, 0)
        th  = hs0[:,1]
        idx_relax = np.argmax(th > t_relax)
        MH   = hs0[idx_relax,2] + hs1[idx_relax,2]
        return MH
    
    def local_vars_for_plots(self, **kwargs):
        def kwargs_or_self(name):
            if name in kwargs.keys():
                return kwargs.get(name)
            elif hasattr(self, name):
                return getattr(self, name)
            else:
                raise RuntimeError(f'Unknown var: {name}')
        loc              = lambda:0
        loc.sims         = kwargs_or_self('sims')
        loc.tlim         = kwargs_or_self('tlim')
        loc.savepng      = kwargs_or_self('savepng')
        loc.showpng      = kwargs_or_self('showpng')
        loc.colors       = kwargs_or_self('colors')
        loc.verbose      = kwargs_or_self('verbose')
        loc.dpi          = kwargs_or_self('dpi')
        loc.plots_labels = kwargs_or_self('plots_labels')
        if not set(loc.sims).issubset(set(self.sims)):
            raise RuntimeError('List of sims specified in plots must be '\
                               'a subset of the ones specified during class-initialization')
        loc.nsims = len(loc.sims)
        if len(loc.plots_labels)!=loc.nsims: #FIXME: set 'auto' options also for labels
            raise ValueError('size of plot_labels incompatible with number of (local) simulations')
        if isinstance(loc.colors, str) and loc.colors=='auto':
            loc.colors = self.auto_colors(loc.nsims) 
        elif len(loc.colors)!=loc.nsims:
            raise ValueError('size of colors incompatible with number of (local) simulations')
        return loc
    
    def plot_psi4_radii(self, figname='plot_psi4_radii.png', **kwargs):

    def plot_eobnr(self, sim_name, M_rescale=None, tlim=None, align_method='cross-corr', plot_time='tmrg', show_diffs=False, plot_rmpi=0, ylim_diffs=[]):
        M      = 1
        sim    = self.all_process_data[sim_name]
        E0     = self.mm_data[sim_name]['E0']
        J0     = self.mm_data[sim_name]['J0']
        if M_rescale is not None:
            print(f'Rescaling IC using M_rescale {M_rescale:.5f}')
            print(f'old E0, J0: {E0:.5f}, {J0:.5f}')
            E0 = E0/M_rescale
            J0 = J0/M_rescale**2
            print(f'new E0, J0: {E0:.5f}, {J0:.5f}')
        #MH = self.return_MH_after_relax(sim, t_relax=30)
        #E0 = sim['metadata']['M_ADM']/MH  
        #J0 = sim['metadata']['J_ADM']/MH**2
        E0_opt = self.mm_data[sim_name]['E0_opt']
        J0_opt = self.mm_data[sim_name]['J0_opt']

        t_eob, hlm_eob_dict, _, _ = self.EOB_h22_from_IC_and_M(E0, J0, M, sim, geo_units=True)
        A_eob = hlm_eob_dict['1'][0]
        p_eob = hlm_eob_dict['1'][1]
        h_eob = A_eob*np.exp(-1j*p_eob)

        t_eob_opt, hlm_eob_opt_dict, _, _ = self.EOB_h22_from_IC_and_M(E0_opt, J0_opt, M, sim, geo_units=True)
        A_eob_opt = hlm_eob_opt_dict['1'][0]
        p_eob_opt = hlm_eob_opt_dict['1'][1]
        h_eob_opt = A_eob_opt*np.exp(-1j*p_eob_opt)
        
        h_nr = sim['h'][(2,2)]/sim['M']/sim['metadata']['nu']
        t_nr = sim['t']/sim['M']
        tmrg,_ = utils.find_Amax(t_nr, np.abs(h_nr))
         
        t_eob     += self.return_shift_waveforms(t_nr, h_nr, t_eob,     h_eob,     align_method=align_method)
        t_eob_opt += self.return_shift_waveforms(t_nr, h_nr, t_eob_opt, h_eob_opt, align_method=align_method) 

        # plot using retarded time
        if plot_time=='tret':
            r_extr = 100 # extraction radius for NR sims
            R_extr = r_extr * (1 + M/(2*r_extr))**2
            rs     = R_extr + 2*M*np.log(R_extr/(2*M)-1)
            gshift = -rs
            xlab   = r'$t-r_*$'
        elif plot_time=='tmrg_nr':
            gshift = -tmrg
            xlab   = r'$t-t_{\rm mrg}$'
        else:
            gshift = 0
            xlab   = r'$t$'
        
        t_nr      += gshift
        t_eob     += gshift
        t_eob_opt += gshift
        
        if len(tlim)!=2:
            tlim  = np.array([0, tmrg+self.Tmax_after_mrg])+gshift
        tlim  = np.array(tlim)

        if show_diffs:
            figsize = (10,8)
            figm    = 2
            fign    = 1
            fig, axs = plt.subplots(figm, fign, figsize=figsize)
            ax = axs[0]
        else:
            fig, ax = plt.subplots(1,1)
            
        ax.plot(t_nr,      h_nr.real,      color='k', label='NR')
        ax.plot(t_eob,     h_eob.real,     color=[0.7,0.7,0.7], label='EOB (not opt)')
        ax.plot(t_eob_opt, h_eob_opt.real, color=[0.8,0,0], label='EOB (optmized)')
        #ax.plot(t_nr+gshift,      h_nr.imag,      color='k', ls='--')
        #ax.plot(t_eob_opt+gshift, h_eob_opt.imag, color='r', ls=':')
        ax.set_xlim(tlim)
        ax.set_xlabel(xlab, fontsize=20)
        ax.set_ylabel(r'$\Re(h_{22})/\nu$', fontsize=20)
        ax.legend(loc='upper left', fontsize=12)
        if plot_time=='tmrg_nr':
            ax.axvline(0,color='k',ls='--')
        if show_diffs:
            amp_nr = np.abs(h_nr)
            phi_nr = -np.unwrap(np.arctan(h_nr.imag/h_nr.real)*2)/2 
            amp_eob_opt = np.abs(h_eob_opt)
            phi_eob_opt = -np.unwrap(np.arctan(h_eob_opt.imag/h_eob_opt.real)*2)/2 
            ts, damp = utils.vec_differences(t_nr,amp_nr,t_eob_opt,amp_eob_opt,tlim[0],tlim[1],0.5,diff_kind='rel',fabs=False)
            _ , dphi = utils.vec_differences(t_nr,phi_nr,t_eob_opt,phi_eob_opt,tlim[0],tlim[1],0.5,diff_kind='phi',fabs=False)
            dphi -= plot_rmpi*np.pi
            axs[1].plot(ts, -damp, color=[1,0.7, 0 ], label=r'$\Delta A^{\rm EOBNR}/A^{\rm NR}$')
            axs[1].plot(ts, -dphi, color=[0,0.8,0.8], label=r'$\Delta \phi^{\rm EOBNR}$')
            #axs[1].plot(t_nr, phi_nr)
            #axs[1].plot(t_eob_opt, phi_eob_opt)
            axs[1].set_xlabel(xlab, fontsize=20)
            axs[1].set_xlim(tlim)
            axs[1].legend(loc='upper left', fontsize=15)
            if plot_time=='tmrg_nr':
                axs[1].axvline(0,color='k',ls='--')
            axs[1].grid()
            if len(ylim_diffs)==2:
                axs[1].set_ylim(ylim_diffs)
        plt.show()
        return
    
    def plot_eobnr_ft(self,sim_name, complex_TimeSeries=True, show_eob=True):
        M      = self.M_ref
        sim    = self.all_process_data[sim_name]
        E0     = self.mm_data[sim_name]['E0']
        J0     = self.mm_data[sim_name]['J0']
        E0_opt = self.mm_data[sim_name]['E0_opt']
        J0_opt = self.mm_data[sim_name]['J0_opt']
        
        h_nr = self.rescale_NR_waveform(sim, M, complex_TimeSeries=complex_TimeSeries)
        h_nr = np.array(h_nr) # TimeSeries to numpy-array
        
        freq_nr    = fftfreq(len(h_nr), 1/self.sample_rate)
        fft_abs_nr = np.abs(fft(h_nr)) # Fourier transform 
        
        plt.figure(figsize=(10,7))
        plt.plot(freq_nr, fft_abs_nr, color='k', label='NR')
        
        if show_eob:
            _,_,h_eob, _     = self.EOB_h22_from_IC_and_M(E0, J0, M, sim, complex_TimeSeries=complex_TimeSeries)
            _,_,h_eob_opt, _ = self.EOB_h22_from_IC_and_M(E0_opt, J0_opt, M, sim, complex_TimeSeries=complex_TimeSeries)
            
            h_eob     = np.array(h_eob)
            h_eob_opt = np.array(h_eob_opt)
            
            freq_eob        = fftfreq(len(h_eob),     1/self.sample_rate)
            freq_eob_opt    = fftfreq(len(h_eob_opt), 1/self.sample_rate)
            fft_abs_eob     = np.abs(fft(h_eob))
            fft_abs_eob_opt = np.abs(fft(h_eob_opt))
            
            plt.plot(freq_eob,     fft_abs_eob,     color=[0.7,0.7,0.7], label='EOB (not opt)')
            plt.plot(freq_eob_opt, fft_abs_eob_opt, color='r', label='EOB (optimized)')
        
        if complex_TimeSeries:
            plt.xlim([-125,25])
        else:
            plt.xlim([-125,125])
        plt.legend()
        plt.show()
        return

    def plot_mismatches_vs_M(self, sim_names, M_min=None, M_max=None, dM=5, labels=None, verbose=None, ylog=True, ylim=None):
        if M_min   is None: M_min   = self.M_min
        if M_max   is None: M_max   = self.M_max
        if verbose is None: verbose = self.verbose
        if labels  is None: labels  = sim_names
        colors = matplotlib.cm.rainbow(np.linspace(0,1,num=len(sim_names)))
        masses = np.linspace(M_min, M_max, num=int((M_max-M_min)/dM)+1 )
        plt.figure
        for i, sim_name in enumerate(sim_names):
            sim    = self.all_process_data[sim_name]
            E0_opt = self.mm_data[sim_name]['E0_opt']
            J0_opt = self.mm_data[sim_name]['J0_opt']
            mm_masses = []
            for j,M in enumerate(masses):
                if verbose:
                    if j==len(masses)-1:
                        end = '\n'
                    else:
                        end = '\r'
                    print(f'Computing mm for {sim_name}: {M:6.2f}/{M_max:.2f}', end=end)
                mm = self.compute_mismatch([E0_opt,J0_opt], M, sim)   
                mm_masses.append(mm)
            plt.plot(masses, mm_masses, label=labels[i], color=colors[i])
        plt.legend()
        if ylog:
            plt.yscale('log')
        if ylim is not None:
            plt.ylim(ylim)
        plt.grid(which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')
        plt.xlabel(r'$M\;[M_\odot]$', fontsize= 15)
        plt.ylabel(r'$\bar{\cal F}$', fontsize= 15)
        plt.show()

    def compute_scattering_angles(self, sim_name, search_low_res=False):
        dashes = '-'*80
        utils.vprint(f'{dashes}\nComputing scattering angles for: {sim_name}\n{dashes}',verbose=self.verbose)
        M      = 1
        E0     = self.mm_data[sim_name]['E0']
        J0     = self.mm_data[sim_name]['J0']
        E0_opt = self.mm_data[sim_name]['E0_opt']
        J0_opt = self.mm_data[sim_name]['J0_opt']
        sim    = self.all_process_data[sim_name]
        rp = np.sqrt((sim['punct_0'][:,2]-sim['punct_1'][:,2])**2 + \
                     (sim['punct_0'][:,3]-sim['punct_1'][:,3])**2)/sim['M']
        
        if rp[-1]<3:
            print(f'+++ Warning +++\n{sim_name} is not a scattering')
            return
        
        nmin      = 2
        nmax      = 4
        n_extract = 3

        # NR scattering angle 
        utils.vprint(f'>>> NR angle: (E0,J0,r0)=({E0:.5f},{J0:.5f},{rp[0]:.2f})', verbose=self.verbose)
        scat_NR = scat.ScatteringAngle(punct0=sim['punct_0'], punct1=sim['punct_1'], file_format='GRA', nmin=nmin, nmax=nmax, n_extract=n_extract,
                                   r_cutoff_out_low=25, r_cutoff_out_high=None, r_cutoff_in_low=25, r_cutoff_in_high=rp[0]-5, 
                                   verbose=self.verbose)
        if search_low_res:
            N = sim['metadata']['nx1'] 
            sim_low_res_found = False
            for s in self.all_process_data.keys():
                if s[:-3]==sim_name[:-3]:
                    Ns = self.all_process_data[s]['metadata']['nx1']
                    if Ns<N:
                        sim_name_low_res = s
                        sim_low_res_found = True
                        break
            if sim_low_res_found:
                utils.vprint(f'Low res sim found: {sim_name_low_res}\n', verbose=self.verbose)
                sim_low_res = self.all_process_data[sim_name_low_res]
                scat.ComputeChiFrom2Sims(punct0_lres=sim_low_res['punct_0'], punct1_lres=sim_low_res['punct_1'],
                                         punct0_hres=sim['punct_0'],         punct1_hres=sim['punct_1'],
                                         file_format='GRA', nmin=nmin, nmax=nmax, n_extract=n_extract, 
                                         r_cutoff_out_low=25, r_cutoff_out_high=None, r_cutoff_in_low=25, r_cutoff_in_high=rp[0]-5, 
                                         verbose=self.verbose)
            else:
                utils.vprint(f'Low res sim not found',verbose=self.verbose)
       
        # Compute EOB IC using u-extrapolation
        xp0 = sim['punct_0'][:,2]/sim['M']
        yp0 = sim['punct_0'][:,3]/sim['M']
        xp1 = sim['punct_1'][:,2]/sim['M']
        yp1 = sim['punct_1'][:,3]/sim['M']
        tp  = sim['punct_0'][:,1]/sim['M']
        rp  = np.sqrt( (xp0-xp1)**2 + (yp0-yp1)**2 )
        tw       = sim['t']/M
        hatErad  = sim['energetics']['E']/sim['M']     # integrated Edot
        hatJrad  = sim['energetics']['Jz']/sim['M']**2 # integrated Jdot
        hatE_sys = E0 - hatErad   
        hatJ_sys = J0 - hatJrad 
        Es = utils.spline(tw, hatE_sys, tp)
        Js = utils.spline(tw, hatJ_sys, tp)
        upoly_E_out = utils.upoly_fits(rp, Es, nmin=1, nmax=5, n_extract=2, direction='in',
                                       r_cutoff_low=12.5, r_cutoff_high=50)
        upoly_J_out = utils.upoly_fits(rp, Js, nmin=1, nmax=5, n_extract=2, direction='in',
                                       r_cutoff_low=12.5, r_cutoff_high=50)
        Einf = upoly_E_out['extrap'] 
        Jinf = upoly_J_out['extrap'] 
        
        rp_after_junk  = 50 # FIXME: this need tuning, fixed for the moment
        idx_after_junk = np.argmax(rp < rp_after_junk)
        
        E_afterjunk  = hatE_sys[idx_after_junk]
        J_afterjunk  = hatJ_sys[idx_after_junk] # use these values but still r0~100
        r0_afterjunk = rp[idx_after_junk]
        
        # E0, J0, r0 (r0=None ---> use 2PN ADM/EOB transformation)
        scat_eob_list = []
        EOB_IC_list = [ [E0, J0, None], [E0_opt, J0_opt, None], [Einf, Jinf, 10000], \
                        [E_afterjunk, J_afterjunk, None], [E_afterjunk, J_afterjunk, r0_afterjunk],  [E_afterjunk, J_afterjunk,10000] ]
        EOB_IC_info = [ 'not-optimized', 'optimized', 'u-extrapolated', 'after junk (old r0)', 'after junk (new r0)', 'after junk (r->inf)']
        counter = 0
        for IC, IC_info in zip(EOB_IC_list, EOB_IC_info):
            E0i = IC[0]
            J0i = IC[1]
            _, _, _, dyn_eob = self.EOB_h22_from_IC_and_M(E0i, J0i, M, sim, geo_units=True, r0=IC[2])
            r0i = dyn_eob['r'][0]
            utils.vprint(f'>>> EOB angle ({IC_info}): (E0,J0,r0)=({E0i:.5f},{J0i:.5f},{r0i:.2f})', verbose=self.verbose)
            if dyn_eob['r'][-1]<3:
                print(f'{IC_info} EOB ID produced bounded configuration!\n')
            else:
                track_eob = np.column_stack((dyn_eob['t'], dyn_eob['r'], dyn_eob['phi']))
                scat_eob  = scat.ScatteringAngle(punct0=track_eob, punct1=None, file_format='EOB', nmin=nmin, nmax=nmax, n_extract=n_extract,
                                       r_cutoff_out_low=25, r_cutoff_out_high=None, r_cutoff_in_low=25, r_cutoff_in_high=None, 
                                       verbose=self.verbose)
                scat_eob_list.append(scat_eob)
                del scat_eob, track_eob
            del dyn_eob
            counter += 1
            if (not self.skip_psi4_extrap) and counter>1 and self.lmmax>5:
                print('Using psi4-extrapolation and llmax>5: skipping EOB runs that involve numerical fluxes')
                break
        return

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dset',         default='GAUSS_2023',  help="select dset: 'GAUSS_2021' or 'GAUSS_2023'")
    parser.add_argument('-s', '--sims', nargs='+', default=[],   help='simulations to consider (either names or indeces)', required=False)
    parser.add_argument('--simfile', type=str,                   help='Txt file with list of sims to consider. If this option is used,'
                                                                      'the -s/--sims option is ignored (only names supported, not indeces)')
    parser.add_argument('-v', '--verbose', action='store_true',  help='print info')
    parser.add_argument('-l', '--list',    action='store_true',  help='Print list of available simulations')
   
    # integration options 
    parser.add_argument('-i','--integral',default='FFI', 
                        choices = ['FFI', 'TDI'],        type=str,   help="method to use for integration")
    parser.add_argument('-f','--FFI_f0',   default=0.01, type=float, help="f0 used in FFI")
    parser.add_argument('-d','--TDI_degree',default=1,   type=int,   help="poly-degree used in TDI")
    parser.add_argument('--TDI_poly_int',  default=None, type=float,
                                            nargs='+',               help="poly-interval used in TDI")
    parser.add_argument('--skip_psi4_extrap', action='store_true',   help="Skip psi4 extrapolation before integrating")
    parser.add_argument('--maxiter', default=1000, type=int,         help="Max iteration in IC optimization")
    parser.add_argument('--peaks_penalty', default=0.2, type=float,  help="Peaks penalty to use in optimization")
    parser.add_argument('--sample_rate', default=8192, type=int,     help="Sample rate to use when resizing waveforms")   
    parser.add_argument('--all', action="store_true",                help="Compute mismatches for all the simulations"
                                                                          " in the dataset")  
    parser.add_argument('--exclude',nargs='+',type=str,default=[],   help="sims to not consider when --all is used")
    parser.add_argument('--json_file',  default=None, type=str,      help="json file where to save results.")
    parser.add_argument('--plots', nargs='+', default=[],
                        choices=['mm_vs_M', 'eobnr', 'eobnr_ft'],    help="Show plots")
    parser.add_argument('--tlim', nargs='+', default=[], type=float, help='tlim to use in time-plots, e.g. 0 700')
    parser.add_argument('--plot_rmpi', default=0, type=int,          help='Subtract specified multiples of pi from phase-diff')
    parser.add_argument('--ylim_diffs',default=[],nargs='+',type=float,help='ylim to use in eob/nr differences')
    parser.add_argument('--epsilon_e', default=0.0001, type=float,   help='Semi-interval in which to optimise the E0'
                                                                          'value guessed from NR. Default is 0.0001')
    parser.add_argument('--epsilon_j', default=0.02,   type=float,   help='Semi-interval in which to optimise the J0'
                                                                          'value guessed from NR. Default is 0.0200')
    parser.add_argument("--lmmax", type=int, default=5,             help='Maximum multipole m=l to consider')
    parser.add_argument("--plot_M_rescale",default=None,type=float, help='Mass used to rescale IC in EOB/NR plot')
    parser.add_argument("--scatterings", action="store_true",       help='Compute scattering angles')

    args = parser.parse_args()
    
    sim = SimClass(dset=args.dset)
    simlist = sim.simlist()
    if args.all:
        simulations = [item for item in simlist if item not in args.exclude]
    else: 
        simulations = [] 
        for elem in args.sims:
            try:
                simulations.append(simlist[int(elem)])
            except ValueError:
                simulations.append(elem)

    if args.simfile is not None:
        with open(args.simfile, 'r') as f: 
            simfile_list = f.readlines()
        simfile_list = [sim.rstrip() for sim in simfile_list]
        if len(simulations)!=0:
            print('Warning: using --simfile, ignoring -s/--sims')
        simulations = simfile_list
    
    for simulation in simulations:
        if simulation not in simlist:
            raise ValueError(f'Unknown simulation: {simulation}')

    if args.list:
        sim.print_simlist()
         
    if len(simulations)==0:
        print('No simulations given in input. Exit...')
        sys.exit(-1)

    eobnr = CompareEOBNR(dset=args.dset, verbose=args.verbose, 
                         integral=args.integral, FFI_f0=args.FFI_f0, TDI_degree=args.TDI_degree, 
                         skip_psi4_extrap=args.skip_psi4_extrap, lmmax=args.lmmax,
                         opt_maxiter=args.maxiter, peaks_penalty=args.peaks_penalty, sample_rate=args.sample_rate,
                         json_file=args.json_file, epsilon_e=args.epsilon_e, epsilon_j=args.epsilon_j)
    eobnr.mismatch_loop(simulations)
    
    for plot in args.plots:
        if plot=='mm_vs_M':
            eobnr.plot_mismatches_vs_M(simulations, ylim=[1e-3, 1e-1])
    
        for sim_name in simulations:
            if plot=='eobnr':
                eobnr.plot_eobnr(sim_name,align_method='cross-corr',show_diffs=True, \
                                 plot_time='tmrg_nr', tlim=args.tlim, plot_rmpi=args.plot_rmpi, \
                                 ylim_diffs=args.ylim_diffs, M_rescale=args.plot_M_rescale)
            if plot=='eobnr_ft':
                eobnr.plot_eobnr_ft(sim_name, complex_TimeSeries=True, show_eob=True)

    for sim_name in simulations:
        if args.scatterings:
            eobnr.compute_scattering_angles(sim_name, search_low_res=True)

