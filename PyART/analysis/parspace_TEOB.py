import time, matplotlib, sys, os, argparse, subprocess, copy
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import EOBRun_module 
from   scipy.signal import find_peaks
from   scipy.optimize import brentq
from   PyART.models.teob import search_radial_turning_points, RadialPotential, SpinHamiltonian

matplotlib.rc("text", usetex=True)


def build_colormap(old_cmp_name, clr, peaks_list, continuous_cmap=False):
    """
    Outputs a colormap similar to old_cmp_name, but with a new specified RGBA color
    associated to val.

    Parameters
    ----------
    old_cmp_name : str
        Name of the colormap to be used as a base.
    clr : list
        RGBA color to be associated to val.
    peaks_list : list
        List of values to be represented in the colormap.
    discrete_cmap : bool, optional
        If True, the colormap will be discrete. Default is False.

    Returns
    -------
    newcmp : ListedColormap
        The modified colormap.
    """
    from matplotlib.colors import ListedColormap
    old_cmp = matplotlib.cm.get_cmap(old_cmp_name)
    if discrete_cmap:
        ncolors = int(max(peaks_list))
        newcolors = old_cmp(np.linspace(0, 1, ncolors))  # default
        newcolors[1] = clr
    else:
        nmax = int(max(peaks_list))
        ncolors = 256
        newcolors = old_cmp(np.linspace(0, 1, ncolors))
        for i in range(ncolors):
            if 0.5 * ncolors / nmax < i and i < ncolors / nmax * 1.5:
                newcolors[i] = clr
    else: # discrete cmap
        ncolors   = int(max(peaks_list))-1
        newcolors = old_cmp(np.linspace(0, 1, num=ncolors)) # default
        newcolors[1] = clr
    newcmp = ListedColormap(newcolors)
    return newcmp

#--------------------------
# EOB 
#-------------------------
def rmin_given_spins(chi1,chi2):
    if chi1!=0 or chi2!=0: # important: rmin should be smaller with spin
        rmin = 1.1
    else:
        rmin = 1.3
    return rmin 

def EnergyLimitsSpin(rmax, pph, q, chi1, chi2, N=1000):
    """
    Compute the maximum energy allowed for a given angular momentum pph_hyp,
    mass ratio q, and spins chi1, chi2 for the orbit to be bound.

    Parameters
    ----------
    rmax : float
        Maximum radial separation.
    q : float
        Mass ratio.
    pph_hyp : float
        Angular momentum.
    chi1 : float
        Dimensionless spin of the primary.
    chi2 : float
        Dimensionless spin of the secondary.
    N : int, optional
        Number of points in the radial grid. Default is 100000.

    Returns
    -------
    Emax : float
        The maximum energy allowed for a bound orbit.
    """
    rmin = rmin_given_spins(chi1,chi2)
    rvec = np.linspace(rmin,rmax,N)
    V    = RadialPotential(rvec,pph,q,chi1,chi2)
    Emax = np.nanmax(V)     
    return Emax

def RadialPotential_MaxMin(rmax, pph, q, chi1, chi2, N=1000):
    rmin = rmin_given_spins(chi1,chi2)
    rvec = np.linspace(rmin,rmax,N)
    V    = RadialPotential(rvec,pph,q,chi1,chi2) 
    peaks_idx, _ = find_peaks(V, height=0.9) 
    if len(peaks_idx)>0:
        max_i0 = peaks_idx[0]
        Vmax = V[max_i0]
        rmax = rvec[max_i0]
        specular_idx, _ = find_peaks(1-V[max_i0:])
        if len(specular_idx)>0:
            min_i0 = max_i0+specular_idx[0]
            Vmin = V[min_i0]
            rmin = rvec[min_i0]
        else:
            Vmin = None
            rmin = None
    else:
        Vmax = None
        Vmin = None
        rmax = None
        rmin = None
    return Vmax, Vmin, rmax, rmin


# ---------------------
# Span parameterspace
# ---------------------
class Spanner(object):
    """
    Class to span the (j,E) parameter space using TEOBResumS
    """

    def __init__(
        self,
        q,
        chi1,
        chi2,
        pph_min,
        pph_max,
        nj=100,
        update_pph_min=True,
        r0=1000,
        input_file=None,
        verbose=False,
        nproc=1,
        dump_npz=False,
        ignore_input_file=False,
        outdir=None,
    ):
        """
        Initialize the Spanner class.

        Parameters
        ----------
        q : float
            Mass ratio.
        chi1 : float
            Dimensionless spin of the primary.
        chi2 : float
            Dimensionless spin of the secondary.
        pph_min : float
            Minimum value of the angular momentum.
        pph_max : float
            Maximum value of the angular momentum.
        nj : int, optional
            Number of angular momenta considered. Default is 100.
        update_pph_min : bool, optional
            If True, update pph_min to the minimum value that allows for a bound orbit.
            Default is True.
        r0 : float, optional
            Initial separation. Default is 1000.
        input_file : str, optional
            File with data points. If provided, the TEOB runs will be skipped and the
            points will be loaded from the file. Default is None.
        verbose : bool, optional
            If True, print verbose output. Default is False.
        nproc : int, optional
            Number of processes to use for parallel TEOB runs. Default is 1.
        dump_npz : bool, optional
            If True, dump the points used in an npz file. Default is False.
        ignore_input_file : bool, optional
            If True, ignore the input_file and run TEOB for all points. Default is False.
        outdir : str, optional
            Output directory for data and plots. If None, use the current working directory. Default is None.
        """
        self.q = q
        self.nu = q / (1 + q) ** 2
        self.chi1 = chi1
        self.chi2 = chi2
        self.nj = nj
        self.r0 = r0
        self.pph_min = pph_min
        self.pph_max = pph_max
        self.update_pph_min = update_pph_min
        self.input_file = input_file
        self.verbose  = verbose
        self.vverbose = vverbose
        if self.vverbose:
            self.verbose = True

        # create outdir
        if outdir is None:
            outdir = os.getcwd()
        os.makedirs(outdir, exist_ok=True)        
        self.outdir  = outdir
      
        # check on the input 
        if r0_type!='auto' and r0_type!='val':
            raise ValueError(f"Unknown option for r0_type: '{r0_type}'. Use 'auto' or 'val'")
        elif r0_type=='val' and r0_val is None:
            raise RuntimeError(f"If r0_type is 'val', r0_val must be specified")

        if self.verbose:
            print("\n", "TEOB module:", EOBRun_module.__file__, "\n", sep="")
            print("---------------------------------------")
            print("Input")
            print("---------------------------------------")
            object_namespace = vars(self)
            for attribute, value in object_namespace.items():
                if isinstance(value, (str, float, int)) or value is None:
                    print(f"{attribute:15s}: {value}")
            print(" ")

        self.eobcommonpars = {
              'M'                  : 1,
              'q'                  : q,
              'chi1'               : chi1,
              'chi2'               : chi2,
              'LambdaAl2'          : 0.,
              'LambdaBl2'          : 0.,
              'dt'                 : 0.5,
              'nqc'                : "manual",
              'nqc_coefs_hlm'      : "none",
              'nqc_coefs_flx'      : "none",
              'ode_tmax'           : 1e+6,
              'domain'             : 0,        #Set 1 for FD. Default = 0
              'ode_tstep_opt'      : 1,        #fixing uniform or adaptive. Default = 1 
              'srate_interp'       : 300000.,  #srate at which to interpolate. Default = 4096.
              'use_geometric_units': "yes",    # Output quantities in geometric units. Default = 1
              'interp_uniform_grid': "yes",    #interpolate mode by mode on a uniform grid. Default = 0 (no interpolation)
              'use_mode_lm'        : [1],      #List of modes to use/output through EOBRunPy
              'output_lm'          : [1],      #List of modes to print on file
              'arg_out'            : "yes",    #Output hlm/hflm. Default = 0
              'output_dynamics'    : "no",     #output of the dynamics
              }
       
        if self.input_file is None:
            self.parspace_populate() # define self.points
            self.run_TEOB_parallel(ignore_input_file=ignore_input_file)
        
        else:
            if self.verbose:
                print(f'Loading npz file {input_file}...\n')
            data = np.load(input_file, allow_pickle=True)
            self.points  = data['points']
            meta_are_equal = True
            
            # check that metadata are consistent
            meta = data['meta'][()]
            for key, val in meta.items():
                self_attr = getattr(self,key)
                if isinstance(val,float) and isinstance(self_attr,float):
                    meta_are_equal = meta_are_equal and np.isclose(self_attr, val, atol=1e-12)
                elif isinstance(val,str):
                    meta_are_equal = meta_are_equal and self_attr==val
                elif val is None and self_attr is not None:
                    meta_are_equal = False
                    break
            
            if not meta_are_equal:           
                print('+++ WARNING +++\ninput file not consistent with values specified in input\nOverwriting input values:')
                self.q       = meta['q']
                self.nu      = self.q/(1+self.q)**2
                self.chi1    = meta['chi1']
                self.chi2    = meta['chi2']
                self.nj      = meta['nj']
                self.r0_type = meta['r0_type']
                self.r0_val  = meta['r0_val']

                object_namespace = vars(self)
                for attribute, value in object_namespace.items():
                    if isinstance(value, (str, float, int)) or value is None:
                        print(f"{attribute:15s}: {value}")
                print(" ")

            npoints = len(self.points)
            self.nj = int(0.5*(np.sqrt(1+8*npoints)-1))
        
        if self.verbose:
            print('---------------------------------------')
            print('Updated angular momenta and energies')
            print('---------------------------------------')
            print('number of points : {:d} (nj={:d})'.format(len(self.points),self.nj))
            print('max pph          : {:.5f}'.format(max(self.points[:,0])))
            print('min pph          : {:.5f}'.format(min(self.points[:,0])))
            print('max E            : {:.5f}'.format(max(self.points[:,1])))
            print('min E            : {:.5f}\n'.format(min(self.points[:,1])))
        return

    def info_string(self, q_prec=2, chi1_prec=2, chi2_prec=2, r0_prec=0):
        """
        Write an info string based on the parameters of the Spanner instance.
        Parameters
        ----------
        q_prec : int, optional
            Precision for the mass ratio in the info string. Default is 2.
        chi1_prec : int, optional
            Precision for the primary spin in the info string. Default is 2.
        chi2_prec : int, optional
            Precision for the secondary spin in the info string. Default is 2.
        r0_prec : int, optional
            Precision for the initial separation in the info string. Default is 0.
        Returns
        -------
        str
            The info string.
        """
        template = "q{:.@q_prec@f}_chi1@SIGN1@{:.@chi1_prec@f}_chi2@SIGN2@{:.@chi2_prec@f}_r0{:.@r0_prec@f}_nj{:d}"
        info_str = template.replace("@q_prec@", str(q_prec))
        info_str = info_str.replace("@chi1_prec@", str(chi1_prec))
        info_str = info_str.replace("@chi2_prec@", str(chi2_prec))
        info_str = info_str.replace("@r0_prec@", str(r0_prec))

        def return_sign_str(chi):
            if chi >= 0:
                sign = "p"
            else:
                sign = "m"
            return sign
        info_str = info_str.replace('@SIGN1@', return_sign_str(self.chi1))
        info_str = info_str.replace('@SIGN2@', return_sign_str(self.chi2))
        info_str = info_str.format(self.q, abs(self.chi1), abs(self.chi2), self.nj)
        if self.r0_type=='auto':
            info_str += '_r0auto'
        else:
            info_str += '_r0{:.2f}'.format(self.r0_val)
        return info_str
    
    def bracketing(self,f,start,end,step_size):
        bracketed_intervals = []
        a  = start
        b  = a + step_size
        fa = f(a)
        while b<=end:
            fb = f(b)
            if fa * fb < 0:
                bracketed_intervals.append([a,b])
                a  = b
                fa = f(a)
            b += step_size
        bracketed_intervals.append([a,end])
        return bracketed_intervals

    def determine_r0_from_pphE(self, pph, E, apa_tol=None, rVmin=None, debug_plot=False):
        if E>=1: # unbound case
            if self.r0_type=='auto':
                r0 = self.r_infty
            elif self.r0_type=='val':
                r0 = self.r0_val
        else: # bound case 
            #r_apa = self.search_apastron(pph, E)    
            _, r_apa = search_radial_turning_points(self.q, self.chi1, self.chi2, pph, E, step_size=0.1) 
            if self.r0_type=='auto':
                r0 = r_apa
                if debug_plot:
                    print(f'   check at pph0, E0, r0 : {pph:8.5f} {E:8.5f} {r0:8.3f}')
                    rvec = np.linspace(2,100,num=1000)
                    V = RadialPotential(rvec,pph,self.q,self.chi1,self.chi2)
                    plt.figure()
                    plt.plot(rvec, V, c='r')
                    plt.axhline(E)
                    plt.axvline(r_apa)
                    plt.show()
                if apa_tol is not None:
                    r0 -= apa_tol
            elif self.r0_type=='val':
                if self.r0_val<=r_apa:
                    r0 = self.r0_val
                else:
                    r0 = None
             
            if rVmin is None: rVmin = 0.0 # fictitious value
            if r0 is not None and r0<rVmin:
                if self.vverbose:
                    print(f'   discarding pph0, E0, r0 : {pph:8.5f} {E:8.5f} {r0:8.3f}  (rVmin :{rVmin:7.3f})')
                r0 = None
            elif r0 is None and self.vverbose:
                print(f'   discarding pph0, E0, r0 : {pph:8.5f} {E:8.5f}   nan     (rVmin :{rVmin:7.3f})')
        return r0
            
    def compute_r0_from_IClist(self, indeces, X, apa_tol):
        for i in indeces:
            row    = X[i,:]
            pph0   = row[0]
            E0     = row[1]
            rVmin  = row[2]
            row[3] = self.determine_r0_from_pphE(pph0, E0, apa_tol=apa_tol, rVmin=rVmin)
            X[i,:] = row
        return X

    def parspace_populate(self):
        t0 = time.perf_counter()
        nj = self.nj
        rmax = 10.0 # used in seach of E_max
        if self.verbose:
            print('>> Creating grid')

        # 1) search pph such that Vmax~1
        if abs(self.chi1+self.chi2)<1e-5:
            pph_search_Vmax1 = np.linspace(3.5,4.0,num=2000)
        else:
            pph_search_Vmax1 = np.linspace(2.0,5.0,num=6000)
        for j in pph_search_Vmax1:
            E_max_j = EnergyLimitsSpin(rmax, j, self.q, self.chi1, self.chi2, N=1000)
            if E_max_j>=1:
                pph_Vmax_approx1 = j
                break
        if self.verbose:
            print('   pph such that Vmax~1: {:.6f} (Vmax={:.6f})'.format(pph_Vmax_approx1, E_max_j))
        if self.Emin>1.0-1e-14:
            self.pph_min = pph_Vmax_approx1
        
        # 2) ubound part of the parspace
        pph_sample_unbound = np.linspace(pph_Vmax_approx1, self.pph_max, num=nj)
        points_unbound = []
        if self.Emax>1.0:
            E_max_pph = np.empty(np.shape(pph_sample_unbound))
            for i,pph in enumerate(pph_sample_unbound):
                E_max_pph[i] = max(1,EnergyLimitsSpin(rmax, pph, self.q, self.chi1, self.chi2, N=1000))
            for i in range(nj):
                for j in range(nj):
                    if j<=i:
                        E0 = E_max_pph[j]
                        if E0>self.Emax:
                            continue
                        pph0 = pph_sample_unbound[i]
                        r0   = self.determine_r0_from_pphE(pph0,E0)
                        points_unbound.append([pph0, E0, r0, 0.0, 0.0 ])
            npoints_unbound = len(points_unbound)
            if self.verbose:
                print(f'   considering {npoints_unbound:5d} configurations for unbound motion')
        else:
            npoints_unbound = 0
        points_unbound = np.array(points_unbound)
        
        # 3) bound part of the parspace
        Emin = self.Emin
        points_bound = []
        if Emin<1.0:
            apa_tol = 1e-4 

            dj          = pph_sample_unbound[2]-pph_sample_unbound[1] # dj used in the unbound case 
            num_new_pph = int((pph_Vmax_approx1-self.pph_min)/dj) # new (lower) pph to consider in the bound case
            new_pph     = np.linspace(self.pph_min, pph_Vmax_approx1, num=num_new_pph)
            pph_sample  = np.unique(np.concatenate( (new_pph, pph_sample_unbound) )) # concatenate
            
            if self.dE_bound is None:
                if 'E_max_pph' in locals(): # defined if Emax>1
                    dE = E_max_pph[-1]-E_max_pph[-2]
                else:
                    dE = 0.001 
            else:
                dE = self.dE_bound
            
            npph       = len(pph_sample)
            Emax_bound = min(self.Emax,1)
            nE         = int((Emax_bound-self.Emin)/dE)
            bounded_E  = np.linspace(self.Emin, Emax_bound, num=nE)
            
            # determine points that are energetically allowed
            if self.verbose:
                print('   checking potentials for bounded cases (serial)' )
            info_to_compute_r0 = []
            for i in range(nE):
                for j in range(npph):
                    E0   = bounded_E[i]
                    pph0 = pph_sample[j]
                    Vmax, Vmin, rVmax, rVmin = RadialPotential_MaxMin(100,pph0,self.q,self.chi1,self.chi2,N=1000)     
                    if Vmax is None or Vmin is None:
                        continue
                    if Vmax>=E0 and Vmin<=E0:
                        info_to_compute_r0.append( [pph0, E0, rVmin, 0.0] ) # last column will be filled with r0  
                    elif E0<Vmin:
                        break
            info_to_compute_r0 = np.array(info_to_compute_r0)
            
            # compute apatra or check r0 given in input for points found above (parallel)
            if self.verbose:
                print('   computing/checking r0 for {:d} bounded configurations - {:d} core(s)'.format(len(info_to_compute_r0),self.nproc))
            Y = self.__parallel_run(self.compute_r0_from_IClist, info_to_compute_r0, apa_tol) 
            for i in range(len(Y[:,0])):
                pph0 = Y[i,0]
                E0   = Y[i,1]
                r0   = Y[i,3]
                if not np.isnan(r0):
                    points_bound.append( [pph0, E0, r0, 0.0, 0.0] ) # pph0, E0, r0, (npeaks, rend)
            npoints_bound = len(points_bound)
             
        else:
            npoints_bound = 0
        points_bound  = np.array(points_bound)
        
        if npoints_bound>0 and npoints_unbound>0:
            points = np.concatenate((points_unbound, points_bound), axis=0)
        elif npoints_bound==0 and npoints_unbound>0:
            points = points_unbound
        elif npoints_bound>0 and npoints_unbound==0:
            points = points_bound
        else:
            raise RuntimeError('Empty list of grid-points')
        
        self.points  = points
        self.npoints = len(points[:,0]) 
        if self.verbose:
            print('   points: {:d} (unbound: {:d}, bound: {:d})'.format(self.npoints, npoints_unbound, npoints_bound))
            print('   elapsed time {:.2f} s\n'.format(time.perf_counter()-t0))
        return

    def single_run_TEOB(self,point):
        """
        Run a single TEOB simulation for given initial conditions.
        Parameters
        ----------
        point : list
            List containing [j_hyp, H_hyp].
        Returns
        -------
        list
            List containing [j_hyp, H_hyp, npeaks, rend].
        """
        eobpars = copy.deepcopy(self.eobcommonpars)
        eobpars['j_hyp'] = point[0]
        eobpars['H_hyp'] = point[1]
        eobpars['r_hyp'] = point[2]
        
        t, hp, hc, hlm, dyn = EOBRun_module.EOBRunPy(eobpars)    
        Eend = dyn['E'][-1]
        rend = dyn['r'][-1]

        # for long runs it may happens that ode_tmax is reached before merger, i.e. mergers can be mistaken for scatterings
        # To avoid issues, check the final energy and the final radius. If E_final<1 but no merger, then increase ode_tmax
        maxiter             = 2
        ode_increase_factor = 10
        strategy            = 'ignore' # Strategy if maxiter is reached:'skip' or 'ignore' 
        # 'skip'   : skip this configuration, i.e. leave a white dot in the plot
        # 'ignore' : raise a print-warning, but then use the data anyway (to avoid white spaces in the plot) 
        counter = 0
        while Eend<1 and rend>3:
            if counter==maxiter:
                if self.verbose:
                    print(f'   Warning: (E0,pph0)=({point[1]:.5f}, {point[0]:.5f}) - ', end='')
                    print(f'(Eend,rend)=({Eend:.10f}, {rend:.1e}) - maxiter ({maxiter}) reached, strategy: {strategy}')
                    if strategy=='skip':
                        dyn['E'] = 0*dyn['E'] # with this value set npeaks and rend to NaN, see next if-statement
                    elif strategy=='ignore':
                        pass # do nothing
                    else:
                        raise RuntimeError(f"Unknown strategy: '{strategy}' - strategy should be 'skip' or 'ignore'")
                break
            eobpars['ode_tmax'] = ode_increase_factor*eobpars['ode_tmax'] 
            t, hp, hc, hlm, dyn = EOBRun_module.EOBRunPy(eobpars)    
            Eend = dyn['E'][-1]
            rend = dyn['r'][-1]
            counter += 1
        
        Energy = dyn['E']
        if Energy[-1]<1e-5 or Energy[-1]>point[1]:
            if self.vverbose:
                print('   TEOB failure at pph0, E0, r0 : {:8.5f} {:8.5f} {:7.3f}, final En: {:9.5f}'.format(point[0], point[1], point[2], Energy[-1]))
            npeaks = np.nan
            rend   = np.nan
        else:
            OmgOrb  = dyn['MOmega_orb']
            T       = dyn['t'] 
            mask    = T>=10
            peaks,_ = find_peaks(OmgOrb[mask])
            npeaks  = len(peaks)
            rend    = dyn['r'][-1]
        return [point[0], point[1], point[2], npeaks, rend]

    def run_TEOB_list(self, indeces, points, eobcommonpars):
        """
        Run TEOB simulations for a list of points
        Parameters
        ----------
        indeces : list
            List of indices of the points to be simulated.
        points : ndarray
            Array of points containing [j_hyp, H_hyp].
        eobcommonpars : dict
            Dictionary containing common EOB parameters.
        Returns
        -------
        ndarray
            Array of points containing [j_hyp, H_hyp, npeaks, rend].
        """
        out = points
        for i in indeces:
            out[i,:] = self.single_run_TEOB(points[i])
        return out
    
    def __parallel_run(self, function, X, *additional_args):
        N       = len(X)
        nbatch  = int(np.ceil(N/self.nproc))
        batches = [list(range(i, min(i + nbatch, N))) for i in range(0, N, nbatch)]
        tasks   = []
        Y = np.empty(np.shape(X))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for i in range(self.nproc):
                task = executor.submit( function, batches[i], X, *additional_args)
                tasks.append(task)
            concurrent.futures.wait(tasks, return_when=concurrent.futures.ALL_COMPLETED)
            results = [future.result() for future in tasks]
        for i, batch in enumerate(batches):
            result = results[i]
            for j in batch:
                Y[j,:] = result[j,:]
        return Y

    def run_TEOB_parallel(self, dump_npz=False, dump_txt=False, ignore_input_file=False):
        if self.input_file is not None and not ignore_input_file:
            if self.verbose:
                print(">> Provided input file and ignore_input_file is False, skipping TEOB-runs")
            return
        
        if self.verbose:
            print(f'>> Running TEOB using {self.nproc} core(s)')
        
        t0 = time.perf_counter() 
        self.points = self.__parallel_run(self.run_TEOB_list, self.points, self.eobcommonpars)
        
        if self.verbose:
            print('   elapsed time  : {:.2f} s\n'.format(time.perf_counter()-t0))
        
    def dump_npz(self):
        fname = "data_"+self.info_string()+".npz"
        meta  = {'q':self.q, 'chi1':self.chi1, 'chi2':self.chi2, 'nj':self.nj, 
                 'r0_type':self.r0_type, 'r0_val':self.r0_val}
        np.savez(os.path.join(self.outdir,fname), points=self.points, meta=meta, allow_pickle=True)
        if self.verbose:
            print(f'>> created {fname} in {self.outdir}')
        return

    def dump_txt(self):
        fname = "data_"+self.info_string()+".txt"
        np.savetxt(os.path.join(self.outdir,fname), self.points)
        if self.verbose:
            print(f'>> created {fname} in {self.outdir}')
        return 

    def qchi1chi2_list_str(self):
        """
        Write a string representation of the (q, chi1, chi2) parameters.

        Returns
        -------
        str
            The string representation of (q, chi1, chi2).
        """
        str0 = ""
        vals = [self.q, self.chi1, self.chi2]
        for i,v in enumerate(vals):
            if abs(v-int(v))<1e-14 or i==0:
                isint = True
                v_str = "{:.0f}".format(abs(v))
            else:
                isint = False
                v_str = '{:.1f}'.format(abs(v))
            if i>0 and not isint:
                if v>=0:
                    v_str = '+'+v_str
                else:
                    v_str = "-" + v_str
            str0 += v_str
            if i < len(vals) - 1:
                str0 += ","
        return str0
    
    def plot_parspace(self, marker_size=1, savepng=False, figname=None, continuous_cmap=False,show='on', 
                      show_fails=True, Nmax=None, qc_line=False, parabolic_line=False, 
                      show_kankani=False, show_gra_fit=False,
                      grey_fill=False):
        """
        Plot the parameter space

        Parameters
        ----------
        marker_size : int, optional
            Marker size in the plot. Default is 1.
        savepng : bool, optional
            If True, save the plot as a PNG file. Default is False.
        discrete_cmap : bool, optional
            If True, use a discrete color map. Default is False.
        show : str, optional
            If "on", show the plot. Default is "on".
        pph_max_plot : float, optional
            Maximum value of pph to be considered in the plot. Default is None.
        show_NR : bool, optional
            If True, show NR points in the plot. Default is False.
        dset : str, optional
            Dataset name for NR points. Default is "GAUSS_2023".

        """
        points2plot = self.points
        
        mask = points2plot[:,1]>=self.Emin 
        mask = np.logical_and(mask, points2plot[:,1]<=self.Emax)
        mask = np.logical_and(mask, points2plot[:,0]>=self.pph_min)
        mask = np.logical_and(mask, points2plot[:,0]<=self.pph_max)
        if Nmax is not None:
            mask = np.logical_and(mask, points2plot[:,3]<=Nmax)
        points2plot = points2plot[mask]
        
        N   = points2plot[:,3] #points2plot[:,2] 
        cmp = build_colormap('jet', [1,0,1,1], N, continuous_cmap=continuous_cmap)
        
        mask_nan = np.isnan(N)
        
        plt.figure
        plt.grid(which='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.5)#, zorder=1)
        plt.scatter(points2plot[~mask_nan,0], points2plot[~mask_nan,1], c=N[~mask_nan], cmap=cmp, s=marker_size)#, zorder=3)
        plt.xlabel(r'$p_\varphi^0$', fontsize=15)
        plt.ylabel(r'$E_0/M$', fontsize=15)
        cb = plt.colorbar()
        cb.locator = matplotlib.ticker.MaxNLocator(integer=True)
        cb.update_ticks()
        cb.set_label(r'$N\;{\rm(number\;of\;\Omega\;peaks)}$', fontsize=15)
        if show_fails:
            plt.scatter(points2plot[mask_nan,0], points2plot[mask_nan,1], c='gray', s=1, marker='x', zorder=15)
        
        Emin    = min(points2plot[:,1])
        Emax    = max(points2plot[:,1])
        pph_min = min(points2plot[:,0])
        pph_max = max(points2plot[:,0])
        def write_text(xrel,yrel,text,fontsize=17,clr='k'):
            xplot = pph_min+xrel*(pph_max-pph_min)
            yplot = Emin+yrel*(Emax-Emin)
            plt.text(xplot, yplot, text, fontsize=fontsize, color=clr)
        write_text(0.02, 0.97, r"$(q,\chi_1,\chi_2)=("+self.qchi1chi2_list_str()+")$")
        write_text(0.70, 0.34, r"${\rm scattering}$", clr=[1,1,1,1], fontsize=18)

        y0 = 0.60
        write_text(0.10, y0, r"$E_0>V_0^{\rm max}$")
        #write_text(0.13, y0, r"$E_0>E_0^{\rm max}$")
        
        if parabolic_line:
            plt.axhline(1, color=[0.7,0.7,0.7], zorder=5, lw=0.4)

        if qc_line and Emin<1:
            npph   = 1000
            qc_pph = np.linspace(pph_min, pph_max, num=npph)
            qc_E   = np.zeros_like(qc_pph)
            if pph_max>5:
                rmax_search = 100
            else:
                rmax_search = 30
            for i in range(npph):
                qc_pph_i = qc_pph[i]
                _, Vmin, _, rVmin = RadialPotential_MaxMin(rmax_search, qc_pph_i, self.q, self.chi1, self.chi2, N=1000)
                if Vmin>Emin:
                    qc_E[i] = Vmin
                else:
                    qc_E[i] = np.nan
            plt.scatter(qc_pph, qc_E, color=[0.6,0.6,0.6], s=1, zorder=5) 
        
        if show_kankani:
            if not np.allclose([self.q,self.chi1,self.chi2], [1,0,0]):
                print('Warning: Kankani-McWilliams fit works only for nonspinning equal mass. Skipping...')
            else:
                ca =  3.8733
                cb = -6.7554
                cc =  2.8823
                E_kk = np.linspace(1+1e-10, Emax+0.1, num=100000)
                J_kk = (ca*E_kk**2+cb*E_kk+cc)/(E_kk-1)
                pph_kk = J_kk*4
                mask = pph_kk<=self.pph_max # old label: 2404.03607
                plt.plot(pph_kk[mask], E_kk[mask], color='r', label="2024 - Kankani,\nMcWilliams", zorder=5)
        
        if show_gra_fit:
            if not np.allclose([self.q,self.chi1,self.chi2], [1,0,0]):
                print('Warning: our fit works only for nonspinning equal mass. Skipping...')
            else:
                ca =  15.3129 
                cb = -26.5987
                cc =  11.2860  
                E_fit = np.linspace(1+1e-10, Emax+0.1, num=100000)
                pph_fit = (ca*E_fit**2+cb*E_fit+cc)/(E_fit-1)
                mask = pph_fit<=self.pph_max
                plt.plot(pph_fit[mask], E_fit[mask], color=[1,0.7,0], label=r'\texttt{GR-Athena++} fit', zorder=5)

        if grey_fill:
            xmin = pph_min*0.990
            xmax = pph_max*1.012
            ymin = Emin*0.996
            ymax = Emax*1.003
            
            # rectangular region
            x = np.linspace(xmin, pph_max, 1000)
            y = np.linspace(Emin, ymax, 1000)
            X, Y = np.meshgrid(x, y)
            
            # create triangle and corresponding mask 
            triangle_points = [(pph_min,Emin-1e-5), (pph_max,Emin), (pph_max,Emax*0.98), (pph_min,Emin-1e-5)]
            triangle_path   = matplotlib.path.Path(triangle_points)
            mask = triangle_path.contains_points(np.vstack((X.flatten(), Y.flatten())).T)
            mask = mask.reshape(X.shape)

            plt.scatter(X[~mask], Y[~mask], color=[0.9,0.9,0.9], s=1, zorder=1, alpha=0.05)            

            plt.xlim([xmin,xmax])
            plt.ylim([ymin,ymax])
        
        if savepng:
            if figname is None:
                figname = 'plot_parspace_'+self.info_string()+'.png'
            #figname = figname.replace('.','d')+'.png'
            plt.savefig(os.path.join(self.outdir,figname), dpi=300)
            if self.verbose:
                print(f">> {figname} saved in {self.outdir}")
        if show == "on":
            plt.show()
        return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n", "--nproc", default=1, type=int, help="number of processes"
    )
    parser.add_argument(
        "--jmin",
        default=3.5,
        type=float,
        help="min value of pph considered (eventually updated)",
    )
    parser.add_argument(
        "--jmax", default=5.0, type=float, help="max value of pph considered"
    )
    parser.add_argument(
        "--jmax_plot",
        default=None,
        type=float,
        help="max value of pph considered in the plot (and only there)",
    )
    parser.add_argument(
        "-q", "--mass_ratio", default=1.0, type=float, help="mass ratio"
    )
    parser.add_argument("--chi1", default=0.0, type=float, help="primary spin")
    parser.add_argument("--chi2", default=0.0, type=float, help="secondary spin")
    parser.add_argument(
        "--nj", default=20, type=int, help="number of angular momenta considered"
    )
    parser.add_argument(
        "--marker_size", default=1, type=int, help="marker size in parspace-plot"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose option")
    parser.add_argument(
        "--savepng", action="store_true", help="save parspace-plot as png"
    )
    parser.add_argument(
        "--discrete_cmap",
        action="store_true",
        help="use discrete color-map in parspace plot",
    )
    parser.add_argument(
        "--dump_npz", action="store_true", help="dump the points used in npz file"
    )
    parser.add_argument(
        "-i", "--input_file", default=None, type=str, help="file with data points"
    )
    parser.add_argument(
        "-o", "--outdir", default=None, type=str, help="outdir for data and plots"
    )
    parser.add_argument(
        "--show",
        choices=["on", "off"],
        default="on",
        type=str,
        help="Show parspace-plot",
    )

    args = parser.parse_args()

    if args.mass_ratio < 1:
        args.mass_ratio = 1 / args.mass_ratio

    nproc = args.nproc
    pph_min = args.jmin
    pph_max = args.jmax
    q = args.mass_ratio
    chi1 = args.chi1
    chi2 = args.chi2
    nj = args.nj

    spanner = Spanner(
        q=q,
        chi1=chi1,
        chi2=chi2,
        pph_min=pph_min,
        pph_max=pph_max,
        nj=nj,
        update_pph_min=True,
        nproc=nproc,
        dump_npz=args.dump_npz,
        ignore_input_file=False,
        input_file=args.input_file,
        verbose=args.verbose,
        outdir=args.outdir,
    )
    spanner.plot_parspace(
        marker_size=args.marker_size,
        savepng=args.savepng,
        discrete_cmap=args.discrete_cmap,
        show=args.show,
        pph_max_plot=args.jmax_plot,
    )
