import time, matplotlib, sys, os, argparse, subprocess
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import concurrent.futures

repo_path = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'],stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
sys.path.insert(1,os.path.join(repo_path,'py/teob'))
sys.path.insert(1,os.path.join(repo_path,'py/sims'))
import EOBRun_module
import simulations

matplotlib.rc('text', usetex=True)

def build_colormap(old_cmp_name, clr, peaks_list, discrete_cmap=False):
    """
    Outputs a colormap similar to old_cmp_name, but with a new specified RGBA color
    associated to val.
    """
    from matplotlib.colors import ListedColormap
    old_cmp   = matplotlib.cm.get_cmap(old_cmp_name)
    if discrete_cmap:
        ncolors   = int(max(peaks_list))
        newcolors = old_cmp(np.linspace(0, 1, ncolors)) # default
        newcolors[1] = clr
    else:
        nmax      = int(max(peaks_list))
        ncolors   = 256
        newcolors = old_cmp(np.linspace(0,1,ncolors))
        for i in range(ncolors):
            if 0.5*ncolors/nmax<i and i<ncolors/nmax*1.5:
                newcolors[i] = clr
    newcmp = ListedColormap(newcolors)
    return newcmp

#--------------------------
# EOB 
#-------------------------

#def Hamiltonian(r, pph, nu):
#    """
#    Circularized EOB Hamiltonian: nonspinning
#    """
#    A, dA, d2A  = EOBRun_module.eob_metric_A5PNlog_py(r, nu)
#    Heff0       = np.sqrt(A*(1.+(pph/r)**2))
#    E0          = np.sqrt(1. + 2.*nu*(Heff0-1.))
#
#    return E0
#
def SpinHamiltonian(r, pph, q, chi1, chi2):
    prstar = 0.
    hatH   = EOBRun_module.eob_ham_s_py(r, q, pph, prstar, chi1, chi2)
    nu     = q/(1+q)**2
    E0     = nu*hatH[0]
    return E0

def RadialPotential(rmin,rmax,pph,q,chi1,chi2,N=100):
    rvec = np.linspace(rmin,rmax,N)
    dr   = rvec[1]-rvec[0]
    V    = np.array([SpinHamiltonian(ri, pph, q, chi1, chi2) for ri in rvec])
    return V, rvec

def EnergyLimitsSpin(rmax, q, pph_hyp, chi1, chi2, N=100000):
    if chi1!=0 or chi2!=0:
        # important: rmin should be smaller with spin
        rmin = 1.1
    else:
        rmin = 1.3
    #Emin = SpinHamiltonian(rmax, pph_hyp, q, chi1, chi2)
    # Determine the max energy allowed. For large q, A will go below zero, so ignore those values by removing nans.
    V, rvec = RadialPotential(rmin,rmax,pph_hyp,q,chi1,chi2,N=N)
    Emax = np.nanmax(V)     
    return Emax

#---------------------
# Run TEOB
#---------------------
# FIXME: not in the Spanner-class because I was having issues with parallelization

def single_run_TEOB(point, eobcommonpars):
    eobpars = eobcommonpars
    eobpars['j_hyp'] = point[0]
    eobpars['H_hyp'] = point[1]
    t, hp, hc, hlm, dyn = EOBRun_module.EOBRunPy(eobpars)    
    T      = dyn['t']
    r      = dyn['r']
    OmgOrb = dyn['MOmega_orb']
    peaks, _ = find_peaks(OmgOrb)
    npeaks = len(peaks)
    rend   = r[-1]
    return [point[0], point[1], npeaks, rend]

def run_TEOB_list(indeces, points, eobcommonpars):
    out = points
    for i in indeces:
        out[i,:] = single_run_TEOB(points[i], eobcommonpars)
    return out

#---------------------
# Span parameterspace
#---------------------
class Spanner(object):
    def __init__(self, q, chi1, chi2, pph_min, pph_max, nj=100, 
                 update_pph_min=True, r0=1000, input_file=None, verbose=False,
                 nproc=1, dump_npz=False, ignore_input_file=False, outdir=None):
        self.q       = q
        self.nu      = q/(1+q)**2
        self.chi1    = chi1
        self.chi2    = chi2
        self.nj      = nj
        self.r0      = r0
        self.pph_min = pph_min
        self.pph_max = pph_max
        self.update_pph_min = update_pph_min
        self.input_file = input_file
        self.verbose = verbose
        if outdir is None:
            outdir = os.getcwd()
        os.makedirs(outdir, exist_ok=True)        
        self.outdir  = outdir

        if self.verbose:
            print('\n','TEOB module:', EOBRun_module.__file__,'\n',sep='')
            print('---------------------------------------')
            print('Input')
            print('---------------------------------------')
            object_namespace = vars(self)
            for attribute, value in object_namespace.items():
                if isinstance(value, (str, float, int)) or value is None:
                    print(f"{attribute:15s}: {value}") 
            print(' ')

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
              'r0'                 : self.r0,
              'r_hyp'              : self.r0,         #r_hyp. Default = 0.
              'ode_tmax'           : 1e+6,
              'domain'             : 0,                 #Set 1 for FD. Default = 0
              'ode_tstep_opt'      : 1,                 #fixing uniform or adaptive. Default = 1 
              'srate_interp'       : 300000.,           #srate at which to interpolate. Default = 4096.
              'use_geometric_units': "yes",             # Output quantities in geometric units. Default = 1
              'interp_uniform_grid': "yes",             #interpolate mode by mode on a uniform grid. Default = 0 (no interpolation)
              'use_mode_lm'        : [1],               #List of modes to use/output through EOBRunPy
              'output_lm'          : [1],               #List of modes to print on file
              'arg_out'            : "yes",             #Output hlm/hflm. Default = 0
              'output_dynamics'    : "no",              #output of the dynamics
              }
       
        if self.input_file is None:
            self.parspace_populate() # define self.points
            self.run_TEOB_parallel(nproc=nproc, dump_npz=dump_npz, ignore_input_file=ignore_input_file)
        else:
            if self.verbose:
                print(f'Loading npz file {input_file}...\n')
            data = np.load(input_file)
            self.points  = data['points']
            if not np.allclose(data['meta'], [self.q, self.chi1, self.chi2, self.r0, self.nj], atol=1e-12):
                print('+++ WARNING +++\ninput file not consistent with (q,chi1,chi2,r0,nj) specified in input\nOverwriting input values:')
                self.q       = data['meta'][0]
                self.chi1    = data['meta'][1]
                self.chi2    = data['meta'][2]
                self.r0      = data['meta'][3]
                self.nj      = data['meta'][4]
                self.pph_min = min(self.points[:,0])
                self.pph_max = max(self.points[:,0])
                object_namespace = vars(self)
                for attribute, value in object_namespace.items():
                    if isinstance(value, (str, float, int)) or value is None:
                        print(f"{attribute:15s}: {value}") 
                print(' ')

            self.pph_min = min(self.points[:,0])
            self.pph_max = max(self.points[:,0])
            npoints = len(self.points)
            self.nj = int(0.5*(np.sqrt(1+8*npoints)-1))
        if args.verbose:
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
        template = 'q{:.@q_prec@f}_chi1@SIGN1@{:.@chi1_prec@f}_chi2@SIGN2@{:.@chi2_prec@f}_r0{:.@r0_prec@f}_nj{:d}'
        info_str = template.replace('@q_prec@', str(q_prec))
        info_str = info_str.replace('@chi1_prec@', str(chi1_prec))
        info_str = info_str.replace('@chi2_prec@', str(chi2_prec))
        info_str = info_str.replace('@r0_prec@', str(r0_prec))
        def return_sign_str(chi):
            if chi>=0:
                sign = 'p'
            else:
                sign = 'm'
            return sign
        info_str = info_str.replace('@SIGN1@', return_sign_str(self.chi1))
        info_str = info_str.replace('@SIGN2@', return_sign_str(self.chi2))
        return info_str.format(self.q, abs(self.chi1), abs(self.chi2), self.r0, self.nj)
    
    def parspace_populate(self):
        nj = self.nj
        if self.update_pph_min:
            # search pph such that Vmax=1
            pph_sample_tmp = np.linspace(3.0,5.0,num=2000)
            for j in pph_sample_tmp:
                E_max = EnergyLimitsSpin(10., self.q, j, self.chi1, self.chi2, N=1000)
                if E_max>=1:
                    self.pph_min = j
                    break
        pph_sample = np.linspace(self.pph_min, self.pph_max, num=nj)
        E_max_pph  = np.empty(np.shape(pph_sample))
        for i,pph in enumerate(pph_sample):
            E_max_pph[i] = max(1,EnergyLimitsSpin(10., self.q, pph, self.chi1, self.chi2, N=1000))
        npoints = int(nj*(nj+1)/2)
        points  = np.empty((npoints,4)) # 0:pph, 1:E, 2:Omg_orb_peaks, 3:r(end)
        k = 0
        for i in range(nj):
            for j in range(nj):
                if j<=i:
                    points[k,0] = pph_sample[i]
                    points[k,1] = E_max_pph[j]
                    k += 1
        self.points = points
        return

    def run_TEOB_parallel(self,nproc, dump_npz=False, ignore_input_file=False):
        if self.input_file is not None and not ignore_input_file:
            if self.verbose:
                print("Provided input file and ignore_input_file is False, skipping TEOB-runs")
            return
         
        t0 = time.perf_counter() 
        if self.verbose:
            print(f'>> Running TEOB using {nproc} core(s)')
        
        npoints = len(self.points)
        nbatch  = int(np.ceil(npoints/nproc))
        batches = []
        tmp_list = []
        for i in range(npoints):
             tmp_list.append(i)
             if len(tmp_list)==nbatch or i==npoints-1:
                  batches.append(tmp_list)
                  tmp_list = []
        tasks   = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for i in range(nproc):
                #print('proc: ', i+1)
                task = executor.submit( run_TEOB_list, batches[i], self.points, self.eobcommonpars)
                tasks.append(task)
            concurrent.futures.wait(tasks, return_when=concurrent.futures.ALL_COMPLETED)
            results = [future.result() for future in tasks]
        for i,batch in enumerate(batches):
            result = results[i]
            for j in batch:
                self.points[j,:] = result[j,:]
        if self.verbose:
            print('>> elapsed time {:.2f} s'.format(time.perf_counter()-t0))
        if dump_npz:
            fname = "data_"+self.info_string()+".npz"
            meta  = [self.q, self.chi1, self.chi2, self.r0, self.nj]
            np.savez(os.path.join(self.outdir,fname), points=self.points, meta=meta)
            if self.verbose:
                print(f'>> created {fname} in {self.outdir}')
        if self.verbose:
            print(' ')
        return 
    
    def qchi1chi2_list_str(self):
        str0 = ''
        vals = [self.q, self.chi1, self.chi2]
        for i,v in enumerate(vals):
            if abs(v-int(v))<1e-14:
                isint = True
                v_str = '{:.0f}'.format(abs(v))
            else:
                isint = False
                v_str = '{:.2f}'.format(abs(v))
            if i>0 and not isint:
                if v>=0:
                    v_str = '+'+v_str
                else:
                    v_str = '-'+v_str
            str0 += v_str 
            if i<len(vals)-1:
                str0 +=','
        return str0
    
    def get_NR_points(self, dset='GAUSS_2023', Emin=1.0, Emax=None, Jmin=None, Jmax=None):
        q  = self.q
        nu = self.nu
        if Emax is None: Emax = max(self.points[:,1])
        if Jmax is None: Jmax = max(self.points[:,0])*nu
        if Jmin is None: Jmin = min(self.points[:,0])*nu
         
        data_NR = []
        NR_path = os.path.join(repo_path, 'data', dset.lower())
        sims    = simulations.Sim(dset=dset)
        NR_list = sims.subset(pars={'q':q, 'chi1z':self.chi1, 'chi2z':self.chi2}, float_tol=5e-3)
        sims_to_skip = ['bbh_q-3.0_E0-1.017_j0-4.350_old-TP_N-96',    'bbh_q-3.0_E0-1.017_j0-4.350_old-TP_N-128', \
                        'bbh_q-3.0_E0-1.017_j0-4.350_old-TP_N-192',   'bbh_q-3.0_E0-1.017_j0-4.350_old-TP_N-256', \
                        'bbh_q-3.0_E0-1.017_j0-4.400_gauge-TP_N-128', 'bbh_q-3.0_E0-1.017_j0-4.400_gauge-TP_N-192',\
                        'bbh_q-3.0_E0-1.017_j0-4.400_gauge-TP_N-256', 'bbh_q-1.0_E0-1.011_j0-3.950_testM2_N-96']
        for name in NR_list:
           sim  = sims.getsim(name)
           M    = sim['M1_ADM']+sim['M2_ADM']
           E0   = sim['M_ADM']/M
           J0   = sim['J_ADM']/M**2
           fname_p0 = os.path.join(NR_path, name, 'collated_smalldata/puncture_0.txt')
           if not os.path.exists(fname_p0) or name in sims_to_skip:
              continue
           punct0 = np.loadtxt(fname_p0)
           punct1 = np.loadtxt(fname_p0.replace('_0.txt', '_1.txt'))
           psi4_l2m2 = np.loadtxt(os.path.join(NR_path, name, 'collated_smalldata/psi4_l2m2_r100.00.txt'))
           r_end      = np.sqrt( (punct0[-1,2]-punct1[-1,2])**2 + (punct0[-1,3]-punct1[-1,3])**2)/M
           psi4_amp   = np.abs( psi4_l2m2[:,1] + 1j*psi4_l2m2[:,2] )*M
           time_distance = 10 
           t  = psi4_l2m2[:,0]/M
           dT = t[1] - t[0]
           psi4_peaks,_ = find_peaks(psi4_amp, height=0.0002, distance=round(time_distance/dT))
           psi4_npeaks  = len(psi4_peaks)
           if J0>=Jmin and J0<=Jmax and E0<=Emax and E0>=Emin:
              #print([E0,J0,r_end,psi4_npeaks])
              data_NR.append([E0,J0,r_end,psi4_npeaks])
        
        data_NR = np.array(data_NR)
        E0 = data_NR[:,0]
        J0 = data_NR[:,1]/nu
        mask_scat       = data_NR[:,2] > 3 # final separation bigger than three --> scat
        mask_multi      = data_NR[:,3] > 1 # multiple encounters (peaks of psi4)
        mask_mrg_multi  = np.logical_and(~mask_scat,  mask_multi)
        mask_mrg_single = np.logical_and(~mask_scat, ~mask_multi)
        
        NR_points = {}
        NR_points['E0_scat']       = data_NR[ mask_scat, 0]
        NR_points['J0_scat']       = data_NR[ mask_scat, 1]/nu
        NR_points['E0_mrg_single'] = data_NR[mask_mrg_single, 0]
        NR_points['J0_mrg_single'] = data_NR[mask_mrg_single, 1]/nu
        NR_points['E0_mrg_multi']  = data_NR[mask_mrg_multi,  0]
        NR_points['J0_mrg_multi']  = data_NR[mask_mrg_multi,  1]/nu
        return NR_points

    def plot_parspace(self, marker_size=1, savepng=False,discrete_cmap=False,show='on', pph_max_plot=None, show_NR=False, dset='GAUSS_2023'):
        points2plot = self.points
        if pph_max_plot is not None:
            mask = points2plot[:,0]<=pph_max_plot
            points2plot = points2plot[mask]
        N   = points2plot[:,2] 
        cmp = build_colormap('jet', [1,0,1,1], N, discrete_cmap=discrete_cmap)
        plt.figure
        plt.scatter(points2plot[:,0], points2plot[:,1], c=points2plot[:,2], cmap=cmp, s=marker_size)
        plt.grid(which='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
        plt.xlabel(r'$p_\varphi^0$', fontsize=15)
        plt.ylabel(r'$E_0/M$', fontsize=15)
        cb = plt.colorbar()
        cb.locator = matplotlib.ticker.MaxNLocator(integer=True)
        cb.update_ticks()
        cb.set_label(r'$N$ (number of $\Omega$ peaks)', fontsize=15)
        Emax = max(points2plot[:,1])
        Emin = min(points2plot[:,1])
        pph_max = max(points2plot[:,0])
        pph_min = min(points2plot[:,0])
        def write_text(xrel,yrel,text,size=17, clr='k'):
            xplot = pph_min+xrel*(pph_max-pph_min)
            yplot = Emin+yrel*(Emax-Emin)
            plt.text(xplot, yplot, text, fontsize=size, color=clr)
        write_text(0.01, 0.97, r"$(q,\chi_1,\chi_2)=("+self.qchi1chi2_list_str()+")$")
        write_text(0.10, 0.65, r"$E_0>V_0^{\rm max}$")
        write_text(0.70, 0.37, r"scattering", clr=[1,1,1,1])

        if show_NR:
            NR_points = self.get_NR_points(dset=dset, Emin=Emin, Emax=Emax, Jmin=pph_min*self.nu, Jmax=pph_max*self.nu)
            plt.scatter(NR_points['J0_mrg_single'],  NR_points['E0_mrg_single'],  color=[ 0 , 1,  0 ], s=5, marker='o', label='direct capture')
            plt.scatter(NR_points['J0_mrg_multi'],   NR_points['E0_mrg_multi'],   color=[ 1 ,0.8, 0 ], s=5, marker='o', label='multi-encounter')
            plt.scatter(NR_points['J0_scat'],        NR_points['E0_scat'],        color=[0.0,0.7,0.9], s=5, marker='o', label='scattering')
            plt.legend(loc='upper left', bbox_to_anchor=(0.03, 0.9))
        
        if savepng:
            figname = 'plot_parspace_'+self.info_string()+'.png'
            #figname = figname.replace('.','d')+'.png'
            plt.savefig(os.path.join(self.outdir,figname), dpi=300)
            if self.verbose:
                print(f'>> {figname} saved in {self.outdir}')
        if show=='on':
            plt.show()
        return
    

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-n','--nproc',       default=1,    type=int,   help="number of processes")
    parser.add_argument('--jmin',             default=3.5,  type=float, help="min value of pph considered (eventually updated)")
    parser.add_argument('--jmax',             default=5.0,  type=float, help="max value of pph considered")
    parser.add_argument('--jmax_plot',        default=None, type=float, help="max value of pph considered in the plot (and only there)")
    parser.add_argument('-q', '--mass_ratio', default=1.0,  type=float, help="mass ratio")
    parser.add_argument('--chi1',             default=0.0,  type=float, help="primary spin")
    parser.add_argument('--chi2',             default=0.0,  type=float, help="secondary spin")
    parser.add_argument('--nj',               default=20,   type=int,   help="number of angular momenta considered") 
    parser.add_argument('--marker_size',      default=1,    type=int,   help="marker size in parspace-plot")
    parser.add_argument('-v', '--verbose',    action="store_true",      help="verbose option")
    parser.add_argument('--savepng',          action="store_true",      help="save parspace-plot as png")
    parser.add_argument('--discrete_cmap',    action="store_true",      help="use discrete color-map in parspace plot")
    parser.add_argument('--dump_npz',         action="store_true",      help="dump the points used in npz file")
    parser.add_argument('--show_NR',          action="store_true",      help="Show NR points on parspace plot (bool)")
    parser.add_argument('--dset', choices=['GAUSS_2021','GAUSS_2023'],\
                                  default='GAUSS_2023', type=str,       help="dset to use for show_NR")
    parser.add_argument('-i', '--input_file', default=None, type=str,   help="file with data points")
    parser.add_argument('-o', '--outdir',     default=None, type=str,   help="outdir for data and plots")
    parser.add_argument('--show', choices=['on', 'off'], default='on', type=str, help="Show parspace-plot")    
    
    args = parser.parse_args()
    
    if args.mass_ratio<1:
        args.mass_ratio = 1/args.mass_ratio

    nproc   = args.nproc
    pph_min = args.jmin
    pph_max = args.jmax
    q       = args.mass_ratio
    chi1    = args.chi1
    chi2    = args.chi2
    nj      = args.nj
    
    spanner = Spanner(q=q, chi1=chi1, chi2=chi2, pph_min=pph_min, pph_max=pph_max, nj=nj, update_pph_min=True, 
                      nproc=nproc, dump_npz=args.dump_npz, ignore_input_file=False,
                      input_file=args.input_file, verbose=args.verbose, outdir=args.outdir)
    spanner.plot_parspace(marker_size=args.marker_size,savepng=args.savepng,discrete_cmap=args.discrete_cmap,show=args.show, 
                          show_NR=args.show_NR, dset=args.dset, pph_max_plot=args.jmax_plot)

