import argparse,time,sys,os,copy
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from PyART.utils.os_utils import runcmd

file_path = os.path.abspath(__file__)
file_dir  = os.path.dirname(file_path)
DUMMY     = os.path.join(file_dir,'twopuncts.dummy')

###########################
# Class to normalize data
###########################
class LinearScaler:
    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B 
        self.C = C
        self.D = D
    def transform(self,x):
        return self.__lin_transf(self.A,self.B,self.C,self.D,x)
    def inverse_transform(self,y):
        return self.__lin_transf(self.C,self.D,self.A,self.B,y)
    def __lin_transf(self,A,B,C,D,x):
        return (D-C)*(x-A)/(B-A)+C 

###########################
# Class to compute NR ICs
###########################
class TwoPunctID(object):
    def __init__(self, **kwargs):
        
        self.M           = 1.0
        self.q           = 1.0
        self.E           = 1.1
        self.E_min       = None
        self.L           = 1.0
        self.D           = 100
        self.chi1z       = 0.0
        self.chi2z       = 0.0
        self.npoints_A   = 8
        self.npoints_B   = 8
        self.npoints_phi = 6
        self.TP_dummy    = DUMMY
        self.TP_exe      = './TwoPunctures.x' 
        self.outdir      = './'
        self.verbose     = False
        self.iteration   = None

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f'Unknown option: {key}')
        
        if self.verbose:
            for name, value in self.__dict__.items():
                if not callable(value):
                    print(f"{name:12s} : {value}")
            print(' ')

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
            if self.verbose:
                print(f'Created outdir: {self.outdir}')

        M     = self.M
        q     = self.q
        chi1z = self.chi1z 
        chi2z = self.chi2z
        D     = self.D

        # create dictionary for TP
        if q<1 : q=1/q
        mp    = M*q/(1+q) # m plus
        mm    = M  /(1+q) # m minus 
        par_b = D/2
        give_bare_mass = 0

        Spx   = 0.0
        Spy   = 0.0
        Spz   = chi1z*mp**2 
        Smx   = 0.0
        Smy   = 0.0
        Smz   = chi2z*mm**2 

        offsetx = -D/2*(q-1)/(q+1)
        offsety = 0.0
        offsetz = 0.0
        
        self.TP_pars = {'par_b':par_b, 'par_m_plus':mp, 'par_m_minus':mm, 'target_M_plus':mp, 'target_M_minus':mm,
                        'par_P_plus1'  : None, 'par_P_plus2'  : None, 'par_P_plus3'  : 0.0,
                        'par_P_minus1' : None, 'par_P_minus2' : None, 'par_P_minus3' : 0.0,
                        'par_S_plus1'  : Spx,  'par_S_plus2'  : Spy,  'par_S_plus3'  : Spz,
                        'par_S_minus1' : Smx,  'par_S_minus2' : Smy,  'par_S_minus3' : Smz,
                        'center_offset1': offsetx, 'center_offset2': offsety, 'center_offset3': offsetz,
                        'npoints_A':self.npoints_A, 'npoints_B':self.npoints_B, 'npoints_phi':self.npoints_phi,
                        'give_bare_mass':give_bare_mass}
        self.TP_int_vars = ['npoints_A', 'npoints_B', 'npoints_phi', 'give_bare_mass']
        return
    
    def create_TP_parfile(self, P, parfile=None, outdir=None):
        if outdir  is None: outdir = self.outdir
        if parfile is None: parfile = f'parfile_P{P:.10f}.par'
        Py = self.L/self.D
        Py2 = Py**2
        P2  = P**2
        if P2<=Py2:
            P  = 0.0
            Py = 0.0
            Px = 0.0
        else:
            Px = np.sqrt(P2-Py2)
        TP_pars_P = copy.deepcopy(self.TP_pars)
        TP_pars_P['par_P_plus1']  = -Px
        TP_pars_P['par_P_plus2']  =  Py
        TP_pars_P['par_P_minus1'] =  Px
        TP_pars_P['par_P_minus2'] = -Py
        parfile_lines = []
        with open(self.TP_dummy, 'r') as file:
            for line in file:
                if '@' in line:
                    idx1 = line.find('@')
                    idx2 = line.rfind('@')
                    var_name = line[idx1+1:idx2]
                    var = TP_pars_P[var_name]
                    if var_name in self.TP_int_vars:
                        str2write = f'{var:d}'
                    else:
                        str2write = f'{var:.15f}'
                    line = line.replace(line[idx1:idx2+1],str2write)
                parfile_lines.append(line)
        with open(os.path.join(outdir,parfile), 'w') as file:
            for line in parfile_lines:
                file.write(line)
        return parfile
        
    def run_TP_from_parfile(self, parfile):
        outfile  = parfile.replace('.par','.out')
        TP_cmd = self.TP_exe+' '+parfile+' > '+outfile+' 2>&1'
        runcmd(TP_cmd, workdir=self.outdir)         
        return outfile
    
    def read_TP_output(self, outfile):
        if os.path.exists(outfile):
            raise RuntimeError(f'file not found: {outfile}')
        with open(os.path.join(self.outdir,outfile), 'r') as file:
            for line in file:        
                if 'Puncture 1 ADM mass is' in line:
                    M1 = float(line.split(' ')[-1])
                elif 'Puncture 2 ADM mass is' in line:
                    M2 = float(line.split(' ')[-1])
                elif 'The total ADM mass is' in line:
                    E_ADM = float(line.split(' ')[-1])
                    break
        return E_ADM, M1, M2
    
    def run_TP_wrapper(self, P, parfile=None, verbose=None):
        if verbose is None: verbose = self.verbose
        if parfile is None: parfile = f'parfile_P{P:.10f}.par'
        self.create_TP_parfile(P, parfile=parfile)
        outfile = self.run_TP_from_parfile(parfile)
        E_ADM, M1, M2 = self.read_TP_output(outfile)
        if verbose:
            print(f'{outfile:10s} : P:{P:.10f}  E_ADM:{E_ADM:.10f}  M1:{M1:.5f}  M2:{M2:.5f}')
        return E_ADM, M1, M2
    
    def run_TP_parallel(self, momenta, batch):
        energies = np.zeros_like(momenta) 
        for idx in batch:
            E_ADM, _, _ = self.run_TP_wrapper(momenta[idx])
            if E_ADM<=self.E_min:
                energies[idx] = np.nan
            else:
                energies[idx] = E_ADM
        return energies

    def fit_iter(self, P0, dP, npoints=4, poly_order=None, verbose=None, show_plot=False, save_plot=True, x0_position='centered', nproc=1):
        if verbose is None: verbose = self.verbose
        P    = P0
        Pmin = 0
        if self.E_min is None:
            if self.E<1:
                if verbose:
                    print('Running TP to find E_min')
                self.E_min,_,_ = self.run_TP_wrapper(Pmin)
            else:
                self.E_min = 1-1e-8
        E_min = self.E_min
        if self.E<E_min:
            raise RuntimeError(f'Invalid energy: {self.E}')
        
        if x0_position=='centered':
            half1   = np.floor((npoints-1)/2)
            half2   = np.ceil((npoints-1)/2)
            momenta = np.linspace(P0-half1*dP, P0+half2*dP, num=npoints)
        elif x0_position=='highest':
            momenta = np.linspace(P0-(npoints-1)*dP, P0, num=npoints)
        elif x0_position=='lowest':
            momenta = np.linspace(P0, P0+(npoints-1)*dP, num=npoints)
        else:
            raise RuntimeError(f'Unknown position option: {x0_position}')
        
        mask_neg = momenta<0
        if np.sum(mask_neg)>0 and verbose:
            momenta[mask_neg] = 0.0
            npoints = len(momenta)
            print(f'Warning: setting negative momenta to zero')
         
        energies = np.zeros_like(momenta) 
        nbatch   = int(np.ceil(npoints/nproc))
        
        batches = []
        start = 0
        while start < npoints:
            batches.append(list(range(start, min(start + nbatch, npoints + 1))))
            start += nbatch
        tasks = []
        nproc_updated = len(batches)
        if nproc_updated<nproc and verbose:
            print(f'Warning: using less processor ({nproc_updated:d}) than requested ({nproc:d})')
        nproc = nproc_updated
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for i in range(nproc):
                task = executor.submit(self.run_TP_parallel, momenta, batches[i])
                tasks.append(task)
            concurrent.futures.wait(tasks, return_when=concurrent.futures.ALL_COMPLETED)
            results = [future.result() for future in tasks]
        for i,r in enumerate(results):
            for j in batches[i]:
                energies[j] = results[i][j]
                
        # remove NaN and duplicates, then sort data
        mask_nan = np.isnan(energies)
        momenta  = np.unique( momenta[~mask_nan])
        energies = np.unique(energies[~mask_nan])
        npoints_updated = len(energies)
        
        if npoints_updated<1:
            raise RuntimeError('Search failed. Tried to change hyperparameters')

        mask     = np.argsort(momenta)
        momenta  = momenta[mask]
        energies = energies[mask] 
        
        # rescale and fit 
        if npoints_updated<npoints:
            npoints = npoints_updated
        if poly_order is None: poly_order = min(max(npoints-1,1),5)
        ncoeffs = poly_order + 1
        if ncoeffs>npoints:
            raise ValueError(f'Cannot perform {poly_order:d}-order polynomial fit with {npoints:d} points')

        if verbose:
            print(f'Performing fit using {npoints:d} points with {poly_order:d}-order polynomial') 
        energy_scaler   = LinearScaler(min(energies), max(energies), 0, 1)
        momentum_scaler = LinearScaler(min(momenta),  max(momenta),  0, 1)
        x = energy_scaler.transform(energies)
        y = momentum_scaler.transform(momenta)
        b = np.polyfit(x, y, poly_order)
        dE_approx     = energies[1] - energies[0]
        energies_fine = np.linspace(energies[0]-dE_approx, energies[-1]+dE_approx, num=1000)
        x_fine = energy_scaler.transform(energies_fine)
        p = np.polyval(b, x_fine) 
        fitted_momenta = momentum_scaler.inverse_transform(p)
        
        E_rescaled  = energy_scaler.transform(self.E)
        Pf_rescaled = np.polyval(b, E_rescaled) 
        Pf          = momentum_scaler.inverse_transform(Pf_rescaled)

        xlabs = [r'$x_{\rm fit}$', r'$E_{\rm ADM}$']
        ylabs = [r'$y_{\rm fit}$', r'$P_{\rm ADM}$']
        fig, axs = plt.subplots(2,1,figsize=(10,7))
        axs[0].scatter(x,y, zorder=2, c='k')
        axs[0].plot(x_fine, p, zorder=1, color=[1,0,0])
        axs[0].axhline(Pf_rescaled, zorder=3)
        axs[1].scatter(energies, momenta, zorder=2)
        axs[1].plot(energies_fine, fitted_momenta, zorder=1, color=[1,0.5,0])
        axs[1].axhline(Pf, zorder=3)
        for i in range(2):
            axs[i].grid()
            axs[i].set_xlabel(xlabs[i], fontsize=15)
            axs[i].set_ylabel(ylabs[i], fontsize=15)
        if save_plot:
            figname = f'tp_E{self.E:.3f}_L{self.L:.3f}_iter{self.iteration:d}.png'
            figname = os.path.join(self.outdir, figname)
            plt.savefig(figname, dpi=200, bbox_inches='tight')
        if show_plot: 
            plt.show()
        else:
            plt.close()
        return Pf
    
    def fit_iterations(self, P0=None, step_rel=0.15, tol=1e-10, resize_factor=100, itermax=5, npoints=4, \
                             poly_order=None, verbose=None, nproc=1, save_plot=True, show_plot=False):
        if verbose is None: verbose = self.verbose
        if P0 is None and self.E > 1-tol:
            nu = self.q/(1+self.q)**2
            P0 = 0.5*np.sqrt( (self.E**2-1)/(4*nu) )
            x0_position_first = 'centered' #'highest'
            x0_position_other = 'centered'
        elif P0 is None and self.E < 1-tol:
            P0 = 0.03
            x0_position_first = 'lowest'
            x0_position_other = 'lowest'
        else:
            x0_position_first = 'centered'
            x0_position_other = 'centered'

        x0_position = x0_position_first
        dP    = step_rel*P0
        P     = P0
        error = 1
        for i in range(itermax):
            self.iteration = i
            if verbose:
                print(f'# iter : {i:d}')
                print(f'P0     : {P:.10f}\ndP     : {dP:.10f}')
            t0 = time.perf_counter()
            P  = self.fit_iter(P, dP, npoints=npoints, poly_order=poly_order, x0_position=x0_position, nproc=nproc,
                             save_plot=save_plot, show_plot=show_plot)
            E_adm,_,_ = self.run_TP_wrapper(P)
            error = abs(E_adm-self.E)/self.E
            if verbose:
                print(f'iteration results')
                print(f'P            : {P:.15f}')
                print(f'E            : {E_adm:.15f}')
                print(f'error        : {error:.5e}')
                print( 'Elapsed time : {:.2f} s\n'.format(time.perf_counter()-t0))
            if error<tol:
                break
            dP *= 1/resize_factor 
            x0_position = x0_position_other
        Py = self.L/self.D
        Px = -np.sqrt(P**2-Py**2)
        return Px, Py, E_adm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe',           required=True,             help='Exectuable (TwoPunctures.x)')
    parser.add_argument('-q',              default=1,     type=float, help='mass ratio')
    parser.add_argument('--chi1z',         default=0.0,   type=float, help='chi1z')
    parser.add_argument('--chi2z',         default=0.0,   type=float, help='chi2z')
    parser.add_argument('-D', '--distance',default=100,   type=float, help='Initial separation') 
    parser.add_argument('-E', '--energy',  default=1.01,  type=float, help='Initial energy') 
    parser.add_argument('-L', '--ang_momentum', default=1,type=float, help='Initial orbital angular momentum') 
    parser.add_argument('-o', '--outdir',  default='./out',           help='Outdir')
    parser.add_argument('-n', '--nproc',   default=1,     type=int,   help='number of process used')
    parser.add_argument('--res',default=[8,8,6],nargs='+',type=int,   help='Resolution to use: [npoints_A,npoints_B,npoints_phi]')                  
    # tunable parameter for the search
    parser.add_argument('--init_guess',    default=None,  type=float, help='Initial guess for the momentum modulo')
    parser.add_argument('--npoints',       default=4,     type=int,   help='Points to consider when searching initial guess')
    parser.add_argument('--step',          default=0.05,  type=float, help='Initial relative momentum step')
    parser.add_argument('--resize',        default=100,   type=float, help='Resize factor used in the search')
    parser.add_argument('--tol',           default=1e-10, type=float, help='Tolerance on the energy')
    args = parser.parse_args()

    # input
    q     = args.q
    chi1z = args.chi1z
    chi2z = args.chi2z
    D     = args.distance
    E     = args.energy
    L     = args.ang_momentum
    res   = args.res

    tpi = TwoPunctID(q=q, D=D, E=E, L=L, chi1z=chi1z, chi2z=chi2z, 
                      npoints_A=res[0], npoints_B=res[1], npoints_phi=res[2],
                      TP_exe=args.exe, outdir=args.outdir, verbose=True)
    
    Px, Py, E_adm = tpi.fit_iterations(P0=args.init_guess, step_rel=args.step, tol=args.tol, \
                                      npoints=args.npoints, resize_factor=args.resize, nproc=args.nproc)
    P = np.sqrt(Px**2 + Py**2)
    
    print(f'Final result : ')
    print(f'E target     : {E:.15e}')
    print(f'E found      : {E_adm:.15e}')
    print(f'P            : {P:.15e}')
    print(f'Px           : {Px:.15e}')
    print(f'Py           : {Py:.15e}')

