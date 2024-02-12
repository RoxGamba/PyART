import sys, os, argparse, subprocess, matplotlib
import matplotlib.pyplot as plt 
import numpy as np
from scipy.fftpack import fft, fftfreq, ifft

repo_path = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], \
                             stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
sys.path.insert(0, os.path.join(repo_path, 'py/sims'))
import simulations, utils, scattering_angle as scat

try:
    from processwave import processwave
except Exception as e:
    print("An error occurred:", str(e))
    print('Solution: create a softlink simulations/scripts/processwave.py in /py/analysis/\n')
    raise e

matplotlib.rc('text', usetex=True)

class ProcessData(object):
    """
    Class to process NR data and produce plots
    """
    def __init__(self, **kwargs):
        self.dset         = 'GAUSS_2023'
        self.sims         = []
        self.tlim         = []
        self.verbose      = False
        self.savepng      = False
        self.showpng      = True
        self.plots_labels = []
        self.integral     = 'FFI'
        self.FFI_f0       = 0.01 
        self.TDI_degree   = 1
        self.TDI_poly_int = None
        self.lmmax        = 5 
        self.extrap_psi4  = False
        self.out_data     = False
        self.rm_last_Dt   = [] # cut last Delta_t of the GW signal
        self.dpi          = 100

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f'Unknown option: {key}')
        
        self.nsims = len(self.sims)
        
        self.colors = self.auto_colors(self.nsims)

        if self.dset=='GAUSS_2021':
            self.datapath = os.path.join(repo_path,'data/gauss_2021')
        elif self.dset=='GAUSS_2023':
            self.datapath = os.path.join(repo_path,'data/gauss_2023')
        else:
            raise RuntimeError('Unknown dset: {}'.format(dset))
        self.jsonfile = os.path.join(repo_path,'runs/simulations.json')
        
        self.Sim = simulations.Sim(dset=self.dset, datapath=self.datapath, jsonfile=self.jsonfile)
        if len(self.sims)<1:
            print('\nList of simulations not given in input! Exit...')
            sys.exit()
            return 
        
        if isinstance(self.FFI_f0, float):
            self.FFI_f0 = np.ones((self.nsims,))*self.FFI_f0
        elif len(self.FFI_f0)==1:
            self.FFI_f0 = np.ones((self.nsims,))*self.FFI_f0[0]
        elif len(self.FFI_f0)!=self.nsims:
            raise ValueError('FFI_f0 can be a float or a list of float with size of nsims')

        if len(self.plots_labels)==0:
            for i,sim in enumerate(self.sims):
                meta = self.Sim.getsim(sim,key="name") 
                q  = meta['q']
                M  = meta['M1_ADM']+meta['M2_ADM']
                E0 = meta['M_ADM']/M
                J0 = meta['J_ADM']/M**2/meta['nu']
                N  = meta['nx1']
                chi1z = meta['chi1z']
                chi2z = meta['chi2z']
                self.plots_labels.append('{:.2f}-{:.1f}-{:.1f}-{:.3f}-{:.3f}-{:d}'.format(q,chi1z,chi2z,E0,J0,N))
        else:
            for i in range(len(self.plots_labels)):
                self.plots_labels[i] = r'' + self.plots_labels[i]

        if self.verbose:
            for attribute in dir(self):
                if not attribute.startswith('__') and not callable(getattr(self, attribute)):
                    value = getattr(self,attribute)
                    print(f'{attribute:20s} : {value}')
            print(' ')

        self.load_process_data()
        return
    
    def auto_colors(self, nsims):
        if nsims<=4:
            colors4 = np.array([[0.2,0.5,1], [0,0.9,0.2], [1,0.7,0.2], [0.9,0,0]])
            colors  = colors4[:nsims]
        else:
            colors = plt.cm.rainbow(np.linspace(0,1,nsims))
            colors = colors[:,:-1]
        return colors 

    def load_process_data(self):
        dashes = '-'*60
        self.data = {}
        sims = self.sims
        for n in range(self.nsims):
            sim = sims[n]
            sim_datapath = os.path.join(self.datapath,sim,'collated_smalldata/')
            if not os.path.exists(sim_datapath):
                raise RuntimeError(f'Data path not found:\n{sim_datapath}\nTry to run rsync data or change dset.\nSkipping this run')

            if not os.path.exists(os.path.join(sim_datapath,'wave_r100.00.txt')):
                raise RuntimeError(f'Data not found in {sim_datapath}.\nSkipping this run')

            metadata   = self.Sim.getsim(sim)
            M          = metadata['M1_ADM'] + metadata['M2_ADM']
            q          = metadata['q']
            nu         = metadata['nu']
            hatM_ADM   = metadata['M_ADM']/M
            hatJ_ADM   = metadata['J_ADM']/(M**2)
            punct_0    = np.loadtxt(os.path.join(sim_datapath,'puncture_0.txt')) # iter,t,x,y,z,(+bx,by,bz if GAUSS_2023)
            punct_1    = np.loadtxt(os.path.join(sim_datapath,'puncture_1.txt')) # iter,t,x,y,z,(+bx,by,bz if GAUSS_2023)
            rp         = np.sqrt( (punct_0[:,2]-punct_1[:,2])**2 + (punct_0[:,3]-punct_1[:,3])**2 )/M
            
            if self.verbose:
                print(f'name         : {sim}')
                print( 'q, ch1, chi2 : {:.5f}, {:.2f}, {:.2f}'.format(q, metadata['chi1z'], metadata['chi2z'])) 
                print(f'M            : {M:.5f}') 
                print(f'E0/M         : {hatM_ADM:.15f}') 
                print(f'J0/M2        : {hatJ_ADM:.15f}') 
                print( 'par_b        : {:.15f}'.format(metadata['par_b']))
                print( 'P+           : [{:18.15f},{:18.15f}]'.format(metadata['par_P_plus'][0],metadata['par_P_plus'][1]) )
                print( 'is_scat      : {}'.format(rp[-1]>3))
                print(' ') 

            if self.dset=='GAUSS_2021':
                def add_beta_shift(p):
                    n  = len(p[:,0])
                    X  = np.empty((n,8))
                    tp = p[:,1]
                    bx = utils.D1(p[:,2],tp,4)
                    by = utils.D1(p[:,3],tp,4)
                    bz = utils.D1(p[:,4],tp,4)
                    X  = np.vstack( (p[:,0], p[:,1], p[:,2], p[:,3], p[:,4], bx, by, bz) )
                    return np.transpose(X)
                punct_0 = add_beta_shift(punct_0)
                punct_1 = add_beta_shift(punct_1)

            radius = 100. 
            if abs(M-2)<1e-8:# this should happen only for bbh_q-1.0_E0-1.011_j0-3.950_testM2_N-96
                if self.verbose:
                    print('Rest-mass M=2, loading wave_r200.00.txt instead of wave_r100.00.txt')
                radius = 200
            fname  = os.path.join(sim_datapath,f'wave_r{radius:.2f}.txt')
            proc_outdir = 'data_'+sim+'_'+self.integral
            if self.extrap_psi4:
                proc_outdir += '_psi4extrap'
               
            dataproc = processwave(ifiles=fname, radii=radius, mass=M, lmmax=self.lmmax, extrap_psi4=self.extrap_psi4, 
                                   strain=self.integral, frequency=self.FFI_f0[n]/M, degree=self.TDI_degree, poly_int=self.TDI_poly_int,
                                   madm=hatM_ADM*M, jadm=hatJ_ADM*M*M, mratio=q, energetics=True, silent=True,
                                   no_output=(not self.out_data), outdir=proc_outdir)
            if len(self.rm_last_Dt)>0:
                t     = dataproc['t']/M
                mask  = t<=t[-1]-self.rm_last_Dt[n]
                keys_lm = ['u', 'psi4', 'dh', 'h', 'phi', 'omg', 'amp']
                keys    = ['t']
                keys_en = ['E', 'Edot', 'Jz', 'Jzdot'] 
                for lm in dataproc['lmmodes']:
                    for k in keys_lm:
                        dataproc[k][lm] = dataproc[k][lm][mask]
                for k in keys:
                    dataproc[k] = dataproc[k][mask]
                for k in keys_en:
                    dataproc['energetics'][k] = dataproc['energetics'][k][mask]

            if self.verbose and self.out_data:
                print(f'Dumped processed data in {proc_outdir}')
           
            if len(self.tlim)==0:
                self.tlim = [dataproc['t'][0]/M, dataproc['t'][-1]/M]
            
            if rp[-1]>3:
                scat_NR = scat.ScatteringAngle(punct0=punct_0, punct1=punct_1, file_format='GRA', nmin=2, nmax=5, n_extract=4,
                                           r_cutoff_out_low=25, r_cutoff_out_high=None, 
                                           r_cutoff_in_low=25, r_cutoff_in_high=rp[0]-5, 
                                           verbose=self.verbose)
                #scat_NR.plot_fit_extrapolation()
                chi = scat_NR.chi
                chi_fit_err = scat_NR.fit_err
            else:
                chi = None
                chi_fit_err = None
            
            if metadata['ahf']:
                # 1:iter 2:time 3:mass 4:Sx 5:Sy 6:Sz 7:S 8:area 9:hrms 10:hmean 11:meanradius
                horizon_s0 = np.loadtxt(os.path.join(sim_datapath,'horizon_summary_0.txt'))
                horizon_s1 = np.loadtxt(os.path.join(sim_datapath,'horizon_summary_1.txt'))
            else:
                horizon_s0 = []
                horizon_s1 = []
            
            data               = dataproc # t,u,psi4,dh,h,amp,phi,omg + energetics
            data['M']          = M 
            data['chi']        = chi
            data['chi_fit_err']= chi_fit_err
            data['punct_0']    = punct_0
            data['punct_1']    = punct_1
            data['horizon_s0'] = horizon_s0
            data['horizon_s1'] = horizon_s1
            data['metadata']   = metadata
            self.data[sim]     = data
        return

    def return_wave(self, sim_dict, wave_var, l, m, wave_func=None):
        w0   = sim_dict[wave_var][(l,m)]
        M    = sim_dict['M']
        u    = sim_dict['u'][(l,m)]
        t    = sim_dict['t']
        ubyM = u/M
        lm   = str(l)+str(m)
        h    = sim_dict['h'][(l,m)]/M
        
        if wave_var=='psi4':
            w1    = w0*M
            w1check = utils.D1(utils.D1(h,t,4),t,4)
            #ylab  = r'$R \psi_{4,'+lm+'} M$'
            ylab  = r'$\psi_{4,'+lm+'} M$'
        elif wave_var=='dh':
            w1    = w0
            w1check = utils.D1(h,t,4)
            ylab  = r'$\dot{h}_{'+lm+'}$'
        elif wave_var=='h':
            w1    = w0/M
            w1check = w1
            #ylab  = r'$R h_{'+lm+'}/M$'
            ylab  = r'$h_{'+lm+'}/M$'
        
        if wave_func is None:
            w    = w1
            wchk = w1check
        elif wave_func=='re':
            w    = w1.real
            wchk = w1check.real
            ylab = '$\Re '+ylab[1:-1]+'$'
        elif wave_func=='im':
            w    = w1.imag
            wchk = w1.imag
            ylab = '$\Im '+ylab[1:-1]+'$'
        elif wave_func=='amp':
            w    = abs(w1)
            wchk = abs(w1check)
            ylab = '$|'+ylab[1:-1]+'|$'
        elif wave_func=='phi':
            w    = -np.unwrap(np.arctan(w1.imag/w1.real)*2)/2
            wchk = w*0
            ylab = r'$\phi_{'+lm+'}$ - ' + ylab
        elif wave_func=='omg':
            phi  = -np.unwrap(np.arctan(w1.imag/w1.real)*2)/2
            w    = utils.D1(phi, t/M, 4)
            wchk = w*0
            ylab = r'$\omega_{'+lm+'}$ - ' + ylab
        return w, u, M, wchk, ylab
    
    ##################################
    # Plotting functions
    ##################################
    
    def plot_psi4_radii(self, figname='plot_psi4_radii.png', **kwargs):
        loc     = utils.local_vars_for_plots(self,**kwargs)
        sims    = loc.sims
        tlim    = loc.tlim 

        data = self.data
        psi4_dict = {}
        for sim in sims:
            M = data[sim]['M']
            sim_datapath = os.path.join(self.datapath,sim,'collated_smalldata/')
            files = os.listdir(sim_datapath)
            for f in files:
                if 'psi4_l2m2_r' in f:
                    tmp = f.replace('psi4_l2m2_', '')
                    key = tmp.replace('.txt', '')
                    R   = float(key.replace('r',''))
                    X = np.loadtxt(os.path.join(sim_datapath,f)) # t,re,im
                    s = {}
                    s['R']    = R
                    s['t']    = X[:,0]
                    s['psi4'] = R*(X[:,1]+1j*X[:,2])
                    psi4_dict[key] = s
            # plot for different extraction radius
            def numerical_value(rkey):
                return float(rkey[1:])
            sorted_keys = sorted(list(psi4_dict.keys()), key=numerical_value)
            colors_radii = plt.cm.rainbow(np.linspace(0, 1, len(sorted_keys)))
            
            plt.figure(dpi=loc.dpi)
            for i,key in enumerate(sorted_keys):
                s     = psi4_dict[key]
                tw    = (s['t']-s['R'])/M
                Rpsi4 = s['psi4']*M
                plt.plot(tw, np.abs(Rpsi4), label=key, color=colors_radii[i])
            plt.xlim(tlim)
            plt.xlabel(r'$(t-R)/M$',fontsize=20)
            plt.ylabel(r'$R \psi_{4,22} M$',fontsize=20)
            plt.legend()
            utils.save_plot(figname, save=loc.savepng, show=loc.showpng, verbose=loc.verbose)
        return

    def plot_summary(self, figname='plot_summary.png', **kwargs):
        loc     = utils.local_vars_for_plots(self,**kwargs)
        sims    = loc.sims
        tlim    = loc.tlim 
        colors  = loc.colors

        data = self.data
        
        figm = 2
        fign = 2
        figsize = (10,8)
        ylabels = [[r'$y/M$',r'$R\psi_{4,22} M$'],[r'$R\dot{h}_{22} $',r'$Rh_{22}/M$']]
        xlabels = [[r'$x/M$',r'$u/M$'],[r'$u/M$',r'$u/M$']]
        fig,axs = plt.subplots(figm,fign, figsize=figsize, dpi=loc.dpi)
        fig.subplots_adjust(hspace=.3,wspace=.3)
        
        for n,sim in enumerate(sims):
            M   = data[sim]['M']
            xp0 = data[sim]['punct_0'][:,2]/M
            yp0 = data[sim]['punct_0'][:,3]/M
            xp1 = data[sim]['punct_1'][:,2]/M
            yp1 = data[sim]['punct_1'][:,3]/M
            u    = data[sim]['u']
            psi4 = data[sim]['psi4']
            dh   = data[sim]['dh']
            h    = data[sim]['h']
            axs[0,0].plot(xp0, yp0, color=colors[n],     label=loc.plots_labels[n]+' +')
            axs[0,0].plot(xp1, yp1, color=colors[n]/1.4, label=loc.plots_labels[n]+' -')
            axs[0,1].plot(u[(2,2)]/M,  psi4[(2,2)].real*M, color=colors[n],  label=loc.plots_labels[n])
            axs[0,1].plot(u[(2,2)]/M,  np.abs(psi4[(2,2)])*M, color=colors[n] )
            axs[1,0].plot(u[(2,2)]/M,  dh[(2,2)].real,     color=colors[n] )
            axs[1,0].plot(u[(2,2)]/M,  np.abs(dh[(2,2)]),  color=colors[n] )
            axs[1,1].plot(u[(2,2)]/M,  h[(2,2)].real/M,    color=colors[n] )
            axs[1,1].plot(u[(2,2)]/M,  np.abs(h[(2,2)])/M, color=colors[n] )

        for i in range(0,figm):
            for j in range(0,fign):
                axs[i,j].set_xlabel(xlabels[i][j], fontsize=20)
                axs[i,j].set_ylabel(ylabels[i][j], fontsize=20)
                if i>0 or j>0:
                    axs[i,j].set_xlim(tlim)
        axs[0,0].set_title(sim)
        axs[0,0].legend()
        utils.save_plot(figname, save=loc.savepng, show=loc.showpng, verbose=loc.verbose)
        return
    
    def plot_ahf(self,figname='plot_ahf.png', **kwargs):
        loc     = utils.local_vars_for_plots(self,**kwargs)
        sims    = loc.sims
        tlim    = loc.tlim 
        colors  = loc.colors
        
        data = self.data
        fig,axs = plt.subplots(1,2,figsize=(11,5),dpi=loc.dpi)
        for n,sim in enumerate(sims):
            M   = data[sim]['M']
            if len(data[sim]['horizon_s0'])==0:
                continue
            th0 = data[sim]['horizon_s0'][:,1]/M
            Mh0 = data[sim]['horizon_s0'][:,2]/M
            Sh0 = data[sim]['horizon_s0'][:,5]/(M*M)
            th1 = data[sim]['horizon_s1'][:,1]/M
            Mh1 = data[sim]['horizon_s1'][:,2]/M
            Sh1 = data[sim]['horizon_s1'][:,5]/(M*M)
            #nx1_str = 'N'+str(data[sim]['metadata']['nx1'])
            axs[0].plot(th0,Mh0,color=colors[n],   label=loc.plots_labels[n]+' +',linestyle='-')
            axs[0].plot(th1,Mh1,color=1-colors[n], label=loc.plots_labels[n]+' -',linestyle='--')
            axs[1].plot(th0,Sh0,color=colors[n],   label=loc.plots_labels[n]+' +',linestyle='-')
            axs[1].plot(th1,Sh1,color=1-colors[n], label=loc.plots_labels[n]+' -',linestyle='--')
        fig.subplots_adjust(wspace=.4)
        axs[0].set_xlim(tlim)
        axs[0].set_xlabel(r'$t/M$', fontsize=20)
        axs[0].set_ylabel(r'$M_{\rm AHF}$', fontsize=20)
        axs[0].legend()
        axs[1].set_xlim(tlim)
        axs[1].set_xlabel(r'$t/M$', fontsize=20)
        axs[1].set_ylabel(r'$S_z^{\rm AHF}$', fontsize=20)
        axs[1].legend()
        utils.save_plot(figname, save=loc.savepng, show=loc.showpng, verbose=loc.verbose)
        return
    
    def plot_waveform(self, l=2, m=2, wave_vars=['psi4'], wave_func='re', integral_check=False, show_tmrg=False,\
                      figname='plot_@wave_func@_@wave_var@.png',  ylog=False, ylabel=None, **kwargs):
        loc     = utils.local_vars_for_plots(self,**kwargs)
        sims    = loc.sims
        tlim    = loc.tlim 
        colors  = loc.colors
        
        data = self.data
        show_legend = True
        for wave_var in wave_vars:
            fig,ax = plt.subplots(1,1,figsize=(8,6),dpi=loc.dpi)
            for n,sim in enumerate(sims):
                w, u, M, wcheck, default_ylab = self.return_wave(data[sim], wave_var, l, m, wave_func=wave_func)
                if len(loc.plots_labels[n])==0:
                    show_legend = False
                ax.plot(u/M, w, color=colors[n], ls='-', label=loc.plots_labels[n])
                if integral_check and wave_var in ['psi4', 'dh'] and wave_func in ['amp','re','im']:
                    check_color = colors[n]/1.5
                    ax.plot(u/M, wcheck, color=check_color, ls='--', label=loc.plots_labels[n]+' - check')
            if show_tmrg:
                amp_h22, _, _, _, _ = self.return_wave(data[sim], 'h', 2, 2, wave_func='amp')
                tmrg, _ = utils.find_Amax(u/M, amp_h22)
                ax.axvline(tmrg, ls='--')
            ax.set_xlim(tlim)
            ax.set_xlabel(r'$u/M$', fontsize=20)
            if ylabel is not None:
                ax.set_ylabel(ylabel, fontsize=20)
            else:
                ax.set_ylabel(default_ylab, fontsize=20)
            if ylog:
                ax.set_yscale('log')
            if wave_func=='omg':
                ax.set_ylim([0,1])
            if show_legend:
                ax.legend()
            figname_wf = figname.replace('@wave_func@', wave_func)
            figname_wf = figname_wf.replace('@wave_var@', wave_var)
            utils.save_plot(figname_wf, save=loc.savepng, show=loc.showpng, verbose=loc.verbose)
        return
    
    def plot_puncts(self,figname='plot_puncts.png', xlim=None, ylim=None, **kwargs):
        loc     = utils.local_vars_for_plots(self,**kwargs)
        sims    = loc.sims
        tlim    = loc.tlim 
        colors  = loc.colors
        
        data    = self.data
        
        fig,ax = plt.subplots(1,1,figsize=(8,6),dpi=loc.dpi)
        for n,sim in enumerate(sims):
            M   = data[sim]['M']
            xp0 = data[sim]['punct_0'][:,2]/M
            yp0 = data[sim]['punct_0'][:,3]/M
            xp1 = data[sim]['punct_1'][:,2]/M
            yp1 = data[sim]['punct_1'][:,3]/M
            nx1_str = 'N'+str(data[sim]['metadata']['nx1'])
            color  = colors[n]
            if loc.nsims==1:
                ccolor = 1-color
            else:
                ccolor = color/1.5
            ax.plot(xp0,yp0,color=color,  label=loc.plots_labels[n]+' +') 
            ax.plot(xp1,yp1,color=ccolor, label=loc.plots_labels[n]+' -')
        ax.set_xlabel(r'$x/M$', fontsize=20)
        ax.set_ylabel(r'$y/M$', fontsize=20)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.legend()
        utils.save_plot(figname, save=loc.savepng, show=loc.showpng, verbose=loc.verbose)
        return
    
    def plot_puncts4(self,figname='plot_puncts4.png', xlim=None, ylim=None, **kwargs):
        loc     = utils.local_vars_for_plots(self,**kwargs)
        sims    = loc.sims
        tlim    = loc.tlim 
        colors  = loc.colors
        
        data = self.data
        
        fig, axs = plt.subplots(2,2,figsize=(12,8), dpi=loc.dpi)
        for n,sim in enumerate(sims):
            M   = data[sim]['M']
            xp0 = data[sim]['punct_0'][:,2]/M
            yp0 = data[sim]['punct_0'][:,3]/M
            xp1 = data[sim]['punct_1'][:,2]/M
            yp1 = data[sim]['punct_1'][:,3]/M
            
            tp  = data[sim]['punct_0'][:,1]/M
            xp  = xp0-xp1
            yp  = yp0-yp1
            rp  = np.sqrt( xp**2 + yp**2 )

            vxp0 = data[sim]['punct_0'][:,5]
            vyp0 = data[sim]['punct_0'][:,6]
            vxp1 = data[sim]['punct_1'][:,5]
            vyp1 = data[sim]['punct_1'][:,6]
            vp0  = np.sqrt(vxp0**2 + vyp0**2)
            vp1  = np.sqrt(vxp1**2 + vyp1**2)
                
            hatE = data[sim]['metadata']['M_ADM']/M
            nu   = data[sim]['metadata']['nu']
            gammaL = (hatE**2 - 1 + 2*nu)/(2*nu)
        
            axs[0,0].plot(xp0,yp0,label=loc.plots_labels[n]+' +', color=colors[n])
            axs[0,0].plot(xp1,yp1,label=loc.plots_labels[n]+' -', color=colors[n]/1.5)
            axs[0,1].plot(tp,  rp,   color=colors[n])
            axs[1,0].plot(tp, vp0,   color=colors[n])
            axs[1,0].plot(tp, vp1,   color=colors[n]/1.5, ls='--')
            
            is_scattering = rp[-1]>1
            if is_scattering:
                vcm_in = np.sqrt( (gammaL-1) / (gammaL+1) )
                axs[1,0].axhline(vcm_in, color=colors[n])
                upoly_out0 = utils.upoly_fits(rp, vp0, nmin=2, nmax=5, n_extract=3, direction='out', \
                                             r_cutoff_low=25, r_cutoff_high=100000) 
                upoly_out1 = utils.upoly_fits(rp, vp1, nmin=2, nmax=5, n_extract=3, direction='out', \
                                             r_cutoff_low=25, r_cutoff_high=100000) 
                vinf0       = upoly_out0['extrap'] 
                vinf0_vec   = upoly_out0['extrap_vec']
                vinf1       = upoly_out1['extrap'] 
                vinf1_vec   = upoly_out1['extrap_vec']
                fit_orders  = upoly_out0['fit_orders']
                axs[1,0].axhline(vinf0,color=colors[n], ls='-.')
                axs[1,0].axhline(vinf1,color=1-colors[n], ls='-.')
                axs[1,1].scatter(fit_orders, vinf0_vec, color=colors[n])
                axs[1,1].scatter(fit_orders, vinf1_vec, color=1-colors[n])
        axs[0,0].set_xlabel(r'$x/M$')
        axs[0,0].set_ylabel(r'$y/M$')
        if xlim is not None:
            axs[0,0].set_xlim(xlim)
        if ylim is not None:
            axs[0,0].set_ylim(ylim)
        axs[0,1].set_xlabel(r'$t/M$')
        axs[0,1].set_ylabel(r'$r/M$')
        axs[0,1].set_xlim(tlim)
        axs[1,0].set_xlabel(r'$t/M$')
        axs[1,0].set_ylabel(r'$|\beta|$')
        axs[1,0].set_xlim(tlim)
        axs[1,1].set_xlabel(r'poly order')
        axs[1,1].set_ylabel(r'$v_\infty$')
        utils.save_plot(figname, save=loc.savepng, show=loc.showpng, verbose=loc.verbose)
        return 

    def plot_ft(self, figname='plot_fouriertrans.png', sample_rate=8192, use_real=False, **kwargs):
        loc     = utils.local_vars_for_plots(self,**kwargs)
        sims    = loc.sims
        tlim    = loc.tlim 
        colors  = loc.colors
        data    = self.data
        
        fig,ax = plt.subplots(1,1,figsize=(10,7),dpi=loc.dpi)
        for n,sim in enumerate(sims):
            M  = data[sim]['M']
            h  = data[sim]['h'][(2,2)]/M
            if use_real:
                h = h.real
            ls = '-'
            freq  = fftfreq(len(h), 1/sample_rate)
            fft_h = fft(h)  # Fourier transform 
            ax.plot(freq, np.abs(fft_h), linestyle=ls, color=colors[n], label=loc.plots_labels[n])
        if use_real:
            ylab = r'${\cal F}\left[ \Re\left( h \right)\right]$'
            ax.set_xlim([-150,150])
        else:
            ylab = r'${\cal F}\left[ h \right]$'
            ax.set_xlim([-125,25])
        ax.set_ylabel(ylab, fontsize=15)
        ax.set_xlabel(r'$\omega$', fontsize=15)
        ax.legend()
        utils.save_plot(figname, save=loc.savepng, show=loc.showpng, verbose=loc.verbose)
        return
    
    def plot_EJ(self, figname='plot_EJ.png', **kwargs):
        loc     = utils.local_vars_for_plots(self,**kwargs)
        sims    = loc.sims
        tlim    = loc.tlim 
        colors  = loc.colors

        data = self.data
        
        fig, axs = plt.subplots(3,1,figsize=(10,7),dpi=loc.dpi)
        for n,sim in enumerate(sims):
            M        = data[sim]['M']
            hatErad  = data[sim]['energetics']['E']/M      # integrated Edot
            hatJrad  = data[sim]['energetics']['Jz']/(M*M) # integrated Jdot
            hatE0    = data[sim]['metadata']['M_ADM']/M
            hatJ0    = data[sim]['metadata']['J_ADM']/(M*M)
            hatE_sys = hatE0 - hatErad
            hatJ_sys = hatJ0 - hatJrad
            t = data[sim]['t']
            ls = '-'
            axs[0].plot(t/M, hatE_sys, linestyle=ls, color=colors[n])
            axs[1].plot(t/M, hatJ_sys, linestyle=ls, color=colors[n])
            axs[2].plot(hatJ_sys, hatE_sys, linestyle=ls, color=colors[n])
        axs[0].set_ylabel(r'$E(t)$')
        axs[1].set_ylabel(r'$J(t)$')
        axs[2].set_ylabel(r'$E(J)$')
        axs[0].set_xlim(tlim)
        axs[1].set_xlim(tlim)
        utils.save_plot(figname, save=loc.savepng, show=loc.showpng, verbose=loc.verbose)
        return

    def plot_E_extrap(self,figname='plot_E_extrap.png', xlim=None, ylim=None, **kwargs):
        loc     = utils.local_vars_for_plots(self,**kwargs)
        sims    = loc.sims
        tlim    = loc.tlim 
        colors  = loc.colors

        data = self.data
        
        fig, axs = plt.subplots(2,2,figsize=(12,8),dpi=loc.dpi)
        for n,sim in enumerate(sims):
            M   = data[sim]['M']
            xp0 = data[sim]['punct_0'][:,2]/M
            yp0 = data[sim]['punct_0'][:,3]/M
            xp1 = data[sim]['punct_1'][:,2]/M
            yp1 = data[sim]['punct_1'][:,3]/M
            tp  = data[sim]['punct_0'][:,1]/M
            xp  = xp0-xp1
            yp  = yp0-yp1
            rp  = np.sqrt( xp**2 + yp**2 )
            tw       = data[sim]['t']/M
            hatErad  = data[sim]['energetics']['E']/M      # integrated Edot
            hatJrad  = data[sim]['energetics']['Jz']/(M*M) # integrated Jdot
            hatE0    = data[sim]['metadata']['M_ADM']/M
            hatE_sys = hatE0 - hatErad   
            
            Es = utils.spline(tw,hatE_sys,tp)
            upoly_out = utils.upoly_fits(rp, Es, nmin=1, nmax=5, n_extract=1, direction='in',
                                         r_cutoff_low=12.5, r_cutoff_high=50)
            Einf       = upoly_out['extrap'] 
            Einf_vec   = upoly_out['extrap_vec']
            fit_orders = upoly_out['fit_orders']
            mask       = upoly_out['mask']
            axs[0,0].plot(tw, hatE_sys, color=colors[n])
            axs[0,0].plot(tp, Es, ls='--', label='check', color=[1,0.7,0])
            axs[0,1].plot(xp0, yp0, color=colors[n])
            axs[0,1].plot(xp1, yp1, color=1-colors[n])
            axs[1,0].plot(rp, Es, label='full', color=colors[n])
            axs[1,0].plot(rp[mask], Es[mask], '--', label='to extrap', color=[1,0.7,0])
            axs[1,1].scatter(fit_orders, Einf_vec, color=colors[n])
            axs[1,1].axhline(hatE_sys[0], color=colors[n])
        
        axs[0,0].set_xlabel(r'$t/M$')
        axs[0,0].set_ylabel(r'$E(t)/M$')
        axs[0,0].set_xlim(tlim)
        axs[0,0].legend()
        axs[0,1].set_xlabel(r'$x/M$')
        axs[0,1].set_ylabel(r'$y/M$')
        if xlim is not None:
            axs[0,1].set_xlim(xlim)
        if ylim is not None:
            axs[0,1].set_ylim(ylim)
        axs[1,0].set_xlabel(r'$r/M$')
        axs[1,0].set_ylabel(r'$E/M$')
        axs[1,0].legend()
        axs[1,1].set_xlabel(r'poly order')
        axs[1,1].set_ylabel(r'$E_{\infty}/M$')
        utils.save_plot(figname, save=loc.savepng, show=loc.showpng, verbose=loc.verbose)
        return  

    def plot_convergence(self, wave_vars=['psi4'], conv_window=[], conv_dt=1, conv_rmpi=[], conv_labels=[], conv_SF=[],\
                         figname='plot_conv_@wave_var@.png', **kwargs):
        if len(conv_window) not in [0,2]:
            raise RuntimeError(f'Invalid input for conv_window: {conv_window}')
        if len(conv_rmpi) not in [0, loc.nsims-1]:
            raise RuntimeError(f'Invalid input for conv_rmpi: {conv_rmpi}')
        if len(conv_labels) not in [0, loc.nsims-1]:
            raise RuntimeError(f'Invalid input for conv_labels: {conv_labels}')
        
        loc     = utils.local_vars_for_plots(self,**kwargs)
        sims    = loc.sims
        tlim    = loc.tlim 
        colors  = loc.colors
         
        data = self.data
        if len(sims)<2:
            raise RuntimeError('Warning: to produce convergence plots you need at least two simulations')
        
        l = self.lm[0]
        m = self.lm[1]
        if len(conv_window):
            a = conv_window[0]
            b = conv_window[1]
        else:
            a = None
            b = None
        dt = conv_dt 
        for wave_var in wave_vars:
            amp_list  = [] # size len(sims)
            phi_list  = []
            t_list    = []
            ts_list   = []
            damp_list = [] # size len(sims), dphi[0] is all zero
            dphi_list = [] 
            fig, axs = plt.subplots(3,1,figsize=(10,8),dpi=loc.dpi)
            for n,sim in enumerate(sims):
                amp,_,M,_,_ = self.return_wave(data[sim], wave_var, l, m, wave_func='amp')
                phi,_,_,_,_ = self.return_wave(data[sim], wave_var, l, m, wave_func='phi')
                re,_,_,_,ylab_re = self.return_wave(data[sim], wave_var, l, m, wave_func='re')
                t = data[sim]['t']/M
                
                if a is None:
                    a = t[0]
                if b is None:
                    b = t[-1]

                axs[0].plot(t/M, re)
                
                if n==0:
                    zeros = np.zeros(np.shape(amp))
                    damp  = zeros
                    dphi  = zeros
                    ts    = zeros
                else:
                    t_prev   =   t_list[n-1]
                    amp_prev = amp_list[n-1]
                    phi_prev = phi_list[n-1]
                    ts, damp = utils.vec_differences(t_prev,amp_prev,t,amp,a,b,dt,diff_kind='rel',fabs=True)
                    _, dphi  = utils.vec_differences(t_prev,phi_prev,t,phi,a,b,dt,diff_kind='phi',fabs=True)
                    
                    if len(conv_rmpi)==loc.nsims-1:
                        dphi -= conv_rmpi[n-1]*np.pi
                        dphi = np.abs(dphi)
                    if len(conv_labels)!=loc.nsims-1:
                        conv_label = "N"+str(data[sims[n-1]]['metadata']['nx1'])+\
                                    "-N"+str(data[sims[ n ]]['metadata']['nx1'])
                    else:
                        conv_label = conv_labels[n-1]
                    axs[1].plot(ts,damp,color=colors[n])
                    axs[2].plot(ts,dphi,color=colors[n],label=conv_label)
                    
                    if n>1 and len(conv_SF)>0:
                        nSF = len(conv_SF)
                        for i,r in enumerate(conv_SF):
                            dxp_L = data[sims[n-2]]['metadata']['delta_xp']
                            dxp_M = data[sims[n-1]]['metadata']['delta_xp']
                            dxp_H = data[sims[ n ]]['metadata']['delta_xp']
                            SFr   = ( dxp_M**r - dxp_L**r  ) / (dxp_H**r - dxp_M**r )
                            cf    = 0.5*(nSF-i-1)/nSF + 1.2 # color-factor
                            axs[1].plot(ts,damp*SFr,color=colors[n]/cf,ls='--')
                            axs[2].plot(ts,dphi*SFr,color=colors[n]/cf,ls='--',\
                                        label=conv_label+f'-SF({r:d})')
                
                t_list.append(t)
                ts_list.append(ts)
                amp_list.append(amp)
                phi_list.append(phi)
                damp_list.append(damp)
                dphi_list.append(dphi)
            lm = f'{l:d}{m:d}' 
            axs[0].set_xlim([a,b])
            axs[0].set_ylabel(ylab_re, fontsize=20)
            axs[1].set_ylabel(r'$\Delta A_{'+lm+'}/A_{'+lm+'}$', fontsize=20)
            axs[2].set_ylabel(r'$\Delta \phi_{'+lm+'}$', fontsize=20)
            axs[2].set_xlabel(r'$t/M$', fontsize=20)
            axs[2].legend(loc='lower right')
            for i in range(1,3): 
                axs[i].set_xlim([a,b])
                axs[i].set_yscale('log')
                axs[i].grid()
            figname_wv = figname.replace('@wave_var@', wave_var)
            utils.save_plot(figname, save=loc.savepng, show=loc.showpng, verbose=loc.verbose)
        return
    
    def plot_scats(self, xvars, xlab='', figname='plot_scat_data.png', **kwargs):
        
        nxvars = len(xvars)
        if nxvars!=loc.nsims:
            raise RuntimeError(f'Size of data_scat input ({nvars}) not compatible with number of sims ({self.nsims})')
        all_scatterings = True
        data   = self.data
        for sim in sims:
            if data[sim]['chi'] is None:
                all_scatterings = False
                print(f'{sim} is not a scattering!')
                break
        if all_scatterings:
            fig_chi,ax_chi = plt.subplots(1,1,figsize=(8,6),dpi=loc.dpi)
            y = []
            y_err = []
            for n,sim in enumerate(sims):
                chi = data[sim]['chi']
                chi_fit_err = data[sim]['chi_fit_err']
                y.append(chi)
                y_err.append(chi_fit_err)
                if self.verbose:
                    print(f'{sim} chi +- fit_err : {chi:.3f} +- {chi_fit_err:.3f}')
            plt.errorbar(xvars, y, yerr=y_err, fmt='o', capsize=5)
            plt.xlabel(xlab, fontsize=20)
            plt.ylabel(r'$\chi [{}^\circ]$', fontsize=20)
            plt.grid()
            utils.save_plot(figname, save=loc.savepng, show=loc.showpng, verbose=loc.verbose)
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dset',          default='GAUSS_2023',  help="select dset: 'GAUSS_2021' or 'GAUSS_2023'")
    parser.add_argument('-s', '--sims',    nargs='+', default=[], help='simulations to consider (either names or indeces)', required=False)
    parser.add_argument('-v', '--verbose', action='store_true',   help='print info')
    parser.add_argument('-l', '--list',    action='store_true',   help='Print list of available simulations')
    parser.add_argument('--lmmax',      type=int, default=5,      help='Maximum multipole m=l to consider')
    parser.add_argument('--rm_last_Dt', type=float, nargs='+',  
                                        default=[],               help='Remove last Delta_t from GW signals; list of float')
    
    # available plots
    parser.add_argument('-p', '--plots', nargs='+', default=[],  help='Plots, e.g. --plots puncts abs',
                        choices=['puncts', 'puncts4', 'ahf', 
                                 'amp', 're', 'im', 'omg', 
                                 'summary', 'ft', 'EJ','psi4_radii', 
                                 'E_extrap', 'conv']) 
    parser.add_argument('-w', '--wave_vars', default=['psi4'],   help='Waves to use in abs/re/im/omg plots', 
                         nargs='+', choices=['psi4','dh','h'])
    parser.add_argument('--plots_scat', type=float, nargs='+',default=[], 
                                                              help='Plot of the scat-angles vs list of floats ' 
                                                                   'given in input, e.g. --plots_scat 0.75 1.00 2.00')

    # generic options to use in plots
    parser.add_argument('-t','--tlim',    nargs='+', default=[], 
                                          type=float,               help='tlim to use in time-plots, e.g. 0 700')
    parser.add_argument('--lm',           nargs='+', default=[2,2],
                                          type=int,                 help='Multipole to consider in amp/re/im/omg plots')
    parser.add_argument('--ft_real',      action='store_true',      help='Use only real part of h in FT of ft-plot')
    parser.add_argument('--ylog',         action='store_true',      help='Use log Y-scale in amp/re/im/omg plots')
    parser.add_argument('--plots_labels', nargs='+', default=[],    help='Labels to use in plots') 
    parser.add_argument('--savepng',      action='store_true',      help='Save figures in PNG format')
    parser.add_argument('--hidefigs',     action='store_true',      help="Don't show figures")
    
    # Integration options
    parser.add_argument('-i','--integral',default='FFI', 
                        choices = ['FFI', 'TDI'],        type=str,   help="method to use for integration")
    parser.add_argument('-f','--FFI_f0',    default=0.01,type=float, nargs='+',  help="f0 used in FFI; float or list (size of sims)")
    parser.add_argument('-d','--TDI_degree',default=1,   type=int,   help="poly-degree used in TDI")
    parser.add_argument('--TDI_poly_int',   default=None,type=float,
                                            nargs='+',               help="poly-interval used in TDI")
    parser.add_argument('--extrap_psi4',    action='store_true',     help="Extrapolate psi4 before integrating")
    parser.add_argument('--integral_check', action='store_true',     help='Add a check on the integration in abs/re/im plots for psi4 and dh')
    parser.add_argument('--out_data',       action='store_true',     help="Dump processed data")

    # Options for convergence plots
    parser.add_argument('--conv_dt',     default=1.0,           type=float, help='Time-step to use in convergence plots')
    parser.add_argument('--conv_window', nargs='+', default=[], type=float, help='Time window to use in convergence plots')
    parser.add_argument('--conv_rmpi',   nargs='+', default=[], type=int,   help='Subtract specified multiples of pi from phase-diff '\
                                                                                 '(list with size self.nsims-1)')
    parser.add_argument('--conv_labels', nargs='+', default=[], type=str,   help='Labels to use in convergence plots (list with size self.nsims-1)')
    parser.add_argument('--conv_SF',     nargs='+', default=[], type=int,   help='Rescale differences with scaling factor'
                                                                                 '(do not scale the first diff)')
    args = parser.parse_args()
    
    SimMain = simulations.Sim(dset=args.dset)
    if args.list:
        SimMain.print_simlist()
    
    sims = [] 
    simlist = SimMain.simlist()
    for elem in args.sims:
        try:
            sims.append(simlist[int(elem)])
        except ValueError:
            sims.append(elem)
    
    for sim in sims:
        if sim not in simlist:
            raise ValueError(f'Unknown simulation: {sim}')
    
    pdata = ProcessData(dset=args.dset, sims=sims, verbose=args.verbose, out_data=args.out_data,
                        savepng=args.savepng, showpng=not args.hidefigs, extrap_psi4=args.extrap_psi4,
                        integral=args.integral, FFI_f0=args.FFI_f0, TDI_degree=args.TDI_degree, TDI_poly_int=args.TDI_poly_int, 
                        plots_labels=args.plots_labels, tlim=args.tlim, lmmax=args.lmmax, rm_last_Dt=args.rm_last_Dt)
    
    if len(args.plots)>0:
        plt.ion()

    for plot_type in args.plots:
        if plot_type=='summary':
            pdata.plot_summary()
        elif plot_type=='ahf':
            pdata.plot_ahf()
        elif plot_type in ['amp', 're', 'img', 'omg']:
            pdata.plot_waveform(wave_vars=args.wave_vars, wave_func=plot_type, integral_check=args.integral_check, l=args.lm[0], m=args.lm[1], ylog=args.ylog)
        elif plot_type=='puncts':
            pdata.plot_puncts()
        elif plot_type=='puncts4':
            pdata.plot_puncts4()
        elif plot_type=='psi4_radii':
            pdata.plot_psi4_radii()    
        elif plot_type=='ft':
            pdata.plot_ft(use_real=args.ft_real)
        elif plot_type=='EJ':
            pdata.plot_EJ()
        elif plot_type=='E_extrap':
            pdata.plot_E_extrap()
        elif plot_type=='conv':
            pdata.plot_convergence(wave_vars=args.wave_vars,
                                          conv_window=args.conv_window, conv_dt=args.conv_dt, conv_SF=args.conv_SF, 
                                          conv_rmpi=args.conv_rmpi, conv_labels=args.conv_labels) 
    if len(args.plots_scat)>0:
        pdata.plot_scats(args.plots_scat, xlab=r'$x-var$')
    
    if len(args.plots)>0:
        user_input = input("Press Enter to close plots...")
        plt.ioff()



