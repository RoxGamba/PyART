"""
Stuff for mismatches, still need to port the parallelization,
the precessing case and debug/test the code
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar, dual_annealing
from ..utils import utils as ut

#PyCBC imports
from pycbc.filter import sigmasq, matched_filter_core, overlap_cplx, optimized_match, matched_filter,compute_max_snr_over_sky_loc_stat_no_phase
from pycbc.types.timeseries import TimeSeries, FrequencySeries
from pycbc.psd import aLIGOZeroDetHighPower, sensitivity_curve_lisa_semi_analytical
from pycbc.psd.read import from_txt

class Matcher(object):
    """
    Class to compute the match between two waveforms.
    """
    def __init__(self,
                 WaveForm1,
                 WaveForm2,
                 settings  = None,
                 pre_align = False,
                 ) -> None:
        
        self.settings = self.__default_parameters__()
        if settings:
            self.settings.update(settings)
        self.modes = settings.get('modes', [])

        # Choose the appropriate mismatch function
        if settings['kind'] == 'single-mode':
            self.match_f = self._compute_mm_single_mode
        elif settings['kind'].lower() == 'hm':
            self.match_f = self._compute_mm_skymax
        else:
            raise ValueError(f"Kind '{settings['kind']}' not recognized")
    
        # Get local objects with TimeSeries
        wf1 = self._wave2locobj(WaveForm1)
        wf2 = self._wave2locobj(WaveForm2)
        
        # align to improve subsequent tapering (applied before matching computation)
        if pre_align: 
            wf1, wf2 = self.pre_alignment(wf1, wf2)
        
        # Determine time length for resizing
        self.settings['tlen'] = self._find_tlen(wf1, wf2, resize_factor=settings['resize_factor'])
        # Compute and store the mismatch
        self.mismatch = 1 - self.match_f(wf1,wf2,self.settings)
    
    def _wave2locobj(self, WaveForm, isgeom=True):
        """
        Extract useful information from WaveForm class 
        (see PyART/waveform.py) and store in a lambda-obj 
        that will be used in this Matcher class
        """
        if not hasattr(WaveForm, 'hp'):
            raise RuntimeError('hp not found! Compute it before calling Matcher')
        
        wf = lambda: None
        wf.domain = WaveForm.domain
        wf.f = None  # Assume TD waveform at the moment
        wf.hlm = WaveForm.hlm
        wf.compute_hphc = WaveForm.compute_hphc
        wf.t = WaveForm.u
        
        # Get updated time and hp/hc-TimeSeries
        wf.hp, wf.hc, wf.u = self._mass_rescaled_TimeSeries(WaveForm.u, WaveForm.hp, WaveForm.hc, isgeom=isgeom)

        # also update the modes in a TimeSeries
        wf.modes = {}
        for k in WaveForm.hlm.keys():
            re = WaveForm.hlm[k]['real']
            im = WaveForm.hlm[k]['imag']
            re_lm, im_lm, _ = self._mass_rescaled_TimeSeries(WaveForm.u, re, im, isgeom=isgeom)
            wf.modes[k] = {'real':re_lm, 'imag':im_lm}

        return wf

    def _mass_rescaled_TimeSeries(self, u, hp, hc, isgeom=True, kind='cubic'):
        """
        Rescale waveforms with the mass used in settings
        and return TimeSeries.
        If the waveform is not in geom-units, the simply
        return the TimeSeries
        """
        # TODO : test isgeom==False
        dT = self.settings['dt']
        if isgeom:
            M       = self.settings['M'] 
            dT_resc = dT/(M*ut.Msun)
            new_u   = np.arange(u[0], u[-1], dT_resc)
            hp = ut.spline(u, hp, new_u, kind=kind) 
            hc = ut.spline(u, hc, new_u, kind=kind) 
        return TimeSeries(hp, dT), TimeSeries(hc, dT), new_u
    
    def pre_alignment(self, wf1, wf2):
        """
        Align waveforms (TimeSeries) before feeding 
        them to the conditioning/matching functions. 
        This is needed to improve tapering-consistency
        """
        if not self.settings['taper']:
            warnings.warn('Pre-alignment is not needed if no tapering is applied!')
        # and now? 
        return wf1, wf2

    def _find_tlen(self, wf1, wf2, resize_factor=2):
        """
        Given two local-waveform objects (see wave2locobj()),
        return the time-length to use in TD-waveform
        conditioning (before match computation)
        """
        dT   = self.settings['dt']
        h1   = TimeSeries(wf1.hp, dT)
        h2   = TimeSeries(wf2.hp, dT)
        LM   = max(len(h1), len(h2))
        tl   = (LM-1)*dT
        tN   = ut.nextpow2(resize_factor*tl)
        tlen = int(tN/dT)
        return tlen

    def __default_parameters__(self):
        """
        Default parameters for the mismatch calculation
        """
        return {
            'initial_frequency_mm' : 20.,
            'final_frequency_mm'   : 2048.,
            'psd'                  : 'aLIGOZeroDetHighPower',
            'dt'                   : 1./4096,
            'M'                    : 100.,
            'iota'                 : 0.,
            'coa_phase'            : np.linspace(0,2*np.pi,1),
            'eff_pols'             : np.linspace(0,np.pi,1),
            'taper'                : True,
            'taper_start'          : 0.05, # % of the waveform to taper at the beginning
            'taper_end'            : 0.00, # % of the waveform to taper at the end
            'taper_alpha'          : 0.01,  # sigmoid-parameter used in tapering
            'resize_factor'        : 2,
            'debug'                : False,
            'geom'                 : True,
        }
    
    def _get_psd(self, flen, df, fmin):
        """
        Get the PSD for the mismatch calculation
        """
        if self.settings['psd'] == 'aLIGOZeroDetHighPower':
            psd = aLIGOZeroDetHighPower(flen, df, fmin)
        elif self.settings['psd'] == 'LISA':
            psd = sensitivity_curve_lisa_semi_analytical(flen, df, fmin)
        elif self.settings['psd'] == 'flat':
            psd = FrequencySeries(np.ones(flen), delta_f=df)
        elif self.settings['psd'] == 'txt':
            psd = from_txt(filename=self.settings['asd_file'],
                           length=flen, delta_f=df, low_freq_cutoff=fmin, is_asd_file=True)
        else:        
            raise ValueError('psd not recognized')
        return psd

    def _compute_mm_single_mode(self, wf1, wf2, settings):
        """
        Compute the mismatch between two waveforms with only a single mode.
        Use either h+ (modes-or-pol = 'pol') or the mode itself (modes-or-pol = 'modes')
        This is true for non-precessing systems with a single (ell, |m|)
        """

        if settings['modes-or-pol'] == 'pol':
            h1_nc = wf1.hp
            h2_nc = wf2.hp
        elif settings['modes-or-pol'] == 'modes':
            if len(settings['modes']) > 1:
                raise ValueError('Only one mode is allowed in this function')
            h1_nc = wf1.modes[settings['modes'][0]]['real']
            h2_nc = wf2.modes[settings['modes'][0]]['real']
        
        # condition TD waveforms (taper, resize, etc)
        if wf1.domain == 'Time':
            h1 = condition_td_waveform(h1_nc, settings)
        if wf2.domain == 'Time':
            h2 = condition_td_waveform(h2_nc, settings)
        
        assert len(h1) == len(h2)
        df   = 1.0 / h1.duration
        flen = len(h1)//2 + 1
        psd  = self._get_psd(flen, df, settings['initial_frequency_mm'])
        
        if settings['debug']:
            self._debug_plot_waveforms(h1_nc, h2_nc, h1, h2, psd, settings)
                  
        m,_  = optimized_match( h1, h2, 
                                psd=psd, 
                                low_frequency_cutoff=settings['initial_frequency_mm'], 
                                high_frequency_cutoff=settings['final_frequency_mm']
                                )

        return m

    def _debug_plot_waveforms(self, h1_nc, h2_nc, h1, h2, psd, settings):
        """
        Plot waveforms and PSD for debugging.
        """
        plt.figure(figsize=(15, 7))

        plt.subplot(231)
        plt.title('Real part of waveforms before conditioning')
        plt.plot(h1_nc.sample_times, h1_nc, label='h1 unconditioned', color='blue', linestyle='-')
        plt.plot(h2_nc.sample_times, h2_nc, label='h2 unconditioned', color='green', linestyle='-')
        plt.legend()

        plt.subplot(232)
        plt.title('Real part of waveforms after conditioning')
        plt.plot(h1.sample_times, h1, label='h1 conditioned', color='blue')
        plt.plot(h2.sample_times, h2, label='h2 conditioned', color='green')
        plt.legend()

        plt.subplot(234)
        plt.title('PSD used for match')
        plt.loglog(psd.sample_frequencies, np.sqrt(psd.data* psd.sample_frequencies), label='PSD', color='black')
        plt.legend()

        plt.subplot(235)
        plt.title('Overlap integrand between h1 and h2')
        freq = np.linspace(settings['initial_frequency_mm'], settings['final_frequency_mm'], len(h1))
        plt.plot(freq, abs(h1.data * h2.data), color='red')
        
        hf1 = h1.to_frequencyseries()
        f1  = hf1.get_sample_frequencies()
        hf2 = h2.to_frequencyseries()
        f2  = hf2.get_sample_frequencies()
        Af1 = np.abs(hf1)
        Af2 = np.abs(hf2)
        for i in [3,6]:
            plt.subplot(2,3,i)
            plt.title('Fourier transforms (abs value)')
            plt.plot(f1, Af1, c='blue',  label='FT h1')
            plt.plot(f2, Af2, c='green', label='FT h2')
            plt.axvline(settings['initial_frequency_mm'], lw=0.8, c='r')
            plt.axvline(settings['final_frequency_mm'],   lw=0.8, c='r')
            plt.grid()
            plt.yscale('log')
            #plt.ylim(1e-6, max(1.2*max(Af1),1.2*max(Af2),0.1))
            plt.legend()
            if i==3:
                plt.xscale('log')
                
        plt.tight_layout()
        if 'save' not in settings.keys():
            print("not saving")
        if 'save' not in settings.keys():
            plt.show()
        else:
            print("Saving to ", settings['save'])
            plt.savefig(f"{settings['save']}", dpi=100, bbox_inches='tight')

    def _compute_mm_skymax(self, wf1, wf2, settings):
        """
        Compute the match between two waveforms with higher modes.
        Use wf1 as a fixed target, and decompose wf2 in modes to find the
        best orbital phase.
        """

        iota = settings['iota']
        mms = []

        for coa_phase in settings['coa_phase']:
            sp,sx     = wf1.compute_hphc(coa_phase, iota, modes=self.modes)
            sp, sx, _ = self._mass_rescaled_TimeSeries(wf1.t, sp, sx, isgeom=settings['geom']) 
            sp        = condition_td_waveform(sp, settings)
            sx        = condition_td_waveform(sx, settings)
            spf       = sp.to_frequencyseries() 
            sxf       = sx.to_frequencyseries()
            psd       = self._get_psd(len(spf), spf.delta_f, settings['initial_frequency_mm'])

            for k in settings['eff_pols']:
                s = np.cos(k)*spf + np.sin(k)*sxf

                mm  = self.skymax_match(
                    s, wf2, 
                    iota,
                    psd,
                    self.modes,
                    dT=settings['dt'],
                    fmin_mm=settings['initial_frequency_mm'],
                    fmax=settings['final_frequency_mm']
                )
                mms.append(mm)
        return np.average(mm)
    
    def skymax_match(self,  
                     s, wf, 
                     inc, psd, modes,
                     dT=1./4096,
                     fmin_mm=20.,
                     fmax=2048.):

        def to_minimize_dphi(x):
            hp, hc   = wf.compute_hphc(x, inc, modes=modes)
            hp, hc, _= self._mass_rescaled_TimeSeries(wf.t, hp, hc, isgeom=self.settings['geom'])
            hps      = condition_td_waveform(hp, self.settings)
            hxs      = condition_td_waveform(hc, self.settings)
            # To FD
            hpf = hps.to_frequencyseries()
            hcf = hxs.to_frequencyseries()
            return 1.-sky_and_time_maxed_overlap(s, hpf, hcf, 
                                                    psd,self.settings['initial_frequency_mm'],
                                                    self.settings['final_frequency_mm'], kind=self.settings['kind'])

        res_ms = minimize_scalar(
                    to_minimize_dphi,
                    method="bounded",
                    bounds=(0, 2.*np.pi),
                    options={'xatol':1e-15}
                    )
        res = to_minimize_dphi(res_ms.x)
        #print('minimize_scalar:', res)
        if  res > 1e-2:
            # try also with (more expensive) dual annealing
            # and choose the minimum
            _, res_da = dual_annealing_wrap(
                    to_minimize_dphi,
                    [(0., 2.*np.pi)],
                    maxfun=100
                )
            res = min(res, res_da)

        return 1. - res
        
### other functions, not just code related to the class
def condition_td_waveform(h, settings):
    """
    Condition the waveforms before computing the mismatch.
    h is already a TimeSeries
    """
    # taper the waveform
    if settings['taper']:
        hlen = len(h)
        t1    = settings['taper_start'] * hlen
        t2    = settings['taper_end']   * hlen
        alpha = settings['taper_alpha']
        t = np.linspace(0, hlen-1, num=hlen)
        h = ut.taper_waveform(t, h, t1=t1, t2=t2, alpha=alpha)
    # resize
    h.resize(settings['tlen'])
    return h

def dual_annealing_wrap(func,bounds,maxfun=2000):
    result= dual_annealing(func, bounds, maxfun=maxfun)#, local_search_options={"method": "Nelder-Mead"})
    opt_pars,opt_val=result['x'],result['fun']
    return opt_pars, opt_val

def sky_and_time_maxed_overlap(s, hp, hc, psd, low_freq, high_freq, kind='hm'):
    """
    Compute the sky and time maximized overlap between a signal and a template.
    See https://arxiv.org/pdf/1709.09181 and Eq. 10 of https://arxiv.org/pdf/2207.01654
    """
    signal_norm = s / np.sqrt(sigmasq(s, 
                                      psd=psd, 
                                      low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq))
    hplus_norm = hp / np.sqrt(sigmasq(hp, 
                                      psd=psd, 
                                      low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq))
    hcross_norm = hc / np.sqrt(sigmasq(hc, 
                                       psd=psd,
                                       low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq))
    hphc_corr = np.real(overlap_cplx(hplus_norm, 
                                        hcross_norm, 
                                        psd=psd, 
                                        low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq,
                                        normalized=False))
    Iplus = matched_filter(hplus_norm, 
                            signal_norm, 
                            psd=psd,
                            sigmasq=1.,
                            low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq,)
    Icross = matched_filter(hcross_norm, 
                            signal_norm, 
                            psd=psd,
                            sigmasq=1.,
                            low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq,)
    
    det_stat = compute_max_snr_over_sky_loc_stat_no_phase(Iplus, 
                                                                Icross, 
                                                                hphccorr=hphc_corr,
                                                                hpnorm=1.,
                                                                hcnorm=1.,
                                                                thresh=0.1,
                                                                analyse_slice=slice(0, len(Iplus.data))
                                                                )

    i = np.argmax(det_stat.data)
    o = det_stat[i]
    if (o > 1.):
        o = 1.

    return o

def time_maxed_overlap(s, hp, hc, psd, low_freq, high_freq, max_pol=True):
    """
    Assume s is + only. 
    We allow for a polarization shift, i.e. a **global** change of sign in the waveform.
    TODO: check if this is implemented correctly, see Sec. VD of https://arxiv.org/abs/1812.07865
    """
    
    ss   = sigmasq(s,  psd=psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)
    hphp = sigmasq(hp, psd=psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)
    hp  /= np.sqrt(hphp)

    rhop, _, nrm  = matched_filter_core(hp, s, psd = psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)
    rhop         *= nrm
    
    # allow for different polarization conventions
    # potential global - sign change
    os = []
    for ang in [np.pi/2, 0.]:
        re_rhop  = np.real(rhop.data*np.exp(2*1j*ang))
        num   = re_rhop
        o     = max(num)/np.sqrt(ss)
        os.append(o)

    # maximize over allowed polarization
    if max_pol: o = max(os)
    else:       o = os[0]

    if (o > 1.):
        o = 1.
    return o
