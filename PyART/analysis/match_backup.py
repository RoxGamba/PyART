"""
Stuff for mismatches, still need to port the parallelization,
the precessing case and debug/test the code
"""

import numpy as np
from scipy.optimize import minimize_scalar, dual_annealing
from ..utils import utils as ut

#PyCBC imports
from pycbc.filter import sigmasq, matched_filter_core, overlap_cplx, optimized_match
from pycbc.types.timeseries import TimeSeries
from pycbc.psd import  aLIGOZeroDetHighPower

class Matcher(object):
    """
    Class to compute the match between two waveforms.
    Assume that one of them is a template, and the other is the fixed target.
    """
    def __init__(self,
                 target_wf,
                 template_wf,
                 modes     = [(2,2)],
                 settings  = None
                 ) -> None:
        
        self.modes    = modes
        self.settings = self.__default_parameters__()
        self.settings.update(settings)

        # choose overlap function based on settings
        if   settings['overlap_kind'] == 'sky_time_phase_maxed_overlap':
            # overlap function for precessing 22 modes
            self.overlap_func = sky_time_phase_maxed_overlap
        elif settings['overlap_kind'] == 'sky_time_maxed_overlap':
            # most generic overlap function, requires numerical optimization
            # over phase
            self.overlap_func = sky_time_maxed_overlap
        elif settings['overlap_kind'] == 'pol_time_maxed_overlap':
            # similar to above, but does not max over skyloc, and assumes 
            # that target is h+ only
            self.overlap_func = pol_time_maxed_overlap
        elif settings['overlap_kind'] == 'phase_time_maxed_overlap':
            # Valid for non-precessing single modes
            self.overlap_func = phase_time_maxed_overlap
        else:
            raise ValueError('kind not recognized')
        
        # compute the mismatch
        m = self.compute_match(target_wf,template_wf,settings)

        return m

    def __default_parameters__(self):
        """
        Default parameters for the mismatch calculation
        """
        return {
            'initial_frequency_mm' : 20.,
            'final_frequency_mm'   : 2048.,
            'taper'                : True,
            'psd'                  : 'aLIGOZeroDetHighPower',
            'M'                    : 100.,
            'iota'                 : 0.,
            'coa_phase'            : np.linspace(0,2*np.pi,4),
            'eff_pols'             : np.linspace(0,np.pi,5),
        }
    
    def _get_psd(self, flen, df, fmin):
        """
        Get the PSD for the mismatch calculation
        """
        if self.settings['psd'] == 'aLIGOZeroDetHighPower':
            psd = aLIGOZeroDetHighPower(flen, df, fmin)
        else:
            raise ValueError('psd not recognized')
        return psd

    def compute_match(self, target_wf, template_wf, settings):
        """
        assumes that the target_wf is fixed and in time domain.
        The template is the 'model', that will be optimized over (if needed)
        """

        # condition the waveforms (taper, resize, etc)
        if target_wf.domain == 'Time':
            ht = condition_td_waveform(target_wf, settings['tlen'], settings['dt'])
            htf = ht.to_frequencyseries()
        else:
            ht = target_wf
        
        if self.psd == None:
            df   = 1.0 / ht.duration
            flen = len(ht)//2 + 1
            psd  = self._get_psd(flen, df, settings['initial_frequency_mm'])
        else:
            psd = self.psd

        m  = minimize_overlap(
                                htf, 
                                template_wf,
                                psd, 
                                self.overlap_func, 
                                settings,
                                regen_wf=False
                                )

        return m

    def sky_average_match(self, wft, wftmp, settings):
        """
        sky average the match
        """

        iota = settings['iota']
        for coa_phase in settings['coa_phase']:
            sp,sx = wft.compute_hphc(iota, coa_phase)
            sp    = condition_td_waveform(sp, settings['tlen'], settings['dt'])
            sx    = condition_td_waveform(sx, settings['tlen'], settings['dt'])
            spf   = sp.to_frequencyseries() 
            sxf   = sx.to_frequencyseries()
            psd   = self._get_psd(len(spf), spf.delta_f, settings['initial_frequency_mm'])
            self.psd = psd

            for k in settings['eff_pols']:
                s  = np.cos(k)*spf + np.sin(k)*sxf
                mm = self.compute_match(s, wftmp, settings):
            
        return mm
    
### other functions, not just code related to the class

def condition_td_waveform(h, tlen, settings):
    """
    Condition the waveforms before computing the mismatch
    """
    # taper the waveform
    if settings['taper']:
        h = h #taper_waveform()
    # make this a TimeSeries
    h = TimeSeries(h, settings['dt'])
    # resize the waveform
    h.resize(tlen)
    return h

def dual_annealing(func,bounds,maxfun=2000):
    result= dual_annealing(func, bounds, maxfun=maxfun)#, local_search_options={"method": "Nelder-Mead"})
    opt_pars,opt_val=result['x'],result['fun']
    return opt_pars, opt_val

### Overlap functions ###

def sky_time_phase_maxed_overlap(s, hp, hc, psd, low_freq, high_freq):
    """
    https://arxiv.org/abs/1603.02444
    """
    ss   = sigmasq(s,  psd=psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)
    hphp = sigmasq(hp, psd=psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)
    hchc = sigmasq(hc, psd=psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)
    hp  /= np.sqrt(hphp)
    hc  /= np.sqrt(hchc)

    rhop, _, nrm     = matched_filter_core(hp,s, psd = psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)
    rhop *= nrm
    rhoc, _, nrm     = matched_filter_core(hc,s, psd = psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)
    rhoc *= nrm
    
    hphccorr       = overlap_cplx(hp, hc, psd=psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)# matched_filter_core(hp,hc)
    hphccorr       = np.real(hphccorr)
    Ipc            = hphccorr

    rhop2 = np.abs(rhop.data)**2
    rhoc2 = np.abs(rhoc.data)**2
    gamma = rhop.data * np.conjugate(rhoc.data)
    gamma = np.real(gamma)

    sqrt_part = np.sqrt((rhop2-rhoc2)**2 + 4*(Ipc*rhop2-gamma)*(Ipc*rhoc2-gamma))
    num       = rhop2 - 2.*Ipc*gamma + rhoc2 + sqrt_part
    den       = 1. - Ipc**2
    
    o = np.sqrt(max(num)/den/2.)/np.sqrt(ss)
    if (o > 1.):
         o = 1.

    return o

def sky_time_maxed_overlap(s, hp, hc, psd, low_freq, high_freq):
    """
    https://arxiv.org/pdf/1709.09181
    """
    
    ss   = sigmasq(s,  psd=psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)
    hphp = sigmasq(hp, psd=psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)
    hchc = sigmasq(hc, psd=psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)
    hp  /= np.sqrt(hphp)
    hc  /= np.sqrt(hchc)

    rhop, _, nrm     = matched_filter_core(hp,s, psd = psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)
    rhop *= nrm
    rhoc, _, nrm     = matched_filter_core(hc,s, psd = psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)
    rhoc *= nrm
    
    hphccorr       = overlap_cplx(hp, hc, psd=psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)# matched_filter_core(hp,hc)
    Ipc            = np.real(hphccorr)
    
    re_rhop  = np.real(rhop.data)
    re_rhoc  = np.real(rhoc.data) 
    re_rhop2 = re_rhop**2
    re_rhoc2 = re_rhoc**2

    num   = re_rhop2 - 2.*Ipc*re_rhop*re_rhoc + re_rhoc2
    den   = 1. - Ipc**2

    o = np.sqrt(max(num)/den)/np.sqrt(ss)

    if (o > 1.):
        o = 1.

    return o

def pol_time_maxed_overlap(s, hp, hc, psd, low_freq, high_freq, max_pol=True):
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

def phase_time_maxed_overlap(s, hp, hc, psd, low_freq, high_freq, max_pol=True):
    """
    TODO: wrap PyCBC's optimized_match
    """
    pass

### minimizators ###

def minimize_overlap(target_wf, template_wf, psd, overlap_func, settings, regen_wf=False):
    """
    Minimize the overlap function to find the best match.
    """

    # list of parameters to numerically optimize over
    numerical_pars = settings['numerical_pars']

    def to_minimize_genwf(x):
        """
        Maximize over parameters that require the generation of the waveform
        """
        pars = template_wf.pars

        # Update the intrinsic parameters
        for i,k in enumerate(numerical_pars):
            if k in ['iota','coa_phase', 'M', 'H_hyp', 'j_hyp']: pars[k] = x[i]
        
        # Rotate the spins
        if 'theta' in numerical_pars:
            # TODO: implement this
            pars = pars

        # gen the waveform
        this_wf = template_wf.generate_wf(pars)
        hp, hc  = this_wf.hp, this_wf.hc
        hp = condition_td_waveform(hp, settings['tlen'], settings['dt'])
        hc = condition_td_waveform(hc, settings['tlen'], settings['dt'])
        hpf = hp.to_frequencyseries()
        hcf = hc.to_frequencyseries()
        return 1.-overlap_func(target_wf, hpf, hcf, psd, settings['initial_frequency_mm'], settings['final_frequency_mm'])
    
    def to_minimize(x):
        """
        Only numerically maximize over the reference phase
        """
        hp, hc = template_wf.compute_hphc(x, settings['iota'], modes=settings['modes'])
        hp = condition_td_waveform(hp, settings['tlen'], settings['dt'])
        hc = condition_td_waveform(hc, settings['tlen'], settings['dt'])
        hpf = hp.to_frequencyseries()
        hcf = hc.to_frequencyseries()
        return 1.-overlap_func(target_wf, hpf, hcf, psd, settings['initial_frequency_mm'], settings['final_frequency_mm'])
    
    def no_minimize():
        """
        Compute the overlap without any minimization
        """
        hp, hc = template_wf.compute_hphc(0., settings['iota'], modes=settings['modes'])
        hp = condition_td_waveform(hp, settings['tlen'], settings['dt'])
        hc = condition_td_waveform(hc, settings['tlen'], settings['dt'])
        hpf = hp.to_frequencyseries()
        hcf = hc.to_frequencyseries()
        return 1.-overlap_func(target_wf, hpf, hcf, psd, settings['initial_frequency_mm'], settings['final_frequency_mm'])

    if numerical_pars == None:
        return no_minimize()
    elif regen_wf:
        func = to_minimize_genwf
    else:
        func = to_minimize

    res = 1.
    if len(numerical_pars) == 1:
        res = minimize_scalar(
                func,
                method="bounded",
                bounds=settings['bounds'][0],
                options={'xatol':1e-15}
                )
        res = to_minimize(res.x)
        if res < 1e-2:
            return res
    else:
        _, res = dual_annealing(
                 func,
                 settings['bounds'],
                 maxfun=100
             )
    return res