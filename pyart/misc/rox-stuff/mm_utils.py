import numpy as np; import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# my stuff
import utils as ut; import wf_utils as wfu; import wf

# PyCBC imports for match
from pycbc.filter import sigmasq, matched_filter_core, overlap_cplx, optimized_match
from pycbc.types.timeseries import TimeSeries
from pycbc.psd import  aLIGOZeroDetHighPower
from pycbc.waveform.utils import taper_timeseries

def hpc_to_timeseries(hp, hc, dT):
    hpTS = TimeSeries(hp, dT)
    hcTS = TimeSeries(hc, dT)
    return hpTS, hcTS

def condition_hpc(hp, hc, tlen, kind='pycbc', M=1., Msuns=1, alpha=0.1, tau=1.):
    if kind == 'pycbc':
        hp   = taper_timeseries(hp,tapermethod="startend")
        hc   = taper_timeseries(hc,tapermethod="startend")
    elif kind == 'manual':
        dT   = hp.delta_t
        hp   = TimeSeries(wfu.taper(hp.sample_times, hp, M, alpha, tau, Msuns=Msuns), dT)
        hc   = TimeSeries(wfu.taper(hc.sample_times, hc,M, alpha, tau, Msuns=Msuns), dT)
    else:
        raise ValueError('kind not recognized')
    hp.resize(tlen)
    hc.resize(tlen)
    return hp, hc

def compute_one_mismatch_hp_22(wf1, wf2, settings, M):
    """
    Assumes waveforms are already interpolated to common time array, and
    given in physical units
    """
    f0 = settings['initial_frequency_mm']
    hs = []
    for i,w in enumerate([wf1, wf2]): 
        h  = w.hp
        # Time Series
        if (settings['taper'] and settings['taper_kind'] == 'manual'):
            h = wfu.taper(w.u, h, M, settings['alpha'][i] , settings['tau'][i], Msuns=ut.Msuns)
        hTS   = TimeSeries(h, settings['dT'])
        if (settings['taper'] and settings['taper_kind'] == 'pycbc'):
                hTS = taper_timeseries(hTS,tapermethod="startend")
        hs.append(hTS)
    
    h1_TS = hs[0]
    h2_TS = hs[1]  
    # Resize both to next power of 2
    LM        = max(len(h1_TS), len(h2_TS))
    tl        = (LM-1)*settings['dT']
    tN        = ut.nextpow2(4*tl)
    tlen      = int(tN/settings['dT'])
    h1_TS.resize(tlen)
    h2_TS.resize(tlen)
                    
    delta_f = 1.0 / h1_TS.duration
    flen    = tlen//2 + 1
    psd     = aLIGOZeroDetHighPower(flen, delta_f, f0)
    m,  _   = optimized_match(h1_TS, h2_TS, psd=psd, low_frequency_cutoff=f0, high_frequency_cutoff=settings['fhigh'])
    return 1.- m

def compute_one_mismatch_22(wf1, wf2, settings, M, geom=True, plot=False):
    """
    Assumes waveforms are in geometric units if geom=True
    """
    A_prf = scale_t = 1.
    if geom:
        A_prf   = M*ut.Msun_m/(settings['distance']*ut.Mpc_m)
        scale_t = M*ut.Msuns
    f0   = settings['initial_frequency_mm']

    # Interpolate NR and EOB 22 mode on array with spacing dT & taper
    amps, phis, tt, hs = [], [], [], []
    for i,w in enumerate([wf1, wf2]): 
        t_SI  = w.u*scale_t
        A, p  = w.extract_hlm(2,2)
        if w.kind == 'EOB':
            # Apply nu rescaling to A
            q  = w.pars['q']
            nu = q/(1.+q)**2
            A *= nu
        if w.kind == 'SXS':
            # change phase sign for SXS
            p = p*(-1)

        t_new  = np.arange(t_SI[0], t_SI[-1], settings['dT'])
        amps.append(np.interp(t_new, t_SI, A_prf*A))
        phis.append(np.interp(t_new, t_SI, p))
        tt.append(t_new)

        t_mrg = t_new[np.argmax(amps[i])]
        h     = amps[i]*np.cos(phis[i])
        
        # Time Series
        if (settings['taper'] and settings['taper_kind'] == 'manual'):
            h = wfu.taper(t_new, h, M, settings['alpha'][i] , settings['tau'][i], Msuns=ut.Msuns)

        hTS   = TimeSeries(h, settings['dT'], epoch=t_mrg)

        if (settings['taper'] and settings['taper_kind'] == 'pycbc'):
                hTS = taper_timeseries(hTS,tapermethod="startend")
        hs.append(hTS)
        if plot:
            hf = hTS.to_frequencyseries()
            plt.loglog(hf.sample_frequencies, abs(hf))
        #plt.plot(hTS.sample_times-t_mrg, hTS)
    if plot: plt.show()
    
    h1_TS = hs[0]
    h2_TS = hs[1]  
    # Resize both to next power of 2
    LM        = max(len(h1_TS), len(h2_TS))
    tl        = (LM-1)*settings['dT']
    tN        = ut.nextpow2(4*tl)
    tlen      = int(tN/settings['dT'])
    h1_TS.resize(tlen)
    h2_TS.resize(tlen)
                    
    delta_f = 1.0 / h1_TS.duration
    flen    = tlen//2 + 1
    psd     = aLIGOZeroDetHighPower(flen, delta_f, f0)
    m,  _   = optimized_match(h1_TS, h2_TS, psd=psd, low_frequency_cutoff=f0, high_frequency_cutoff=settings['fhigh'])
    return 1.- m


# higher modes stuff

def snr(h, psd, low_freq, high_freq):
    hh = sigmasq(h,  psd=psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)
    return np.sqrt(hh)

def skypos_and_time_maximized_overlap(s, hp, hc, psd, low_freq, high_freq):
    """
    Maximize the overlap over sky position and time (via ifft, as usual)
    See https://arxiv.org/pdf/1603.02444.pdf
    TODO: implement some kind of time interpolation/subsampling to improve the timeshift accuracy
    """
    ss   = sigmasq(s,  psd=psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)
    hphp = sigmasq(hp, psd=psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)
    hchc = sigmasq(hc, psd=psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)
    hp  /= np.sqrt(hphp)
    hc  /= np.sqrt(hchc)

    rhop, _, nrm  = matched_filter_core(hp,s, psd = psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)
    rhop *= nrm
    rhoc, _, nrm  = matched_filter_core(hc,s, psd = psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)
    rhoc *= nrm
    
    hphccorr  = overlap_cplx(hp, hc, psd=psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)# matched_filter_core(hp,hc)
    hphccorr  = np.real(hphccorr)
    Ipc       = hphccorr

    gam   = rhop.data * np.conjugate(rhoc.data)
    gam   = np.real(gam)
    rhop2 = np.abs(rhop.data)**2
    rhoc2 = np.abs(rhoc.data)**2

    sqrt  = np.sqrt((rhop2-rhoc2)**2 + 4*(Ipc*rhop2-gam)*(Ipc*rhoc2-gam))
    num   = rhop2 - 2.*Ipc*gam + rhoc2 + sqrt
    den   = 1. - Ipc**2
    
    o = np.sqrt(max(num)/den/2.)/np.sqrt(ss)
    if (o > 1.):
         o = 1.

    return o

def snr_weighted_match(ms, snrs):
    ms   = np.array(ms) 
    snrs = np.array(snrs)
    ms3  = ms**3
    snr3 = snrs**(3.)

    num  = np.sum(ms3*snr3)
    den  = np.sum(snr3)
    snrw = (num/den)**(1./3.)
    return snrw


def higher_modes_match_k_phic(s, hlm, tlen, inc, psd, modes_k, dT=1./4096, fmin_mm=20,fmax=2048, 
                              kind='pycbc',  M=1., Msuns=1, alpha=0.1, tau=1. ):
    
    def to_minimize_dphi(x):
        hp, hc  = wfu.compute_hphc(hlm, x, inc, modes=modes_k)
        hp_TS, hc_TS = hpc_to_timeseries(hp, hc, dT)
        hp_TS, hc_TS = condition_hpc(hp_TS, hc_TS, tlen,  kind='pycbc',  M=1., Msuns=1, alpha=0.1, tau=1. )
        # To FD
        hpf = hp_TS.to_frequencyseries()
        hcf = hc_TS.to_frequencyseries()
        mis = 1.-skypos_and_time_maximized_overlap(s, hpf, hcf, psd,fmin_mm,fmax)
        return mis

    res = minimize_scalar(
                to_minimize_dphi,
                method="bounded",
                bounds=(0, 2.*np.pi),
                #options={'xatol':1e-15}
                )
    return to_minimize_dphi(res.x), res.x


if __name__ == '__main__':

    # test that the HM mismatch is ~zero for the same EOB waveform
    # vary the effective polarization, inclination and coalescence phase
    eff_pols  = [0., np.pi/4, np.pi/1.9999, 3*np.pi/4]
    iotas     = [0., np.pi/4, np.pi/2, 3*np.pi/4]
    coa_phases= [0., np.pi/4, np.pi/2, 3*np.pi/4]

    f0        = 20.
    modes     = [0, 1, 4, 8]
    print("Check that the HM mismatch is approximately zero for the same waveform")
    print(f"Using modes = {modes}")
    print("-------------------------------------------------------")

    for iota in iotas:
        print(f"iota={iota:.2f}")
        for coa_phase in coa_phases:
            print(f"\t coa_phase={coa_phase:.2f}")
            pars = wf.CreateDict(M=20, q=2, f0=20., ecc=1e-8, use_geom="no", iota=iota)
            pars['use_mode_lm']       = modes
            pars['coalescence_angle'] = np.pi/2.-coa_phase

            eob_target = wf.Waveform_EOB(pars)

            # setup target waveform
            hpt, hct = eob_target.hp, eob_target.hc
            hpt, hct = hpc_to_timeseries(hpt, hct, 1./pars['srate_interp'])
            tlen     = len(eob_target.u)
            tlen     = 2**(int(np.log2(tlen))+2)
            hpt, hct = condition_hpc(hpt, hct, tlen)
            hpf      = hpt.to_frequencyseries()
            hcf      = hct.to_frequencyseries()
        
            # psd
            psd      = aLIGOZeroDetHighPower(len(hpf), hpf.delta_f, f0)

            for k in eff_pols:
                h = np.cos(k)*hpf + np.sin(k)*hcf
                mm, phi = higher_modes_match_k_phic(
                        h,
                        eob_target.hlm,
                        tlen,
                        iota,
                        psd,
                        [str(k) for k in pars['use_mode_lm']],
                        dT=1./pars['srate_interp'],
                        fmin_mm=f0,
                        fmax=2048
                        )
                assert(mm < 1e-10)
                print(f"\t\t eff_pol={k:.2f} mm={mm:.2e}")

    print("Noice.")