#!/usr/bin/python
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from scipy import integrate    

from ..utils.utils import D1, safe_sigmoid 

class Multipole:
    """
    Class for gravitational-wave multipole
    """
    def __init__(self, l, m, t, data, mass, radius, integrand='psi4'):
        self.l         = l
        self.m         = m
        self.mass      = mass
        self.radius    = radius
        self.integrand = integrand
        self.t = t
        self.u = self.retarded_time()

        if self.integrand=='psi4':
            self.psi = radius * data
            self.dh  = np.array([])
        elif self.integrand=='news':
            self.psi = np.array([]) 
            self.dh  = radius * data
        else:
            raise RuntimeError(f"Unknown integrand: {self.integrand}, use 'psi4' or 'news'")
        self.h = np.array([])
        
        self.psi4_extrapolated = False 
        self.window_applied    = False
    
    def areal_radius(self, r, M):
        return r * (1 + M/(2*r))**2

    def retarded_time(self):
        M = self.mass
        r = self.radius
        if r <= 0.:
            return self.t
        R = self.areal_radius(r=r, M=M)
        rstar = R + 2*M*np.log(R/(2*M) - 1)
        return self.t - rstar

    def extrapolate_psi4(self, integration, fcut=0, deg=-1, poly_int=None):
        if self.psi4_extrapolated:
            raise RuntimeError('psi4 has been already extrapolated!')
        r = self.radius
        M = self.mass
        R = self.areal_radius(r=r,M=M)
        psi0 = self.psi / r * R # self.psi = radius * data, see __init__
            
        if integration=='FFI':
            f   = self.freq_interval(self.psi,fcut=fcut)
            dh0 = ifft(-1j*fft(psi0)/(2*np.pi*f))
        elif integration=='TDI':
            dh_tmp = integrate.cumtrapz(psi0,self.t,initial=0)
            dh0    = self.remove_time_drift(dh_tmp, deg=deg, poly_int=poly_int)
        else:
            raise RuntimeError(f'Unknown integration option: {integration}')
        l = self.l
        A = 1 - 2*M/R
        self.psi = A*(psi0 - (l-1)*(l+2)*dh0/(2*R))
        self.psi4_extrapolated = True
        return
    
    def apply_window(self, window=[10,-10], walpha=3):
        if self.window_applied:
            raise RuntimeError('Time-window already applied!')
        
        if window is None:
            return 
        
        if self.integrand=='psi4':
            signal = self.psi
        elif self.integrand=='news':
            signal = self.dh

        walpha   = 3
        towindow = 50.0
        t      = self.t

        # apply window
        clip_val = np.log(1e+20)
        #FIXME : this if-statement is error-prone, to improve
        if window[0]>=0 and window[1]<=0:
            w_t1   = window[0]
            w_t2   = t[-1] + window[1]
            sig1    = safe_sigmoid(t-w_t1, alpha=walpha, clip=clip_val)
            sig2    = safe_sigmoid(w_t2-t, alpha=walpha, clip=clip_val)
            signal *= sig1
            signal *= sig2
        elif window[1]>window[0]:
            sig = safe_sigmoid(window[0]-t, walpha=walpha, clip=clip_val) + \
                  safe_sigmoid(t-window[1], walpha=walpha, clip=clip_val)
            signal *= sig 
        else:
            raise RuntimeError('Invalid window option:: [{:f} {:f}]'.format(*window))

        # set signal equal to zero below threshold
#        threshold = 1e-14
#        amp   = np.abs(signal)
#        bool1 = amp>threshold
#        if any(bool1):
#            idx1  = np.where(bool1)[0][0]
#            signal[:idx1+1] = 0*signal[:idx1+1]
#        bool2 = np.logical_and(amp<threshold, t>t[idx1+10])
#        if any(bool2):
#            idx2  = np.where(bool2)[0][0]
#            signal[idx2:] = 0*signal[idx2:]
        
        if self.integrand=='psi4':
            self.psi = signal
        elif self.integrand=='news':
            self.dh = signal
        return

    def freq_interval(self, signal, fcut=0):
        dt = np.diff(self.t)[0]
        f  = fftfreq(signal.shape[0], dt)
        idx_p = np.logical_and(f >= 0, f < fcut)
        idx_m = np.logical_and(f <  0, f > -fcut)
        f[idx_p] =  fcut
        f[idx_m] = -fcut
        return f

    def fixed_freq_int(self, fcut=0, extrap_psi4=False, window=None):
        """
        Fixed frequency double time integration
        """
        if window is not None:
            self.apply_window(window=window)
        if extrap_psi4:
            self.extrapolate_psi4(integration='FFI',fcut=fcut)
        
        if self.integrand=='psi4':
            signal  = self.psi
            f       = self.freq_interval(signal,fcut=fcut)
            self.dh = ifft(-1j*fft(signal)/(2*np.pi*f))
            self.h  = ifft(-fft(signal)/(2*np.pi*f)**2)
        else:
            signal  = self.dh 
            f       = self.freq_interval(signal,fcut=fcut)
            self.h  = ifft(-1j*fft(signal)/(2*np.pi*f))
        return

    def remove_time_drift(self, signal, deg=-1, poly_int=None):
        """
        Remove drift in TD integration using a fit.
        If poly_int is specified, then fit only that part of the signal. 
        """
        out = signal
        if deg>=0:
            if poly_int is None:
                t_tofit      = self.t
                signal_tofit = signal
            else:
                if poly_int[1]>self.t[-1]:
                    raise RuntimeError("Polynomial interval ends after simulation's end (t : [{:.2f}, {:.2f}] M)".format(self.t[0], self.t[-1]))
                mask = np.logical_and(self.t >= poly_int[0],\
                                      self.t <= poly_int[1])
                t_tofit      = self.t[mask]
                signal_tofit = signal[mask]
            p = np.polyfit(t_tofit, signal_tofit, deg)
            out -= np.polyval(p, self.t)       
        return out

    def time_domain_int(self, deg=-1, poly_int=None, extrap_psi4=False, window=None):
        """
        Time domain integration with polynomial correction
        The polynomial is obtained fitting the whole signal if poly_int is none,
        otherwise consider only the interval specified; see remove_time_drift
        """
        if window is not None:
            self.apply_window(window=window)
        if extrap_psi4:
            self.extrapolate_psi4(integration='TDI',deg=deg,poly_int=poly_int)
        
        if self.integrand=='psi4':
            dh0 = integrate.cumtrapz(self.psi,self.t,initial=0)
            dh  = self.remove_time_drift(dh0,deg=deg,poly_int=poly_int)
        else:
            dh  = self.dh 

        h0  = integrate.cumtrapz(dh,self.t,initial=0)
        h   = self.remove_time_drift(h0,deg=deg,poly_int=poly_int)
        
        self.dh = dh
        self.h  = h 
        return

    def integrate_wave(self, integr_opts={}):
        """ 
        Integrate according to specified methods
        """
        method      = integr_opts['method']      if 'method'      in integr_opts else 'FFI'
        f0          = integr_opts['f0']          if 'f0'          in integr_opts else 0.007
        deg         = integr_opts['deg']         if 'deg'         in integr_opts else 0 
        poly_int    = integr_opts['poly_int']    if 'poly_int'    in integr_opts else None
        extrap_psi4 = integr_opts['extrap_psi4'] if 'extrap_psi4' in integr_opts else False
        window      = integr_opts['window']      if 'window'      in integr_opts else None
        
        # integrate psi4 and get self.dh and self.h
        if method=='FFI':
            self.fixed_freq_int(fcut=2*f0/max(1,abs(self.m)), extrap_psi4=extrap_psi4, window=window)
        elif method=='TDI':
            self.time_domain_int(deg=deg, poly_int=poly_int, extrap_psi4=extrap_psi4, window=window) 
        else:
            raise RuntimeError('Unknown method: {:s}'.format(integration['method']))
        
        if self.integrand=='news':
            self.psi = D1(self.dh,self.t,4)

        return self.h, self.dh



