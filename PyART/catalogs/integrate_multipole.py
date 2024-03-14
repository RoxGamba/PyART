#!/usr/bin/python

import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from scipy import integrate    

class Multipole:
    """
    Class for gravitational-wave multipole
    """
    def __init__(self, l, m, t, data, mass, radius):
        self.l, self.m = l, m
        self.mass   = mass
        self.radius = radius
        self.t = t
        self.u = self.retarded_time()

        self.psi = radius * data
        self.dh  = np.array([])
        self.h   = np.array([])
        
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
            f   = self.freq_interval(fcut=fcut)
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

        walpha   = 3
        towindow = 50.0
        t      = self.t

        # apply window
        w_t1   = window[0]
        w_t2   = t[-1] + window[1]
        sig1   = 1/(1 + np.exp(-walpha*(t-w_t1)) ) 
        sig2   = 1/(1 + np.exp(-walpha*(w_t2-t)) ) 
        self.psi *= sig1
        self.psi *= sig2
        
        # set signal equal to zero below threshold
        threshold = 1e-14
        amp   = np.abs(self.psi)
        bool1 = amp>threshold
        if any(bool1):
            idx1  = np.where(bool1)[0][0]
            self.psi[:idx1+1] = 0*self.psi[:idx1+1]
        bool2 = np.logical_and(amp<threshold, t>t[idx1+10])
        if any(bool2):
            idx2  = np.where(bool2)[0][0]
            self.psi[idx2:]   = 0*self.psi[idx2:]
        self.window_applied = True
        return

    def freq_interval(self, fcut=0, signal=None):
        if signal is None: signal = self.psi
        dt = np.diff(self.t)[0]
        f  = fftfreq(signal.shape[0], dt)
        idx_p = np.logical_and(f >= 0, f < fcut)
        idx_m = np.logical_and(f <  0, f > -fcut)
        f[idx_p] =  fcut
        f[idx_m] = -fcut
        return f

    def fixed_freq_int(self, fcut=0, extrap_psi4=False, window=None, dh=None):
        """
        Fixed frequency double time integration
        """
        if window is not None:
            self.apply_window(window=window)
        if extrap_psi4:
            self.extrapolate_psi4(integration='FFI',fcut=fcut)
        
        f        = self.freq_interval(fcut=fcut)
        if dh is None:
            signal   = self.psi
            self.dh  = ifft(-1j*fft(signal)/(2*np.pi*f))
            self.h   = ifft(-fft(signal)/(2*np.pi*f)**2)
        else:
            self.dh = dh 
            self.h  = ifft(-1j*fft(dh)/(2*np.pi*f))
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

    def time_domain_int(self, deg=-1, poly_int=None, extrap_psi4=False, window=None, dh=None):
        """
        Time domain integration with polynomial correction
        The polynomial is obtained fitting the whole signal if poly_int is none,
        otherwise consider only the interval specified; see remove_time_drift
        """
        if window is not None:
            self.apply_window(window=window)
        if extrap_psi4:
            self.extrapolate_psi4(integration='TDI',deg=deg,poly_int=poly_int)
        
        if dh in None:
            dh0 = integrate.cumtrapz(self.psi,self.t,initial=0)
            dh  = self.remove_time_drift(dh0,deg=deg,poly_int=poly_int)

        h0  = integrate.cumtrapz(dh,self.t,initial=0)
        h   = self.remove_time_drift(h0,deg=deg,poly_int=poly_int)
        
        self.dh = dh
        self.h  = h 
        return

    def integrate_psi4(self, integr_opts={}):
        """ 
        Integrate psi4 according to specified methods
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
        return self.h, self.dh



