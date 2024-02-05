#!/usr/bin/python
# Process GR-Athena++ wave?.txt files with Psi4 multipolar output,
# computes energetics and writes into CoRe format.

import os, sys, re
import argparse
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from scipy import integrate    
import matplotlib.pyplot as plt
import h5py

# #################################################################################

# c=G=Mo=1 to ms, km and g/cm^3
ms = 4.925490949141889e-6 * 1e3
km = 1.47662506140
gcm3 = 6.175828283964599e+17 # 1.98840990e33/(1.47662506140e5)**3

# #################################################################################

def fheader(l,m,r,mass):
    """
    Process mass and radius for file header
    """
    if(r>0):
        hstr    = "r=%e\nM=%e\n" % (r, mass)
        rstr    = '_r%05d' % (r)
        lmrstr  = 'l%d_m%d_r%05d' % (l, m, r)
    else:
        hstr    = "r=Infinity\nM=%e\n" % (mass)
        rstr    = '_rInf'
        lmrstr  = 'l%d_m%d_rInf' % (l, m)
    return hstr, rstr, lmrstr

class Multipole:
    """
    Class for gravitational-wave multipole
    """
    def __init__(self, l, m, t, data, mass, radius, path):
        self.l, self.m = l, m
        self.mass = mass
        self.radius = radius
        self.t = t
        self.u = self.retarded_time()

        self.psi = radius * data
        self.phi_p, self.amp_p, self.omg_p = self.get_pao(self.psi)
        self.h = np.array([])

        self.path = path
        
        self.psi4_extrapolated = False 

    def get_pao(self,data):
        """
        Phase, amplitude, frequency decomposition
        """
        phi = - np.unwrap(np.angle(data))
        amp = np.abs(data)
        omg = self.diff1(self.t,phi)
        return phi, amp, omg
    
    def extrapolate_psi4(self, integration, fcut=0, deg=-1, poly_int=None):
        if self.psi4_extrapolated:
            raise RuntimeError('psi4 has been already extrapolated!')
        r = self.radius
        M = self.mass
        R = r * (1 + M/(2*r))**2
        psi0 = self.psi / r * R # self.psi = radius * data, see __init__
        if integration=='FFI':
            dt    = np.diff(self.t)[0]
            f     = fftfreq(psi0.shape[0], dt)
            idx_p = np.logical_and(f >= 0, f < fcut)
            idx_m = np.logical_and(f <  0, f > -fcut)
            f[idx_p] = fcut
            f[idx_m] = -fcut
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

    def fixed_freq_int(self, fcut=0, extrap_psi4=False):
        """
        Fixed frequency double time integration
        """
        if extrap_psi4:
            self.extrapolate_psi4(integration='FFI',fcut=fcut)
        signal = self.psi
        dt = np.diff(self.t)[0]
        f = fftfreq(signal.shape[0], dt)
        idx_p = np.logical_and(f >= 0, f < fcut)
        idx_m = np.logical_and(f <  0, f > -fcut)
        f[idx_p] = fcut
        f[idx_m] = -fcut
        self.dh = ifft(-1j*fft(signal)/(2*np.pi*f))
        self.h = ifft(-fft(signal)/(2*np.pi*f)**2)
        self.phi, self.amp, self.omg = self.get_pao(self.h)
    
    def remove_time_drift(self, signal, deg=-1, poly_int=None):
        """
        Remove drift in TD integration.
        If poly_int is specified, then fit only that part of the signal. 
        """
        out = signal
        if deg>=0:
            if poly_int is None:
                t_tofit      = self.t
                signal_tofit = signal
            else:
                if poly_int[1]>self.t[-1]:
                    raise RuntimeError("Polynomial interval ends after simulation's end ({:.2f} M)".format(self.t[-1]))
                mask = np.logical_and(self.t >= poly_int[0],\
                                      self.t <= poly_int[1])
                t_tofit      = self.t[mask]
                signal_tofit = signal[mask]
            p = np.polyfit(t_tofit, signal_tofit, deg)
            out -= np.polyval(p, self.t)       
        return out

    def time_domain_int(self, deg=-1, poly_int=None, extrap_psi4=False):
        """
        Time domain integration with polynomial correction
        The polynomial is obtained fitting the whole signal if poly_int is none,
        otherwise consider only the interval specified; see remove_time_drift
        """
        if extrap_psi4:
            self.extrapolate_psi4(integration='TDI',deg=deg,poly_int=poly_int)
        
        dh0 = integrate.cumtrapz(self.psi,self.t,initial=0)
        dh  = self.remove_time_drift(dh0,deg=deg,poly_int=poly_int)
        h0  = integrate.cumtrapz(dh,self.t,initial=0)
        h   = self.remove_time_drift(h0,deg=deg,poly_int=poly_int)
        
        self.dh = dh
        self.h  = h 
        self.phi, self.amp, self.omg = self.get_pao(self.h)
        
    def retarded_time(self):
        M = self.mass
        r = self.radius
        if r <= 0.:
            return self.t
        R = r * (1 + M/(2*r))**2
        rstar = R + 2*M*np.log(R/(2*M) - 1)
        return self.t - rstar

    def diff1(self, xp, yp, pad=True):
        """
        Computes the first derivative of y(x) using centered 2nd order
        accurate finite-differencing
        """
        dyp = [(yp[i+1] - yp[i-1])/(xp[i+1] - xp[i-1]) \
               for i in range(1, xp.shape[0]-1)]
        dyp = np.array(dyp)
        if pad==True:
            dyp = np.insert(dyp, 0, dyp[0])
            dyp = np.append(dyp, dyp[-1])
        return dyp

    def to_file(self, var=['all']):
        """ 
        Writes waveform data in CoRe format 
        Writes merger info from (2,2) mode
        """
        os.makedirs(self.path, exist_ok=True)
        r = self.radius
        M = self.mass
        l = self.l
        m = self.m
        headstr, rstr, lmrstr = fheader(l,m,r,M)
        if 'all' or 'Psi4' in var:
            header = headstr
            header += "u/M:0 RePsi4/M:1 ImPsi4/M:2 Momega:3 A/M:4 phi:5 t:6"
            data = np.c_[self.u/M, self.psi.real/M, self.psi.imag/M,
                         M*self.omg_p, self.amp_p/M, self.phi_p, self.t]
            np.savetxt(os.path.join(self.path,'Rpsi4_'+lmrstr+'.txt'),
                       data, header=header)
        if ('all' or 'h' in var) and self.h.size != 0:
            header = headstr
            header += "u/M:0 ReRh/M:1 ImRh/M:2 Momega:3 A/M:4 phi:5 t:6"
            data = np.c_[self.u/M, self.h.real/M, self.h.imag/M,
                         M*self.omg, self.amp/M, self.phi, self.t]
            np.savetxt(os.path.join(self.path,'Rh_'+lmrstr+'.txt'),
                       data, header=header)
            if l==2 and m==2:
                imrg = np.argmax(self.amp)
                tmrg = self.t[imrg]
                umrg = self.u[imrg]
                header = headstr
                header += '1:imrg 2:tmrg 3:umrg 4:umrg(ms)'
                np.savetxt(os.path.join(self.path,'merger'+rstr+'.txt'),
                           np.c_[imrg, tmrg, umrg, umrg*ms],
                           header = header)

# Various multipolar coefficients (mc) needed below
def mc_f(l,m):
    return np.sqrt(l*(l+1) - m*(m+1))
def mc_a(l,m):
    return np.sqrt((l-m)*(l+m+1))/(l*(l+1))
def mc_b(l,m):
    return np.sqrt(((l-2)*(l+2)*(l+m)*(l+m-1))/((2*l-1)*(2*l+1)))/(2*l)
def mc_c(l,m):
    return 2*m/(l*(l+1))
def mc_d(l,m):
    return np.sqrt(((l-2)*(l+2)*(l-m)*(l+m))/((2*l-1)*(2*l+1)))/l

def waveform2energetics(h, doth, t, u, lmmodes,
                        mnegative=False,
                        to_file=None, attrs ={}):
    """
    Compute GW energy and angular momentum from multipolar waveform
    See e.g. https://arxiv.org/abs/0912.1285

    * h[(l,m)]     : multipolar strain 
    * doth[(l,m)]  : time-derivative of multipolar strain
    * t            : time array
    * modes        : (l,m) indexes
    * mnegative    : if True, account for the factor 2 due to m<0 modes 
    """    
    dt = np.diff(t)[0]
    oo16pi  = 1./(16*np.pi)

    lmodes = [lm[0] for lm in lmmodes]
    mmodes = [lm[1] for lm in lmmodes]
    lmax = max(lmodes)
    lmin = min(lmodes)
    if lmin < 2:
        raise ValueError("l>2")
    if lmin != 2:
        print("Warning: lmin > 2")
        
    mnfactor = np.ones_like(mmodes)
    if mnegative:
        mnfactor = [1 if m == 0 else 2 for m in mmodes]
    else:
        if all(m >= 0 for m in mmodes):
            print("Warning: m>=0 but not accouting for it!")

    dotE, E = {}, {}
    dotJ, J = {}, {}
    dotJz, dotJy, dotJx = {}, {}, {}
    Jz, Jy, Jx = {}, {}, {}
    dotP, P = {}, {}
    dotPz, dotPy, dotPx = {}, {}, {}
    Pz, Py, Px = {}, {}, {}

    dotE_all, E_all = np.zeros_like(t), np.zeros_like(t)
    dotJ_all, J_all = np.zeros_like(t), np.zeros_like(t)
    dotJz_all, dotJy_all, dotJx_all = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    Jz_all, Jy_all, Jx_all = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    dotP_all, P_all = np.zeros_like(t), np.zeros_like(t)
    dotPz_all, dotPy_all, dotPx_all = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    Pz_all, Py_all, Px_all = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    
    for k, (l,m) in enumerate(lmmodes):

        fact = mnfactor[k] * oo16pi
        
        # Energy
        dotE[(l,m)] = fact * np.abs(doth[(l,m)])**2 

        # Angular momentum
        dotJz[(l,m)] = fact * m * np.imag(h[(l,m)] * np.conj(doth[(l,m)]))

        dothlm_1 = doth[(l,m-1)] if (l,m-1) in doth else 0*h[(l,m)]
        dothlm1  = doth[(l,m+1)] if (l,m+1) in doth else 0*h[(l,m)]
        
        dotJy[(l,m)] = 0.5 * fact * \
            np.real( h[(l,m)] * (mc_f(l,m) * np.conj(dothlm1) - mc_f(l,-m) * np.conj(dothlm_1) ))
        dotJx[(l,m)] = 0.5 * fact * \
            np.real( h[(l,m)] * (mc_f(l,m) * np.conj(dothlm1) + mc_f(l,-m) * np.conj(dothlm_1) ))

        dotJ[(l,m)] = np.sqrt(dotJx[(l,m)]**2 + dotJy[(l,m)]**2 + dotJz[(l,m)]**2)

        # Linear momentum
        dothlm1   = doth[(l,m+1)]   if (l,m+1)   in doth else 0*h[(l,m)]
        dothl_1m1 = doth[(l-1,m+1)] if (l-1,m+1) in doth else 0*h[(l,m)]
        dothl1m1  = doth[(l+1,m+1)] if (l+1,m+1) in doth else 0*h[(l,m)]
        dotl_1m   = doth[(l-1,m)]   if (l-1,m)   in doth else 0*h[(l,m)]
        dothl1m   = doth[(l+1,m)]   if (l+1,m)   in doth else 0*h[(l,m)]
        
        dotPxiy = 2.0 * fact * doth[(l,m)] * \
            (mc_a(l,m) * np.conj(dothlm1) + mc_b(l,-m) * np.conj(dothl_1m1) - mc_b(l+1,m+1) * np.conj(dothl1m1))
        dotPy[(l,m)] = np.imag(dotPxiy)
        dotPx[(l,m)] = np.real(dotPxiy)
        dotPz[(l,m)] = fact * np.imag( doth[(l,m)] * \
            (mc_c(l,m) * np.conj(doth[(l,m)]) + mc_d(l,m) * np.conj(dotl_1m) + mc_d(l+1,m) * np.conj(dothl1m)) )

        dotP[(l,m)] = np.sqrt(dotPx[(l,m)]**2 + dotPy[(l,m)]**2 + dotPz[(l,m)]**2)

        # Sum up
        E[(l,m)]  = integrate.cumtrapz(dotE[(l,m)],t,initial=0)        

        Jz[(l,m)] = integrate.cumtrapz(dotJz[(l,m)],t,initial=0)
        Jy[(l,m)] = integrate.cumtrapz(dotJy[(l,m)],t,initial=0)
        Jx[(l,m)] = integrate.cumtrapz(dotJx[(l,m)],t,initial=0)
        J[(l,m)]  = integrate.cumtrapz(dotJ[(l,m)],t,initial=0)

        Pz[(l,m)] = integrate.cumtrapz(dotPz[(l,m)],t,initial=0)
        Py[(l,m)] = integrate.cumtrapz(dotPy[(l,m)],t,initial=0)
        Px[(l,m)] = integrate.cumtrapz(dotPx[(l,m)],t,initial=0)
        P[(l,m)]  = integrate.cumtrapz(dotP[(l,m)],t,initial=0)

        dotE_all  += dotE[(l,m)]
        dotJz_all += dotJz[(l,m)]
        dotJz_all += dotJy[(l,m)]
        dotJz_all += dotJx[(l,m)]
        dotJ_all  += dotJ[(l,m)]
        dotPz_all += dotPz[(l,m)]
        dotPy_all += dotPy[(l,m)]
        dotPx_all += dotPx[(l,m)]
        dotP_all  += dotP[(l,m)]
        
        E_all  += E[(l,m)]
        Jz_all += Jz[(l,m)]
        Jx_all += Jx[(l,m)]
        Jy_all += Jy[(l,m)]
        J_all  += J[(l,m)]
        Pz_all += Pz[(l,m)]
        Px_all += Px[(l,m)]
        Py_all += Py[(l,m)]
        P_all  += P[(l,m)]

    if to_file:
        
        data = {}
        data['t'] = t
        data['u'] = u[(2,2)]
        data['lmmodes'] = lmmodes
        data['mnegative'] = mnegative
        
        # Integrated data
        data['E_all'], data['dotE_all'] = E_all, dotE_all
        data['Jz_all'], data['dotJz_all'] = Jz_all, dotJz_all
        data['Jy_all'], data['dotJy_all'] = Jy_all, dotJy_all
        data['Jx_all'], data['dotJx_all'] = Jx_all, dotJx_all
        data['J_all'], data['dotJ_all'] = J_all, dotJ_all
        data['Pz_all'], data['dotPz_all'] = Pz_all, dotPz_all
        data['Py_all'], data['dotPy_all'] = Py_all, dotPy_all
        data['Px_all'], data['dotPx_all'] = Px_all, dotPx_all
        data['P_all'], data['dotP_all'] = P_all, dotP_all

        # Multipolar data
        mdata = {}
        mdata['h'], mdata['doth'] = h, doth
        mdata['E'], mdata['dotE'] = E, dotE
        mdata['J'], mdata['dotJ'] = J, dotJ
        mdata['P'], mdata['dotP'] = P, dotP
        mdata['Jz'], mdata['Jy'], mdata['Jx'] = Jz, Jy, Jx
        mdata['Pz'], mdata['Py'], mdata['Px'] = Pz, Py, Px
        mdata['dotJz'], mdata['dotJy'], mdata['dotJx'] = dotJz, dotJy, dotJx
        mdata['dotPz'], mdata['dotPy'], mdata['dotPx'] = dotPz, dotPy, dotPx
        
        with h5py.File(to_file, 'w') as f:
            for a in attrs.keys():
                f.attrs[a] = attrs[a]
            for k in data.keys():
                dset = f.create_dataset(k, data=data[k])            
            for (l,m) in lmmodes:
                gm = f.create_group("Multipole/l{}m{}".format(l,m));
                for k in mdata.keys():
                    dset = gm.create_dataset(k, data=mdata[k][(l,m)])                
    
    return E_all, dotE_all, Jz_all, dotJz_all, Jy_all, dotJy_all, Jx_all, dotJx_all, J_all, dotJ_all

# #################################################################################
def processwave(**kwargs):
    
    if 'is_argparse' in kwargs and kwargs['is_argparse']:
        args = argparse.Namespace(**kwargs)
    
    else:
        # handle input 
        default_dict                = {}
        default_dict['ifiles']      = []
        default_dict['outdir']      = 'waveforms' 
        default_dict['mass']        = 1. 
        default_dict['radii']       = [1.]
        default_dict['strain']      = 'No'
        default_dict['frequency']   = -1
        default_dict['degree']      = 4
        default_dict['poly_int']    = None
        default_dict['extrap_psi4'] = False
        default_dict['energetics']  = False
        default_dict['mratio']      = -1.
        default_dict['madm']        = -1.
        default_dict['jadm']        = -1.
        default_dict['lmmax']       = -1.
        default_dict['mneg']        = False
        default_dict['no_output']   = False
        default_dict['silent']      = False 

        input_dict = kwargs
        if len(input_dict['ifiles'])<1:
            raise RuntimeError('Input file(s) not specified (key: ifiles)!')
        for key in input_dict.keys():
            if key not in default_dict.keys():
                print(f"+++ Warning +++ : unknown key '{key}'! Ignoring this key")
        # update input dict using default values when not specified otherwise
        for key in default_dict.keys():
            if key not in input_dict:
                input_dict[key] = default_dict[key]
            elif isinstance(default_dict[key],list) and not isinstance(input_dict[key],list):
                input_dict[key] = [input_dict[key]] # transform single items in list if needed
        
        if input_dict['strain'] not in ['FFI', 'TDI','No']:
            raise ValueError(f"Unknown value for 'strain': {args.strain}")

        # convert to namespace
        args = argparse.Namespace(**input_dict)
    
    # checks on the input
    if args.strain == 'FFI' and args.frequency <= 0:
        raise ValueError("Need to specify the initial frequency")

    if args.energetics and (args.madm <= 0. or
                            args.jadm <= 0. or
                            args.mratio <= 0.):
        raise ValueError("Need to specify the ADM values and the mass ratio")
    
    write_output = not args.no_output

    output_data = []
    for n,fname in enumerate(args.ifiles):
        output_data_dict = {}
        output_data_dict['fname'] = fname
        
        if not args.silent:
            print('{}: '.format(fname))
        
        # Read the wave?.txt file & Extract the modes
        with open(fname) as f:
            header = f.readline().replace('#','')
        # Waveform data start at cstart, after 'iter' and 'time' cols,
        # NB re.split('\d+:',header) = 
        # [' ', 'iter ', 'time ', 'l=2 m=-2 R ', ' l=2 m=-2 I ', ' l=2 m=-1 R ', ...
        cstart = 2
        header = re.split('\d+:',header)[cstart+1:] 
        
        # Load the data & Make sure no double entries
        data = np.loadtxt(fname)
        t, uniq = np.unique(data[:,1], axis=0, return_index=True)
        data = data[uniq, :]
        
        # get lmax from the number of columns
        ncols_data   = len(data[0,:])
        lmax_in_data = np.sqrt(4+(ncols_data-2)/2)-1 # in each row there are 2+2*nmodes columns
        if abs(int(lmax_in_data)-lmax_in_data)>1e-14:
            raise RuntimeError('Something wrong when computing lmax from number of cols')
        else:
            lmax_in_data = int(lmax_in_data)
        if lmax_in_data<args.lmmax:
            raise ValueError(f'Using lmmax={args.lmmax} but only modes up to l={lmax_in_data} are available')
        
        # Use up to last available input radius, and keep using the last
        rad = args.radii[n if n<len(args.radii) else -1]

        # Collect the multipolar waveform on the way
        h, dh, psi4, u, phi, omg, amp = {}, {}, {}, {}, {}, {}, {}
        lmmodes     = []
        col_indices = {}

        # Get list of modes and col-indices
        if len(header)+2==ncols_data: # if all columns are listed in the header
            for c in range(0,len(header),2):

                # Get the mode indexes
                col = header[c].strip()
                
                col_tmp = re.match('l=(\d+) m=([+-]?\d+) (\w)',col) # 'l=2 m=-2 R'
                if col_tmp is None:
                    col_tmp = re.match('l=(\d+)-m=([+-]?\d+)-(\w)',col) # 'l=2-m=-2-R'
                col = col_tmp
                 
                if col.group(3) != 'R':
                    raise ValueError('Something wrong with column {}'.format(cstart+c))
                l, m = int(col.group(1)), int(col.group(2))

                if l > args.lmmax:
                    # NB this assumes (l,m) are sorted in ascending order!
                    break
                if args.mneg and m<0:
                    continue
                
                lmmodes.append((l,m))
                col_indices[(l,m)] = (cstart+c,cstart+c+1)
        else:
            # this assumes lmin=2
            c = 0
            for l in range(2, args.lmmax+1):
                for m in range(-l, l+1):
                    lmmodes.append((l,m))
                    col_indices[(l,m)] = (cstart+c, cstart+c+1)
                    if args.mneg and m<0:
                        continue
                    c += 2
        
        # Now, Process and write multipolar data for each pair of columns
            # Multipole data & strain
            #psi4_tmp = data[:,cstart+c] + 1j*data[:,cstart+c+1]
        for lmmode in lmmodes:
            l = lmmode[0]
            m = lmmode[1]
            col_idx_re = col_indices[(l,m)][0]
            col_idx_im = col_indices[(l,m)][1]
            if not args.silent:
                 print(' * load l={} m={} col={} {}'.format(l,m,col_idx_re,col_idx_im))
            psi4_tmp = data[:,col_idx_re] + 1j*data[:,col_idx_im]
            mode = Multipole(l, m, t, psi4_tmp, args.mass, rad, args.outdir)
            psi4[(l,m)] = psi4_tmp*rad

            if args.strain == 'FFI':
                mode.fixed_freq_int(fcut=2*args.frequency/max(1,abs(m)),extrap_psi4=args.extrap_psi4)
            if args.strain == 'TDI':
                mode.time_domain_int(deg=args.degree,extrap_psi4=args.extrap_psi4,poly_int=args.poly_int) 

            # Write multipole in CoRe format
            if write_output:
                mode.to_file()
            
            # Store this mode for later
            h[(l,m)], dh[(l,m)], u[(l,m)] = mode.h, mode.dh, mode.u 
            phi[(l,m)], omg[(l,m)], amp[(l,m)] = mode.phi, mode.omg, mode.amp 
        
        output_data_dict['lmmodes'] = lmmodes
        output_data_dict['u']       = u
        output_data_dict['t']       = t
        output_data_dict['psi4']    = psi4
        output_data_dict['dh']      = dh
        output_data_dict['h']       = h
        output_data_dict['phi']     = phi
        output_data_dict['omg']     = omg
        output_data_dict['amp']     = amp

        # Compute the energetics for this input file / radius    
        energetics_dict = {}
        if args.energetics:
        
            # Symmetric mass ratio
            q = args.mratio
            if q < 1.0: q = 1.0/q
            nu = q/((1.+q)**2)

            # Output file header and h5 attributes
            header, rstr, _ = fheader(0,0,rad,args.mass)
            attrs = {}
            attrs['mass'] = args.mass
            attrs['radius'] = rad
            attrs['mass_ratio'] = q
            attrs['symmetric_mass_ratio'] = nu
            attrs['ADM_mass'] = args.madm
            attrs['ADM_angular_momentum'] = args.jadm

            # Energetics and multipoles
            if write_output:
                to_file = os.path.join(mode.path,'data'+rstr+'.h5')
            else:
                to_file = None
            e,edot,jz,jzdot,_,_,_,_,j,jdot = \
                waveform2energetics(h, dh, t, u, 
                                    lmmodes, args.mneg,
                                    to_file = to_file,
                                    attrs = attrs)

            # Energetics file in CoRe Format
            eb = ((args.madm-e)/args.mass-1.0)/nu
            jorb = (args.jadm-j)/(args.mass**2*nu)

            header += "J_orb:0 E_b:1 u/M:2 E_rad:3 J_rad:4 t:5"
            data = np.c_[jorb, eb, u[(2,2)]/args.mass, e, j, t]
            if write_output:
                np.savetxt(os.path.join(mode.path,'EJ'+rstr+'.txt'),
                           data, header=header)
            
            energetics_dict['E']     = e
            energetics_dict['Edot']  = edot
            energetics_dict['Jz']    = jz
            energetics_dict['Jzdot'] = jzdot
        
        output_data_dict['energetics'] = energetics_dict
        
        return output_data_dict


# #################################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process GR-Athena++ wave?.txt files into CoRe format.')
    parser.add_argument("-i", "--input", nargs='+', required=True, dest='ifiles', metavar="FILE",
                        help="Input files")
    parser.add_argument("-o", "--outdir", default='waveforms',
                        help="Output folder")
    parser.add_argument("-m", "--mass", type=float, default=1.,
                        help="Source gravitational mass")
    parser.add_argument("-r", "--radii", nargs='+', type=float, default=[1.],
                        help="Extraction radii (uses as many as provided, negative values are assumed as Inf)")
    parser.add_argument("-s", "--strain", default='No', const='FFI', nargs='?',
                        choices=['FFI', 'TDI','No'],
                        help="Calculate strain mode (default: %(default)s)")
    parser.add_argument("-f", "--frequency", type=float, default=-1,
                        help="Initial frequency for FFI cut")
    parser.add_argument("-d", "--degree", type=int, default=4,
                        help="Polynomial degree for TDI correction")
    parser.add_argument("--poly_int", type=float, nargs='+', default=None, 
                        help="Interval on which fit the polynomial in TDI")
    parser.add_argument("--extrap_psi4", action='store_true',
                        help="Extrapolate psi4 before integration")
    parser.add_argument("-e", "--energetics", action='store_true',
                        help="Compute the energetics from multipolar waveform")
    parser.add_argument("-q", "--mratio", type=float, default=-1.,
                        help="Source mass ratio")
    parser.add_argument("-a", "--madm", type=float, default=-1.,
                        help="ADM mass")
    parser.add_argument("-j", "--jadm", type=float, default=-1.,
                        help="ADM angular momentum")
    parser.add_argument("-k", "--lmmax", type=int, default=-1,
                        help="Maximum multipole m=l to consider.\
                        This assumes columns are sorted in ascending order of (l,m)")
    parser.add_argument("-n", "--mneg", action='store_true',
                        help="Skip negative mode m<0")
    parser.add_argument("--no_output", action='store_true',
                        help="Avoid writing output files")
    parser.add_argument("--silent", action='store_true', 
                        help="Do not print on the stdout")
        
    args_shell = parser.parse_args()
    
    # This works also with is_argparse==False if args_shell is from parser.parse_rgs,
    # use is_argparse==True just to avoid useless operations 
    processwave(**vars(args_shell), is_argparse=True) 

# #################################################################################

