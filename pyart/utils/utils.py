import numpy as np; from scipy import interpolate
from math import factorial as fact
from math import ceil

Msuns  = 4.925491025543575903411922162094833998e-6
Mpc_m  = 3.085677581491367278913937957796471611e22
Msun_m = 1.476625061404649406193430731479084713e3

def delta_a_b(a, xa, b, xb, N=500):
    x_max = min(max(xa),max(xb))
    x_min = max(min(xa),min(xb))
    x     = np.linspace(x_min, x_max,N)
    
    fa =interpolate.interp1d(xa,a)
    fb =interpolate.interp1d(xb,b)

    a_n = fa(x)
    b_n = fb(x)
    delta_ab = a_n - b_n 
    return x, delta_ab, a_n, b_n

def nextpow2(x):
    """
    Return the next closest power of 2
    """
    return pow(2, ceil(np.log(x)/np.log(2)))

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def D02(xp, yp, pad=True):
    """
    Computes the first derivative of y(x) using centered 2nd order
    accurate finite-differencing

    This function returns an array of yp.shape[0]-2 elements

    NOTE: the data needs not to be equally spaced
    """
    dyp = [(yp[i+1] - yp[i-1])/(xp[i+1] - xp[i-1]) \
            for i in range(1, xp.shape[0]-1)]
    dyp = np.array(dyp)
    if pad:
        dyp = np.insert(dyp, 0, dyp[0])
        dyp = np.append(dyp, dyp[-1])
    return dyp

def D04(f, dx):
    """
    4th order centered stencil for first derivative
    """
    n  = len(f)
    f  = np.array(f)
    df = np.zeros_like(f)
    oo12dx  = 1./(12*dx);
    
    i = np.arange(2,n-2)
    df[i] = (8.*(f[i+1]-f[i-1]) - f[i+2] + f[i-2])*oo12dx

    i = 0;
    df[i] = (-25.*f[i] + 48.*f[i+1] - 36.*f[i+2] + 16.*f[i+3] - 3.*f[i+4])*oo12dx
    i = 1;
    df[i] = (-3.*f[i-1] - 10.*f[i] + 18.*f[i+1] - 6.*f[i+2] + f[i+3])*oo12dx
    i = n-2;
    df[i] = - (-3.*f[i+1] - 10.*f[i] + 18.*f[i-1] - 6.*f[i-2] + f[i-3])*oo12dx
    i = n-1;
    df[i] = - (-25.*f[i] + 48.*f[i-1] - 36.*f[i-2] + 16.*f[i-3] - 3.*f[i-4])*oo12dx
    return df

def integrate(ff):
    """
    Computes the anti-derivative of a discrete function using a
    2nd order accurate formula
    """
    out = np.empty_like(ff)
    out[0] = 0.0
    out[1:] = np.cumsum(0.5*(ff[:-1] + ff[1:]))
    return out

def powspace(start, stop, power, num):
    """
    Compute equally spaced grid in log-space
    """
    start = np.power(start, 1./float(power))
    stop  = np.power(stop,  1./float(power))
    return np.power( np.linspace(start, stop, num=num), power) 

def spinsphericalharm(s, l, m, phi, i):
    """
    Compute spin-weighted spherical harmonics
    """
    c = pow(-1.,-s) * np.sqrt( (2.*l+1.)/(4.*np.pi) )
    dWigner = c * wigner_d_function(l,m,-s,i)
    rY = np.cos(m*phi) * dWigner
    iY = np.sin(m*phi) * dWigner
    return rY + 1j*iY

def wigner_d_function(l,m,s,i):
    """
    Compute wigner d functions, following Ref
    TODO: add reference
    """
    costheta = np.cos(i*0.5)
    sintheta = np.sin(i*0.5)
    norm = np.sqrt( (fact(l+m) * fact(l-m) * fact(l+s) * fact(l-s)) )
    ki = max( 0  , m-s )
    kf = min( l+m, l-s )
    dWig = 0.
    for k in range(int(ki), int(kf)+1):
        div = 1.0/( fact(k) * fact(l+m-k) * fact(l-s-k) * fact(s-m+k) )
        dWig = dWig+div*( pow(-1.,k) * pow(costheta,2*l+m-s-2*k) * pow(sintheta,2*k+s-m) )
    return (norm * dWig)

# function to add to JSON
def write_json(new_data, filename='nr_ecc_sims.json'):
    import json
    """
    Append to log.json
    """
    with open(filename,'r') as file:
        file_data = json.load(file)
        file.close()
    with open(filename,'w') as file:
        file_data["SXS"].append(new_data)
        file.seek(0)
        json.dump(file_data, file, indent = 4)
    return 0
    # function above useful for things like the following:
    # for id in eob_ics.keys():
    #     e0, omg0, f0_mm = eob_ics[id]
    #     alpha_nr, tau_nr = LookupTaper(id, "NR")
    #     alpha_eob, tau_eob = LookupTaper(id, "EOB")
    #     data = {id : {'eob_ecc': e0, 'eob_omg': omg0, 'f0_mm': f0_mm, 
    #             'alpha_nr': alpha_nr, 'tau_nr': tau_nr, 
    #             'alpha_eob': alpha_eob, 'tau_eob': tau_eob
    #             }}
    #     write_json(data)