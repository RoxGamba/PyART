import re, copy
import numpy as np
from scipy import interpolate
from scipy.signal import find_peaks
from scipy.signal.windows import tukey
from math import factorial as fact
from math import ceil
import matplotlib.pyplot as plt
from astropy.constants import G, c, M_sun

Msun = G.value * M_sun.value / (c.value**3)  # Solar mass
## Misc


def rotate3_axis(vector, theta=0.0, axis=[0, 0, 1]):
    """
    Rotate a 3 vector around a provided axis of an angle theta

    Parameters
    ----------
    vector: array-like
        vector to rotate
    theta: float
        rotation angle (in radians)
    axis: array-like
        rotation axis
    Returns
    -------
    out: array-like
        rotated vector
    """
    from scipy.spatial.transform import Rotation

    zaxis = np.array(axis)
    r = Rotation.from_rotvec(theta * zaxis)
    vector_r = r.apply(vector)
    return vector_r


# Rotate a 3 vector using Euler angles
def rotate3(vector, alpha, beta, gamma, invert=False):
    """
    Rotate a 3 vector using Euler angles under conventions defined at:
    https://en.wikipedia.org/wiki/Euler_angles
    https://en.wikipedia.org/wiki/Rotation_matrix

    Science reference: https://arxiv.org/pdf/1110.2965.pdf (Appendix)

    Specifically, the Z1,Y2,Z3 ordering is used: https://wikimedia.org/api/rest_v1/media/math/render/svg/547e522037de6467d948ecf3f7409975fe849d07

    *  alpha represents a rotation around the z axis
    *  beta represents a rotation around the x' axis
    *  gamma represents a rotation around the z'' axis

    NOTE that in order to perform the inverse rotation, it is *not* enough to input different rotation angles. One must use the invert=True keyword.
    This takes the same angle inputs as the forward rotation, but correctly applies the transposed rotation matricies in the reversed order.

    spxll'18
    """

    # Import usefuls
    from numpy import cos, sin, array, dot, ndarray, vstack

    # Hangle angles as arrays
    angles_are_arrays = (
        isinstance(alpha, np.ndarray)
        and isinstance(beta, np.ndarray)
        and isinstance(gamma, np.ndarray)
    )
    if angles_are_arrays:
        # Check for consistent array shapes
        if not (alpha.shape == beta.shape == gamma.shape):
            # Let the people know and halt
            raise ValueError("input angles as arrays must have identical array shapes")

    # Validate input(s)
    if isinstance(vector, (list, tuple, ndarray)):
        vector = array(vector)
    else:
        raise ValueError(
            "first input must be iterable compatible 3D vector; please check"
        )

    # Rotation around z''
    Ra = array([[cos(alpha), -sin(alpha), 0], [sin(alpha), cos(alpha), 0], [0, 0, 1]])

    # Rotation around y
    Rb = array([[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]])

    # Rotation around z
    Rg = array([[cos(gamma), -sin(gamma), 0], [sin(gamma), cos(gamma), 0], [0, 0, 1]])

    # Perform the rotation
    # ans = (  Ra * ( Rb * ( Rg * vector ) )  )
    # NOTE that this is the same convention of equation A9 of Boyle et al : https://arxiv.org/pdf/1110.2965.pdf
    R = dot(Ra, dot(Rb, Rg))
    if invert:
        R = R.T
    ans = dot(R, vector)

    # If angles are arrays, then format the input such that rows in ans correspond to rows in alpha, beta and gamma
    if angles_are_arrays:
        ans = vstack(ans).T

    return ans


def reflect_unwrap(vec):
    """
    Reflect points in an array
    """

    ans = np.array(vec)
    for k in range(len(vec)):
        if (k > 0) and ((k + 1) < len(vec)):
            l = vec[k - 1]
            c = vec[k]
            r = vec[k + 1]
            reflect = (np.sign(l) == np.sign(r)) and (np.sign(l) == -np.sign(c))
            if reflect:
                ans[k] *= -1
    return ans


def minmax_array(x, tol=1e-20):
    """
    Find the maximum and minimum of an array

    Parameters
    ----------
    x: array-like
        input array
    tol: float
        tolerance to add/subtract to min/max
    Returns
    -------
    out: list
        [min(x)-tol, max(x)+tol]
    """
    return [min(x) - tol, max(x) + tol]


def nextpow2(x):
    """
    Return the next closest power of 2

    Parameters
    ----------
    x: float
        input number

    Returns
    -------
    out: int
        next closest power of 2
    """
    return pow(2, ceil(np.log(x) / np.log(2)))


def safe_sigmoid(x, alpha, clip=None):
    """
    Sigmoid function with clips on the exponent

    Parameters
    ----------
    x: array-like
        input array
    alpha: float
        slope parameter
    clip: float or None
        if not None, clip the exponent to [-clip, clip]
    Returns
    -------
    out: array-like
        sigmoid values
    """
    if clip is None:
        exponent = -alpha * x
    elif isinstance(clip, (int, float)):
        exponent = np.clip(-alpha * x, -clip, clip)
    else:
        raise ValueError(f"Invalid clip value: {clip}")
    return 1 / (1 + np.exp(exponent))


def taper_waveform(
    t, h, t1=None, t2=None, alpha=0.2, alpha_end=None, kind="sigmoid", debug=False
):
    """
    Waveform tapering in Matcher-class.
    The meaning of t1, t2, and alpha depends on the kind used!

    Parameters
    ----------
    t: array-like
        time array
    h: array-like
        strain array
    t1: float or None
        start time of the tapering
    t2: float or None
        end time of the tapering (if None, tapering is only applied at t1)
    alpha: float
        slope parameter for the tapering
    alpha_end: float or None
        slope parameter for the tapering at t2 (if None, alpha_end=alpha)
    kind: str
        tapering method, can be 'sigmoid' or 'tukey' (default: 'sigmoid')
    debug: bool
        if True, plot the tapering window
    Returns
    -------
    out: array-like
        tapered strain
    """
    if kind is None:
        return h

    elif kind == "sigmoid":
        if alpha_end is None:
            alpha_end = alpha
        window = np.ones_like(h)
        if t1 is not None:
            window *= safe_sigmoid(t - t1, alpha=alpha, clip=50)
        if t2 is not None:
            window *= safe_sigmoid(t2 - t, alpha=alpha_end, clip=50)

    elif kind == "tukey":
        n = len(h)
        idx1 = np.where(t > t1)[0][0]
        if t2 is not None:
            idx2 = np.where(t > t2)[0][0]
            window = tukey(idx2 - idx1, alpha)
            window = np.pad(window, (0, n - idx2), mode="constant")
        else:
            window = tukey(n - idx1, alpha)
            nby2 = int(n / 2)
            window[nby2:] = np.ones_like(window[nby2:])

        window = np.pad(window, (idx1, 0), mode="constant")
    else:
        raise ValueError("Unknown tapering method")

    out = h * window
    if debug:
        import matplotlib.pyplot as plt

        plt.figure
        plt.plot(t, h)
        plt.plot(t, out)
        plt.plot(t, window)
        if t1 is not None:
            plt.axvline(t1)
        if t2 is not None:
            plt.axvline(t2)
        plt.show()
    return out


def windowing(h, alpha=0.1):
    """
    Windowing with Tukey window on a given strain (time-domain)
    h     : strain to be tapered
    alpha : Tukey filter slope parameter. Suggested value: alpha = 1/4/seglen

    Parameters
    ----------
    h: array-like
        strain array
    alpha: float
        slope parameter for the tapering
    Returns
    -------
    out: (array-like, float)
        tapered strain and wfact = <w^2>
    """
    window = tukey(len(h), alpha)
    wfact = np.mean(window**2)
    return h * window, wfact


def fft(h, dt):
    """
    FFT of a time-domain signal h(t)
    Parameters
    ----------
    h: ndarray
        time-domain signal
    dt: float
        time spacing (assumed uniform)
    Returns
    -------
    out: (f, hfft)
        frequency array and FFT of h
    """
    N = len(h)
    hfft = np.fft.rfft(h) * dt
    f = np.fft.rfftfreq(N, d=dt)
    return f, hfft


def ifft(u, srate, seglen, t0=0.0):
    """
    IFFT of a frequency-domain signal u(f)
    Parameters
    ----------
    u: ndarray
        frequency-domain signal
    srate: float
        sampling rate
    seglen: float
        segment length
    t0: float
        time shift

    Returns
    -------
    out: (t, s)
        time array and IFFT of u
    """
    N = int(srate * seglen)
    s = np.fft.irfft(u, N)
    dt = 1.0 / srate
    seglen = N * dt
    t = np.arange(N) * dt + t0 - seglen / 2.0
    return t, s * srate


def find_nearest(array, value):
    """
    Find the index of the element in array closest to value
    Parameters
    ----------
    array: array-like
        input array
    value: float
        target value
    Returns
    -------
    out: int
        index of the element in array closest to value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def zero_pad_before(array, N, return_column=True):
    """
    Pad an array with N zeroes (at the beginning)

    Parameters
    ----------
    array: array-like
        input array
    N: int
        length of the output array
    return_column: bool
        if True, return a column vector, otherwise a row vector
    Returns
    -------
    out: array-like
        zero-padded array
    """
    n = len(array)
    v = np.zeros((N,))
    for i in range(n):
        v[N - n + i] = array[i]
    if not return_column:
        v = np.transpose(v)
    return v


def powspace(start, stop, power, num):
    """
    Compute equally spaced grid in log-space

    Parameters
    ----------
    start: float
        start value
    stop: float
        stop value
    power: float
        power
    num: int
        number of points
    Returns
    -------
    out: array-like
        array of num points between start and stop in log-space
    """
    start = np.power(start, 1.0 / float(power))
    stop = np.power(stop, 1.0 / float(power))
    return np.power(np.linspace(start, stop, num=num), power)


def delta_a_b(a, xa, b, xb, N=500):
    """
    Given two sets (xa, a=f(xa)) and (xb, b=g(xb)),
    interpolate them to a common grid x and compute their
    difference f(x)-g(x)

    Parameters
    ----------
    a: array-like
        first array
    xa: array-like
        x values for the first array
    b: array-like
        second array
    xb: array-like
        x values for the second array
    N: int
        number of points in the common grid
    Returns
    -------
    out: (x, delta_ab, a_n, b_n)
        x: common grid
        delta_ab: a_n - b_n
        a_n: interpolated a on the common grid
        b_n: interpolated b on the common grid
    """
    x_max = min(max(xa), max(xb))
    x_min = max(min(xa), min(xb))
    x = np.linspace(x_min, x_max, N)

    fa = interpolate.interp1d(xa, a)
    fb = interpolate.interp1d(xb, b)

    a_n = fa(x)
    b_n = fb(x)
    delta_ab = a_n - b_n
    return x, delta_ab, a_n, b_n


def vec_differences(
    x1,
    y1,
    x2,
    y2,
    a,
    b,
    dx,
    diff_kind="abs",
    fabs=False,
    interp_kind="cubic",
    ensure_unique=False,
):
    """
    Compute differences between arrays with different lengths
    Similar to delta_a_b, but with more options

    Parameters
    ----------
    x1: array-like
        x values for the first array
    y1: array-like
        first array
    x2: array-like
        x values for the second array
    y2: array-like
        second array
    a: float
        start value for the common grid
    b: float
        stop value for the common grid
    dx: float
        spacing for the common grid
    diff_kind: str
        kind of difference, can be 'abs' (absolute), 'rel' (relative), or 'phi' (phase)
    fabs: bool
        if True, return the absolute value of the difference
    interp_kind: str
        kind of interpolation, passed to spline function
    ensure_unique: bool
        if True, ensure that x1 and x2 are unique before interpolation
    Returns
    -------
    out: (x, dy)
        x: common grid
        dy: differences on the common grid
    """
    xs = np.linspace(a, b, num=int((b - a) / dx))

    y1s = spline(x1, y1, xs, kind=interp_kind, ensure_unique=ensure_unique)
    y2s = spline(x2, y2, xs, kind=interp_kind, ensure_unique=ensure_unique)
    if diff_kind == "abs":
        dys = y1s - y2s
    elif diff_kind == "rel":
        dys = (y1s - y2s) / y1s
    elif diff_kind == "phi":
        dys = y1s - y2s
        maxd = max(dys)
        n = round(maxd / (2 * np.pi))
        dys = dys - 2 * np.pi * n
    if fabs:
        dys = np.abs(dys)
    return xs, dys


def spline(x, y, xs, kind="cubic", ensure_unique=False):
    """
    Compute the spline of y(x), return y(xs)
    Parameters
    ----------
    x: array-like
        x values
    y: array-like
        y values
    xs: array-like
        x values for the output
    kind: str
        kind of interpolation, passed to interp1d
    ensure_unique: bool
        if True, ensure that x is unique before interpolation
    Returns
    -------
    out: array-like
        interpolated values y(xs)
    """
    if ensure_unique:
        x, unique_indices = np.unique(x, return_index=True)
        y = y[unique_indices]
    f = interpolate.interp1d(x, y, kind=kind, fill_value="extrapolate")
    return f(xs)


def spline_diff(t, y, k=3, n=1):
    """
    Wrapper for InterpolatedUnivariateSpline derivative function
    Parameters
    ----------
    t: array-like
        x values
    y: array-like
        y values
    k: int
        spline order
    n: int
        number of derivatives

    Returns
    -------
    out: array-like
        n-th derivative of y at t
    """
    #
    from numpy import sum
    from scipy.interpolate import InterpolatedUnivariateSpline as spline

    # Calculate the desired number of derivatives
    ans = spline(t, y.real, k=k).derivative(n=n)(t) + (
        1j * spline(t, y.imag, k=k).derivative(n=n)(t) if (sum(abs(y.imag)) != 0) else 0
    )

    return ans


#
def spline_antidiff(t, y, k=3, n=1):
    """
    Wrapper for InterpolatedUnivariateSpline antiderivative function

    Parameters
    ----------
    t: array-like
        x values
    y: array-like
        y values
    k: int
        spline order
    n: int
        number of derivatives

    Returns
    -------
    out: array-like
        n-th anti-derivative of y at t
    """

    #
    from scipy.interpolate import InterpolatedUnivariateSpline as spline

    # Calculate the desired number of integrals
    ans = spline(t, y.real, k=k).antiderivative(n=n)(t) + (
        1j * spline(t, y.imag, k=k).antiderivative(n=n)(t)
        if isinstance(y[0], complex)
        else 0
    )

    # Return the answer
    return ans


def polyfit_svd(x, y, order):
    """
    Polynomial fit using SVD
    Parameters
    ----------
    x: array-like
        x values
    y: array-like
        y values
    order: int
        polynomial order
    Returns
    -------
    out: array-like
        polynomial coefficients
    """

    X = np.vander(x, order + 1)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # threshold = np.finfo(float).eps * max(X.shape) * S[0]
    threshold = 1e-13 * np.amax(S)
    mask = S > threshold
    S_inv = np.zeros_like(S)
    S_inv[mask] = 1 / S[mask]
    coeffs = Vt.T @ np.diag(S_inv) @ U.T @ y
    return coeffs


def upoly_fits(
    r0,
    y0,
    nmin=1,
    nmax=10,
    n_extract=None,
    r_cutoff_low=None,
    r_cutoff_high=None,
    direction="in",
    svd=True,
):
    """
    Perform polynomial fits in 1/r to the data (r0, y0)
    Parameters
    ----------
    r0: array-like
        radial coordinate
    y0: array-like
        function values at r0
    nmin: int
        minimum polynomial order
    nmax: int
        maximum polynomial order
    n_extract: int or None
        if not None, extract the extrapolated value from this fit order
    r_cutoff_low: float or None
        if not None, only use data with r0 >= r_cutoff_low
    r_cutoff_high: float or None
        if not None, only use data with r0 <= r_cutoff_high
    direction: str
        'in' or 'out', whether the data is from an inward or outward
        moving observer
    svd: bool
        if True, use SVD for the polynomial fits
    Returns
    -------
    out: dict
        dictionary with the following keys
        'extrap': extrapolated value (mean over fit orders or from n_extract)
        'extrap_vec': extrapolated values for each fit order
        'fit_orders': list of fit orders
        'coeffs': polynomial coefficients for each fit order
        'polynomials': polynomial values at the input r0 for each fit order
        'mask': boolean mask used to select the data
    """

    if n_extract is None:
        n_extract = nmax  # safe if SVD is used
    if r_cutoff_low is None:
        r_cutoff_low = min(r0)
    if r_cutoff_high is None:
        r_cutoff_high = max(r0)
    if nmin > nmax:
        raise ValueError(f"nmin>nmax: {nmin}>{nmax} !")

    i_rmin = np.argmin(r0)
    if direction == "in":
        index_condition = np.arange(len(r0)) < i_rmin
    elif direction == "out":
        index_condition = np.arange(len(r0)) > i_rmin
    else:
        raise ValueError(f"Unknown direction: {direction}! Use 'in' or 'out'")
    interval_condition = np.logical_and(r0 >= r_cutoff_low, r0 <= r_cutoff_high)
    mask = np.logical_and(index_condition, interval_condition)

    r = r0[mask]
    y = y0[mask]
    u = 1 / r

    nfits = nmax - nmin + 1
    p = np.empty((len(u), nfits))
    b = np.empty((nmax + 1, nfits))
    ye_vec = np.empty((nfits,))
    fit_orders = np.empty((nfits,))
    for i in range(0, nfits):
        fit_order = nmin + i
        fit_orders[i] = fit_order
        if svd:
            b_tmp = polyfit_svd(u, y, fit_order)
        else:
            b_tmp = np.polyfit(u, y, fit_order)
        b[:, i] = zero_pad_before(b_tmp, nmax + 1)
        p[:, i] = np.polyval(b_tmp, u)
        ye_vec[i] = b[-1, i]
        if n_extract is not None and fit_order == n_extract:
            ye = ye_vec[i]
    if n_extract is None:
        ye = np.mean(ye_vec)
    out = {
        "extrap": ye,
        "extrap_vec": ye_vec,
        "fit_orders": fit_orders.astype(int).tolist(),
        "coeffs": b,
        "polynomials": p,
        "mask": mask,
    }
    return out


# Deprecated
# def find_Amax(t, Alm, height=0.15, return_idx=False):
#     peaks, _ = find_peaks(Alm, height=height)
#     i_mrg = peaks[-1]
#     if return_idx:
#         return t[i_mrg], Alm[i_mrg], i_mrg
#     else:
#         return t[i_mrg], Alm[i_mrg]


## Derivatives and integration
def D02(xp, yp, pad=True):
    """
    Computes the first derivative of y(x) using centered 2nd order
    accurate finite-differencing

    This function returns an array of yp.shape[0]-2 elements

    NOTE: the data needs not to be equally spaced

    Parameters
    ----------
    xp: array-like
        x values
    yp: array-like
        y values
    pad: bool
        if True, pad the output with the first and last value to
        return an array of the same length as the input

    Returns
    -------
    out: array-like
        first derivative of y with respect to x
    """
    dyp = [
        (yp[i + 1] - yp[i - 1]) / (xp[i + 1] - xp[i - 1])
        for i in range(1, xp.shape[0] - 1)
    ]
    dyp = np.array(dyp)
    if pad:
        dyp = np.insert(dyp, 0, dyp[0])
        dyp = np.append(dyp, dyp[-1])
    return dyp


def D1(f, x, order):
    """
    Computes the first derivative of function f(x)

    Parameters
    ----------
    f : float (list/numpy array of)
       uniformly sampled function
    x : float (list/numpy array of)
       uniformly sampled function domain or grid spacing
    order : int
       finite differencing order

    Returns
    -------
    df : list (or numpy array)
       finite differences at given order
    """

    df = np.zeros_like(f)

    Nmin = 0
    Nmax = len(f) - 1

    if len(x) == 1:
        oodx = 1.0 / x
    else:
        oodx = 1.0 / (x[1] - x[0])

    if order == 1:
        i = np.arange(Nmin, Nmax)
        df[i] = (f[i + 1] - f[i]) * oodx

        f[Nmax] = f[Nmax - 1]
    elif order == 2:
        i = np.arange(Nmin + 1, Nmax)
        df[i] = 0.5 * (f[i + 1] - f[i - 1]) * oodx

        df[Nmin] = -0.5 * (3 * f[Nmin] - 4 * f[Nmin + 1] + f[Nmin + 2]) * oodx
        df[Nmax] = 0.5 * (3 * f[Nmax] - 4 * f[Nmax - 1] + f[Nmax - 2]) * oodx
    elif order == 4:

        i = np.arange(Nmin + 2, Nmax - 1)
        oodx12 = oodx / 12.0
        df[i] = (8 * f[i + 1] - f[i + 2] - 8 * f[i - 1] + f[i - 2]) * oodx12

        df[Nmin] = (
            -25 * f[Nmin]
            + 48 * f[Nmin + 1]
            - 36 * f[Nmin + 2]
            + 16 * f[Nmin + 3]
            - 3 * f[Nmin + 4]
        ) * oodx12
        df[Nmin + 1] = (
            -25 * f[Nmin + 1]
            + 48 * f[Nmin + 2]
            - 36 * f[Nmin + 3]
            + 16 * f[Nmin + 4]
            - 3 * f[Nmin + 5]
        ) * oodx12
        df[Nmax] = (
            3 * f[Nmax - 4]
            - 16 * f[Nmax - 3]
            + 36 * f[Nmax - 2]
            - 48 * f[Nmax - 1]
            + 25 * f[Nmax]
        ) * oodx12
        df[Nmax - 1] = (
            3 * f[Nmax - 5]
            - 16 * f[Nmax - 4]
            + 36 * f[Nmax - 3]
            - 48 * f[Nmax - 2]
            + 25 * f[Nmax - 1]
        ) * oodx12
    else:
        raise NotImplementedError("Supported order are 1,2,4")
    return df


def integrate(ff):
    """
    Computes the anti-derivative of a discrete function using a
    2nd order accurate formula

    Parameters
    ----------
    ff: array-like
        input function values
    Returns
    -------
    out: array-like
        anti-derivative of ff
    """
    out = np.empty_like(ff)
    out[0] = 0.0
    out[1:] = np.cumsum(0.5 * (ff[:-1] + ff[1:]))
    return out


## Special functions


def spinsphericalharm(s, l, m, phi, i):
    """
    Compute spin-weighted spherical harmonics

    Parameters
    ----------
    s, l, m: int
        spin, l and m indices
    phi: float
        azimuthal angle
    i: float
        inclination angle
    Returns
    -------
    out: complex
        value of the spin-weighted spherical harmonic
    """
    c = pow(-1.0, -s) * np.sqrt((2.0 * l + 1.0) / (4.0 * np.pi))
    dWigner = c * wigner_d_function(l, m, -s, i)
    rY = np.cos(m * phi) * dWigner
    iY = np.sin(m * phi) * dWigner
    return rY + 1j * iY


# Small wigner d matrices
def wigner_d_function(l, m, s, i):
    """
    Compute wigner d functions, following Ref
    TODO: add reference

    Parameters
    ----------
    l, m, s: int
        l, m and s indices
    i: float
        inclination angle
    Returns
    -------
    out: float
        value of the wigner d function
    """
    costheta = np.cos(i * 0.5)
    sintheta = np.sin(i * 0.5)
    norm = np.sqrt((fact(l + m) * fact(l - m) * fact(l + s) * fact(l - s)))
    ki = max(0, m - s)
    kf = min(l + m, l - s)
    dWig = 0.0
    for k in range(int(ki), int(kf) + 1):
        div = 1.0 / (fact(k) * fact(l + m - k) * fact(l - s - k) * fact(s - m + k))
        dWig = dWig + div * (
            pow(-1.0, k)
            * pow(costheta, 2 * l + m - s - 2 * k)
            * pow(sintheta, 2 * k + s - m)
        )
    return norm * dWig


# Calculate Widger D-Matrix Element
def wdelement(
    ll,  # polar index (eigenvalue) of multipole to be rotated (set of m's for single ll )
    mp,  # member of {all em for |em|<=l} -- potential projection spaceof m
    mm,  # member of {all em for |em|<=l} -- the starting space of m
    alpha,  # -.
    beta,  #  |- Euler angles for rotation
    gamma,
):  # -'

    # ** James Healy 6/18/2012
    # ** wignerDelement
    # *  calculates an element of the wignerD matrix
    # Modified by llondon6 in 2012 and 2014
    # Converted to python by spxll 2016
    #
    # This implementation apparently uses the formula given in:
    # https://en.wikipedia.org/wiki/Wigner_D-matrix
    #
    # Specifically, this the formula located here:
    # https://wikimedia.org/api/rest_v1/media/math/render/svg/53fd7befce1972763f7f53f5bcf4dd158c324b55

    #
    if (
        (type(alpha) is np.ndarray)
        and (type(beta) is np.ndarray)
        and (type(gamma) is np.ndarray)
    ):
        alpha, beta, gamma = (
            alpha.astype(float),
            beta.astype(float),
            gamma.astype(float),
        )
    else:
        alpha, beta, gamma = float(alpha), float(beta), float(gamma)

    coefficient = np.sqrt(
        fact(ll + mp) * fact(ll - mp) * fact(ll + mm) * fact(ll - mm)
    ) * np.exp(1j * (mp * alpha + mm * gamma))

    total = 0
    # find smin
    if (mm - mp) >= 0:
        smin = mm - mp
    else:
        smin = 0
    # find smax
    if (ll + mm) > (ll - mp):
        smax = ll - mp
    else:
        smax = ll + mm

    if smin <= smax:
        for ss in range(smin, smax + 1):
            A = (-1) ** (mp - mm + ss)
            A *= np.cos(beta / 2) ** (2 * ll + mm - mp - 2 * ss) * np.sin(beta / 2) ** (
                mp - mm + 2 * ss
            )
            B = fact(ll + mm - ss) * fact(ss) * fact(mp - mm + ss) * fact(ll - mp - ss)
            total += A / B

    element = coefficient * total
    return element


## Plot utils
def save_plot(figname, show=True, save=False, verbose=False):
    """
    Save or show a plot
    Parameters
    ----------
    figname: str
        name of the figure file
    show: bool
        if True, show the plot
    save: bool
        if True, save the plot
    verbose: bool
        if True, print messages
    """
    if save:
        plt.savefig(figname, dpi=200, bbox_inches="tight")
        if verbose:
            print(f"figure saved: {figname}")
    if show:
        # plt.show()
        plt.draw()
        plt.pause(0.0001)
    else:
        plt.close()
    return


def retarded_time(t, r, M=1):
    """
    Compute the retarded time in Schwarzschild spacetime
    Parameters
    ----------
    t: float
        coordinate time
    r: float
        areal radius
    M: float
        mass of the black hole
    Returns
    -------
    out: float
        retarded time
    """
    # FIXME would remove from here!
    R = r * (1 + M / (2 * r)) ** 2
    rstar = R + 2 * M * np.log(R / (2 * M) - 1)
    return t - rstar


def are_dictionaries_equal(
    dict1_in, dict2_in, excluded_keys=[], float_tol=1e-14, verbose=False
):
    """
    Check if two dictionaries are equal,
    modulo excluded keys

    Parameters
    ----------
    dict1_in: dict
        first dictionary
    dict2_in: dict
        second dictionary
    excluded_keys: list
        list of keys to exclude from the comparison
    float_tol: float
        tolerance for float comparison
    verbose: bool
        if True, print messages
    Returns
    -------
    out: bool
        True if the dictionaries are equal, False otherwise
    """
    dict1 = copy.deepcopy(dict1_in)
    dict2 = copy.deepcopy(dict2_in)
    # remove excluded keys
    for ke in excluded_keys:
        if ke in dict1:
            del dict1[ke]
        if ke in dict2:
            del dict2[ke]

    setk1 = set(dict1.keys())
    setk2 = set(dict2.keys())
    # setke = set(excluded_keys)
    # if setk1-setke != setk2-setke:
    if setk1 != setk2:
        if verbose:
            print("+++ Different number of keys +++")
        return False
    for key in setk1:
        val1 = dict1[key]
        val2 = dict2[key]
        if isinstance(val1, float) or isinstance(val2, float):
            kbool = abs(val1 - val2) < float_tol
        else:
            kbool = val1 == val2
        if not kbool:
            if verbose:
                print(f"+++ Issues with key: {key} +++")
            return False
    return True


def print_dict_comparison(
    dict1_in, dict2_in, excluded_keys=[], dict1_name="dict1", dict2_name="dict2"
):
    """
    Print a comparison between two dictionaries,
    modulo excluded keys
    Parameters
    ----------
    dict1_in: dict
        first dictionary
    dict2_in: dict
        second dictionary
    excluded_keys: list
        list of keys to exclude from the comparison
    dict1_name: str
        name of the first dictionary
    dict2_name: str
        name of the second dictionary
    Returns
    -------
    out: None
    """
    dict1 = copy.deepcopy(dict1_in)
    dict2 = copy.deepcopy(dict2_in)
    # remove excluded keys
    for ke in excluded_keys:
        if ke in dict1:
            del dict1[ke]
        if ke in dict2:
            del dict2[ke]

    def list_to_str(mylist):
        mystr = "("
        for elem in mylist:
            if isinstance(elem, (list, tuple)):
                return "too deep: list of list of list"
            mystr += str(elem) + ", "
        mystr += ")"
        return mystr.replace(", )", ")")

    for key, value1 in dict1.items():
        value2 = dict2[key]
        if isinstance(value1, dict):
            dbool = are_dictionaries_equal(value1, value2, verbose=True)
            if dbool:
                print(f">> issues with {key:16s} (dictionary)")
            else:
                print(f">> {key:16s} is dict:")
        else:
            if value1 is None:
                value1 = "None"
            if value2 is None:
                value2 = "None"
            if isinstance(value1, list) or isinstance(value2, list):
                n1 = len(value1)
                n2 = len(value2)
                print(f">> {key:22s} is list:")
                for i in range(max(n1, n2)):
                    elem1 = value1[i] if i < n1 else " "
                    elem2 = value2[i] if i < n2 else " "
                    if isinstance(elem1, (list, tuple)):
                        elem1 = list_to_str(elem1)
                    if isinstance(elem2, (list, tuple)):
                        elem2 = list_to_str(elem2)
                    print(
                        " " * 26
                        + f"elem n.{i:d} ---> {dict1_name:10s}: {elem1:<10}   {dict2_name:10s}: {elem2:<10}"
                    )
            else:
                print(
                    f">> {key:22s} - {dict1_name:10s}: {value1:<22}   {dict2_name:10s}: {value2:<22}"
                )
    pass


def safe_loadtxt(fname, remove_nans=True, remove_overlaps=True, time_idx=0):
    """
    Load a text file and optionally remove rows with NaNs
    and rows with overlapping time values
    Parameters
    ----------
    fname: str
        name of the file to load
    remove_nans: bool
        if True, remove rows with NaNs
    remove_overlaps: bool
        if True, remove rows with overlapping time values
    time_idx: int
        index of the time column
    Returns
    -------
    out: array-like
        loaded data
    """
    X = np.loadtxt(fname)
    if remove_nans:
        ncols = len(X[0, :])
        X = X[~np.isnan(X).any(axis=1)]
    if remove_overlaps:
        t = X[:, time_idx]
        mask = t == np.maximum.accumulate(t)
        X = X[mask]
        # ensure unicity
        _, unique_indices = np.unique(X[:, time_idx], return_index=True)
        X = X[unique_indices]
    return X


def extract_value_from_str(string, key):
    """
    Extract a numerical value from a string given a key
    Parameters
    ----------
    string: str
        input string
    key: str
        key to search for
    Returns
    -------
    out: float or int or None
        extracted value, or None if not found
    """
    pattern = rf"{key}(\d+(\.\d+|p\d+)?|\d+)"
    match = re.search(pattern, string)
    if match:
        value = match.group(1)
        value = value.replace("p", ".")
        if "." in value:
            return float(value)
        else:
            return int(value)
    return None
