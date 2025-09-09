"""
Module to fit rotated/translated hyperbola and extract the angle between asymptotes.

This module provides functions to fit a generic quadratic form to a set of points,
convert the quadratic coefficients to canonical hyperbola parameters, and visualize
the fitted hyperbola. It also includes utilities for hyperbola parametrization and
branch extraction.

Functions
---------
fit_quadratic(x, y)
    Fit a generic quadratic form to the given x, y data points.

quadratic_to_canonical(ABCDF)
    Convert quadratic coefficients to canonical hyperbola parameters.

hyp_parametrization(th, canonical)
    Parametrize the hyperbola using canonical parameters and angle array.

hyp_branches(canonical)
    Generate the two branches of the hyperbola from canonical parameters.

plot_hypfit(x, y, canonical_fit, rlim=10, swap_ab=False)
    Plot the input data points and the fitted hyperbola branches.

Example
-------
Run this module as a script to generate synthetic hyperbola data, fit the hyperbola,
and compare the fitted parameters to the original ones. Usage:

    python3 hypfit.py seed th_min th_max npoints scale_noise

If no arguments are provided, default values are used.

Notes
-----
- The fitting assumes a generic quadratic form and extracts canonical parameters
  (semi-axes, rotation, center).
- Visualization includes both branches of the hyperbola and optionally swapped axes.
- The module prints a comparison between input and fitted parameters for validation.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys


def fit_quadratic(x, y):
    """
    Fits a generic quadratic form to the given x and y data points.
    The quadratic form is defined as:
        Q(x, y) = a*x^2 + b*2*x*y + c*y^2 + d*2*x + e*2*y
    Parameters
    ----------
    x : array_like
        1D array of x-coordinates of the data points.
    y : array_like
        1D array of y-coordinates of the data points.
    Returns
    -------
    coeffs : ndarray
        Solution vector [a, b, c, d, e] representing the coefficients of the quadratic form.
    """

    # Fit generic quadratic form
    x = np.array(x)
    y = np.array(y)
    xmtr = np.vstack([x * x, 2 * x * y, y * y, 2 * x, 2 * y])
    xvec = xmtr.sum(axis=1)
    ymtr = np.dot(xmtr, xmtr.transpose())
    return np.linalg.solve(ymtr, xvec)


def quadratic_to_canonical(ABCDF):
    """
    Converts quadratic curve coefficients to canonical hyperbola parameters.
    Parameters
    ----------
    ABCDF : array-like, shape (5,)
        Coefficients [A, B, C, D, F] of the quadratic equation:
        A*x^2 + 2*B*x*y + C*y^2 + 2*D*x + 2*F*y = 1
    Returns
    -------
    params : list
        List containing canonical ellipse parameters:
        [a, b, ph, x0, y0], where
        a : float
            Semi-major axis length.
        b : float
            Semi-minor axis length.
        ph : float
            Rotation angle of the hyperbola (radians).
        x0 : float
            x-coordinate of the hyperbola center.
        y0 : float
            y-coordinate of the hyperbola center.
    """

    # Compute canonical coefficients from quadratic ones
    A = ABCDF[0]
    B = ABCDF[1]
    C = ABCDF[2]
    D = ABCDF[3]
    F = ABCDF[4]
    delta = A * C - B * B
    Delta = np.linalg.det([[A, B, D], [B, C, F], [D, F, -1]])
    lambda1 = 0.5 * (A + C + np.sqrt((A + C) ** 2 - 4 * delta))
    lambda2 = 0.5 * (A + C - np.sqrt((A + C) ** 2 - 4 * delta))
    a = np.sqrt(np.abs(Delta / lambda1 / delta))  # x^2/a^2 + y^2/b^2 = 1
    b = np.sqrt(np.abs(Delta / lambda2 / delta))  # x^2/a^2 + y^2/b^2 = 1
    ph = 0.5 * np.arctan(2 * B / (A - C))  # rotation angle
    x0 = -(C * D - B * F) / delta  # x-component of the center
    y0 = -(A * F - B * D) / delta  # y-component of the center
    return [a, b, ph, x0, y0]


def hyp_parametrization(th, canonical):
    """
    Parametrize a hyperbola given angle and canonical parameters.
    Parameters
    ----------
    th : float or array-like
        Angle(s) at which to evaluate the hyperbola.
    canonical : array-like
        Canonical parameters of the hyperbola [a, b, ph, x0, y0]:
            a : float
                Semi-major axis length.
            b : float
                Semi-minor axis length.
            ph : float
                Rotation angle (radians).
            x0 : float
                X-coordinate of the center.
            y0 : float
                Y-coordinate of the center.
    Returns
    -------
    x : float or ndarray
        X-coordinate(s) of the hyperbola at angle(s) th.
    y : float or ndarray
        Y-coordinate(s) of the hyperbola at angle(s) th.
    """

    a = canonical[0]
    b = canonical[1]
    ph = canonical[2]
    x0 = canonical[3]
    y0 = canonical[4]
    x = a * np.cos(ph) / np.cos(th) - b * np.sin(ph) * np.tan(th) + x0
    y = a * np.sin(ph) / np.cos(th) + b * np.cos(ph) * np.tan(th) + y0
    return x, y


def hyp_branches(canonical):
    """
    Computes the two branches of a hyperbola given its canonical parameters.
    Parameters
    ----------
    canonical : array-like
        Canonical parameters of the hyperbola.
    Returns
    -------
    x1, y1 : ndarray
        Coordinates of the first branch of the hyperbola.
    x2, y2 : ndarray
        Coordinates of the second branch of the hyperbola.
    """

    eps = 1e-4
    th1 = np.linspace(-np.pi / 2 + eps, np.pi / 2 - eps, num=1000)
    th2 = np.linspace(+np.pi / 2 + eps, 3 * np.pi / 2 - eps, num=1000)
    x1, y1 = hyp_parametrization(th1, canonical)
    x2, y2 = hyp_parametrization(th2, canonical)
    return x1, y1, x2, y2


def plot_hypfit(x, y, canonical_fit, rlim=10, swap_ab=False):
    """
    Plot the canonical hyperbola fit and its rotated version along with data points.
    Parameters
    ----------
    x : array_like
        X-coordinates of the data points.
    y : array_like
        Y-coordinates of the data points.
    canonical_fit : array_like
        Parameters of the canonical hyperbola fit.
    rlim : float, optional
        Range limit for the plot axes (default is 10).
    swap_ab : bool, optional
        If True, swap the first two parameters of canonical_fit before plotting.
    Returns
    -------
    None
    """

    if swap_ab:
        tmp = canonical_fit[0]
        canonical_fit[0] = canonical_fit[1]
        canonical_fit[1] = tmp
    x1_fit, y1_fit, x2_fit, y2_fit = hyp_branches(canonical_fit)
    canonical_fit[2] -= np.pi / 2
    x3_fit, y3_fit, x4_fit, y4_fit = hyp_branches(canonical_fit)
    plt.figure()
    plt.scatter(x, y, s=5, color="r")
    plt.plot(x1_fit, y1_fit, color="k")
    plt.plot(x2_fit, y2_fit, color="k")
    plt.plot(x3_fit, y3_fit, color="b")
    plt.plot(x4_fit, y4_fit, color="b")
    plt.xlim([-rlim, rlim])
    plt.ylim([-rlim, rlim])
    plt.show()
    return


if __name__ == "__main__":

    # -----------------
    # Simple example
    # -----------------

    if len(sys.argv) != 6:
        print("Usage: python3 hypfit.py seed th_min th_max npoints scale_noise")
        print("Using default values\n")
        seed = 1
        th_min = 0.0
        th_max = 3.14
        npoints = 100
        noise = 0.0
    else:
        seed = int(sys.argv[1])
        th_min = float(sys.argv[2])
        th_max = float(sys.argv[3])
        npoints = int(sys.argv[4])
        noise = float(sys.argv[5])

    np.random.seed(seed)

    # hyperbola points
    th = np.linspace(th_min, th_max, num=npoints)
    a = np.random.rand() * 4 + 1
    b = np.random.rand() * 4 + 1
    ph = np.random.rand() * np.pi / 2
    x0 = np.random.rand() * 2
    y0 = np.random.rand() * 2

    canonical_0 = np.array([a, b, ph, x0, y0])
    x, y = hyp_parametrization(th, canonical_0)

    scale_noise = noise
    if scale_noise > 0:
        xnoise = np.random.normal(loc=1.0, scale=scale_noise, size=np.shape(x))
        ynoise = np.random.normal(loc=1.0, scale=scale_noise, size=np.shape(y))
        x *= xnoise
        y *= ynoise

    # fit data
    ABCDF = fit_quadratic(x, y)
    canonical_fit = quadratic_to_canonical(ABCDF)

    if abs(canonical_fit[0] - canonical_0[1]) < abs(canonical_fit[0] - canonical_0[0]):
        swap_ab = True
    else:
        swap_ab = False

    plot_hypfit(x, y, canonical_fit, swap_ab=swap_ab)

    # Compare fitted coeff with original ones
    names = ["a", "b", "ph", "x0", "y0"]
    err = (canonical_0 - canonical_fit) / canonical_0

    dashes = "---------------------------------------------------------"
    print(dashes, "\n           Input     vs      fit", "\n", dashes, sep="")
    for i in range(5):
        print(
            "{:2s} : {:15.5e}  {:15.5e}  ->  {:9.2e} %".format(
                names[i], canonical_0[i], canonical_fit[i], err[i]
            )
        )

    print(dashes, "\nAdditional comparisons, in case of swapping\n", dashes, sep="")
    print(
        "(a_0-b_fit)/a_0         : {:9.2e} %".format(
            (canonical_0[0] - canonical_fit[1]) / canonical_0[0]
        )
    )
    print(
        "(b_0-a_fit)/b_0         : {:9.2e} %".format(
            (canonical_0[1] - canonical_fit[0]) / canonical_0[1]
        )
    )
    print(
        "abs(ph_0 - ph_fit)-pi/2 : {:9.2e}".format(
            abs(canonical_0[2] - canonical_fit[2]) - np.pi / 2
        )
    )
