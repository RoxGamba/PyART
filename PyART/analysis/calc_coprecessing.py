import logging
import numpy as np
from ..utils import utils as ut


# Import useful routines
from scipy.linalg import eig, norm
from scipy.interpolate import InterpolatedUnivariateSpline as IUS


# Given dictionary of multipoles all with the same l, calculate the roated multipole with (l,mp)
def rotate_wfarrs_at_all_times(
    l,  # the l of the new multipole (everything should have the same l)
    m,  # the m of the new multipole
    like_l_multipoles_dict,  # dictionary in the format { (l,m): array([domain_values,re,img]) }
    euler_alpha_beta_gamma,
    ref_orientation=None,
):
    """
    Given dictionary of multipoles all with the same l, calculate the roated multipole with (l,mp).
    Key reference -- arxiv:1012:2879
    Based on LL,EZH 2018
    """
    # unpack the euler angles
    alpha, beta, gamma = euler_alpha_beta_gamma

    # Handle the default behavior for the reference orientation
    if ref_orientation is None:
        ref_orientation = np.ones(3)

    # Apply the desired reflection for the reference orientation.
    # NOTE that this is primarily useful for BAM run which have an atypical coordinate setup if Jz<0

    gamma *= np.sign(ref_orientation[-1])
    alpha *= np.sign(ref_orientation[-1])

    new_plus = 0
    new_cross = 0

    for lm in like_l_multipoles_dict:
        # See eq A9 of arxiv:1012:2879
        l, mp = lm
        old_wfarr = like_l_multipoles_dict[lm]

        d = ut.wdelement(l, m, mp, alpha, beta, gamma)
        a, b = d.real, d.imag
        p = old_wfarr[1]
        c = old_wfarr[2]

        new_plus += a * p - b * c
        new_cross += b * p + a * c

    # Construct the new waveform array

    return {
        "real": new_plus,
        "imag": new_cross,
        "A": np.sqrt(new_plus**2 + new_cross**2),
        "p": np.arctan2(new_cross, new_plus),
    }


# Given a dictionary of multipole data, calculate the Euler angles corresponding to a co-precessing frame
# Taken from nrutils_dev, credits to Lionel London
def calc_coprecessing_angles(
    multipole_dict,  # Dict of multipoles { ... l,m:data_lm ... }
    domain_vals=None,  # The time or freq series for multipole data
    ref_orientation=None,  # e.g. initial J; used for breaking degeneracies in calculation
    return_xyz=False,
    safe_domain_range=None,
):
    """
    Given a dictionary of multipole data, calculate the Euler angles corresponding to a co-precessing frame
    Key referece: https://arxiv.org/pdf/1304.3176.pdf
    Secondary ref: https://arxiv.org/pdf/1205.2287.pdf
    INPUT
    ---
    multipole_dict,       # dict of multipoles { ... l,m:data_lm ... }
    t,                    # The time series corresponding to multipole data; needed
                            only to calculate gamma; Optional
    OUTPUT
    ---
    alpha,beta,gamma euler angles as defined in https://arxiv.org/pdf/1205.2287.pdf
    AUTHOR
    ---
    Lionel London (spxll) 2017
    """

    # Handle optional input
    if ref_orientation is None:
        ref_orientation = np.ones(3)

    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
    # Enforce that multipole data is array typed with a well defined length
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
    y = multipole_dict
    for l, m in y:
        if isinstance(y[l, m], (float, int)):
            y[l, m] = np.array(
                [
                    y[l, m],
                ]
            )

    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
    # Calculate the emission tensor corresponding to the input data
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
    L = calc_Lab_tensor(multipole_dict)

    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
    # Compute the eigenvectors and values of this tensor
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#

    # NOTE that members of L have the same length as each y[l,m]; the latter has been
    # forced to always have a length above

    # Initialize idents for angles.
    # NOTE that gamma will be handled below
    alpha, beta = [], []
    X, Y, Z = [], [], []
    old_dom_dex = None

    # For all multipole instances
    ref_x, ref_y, ref_z = None, None, None
    flip_z_convention = False
    for k in range(len(L[0, 0, :])):

        # Select the emission matrix for this instance, k
        _L = L[:, :, k]

        # Compute the eigen vals and vecs for this instance
        vals, vec = eig(_L)

        # Find the dominant direction's index
        dominant_dex = np.argmax(vals)
        if old_dom_dex is None:
            old_dom_dex = dominant_dex
        if old_dom_dex != dominant_dex:
            # print dominant_dex
            old_dom_dex = dominant_dex

        # Select the corresponding vector
        dominant_vec = vec[:, dominant_dex]

        # There is a z axis degeneracy that we will break here
        # by imposing that the z component is always consistent with the initial L
        if not flip_z_convention:
            if np.sign(dominant_vec[-1]) == -np.sign(ref_orientation[-1]):
                dominant_vec *= -1
        else:
            if np.sign(dominant_vec[-1]) == np.sign(ref_orientation[-1]):
                dominant_vec *= -1

        # dominant_vec *= sign(domain_vals[k])

        # Extract the components of the dominant eigenvector
        _x, _y, _z = dominant_vec

        # Store reference values if they are None
        if ref_x == None:
            ref_x = _x
            ref_y = _y
            ref_z = _z
        else:
            if (ref_x * _x < 0) and (ref_y * _y < 0):
                _x *= -1
                _y *= -1
                _x *= -1

        # Store unit components for reference in the next iternation
        ref_x = _x
        ref_y = _y
        ref_z = _z

        # Look for and handle trivial cases
        if abs(_x) + abs(_y) < 1e-8:
            _x = _y = 0

        X.append(_x)
        Y.append(_y)
        Z.append(_z)

    # Look for point reflection in X
    X = ut.reflect_unwrap(np.array(X))
    Y = np.array(Y)
    Z = np.array(Z)

    # 3-point vector reflect unwrapping
    # print safe_domain_range
    tol = 0.1
    if safe_domain_range is None:
        safe_domain_range = ut.minmax_array(abs(domain_vals))
    safe_domain_range = np.array(safe_domain_range)
    for k in range(len(X))[1:-1]:
        if k > 0 and k < (len(domain_vals) - 1):

            if (abs(domain_vals[k]) > min(abs(safe_domain_range))) and (
                abs(domain_vals[k]) < max(abs(safe_domain_range))
            ):

                left_x_has_reflected = abs(X[k] + X[k - 1]) < tol * abs(X[k - 1])
                left_y_has_reflected = abs(Y[k] + Y[k - 1]) < tol * abs(X[k - 1])

                right_x_has_reflected = abs(X[k] + X[k + 1]) < tol * abs(X[k])
                right_y_has_reflected = abs(Y[k] + Y[k + 1]) < tol * abs(X[k])

                x_has_reflected = right_x_has_reflected or left_x_has_reflected
                y_has_reflected = left_y_has_reflected or right_y_has_reflected

                if x_has_reflected and y_has_reflected:

                    if left_x_has_reflected:
                        X[k:] *= -1
                    if right_x_has_reflected:
                        X[k + 1 :] *= -1

                    if left_y_has_reflected:
                        Y[k:] *= -1
                    if right_y_has_reflected:
                        Y[k + 1 :] *= -1

                    Z[k:] *= -1

    # Make sure that imag parts are gone
    X = np.double(X)
    Y = np.double(Y)
    Z = np.double(Z)

    #################################################
    # Reflect Y according to nrutils conventions    #
    Y *= -1  #
    #################################################

    a = np.array(ref_orientation) / norm(ref_orientation)
    B = np.array([X, Y, Z]).T
    b = B.T / norm(B, axis=1)

    # Here we define a test quantity that is always sensitive to each dimension.
    # NOTE that a simple dot product does not have this property if eg
    # a component of the reference orientation is zero.
    # There is likely a better solution here.
    test_quantity = sum([a[k] * b[k] if a[k] else b[k] for k in range(3)])

    mask = (domain_vals >= min(safe_domain_range)) & (
        domain_vals <= max(safe_domain_range)
    )
    if 1 * (test_quantity[mask][0]) < 0:
        logging.info("flipping manually for negative domain")
        X = -X
        Y = -Y
        Z = -Z

    # Calculate Angles

    alpha = np.arctan2(Y, X)
    beta = np.arccos(Z)

    # Make sure that angles are unwrapped
    alpha = np.unwrap(alpha)
    beta = np.unwrap(beta)

    # Calculate gamma (Eq. A4 of of arxiv:1304.3176)
    if len(alpha) > 1:
        k = 1
        gamma = -ut.spline_antidiff(
            domain_vals, np.cos(beta) * ut.spline_diff(domain_vals, alpha, k=k), k=k
        )
        gamma = np.unwrap(gamma)
        # Enforce like integration constant for neg and positive frequency gamma;
        # this assumes time series will not have negative values (i.e. the code should work for TD and FD cases)
        neg_mask = domain_vals < 0
        _mask = (-domain_vals) > 0.01
        mask_ = domain_vals > 0.01
        if sum(neg_mask):
            gamma[neg_mask] = gamma[neg_mask] - gamma[_mask][-1] + gamma[mask_][0]
    else:
        # NOTE that this is the same as above, but here we're choosing an integration constant such that
        # the value is zero. Above, no explicit integration constant is chosen.
        gamma = 0

    # Return answer
    if return_xyz == "all":
        return alpha, beta, gamma, X, Y, Z
    elif return_xyz:
        return X, Y, Z
    else:
        return alpha, beta, gamma


# Calculate the emission tensor given a dictionary of multipole data
# Taken from nrutils_dev, credits to Lionel London
def calc_Lab_tensor(multipole_dict):
    """
    Given a dictionary of multipole moments (single values or time series)
    determine the emission tensor, <L(aLb)>.

    See:
    - https://arxiv.org/pdf/1304.3176.pdf
    - https://arxiv.org/pdf/1205.2287.pdf
    """

    # Rename multipole_dict for short-hand
    y = multipole_dict

    # Allow user to input real and imag parts separately -- this helps with sanity checks
    x = {}
    if ("real" in y[2, 2]) or ("imag" in y[2, 2]):
        lmlist = y.keys()
        for l, m in lmlist:
            x[l, m] = y[l, m]["real"] + 1j * y[l, m]["imag"]
            x[l, m, "conj"] = x[l, m].conj()
    elif ("A" in y[2, 2]) or ("p" in y[2, 2]):
        lmlist = y.keys()
        for l, m in lmlist:
            x[l, m] = y[l, m]["A"] * np.exp(-1j * y[l, m]["p"])
            x[l, m, "conj"] = x[l, m].conj()
    else:
        raise TypeError(
            "Input must be a dictionary containing either real/imaginary or amplitude/phase"
        )
    y = x

    # Check type of dictionary values and pre-allocate output
    if isinstance(y[2, 2], (float, int, complex)):
        L = np.zeros((3, 3), dtype=complex)
    elif isinstance(y[2, 2], np.ndarray):
        L = np.zeros((3, 3, len(y[2, 2])), dtype=complex)
    else:
        logging.error("Dictionary values of handled type; must be float or array")

    # define lambda function for useful coeffs
    c = lambda l, m: np.sqrt(l * (l + 1) - m * (m + 1)) if abs(m) <= l else 0

    # Compute tensor elements (Eqs. A1-A2 of https://arxiv.org/pdf/1304.3176.pdf)
    I0, I1, I2, Izz = (
        np.zeros_like(y[2, 2]),
        np.zeros_like(y[2, 2]),
        np.zeros_like(y[2, 2]),
        np.zeros_like(y[2, 2]),
    )

    # Sum contributions from input multipoles
    for l, m in lmlist:
        # Eq. A2c
        I0 += 0.5 * (l * (l + 1) - m * m) * y[l, m] * y[l, m, "conj"]
        # Eq. A2b
        I1 += (
            c(l, m)
            * (m + 0.5)
            * (y[l, m + 1, "conj"] if (l, m + 1) in y else 0)
            * y[l, m]
        )
        # Eq. A2a
        I2 += (
            0.5
            * c(l, m)
            * c(l, m + 1)
            * y[l, m]
            * (y[l, m + 2, "conj"] if (l, m + 2) in y else 0)
        )
        # Eq. A2d
        Izz += m * m * y[l, m] * y[l, m, "conj"]

    # Compute the net power (amplitude squared) of the multipoles
    N = sum([y[l, m] * y[l, m, "conj"] for l, m in lmlist]).real

    # Populate the emission tensor ( Eq. A2e )
    # Populate asymmetric elements
    L[0, 0] = I0 + I2.real
    L[0, 1] = I2.imag
    L[0, 2] = I1.real
    L[1, 1] = I0 - I2.real
    L[1, 2] = I1.imag
    L[2, 2] = Izz
    # Populate symmetric elements
    L[1, 0] = L[0, 1]
    L[2, 0] = L[0, 2]
    L[2, 1] = L[1, 2]

    # Normalize
    N[N == 0] = min(N[N > 0])
    L = L.real / N

    return L


def calc_initial_jframe(u, dyn, wlm):
    """
    Rotate multipoles wlm to an initial frame aligned with the
    total angular momentum J = L + S
    """
    J = dyn["id"]["J0"]
    L = dyn["id"]["L0"]

    # normalize J and get the rotation angles
    J_norm = norm(J)
    JJ = J / J_norm
    thetaJ = np.arccos(JJ[2])
    phiJ = np.arctan2(JJ[1], JJ[0])

    # euler angles for the rotation
    beta = -thetaJ
    gamma = -phiJ

    # we have one additional dof to set. We choose it such that Lx = 0
    LL = ut.rotate3(L, 0, beta, gamma)
    psiL = np.arctan2(LL.T[1], LL.T[0])
    alpha = -psiL

    euler_ang = np.array([alpha, beta, gamma])

    # Rotate the multipoles
    new_wvf = {}
    for ell, emm in wlm.keys():

        # find relevant multipoles to rotate
        same_ells = [(l, m) for (l, m) in wlm.keys() if l == ell]
        same_ells_dict = {
            (l, m): [u, wlm[(l, m)]["real"], wlm[(l, m)]["imag"]]
            for (l, m) in same_ells
        }
        # perform rotation
        rotd_hlm = rotate_wfarrs_at_all_times(ell, emm, same_ells_dict, euler_ang)
        new_wvf[(ell, emm)] = rotd_hlm

    return new_wvf
