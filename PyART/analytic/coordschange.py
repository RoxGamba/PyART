"""
Various utility functions for coordinate transformations
and calculation of PN energy and angular momentum.

SA, DC, RG, 2020-2025
"""

import numpy as np

pi = np.pi


def Polar2Cartesian(r, phi, pr, pphi):
    """
    Transform polar coordinates (r, phi) and momenta (pr, pphi)
    to Cartesian coordinates (x, y) and momenta (px, py).

    Parameters
    ----------
    r : array_like
        Radial coordinate.
    phi : array_like
        Angular coordinate.
    pr : array_like
        Radial momentum.
    pphi : array_like
        Angular momentum.

    Returns
    -------
    x : array_like
        Cartesian x coordinate.
    y : array_like
        Cartesian y coordinate.
    px : array_like
        Cartesian x momentum.
    py : array_like
        Cartesian y momentum.
    """

    x = r * np.cos(phi)
    y = r * np.sin(phi)
    px = np.cos(phi) * pr - np.sin(phi) * pphi / r
    py = np.sin(phi) * pr + np.cos(phi) * pphi / r

    return x, y, px, py


def Cartesian2Polar(x, y, px, py):
    """
    Transform Cartesian coordinates (x, y) and momenta (px, py)
    to polar coordinates (r, phi) and momenta (pr, pphi).

    Parameters
    ----------
    x : array_like
        Cartesian x coordinate.
    y : array_like
        Cartesian y coordinate.
    px : array_like
        Cartesian x momentum.
    py : array_like
        Cartesian y momentum.

    Returns
    -------
    r : array_like
        Radial coordinate.
    phi : array_like
        Angular coordinate.
    pr : array_like
        Radial momentum.
    pphi : array_like
        Angular momentum.
    """
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan(y / x) + pi * np.logical_and(x < 0, 1)
    pr = (x * px + y * py) / r
    pphi = x * py - y * px

    return r, phi, pr, pphi


###########################
##      EOB <-> ADM      ##
###########################
def Eob2Adm(qe_vec, pe_vec, nu, PN_order):
    """
    Convert EOB coordinates and momenta to ADM coordinates and momenta
    up to 2PN order. Use the transformation in
    Buonanno, Damour:9811091 and Bini, Damour:1210.2834 (Appendix E)
    Note: momenta are mu-normalized

    Parameters
    ----------
    qe_vec : array_like
        EOB coordinates (x, y).
    pe_vec : array_like
        EOB momenta (px, py), mu-normalized.
    nu : float
        Symmetric mass ratio.
    PN_order : int
        Post-Newtonian order (0, 1, or 2).

    Returns
    -------
    qa_vec : array_like
        ADM coordinates (x, y).
    pa_vec : array_like
        ADM momenta (px, py), mu-normalized.
    """
    # shorthands
    qe2 = np.dot(qe_vec, qe_vec)  # x, y
    qe = np.sqrt(qe2)
    qe3 = qe * qe2
    pe2 = np.dot(pe_vec, pe_vec)
    pe = np.sqrt(pe2)
    pe3 = pe * pe2
    pe4 = pe * pe3
    qedotpe = np.dot(qe_vec, pe_vec)
    qedotpe2 = qedotpe * qedotpe
    nu2 = nu * nu

    # coefficients for ADM coordinates
    cqa_1PN_q = nu * pe2 / 2 - (1 + nu / 2) / qe
    cqa_1PN_p = nu * qedotpe
    cqa_2PN_q = (
        -nu / 8 * (1 + nu) * pe4
        + 3 / 4 * nu * (nu / 2 - 1) * pe2 / qe
        - nu * (2 + 5 / 8 * nu) * qedotpe2 / qe3
        + (-nu2 + 7 * nu - 1) / 4 / qe2
    )
    cqa_2PN_p = qedotpe * (nu * (nu - 1) / 2 * pe2 + nu / 2 * (-5 + nu / 2) / qe)

    # coefficients for ADM momenta
    cpa_1PN_q = -(1 + nu / 2) * qedotpe / qe3
    cpa_1PN_p = -nu / 2 * pe2 + (1 + nu / 2) / qe
    cpa_2PN_q = (
        qedotpe
        / qe3
        * (
            3 / 4 * nu * (nu / 2 - 1) * pe2
            + 3 / 8 * nu2 * qedotpe2 / qe2
            + (-3 / 2 + 5 / 2 * nu - 3 / 4 * nu2) / qe
        )
    )
    cpa_2PN_p = (
        nu * (1 + 3 * nu) / 8 * pe4
        - nu / 4 * (1 + 7 / 2 * nu) * pe2 / qe
        + nu * (1 + nu / 8) * qedotpe2 / qe3
        + (5 / 4 - 3 / 4 * nu + nu2 / 2) / qe2
    )

    # Put all together
    qa_vec = qe_vec
    pa_vec = pe_vec
    if PN_order > 0:
        qa_vec = qa_vec + cqa_1PN_q * qe_vec + cqa_1PN_p * pe_vec
        pa_vec = pa_vec + cpa_1PN_q * qe_vec + cpa_1PN_p * pe_vec
    if PN_order > 1:
        qa_vec = qa_vec + cqa_2PN_q * qe_vec + cqa_2PN_p * pe_vec
        pa_vec = pa_vec + cpa_2PN_q * qe_vec + cpa_2PN_p * pe_vec
    if PN_order > 2:
        print("2PN is the max PN order available")
    return qa_vec, pa_vec


def Adm2Eob(qa_vec, pa_vec, nu, PN_order):
    """
    Convert ADM coordinates and momenta to EOB coordinates and momenta
    up to 2PN order. Use the transformation in
    Buonanno, Damour:9811091 and Bini, Damour:1210.2834 (Appendix E)
    Note: momenta are mu-normalized

    Parameters
    ----------
    qa_vec : array_like
        ADM coordinates (x, y).
    pa_vec : array_like
        ADM momenta (px, py), mu-normalized.
    nu : float
        Symmetric mass ratio.
    PN_order : int
        Post-Newtonian order (0, 1, or 2).

    Returns
    -------
    qe_vec : array_like
        EOB coordinates (x, y).
    pe_vec : array_like
        EOB momenta (px, py), mu-normalized.
    """

    # shorthands
    qa2 = np.dot(qa_vec, qa_vec)  # x, y
    qa = np.sqrt(qa2)
    qa3 = qa * qa2
    pa2 = np.dot(pa_vec, pa_vec)
    pa = np.sqrt(pa2)
    pa3 = pa * pa2
    pa4 = pa * pa3
    qadotpa = np.dot(qa_vec, pa_vec)
    qadotpa2 = qadotpa * qadotpa
    nu2 = nu * nu

    # coefficients for EOB coordinates
    cqe_1PN_q = -nu / 2 * pa2 + 1 / qa * (1 + nu / 2)
    cqe_1PN_p = -qadotpa * nu
    cqe_2PN_q = (
        nu / 8 * (1 - nu) * pa4
        + nu / 4 * (5 - nu / 2) * pa2 / qa
        + nu * (1 + nu / 8) * qadotpa2 / qa3
        + 1 / 4 * (1 - 7 * nu + nu2) / qa2
    )
    cqe_2PN_p = qadotpa * (nu / 2 * (1 + nu) * pa2 + 3 / 2 * nu * (1 - nu / 2) / qa)

    # coefficients for EOB momenta
    cpe_1PN_q = qadotpa / qa3 * (1 + nu / 2)
    cpe_1PN_p = nu / 2 * pa2 - 1 / qa * (1 + nu / 2)
    cpe_2PN_q = (
        qadotpa
        / qa3
        * (
            nu / 8 * (10 - nu) * pa2
            + 3 / 8 * nu * (8 + 3 * nu) * qadotpa2 / qa2
            + 1 / 4 * (-2 - 18 * nu + nu2) / qa
        )
    )
    cpe_2PN_p = (
        nu / 8 * (-1 + 3 * nu) * pa4
        - 3 / 4 * nu * (3 + nu / 2) * pa2 / qa
        - nu / 8 * (16 + 5 * nu) * qadotpa2 / qa3
        + 1 / 4 * (3 + 11 * nu) / qa2
    )

    # Put all together
    qe_vec = qa_vec
    pe_vec = pa_vec
    if PN_order > 0:
        qe_vec = qe_vec + cqe_1PN_q * qa_vec + cqe_1PN_p * pa_vec
        pe_vec = pe_vec + cpe_1PN_q * qa_vec + cpe_1PN_p * pa_vec
    if PN_order > 1:
        qe_vec = qe_vec + cqe_2PN_q * qa_vec + cqe_2PN_p * pa_vec
        pe_vec = pe_vec + cpe_2PN_q * qa_vec + cpe_2PN_p * pa_vec
    if PN_order > 2:
        print("2PN is the max PN order available")

    return qe_vec, pe_vec


def eob_ID_to_ADM(eob_Wave, verbose=False, PN_order=2, rotate_on_x_axis=True):
    """
    Generate initial ID for NR simulations with initial
    data from TwoPuncturesC

    Parameters
    ----------
    eob_Wave : object
        EOB waveform, instance of PyART.models.teob.Waveform_EOB
    verbose : bool, optional
        If True, print out info for testing. Default is False.
    PN_order : int, optional
        Post-Newtonian order for EOB <-> ADM transformation.
        Default is 2 (max available).
    rotate_on_x_axis : bool, optional
        If True, rotate system so that punctures are on x-axis at t=0.
        Default is True.

    Returns
    -------
    out : dict
        Dictionary with the following keys:
        'q_cart' : ADM coordinates (x, y)
        'p_cart' : ADM momenta (px, py), mu-normalized
        'px' : ADM momentum px, M-normalized
        'py' : ADM momentum py, M-normalized
        'x1' : x coordinate of puncture 1
        'x2' : x coordinate of puncture 2
        'D' : coordinate separation between punctures
        'x_offset' : offset to be added to x coordinates
    """
    # Get info from EOB dynamics
    q = eob_Wave.pars["q"]
    nu = q / (1 + q) ** 2
    r0 = eob_Wave.dyn["r"][0]
    phi0 = eob_Wave.dyn["phi"][0]
    pph0 = eob_Wave.dyn["Pphi"][0]
    pr = eob_Wave.get_Pr()
    pr0 = pr[0]

    # Convert to EOB Cartesian
    x, y, px, py = Polar2Cartesian(r0, phi0, pr0, pph0)
    qe = np.array([x, y])
    pe = np.array([px, py])  # already divided by nu

    # Convert to ADM Cartesian
    qa, pa = Eob2Adm(qe, pe, nu, PN_order=PN_order)
    d_ADM = np.sqrt(np.dot(qa, qa))

    halfx = qa[0] / 2
    b_par = d_ADM / 2
    if rotate_on_x_axis:
        # Rotate so that the punctures will be on the x-axis at t=0
        cosa = halfx / b_par
        sina = np.sqrt(1 - cosa * cosa)
        if qa[1] > 0:
            sina = -sina
    else:
        cosa = 1.0
        sina = 0.0
    pxbynu = cosa * pa[0] - sina * pa[1]
    pybynu = sina * pa[0] + cosa * pa[1]

    x1 = d_ADM / (q + 1)
    x2 = -d_ADM * q / (q + 1)
    x_offset = -b_par + d_ADM / (q + 1)

    # wrap output
    out = {
        "q_cart": qa,
        "p_cart": pa,
        "px": pxbynu * nu,
        "py": pybynu * nu,
        "x1": x1,
        "x2": x2,
        "D": d_ADM,
        "x_offset": x_offset,
    }

    if verbose:
        # for testing
        qe_check, pe_check = Adm2Eob(qa, pa, nu, PN_order=PN_order)

        dashes = "-" * 50
        print("{}\nPunctures\n{}".format(dashes, dashes))
        print("b_par    : {:.15f}".format(b_par))
        print("D        : {:.15f}".format(b_par * 2))
        print("x_offset : {:.15f}".format(x_offset))
        print("px       : {:.15f}".format(pxbynu * nu))
        print("py       : {:.15f}\n".format(pybynu * nu))

        print("{}\nEOB-ADM 2PN transformation\n{}".format(dashes, dashes))
        print("q EOB      : {:.5e}, {:.5e}".format(qe[0], qe[1]))
        print("q EOB->ADM : {:.5e}, {:.5e}".format(qa[0], qa[1]))
        print("q ADM->EOB : {:.5e}, {:.5e}\n".format(qe_check[0], qe_check[1]))

        print("p EOB      : {:.5e}, {:.5e}".format(pe[0], pe[1]))
        print("p EOB->ADM : {:.5e}, {:.5e}".format(pa[0], pa[1]))
        print("p ADM->EOB : {:.5e}, {:.5e}\n".format(pe_check[0], pe_check[1]))

    return out


def H_ADM(r, pr, L, chi1, chi2, nu, pn=2):
    """
    ADM Hamiltonian
    Eq C1 from https://arxiv.org/pdf/2209.00611

    Parameters
    ----------
    r : float
        Radial coordinate.
    pr : float
        Radial momentum.
    L : float
        Angular momentum.
    chi1 : float
        Dimensionless spin of body 1.
    chi2 : float
        Dimensionless spin of body 2.
    nu : float
        Symmetric mass ratio. Default is 0.25.
    pn : int, optional
        Post-Newtonian order (0, 1, or 2). Default is 2.

    Returns
    -------
    H : float
        ADM Hamiltonian at given PN order.
    """

    u = 1 / r
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u

    dm = np.sqrt(1 - 4 * nu)
    m1 = (1 + dm) / 2
    m2 = 1 - m1

    H_0 = 0.5 * pr**2 - u + 0.5 * L**2 * u2
    H_1 = (
        0.5 * u2
        - pr**2 * (nu + 3 / 2) * u
        + +(pr**4) * (3 * nu / 8 - 1.0 / 8)
        + +(L**2) * (-nu / 2 * u3 - 3.0 / 2 * u3 + pr**2 * (3 * nu / 4 - 1.0 / 4) * u2)
        + L**4 * (3.0 * nu / 8 * u4 - 1.0 / 8 * u4)
    )

    H_2 = (
        -nu / 4 * u3
        - 0.5 * u3
        + pr**2 * (9 * nu / 2 * u2 + 9 / 4 * u2)
        + +(pr**4) * (-nu * nu - 5 * nu / 2 + 5 / 8) * u
        + pr**6 * (5 * nu * nu / 16 - 5 * nu / 16 + 1.0 / 16)
        + +(L**2)
        * (
            (3 * nu + 11 / 4) * u4
            + pr**2 * (-nu * nu - 9 * nu / 2 + 5 / 4) * u3
            + pr**4 * (15 * nu * nu / 16 - 15 * nu / 16 + 3.0 / 16) * u2
        )
        + +(L**4)
        * (
            (-3 * nu * nu / 8 - 2 * nu + 5 / 8) * u4 * u
            + pr**2 * (15 * nu * nu / 16 - 15 * nu / 16 + 3.0 / 16) * u4
        )
        + L**6 * (5 * nu * nu / 16 - 5 * nu / 16 + 1.0 / 16) * u3 * u3
    )

    # Add spin part of Hamiltonian

    H_S_LO = L * (u3) * (1.5 * nu * (chi1 + chi2) + 2 * (m1**2 * chi1 + m2**2 * chi2))

    if pn == 0:
        return H_0
    elif pn == 1:
        return H_0 + H_1
    elif pn == 2:
        return H_0 + H_1 + H_2 + H_S_LO
    else:
        raise ValueError("PN order not available, max is 2")


## EJ from dynamical variables (harmonic coordinates)
def EJ_from_rv(r, v, S, Sig, nu):
    """
    Calculates PN energy and ang momentum from the dynamical variables,
    harmonic coordinates up to 3PN order, including spin-orbit and
    spin-spin couplings.
    TODO: add reference (Blanchet etc)
    """

    nu2 = nu**2

    q = (1 - 2 * nu + np.sqrt(1 - 4 * nu)) / (2 * nu)
    M1 = q / (1 + q)
    M2 = 1 / (1 + q)
    deltam = M1 - M2
    mu = nu / (M1 + M2)

    rx = r[0][:]
    ry = r[1][:]
    rz = r[2][:]

    vx = v[0][:]
    vy = v[1][:]
    vz = v[2][:]

    rq = rx**2 + ry**2 + rz**2
    vq = vx**2 + vy**2 + vz**2

    rad = np.sqrt(rq)
    vel = np.sqrt(vq)

    rdot = (rx * vx + ry * vy + rz * vz) / rad

    Sx = S[0][:]
    Sy = S[1][:]
    Sz = S[2][:]

    Sigx = Sig[0][:]
    Sigy = Sig[1][:]
    Sigz = Sig[2][:]

    nSv = (
        rx * Sy * vz
        - rx * Sz * vy
        + ry * Sz * vx
        - ry * Sx * vz
        + rz * Sx * vy
        - rz * Sy * vx
    ) / rad
    nSigv = (
        -(
            rx * Sigy * vz
            - rx * Sigz * vy
            + ry * Sigz * vx
            - ry * Sigx * vz
            + rz * Sigx * vy
            - rz * Sigy * vx
        )
        / rad
    )

    Sigv = Sigx * vx + Sigy * vy + Sigz * vz
    Sv = Sx * vx + Sy * vy + Sz * vz
    nSig = (rx * Sigx + ry * Sigy + rz * Sigz) / rad
    nS = (rx * Sx + ry * Sy + rz * Sz) / rad

    # Energy
    # N + 1PN

    Energy = (
        vq / 2
        - 1 / rad
        + 3 / 8 * (1 - 3 * nu) * vq**2
        + 1 / 2 * (3 + nu) * vq / rad
        + 1 / 2 * nu * rdot**2 / rad
        + 1 / 2 / rq
    )

    # 1.5PN

    Energy += (-deltam * nSigv - nSv) / rq

    # 2PN

    Energy += (
        vq**3 * 5 / 16 * (1 - 7 * nu + 13 * nu2)
        + 1 / 8 * (21 - 23 * nu - 27 * nu2) * vq**2 / rad
        + 1 / 4 * nu * (1 - 15 * nu) * vq * rdot**2 / rad
        - 3 / 8 * nu * (1 - 3 * nu) * rdot**4 / rad
        + 1 / 8 * (14 - 55 * nu + 4 * nu2) * vq / rq
        + 1 / 8 * (4 + 69 * nu + 12 * nu2) * rdot**2 / rq
        - 1 / 4 * (2 + 15 * nu) / (rq * rad)
    )

    # 2PN SS

    Energy += 0

    # 2.5PN

    Energy += (
        nSigv * deltam * (-1 + 5 * nu) / 2 * vq
        + nSv * (3 / 2 * nu * rdot**2 + 3 / 2 * (1 + nu) * vq)
    ) / rq + (-3 / 2 * deltam * nSigv - 2 * nSv) * nu / (rq * rad)

    # 3PN

    Energy += (
        (315 + 18469 * nu) / (840 * rad**4)
        - ((rdot**2) * vq**2 * nu * (21 + 75 * nu - 375 * nu**2)) / (16 * rad)
        - (rdot**4 * nu * (731 - 429 * nu - 288 * nu**2)) / (48 * rad**2)
        + (rdot**6 * nu * (5 - 25 * nu + 25 * nu**2)) / (16 * rad)
        - (rdot**4 * vq * nu * (9 - 84 * nu + 165 * nu**2)) / (16 * rad)
        + 1 / 128 * vq**4 * (35 - 413 * nu + 1666 * nu**2 - 2261 * nu**3)
        + (rdot**2 * vq * (12 + 248 * nu - 815 * nu**2 - 324 * nu**3)) / (16 * rad**2)
        + (vq**2 * (135 - 194 * nu + 406 * nu**2 - 108 * nu**3)) / (16 * rad**2)
        + (vq**3 * (55 - 215 * nu + 116 * nu**2 + 325 * nu**3)) / (16 * rad)
        + (vq * (2800 - (53976 - 1435 * np.pi**2) * nu - 11760 * nu**2 + 1120 * nu**3))
        / (2240 * rad**3)
        + (
            rdot**2
            * (3360 + (18568 - 4305 * np.pi**2) * nu + 28560 * nu**2 + 7840 * nu**3)
        )
        / (2240 * rad**3)
    )

    # Angular momentum
    # N + 1PN

    Jang = 1 + 1 / 2 * (1 - 3 * nu) * vq + (3 - nu) / rad

    # 2PN

    Jang += (
        -((rdot**2 * nu * (2 + 5 * nu)) / (2 * rad))
        + (vq * (7 - 10 * nu - 9 * nu**2)) / (2 * rad)
        + (14 - 41 * nu + 4 * nu**2) / (4 * rad**2)
        + 3 / 8 * vq**2 * (1 - 7 * nu + 13 * nu**2)
    )

    # 3PN

    Jang += (
        -((rdot**2 * vq * nu * (12 - 7 * nu - 75 * nu**2)) / (4 * rad))
        + (3 * rdot**4 * nu * (2 - 2 * nu - 11 * nu**2)) / (8 * rad)
        + (rdot**2 * (12 - 287 * nu - 951 * nu**2 - 324 * nu**3)) / (24 * rad**2)
        + 1 / 16 * vq**3 * (5 - 59 * nu + 238 * nu**2 - 323 * nu**3)
        + (vq * (135 - 322 * nu + 315 * nu**2 - 108 * nu**3)) / (12 * rad**2)
        + (5 / 2 - ((20796 - 1435 * np.pi**2) * nu) / 1120 - 7 * nu**2 + nu**3) / rad**3
        + (vq**2 * (33 - 142 * nu + 106 * nu**2 + 195 * nu**3)) / (8 * rad)
    )

    # Jang_norm = Jang*rad*np.sqrt(vq - rdot**2)

    # Components, only non-spin

    Jangx = Jang * (ry * vz - rz * vy)
    Jangy = Jang * (rz * vx - rx * vz)
    Jangz = Jang * (rx * vy - ry * vx)

    # 1.5PN, SO

    Jangx += (
        -1 / 2 * deltam * Sigv * vx
        - 1 / 2 * Sv * vx
        + 1 / 2 * vq * Sx
        + 1 / 2 * deltam * vq * Sigx
    ) + ((deltam * nSig + 3 * nS) * rx / rad - 3 * Sx - deltam * Sigx) / rad
    Jangy += (
        -1 / 2 * deltam * Sigv * vy
        - 1 / 2 * Sv * vy
        + 1 / 2 * vq * Sy
        + 1 / 2 * deltam * vq * Sigy
    ) + ((deltam * nSig + 3 * nS) * ry / rad - 3 * Sy - deltam * Sigy) / rad
    Jangz += (
        -1 / 2 * deltam * Sigv * vz
        - 1 / 2 * Sv * vz
        + 1 / 2 * vq * Sz
        + 1 / 2 * deltam * vq * Sigz
    ) + ((deltam * nSig + 3 * nS) * rz / rad - 3 * Sz - deltam * Sigz) / rad

    # 2.5PN, SO

    Jangx += (
        (
            (Sigv * deltam * (-3 / 8 + 5 / 4 * nu) + Sv * (-3 / 8 + 9 / 8 * nu))
            * vx
            * vq
            + vq**2 * (3 / 8 - 9 / 8 * nu) * Sx
            + deltam * Sigx * (3 / 8 - 5 / 4 * nu) * vq**2
        )
        + (
            (
                nSig * deltam * (-3 * nu * rdot**2 + (1 + 3 * nu) / 2 * vq)
                + Sigv * deltam * rdot * (-1 / 2 - 13 / 4 * nu)
                + nS * (-9 / 2 * nu * rdot**2 + (7 - nu) / 2 * vq)
                + Sv * rdot * (-3 - 7 * nu) / 2
            )
            * rx
            / rad
            + (
                7 / 4 * nu * deltam * rdot * nSig
                + Sigv * deltam * (-3 + nu) / 2
                + nS * (-3 + 6 * nu) * rdot
                + Sv * (1 - nu) / 2
            )
            * vx
            + ((2 + 5 / 2 * nu) * rdot**2 + (-3 + nu) / 2 * vq) * Sx
            + Sigx * deltam * (7 / 2 * nu * rdot**2 + (3 / 2 - nu) * vq)
        )
        / rad
        + (
            rx / rad * (nSig * deltam * (-1 - 3 * nu) / 2 + nS * (-1 / 2 - 2 * nu))
            + Sx * (1 / 2 + 2 * nu)
            + Sigx * deltam * (1 / 2 + 3 / 2 * nu)
        )
        / rq
    )
    Jangy += (
        (
            (Sigv * deltam * (-3 / 8 + 5 / 4 * nu) + Sv * (-3 / 8 + 9 / 8 * nu))
            * vy
            * vq
            + vq**2 * (3 / 8 - 9 / 8 * nu) * Sy
            + deltam * Sigy * (3 / 8 - 5 / 4 * nu) * vq**2
        )
        + (
            (
                nSig * deltam * (-3 * nu * rdot**2 + (1 + 3 * nu) / 2 * vq)
                + Sigv * deltam * rdot * (-1 / 2 - 13 / 4 * nu)
                + nS * (-9 / 2 * nu * rdot**2 + (7 - nu) / 2 * vq)
                + Sv * rdot * (-3 - 7 * nu) / 2
            )
            * ry
            / rad
            + (
                7 / 4 * nu * deltam * rdot * nSig
                + Sigv * deltam * (-3 + nu) / 2
                + nS * (-3 + 6 * nu) * rdot
                + Sv * (1 - nu) / 2
            )
            * vy
            + ((2 + 5 / 2 * nu) * rdot**2 + (-3 + nu) / 2 * vq) * Sy
            + Sigy * deltam * (7 / 2 * nu * rdot**2 + (3 / 2 - nu) * vq)
        )
        / rad
        + (
            ry / rad * (nSig * deltam * (-1 - 3 * nu) / 2 + nS * (-1 / 2 - 2 * nu))
            + Sy * (1 / 2 + 2 * nu)
            + Sigy * deltam * (1 / 2 + 3 / 2 * nu)
        )
        / rq
    )
    Jangz += (
        (
            (Sigv * deltam * (-3 / 8 + 5 / 4 * nu) + Sv * (-3 / 8 + 9 / 8 * nu))
            * vz
            * vq
            + vq**2 * (3 / 8 - 9 / 8 * nu) * Sz
            + deltam * Sigz * (3 / 8 - 5 / 4 * nu) * vq**2
        )
        + (
            (
                nSig * deltam * (-3 * nu * rdot**2 + (1 + 3 * nu) / 2 * vq)
                + Sigv * deltam * rdot * (-1 / 2 - 13 / 4 * nu)
                + nS * (-9 / 2 * nu * rdot**2 + (7 - nu) / 2 * vq)
                + Sv * rdot * (-3 - 7 * nu) / 2
            )
            * rz
            / rad
            + (
                7 / 4 * nu * deltam * rdot * nSig
                + Sigv * deltam * (-3 + nu) / 2
                + nS * (-3 + 6 * nu) * rdot
                + Sv * (1 - nu) / 2
            )
            * vz
            + ((2 + 5 / 2 * nu) * rdot**2 + (-3 + nu) / 2 * vq) * Sz
            + Sigz * deltam * (7 / 2 * nu * rdot**2 + (3 / 2 - nu) * vq)
        )
        / rad
        + (
            rz / rad * (nSig * deltam * (-1 - 3 * nu) / 2 + nS * (-1 / 2 - 2 * nu))
            + Sz * (1 / 2 + 2 * nu)
            + Sigz * deltam * (1 / 2 + 3 / 2 * nu)
        )
        / rq
    )

    # Norm

    Jang_norm = np.sqrt(Jangx**2 + Jangy**2 + Jangz**2)

    return Energy, mu * Jang_norm, [mu * Jangx, mu * Jangy, mu * Jangz]
