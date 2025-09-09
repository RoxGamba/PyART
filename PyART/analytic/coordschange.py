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
