"""
Horizon fluxes in EOB coordinates on generic orbits, order by order.
Also including circular-non-circular factorization.
"""

import numpy as np
import scipy as sp

pi = np.pi


def mj1_dot_QC_NNLO(m1, m2, x, r, chi1, chi2, lo="x"):
    """
    Tidal torquing for BH, to NNLO
    From Saketh+, https://arxiv.org/pdf/2212.13095

    Note: they use x = (Momg_orb)^1/3
    """
    M = m1 + m2
    nu = m1 * m2 / M**2
    X1 = m1 / M
    X2 = m2 / M
    sigma1 = np.sqrt(1 - chi1**2)
    B2chi1 = B2(chi1)
    Omg = x**3  # orbital frequency
    OmgH = chi1 / (2 * m1 * (1 + sigma1))

    # LO
    fact = -16.0 / 5 * m1**4 * nu**2 * (1 + sigma1)
    c_LO = 1 + 3 * chi1**2
    c_NLO = 0.25 * (3 * (2 + chi1**2) + 2 * X1 * c_LO * (2 + 3 * X1))
    c_NNLO = (
        0.5 * (-4 + 3 * chi1**2) * chi2
        - 2 * X1 * c_LO * (X1 * (chi1 + chi2) + 4 * B2chi1)
        + X1
        * (
            -2.0 / 3 * (23 + 30 * sigma1) * chi1
            + (7 - 12 * sigma1) * chi1**3
            + 4 * chi2
            + 9 / 2 * chi1**2 * chi2
        )
    )

    if lo == "x":
        lo_f = x**12
    elif lo == "r":
        lo_f = 1 / r**6
    else:
        raise ValueError("LO spec not recognized; only x and r supported")

    j1dot = (OmgH - Omg) * fact * lo_f * (c_LO + x**2 * c_NLO + x**3 * c_NNLO)

    return Omg * j1dot, j1dot


def m1_dot_spin_pn(m1, m2, r, rdot, phidot, chi1, chi2, order="NNLO"):
    """
    Tidal heating for spinning BH, up to 1.5PN
    """

    chi1q = chi1**2
    sigma1 = np.sqrt(1 - chi1q)

    b1 = -1
    b2 = -1

    # LO expressions for second time derivatives are enough
    phiddot = -2 * phidot * rdot / r
    rddot = r * phidot**2 - 1 / r**2

    # LO term
    dm1 = r * phidot * (1 + 3 * chi1q)

    # 1PN term
    if "NLO" in order:
        dm1 -= 0.25 * (
            2
            * (1 + 3 * chi1q)
            * phidot
            * (
                m2 * (2 * m1 + m2) * r**2 * rddot
                + (6 * m1**2 + 2 * m2 * m1 + m2**2) * rdot**2 * r
                + 3 * m2 * (m2 + 4)
                + 2 * m1 * (2 * m2 + 5)
            )
            + 2 * m2 * (2 * m1 + m2) * r**2 * rdot * phiddot * (1 + 3 * chi1q)
            - 3 * r**3 * phidot**3 * (4 + 7 * chi1q)
        )

    if "NNLO" in order:
        B2chi1 = B2(chi1)

        dm1 += (
            ((1 + 3 * chi1q) * m2 * (chi1 * m1 + chi2 * m2)) / r**2
            + ((1 + 3 * chi1q) * chi2 * m2 * rddot)
            - 1
            / 18
            * r
            * phidot**2
            * (
                144 * B2chi1 * (m1 + 3 * chi1q * m1)
                + 27 * (4 + 7 * chi1q) * chi2 * m2
                + 6 * chi1**3 * m1 * (-4 + 3 * b1 + 40 * b1 * b2 + 36 * sigma1)
                + 4 * chi1 * m1 * (97 - 6 * b1 + 20 * b1 * b2 + 117 * sigma1)
            )
            - (
                rdot**2
                * (
                    36 * B2chi1 * (m1 + 3 * chi1q * m1)
                    + 2 * (1 + 3 * chi1q) * chi2 * m2
                    + chi1 * m1 * (91 + 111 * sigma1 + 3 * chi1q * (-1 + 19 * sigma1))
                )
            )
            / (2 * r)
        )

        deltamnnlo = -2 * m1 * (1 + sigma1) * (3 * rdot**2 / r + r * phidot**2)
    else:
        deltamnnlo = 0.0

    return (-1.6 * m1**5 * m2**2 / (m1 + m2) ** 4 / r**7) * (chi1 * dm1 + deltamnnlo)


def j1_dot_spin_pn(m1, m2, r, rdot, phidot, chi1, chi2, order="NNLO"):
    """
    Tidal torquing for spinning BH, up to 1.5PN
    Corrected to have the right QC limit via a1, a2, a3
    factors
    """

    chi1q = chi1**2
    sigma1 = np.sqrt(1 - chi1q)

    a1 = 2
    a2 = 2
    a3 = -1

    # LO term
    dj1 = 1 + 3 * chi1q

    # 1PN term
    if "NLO" in order:
        dj1 -= (1 + 3 * chi1q) / (4 * r) * (
            r**3 * phidot**2 * 2 * m2**2
            + 2 * (6 * m1**2 + m2**2) * rdot**2 * r
            + 4 * (5 * m1 + 7 * m2)
        ) - 0.75 * r**2 * phidot**2 * (7 * chi1q + 4)

    if "NNLO" in order:
        B2chi1 = B2(chi1)

        dj1 += (
            -(1 / 18)
            * phidot
            * (
                72 * a1 * B2chi1 * (1 + 3 * chi1q) * m1
                + 27 * (4 + 7 * chi1q) * chi2 * m2
                - 3 * chi1**3 * m1 * (a2 * (3 + 40 * a3) + a1 * (4 - 36 * sigma1))
                + 2 * chi1 * m1 * (6 * a2 - 20 * a2 * a3 + a1 * (97 + 117 * sigma1))
            )
        )

        deltajnnlo = -m1 * a1 * (1 + sigma1) * phidot
    else:
        deltajnnlo = 0.0

    return (-1.6 * m1**3 * m1**2 * m2**2 / (m1 + m2) ** 4 / r**6) * (
        chi1 * dj1 + deltajnnlo
    )


def m1_dot_spin_eob(m1, m2, r, pr, pph, chi1, chi2, order="NNLO", omg=None, rdot=None):
    """
    Calculate energy flux.
    Up to 1.5PN in EOB coordinates.
    Truncation options:
    • LO        --> Only LO terms
    • NLO       --> Adds 1PN terms
    • NNLO      --> Adds 1.5PN terms
    • NNLO_fact --> Up to 1.5PN with superradiance prefactor
    Resummation options (only use for NNLO!):
    • [nothing] --> No resummation
    • CF        --> Factorize circular x non-circular terms
    • P[k][l]   --> Padé (k,l) of (non-circular) PN terms
    • CP[k][l]  --> Padé (k,l) of circular PN terms
    Superradiance prefactor options:
    • [nothing] --> Standard
    • omg       --> Using exact orbital frequency (needs to be in input)
    • rdot      --> Using exact radial velocity instead of pr (needs to be in input)
    """

    nu = m1 * m2 / (m1 + m2) ** 2

    sigma1 = np.sqrt(1.0 - chi1**2)
    omgH = 0.5 * chi1 / m1 / (1.0 + sigma1)

    u = 1.0 / r
    u2 = u * u

    factQ = False
    CQ = "CF" in order

    # Overall prefactor
    if "fact" in order:
        factQ = True
        dm1_0 = -3.2 * nu**2 * m1**4 * u2**3 * (1 + sigma1)
    else:
        dm1_0 = -1.6 * nu**2 * m1**3 * u2**3
    if CQ:
        dm1_0 *= u * np.sqrt(u)

    # Superradiance prefactor
    if factQ:
        if "omg" in order:
            if omg is not None:
                omgT_ang = omg
            else:
                raise ValueError(
                    "j1_dot_spin_eob(): give orbital frequency to use omg in H prefactor."
                )
        else:
            omgT_ang = pph * u2
        if "rdot" in order:
            if rdot is not None:
                omgT_rad = 3.0 * rdot**2 * u2 / omg
            else:
                raise ValueError(
                    "m1_dot_spin_eob(): give radial velocity to use rdot in H prefactor."
                )
        else:
            omgT_rad = 3.0 * pr**2 / pph
        H_fac = omgH - omgT_ang - omgT_rad
    else:
        H_fac = 1.0

    if CQ:
        dm1_clo = m1_dot_spin_eob_lo_circ(chi1, r, factQ)
        dm1_lo = m1_dot_spin_eob_lo_nc(r, pph)
    else:
        dm1_lo = m1_dot_spin_eob_lo(chi1, pph, r, factQ)

    if "NLO" in order:
        if CQ:
            dm1_cnlo = m1_dot_spin_eob_nlo_circ(m1, m2, r, chi1, factQ)
            dm1_nlo = m1_dot_spin_eob_nlo_nc(m1, m2, r, pr, pph, chi1)
        else:
            dm1_cnlo = 0.0
            dm1_nlo = m1_dot_spin_eob_nlo(m1, m2, r, pr, pph, chi1, factQ)
    else:
        dm1_nlo = 0.0
        dm1_cnlo = 0.0

    if "NNLO" in order:
        if CQ:
            dm1_cnnlo = m1_dot_spin_eob_nnlo_circ(m1, m2, r, chi1, chi2, factQ)
            dm1_nnlo = m1_dot_spin_eob_nnlo_nc(m1, m2, r, pr, pph, chi1, chi2, factQ)
        else:
            dm1_cnnlo = 0.0
            dm1_nnlo = m1_dot_spin_eob_nnlo(m1, m2, r, pr, pph, chi1, chi2, factQ)
    else:
        dm1_nnlo = 0.0
        dm1_cnnlo = 0.0

    # Padé resummation
    if "_P" in order:
        num, den = (eval(nstr) for nstr in order.split("_P")[1][:2])
    else:
        num, den = 3, 0
    if "CP" in order:
        cnum, cden = (eval(nstr) for nstr in order.split("CP")[1][:2])
    else:
        cnum, cden = 3, 0

    if CQ:
        pn_c_fact = PN_pade([dm1_clo, 0.0, dm1_cnlo, dm1_cnnlo], cnum, cden)
        pn_fact = PN_pade([dm1_lo, 0.0, dm1_nlo, dm1_nnlo], num, den)
    else:
        pn_c_fact = 1.0
        pn_fact = PN_pade([dm1_lo, 0.0, dm1_nlo, dm1_nnlo], num, den)

    return dm1_0 * H_fac * pn_c_fact * pn_fact


def j1_dot_spin_eob(m1, m2, r, pr, pph, chi1, chi2, order="NNLO", omg=None):
    """
    Calculate angular momentum flux.
    Up to 1.5PN in EOB coordinates.
    Truncation options:
    • LO        --> Only LO terms
    • NLO       --> Adds 1PN terms
    • NNLO      --> Adds 1.5PN terms
    • NNLO_fact --> Up to 1.5PN with superradiance prefactor
    Resummation options (only use for NNLO!):
    • [nothing] --> No resummation
    • CF        --> Factorize circular x non-circular terms
    • P[k][l]   --> Padé (k,l) of (non-circular) PN terms
    • CP[k][l]  --> Padé (k,l) of circular PN terms
    Superradiance prefactor options:
    • [nothing] --> Standard
    • omg       --> Using exact orbital frequency (needs to be in input)
    """

    nu = m1 * m2 / (m1 + m2) ** 2

    sigma1 = np.sqrt(1.0 - chi1**2)
    omgH = 0.5 * chi1 / m1 / (1.0 + sigma1)

    u = 1.0 / r
    u2 = u * u

    factQ = False
    CQ = False

    # Overall prefactor
    if "fact" in order:
        factQ = True
        dj1_0 = -3.2 * nu**2 * m1**4 * u2**3 * (1 + sigma1)
    else:
        dj1_0 = -1.6 * nu**2 * m1**3 * u2**3

    # Superradiance prefactor
    if factQ:
        if "omg" in order:
            if omg is not None:
                H_fac = omgH - omg
            else:
                raise ValueError(
                    "j1_dot_spin_eob(): give orbital frequency to use omg in H prefactor."
                )
        else:
            H_fac = omgH - pph * u2
    else:
        H_fac = 1.0

    dj1_lo = j1_dot_spin_eob_lo(chi1, factQ)

    if "NLO" in order:
        if "CF" in order:
            CQ = True
            dj1_cnlo = j1_dot_spin_eob_nlo_circ(m1, m2, r, chi1, factQ)
            dj1_nlo = j1_dot_spin_eob_nlo_nc(m1, m2, r, pr, pph, chi1)
        else:
            dj1_cnlo = 0.0
            dj1_nlo = j1_dot_spin_eob_nlo(m1, m2, r, pr, pph, chi1, chi2, factQ)
    else:
        dj1_nlo = 0.0
        dj1_cnlo = 0.0

    if "NNLO" in order:
        if CQ:
            dj1_cnnlo = j1_dot_spin_eob_nnlo_circ(m1, m2, r, chi1, chi2, factQ)
            dj1_nnlo = j1_dot_spin_eob_nnlo_nc(m1, m2, r, pr, pph, chi1, chi2, factQ)
        else:
            dj1_cnnlo = 0.0
            dj1_nnlo = j1_dot_spin_eob_nnlo(m1, m2, r, pr, pph, chi1, chi2, factQ)
    else:
        dj1_nnlo = 0.0
        dj1_cnnlo = 0.0

    # Padé resummation
    if "_P" in order:
        num, den = (eval(nstr) for nstr in order.split("_P")[1][:2])
    else:
        num, den = 3, 0
    if "CP" in order:
        cnum, cden = (eval(nstr) for nstr in order.split("CP")[1][:2])
    else:
        cnum, cden = 3, 0

    if CQ:
        pn_c_fact = PN_pade([dj1_lo, 0.0, dj1_cnlo, dj1_cnnlo], cnum, cden)
        pn_fact = PN_pade([1.0, 0.0, dj1_nlo, dj1_nnlo], num, den)
    else:
        pn_c_fact = 1.0
        pn_fact = PN_pade([dj1_lo, 0.0, dj1_nlo, dj1_nnlo], num, den)

    return dj1_0 * H_fac * pn_c_fact * pn_fact


def PN_pade(a, n, m):
    """
    Computes Padé (n, m) approximant of PN series with terms a (array)
    """

    if n < 0 or m < 0:
        raise ValueError("PN_pade(): n and m must be positive integers.")
    if n + m >= len(a):
        raise ValueError(
            "PN_pade(): n + m must be less than the length of the input array."
        )

    if n == 3 and m == 0:
        return a[0] + a[1] + a[2] + a[3]
    elif n == 1 and m == 2:
        return (
            a[1] ** 3 + a[0] * a[1] * (a[1] - 2.0 * a[2]) + a[0] ** 2 * (a[3] - a[2])
        ) / (a[1] ** 2 + a[2] ** 2 + a[0] * (a[3] - a[2]) - a[1] * (a[2] + a[3]))
    elif n == 2 and m == 1:
        return a[0] + a[1] + a[2] ** 2 / (a[2] - a[3])
    elif n == 0 and m == 3:
        return a[0] ** 4 / (
            a[0] ** 3
            - a[1] ** 3
            + a[0] * a[1] * (a[1] + 2.0 * a[2])
            - a[0] ** 2 * (a[1] + a[2] + a[3])
        )
    else:
        raise ValueError(f"PN_pade(): ({n}, {m}) Padé approximant not implemented.")


############### Auxiliary functions (order by order etc) ###############


def B2(chi):
    if abs(abs(chi) - 1.0) < 1e-7:
        B2chi1 = np.sign(chi) * np.pi / 2.0
    else:
        B2chi1 = np.imag(sp.special.digamma(3 + 2j * chi / np.sqrt(1.0 - chi**2)))
    return B2chi1


# Angular momentum flux
def j1_dot_spin_eob_lo(chi1, fact=False):
    """
    LO normalized angular momentum horizon flux in EOB coordinates.
    fact is True if using superradiance prefactor.
    Same for generic and circular orbits.
    """

    lo = 1.0 + 3.0 * chi1**2

    return lo if fact else lo * chi1


def j1_dot_spin_eob_nlo(m1, m2, r, pr, pph, chi1, chi2, fact=False):
    """
    NLO normalized angular momentum horizon flux in EOB coordinates.
    fact is True if using superradiance prefactor.
    """

    chi1q = chi1 * chi1
    u = 1.0 / r
    u2 = u * u
    nu = m1 * m2 / (m1 + m2) ** 2

    nlo = (1.0 + 3 * chi1q) * (
        pr**2 * (-3 + 5 / 2 * m2 - 11 / 2 * nu)
        - 0.5 * pph**2 * u2 * (m2 + 5 * nu)
        - u * (2 * m2 - 1 - 3 * nu)
    ) + 0.75 * pph**2 * u2 * (4 + 7 * chi1q)

    return nlo if fact else nlo * chi1


def j1_dot_spin_eob_nnlo(m1, m2, r, pr, pph, chi1, chi2, fact=False):
    """
    NNLO normalized angular momentum horizon flux in EOB coordinates.
    fact is True if using superradiance prefactor.
    """

    chi1q = chi1 * chi1
    u = 1.0 / r
    u2 = u * u
    nu = m1 * m2 / (m1 + m2) ** 2

    B2chi1 = B2(chi1)
    sigma1 = np.sqrt(1.0 - chi1q)

    lin_in_pr = -16.0 * m1 * pr * (1 + 3.0 * chi1q) * u

    if fact:
        dj1_nnlo = (
            pph
            * u2
            * (
                (
                    -8 * B2chi1 * (1 + 3 * chi1q) * m1
                    + 3 * chi1 * nu
                    + (3 / 2) * chi2 * (2 * (-2 * m2 + nu) + chi1q * (-7 * m2 + 6 * nu))
                    + (1 / 6) * chi1**3 * (-66 * m1 + 54 * nu - 72 * m1 * sigma1)
                    - (2 / 9) * chi1 * m1 * (96 + 90 * sigma1)
                )
            )
            + lin_in_pr
        )
    else:
        dj1_nnlo = chi1 * (
            pph
            * u2
            * (
                3 * nu * chi1
                - chi1 * m1 / 9 * (52 + 2 * (97 + 117 * sigma1))
                + chi1q * chi1 / 6 * (-74 * m1 + 54 * nu + m1 * 2 * (4 - 36 * sigma1))
                + 1.5 * chi2 * (2 * (-2 * m2 + nu) + (-7 * m2 + 6 * nu) * chi1q)
                - 8 * m1 * (1 + 3 * chi1q) * B2chi1
            )
            + lin_in_pr
        ) - 2.0 * pph * u2 * m1 * (1.0 + sigma1)

    return dj1_nnlo


def j1_dot_spin_eob_nlo_circ(m1, m2, r, chi1, fact=False):
    """
    NLO normalized angular momentum horizon flux in EOB coordinates on circular orbits.
    fact is True if using superradiance prefactor.
    """

    nu = m1 * m2 / (m1 + m2) ** 2

    nlo = 0.5 / r * (8.0 + 16.5 * chi1**2 + (nu - 2.5 * m2) * (1.0 + 3.0 * chi1**2))

    return nlo if fact else nlo * chi1


def j1_dot_spin_eob_nnlo_circ(m1, m2, r, chi1, chi2, fact=False):
    """
    NNLO normalized angular momentum horizon flux in EOB coordinates on circular orbits.
    fact is True if using superradiance prefactor.
    """

    nu = m1 * m2 / (m1 + m2) ** 2
    u = 1.0 / r
    usq = np.sqrt(u)
    u15 = usq * u

    chi1q = chi1 * chi1
    chi1qf = 1.0 + 3.0 * chi1q
    sigma1 = np.sqrt(1.0 - chi1q)

    nnlo = (
        u15
        / 12.0
        * (
            -4.0
            * m1
            * (
                82.0 * chi1
                + 33.0 * chi1 * chi1q
                + 6.0 * sigma1 * chi1 * (13.0 + 6.0 * chi1q)
            )
            + 18.0
            * (-m2 * (4.0 + 7.0 * chi1q) * chi2 + 2.0 * nu * chi1qf * (chi1 + chi2))
            - 96.0 * m1 * chi1qf * B2(chi1)
        )
    )

    if not fact:
        nnlo = nnlo * chi1 - 2.0 * u15 * m1 * (1.0 + sigma1)

    return nnlo


def j1_dot_spin_eob_nlo_nc(m1, m2, r, pr, pph, chi1):
    """
    NLO non-circular correction to normalized angular momentum horizon flux in EOB coordinates.
    """

    u = 1.0 / r
    u2 = u * u
    pr2 = pr * pr

    nu = m1 * m2 / (m1 + m2) ** 2

    chi1q = chi1 * chi1
    chi1qf = 1.0 + 3.0 * chi1q

    dj1nc_nlo = (
        0.25
        * u2
        * (
            r
            * (
                -7.0
                - 5.0 / chi1qf
                + 10.0 * nu
                + 2.0 * m2 * (1.0 + 5.0 * pr2 * r)
                - 2.0 * pr2 * r * (6.0 + 11.0 * nu)
            )
            - pph**2 * (10.0 * nu + 2.0 * m2 - 3.0 * (4.0 + 7.0 * chi1q) / chi1qf)
        )
    )

    return dj1nc_nlo


def j1_dot_spin_eob_nnlo_nc(m1, m2, r, pr, pph, chi1, chi2, fact=False):
    """
    NNLO non-circular correction to normalized angular momentum horizon flux in EOB coordinates.
    """

    rsq = np.sqrt(r)
    u = 1.0 / r
    u2 = u * u

    nu = m1 * m2 / (m1 + m2) ** 2

    chi1q = chi1 * chi1
    sigma1 = np.sqrt(1.0 - chi1q)
    chi1qf = 1.0 + 3.0 * chi1q

    dj1nc_nnlo = (
        u2
        / 6.0
        * (
            -2.0
            * m1
            * (
                (pph - rsq)
                * (
                    chi1 * (64.0 + 33.0 * chi1q)
                    + 12.0 * sigma1 * chi1 * (5.0 + 3.0 * chi1q)
                )
                / chi1qf
                + 48.0 * pr * r
            )
            + (pph - rsq)
            * (
                9.0
                * (-m2 * (4.0 + 7.0 * chi1q) * chi2 / chi1qf + 2.0 * nu * (chi1 + chi2))
                - 48.0 * m1 * B2(chi1)
            )
        )
    )

    if not fact:
        dj1nc_nnlo += -2.0 * u2 / chi1 * m1 * (1.0 + sigma1) * (pph - rsq)

    return dj1nc_nnlo


# Energy flux
def m1_dot_spin_eob_lo(chi1, pph, r, fact=False):
    """
    LO normalized horizon energy flux in EOB coordinates.
    fact is True if using superradiance prefactor.
    """

    lo = (1.0 + 3.0 * chi1 * chi1) * pph / r**2

    return lo if fact else lo * chi1


def m1_dot_spin_eob_nlo(m1, m2, r, pr, pph, chi1, fact=False):
    """
    NLO normalized horizon energy flux in EOB coordinates.
    fact is True if using superradiance prefactor.
    """

    nu = m1 * m2 / (m1 + m2) ** 2
    u2 = 1.0 / r**2
    chi1q = chi1**2
    pr2 = pr**2

    nlo = (
        (
            (3 * chi1q * (3 + 2 * m1**2 - 14 * nu) + 2 * (4 + m1**2 - 7 * nu)) * pph**3
            - 2
            * (1 + 3 * chi1q)
            * pph
            * r
            * (4 - 8 * nu + 13 * nu * pr2 * r + m1 * (-4 + (6 + m1) * pr2 * r))
        )
        * u2**2
        / 4
    )
    return nlo if fact else nlo * chi1


def m1_dot_spin_eob_nnlo(m1, m2, r, pr, pph, chi1, chi2, fact=False):
    """
    NNLO normalized horizon energy flux in EOB coordinates.
    fact is True if using superradiance prefactor.
    """

    nu = m1 * m2 / (m1 + m2) ** 2
    chi1q = chi1 * chi1
    sigma1 = np.sqrt(1.0 - chi1q)
    u = 1.0 / r
    u2 = u * u
    pr2 = pr * pr
    pph2 = pph * pph

    # Terms linear in pr are common to fact and non-fact
    lin_in_pr = -64 / 3 * m1 * pph * pr * (1 + 3 * chi1q) * u2 * u

    if fact:
        nnlo = (1 / 18 * u2**2) * (
            36 * chi1 * (1.0 + 3 * chi1q) * (m1 - nu) * r
            - 36 * B2(chi1) * (1 + 3 * chi1q) * m1 * (4 * pph2 + 9 * pr2 * r**2)
            + 9
            * chi2
            * m2
            * (
                -5 * (2 + 3 * chi1q) * pph2
                - 2 * (1 + 3 * chi1q) * r * (-2 + 3 * m1 + pr2 * r)
            )
            + m1
            * (
                2
                * pph2
                * (
                    27 * chi2
                    - 27 * chi2 * (-3 * chi1q * m2 + m1)
                    - 3 * chi1q * chi1 * (6 + 27 * m1 + 36 * sigma1)
                    - chi1 * (165 + 27 * m1 + 180 * sigma1)
                )
                + 9
                * r
                * (
                    (1 + 3 * chi1q) * (5 * chi1 + 3 * chi2) * m2
                    - chi1 * pr2 * r * (55 - 3 * chi1q + (75 + 57 * chi1q) * sigma1)
                )
            )
        ) + lin_in_pr
    else:
        nnlo = (
            (1 / 18 * u2**2)
            * (
                -36 * B2(chi1) * (1 + 3 * chi1q) * m1 * (4 * pph2 + 9 * pr2 * r**2)
                + 9
                * chi2
                * (
                    (2 * (-5 + 5 * m1 + 3 * nu) + 3 * chi1q * (-5 + 5 * m1 + 6 * nu))
                    * pph2
                    + (1 + 3 * chi1q) * r * (-3 * nu + 2 * (-1 + m1) * (-2 + pr2 * r))
                )
                + chi1
                * (
                    9 * (1 + 3 * chi1q) * nu * (6 * pph2 + r)
                    + 2
                    * m1
                    * pph2
                    * (-3 * chi1q * (33 + 36 * sigma1) - 2 * (123 + 117 * sigma1))
                    + 9
                    * m1
                    * r
                    * (
                        4
                        + 12 * chi1q
                        - pr2 * r * (91 + 111 * sigma1 + 3 * chi1q * (-1 + 19 * sigma1))
                    )
                )
            )
            + lin_in_pr
        ) * chi1 - 2 * m1 * (1 + sigma1) * (pph2 * u2 + 3 * pr2) * u2
    return nnlo


def m1_dot_spin_eob_lo_circ(chi1, r, fact=False):
    """
    LO normalized energy flux in EOB coordinates on circular orbits.
    fact is True if using superradiance prefactor.
    """

    lo = 1.0 + 3.0 * chi1 * chi1

    return lo if fact else lo * chi1


def m1_dot_spin_eob_nlo_circ(m1, m2, r, chi1, fact):
    """
    NLO normalized energy flux in EOB coordinates on circular orbits.
    fact is True if using superradiance prefactor.
    """

    u = 1.0 / r
    chi1q = chi1 * chi1
    chi1qf = 1.0 + 3.0 * chi1q

    nlo = (
        0.25
        * u
        * (-20.0 * m1 * chi1qf - 3.0 * (-12.0 - 31.0 * chi1q + 10.0 * m2 * chi1qf))
    )

    return nlo if fact else nlo * chi1


def m1_dot_spin_eob_nnlo_circ(m1, m2, r, chi1, chi2, fact=False):
    """
    NNLO normalized energy flux in EOB coordinates on circular orbits.
    fact is True if using superradiance prefactor.
    """

    nu = m1 * m2 / (m1 + m2) ** 2
    u = 1.0 / r
    u15 = np.sqrt(u) * u
    chi1q = chi1**2
    chi1qf = 1.0 + 3.0 * chi1q
    sigma1 = np.sqrt(1.0 - chi1q)

    nnlo = (
        u15
        / 12.0
        * (
            -12.0 * m1**2 * chi1 * chi1qf
            - 4.0
            * m1
            * (chi1 * (82.0 + 78.0 * sigma1 + chi1q * (33.0 + 36.0 * sigma1)))
            + 3.0
            * (
                -6.0 * m2 * (4.0 + 7.0 * chi1q) * chi2
                + nu * chi1qf * (13.0 * chi1 + 9.0 * chi2)
            )
            - 96.0 * m1 * chi1qf * B2(chi1)
        )
    )

    if not fact:
        nnlo = chi1 * nnlo - 2.0 * u15 * m1 * (1.0 + sigma1)

    return nnlo


def m1_dot_spin_eob_lo_nc(r, pph):
    """
    LO non-circular correction to normalized horizon energy flux in EOB coordinates.
    """

    return pph / np.sqrt(r)


def m1_dot_spin_eob_nlo_nc(m1, m2, r, pr, pph, chi1):
    """
    NLO non-circular correction to normalized horizon energy flux in EOB coordinates.
    """

    nu = m1 * m2 / (m1 + m2) ** 2
    chi1qf = 1.0 + 3.0 * chi1**2
    u = 1.0 / r
    u15 = np.sqrt(u) * u

    nlo = (
        -0.25
        * pph
        * u15
        * u
        * (
            r
            * (
                5.0 / chi1qf
                + 11.0
                - 18.0 * nu
                + 2.0 * pr**2 * r * (1.0 + 6.0 * m1**2 + 17.0 * nu - m2**2)
                - 2.0 * m2**2
            )
            + pph**2 * (18.0 * nu - 5.0 - 5.0 / chi1qf + 2 * m2**2)
        )
    )
    return nlo


def m1_dot_spin_eob_nnlo_nc(m1, m2, r, pr, pph, chi1, chi2, fact=False):
    """
    NNLO non-circular correction to normalized horizon energy flux in EOB coordinates.
    """

    rsq = np.sqrt(r)
    u = 1.0 / r
    u2 = u * u
    r2 = r * r
    u15 = u / rsq
    pr2 = pr**2
    pph2 = pph**2

    nu = m1 * m2 / (m1 + m2) ** 2

    chi1q = chi1 * chi1
    sigma1 = np.sqrt(1.0 - chi1q)
    chi1qf = 1.0 + 3.0 * chi1q

    nnlo = (
        u15
        * u
        / 12.0
        * (
            12 * m1**2 * (pph + 2 * rsq) * rsq * chi1
            - 2
            * m1
            * (
                128.0 * pr * pph * r
                + 2.0
                * pph
                * (pph - rsq)
                * chi1
                * (64.0 + 33.0 * chi1q + 12.0 * sigma1 * (5.0 + 3.0 * chi1q))
                / chi1qf
                + 3.0
                * pr2
                * r2
                * chi1
                * (55.0 - 3.0 * chi1q + sigma1 * (75.0 + 57.0 * chi1q))
                / chi1qf
            )
            + 3
            * (
                2
                * pph2
                * (-5 * m2 * (2 + 3 * chi1q) * chi2 / chi1qf + 6 * nu * (chi1 + chi2))
                + 2
                * r
                * (2 * m2 * (-1 + 3 * m2 - pr2 * r) * chi2 + nu * (5 * chi1 + 3 * chi2))
                - pph
                * rsq
                * (
                    -6 * m2 * (4 + 7 * chi1q) * chi2 / chi1qf
                    + nu * (13 * chi1 + 9 * chi2)
                )
            )
            - 24 * m1 * (4 * pph2 - 4 * pph * rsq + 9 * pr2 * r2) * B2(chi1)
        )
    )
    if not fact:
        nnlo += (
            -2.0
            * m1
            * u15
            * u
            / chi1
            * (1.0 + sigma1)
            * (pph2 - pph * rsq + 3.0 * pr2 * r2)
        )
    return nnlo
