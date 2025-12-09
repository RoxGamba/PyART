"""
Horizon fluxes in EOB coordinates on generic orbits, order by order.
Also including circular-non-circular factorization.
"""

import numpy as np
import scipy as sp
pi = np.pi

def B2(chi):
    if abs(abs(chi) - 1.) < 1e-7:
        B2chi1 = np.sign(chi)*np.pi/2.
    else:
        B2chi1 = np.imag(sp.special.digamma(3 + 2j*chi/np.sqrt(1. - chi**2)))
    return B2chi1

# Angular momentum flux
def j1_dot_spin_eob_lo(chi1, fact=False):
    """
    LO normalized angular momentum horizon flux in EOB coordinates.
    fact is True if using superradiance prefactor.
    Same for generic and circular orbits.
    """

    lo = 1. + 3.*chi1**2

    return lo if fact else lo*chi1

def j1_dot_spin_eob_nlo(m1, m2, r, pr, pph, chi1, chi2, fact=False):
    """
    NLO normalized angular momentum horizon flux in EOB coordinates.
    fact is True if using superradiance prefactor.
    """

    chi1q = chi1*chi1
    u     = 1./r
    u2    = u*u
    nu    = m1*m2/(m1 + m2)**2
    
    nlo = (1. + 3*chi1q)*(pr**2*(-3 + 5/2*m2 - 11/2*nu) - \
                                0.5*pph**2*u2*(m2 + 5*nu) - \
                                u*(2*m2 - 1 - 3*nu)) + \
                0.75*pph**2*u2*(4 + 7*chi1q)
    
    return nlo if fact else nlo*chi1

def j1_dot_spin_eob_nnlo(m1, m2, r, pr, pph, chi1, chi2, fact=False):
    """
    NNLO normalized angular momentum horizon flux in EOB coordinates.
    fact is True if using superradiance prefactor.
    """

    chi1q  = chi1*chi1
    u      = 1./r
    u2     = u*u
    nu     = m1*m2/(m1 + m2)**2

    B2chi1 = B2(chi1)
    sigma1 = np.sqrt(1. - chi1q)

    lin_in_pr = -16.*m1*pr*(1 + 3.*chi1q)*u

    if fact:
        dj1_nnlo = pph*u2*((-8 * B2chi1 * (1 + 3 * chi1q) * m1 + 3 * chi1 * nu + \
                            (3/2) * chi2 * (2 * (-2*m2 + nu) + chi1q * (-7*m2 + 6 * nu)) + \
                            (1/6) * chi1**3 * (-66 * m1 + 54 * nu - 72 * m1 * sigma1) - \
                            (2/9) * chi1 * m1 * (96 + 90 * sigma1))) + lin_in_pr
    else:
        dj1_nnlo = chi1*(pph*u2*(3*nu*chi1 - chi1*m1/9*(52 + 2*(97 + 117*sigma1)) + \
                        chi1q*chi1/6*(-74*m1 + 54*nu + m1*2*(4 - 36*sigma1)) + \
                        1.5*chi2*(2*(-2*m2 + nu) + (-7*m2 + 6*nu)*chi1q) - 8*m1*(1 + 3*chi1q)*B2chi1) + \
                        lin_in_pr) - 2.*pph*u2*m1*(1. + sigma1)
    
    return dj1_nnlo

def j1_dot_spin_eob_nlo_circ(m1, m2, r, chi1, fact=False):
    """
    NLO normalized angular momentum horizon flux in EOB coordinates on circular orbits.
    fact is True if using superradiance prefactor.
    """

    nu = m1*m2/(m1 + m2)**2

    nlo = 0.5/r*(8. + 16.5*chi1**2 + (nu - 2.5*m2)*(1. + 3.*chi1**2))

    return nlo if fact else nlo*chi1

def j1_dot_spin_eob_nnlo_circ(m1, m2, r, chi1, chi2, fact=False):
    """
    NNLO normalized angular momentum horizon flux in EOB coordinates on circular orbits.
    fact is True if using superradiance prefactor.
    """

    nu    = m1*m2/(m1 + m2)**2
    u     = 1./r
    usq   = np.sqrt(u)
    u15   = usq*u
    
    chi1q  = chi1*chi1
    chi1qf = 1. + 3.*chi1q
    sigma1 = np.sqrt(1. - chi1q)

    nnlo  = u15/12.*(- 4.*m1*(82.*chi1 + 33.*chi1*chi1q + 6.*sigma1*chi1*(13. + 6.*chi1q)) \
                              + 18.*(-m2*(4. + 7.*chi1q)*chi2 + 2.*nu*chi1qf*(chi1 + chi2)) \
                              - 96.*m1*chi1qf*B2(chi1))
    
    if not fact:
        nnlo = nnlo*chi1 -2.*u15*m1*(1. + sigma1)
    
    return nnlo

def j1_dot_spin_eob_nlo_nc(m1, m2, r, pr, pph, chi1):
    """
    NLO non-circular correction to normalized angular momentum horizon flux in EOB coordinates.
    """

    u   = 1./r
    u2  = u*u
    pr2 = pr*pr

    nu  = m1*m2/(m1 + m2)**2

    chi1q  = chi1*chi1
    chi1qf = 1. + 3.*chi1q

    dj1nc_nlo = 0.25*u2*(r*(-7. -5./chi1qf + 10.*nu + 2.*m2*(1. + 5.*pr2*r) - 2.*pr2*r*(6. + 11.*nu)) \
                             -pph**2*(10.*nu + 2.*m2 - 3.*(4. + 7.*chi1q)/chi1qf))
        
    return dj1nc_nlo

def j1_dot_spin_eob_nnlo_nc(m1, m2, r, pr, pph, chi1, chi2, fact=False):
    """
    NNLO non-circular correction to normalized angular momentum horizon flux in EOB coordinates.
    """

    rsq = np.sqrt(r)
    u   = 1./r
    u2  = u*u

    nu  = m1*m2/(m1 + m2)**2

    chi1q  = chi1*chi1
    sigma1 = np.sqrt(1. - chi1q)
    chi1qf = 1. + 3.*chi1q

    dj1nc_nnlo = u2/6.*(-2.*m1*((pph - rsq)*(chi1*(64. + 33.*chi1q) + 12.*sigma1*chi1*(5. + 3.*chi1q))/chi1qf + 48.*pr*r) \
                                 + (pph - rsq)*(9.*(-m2*(4. + 7.*chi1q)*chi2/chi1qf + 2.*nu*(chi1 + chi2)) - 48.*m1*B2(chi1)))

    if not fact:
        dj1nc_nnlo += -2.*u2/chi1*m1*(1. + sigma1)*(pph - rsq)

    return dj1nc_nnlo

# Energy flux
def m1_dot_spin_eob_lo(chi1, pph, r, fact=False):
    """
    LO normalized horizon energy flux in EOB coordinates.
    fact is True if using superradiance prefactor.
    """

    lo = (1. + 3.*chi1*chi1)*pph/r**2

    return lo if fact else lo*chi1

def m1_dot_spin_eob_nlo(m1, m2, r, pr, pph, chi1, fact=False):
    """
    NLO normalized horizon energy flux in EOB coordinates.
    fact is True if using superradiance prefactor.
    """

    nu    = m1*m2/(m1 + m2)**2
    u2    = 1./r**2
    chi1q = chi1**2
    pr2   = pr**2

    nlo = ((3 * chi1q * (3 + 2 * m1**2 - 14 * nu) + 2 * (4 + m1**2 - 7 * nu)) * pph**3 -  \
                    2 * (1 + 3 * chi1q) * pph * r * (4 - 8 * nu + 13 * nu * pr2 * r + \
                                                    m1 * (-4 + (6 + m1) * pr2 * r))) * u2**2 / 4
    return nlo if fact else nlo*chi1

def m1_dot_spin_eob_nnlo(m1, m2, r, pr, pph, chi1, chi2, fact=False):
    """
    NNLO normalized horizon energy flux in EOB coordinates.
    fact is True if using superradiance prefactor.
    """

    nu     = m1*m2/(m1 + m2)**2
    chi1q  = chi1*chi1
    sigma1 = np.sqrt(1. - chi1q)
    u      = 1./r
    u2     = u*u
    pr2    = pr*pr
    pph2   = pph*pph

    # Terms linear in pr are common to fact and non-fact
    lin_in_pr = -64/3*m1*pph*pr*(1 + 3*chi1q)*u2*u

    if fact:
        nnlo = (1/18 * u2**2) * (36 * chi1*(1. + 3 * chi1q) * (m1 - nu) * r - 36 * B2(chi1) * (1 + 3 * chi1q) * m1 * (4 * pph2 + 9 * pr2 * r**2) + \
                                 9 * chi2 * m2 * (-5 * (2 + 3 * chi1q) * pph2 - 2 * (1 + 3 * chi1q) * r * (-2 + 3 * m1 + pr2 * r)) + \
                                 m1 * (2 * pph2 * (27 * chi2 - 27 * chi2 * (-3 * chi1q * m2 + m1) - 3 * chi1q*chi1 * (6 + 27 * m1 + 36 * sigma1) - \
                                 chi1 * (165 + 27 * m1 + 180 * sigma1)) + 9 * r * ((1 + 3 * chi1q) * (5 * chi1 + 3 * chi2) * m2 - \
                                 chi1 * pr2 * r * (55 - 3 * chi1q + (75 + 57 * chi1q) * sigma1)))) + lin_in_pr
    else:
        nnlo = ((1/18 * u2**2) * (-36 * B2(chi1) * (1 + 3 * chi1q) * m1 * (4 * pph2 + 9 * pr2 * r**2) + \
                                 9 * chi2 * ((2 * (-5 + 5 * m1 + 3 * nu) + 3 * chi1q * (-5 + 5 * m1 + 6 * nu)) * pph2 + \
                                 (1 + 3 * chi1q) * r * (-3 * nu + 2 * (-1 + m1) * (-2 + pr2 * r))) + \
                                 chi1 * (9 * (1 + 3 * chi1q) * nu * (6 * pph2 + r) + \
                                 2 * m1 * pph2 * (-3 * chi1q * (33 + 36 * sigma1) - \
                                 2 * (123 + 117 * sigma1)) + \
                                 9 * m1 * r * (4 + 12 * chi1q - pr2 * r * (91 + 111 * sigma1 + 3 * chi1q * (-1 + 19 * sigma1))))) + \
                                 lin_in_pr)*chi1 - 2*m1*(1 + sigma1)*(pph2*u2 + 3*pr2)*u2
    return nnlo

def m1_dot_spin_eob_lo_circ(chi1, r, fact=False):
    """
    LO normalized energy flux in EOB coordinates on circular orbits.
    fact is True if using superradiance prefactor.
    """

    lo = (1. + 3.*chi1*chi1)
    
    return lo if fact else lo*chi1

def m1_dot_spin_eob_nlo_circ(m1, m2, r, chi1, fact):
    """
    NLO normalized energy flux in EOB coordinates on circular orbits.
    fact is True if using superradiance prefactor.
    """

    u      = 1./r
    chi1q  = chi1*chi1
    chi1qf = 1. + 3.*chi1q

    nlo = 0.25*u*(-20.*m1*chi1qf - 3.*(-12. - 31.*chi1q + 10.*m2*chi1qf))

    return nlo if fact else nlo*chi1

def m1_dot_spin_eob_nnlo_circ(m1, m2, r, chi1, chi2, fact=False):
    """
    NNLO normalized energy flux in EOB coordinates on circular orbits.
    fact is True if using superradiance prefactor.
    """

    nu     = m1*m2/(m1 + m2)**2
    u      = 1./r
    u15    = np.sqrt(u)*u
    chi1q  = chi1**2
    chi1qf = 1. + 3.*chi1q
    sigma1 = np.sqrt(1. - chi1q)

    nnlo = u15/12.*(-12.*m1**2*chi1*chi1qf - 4.*m1*(chi1*(82. + 78.*sigma1 + chi1q*(33. + 36.*sigma1))) \
                             +3.*(-6.*m2*(4. + 7.*chi1q)*chi2 + nu*chi1qf*(13.*chi1 + 9.*chi2)) - 96.*m1*chi1qf*B2(chi1))
    
    if not fact:
        nnlo = chi1*nnlo - 2.*u15*m1*(1. + sigma1)
    
    return nnlo

def m1_dot_spin_eob_lo_nc(r, pph):
    """
    LO non-circular correction to normalized horizon energy flux in EOB coordinates.
    """

    return pph/np.sqrt(r)

def m1_dot_spin_eob_nlo_nc(m1, m2, r, pr, pph, chi1):
    """
    NLO non-circular correction to normalized horizon energy flux in EOB coordinates.
    """

    nu     = m1*m2/(m1 + m2)**2
    chi1qf = 1. + 3.*chi1**2
    u      = 1./r
    u15    = np.sqrt(u)*u

    nlo = -0.25*pph*u15*u*(r*(5./chi1qf + 11. - 18.*nu + 2.*pr**2*r*(1. + 6.*m1**2 + 17.*nu - m2**2) - 2.*m2**2) \
                                     + pph**2*(18.*nu - 5. - 5./chi1qf + 2*m2**2))
    return nlo

def m1_dot_spin_eob_nnlo_nc(m1, m2, r, pr, pph, chi1, chi2, fact=False):
    """
    NNLO non-circular correction to normalized horizon energy flux in EOB coordinates.
    """

    rsq  = np.sqrt(r)
    u    = 1./r
    u2   = u*u
    r2   = r*r
    u15  = u/rsq
    pr2  = pr**2
    pph2 = pph**2

    nu  = m1*m2/(m1 + m2)**2

    chi1q  = chi1*chi1
    sigma1 = np.sqrt(1. - chi1q)
    chi1qf = 1. + 3.*chi1q

    nnlo = u15*u/12.*(12 * m1**2 * (pph + 2 * rsq) * rsq * chi1 \
                                    - 2 * m1 * (128. * pr * pph * r + 2. * pph * (pph - rsq) * chi1 * (64. + 33. * chi1q + 12. * sigma1 * (5. + 3. * chi1q))/chi1qf \
                                    + 3. * pr2 * r2 * chi1 * (55. - 3. * chi1q + sigma1 * (75. + 57. * chi1q))/chi1qf) \
                                    + 3 * (2 * pph2 * (-5 * m2 * (2 + 3 * chi1q) * chi2/chi1qf + 6 * nu * (chi1 + chi2)) \
                                    + 2 * r * (2 * m2 * (-1 + 3 * m2 - pr2 * r) * chi2 + nu * (5 * chi1 + 3 * chi2)) \
                                    - pph * rsq * (-6 * m2 * (4 + 7 * chi1q) * chi2/chi1qf + nu * (13 * chi1 + 9 * chi2))) \
                                    - 24 * m1 * (4 * pph2 - 4 * pph * rsq + 9 * pr2 * r2) * B2(chi1))
    if not fact:
        nnlo += -2.*m1*u15*u/chi1*(1. + sigma1)*(pph2 - pph*rsq + 3.*pr2*r2)
    return nnlo