import numpy as np
import scipy as sp
import pn_eob_fluxes as pneob
pi = np.pi

## PN functions for tidal heating/torquing in harmonic coordinates
def m1_dot(m1, m2, r, rdot, phidot):
    """
    Tidal heating for non-spinning BH, to LO
    """

    return (32*m1**6 * m2**2 *(3*rdot**2 + r**2 * phidot**2))/(5*r**8)

def j1_dot(m1, m2, r, rdot, phidot):
    """
    Tidal torquing for non-spinning BH, to LO
    From rigid rotation relation
    """

    return m1_dot(m1, m2, r, rdot, phidot)/phidot

def mj1_dot_QC_NNLO(m1, m2, x, r, chi1, chi2, lo='x'):
    """
    Tidal torquing for BH, to NNLO
    From Saketh+, https://arxiv.org/pdf/2212.13095
    
    Note: they use x = (Momg_orb)^1/3
    """
    M      = m1 + m2
    nu     = m1*m2/M**2
    X1     = m1/M; X2 = m2/M
    sigma1 = np.sqrt(1 - chi1**2)
    B2chi1 = pneob.B2(chi1)
    Omg    = x**3 # orbital frequency
    OmgH   = chi1/(2*m1*(1+sigma1))

    # LO
    fact   = -16./5*m1**4*nu**2*(1+sigma1)
    c_LO   = 1 + 3*chi1**2
    c_NLO  = 0.25*(3*(2+chi1**2) + 2* X1 * c_LO*(2 + 3*X1))
    c_NNLO = 0.5*(-4 +3*chi1**2)*chi2 - 2*X1*c_LO*(X1*(chi1 + chi2) + 4*B2chi1) + \
            X1*(-2./3*(23+30*sigma1)*chi1 + (7 - 12*sigma1)*chi1**3 + 4*chi2 + 9/2*chi1**2*chi2)
    
    if lo == 'x':
        lo_f = x**12
    elif lo == 'r':
        lo_f = 1/r**6
    else:
        raise ValueError("LO spec not recognized; only x and r supported")

    j1dot  = (OmgH-Omg)*fact*lo_f*(c_LO + x**2*c_NLO + x**3*c_NNLO)

    return Omg*j1dot, j1dot



def m1_dot_spin_pn(m1, m2, r, rdot, phidot, chi1, chi2, order='NNLO'):
    """
    Tidal heating for spinning BH, up to 1.5PN
    Eq. (32) from notes
    Two versions for NNLO (1.5PN):
    • b1 = b2 = 1    ---------> Saketh formulas exactly                ('NNLO', default)
    • b1 = b2 = -1   ---------> Corrected to recover correct QC limit  ('NNLO_mod')
    TODO: update to remove b's
    """

    chi1q  = chi1**2
    sigma1 = np.sqrt(1 - chi1q)

    if order == 'NNLO_mod':
        b1 = -1; b2 = -1
    else:
        b1 = 1; b2 = 1

    # LO expressions for second time derivatives are enough
    phiddot = -2*phidot*rdot/r
    rddot   = r*phidot**2 - 1/r**2

    # LO term
    dm1 = r*phidot*(1 + 3*chi1q)

    #1PN term
    if 'NLO' in order:
        dm1 -= 0.25*(2*(1 + 3*chi1q)*phidot*(m2*(2*m1 + m2)*r**2*rddot + (6*m1**2 + 2*m2*m1 + m2**2)*rdot**2*r + \
                                           3*m2*(m2 + 4) + 2*m1*(2*m2 + 5)) + 2*m2*(2*m1 + m2)*r**2*rdot*phiddot*(1 + 3*chi1q) - \
                                          3*r**3*phidot**3*(4 + 7*chi1q))
        
    if 'NNLO' in order:
        B2chi1 = pneob.B2(chi1)

        dm1 += ((1 + 3 * chi1q) * m2 * (chi1 * m1 + chi2 * m2)) / r**2 + \
               ((1 + 3 * chi1q) * chi2 * m2 * rddot) - \
                 1/18 * r*phidot**2 * (144 * B2chi1 * (m1 + 3 * chi1q * m1) + 27 * (4 + 7 * chi1q) * chi2 * m2 + \
                              6 * chi1**3 * m1 * (-4 + 3 * b1 + 40 * b1 * b2 + 36 * sigma1) + 4 * chi1 * m1 * (97 - 6 * b1 + 20 * b1 * b2 + 117 * sigma1)) - \
               (rdot**2 * (36 * B2chi1 * (m1 + 3 * chi1q * m1) + 2 * (1 + 3 * chi1q) * chi2 * m2 + \
                      chi1 * m1 * (91 + 111 * sigma1 + 3 * chi1q * (-1 + 19 * sigma1)))) / (2 * r)
        
        deltamnnlo = -2*m1*(1 + sigma1)*(3*rdot**2/r + r*phidot**2)
    else:
        deltamnnlo = 0.
    
    return (-1.6*m1**5*m2**2/(m1 + m2)**4/r**7)*(chi1*dm1 + deltamnnlo)

def j1_dot_spin_pn(m1, m2, r, rdot, phidot, chi1, chi2, order='NNLO'):
    """
    Tidal torquing for spinning BH, up to 1.5PN
    Eq. (32) from notes
    The coefficients a1,2,3 enter the 1.5PN part, and modify eq. (4.6) of [https://arxiv.org/pdf/2212.13095]:
    • a1 = a2 = a3 = 1     --> Exactly as in paper                 ('NNLO')
    • a1 = a2 = 2, a3 = -1 --> Changes to recover correct QC limit ('NNLO_mod')
    TODO: update to remove a's
    """

    chi1q  = chi1**2
    sigma1 = np.sqrt(1 - chi1q)

    if order == 'NNLO_mod':
        a1 = 2; a2 = 2; a3 = -1
    else:
        a1 = 1; a2 = 1; a3 = 1

    # LO term
    dj1 = 1 + 3*chi1q

    # 1PN term
    if 'NLO' in order:
        dj1 -= (1 + 3*chi1q)/(4*r)*(r**3*phidot**2*2*m2**2 + 2*(6*m1**2 + m2**2)*rdot**2*r + \
                                       4*(5*m1 + 7*m2)) - 0.75*r**2*phidot**2*(7*chi1q + 4)
    
    if 'NNLO' in order:
        B2chi1 = pneob.B2(chi1)

        dj1 += -(1/18) * phidot * (72 * a1 * B2chi1 * (1 + 3 * chi1q) * m1 + 27 * (4 + 7 * chi1q) * chi2 * m2 - \
                                   3 * chi1**3 * m1 * (a2 * (3 + 40 * a3) + a1 * (4 - 36 * sigma1)) + \
                                   2 * chi1 * m1 * (6 * a2 - 20 * a2 * a3 + a1 * (97 + 117 * sigma1)))
        
        deltajnnlo = -m1*a1*(1 + sigma1)*phidot
    else:
        deltajnnlo = 0.

    return (-1.6*m1**3*m1**2*m2**2/(m1 + m2)**4/r**6)*(chi1*dj1 + deltajnnlo)

def m1_dot_spin_eob(m1, m2, r, pr, pph, chi1, chi2, order='NNLO', omg=None, rdot=None):
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

    nu = m1*m2/(m1 + m2)**2

    sigma1 = np.sqrt(1. - chi1**2)
    omgH = 0.5*chi1/m1/(1. + sigma1)

    u  = 1./r
    u2 = u*u

    factQ = False
    CQ    = 'CF' in order

    # Overall prefactor
    if 'fact' in order:
        factQ = True
        dm1_0 = -3.2*nu**2*m1**4*u2**3*(1 + sigma1)
    else:
        dm1_0 = -1.6*nu**2*m1**3*u2**3
    if CQ:
        dm1_0 *= u*np.sqrt(u)

    # Superradiance prefactor
    if factQ:
        if 'omg' in order:
            if omg is not None:
                omgT_ang = omg
            else:
                raise ValueError("j1_dot_spin_eob(): give orbital frequency to use omg in H prefactor.")
        else:
            omgT_ang = pph*u2
        if 'rdot' in order:
            if rdot is not None:
                omgT_rad = 3.*rdot**2*u2/omg
            else:
                raise ValueError("m1_dot_spin_eob(): give radial velocity to use rdot in H prefactor.")
        else:
            omgT_rad = 3.*pr**2/pph
        H_fac = omgH - omgT_ang - omgT_rad
    else:
        H_fac = 1.
    
    if CQ:
        dm1_clo = pneob.m1_dot_spin_eob_lo_circ(chi1, r, factQ)
        dm1_lo  = pneob.m1_dot_spin_eob_lo_nc(r, pph)
    else:
        dm1_lo = pneob.m1_dot_spin_eob_lo(chi1, pph, r, factQ)

    if 'NLO' in order:
        if CQ:
            dm1_cnlo = pneob.m1_dot_spin_eob_nlo_circ(m1, m2, r, chi1, factQ)
            dm1_nlo  = pneob.m1_dot_spin_eob_nlo_nc(m1, m2, r, pr, pph, chi1)
        else:
            dm1_cnlo = 0.
            dm1_nlo  = pneob.m1_dot_spin_eob_nlo(m1, m2, r, pr, pph, chi1, factQ)
    else:
        dm1_nlo  = 0.
        dm1_cnlo = 0.
    
    if 'NNLO' in order:
        if CQ:
            dm1_cnnlo = pneob.m1_dot_spin_eob_nnlo_circ(m1, m2, r, chi1, chi2, factQ)
            dm1_nnlo  = pneob.m1_dot_spin_eob_nnlo_nc(m1, m2, r, pr, pph, chi1, chi2, factQ)
        else:
            dm1_cnnlo = 0.
            dm1_nnlo  = pneob.m1_dot_spin_eob_nnlo(m1, m2, r, pr, pph, chi1, chi2, factQ)
    else:
        dm1_nnlo  = 0.
        dm1_cnnlo = 0.

    # Padé resummation
    if '_P' in order:
        num, den = (eval(nstr) for nstr in order.split('_P')[1][:2])
    else:
        num, den = 3, 0
    if 'CP' in order:
        cnum, cden = (eval(nstr) for nstr in order.split('CP')[1][:2])
    else:
        cnum, cden = 3, 0
    
    if CQ:
        pn_c_fact = PN_pade([dm1_clo, 0., dm1_cnlo, dm1_cnnlo], cnum, cden)
        pn_fact   = PN_pade([dm1_lo,  0., dm1_nlo,  dm1_nnlo],  num,  den)
    else:
        pn_c_fact = 1.
        pn_fact   = PN_pade([dm1_lo, 0., dm1_nlo,  dm1_nnlo],  num,  den)
    
    return dm1_0*H_fac*pn_c_fact*pn_fact

def j1_dot_spin_eob(m1, m2, r, pr, pph, chi1, chi2, order='NNLO', omg=None):
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

    nu = m1*m2/(m1 + m2)**2

    sigma1 = np.sqrt(1. - chi1**2)
    omgH = 0.5*chi1/m1/(1. + sigma1)

    u  = 1./r
    u2 = u*u

    factQ = False
    CQ    = False

    # Overall prefactor
    if 'fact' in order:
        factQ = True
        dj1_0 = -3.2*nu**2*m1**4*u2**3*(1 + sigma1)
    else:
        dj1_0 = -1.6*nu**2*m1**3*u2**3

    # Superradiance prefactor
    if factQ:
        if 'omg' in order:
            if omg is not None:
                H_fac = omgH - omg
            else:
                raise ValueError("j1_dot_spin_eob(): give orbital frequency to use omg in H prefactor.")
        else:
            H_fac = omgH - pph*u2
    else:
        H_fac = 1.
    
    dj1_lo  = pneob.j1_dot_spin_eob_lo(chi1, factQ)

    if 'NLO' in order:
        if 'CF' in order:
            CQ = True
            dj1_cnlo = pneob.j1_dot_spin_eob_nlo_circ(m1, m2, r, chi1, factQ)
            dj1_nlo  = pneob.j1_dot_spin_eob_nlo_nc(m1, m2, r, pr, pph, chi1)
        else:
            dj1_cnlo = 0.
            dj1_nlo = pneob.j1_dot_spin_eob_nlo(m1, m2, r, pr, pph, chi1, chi2, factQ)
    else:
        dj1_nlo  = 0.
        dj1_cnlo = 0.
    
    if 'NNLO' in order:
        if CQ:
            dj1_cnnlo = pneob.j1_dot_spin_eob_nnlo_circ(m1, m2, r, chi1, chi2, factQ)
            dj1_nnlo  = pneob.j1_dot_spin_eob_nnlo_nc(m1, m2, r, pr, pph, chi1, chi2, factQ)
        else:
            dj1_cnnlo = 0.
            dj1_nnlo = pneob.j1_dot_spin_eob_nnlo(m1, m2, r, pr, pph, chi1, chi2, factQ)
    else:
        dj1_nnlo  = 0.
        dj1_cnnlo = 0.

    # Padé resummation
    if '_P' in order:
        num, den = (eval(nstr) for nstr in order.split('_P')[1][:2])
    else:
        num, den = 3, 0
    if 'CP' in order:
        cnum, cden = (eval(nstr) for nstr in order.split('CP')[1][:2])
    else:
        cnum, cden = 3, 0
    
    if CQ:
        pn_c_fact = PN_pade([dj1_lo, 0., dj1_cnlo, dj1_cnnlo], cnum, cden)
        pn_fact   = PN_pade([1.,     0., dj1_nlo,  dj1_nnlo],  num,  den)
    else:
        pn_c_fact = 1.
        pn_fact   = PN_pade([dj1_lo, 0., dj1_nlo,  dj1_nnlo],  num,  den)
    
    return dj1_0*H_fac*pn_c_fact*pn_fact

def PN_pade(a, n, m):
    """
    Computes Padé (n, m) approximant of PN series with terms a (array)
    """
    
    if n < 0 or m < 0:
        raise ValueError("PN_pade(): n and m must be positive integers.")
    if n + m >= len(a):
        raise ValueError("PN_pade(): n + m must be less than the length of the input array.")
    
    if n == 3 and m == 0:
        return a[0] + a[1] + a[2] + a[3]
    elif n == 1 and m == 2:
        return (a[1]**3 + a[0]*a[1]*(a[1] - 2.*a[2]) + a[0]**2*(a[3] - a[2]))/(a[1]**2 + a[2]**2 + a[0]*(a[3] - a[2]) - a[1]*(a[2] + a[3]))
    elif n == 2 and m == 1:
        return a[0] + a[1] + a[2]**2/(a[2] - a[3])
    elif n == 0 and m == 3:
        return a[0]**4/(a[0]**3 - a[1]**3 + a[0]*a[1]*(a[1] + 2.*a[2]) - a[0]**2*(a[1] + a[2] + a[3]))
    else:
        raise ValueError(f"PN_pade(): ({n}, {m}) Padé approximant not implemented.")

def H_ADM(r, pr, L, chi1, chi2, nu=0.25, pn=2):
    """
    ADM Hamiltonian
    Eq C1 from https://arxiv.org/pdf/2209.00611
    """

    u  = 1/r
    u2 = u*u
    u3 = u2*u
    u4 = u3*u

    dm = np.sqrt(1 - 4*nu)
    m1 = (1 + dm)/2
    m2 = 1 - m1

    H_0 = 0.5*pr**2 - u + 0.5*L**2*u2
    H_1 = 0.5*u2 - pr**2*(nu + 3/2)*u + \
          + pr**4*(3*nu/8 - 1./8) + \
          + L**2*(-nu/2*u3 - 3./2*u3 + pr**2*(3*nu/4 - 1./4)*u2) \
          + L**4*(3.*nu/8*u4 - 1./8*u4)
    
    H_2 = -nu/4*u3 - 0.5*u3 + pr**2*(9*nu/2*u2 + 9/4*u2) + \
          + pr**4*(-nu*nu - 5*nu/2 + 5/8)*u \
          + pr**6*(5*nu*nu/16 - 5*nu/16 + 1./16) + \
          + L**2*((3*nu + 11/4)*u4 + pr**2*(-nu*nu - 9*nu/2 + 5/4)*u3 + pr**4*(15*nu*nu/16 -15*nu/16 + 3./16)*u2) + \
          + L**4*((-3*nu*nu/8 - 2*nu + 5/8)*u4*u + pr**2*(15*nu*nu/16 - 15*nu/16 + 3./16)*u4) \
          + L**6*(5*nu*nu/16 - 5*nu/16 + 1./16)*u3*u3
    
    # Add spin part of Hamiltonian

    H_S_LO = L*(u3)*(1.5*nu*(chi1 + chi2) + 2*(m1**2*chi1 + m2**2*chi2))

    if pn == 0:
        return H_0
    elif pn == 1:
        return H_0 + H_1
    elif pn == 2:
        return H_0 + H_1 + H_2 + H_S_LO
    else:
        raise ValueError('PN order not available, max is 2')

## EJ from dynamical variables (harmonic coordinates)
def EJ_from_rv(r, v, S, Sig, nu):
    """
    Calculates PN energy and ang momentum from the dynamical variables.
    """
    
    nu2 = nu**2
    
    q = (1 - 2*nu + np.sqrt(1 - 4*nu))/(2*nu)
    M1 = q/(1+q)
    M2 = 1/(1+q)
    deltam = M1 - M2
    mu = nu/(M1 + M2)
    
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
    
    rdot   = (rx*vx + ry*vy + rz*vz)/rad
    
    Sx = S[0][:]
    Sy = S[1][:]
    Sz = S[2][:]
    
    Sigx = Sig[0][:]
    Sigy = Sig[1][:]
    Sigz = Sig[2][:]
    
    nSv   = (rx*Sy*vz - rx*Sz*vy + ry*Sz*vx - ry*Sx*vz + rz*Sx*vy - rz*Sy*vx)/rad
    nSigv = -(rx*Sigy*vz - rx*Sigz*vy + ry*Sigz*vx - ry*Sigx*vz + rz*Sigx*vy - rz*Sigy*vx)/rad
    
    Sigv = Sigx*vx + Sigy*vy + Sigz*vz
    Sv   = Sx*vx + Sy*vy + Sz*vz
    nSig = (rx*Sigx + ry*Sigy + rz*Sigz)/rad
    nS   = (rx*Sx + ry*Sy + rz*Sz)/rad
    
    # Energy
    # N + 1PN
    
    Energy = vq/2 - 1/rad + 3/8*(1 - 3*nu)*vq**2 + 1/2*(3 + nu)*vq/rad + 1/2*nu*rdot**2/rad + 1/2/rq
    
    # 1.5PN
    
    Energy += (-deltam*nSigv - nSv)/rq
    
    # 2PN
    
    Energy += vq**3*5/16*(1 - 7*nu + 13*nu2) + 1/8*(21 - 23*nu - 27*nu2)*vq**2/rad + 1/4*nu*(1 - 15*nu)*vq*rdot**2/rad - 3/8*nu*(1 - 3*nu)*rdot**4/rad + 1/8*(14 - 55*nu + 4*nu2)*vq/rq + 1/8*(4 + 69*nu + 12*nu2)*rdot**2/rq - 1/4*(2 + 15*nu)/(rq*rad)
    
    # 2PN SS
    
    Energy += 0
    
    # 2.5PN
    
    Energy += (nSigv*deltam*(-1+5*nu)/2*vq + nSv*(3/2*nu*rdot**2 + 3/2*(1 + nu)*vq))/rq + (-3/2*deltam*nSigv - 2*nSv)*nu/(rq*rad)
    
    # 3PN
    
    Energy += (315 + 18469 * nu)/(840 * rad**4) - \
           ((rdot**2) * vq**2 * nu * (21 + 75 * nu - 375 * nu**2))/(16 * rad) - \
           (rdot**4 * nu * (731 - 429 * nu - 288 * nu**2))/(48 * rad**2) + \
           (rdot**6 * nu * (5 - 25 * nu + 25 * nu**2))/(16 * rad) - \
           (rdot**4 * vq * nu * (9 - 84 * nu + 165 * nu**2))/(16 * rad) + \
           1/128 * vq**4 * (35 - 413 * nu + 1666 * nu**2 - 2261 * nu**3) + \
           (rdot**2 * vq * (12 + 248 * nu - 815 * nu**2 - 324 * nu**3))/(16 * rad**2) + \
           (vq**2 * (135 - 194 * nu + 406 * nu**2 - 108 * nu**3))/(16 * rad**2) + \
           (vq**3 * (55 - 215 * nu + 116 * nu**2 + 325 * nu**3))/(16 * rad) + \
           (vq * (2800 - (53976 - 1435 * np.pi**2) * nu - 11760 * nu**2 + 1120 * nu**3))/(2240 * rad**3) + \
           (rdot**2 * (3360 + (18568 - 4305 * np.pi**2) * nu + 28560 * nu**2 + 7840 * nu**3))/(2240 * rad**3)
           
    # Angular momentum
    # N + 1PN
    
    Jang = 1 + 1/2*(1 - 3*nu)*vq + (3 - nu)/rad
    
    # 2PN
    
    Jang += -((rdot**2 * nu * (2 + 5 * nu))/(2 * rad)) + \
           (vq * (7 - 10 * nu - 9 * nu**2))/(2 * rad) + \
           (14 - 41 * nu + 4 * nu**2)/(4 * rad**2) + \
           3/8 * vq**2 * (1 - 7 * nu + 13 * nu**2)
           
    # 3PN
    
    Jang += -((rdot**2 * vq * nu * (12 - 7 * nu - 75 * nu**2))/(4 * rad)) + \
           (3 * rdot**4 * nu * (2 - 2 * nu - 11 * nu**2))/(8 * rad) + \
           (rdot**2 * (12 - 287 * nu - 951 * nu**2 - 324 * nu**3))/(24 * rad**2) + \
           1/16 * vq**3 * (5 - 59 * nu + 238 * nu**2 - 323 * nu**3) + \
           (vq * (135 - 322 * nu + 315 * nu**2 - 108 * nu**3))/(12 * rad**2) + \
           (5/2 - ((20796 - 1435 * np.pi**2) * nu)/1120 - 7 * nu**2 + nu**3)/rad**3 + \
           (vq**2 * (33 - 142 * nu + 106 * nu**2 + 195 * nu**3))/(8 * rad)
           
    #Jang_norm = Jang*rad*np.sqrt(vq - rdot**2)
           
    # Components, only non-spin
    
    Jangx = Jang*(ry*vz - rz*vy)
    Jangy = Jang*(rz*vx - rx*vz)
    Jangz = Jang*(rx*vy - ry*vx)
    
    # 1.5PN, SO
    
    Jangx += (-1/2*deltam*Sigv*vx - 1/2*Sv*vx + 1/2*vq*Sx + 1/2*deltam*vq*Sigx) + ((deltam*nSig + 3*nS)*rx/rad - 3*Sx - deltam*Sigx)/rad
    Jangy += (-1/2*deltam*Sigv*vy - 1/2*Sv*vy + 1/2*vq*Sy + 1/2*deltam*vq*Sigy) + ((deltam*nSig + 3*nS)*ry/rad - 3*Sy - deltam*Sigy)/rad
    Jangz += (-1/2*deltam*Sigv*vz - 1/2*Sv*vz + 1/2*vq*Sz + 1/2*deltam*vq*Sigz) + ((deltam*nSig + 3*nS)*rz/rad - 3*Sz - deltam*Sigz)/rad
    
    # 2.5PN, SO
    
    Jangx += ((Sigv*deltam*(-3/8 + 5/4*nu) + Sv*(-3/8 + 9/8*nu))*vx*vq + vq**2*(3/8 - 9/8*nu)*Sx + deltam*Sigx*(3/8 - 5/4*nu)*vq**2) + \
             ((nSig*deltam*(-3*nu*rdot**2 + (1 + 3*nu)/2*vq) + Sigv*deltam*rdot*(-1/2 - 13/4*nu) + nS*(-9/2*nu*rdot**2 + (7 - nu)/2*vq) + Sv*rdot*(-3 - 7*nu)/2)*rx/rad + \
             (7/4*nu*deltam*rdot*nSig + Sigv*deltam*(-3 + nu)/2 + nS*(-3 + 6*nu)*rdot + Sv*(1 - nu)/2)*vx + \
             ((2 + 5/2*nu)*rdot**2 + (-3+nu)/2*vq)*Sx + Sigx*deltam*(7/2*nu*rdot**2 + (3/2 - nu)*vq))/rad + \
             (rx/rad*(nSig*deltam*(-1 - 3*nu)/2 + nS*(-1/2 - 2*nu)) + Sx*(1/2 + 2*nu) + Sigx*deltam*(1/2 + 3/2*nu))/rq
    Jangy += ((Sigv*deltam*(-3/8 + 5/4*nu) + Sv*(-3/8 + 9/8*nu))*vy*vq + vq**2*(3/8 - 9/8*nu)*Sy + deltam*Sigy*(3/8 - 5/4*nu)*vq**2) + \
             ((nSig*deltam*(-3*nu*rdot**2 + (1 + 3*nu)/2*vq) + Sigv*deltam*rdot*(-1/2 - 13/4*nu) + nS*(-9/2*nu*rdot**2 + (7 - nu)/2*vq) + Sv*rdot*(-3 - 7*nu)/2)*ry/rad + \
             (7/4*nu*deltam*rdot*nSig + Sigv*deltam*(-3 + nu)/2 + nS*(-3 + 6*nu)*rdot + Sv*(1 - nu)/2)*vy + \
             ((2 + 5/2*nu)*rdot**2 + (-3+nu)/2*vq)*Sy + Sigy*deltam*(7/2*nu*rdot**2 + (3/2 - nu)*vq))/rad + \
             (ry/rad*(nSig*deltam*(-1 - 3*nu)/2 + nS*(-1/2 - 2*nu)) + Sy*(1/2 + 2*nu) + Sigy*deltam*(1/2 + 3/2*nu))/rq
    Jangz += ((Sigv*deltam*(-3/8 + 5/4*nu) + Sv*(-3/8 + 9/8*nu))*vz*vq + vq**2*(3/8 - 9/8*nu)*Sz + deltam*Sigz*(3/8 - 5/4*nu)*vq**2) + \
             ((nSig*deltam*(-3*nu*rdot**2 + (1 + 3*nu)/2*vq) + Sigv*deltam*rdot*(-1/2 - 13/4*nu) + nS*(-9/2*nu*rdot**2 + (7 - nu)/2*vq) + Sv*rdot*(-3 - 7*nu)/2)*rz/rad + \
             (7/4*nu*deltam*rdot*nSig + Sigv*deltam*(-3 + nu)/2 + nS*(-3 + 6*nu)*rdot + Sv*(1 - nu)/2)*vz + \
             ((2 + 5/2*nu)*rdot**2 + (-3+nu)/2*vq)*Sz + Sigz*deltam*(7/2*nu*rdot**2 + (3/2 - nu)*vq))/rad + \
             (rz/rad*(nSig*deltam*(-1 - 3*nu)/2 + nS*(-1/2 - 2*nu)) + Sz*(1/2 + 2*nu) + Sigz*deltam*(1/2 + 3/2*nu))/rq
    
    # Norm
    
    Jang_norm = np.sqrt(Jangx**2 + Jangy**2 + Jangz**2)
           
    return Energy, mu*Jang_norm, [mu*Jangx, mu*Jangy, mu*Jangz]

###########################
##  Polar <-> Cartesian  ##
###########################
# r, phi, etc. are all numpy arrays
def Polar2Cartesian(r, phi, pr, pphi):
    x  = r*np.cos(phi)
    y  = r*np.sin(phi) 
    px = np.cos(phi)*pr - np.sin(phi)*pphi/r
    py = np.sin(phi)*pr + np.cos(phi)*pphi/r

    return x, y, px, py

def Cartesian2Polar(x, y, px, py):
    r    = np.sqrt(x**2 + y**2)
    phi  = np.arctan(y/x) + pi*np.logical_and(x<0,1)
    pr   = (x*px + y*py)/r
    pphi = x*py - y*px

    return r, phi, pr, pphi

###########################
##      EOB <-> ADM      ##
###########################
# functions to convert EOB (cartesian) coordinates and momenta (qe,pe) to ADM (qa, pa). 
# memo: here momenta are mu-normalized
def Eob2Adm(qe_vec, pe_vec, nu, PN_order):
    # shorthands
    qe2       = np.dot(qe_vec, qe_vec) # x, y
    qe        = np.sqrt(qe2)
    qe3       = qe*qe2
    qe4       = qe*qe3
    pe2       = np.dot(pe_vec, pe_vec)
    pe        = np.sqrt(pe2)
    pe3       = pe*pe2
    pe4       = pe*pe3 
    qedotpe   = np.dot(qe_vec, pe_vec)
    qedotpe2  = qedotpe*qedotpe
    nu2       = nu*nu

    # coefficients for ADM coordinates
    cqa_1PN_q = nu*pe2/2 - (1 + nu/2)/qe
    cqa_1PN_p = nu*qedotpe
    cqa_2PN_q = -nu/8*(1 + nu)*pe4 + 3/4*nu*(nu/2 - 1)*pe2/qe - nu*(2 + 5/8*nu)*qedotpe2/qe3 + (-nu2 + 7*nu - 1)/4/qe2
    cqa_2PN_p = qedotpe*(nu*(nu - 1)/2*pe2 + nu/2*(-5 + nu/2)/qe)
    
    # coefficients for ADM momenta
    cpa_1PN_q = -(1 + nu/2)*qedotpe/qe3
    cpa_1PN_p = -nu/2*pe2 + (1 + nu/2)/qe
    cpa_2PN_q = qedotpe/qe3*(3/4*nu*(nu/2 - 1)*pe2 + 3/8*nu2*qedotpe2/qe2 + (-3/2 + 5/2*nu - 3/4*nu2)/qe)
    cpa_2PN_p = nu*(1 + 3*nu)/8*pe4 - nu/4*(1 + 7/2*nu)*pe2/qe + nu*(1 + nu/8)*qedotpe2/qe3 + (5/4 - 3/4*nu + nu2/2)/qe2

    # Put all together
    qa_vec = qe_vec
    pa_vec = pe_vec
    if PN_order > 0:
        qa_vec    = qa_vec + cqa_1PN_q*qe_vec + cqa_1PN_p*pe_vec
        pa_vec    = pa_vec + cpa_1PN_q*qe_vec + cpa_1PN_p*pe_vec
    if PN_order > 1:
        qa_vec    = qa_vec + cqa_2PN_q*qe_vec + cqa_2PN_p*pe_vec
        pa_vec    = pa_vec + cpa_2PN_q*qe_vec + cpa_2PN_p*pe_vec
    if PN_order > 2:
        print('2PN is the max PN order available')
    return qa_vec, pa_vec

def Adm2Eob(qa_vec, pa_vec, nu, PN_order):
    # shorthands
    qa2      = np.dot(qa_vec, qa_vec) # x, y
    qa       = np.sqrt(qa2)
    qa3      = qa*qa2
    qa4      = qa*qa3
    pa2      = np.dot(pa_vec, pa_vec)
    pa       = np.sqrt(pa2)
    pa3      = pa*pa2
    pa4      = pa*pa3
    qadotpa  = np.dot(qa_vec, pa_vec)
    qadotpa2 = qadotpa*qadotpa
    nu2      = nu*nu

    # coefficients for EOB coordinates
    cqe_1PN_q = -nu/2*pa2 + 1/qa*(1 + nu/2)
    cqe_1PN_p = -qadotpa*nu 
    cqe_2PN_q = nu/8*(1 - nu)*pa4 + nu/4*(5 - nu/2)*pa2/qa + nu*(1 + nu/8)*qadotpa2/qa3 + 1/4*(1 - 7*nu + nu2)/qa2
    cqe_2PN_p = qadotpa*(nu/2*(1 + nu)*pa2 + 3/2*nu*(1 - nu/2)/qa)
    
    # coefficients for EOB momenta
    cpe_1PN_q = qadotpa/qa3*(1 + nu/2)
    cpe_1PN_p = nu/2*pa2 - 1/qa*(1 + nu/2)
    cpe_2PN_q = qadotpa/qa3*(nu/8*(10 - nu)*pa2 + 3/8*nu*(8 + 3*nu)*qadotpa2/qa2 + 1/4*(-2 - 18*nu + nu2)/qa)
    cpe_2PN_p = nu/8*(-1 + 3*nu)*pa4 - 3/4*nu*(3 + nu/2)*pa2/qa  - nu/8*(16 + 5*nu)*qadotpa2/qa3 + 1/4*(3 + 11*nu)/qa2

    # Put all together 
    qe_vec = qa_vec
    pe_vec = pa_vec
    if PN_order > 0:
        qe_vec    = qe_vec + cqe_1PN_q*qa_vec + cqe_1PN_p*pa_vec
        pe_vec    = pe_vec + cpe_1PN_q*qa_vec + cpe_1PN_p*pa_vec
    if PN_order > 1:
        qe_vec    = qe_vec + cqe_2PN_q*qa_vec + cqe_2PN_p*pa_vec
        pe_vec    = pe_vec + cpe_2PN_q*qa_vec + cpe_2PN_p*pa_vec
    if PN_order > 2:
        print('2PN is the max PN order available')

    return qe_vec, pe_vec