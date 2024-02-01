import numpy as np

pi = np.pi;

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
    pe        = np.sqrt(pe2);
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
    qa_vec = qe_vec;
    pa_vec = pe_vec;
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
    qe_vec = qa_vec;
    pe_vec = pa_vec;
    if PN_order > 0:
        qe_vec    = qe_vec + cqe_1PN_q*qa_vec + cqe_1PN_p*pa_vec
        pe_vec    = pe_vec + cpe_1PN_q*qa_vec + cpe_1PN_p*pa_vec
    if PN_order > 1:
        qe_vec    = qe_vec + cqe_2PN_q*qa_vec + cqe_2PN_p*pa_vec
        pe_vec    = pe_vec + cpe_2PN_q*qa_vec + cpe_2PN_p*pa_vec
    if PN_order > 2:
        print('2PN is the max PN order available')

    return qe_vec, pe_vec

