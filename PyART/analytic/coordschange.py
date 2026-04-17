import numpy as np
import sympy as sp
from .expr import AnalyticExpression, MathDispatcher, _is_sympy

pi = np.pi


class CoordsChange:
    """
    Collection of coordinate transformations with unified symbolic and
    numeric interfaces.

    Provides both numerical and symbolic (sympy) APIs for standard
    coordinate transformations used in Post-Newtonian and EOB/ADM context.
    Each transformation returns AnalyticExpression objects or plain numeric
    arrays depending on the calling method.

    Supported transformations:
    - Polar <-> Cartesian  (2D phase-space)
    - EOB <-> ADM coordinates/momenta

    References for EOB <-> ADM formulas:
    - Buonanno & Damour (1998) arXiv:gr-qc/9811091 (Appendix E)
    - Bini & Damour (2012) arXiv:1210.28...
    """

    @staticmethod
    def _validate_pn_order(PN_order):
        if PN_order not in (0, 1, 2):
            raise ValueError("PN_order must be 0, 1, or 2")

    @staticmethod
    def Eob2Adm(qe_vec, pe_vec, nu, PN_order=2):
        """
        Transforms EOB coordinates to ADM coordinates.
        Works seamlessly for both numeric arrays and sympy symbols.
        """
        CoordsChange._validate_pn_order(PN_order)

        use_sym = _is_sympy(nu) or _is_sympy(qe_vec) or _is_sympy(pe_vec)
        math = MathDispatcher(use_sym)

        qe = math.norm(qe_vec)
        qe2 = qe * qe
        qe3 = qe * qe2

        pe2 = math.dot(pe_vec, pe_vec)
        pe4 = pe2 * pe2

        qedotpe = math.dot(qe_vec, pe_vec)
        qedotpe2 = qedotpe * qedotpe
        nu2 = nu * nu

        cqa_1PN_q = nu * pe2 / 2 - (1 + nu / 2) / qe
        cqa_1PN_p = nu * qedotpe

        cqa_2PN_q = (
            -nu / 8 * (1 + nu) * pe4
            + 3 / 4 * nu * (nu / 2 - 1) * pe2 / qe
            - nu * (2 + 5 / 8 * nu) * qedotpe2 / qe3
            + (-nu2 + 7 * nu - 1) / 4 / qe2
        )
        cqa_2PN_p = qedotpe * (nu * (nu - 1) / 2 * pe2 + nu / 2 * (-5 + nu / 2) / qe)

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

        q_position_coeff = 1
        q_momentum_coeff = 0
        p_position_coeff = 0
        p_momentum_coeff = 1

        if PN_order > 0:
            q_position_coeff += cqa_1PN_q
            q_momentum_coeff += cqa_1PN_p
            p_position_coeff += cpa_1PN_q
            p_momentum_coeff += cpa_1PN_p
        if PN_order > 1:
            q_position_coeff += cqa_2PN_q
            q_momentum_coeff += cqa_2PN_p
            p_position_coeff += cpa_2PN_q
            p_momentum_coeff += cpa_2PN_p

        Q_ADM = [
            qe_vec[i] * q_position_coeff + pe_vec[i] * q_momentum_coeff
            for i in range(len(qe_vec))
        ]

        P_ADM = [
            qe_vec[i] * p_position_coeff + pe_vec[i] * p_momentum_coeff
            for i in range(len(pe_vec))
        ]

        if use_sym:
            return (
                [AnalyticExpression(q) for q in Q_ADM],
                [AnalyticExpression(p) for p in P_ADM],
            )
        return np.array(Q_ADM), np.array(P_ADM)

    @staticmethod
    def Adm2Eob(qa_vec, pa_vec, nu, PN_order=2):
        """
        Transforms ADM coordinates to EOB coordinates.
        Works seamlessly for both numeric arrays and sympy symbols.
        """
        CoordsChange._validate_pn_order(PN_order)

        use_sym = _is_sympy(nu) or _is_sympy(qa_vec) or _is_sympy(pa_vec)
        math = MathDispatcher(use_sym)

        qa = math.norm(qa_vec)
        qa2 = qa * qa
        qa3 = qa * qa2

        pa2 = math.dot(pa_vec, pa_vec)
        pa4 = pa2 * pa2

        qadotpa = math.dot(qa_vec, pa_vec)
        qadotpa2 = qadotpa * qadotpa
        nu2 = nu * nu

        cqe_1PN_q = -nu / 2 * pa2 + 1 / qa * (1 + nu / 2)
        cqe_1PN_p = -qadotpa * nu

        cqe_2PN_q = (
            nu / 8 * (1 - nu) * pa4
            + nu / 4 * (5 - nu / 2) * pa2 / qa
            + nu * (1 + nu / 8) * qadotpa2 / qa3
            + 1 / 4 * (1 - 7 * nu + nu2) / qa2
        )
        cqe_2PN_p = qadotpa * (nu / 2 * (1 + nu) * pa2 + 3 / 2 * nu * (1 - nu / 2) / qa)

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

        q_position_coeff = 1
        q_momentum_coeff = 0
        p_position_coeff = 0
        p_momentum_coeff = 1

        if PN_order > 0:
            q_position_coeff += cqe_1PN_q
            q_momentum_coeff += cqe_1PN_p
            p_position_coeff += cpe_1PN_q
            p_momentum_coeff += cpe_1PN_p
        if PN_order > 1:
            q_position_coeff += cqe_2PN_q
            q_momentum_coeff += cqe_2PN_p
            p_position_coeff += cpe_2PN_q
            p_momentum_coeff += cpe_2PN_p

        Q_EOB = [
            qa_vec[i] * q_position_coeff + pa_vec[i] * q_momentum_coeff
            for i in range(len(qa_vec))
        ]

        P_EOB = [
            qa_vec[i] * p_position_coeff + pa_vec[i] * p_momentum_coeff
            for i in range(len(pa_vec))
        ]

        if use_sym:
            return (
                [AnalyticExpression(q) for q in Q_EOB],
                [AnalyticExpression(p) for p in P_EOB],
            )
        return np.array(Q_EOB), np.array(P_EOB)

    @staticmethod
    def Polar2Cartesian(r, phi, pr, pphi):
        """
        Transforms 2D Polar coordinates to Cartesian coordinates.
        Inputs: r, phi, p_r, p_phi
        Outputs: x, y, p_x, p_y
        """
        use_sym = _is_sympy([r, phi, pr, pphi])
        math = MathDispatcher(use_sym)

        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)

        x = r * cos_phi
        y = r * sin_phi

        px = pr * cos_phi - (pphi / r) * sin_phi
        py = pr * sin_phi + (pphi / r) * cos_phi

        if use_sym:
            return (
                AnalyticExpression(x),
                AnalyticExpression(y),
                AnalyticExpression(px),
                AnalyticExpression(py),
            )
        return x, y, px, py

    @staticmethod
    def Cartesian2Polar(x, y, px, py):
        """
        Transforms 2D Cartesian coordinates to Polar coordinates.
        Inputs: x, y, p_x, p_y
        Outputs: r, phi, p_r, p_phi
        """
        use_sym = _is_sympy([x, y, px, py])
        math = MathDispatcher(use_sym)

        r = math.sqrt(x**2 + y**2)
        phi = math.arctan2(y, x)

        pr = (x * px + y * py) / r
        pphi = x * py - y * px

        if use_sym:
            return (
                AnalyticExpression(r),
                AnalyticExpression(phi),
                AnalyticExpression(pr),
                AnalyticExpression(pphi),
            )
        return r, phi, pr, pphi


def eob_ID_to_ADM(eob_Wave, verbose=False, PN_order=2, rotate_on_x_axis=True):
    """Convert EOB initial data into ADM initial data.


    Parameters
    ----------
    eob_Wave : object
        Object with attributes pars and dyn and method get_Pr().
        Must provide:
        - pars["q"]
        - dyn["r"][0], dyn["phi"][0], dyn["Pphi"][0]
        - get_Pr()[0]
    verbose : bool
        Print diagnostics if True.
    PN_order : int
        EOB->ADM conversion order (0,1,2).
    rotate_on_x_axis : bool
        If True, rotate output to x-axis.

    Returns
    -------
    dict
        Contains keys: q_cart, p_cart, px, py, x1, x2, D,
        x_offset, qe, pe, qe_chk, pe_chk.
    """
    q = eob_Wave.pars["q"]
    nu = q / (1 + q) ** 2
    r0 = eob_Wave.dyn["r"][0]
    phi0 = eob_Wave.dyn["phi"][0]
    pph0 = eob_Wave.dyn["Pphi"][0]
    pr = eob_Wave.get_Pr()
    pr0 = pr[0]

    x, y, px, py = CoordsChange.Polar2Cartesian(r0, phi0, pr0, pph0)
    qe = np.array([x, y])
    pe = np.array([px, py])

    qa, pa = CoordsChange.Eob2Adm(qe, pe, nu, PN_order=PN_order)

    d_adm = np.sqrt(np.dot(qa, qa))
    if d_adm == 0:
        raise ValueError("ADM separation is zero")

    halfx = qa[0] / 2
    b_par = d_adm / 2
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

    x1 = d_adm / (q + 1)
    x2 = -d_adm * q / (q + 1)
    x_offset = -b_par + d_adm / (q + 1)

    qe_chk, pe_chk = CoordsChange.Adm2Eob(qa, pa, nu, PN_order=PN_order)

    out = {
        "q_cart": qa,
        "p_cart": pa,
        "px": pxbynu * nu,
        "py": pybynu * nu,
        "x1": x1,
        "x2": x2,
        "D": d_adm,
        "x_offset": x_offset,
        "qe": qe,
        "pe": pe,
        "qe_chk": qe_chk,
        "pe_chk": pe_chk,
    }
    if verbose:
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
        print("q ADM->EOB : {:.5e}, {:.5e}\n".format(qe_chk[0], qe_chk[1]))

        print("p EOB      : {:.5e}, {:.5e}".format(pe[0], pe[1]))
        print("p EOB->ADM : {:.5e}, {:.5e}".format(pa[0], pa[1]))
        print("p ADM->EOB : {:.5e}, {:.5e}\n".format(pe_chk[0], pe_chk[1]))
    return out
