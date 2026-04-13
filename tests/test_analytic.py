import numpy as np
import sympy as sp
import pytest

from PyART.analytic import (
    AnalyticExpression,
    CoordsChange,
    eob_ID_to_ADM,
)


def test_analytic_expression_basic():
    x, y = sp.symbols("x y")
    expr = AnalyticExpression("x**2 + y", var=(x, y))

    value = float(expr(x=3, y=2))
    assert value == 11

    dx = expr.derivative("x")
    assert isinstance(dx, AnalyticExpression)
    assert sp.simplify(dx.expr - 2 * x) == 0

    order = AnalyticExpression("x**3 + x**2", var="x").pn_order("x")
    assert order == (sp.Integer(3), sp.Integer(1))

    order2 = AnalyticExpression("x**2 + x*log(x)", var="x").pn_order("x")
    assert order2 == (sp.Integer(2), sp.Integer(1))

    truncated = AnalyticExpression("x + x**2 + x**3", var="x").truncate("x", 2)
    assert sp.expand(truncated.expr) == sp.expand(x + x**2)

    func, variables = expr.to_function()
    assert variables == (x, y)
    assert float(func(3, 2)) == 11.0


def test_coordschange_polar_cartesian_round_trip_numeric():
    x, y, px, py = CoordsChange.Polar2Cartesian(1.0, np.pi / 2, 0.1, 0.2)
    assert np.allclose([x, y], [0.0, 1.0], atol=1e-12)

    r, phi, pr, pphi = CoordsChange.Cartesian2Polar(x, y, px, py)
    assert np.allclose(r, 1.0, atol=1e-12)
    assert np.allclose(phi, np.pi / 2, atol=1e-12)
    assert np.allclose(pr, 0.1, atol=1e-12)
    assert np.allclose(pphi, 0.2, atol=1e-12)


def test_coordschange_symbolic_outputs_are_analytic_expressions():
    r, phi, pr, pphi = sp.symbols("r phi pr pphi")

    x_expr, y_expr, px_expr, py_expr = CoordsChange.Polar2Cartesian(r, phi, pr, pphi)

    assert all(
        isinstance(expr, AnalyticExpression)
        for expr in (x_expr, y_expr, px_expr, py_expr)
    )
    assert sp.simplify(x_expr.expr - r * sp.cos(phi)) == 0
    assert sp.simplify(y_expr.expr - r * sp.sin(phi)) == 0

    x_val = float(x_expr(r=1.0, phi=np.pi / 2, pr=0.1, pphi=0.2))
    y_val = float(y_expr(r=1.0, phi=np.pi / 2, pr=0.1, pphi=0.2))
    assert np.allclose(x_val, 0.0, atol=1e-12)
    assert np.allclose(y_val, 1.0, atol=1e-12)


def test_coordschange_eob_adm_round_trip_and_validation():
    qe = np.array([12.0, 0.3])
    pe = np.array([-0.02, 0.08])
    nu = 0.24

    qa, pa = CoordsChange.Eob2Adm(qe, pe, nu, PN_order=2)
    qe_back, pe_back = CoordsChange.Adm2Eob(qa, pa, nu, PN_order=2)

    assert np.max(np.abs(qe_back - qe)) < 1e-2
    assert np.max(np.abs(pe_back - pe)) < 1e-3

    with pytest.raises(ValueError, match="PN_order must be 0, 1, or 2"):
        CoordsChange.Eob2Adm(qe, pe, nu, PN_order=3)


def test_coordschange_eob_adm_symbolic_returns_wrappers():
    qe1, qe2, pe1, pe2, nu = sp.symbols("qe1 qe2 pe1 pe2 nu")

    qa_exprs, pa_exprs = CoordsChange.Eob2Adm((qe1, qe2), (pe1, pe2), nu, PN_order=1)

    assert len(qa_exprs) == 2
    assert len(pa_exprs) == 2
    assert all(isinstance(expr, AnalyticExpression) for expr in qa_exprs + pa_exprs)
    assert qa_exprs[0].free_symbols
    assert pa_exprs[0].free_symbols


def test_eob_ID_to_ADM_preserved():
    class FakeEobWave:
        def __init__(self):
            self.pars = {"q": 1.0}
            self.dyn = {"r": [1.0], "phi": [0.0], "Pphi": [0.0]}

        def get_Pr(self):
            return np.array([0.0])

    fake = FakeEobWave()
    result = eob_ID_to_ADM(fake, verbose=False, PN_order=0, rotate_on_x_axis=False)
    assert isinstance(result, dict)
    assert set(result) == {
        "D",
        "p_cart",
        "pe",
        "pe_chk",
        "px",
        "py",
        "q_cart",
        "qe",
        "qe_chk",
        "x1",
        "x2",
        "x_offset",
    }
    assert np.allclose(result["q_cart"], np.array([1.0, 0.0]))
    assert np.allclose(result["p_cart"], np.array([0.0, 0.0]))
