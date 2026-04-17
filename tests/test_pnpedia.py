import pytest

from PyART.analytic import MathematicaParser, pnpedia


@pytest.fixture
def pnpedia_instance():
    return pnpedia.PNPedia(path="./PNPedia", download=True)


def test_pnpedia_get_dummy__quantity(tmp_path):
    pnpedia_file = tmp_path / "dummy_energy.txt"
    pnpedia_file.write_text("x^2")

    quantity = pnpedia.PNPedia(str(tmp_path)).get_pn_quantity(
        "dummy_energy", "2", path=str(pnpedia_file), variable="x"
    )

    assert isinstance(quantity, pnpedia.AnalyticExpression)
    assert quantity.var == (pnpedia.sp.symbols("x"),)
    assert float(quantity(2.0)) == 4.0


def test_pnpedia_order_truncation(pnpedia_instance):
    """Test that PNPedia returns a truncated analytic expression."""
    quantity = pnpedia_instance.get_pn_quantity(
        name="energy_circular_nonspinning_binding", order="3"
    )

    assert isinstance(quantity, pnpedia.AnalyticExpression)
    assert quantity.var

    expr = quantity.expr
    x_symbol = pnpedia.sp.symbols("x")

    powers = [
        term.as_coeff_exponent(x_symbol)[1]
        for term in expr.as_ordered_terms()
    ]
    assert float(min(powers)) >= 0
    assert float(max(powers)) <= 4


def test_pnpedia_fractional_power_handling():
    expr = (
        pnpedia.sp.symbols("x") ** pnpedia.sp.Rational(3, 2)
        + pnpedia.sp.symbols("x") ** 2
        + pnpedia.sp.symbols("x") ** pnpedia.sp.Rational(5, 2)
    )

    min_order, max_order = pnpedia._get_x_power_range(expr)
    assert min_order == pnpedia.sp.Rational(3, 2)
    assert max_order == pnpedia.sp.Rational(5, 2)

    term_powers = [
        pnpedia._get_x_exponent(term)
        for term in expr.as_ordered_terms()
    ]
    assert sorted(term_powers) == [
        pnpedia.sp.Rational(3, 2),
        pnpedia.sp.Rational(2),
        pnpedia.sp.Rational(5, 2),
    ]

    target_power = pnpedia.sp.Rational(9, 2)
    selected = sum(
        term
        for term in expr.as_ordered_terms()
        if pnpedia._get_x_exponent(term) <= target_power
    )
    assert selected == expr


def test_pnpedia_custom_variable_counting():
    v = pnpedia.sp.symbols("v")
    expr = v**2 + v ** pnpedia.sp.Rational(5, 2)

    min_order, max_order = pnpedia._get_x_power_range(expr, v)
    assert min_order == pnpedia.sp.Rational(2)
    assert max_order == pnpedia.sp.Rational(5, 2)

    term_powers = [
        pnpedia._get_x_exponent(term, v)
        for term in expr.as_ordered_terms()
    ]
    assert sorted(term_powers) == [
        pnpedia.sp.Rational(2),
        pnpedia.sp.Rational(5, 2),
    ]


def test_mathematica_to_python_vars_expanded(pnpedia_instance):
    expr = (
        "x * \\[Nu] + \\[Delta]^2 + \\[Theta] + \\[Lambda]0[e] + "
        "\\[Lambda]0'[e] + Sqrt[x] + Log[x]"
    )
    converted = pnpedia_instance.mathematica_to_python_vars(expr)

    assert "\\[" not in converted
    assert "nu" in converted
    assert "delta" in converted
    assert "theta" in converted
    assert "lambda0e" in converted
    assert "lambda0eprime" in converted

    parsed = MathematicaParser().parse_expression(converted)
    assert parsed.has(pnpedia.sp.symbols("x"))


def test_pnpedia_preserves_x_logs_during_truncation(tmp_path):
    pnpedia_file = tmp_path / "dummy_log_series.txt"
    pnpedia_file.write_text("Log[x] + x*Log[x] + x^2 + x^3")

    quantity = pnpedia.PNPedia(str(tmp_path)).get_pn_quantity(
        "dummy_log_series", "1", path=str(pnpedia_file), variable="x"
    )
    x_symbol = pnpedia.sp.symbols("x")
    expected = pnpedia.sp.log(x_symbol) + x_symbol * pnpedia.sp.log(x_symbol)

    assert isinstance(quantity, pnpedia.AnalyticExpression)
    assert pnpedia.sp.simplify(quantity.expr - expected) == 0


def test_pnpedia_high_order_truncation_returns_full_expression(tmp_path):
    pnpedia_file = tmp_path / "dummy_full_log_series.txt"
    pnpedia_file.write_text("Log[x] + x*Log[x] + x^2 + x^3")

    quantity = pnpedia.PNPedia(str(tmp_path)).get_pn_quantity(
        "dummy_full_log_series", "99", path=str(pnpedia_file), variable="x"
    )
    x_symbol = pnpedia.sp.symbols("x")
    expected = (
        pnpedia.sp.log(x_symbol)
        + x_symbol * pnpedia.sp.log(x_symbol)
        + x_symbol**2
        + x_symbol**3
    )

    assert isinstance(quantity, pnpedia.AnalyticExpression)
    assert pnpedia.sp.simplify(quantity.expr - expected) == 0


def test_pnpedia_parse_many_files(pnpedia_instance):
    candidates = sorted(pnpedia_instance.pnpedia_structure.items())[:150]

    failures = []
    for key, path in candidates:
        try:
            quantity = pnpedia_instance.get_pn_quantity(
                name=key,
                order="1",
                path=path,
            )
            assert isinstance(quantity, pnpedia.AnalyticExpression)
            assert isinstance(quantity.var, tuple)
        except Exception as exc:
            failures.append((key, path, str(exc)))

    assert not failures, (
        f"PNPedia parse failures in subset: {len(failures)} failures. "
        f"First: {failures[:3]}"
    )


def test_pnpedia_noninteger_order_truncation(pnpedia_instance):
    quantity = pnpedia_instance.get_pn_quantity(
        name="energy_circular_nonspinning_binding", order="2.5"
    )

    assert isinstance(quantity, pnpedia.AnalyticExpression)

    expr = quantity.expr
    x_symbol = pnpedia.sp.symbols("x")
    powers = [
        term.as_coeff_exponent(x_symbol)[1]
        for term in expr.as_ordered_terms()
    ]

    assert float(max(powers)) <= 1 + 2.5
