from PyART.analytic import pnpedia


def test_pnpedia_order_truncation():
    """Test that PNPedia returns a callable expression truncated to requested PN order."""
    pnpedia_instance = pnpedia.PNPedia(path="./PNPedia", dowload=False)

    # Capture the resulting symbolic expression used to build the numeric function
    captured = {}
    original_pn_to_function = pnpedia_instance.pn_to_function

    def capture(expr):
        captured["expr"] = expr
        return original_pn_to_function(expr)

    pnpedia_instance.pn_to_function = capture

    func, sym = pnpedia_instance.get_pn_quantity(
        name="energy_circular_nonspinning_binding", order="3"
    )

    assert callable(func)
    assert isinstance(sym, tuple)

    assert "expr" in captured
    expr = captured["expr"]

    # The raw expression should be truncated at most to x^4 for 3PN relative.
    powers = [
        t.as_coeff_exponent(pnpedia.sp.symbols("x"))[1] for t in expr.as_ordered_terms()
    ]
    assert float(min(powers)) >= 0
    assert float(max(powers)) <= 4


def test_pnpedia_fractional_power_handling():
    pnpedia_instance = pnpedia.PNPedia(path="./PNPedia", dowload=False)

    expr = (
        pnpedia.sp.symbols("x") ** pnpedia.sp.Rational(3, 2)
        + pnpedia.sp.symbols("x") ** 2
        + pnpedia.sp.symbols("x") ** pnpedia.sp.Rational(5, 2)
    )

    min_order, max_order = pnpedia_instance._get_x_power_range(expr)
    assert min_order == pnpedia.sp.Rational(3, 2)
    assert max_order == pnpedia.sp.Rational(5, 2)

    term_powers = [
        pnpedia_instance._get_x_exponent(term) for term in expr.as_ordered_terms()
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
        if pnpedia_instance._get_x_exponent(term) <= target_power
    )
    assert selected == expr


def test_pnpedia_custom_variable_counting():
    pnpedia_instance = pnpedia.PNPedia(path="./PNPedia", dowload=False)

    v = pnpedia.sp.symbols("v")
    expr = v**2 + v ** pnpedia.sp.Rational(5, 2)

    min_order, max_order = pnpedia_instance._get_x_power_range(expr, v)
    assert min_order == pnpedia.sp.Rational(2)
    assert max_order == pnpedia.sp.Rational(5, 2)

    term_powers = [
        pnpedia_instance._get_x_exponent(term, v) for term in expr.as_ordered_terms()
    ]
    assert sorted(term_powers) == [pnpedia.sp.Rational(2), pnpedia.sp.Rational(5, 2)]


def test_mathematica_to_python_vars_expanded():
    pnpedia_instance = pnpedia.PNPedia(path="./PNPedia", dowload=False)

    expr = "x * \\[Nu] + \\[Delta]^2 + \\[Theta] + \\[Lambda]0[e] + \\[Lambda]0'[e] + Sqrt[x] + Log[x]"
    converted = pnpedia_instance.mathematica_to_python_vars(expr)

    assert "\\[" not in converted
    assert "nu" in converted
    assert "delta" in converted
    assert "theta" in converted
    assert "lambda0e" in converted
    assert "lambda0eprime" in converted

    parsed = pnpedia.parse_mathematica(converted)
    assert parsed.has(pnpedia.sp.symbols("x"))


def test_pnpedia_parse_many_files():
    pnpedia_instance = pnpedia.PNPedia(path="./PNPedia", dowload=False)

    # Test a reasonable subset to avoid linear full-repo run-time in CI while still covering many expressions.
    candidates = list(pnpedia_instance.pnpedia_structure.items())[:150]

    failures = []
    for key, path in candidates:
        try:
            func, symbols = pnpedia_instance.get_pn_quantity(
                name=key, order="1", path=path
            )
            assert callable(func)
            assert isinstance(symbols, tuple)
        except Exception as exc:
            failures.append((key, path, str(exc)))

    assert (
        not failures
    ), f"PNPedia parse failures in subset: {len(failures)} failures. First: {failures[:3]}"


def test_pnpedia_noninteger_order_truncation():
    pnpedia_instance = pnpedia.PNPedia(path="./PNPedia", dowload=False)
    captured = {}
    original_pn_to_function = pnpedia_instance.pn_to_function

    def capture(expr):
        captured["expr"] = expr
        return original_pn_to_function(expr)

    pnpedia_instance.pn_to_function = capture

    func, sym = pnpedia_instance.get_pn_quantity(
        name="energy_circular_nonspinning_binding", order="2.5"
    )

    assert callable(func)
    assert isinstance(sym, tuple)

    expr = captured["expr"]
    powers = [
        t.as_coeff_exponent(pnpedia.sp.symbols("x"))[1] for t in expr.as_ordered_terms()
    ]

    assert float(max(powers)) <= 1 + 2.5


if __name__ == "__main__":
    test_pnpedia_order_truncation()
    test_pnpedia_fractional_power_handling()
