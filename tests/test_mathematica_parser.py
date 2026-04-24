import sympy as sp

from PyART.analytic import MathematicaParser, ParsedMathematicaSource

PARSER = MathematicaParser()


def test_parser_extracts_module_assignments_and_metadata():
    source = (
        "Module[{DeltaU}, "
        "DeltaU = SeriesData[y, 0, {1, 2}, 0, 2, 1]; "
        '<|"Name" -> "Toy", "Series" -> DeltaU|>]'
    )

    parsed = PARSER.parse_source(source)

    assert isinstance(parsed, ParsedMathematicaSource)
    assert parsed.metadata["Name"] == "Toy"
    assert parsed.metadata["Series"] == "DeltaU"
    assert parsed.assignments["DeltaU"] == "SeriesData[y, 0, {1, 2}, 0, 2, 1]"
    assert parsed.resolve_expression(parsed.metadata["Series"]) == (
        "SeriesData[y, 0, {1, 2}, 0, 2, 1]"
    )


def test_parser_extracts_top_level_association():
    source = (
        '<|"Name" -> "Association toy", '
        '"Series" -> SeriesData[p, Infinity, {1, 2}, 1, 3, 1]|>'
    )

    parsed = PARSER.parse_source(source)

    assert isinstance(parsed, ParsedMathematicaSource)
    assert parsed.metadata["Name"] == "Association toy"
    assert parsed.resolve_expression(parsed.metadata["Series"]) == (
        "SeriesData[p, Infinity, {1, 2}, 1, 3, 1]"
    )


def test_parser_converts_resummed_series_and_holdform():
    expr = PARSER.parse_expression(
        "HoldForm[ResummedSeriesData[1 + Logy, " "SeriesData[y, 0, {1, 2}, 0, 2, 1]]]"
    )
    y = sp.symbols("y")

    assert sp.simplify(expr - ((1 + sp.log(y)) * (1 + 2 * y))) == 0


def test_parser_converts_empty_seriesdata_to_zero():
    expr = PARSER.parse_expression(
        "Logy ResummedSeriesData[1 + y, 1 + SeriesData[y, 0, {}, 0, 3, 1]]"
    )
    y = sp.symbols("y")

    assert sp.simplify(expr - ((1 + y) * sp.log(y))) == 0


def test_normalize_source_strips_mathematica_comments():
    source = "x^2 (* this is a comment *) + y"
    result = PARSER.normalize_source(source)
    assert "(*" not in result
    assert "*)" not in result
    assert "this is a comment" not in result
    # The actual expression content is preserved
    assert "x" in result
    assert "y" in result


def test_normalize_source_strips_mathematica_comments_multiline():
    source = "a + (* multi\nline\ncomment *) b"
    result = PARSER.normalize_source(source)
    assert "multi" not in result
    assert "a" in result
    assert "b" in result


def test_normalize_source_strips_nested_mathematica_comments():
    source = "a + (* outer (* inner *) still outer *) b"
    result = PARSER.normalize_source(source)
    assert "outer" not in result
    assert "inner" not in result
    assert result == "a +  b"


def test_normalize_source_replaces_box_expressions():
    # Box expressions appear as \!\(\*...\) in Mathematica notebooks
    source = r"x + \!\(\*SuperscriptBox[y, 2]\)"
    result = PARSER.normalize_source(source)
    assert r"\!\(" not in result
    # Replaced by a BoxExpr0 placeholder token
    assert "BoxExpr0" in result
    assert "x" in result
