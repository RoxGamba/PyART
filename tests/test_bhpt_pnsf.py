from pathlib import Path

import pytest
import sympy as sp

from PyART.analytic import BHPTEntry, BHPTPN
from PyART.analytic.expr import AnalyticExpression


@pytest.fixture(scope="session")
def bhptpn_instance(tmp_path_factory):
    download_path = (
        tmp_path_factory.mktemp("bhptpn-runtime-download") / "PostNewtonianSelfForce"
    )
    return BHPTPN(path=str(download_path), download=True)


def test_bhpt_parses_seriesdata_module_body(tmp_path):
    bhpt_file = tmp_path / "Toy.m"
    bhpt_file.write_text(
        (
            "Module[{DeltaU}, "
            "DeltaU = SeriesData[y, 0, {1, 2, 3}, 0, 3, 1]; "
            '<|"Name" -> "Toy", "Series" -> DeltaU|>]'
        ),
        encoding="utf-8",
    )

    loader = BHPTPN(str(tmp_path))
    entry = loader.get_entry(path=str(bhpt_file))
    y = sp.symbols("y")

    assert isinstance(entry, BHPTEntry)
    assert entry["metadata"]["Name"] == "Toy"
    assert entry["metadata"]["Series"] == "DeltaU"
    assert sp.simplify(entry["expr"] - (1 + 2 * y + 3 * y**2)) == 0
    assert isinstance(entry["quantity"], AnalyticExpression)
    assert entry["quantity"].var == (y,)


def test_bhpt_parses_resummed_series_and_holdform(tmp_path):
    bhpt_file = tmp_path / "Flux.m"
    bhpt_file.write_text(
        (
            "Module[{Flux}, "
            "Flux = HoldForm[ResummedSeriesData[1 + Logy, "
            "SeriesData[y, 0, {1, 2}, 0, 2, 1]]]; "
            '<|"Series" -> Flux|>]'
        ),
        encoding="utf-8",
    )

    loader = BHPTPN(str(tmp_path))
    quantity = loader.get_pn_quantity(path=str(bhpt_file))
    y = sp.symbols("y")
    expected = (1 + sp.log(y)) * (1 + 2 * y)

    assert isinstance(quantity, AnalyticExpression)
    assert sp.simplify(quantity.expr - expected) == 0
    assert quantity.var == (y,)


def test_bhpt_parses_top_level_association_file(tmp_path):
    bhpt_file = tmp_path / "Orbit.m"
    bhpt_file.write_text(
        (
            '<|"Name" -> "Association toy", '
            '"Series" -> SeriesData[p, Infinity, {1, 2}, 1, 3, 1]|>'
        ),
        encoding="utf-8",
    )

    loader = BHPTPN(str(tmp_path))
    entry = loader.get_entry(path=str(bhpt_file))
    p = sp.symbols("p")

    assert isinstance(entry, BHPTEntry)
    assert entry["metadata"]["Name"] == "Association toy"
    assert sp.simplify(entry["expr"] - (p**-1 + 2 * p**-2)) == 0
    assert isinstance(entry["quantity"], AnalyticExpression)
    assert entry["quantity"].var == (p,)


def test_bhpt_parses_reciprocal_series_encoded_with_infinity(tmp_path):
    """`SeriesData[p, Infinity, ...]` maps to a series in `1/p` about zero."""
    bhpt_file = tmp_path / "Asymptotic.m"
    bhpt_file.write_text(
        (
            "Module[{Asymptotic}, "
            "Asymptotic = SeriesData[q, Infinity, {1, 2}, 1, 3, 1]; "
            '<|"Series" -> Asymptotic|>]'
        ),
        encoding="utf-8",
    )

    loader = BHPTPN(str(tmp_path))
    quantity = loader.get_pn_quantity(path=str(bhpt_file))
    q = sp.symbols("q")

    assert sp.simplify(quantity.expr - (q**-1 + 2 * q**-2)) == 0
    assert quantity.var == (q,)


def test_bhpt_parses_empty_nested_seriesdata_as_zero(tmp_path):
    bhpt_file = tmp_path / "EmptySeries.m"
    bhpt_file.write_text(
        (
            "Module[{Flux}, "
            "Flux = Logy ResummedSeriesData[1 + y, "
            "1 + SeriesData[y, 0, {}, 0, 3, 1]]; "
            '<|"Series" -> Flux|>]'
        ),
        encoding="utf-8",
    )

    loader = BHPTPN(str(tmp_path))
    quantity = loader.get_pn_quantity(path=str(bhpt_file))
    y = sp.symbols("y")

    assert isinstance(quantity, AnalyticExpression)
    assert sp.simplify(quantity.expr - ((1 + y) * sp.log(y))) == 0
    assert quantity.var == (y,)


def test_bhpt_truncation_infers_single_variable_when_default_missing(tmp_path):
    bhpt_file = tmp_path / "InferVar.m"
    bhpt_file.write_text(
        (
            "Module[{Toy}, "
            "Toy = SeriesData[y, 0, {1, 2, 3, 4}, 0, 4, 1]; "
            '<|"Series" -> Toy|>]'
        ),
        encoding="utf-8",
    )

    loader = BHPTPN(str(tmp_path))
    quantity = loader.get_pn_quantity(path=str(bhpt_file), order="1")
    y = sp.symbols("y")

    assert isinstance(quantity, AnalyticExpression)
    assert sp.simplify(quantity.expr - (1 + 2 * y)) == 0
    assert quantity.var == (y,)


def test_bhpt_truncation_requires_explicit_variable_for_multivariate(tmp_path):
    bhpt_file = tmp_path / "AmbiguousVar.m"
    bhpt_file.write_text(
        '<|"Series" -> a + p|>',
        encoding="utf-8",
    )

    loader = BHPTPN(str(tmp_path))
    with pytest.raises(ValueError, match="Please pass `variable` explicitly"):
        loader.get_pn_quantity(path=str(bhpt_file), order="1")


def test_bhpt_parses_real_redshift_file(bhptpn_instance):
    bhpt_root = Path(bhptpn_instance.path)
    redshift_path = (
        bhpt_root / "SeriesData" / "Schwarzschild" / "Circular" / "Local" / "Redshift.m"
    )

    entry = bhptpn_instance.get_entry(path=str(redshift_path))
    cached_entry = bhptpn_instance.get_entry(path=str(redshift_path))
    y = sp.symbols("y")

    assert isinstance(entry, BHPTEntry)
    assert entry["key"] == "schwarzschild_circular_local_redshift"
    assert entry["metadata"]["Name"] == (
        "Schwarzschild Circular Orbit Red Shift Invariant"
    )
    assert any("Barry Wardell" in author for author in entry["metadata"]["Authors"])
    assert isinstance(entry["quantity"], AnalyticExpression)
    assert entry["quantity"].var == (y,)
    assert entry["expr"].has(y)
    assert cached_entry is entry


def test_bhpt_parses_real_eccentric_redshift_y_file(bhptpn_instance):
    bhpt_root = Path(bhptpn_instance.path)
    redshift_path = (
        bhpt_root
        / "SeriesData"
        / "Schwarzschild"
        / "Eccentric"
        / "Local"
        / "Redshift-y.m"
    )

    entry = bhptpn_instance.get_entry(path=str(redshift_path))
    e, y = sp.symbols("e y")

    assert isinstance(entry, BHPTEntry)
    assert entry["metadata"]["Name"] == (
        "Schwarzschild Eccentric Orbit Redshift Invariant"
    )
    assert isinstance(entry["quantity"], AnalyticExpression)
    assert entry["expr"].has(e)
    assert entry["expr"].has(y)
    # The leading pure-y (e-independent) term must be -y, i.e. the 1SF
    # Newtonian piece of the redshift invariant.
    pure_y_terms = [
        t
        for t in entry["expr"].as_ordered_terms()
        if not t.has(e) and t.as_coeff_exponent(y)[1] == 1
    ]
    assert len(pure_y_terms) == 1
    assert pure_y_terms[0] == -y


def test_bhpt_parses_real_kerr_spherical_orbit_energy_file(bhptpn_instance):
    bhpt_root = Path(bhptpn_instance.path)
    energy_path = bhpt_root / "SeriesData" / "Kerr" / "Spherical" / "Orbit" / "Energy.m"

    entry = bhptpn_instance.get_entry(path=str(energy_path))
    p, a, x = sp.symbols("p a x")

    assert isinstance(entry, BHPTEntry)
    assert entry["metadata"]["Name"] == "Kerr Spherical Orbit Energy"
    assert isinstance(entry["quantity"], AnalyticExpression)
    assert entry["quantity"].var == (a, p, x)
    assert entry["expr"].has(p)
    assert entry["expr"].has(a)
    assert entry["expr"].has(x)
