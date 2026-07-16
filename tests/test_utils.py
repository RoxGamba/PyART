"""
Test the numerical helpers in PyART.utils.utils:
- D1 finite differencing
- upoly_fits polynomial extrapolation
- print_dict_comparison reporting
"""

import logging

import numpy as np
import pytest

from PyART.utils.utils import D1, print_dict_comparison, upoly_fits

##############################
# D1
##############################


@pytest.mark.parametrize("order", [1, 2, 4])
def test_D1_does_not_mutate_input(order):
    """
    D1 must treat its input as read-only: callers pass waveform arrays that are
    reused after differentiation.
    """
    x = np.linspace(0.0, 1.0, 50)
    f = np.sin(x)
    f_ref = f.copy()

    D1(f, x, order)

    assert np.array_equal(f, f_ref), f"D1(order={order}) modified its input array"


@pytest.mark.parametrize("order", [1, 2, 4])
def test_D1_exact_on_linear_function(order):
    """
    Every differencing order is exact for a linear function, endpoints included.
    """
    slope = 2.0
    x = np.linspace(0.0, 5.0, 40)
    f = slope * x + 3.0

    df = D1(f, x, order)

    assert np.allclose(
        df, slope
    ), f"D1(order={order}) is not exact on a linear function"


def test_D1_order1_fills_last_point():
    """
    Forward differencing has no stencil at the last point, so it is copied from
    the previous one. It must not be left at zero.
    """
    x = np.linspace(0.0, 1.0, 20)
    f = np.exp(x)

    df = D1(f, x, order=1)

    assert df[-1] == df[-2]
    assert df[-1] != 0.0


@pytest.mark.parametrize("order, tol", [(2, 1e-4), (4, 1e-8)])
def test_D1_converges_on_sine(order, tol):
    """Interior accuracy against a known derivative."""
    x = np.linspace(0.0, 2.0 * np.pi, 400)
    f = np.sin(x)

    df = D1(f, x, order)

    # skip the boundary stencils, which are lower order
    assert np.allclose(df[5:-5], np.cos(x)[5:-5], atol=tol)


def test_D1_rejects_nonuniform_grid():
    x = np.array([0.0, 1.0, 3.0, 6.0])
    f = x**2
    with pytest.raises(RuntimeError, match="not uniformly spaced"):
        D1(f, x, order=2)


##############################
# upoly_fits
##############################


def _upoly_data():
    """y = 1 + 2u + 3u^2 in u=1/r, so the r->inf extrapolation is exactly 1."""
    r = np.linspace(100.0, 10.0, 60)  # inward-moving
    u = 1.0 / r
    y = 1.0 + 2.0 * u + 3.0 * u**2
    return r, y


def test_upoly_fits_extrapolates_to_known_value():
    r, y = _upoly_data()
    out = upoly_fits(r, y, nmin=2, nmax=4, direction="in")
    assert np.isclose(out["extrap"], 1.0, atol=1e-8)


def test_upoly_fits_n_extract_selects_fit_order():
    """'extrap' must be the entry of 'extrap_vec' belonging to n_extract."""
    r, y = _upoly_data()
    nmin, nmax = 2, 5
    for n_extract in range(nmin, nmax + 1):
        out = upoly_fits(
            r, y, nmin=nmin, nmax=nmax, n_extract=n_extract, direction="in"
        )
        idx = list(out["fit_orders"]).index(n_extract)
        assert out["extrap"] == out["extrap_vec"][idx]


def test_upoly_fits_n_extract_defaults_to_nmax():
    r, y = _upoly_data()
    nmin, nmax = 2, 5
    default = upoly_fits(r, y, nmin=nmin, nmax=nmax, direction="in")
    explicit = upoly_fits(r, y, nmin=nmin, nmax=nmax, n_extract=nmax, direction="in")
    assert default["extrap"] == explicit["extrap"]


@pytest.mark.parametrize("n_extract", [0, 1, 6, 99])
def test_upoly_fits_rejects_n_extract_out_of_range(n_extract):
    """
    An out-of-range n_extract used to fall through the loop and raise
    UnboundLocalError; it must be reported as a ValueError instead.
    """
    r, y = _upoly_data()
    with pytest.raises(ValueError, match="n_extract"):
        upoly_fits(r, y, nmin=2, nmax=5, n_extract=n_extract, direction="in")


def test_upoly_fits_rejects_nmin_above_nmax():
    r, y = _upoly_data()
    with pytest.raises(ValueError, match="nmin>nmax"):
        upoly_fits(r, y, nmin=5, nmax=2, direction="in")


##############################
# print_dict_comparison
##############################


def test_print_dict_comparison_reports_issues_only_when_nested_dicts_differ(caplog):
    """
    The 'issues with ...' line must appear when the nested dicts differ, and
    only then.
    """
    base = {"sub": {"a": 1, "b": 2}}
    same = {"sub": {"a": 1, "b": 2}}
    diff = {"sub": {"a": 1, "b": 999}}

    with caplog.at_level(logging.INFO):
        print_dict_comparison(base, same)
    assert "issues with" not in caplog.text, "reported issues for identical dicts"

    caplog.clear()
    with caplog.at_level(logging.INFO):
        print_dict_comparison(base, diff)
    assert "issues with" in caplog.text, "failed to report issues for differing dicts"


def test_print_dict_comparison_excluded_keys(caplog):
    d1 = {"sub": {"a": 1}, "skipme": {"x": 1}}
    d2 = {"sub": {"a": 1}, "skipme": {"x": 2}}

    with caplog.at_level(logging.INFO):
        print_dict_comparison(d1, d2, excluded_keys=["skipme"])

    assert "issues with" not in caplog.text
    assert "skipme" not in caplog.text
