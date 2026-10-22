"""
Tests for the phase alignment functions (Align, Align_continuous) in PyART
"""

from PyART.utils import wf_utils
import numpy as np
import pytest


def chirp_phase(t):
    # phase with a frequency increasing in time, so that tau and dphi are not degenerate
    return 0.2 * t + 0.002 * t**2


def mock_phases(tau0, dphi0):
    """
    Two phase evolutions such that phi_a(t - tau0) - dphi0 = phi_b(t)
    (Align shifts the time of the first one by +tau)
    """
    t_b = np.linspace(-50.0, 250.0, 6001)
    phi_b = chirp_phase(t_b)
    t_a = np.linspace(-60.0, 260.0, 5001)  # different sampling than b
    phi_a = chirp_phase(t_a + tau0) + dphi0
    return t_a, phi_a, t_b, phi_b


@pytest.mark.parametrize("align", [wf_utils.Align, wf_utils.Align_continuous])
def test_align_on_grid(align):

    # tau0 is a multiple of the grid spacing, so both functions should recover it
    tau0, dphi0 = 3.0, 0.7
    t_a, phi_a, t_b, phi_b = mock_phases(tau0, dphi0)
    t = np.arange(0.0, 200.0, 0.5)

    tau, dphi, chi2 = align(t, 200.0, 20.0, t_a, phi_a, t_b, phi_b)
    assert tau == pytest.approx(tau0, abs=1e-4)
    assert dphi == pytest.approx(dphi0, abs=1e-4)
    assert chi2 == pytest.approx(0.0, abs=1e-6)


def test_align_continuous_off_grid():

    # tau0 is not a multiple of the grid spacing: Align can only find the closest
    # grid value, while Align_continuous recovers it
    tau0, dphi0 = 3.37, -1.2
    t_a, phi_a, t_b, phi_b = mock_phases(tau0, dphi0)
    t = np.arange(0.0, 200.0, 0.5)

    tau_g, dphi_g, chi2_g = wf_utils.Align(t, 200.0, 20.0, t_a, phi_a, t_b, phi_b)
    tau_c, dphi_c, chi2_c = wf_utils.Align_continuous(
        t, 200.0, 20.0, t_a, phi_a, t_b, phi_b
    )

    assert tau_g == pytest.approx(3.5)
    assert tau_c == pytest.approx(tau0, abs=1e-4)
    assert dphi_c == pytest.approx(dphi0, abs=1e-4)
    assert chi2_c < chi2_g


@pytest.mark.parametrize("align", [wf_utils.Align, wf_utils.Align_continuous])
def test_align_window(align):

    # the phases agree (up to tau0, dphi0) only in [50, 150]: restricting the grid
    # to this window, the disagreement outside it must not affect the alignment
    tau0, dphi0 = 2.0, 0.3
    t_a, phi_a, t_b, phi_b = mock_phases(tau0, dphi0)
    outside = (t_b < 40.0) | (t_b > 160.0)
    phi_b[outside] += 0.01 * (t_b[outside] - 100.0) ** 2

    t = np.arange(50.0, 150.0, 0.5)
    tau, dphi, _ = align(t, 150.0 + 0.5, 5.0, t_a, phi_a, t_b, phi_b)
    assert tau == pytest.approx(tau0, abs=1e-4)
    assert dphi == pytest.approx(dphi0, abs=1e-4)
