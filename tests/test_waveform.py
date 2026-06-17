"""
General tests for the waveform class in PyART
"""

from PyART import waveform
from PyART.utils import wf_utils
import copy
import numpy as np
import pytest


def test_waveform_attributes_and_mul():

    # Create a mock waveform
    wf = waveform.Waveform()

    # check that it has the right attributes
    for attr in ["hlm", "u", "t", "f", "hp", "hc", "dothlm", "psi4lm", "dyn", "kind"]:
        assert hasattr(wf, attr), f"Waveform object does not have attribute {attr}"

    # fill the waveform with some mock data
    u = np.linspace(0.0, 10.0, 20)
    z = np.exp(-1j * 0.2 * u) * (1.0 + 0.2 * np.sin(u))
    re = z.real
    im = z.imag
    h_dict = {
        "z": z,
        "A": np.abs(z),
        "p": -np.unwrap(np.angle(z)),
        "real": re,
        "imag": im,
    }
    wf._hlm[(2, 2)] = h_dict
    wf._psi4lm[(2, 2)] = copy.deepcopy(h_dict)
    wf._dothlm[(2, 2)] = copy.deepcopy(h_dict)
    wf._u = u.copy()

    # check that multiplication and division by a factor works
    original_modes = {
        var: copy.deepcopy(wf.__getattribute__(var)[(2, 2)])
        for var in ["hlm", "dothlm", "psi4lm"]
    }

    wf2 = wf * 2
    for var in ["hlm", "dothlm", "psi4lm"]:
        assert np.all(wf2.__getattribute__(var)[(2, 2)]["real"] == 2 * re)

    wf3 = 2 * wf
    for var in ["hlm", "dothlm", "psi4lm"]:
        assert np.all(wf3.__getattribute__(var)[(2, 2)]["real"] == 2 * re)

    wf4 = wf * np.int64(2)
    for var in ["hlm", "dothlm", "psi4lm"]:
        assert np.all(wf4.__getattribute__(var)[(2, 2)]["real"] == 2 * re)

    for var in ["hlm", "dothlm", "psi4lm"]:
        assert np.all(
            wf.__getattribute__(var)[(2, 2)]["real"] == original_modes[var]["real"]
        )
        assert np.all(
            wf.__getattribute__(var)[(2, 2)]["imag"] == original_modes[var]["imag"]
        )
        assert np.all(wf.__getattribute__(var)[(2, 2)]["z"] == original_modes[var]["z"])
        assert np.all(wf.__getattribute__(var)[(2, 2)]["A"] == original_modes[var]["A"])
        assert np.all(wf.__getattribute__(var)[(2, 2)]["p"] == original_modes[var]["p"])

    wfo2 = wf / 2
    for var in ["hlm", "dothlm", "psi4lm"]:
        assert np.all(wfo2.__getattribute__(var)[(2, 2)]["real"] == 0.5 * re)

    with pytest.raises(TypeError):
        wf * "2"
    pass


def test_find_max_variants_and_errors():

    # mock waveform
    wf = waveform.Waveform()
    wf._u = np.arange(6, dtype=float)
    amp = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 0.5])
    phase = np.linspace(0.0, 1.0, len(amp))
    wf._hlm[(2, 2)] = wf_utils.get_multipole_dict(amp * np.exp(-1j * phase))
    wf._psi4lm[(2, 2)] = wf_utils.get_multipole_dict(amp * np.exp(-1j * phase))

    # test identification of max
    t_mrg, A_mrg, _, _, idx = wf.find_max(kind="last-peak", return_idx=True)
    assert idx == 3
    assert t_mrg == pytest.approx(3.0)
    assert A_mrg == pytest.approx(2.0)

    t_mrg_g, A_mrg_g, _, _ = wf.find_max(kind="global")
    assert t_mrg_g == pytest.approx(3.0)
    assert A_mrg_g == pytest.approx(2.0)

    t_after, _, _, _, idx_after = wf.find_max(
        kind="first-max-after-t", umin=2.0, return_idx=True
    )
    assert idx_after == 3
    assert t_after == pytest.approx(3.0)

    with pytest.raises(ValueError):
        wf.find_max(kind="not-a-valid-option")

    # repeat also for psi4lm
    wf._t_psi4 = wf.u.copy()
    t_mrg_psi4, _, _, _ = wf.find_max(wave="psi4lm", kind="global")
    assert t_mrg_psi4 == pytest.approx(3.0)


def test_compute_dothlm_and_psi4lm():

    # mock waveform
    wf = waveform.Waveform()
    u = np.linspace(0.0, 5.0, 64)
    amp = 1.7
    omega = 1.3
    z = amp * np.sin(omega * u)
    wf._u = u
    wf._hlm[(2, 2)] = wf_utils.get_multipole_dict(z)

    wf.compute_dothlm(factor=2.0)
    assert (2, 2) in wf.dothlm
    assert len(wf.dothlm[(2, 2)]["z"]) == len(u)

    # d/dt[A sin(omega t)] = A omega cos(omega t), then scaled by factor=2.
    expected_doth = 2.0 * amp * omega * np.cos(omega * u)
    interior = slice(5, -5)
    assert np.allclose(
        wf.dothlm[(2, 2)]["z"][interior].real,
        expected_doth[interior],
        rtol=1e-4,
        atol=1e-4,
    )
    assert np.allclose(wf.dothlm[(2, 2)]["z"][interior].imag, 0.0, atol=1e-10)

    wf.compute_psi4lm(factor=0.5)
    assert (2, 2) in wf.psi4lm
    assert len(wf.psi4lm[(2, 2)]["z"]) == len(u)

    # d/dt[doth] = -2 A omega^2 sin(omega t), then scaled by factor=0.5.
    expected_psi4 = -amp * omega * omega * np.sin(omega * u)
    assert np.allclose(
        wf.psi4lm[(2, 2)]["z"][interior].real,
        expected_psi4[interior],
        rtol=2e-4,
        atol=2e-4,
    )
    assert np.allclose(wf.psi4lm[(2, 2)]["z"][interior].imag, 0.0, atol=1e-10)

    empty = waveform.Waveform()
    with pytest.raises(RuntimeError):
        empty.compute_dothlm()
    with pytest.raises(RuntimeError):
        empty.compute_psi4lm()

    # only_warn=True should not raise
    empty.compute_dothlm(only_warn=True)
    empty.compute_psi4lm(only_warn=True)

    non_uniform = waveform.Waveform()
    non_uniform._u = np.array([0.0, 0.1, 0.21, 0.33, 0.46, 0.6, 0.75])
    z_nu = np.sin(non_uniform.u)
    non_uniform._hlm[(2, 2)] = wf_utils.get_multipole_dict(z_nu)
    with pytest.raises(ValueError, match="uniformly sampled u-grid"):
        non_uniform.compute_dothlm()

    non_uniform._dothlm[(2, 2)] = wf_utils.get_multipole_dict(z_nu)
    with pytest.raises(ValueError, match="uniformly sampled u-grid"):
        non_uniform.compute_psi4lm()


def test_waveform_phase_shift():

    # mock waveform
    wf = waveform.Waveform()
    u = np.linspace(0.0, 5.0, 64)
    amp = 1.7
    omega = 1.3
    z = amp * np.exp(-1j * omega * u)
    wf._u = u
    wf._hlm[(2, 2)] = wf_utils.get_multipole_dict(z)

    wf.phase_shift(0.5, var="hlm")
    expected_phase = (
        omega * u + 0.5 * 2
    )  # since the mode is (2, 2), the phase shift is 2 times the input value

    interior = slice(5, -5)
    assert np.allclose(
        wf.hlm[(2, 2)]["p"][interior], expected_phase[interior], rtol=1e-4, atol=1e-4
    )
    assert np.allclose(wf.hlm[(2, 2)]["A"][interior], amp, rtol=1e-4, atol=1e-4)


def test_waveform_plots():
    # Create a mock waveform
    wf = waveform.Waveform()

    # fill the waveform with some mock data
    u = np.linspace(0.0, 10.0, 20)
    z = np.exp(-1j * 0.2 * u) * (1.0 + 0.2 * np.sin(u))
    re = z.real
    im = z.imag
    h_dict = {
        "z": z,
        "A": np.abs(z),
        "p": -np.unwrap(np.angle(z)),
        "real": re,
        "imag": im,
    }
    wf._hlm[(2, 2)] = h_dict
    wf._u = u.copy()

    # also add a fake dyn quantity for testing
    wf._dyn["r"] = np.linspace(1.0, 2.0, len(u))
    wf._dyn["t"] = u.copy()
    wf.dyn["x"] = np.linspace(0.0, 1.0, len(u))
    wf.dyn["y"] = np.linspace(0.0, 1.0, len(u))

    # compute pols
    wf.compute_hphc(phi=0.0, i=np.pi / 3)

    # Test plotting methods
    for quantity in ["hlm", "hp", "hc", "dyn"]:
        ax = wf.plot(quantity, show=False)
        assert ax is not None
        assert len(ax.lines) == 1  # Should have one line for the mode or pol
        assert ax.get_xlabel() == r"$t~[M]$"  # Check that the x-axis label is correct

    # check that kwargs are passed to the plot function
    ax = wf.plot("hlm", color="red", linestyle="--", show=False)
    assert ax.lines[0].get_color() == "red"
    assert ax.lines[0].get_linestyle() == "--"

    # check that dynamics dics are plotted correctly
    ax = wf.plot(
        "dyn", show=False, dyn_quantities=["y", "x"], color="blue", linestyle="-"
    )
    # the x array should be the x values, and the y array should be the y values
    assert np.allclose(ax.lines[0].get_xdata(), wf.dyn["x"])
    assert np.allclose(ax.lines[0].get_ydata(), wf.dyn["y"])
    assert ax.lines[0].get_color() == "blue"
    assert ax.lines[0].get_linestyle() == "-"
    assert ax.get_xlabel() == "$x$"
    assert ax.get_ylabel() == "$y$"
