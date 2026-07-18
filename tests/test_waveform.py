"""
General tests for the waveform class in PyART
"""

from PyART import waveform
from PyART.utils import wf_utils
import copy
import inspect
import numpy as np
import pytest

# deactivate all plots during testing
import matplotlib

matplotlib.use("Agg")


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


def test_find_max_first_max_after_t_without_peak_after_umin():
    """
    If no peak lies after umin the loop used to fall through and silently
    return the last peak, i.e. a merger time *before* the requested umin.
    """
    wf = waveform.Waveform()
    wf._u = np.arange(6, dtype=float)
    amp = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 0.5])
    phase = np.linspace(0.0, 1.0, len(amp))
    wf._hlm[(2, 2)] = wf_utils.get_multipole_dict(amp * np.exp(-1j * phase))

    with pytest.raises(ValueError, match="No peak found after"):
        wf.find_max(kind="first-max-after-t", umin=4.0)


@pytest.mark.parametrize("kind", ["first-max-after-t", "last-peak", "global"])
def test_find_max_without_any_peak(kind):
    """
    A flat amplitude has no peaks; every kind must say so rather than raise
    NameError on an unbound index.
    """
    wf = waveform.Waveform()
    wf._u = np.arange(6, dtype=float)
    amp = np.ones(6)
    wf._hlm[(2, 2)] = wf_utils.get_multipole_dict(amp * np.exp(-1j * wf.u))

    with pytest.raises(ValueError, match="No peaks found"):
        wf.find_max(kind=kind, umin=0.0)


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

    # check that the correct mode is plotted when mode is specified
    wf._hlm[(3, 3)] = copy.deepcopy(h_dict)
    wf._hlm[(3, 3)]["real"] = np.sin(0.3 * u)  # change the data to distinguish it
    ax = wf.plot("hlm", mode=(3, 3), show=False)
    assert np.allclose(ax.lines[0].get_xdata(), wf.u)
    assert np.allclose(ax.lines[0].get_ydata(), wf.hlm[(3, 3)]["real"])

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

    # check labels_on functionality (no legends or axis labels)
    ax = wf.plot("hp", show=False, labels_on=False)
    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == ""
    assert ax.get_legend() is None

    # Finally check the Errors
    with pytest.raises(ValueError):
        wf.plot("invalid_quantity", show=False)
        wf.plot("dyn", show=False, dyn_quantities=["t", "x", "y"])
        wf.plot("hlm", show=False, mode=(5, 5))

    wf._hp = None
    wf._dyn = None
    with pytest.raises(RuntimeError):
        wf.plot("hp", show=False)
        wf.plot("dyn", show=False)
        wf.plot("psi4lm", show=False)

    # now let's look at the plot_modes method
    ax = wf.plot_modes(show=False)
    assert ax is not None
    assert wf.plot_modes(show=True) is None
    # check modes behavior
    ax = wf.plot_modes(modes=None, show=False)
    assert len(ax) == len(wf.hlm)  # should have one subplot per mode
    ax = wf.plot_modes(modes=[(2, 2), (3, 3)], show=False)
    assert len(ax) == 2  # should have two subplots


##############################
# to_frequency
##############################


def make_td_waveform(f0=2.0, dt=1.0 / 64.0, tmax=6.0):
    """A time-domain waveform with hp/hc a quarter cycle out of phase."""
    wf = waveform.Waveform()
    u = np.arange(0.0, tmax, dt)
    wf._u = u
    wf._t = u.copy()
    wf._hp = np.sin(2 * np.pi * f0 * u)
    wf._hc = np.cos(2 * np.pi * f0 * u)
    wf._domain = "Time"
    wf._units = "geom"
    return wf, f0, dt


@pytest.mark.parametrize("taper", [False, True])
def test_to_frequency_returns_arrays(taper):
    """
    windowing() returns an (array, wfact) tuple; assigning it straight to
    _hp/_hc left a tuple where an array was expected.
    """
    wf, f0, dt = make_td_waveform()

    wf.to_frequency(taper=taper, pad=False)

    assert wf.domain == "Freq"
    for arr in (wf.f, wf.hp, wf.hc):
        assert isinstance(arr, np.ndarray), "to_frequency did not produce an array"
    assert len(wf.f) == len(wf.hp) == len(wf.hc)


def test_to_frequency_pad_lengths_and_spectrum():
    """Padding must reach the next power of two and keep u/hp/hc consistent."""
    wf, f0, dt = make_td_waveform()
    n_in = len(wf.u)

    wf.to_frequency(taper=False, pad=True)

    # seglen = nextpow2(6.0) = 8 -> 8/dt = 512 samples
    n_pad = int(8.0 / dt)
    assert len(wf.u) == n_pad
    assert len(wf.t) == n_pad
    assert n_pad > n_in
    assert len(wf.f) == n_pad // 2 + 1  # rfft length
    assert len(wf.hp) == len(wf.f)
    assert len(wf.hc) == len(wf.f)

    # the spectrum still peaks at the injected frequency
    df = wf.f[1] - wf.f[0]
    assert wf.f[np.argmax(np.abs(wf.hp))] == pytest.approx(f0, abs=df)
    assert wf.f[np.argmax(np.abs(wf.hc))] == pytest.approx(f0, abs=df)


def test_to_frequency_hc_is_not_a_copy_of_hp():
    """The pad branch built hc out of self.hp, silently duplicating hp."""
    wf, f0, dt = make_td_waveform()

    wf.to_frequency(taper=False, pad=True)

    assert not np.array_equal(wf.hp, wf.hc), "hc is a copy of hp"
    # sin and cos differ by a 90 degree phase at the peak
    ipk = np.argmax(np.abs(wf.hp))
    dphase = np.angle(wf.hc[ipk]) - np.angle(wf.hp[ipk])
    dphase = (dphase + np.pi) % (2 * np.pi) - np.pi
    assert abs(dphase) == pytest.approx(np.pi / 2, abs=0.2)


##############################
# to_geom / to_SI
##############################


def make_unit_waveform():
    wf = waveform.Waveform()
    u = np.linspace(1.0, 10.0, 20)
    z = np.exp(-1j * 0.2 * u)
    wf._u = u.copy()
    wf._t = u.copy()
    wf._t_psi4 = u.copy()
    wf._hlm[(2, 2)] = wf_utils.get_multipole_dict(z)
    wf._domain = "Time"
    wf._units = "geom"
    return wf


def test_to_SI_and_to_geom_roundtrip_with_t_psi4():
    """
    The attribute lists named the read-only 't_psi4' property instead of the
    '_t_psi4' attribute, so setattr raised AttributeError whenever psi4 times
    were set.
    """
    wf = make_unit_waveform()
    u0 = wf.u.copy()
    t_psi40 = wf.t_psi4.copy()
    z0 = wf.hlm[(2, 2)]["z"].copy()
    M, distance = 50.0, 100.0

    wf.to_SI(M, distance)
    assert wf.units == "SI"
    assert wf.t_psi4 is not None
    assert not np.allclose(wf.t_psi4, t_psi40), "t_psi4 was not converted"

    wf.to_geom(M, distance)
    assert wf.units == "geom"
    assert np.allclose(wf.u, u0)
    assert np.allclose(wf.t_psi4, t_psi40)
    assert np.allclose(wf.hlm[(2, 2)]["z"], z0)


def test_to_SI_scales_t_psi4_like_u():
    wf = make_unit_waveform()
    ratio0 = wf.t_psi4 / wf.u

    wf.to_SI(50.0, 100.0)

    assert np.allclose(wf.t_psi4 / wf.u, ratio0)


def test_unit_conversion_rejects_wrong_units():
    wf = make_unit_waveform()
    with pytest.raises(RuntimeError, match="Already using geom"):
        wf.to_geom(50.0, 100.0)
    wf.to_SI(50.0, 100.0)
    with pytest.raises(RuntimeError, match="Already using SI"):
        wf.to_SI(50.0, 100.0)


##############################
# mutable default arguments
##############################


@pytest.mark.parametrize(
    "func",
    [waveform.Waveform.integrate_data, waveform.WaveIntegrated.__init__],
    ids=["integrate_data", "WaveIntegrated.__init__"],
)
def test_no_mutable_dict_defaults(func):
    """
    Both fill missing keys into integr_opts, so a dict default would be shared
    state across every call that relies on it.
    """
    for name, par in inspect.signature(func).parameters.items():
        assert not isinstance(
            par.default, dict
        ), f"{func.__qualname__} has a mutable dict default for '{name}'"


def test_integrate_data_does_not_mutate_caller_opts():
    wf = waveform.Waveform()
    t = np.linspace(0.0, 40.0, 512)
    z = np.exp(-1j * 0.3 * t) * np.exp(-(((t - 20.0) / 8.0) ** 2))
    wf._psi4lm[(2, 2)] = wf_utils.get_multipole_dict(z)

    integr_opts = {"method": "FFI", "f0": 0.01}
    before = copy.deepcopy(integr_opts)

    wf.integrate_data(t_psi4=t, radius=100.0, integr_opts=integr_opts, M=1.0)

    assert integr_opts == before, "integrate_data mutated the caller's integr_opts"
    assert (2, 2) in wf.hlm


def test_integrate_data_default_opts_are_not_shared():
    """Two successive default calls must behave identically."""

    def run():
        wf = waveform.Waveform()
        t = np.linspace(0.0, 40.0, 512)
        z = np.exp(-1j * 0.3 * t) * np.exp(-(((t - 20.0) / 8.0) ** 2))
        wf._psi4lm[(2, 2)] = wf_utils.get_multipole_dict(z)
        return wf.integrate_data(t_psi4=t, radius=100.0, M=1.0)

    assert run() == run()
