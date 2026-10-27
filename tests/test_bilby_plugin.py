"""
Tests for the bilby NR source models in PyART.plugin.bilby_plugin.

Everything here runs against a synthetic in-memory Waveform injected in place of
the catalogue loader, so nothing touches the network or the local_data
directory.
"""

import inspect

import numpy as np
import pytest

from PyART import waveform
from PyART.utils import utils as ut
from PyART.utils import wf_utils

pytest.importorskip("lal", reason="LAL not installed")

from PyART.plugin import bilby_plugin  # noqa: E402

TOTAL_MASS = 300.0
DISTANCE = 400.0
IOTA = 1.2
PHASE = 0.35
SAMPLING_FREQUENCY = 4096.0
MINIMUM_FREQUENCY = 20.0
MAXIMUM_FREQUENCY = 1024.0
DURATION = 32.0


def make_metadata(m1=0.6, m2=0.4, spin_1=(0.1, 0.2, 0.3), spin_2=(-0.15, 0.05, -0.2)):
    metadata = {"m1": m1, "m2": m2, "q": max(m1, m2) / min(m1, m2)}
    for label, spin in ((1, spin_1), (2, spin_2)):
        for component, value in zip("xyz", spin):
            metadata[f"chi{label}{component}"] = value
    return metadata


def make_waveform(
    metadata=None, modes=((2, 2), (2, -2), (3, 3), (3, -3)), u_max=20000.0
):
    """
    A long, slowly chirping synthetic NR waveform in geometric units.

    It has to be long enough in seconds that the SimInspiralFD conditioning
    taper fits inside it at TOTAL_MASS.
    """
    wf = waveform.Waveform()
    u = np.arange(0.0, u_max, 0.5)
    envelope = np.exp(-(((u - 0.85 * u_max) / (0.06 * u_max)) ** 2))
    orbital_phase = 0.012 * u + 2.0e-7 * u**2

    hlm = {}
    for ell, emm in modes:
        amplitude = envelope * (1.0 if ell == 2 else 0.3)
        hlm[(ell, emm)] = wf_utils.get_multipole_dict(
            amplitude * np.exp(-1j * emm * orbital_phase)
        )

    wf._u = u
    wf._t = u.copy()
    wf._hlm = hlm
    wf._domain = "Time"
    wf._units = "geom"
    wf.metadata = metadata if metadata is not None else make_metadata()
    return wf


@pytest.fixture
def waveform_arguments():
    return dict(
        catalog="sxs",
        ID="0001",
        path="/nonexistent",
        minimum_frequency=MINIMUM_FREQUENCY,
        maximum_frequency=MAXIMUM_FREQUENCY,
        sampling_frequency=SAMPLING_FREQUENCY,
    )


@pytest.fixture
def frequency_array():
    return np.arange(0.0, SAMPLING_FREQUENCY / 2 + 1.0 / DURATION, 1.0 / DURATION)


@pytest.fixture
def patched_loader(monkeypatch):
    """Replace the catalogue loader with a cached synthetic waveform."""

    def install(wf):
        calls = {"n": 0}

        def fake_loader(**kwargs):
            calls["n"] += 1
            return wf

        monkeypatch.setattr(bilby_plugin, "load_nr_waveform", fake_loader)
        return calls

    yield install
    bilby_plugin.clear_waveform_cache()


##############################
# component_masses_and_spins
##############################


def test_label_switch_follows_the_lal_convention():
    """
    Port of PrecessingNRSur_switch_labels_if_needed: when m1 < m2 the labels are
    exchanged, the in-plane spin components are swapped AND negated, and the z
    components are swapped as they are.
    """
    spin_1, spin_2 = (0.1, 0.2, 0.3), (-0.15, 0.05, -0.2)
    metadata = make_metadata(m1=0.4, m2=0.6, spin_1=spin_1, spin_2=spin_2)

    mass_1, mass_2, new_1, new_2, switched = bilby_plugin.component_masses_and_spins(
        metadata, TOTAL_MASS
    )

    assert switched is True
    assert mass_1 > mass_2
    assert mass_1 == pytest.approx(0.6 * TOTAL_MASS)
    assert mass_2 == pytest.approx(0.4 * TOTAL_MASS)
    assert np.allclose(new_1, [-spin_2[0], -spin_2[1], spin_2[2]])
    assert np.allclose(new_2, [-spin_1[0], -spin_1[1], spin_1[2]])


def test_no_label_switch_when_already_ordered():
    spin_1, spin_2 = (0.1, 0.2, 0.3), (-0.15, 0.05, -0.2)
    metadata = make_metadata(m1=0.6, m2=0.4, spin_1=spin_1, spin_2=spin_2)

    mass_1, mass_2, new_1, new_2, switched = bilby_plugin.component_masses_and_spins(
        metadata, TOTAL_MASS
    )

    assert switched is False
    assert mass_1 > mass_2
    assert np.allclose(new_1, spin_1)
    assert np.allclose(new_2, spin_2)


def test_label_switch_maps_relabelled_metadata_onto_the_same_parameters():
    """
    Metadata and its label-exchanged twin describe the same physical binary, so
    the LAL-ordered parameters must come out identical. Only the flag differs,
    since only one of the two needed the rotation.
    """
    spin_1, spin_2 = (0.1, 0.2, 0.3), (-0.15, 0.05, -0.2)
    original = make_metadata(m1=0.4, m2=0.6, spin_1=spin_1, spin_2=spin_2)
    relabelled = make_metadata(
        m1=0.6,
        m2=0.4,
        spin_1=(-spin_2[0], -spin_2[1], spin_2[2]),
        spin_2=(-spin_1[0], -spin_1[1], spin_1[2]),
    )

    a = bilby_plugin.component_masses_and_spins(original, TOTAL_MASS)
    b = bilby_plugin.component_masses_and_spins(relabelled, TOTAL_MASS)

    assert a[0] == pytest.approx(b[0])
    assert a[1] == pytest.approx(b[1])
    assert np.allclose(a[2], b[2])
    assert np.allclose(a[3], b[3])
    assert a[4] is True and b[4] is False


def test_label_switch_masses_sum_to_the_total():
    metadata = make_metadata(m1=0.3, m2=0.7)
    mass_1, mass_2, *_ = bilby_plugin.component_masses_and_spins(metadata, TOTAL_MASS)
    assert mass_1 + mass_2 == pytest.approx(TOTAL_MASS)


##############################
# the odd-m rotation
##############################


def test_even_m_modes_are_insensitive_to_the_label_switch(
    patched_loader, waveform_arguments
):
    """A pi rotation about z leaves the even-m modes alone."""
    modes = ((2, 2), (2, -2))
    patched_loader(make_waveform(make_metadata(m1=0.6, m2=0.4), modes=modes))
    _, hp_ordered, hc_ordered = bilby_plugin.nr_time_domain_polarisations(
        TOTAL_MASS, DISTANCE, IOTA, PHASE, **waveform_arguments
    )

    patched_loader(make_waveform(make_metadata(m1=0.4, m2=0.6), modes=modes))
    _, hp_switched, hc_switched = bilby_plugin.nr_time_domain_polarisations(
        TOTAL_MASS, DISTANCE, IOTA, PHASE, **waveform_arguments
    )

    # the strain is of order 1e-21, so atol must be 0 for this to mean anything
    scale = np.max(np.abs(hp_ordered))
    assert scale > 0.0
    assert np.allclose(hp_ordered, hp_switched, rtol=1e-12, atol=0.0)
    assert np.allclose(hc_ordered, hc_switched, rtol=1e-12, atol=0.0)


def test_odd_m_modes_flip_sign_under_the_label_switch(
    patched_loader, waveform_arguments
):
    """
    LALSimIMRPrecessingNRSur.c undoes the relabelling rotation by multiplying
    the odd-m modes by -1. With only (3, +-3) loaded, the whole waveform flips.
    """
    modes = ((3, 3), (3, -3))
    patched_loader(make_waveform(make_metadata(m1=0.6, m2=0.4), modes=modes))
    _, hp_ordered, hc_ordered = bilby_plugin.nr_time_domain_polarisations(
        TOTAL_MASS, DISTANCE, IOTA, PHASE, **waveform_arguments
    )

    patched_loader(make_waveform(make_metadata(m1=0.4, m2=0.6), modes=modes))
    _, hp_switched, hc_switched = bilby_plugin.nr_time_domain_polarisations(
        TOTAL_MASS, DISTANCE, IOTA, PHASE, **waveform_arguments
    )

    scale = np.max(np.abs(hp_ordered))
    assert scale > 0.0
    # atol=0: with the default 1e-8 any two ~1e-21 arrays compare equal
    assert np.allclose(hp_switched, -hp_ordered, rtol=1e-12, atol=0.0)
    assert np.allclose(hc_switched, -hc_ordered, rtol=1e-12, atol=0.0)
    # and the flip is a real change, not a no-op on near-zero data
    assert np.max(np.abs(hp_switched - hp_ordered)) > scale


##############################
# the cached waveform must survive
##############################


def test_source_model_does_not_mutate_the_cached_waveform(
    patched_loader, waveform_arguments, frequency_array
):
    """
    The loader memoises, so every likelihood evaluation gets the same object.
    cut() and to_SI() work in place, and to_SI() raises if called twice, so the
    plugin has to copy first. This is the regression for that.
    """
    wf = make_waveform()
    patched_loader(wf)

    u_before = wf.u.copy()
    hlm_before = {k: v["z"].copy() for k, v in wf.hlm.items()}
    units_before = wf.units

    for _ in range(2):
        bilby_plugin.nr_frequency_domain_source_model(
            frequency_array, TOTAL_MASS, DISTANCE, IOTA, PHASE, **waveform_arguments
        )

    assert np.array_equal(wf.u, u_before)
    assert wf.units == units_before
    for key, before in hlm_before.items():
        assert np.array_equal(wf.hlm[key]["z"], before)


def test_repeated_calls_are_reproducible(
    patched_loader, waveform_arguments, frequency_array
):
    patched_loader(make_waveform())
    first = bilby_plugin.nr_frequency_domain_source_model(
        frequency_array, TOTAL_MASS, DISTANCE, IOTA, PHASE, **waveform_arguments
    )
    second = bilby_plugin.nr_frequency_domain_source_model(
        frequency_array, TOTAL_MASS, DISTANCE, IOTA, PHASE, **waveform_arguments
    )
    assert np.array_equal(first["plus"], second["plus"])
    assert np.array_equal(first["cross"], second["cross"])


##############################
# time-domain output
##############################


def test_time_domain_peak_sits_at_t_zero(patched_loader, waveform_arguments):
    """
    t = 0 goes on the peak of the frame-invariant amplitude. Use a single
    multipole so |h| is exactly the envelope: with both +m and -m present the
    two beat against each other and |h| picks up a modulation whose maximum is
    not the envelope maximum.
    """
    patched_loader(make_waveform(modes=((2, 2),)))
    time, hplus, hcross = bilby_plugin.nr_time_domain_polarisations(
        TOTAL_MASS, DISTANCE, IOTA, PHASE, **waveform_arguments
    )
    peak_time = time[np.argmax(np.abs(hplus - 1j * hcross))]
    assert abs(peak_time) <= 2.0 / SAMPLING_FREQUENCY


def test_time_domain_is_uniformly_sampled(patched_loader, waveform_arguments):
    patched_loader(make_waveform())
    time, _, _ = bilby_plugin.nr_time_domain_polarisations(
        TOTAL_MASS, DISTANCE, IOTA, PHASE, **waveform_arguments
    )
    assert np.allclose(np.diff(time), 1.0 / SAMPLING_FREQUENCY)


def test_time_domain_amplitude_scales_inversely_with_distance(
    patched_loader, waveform_arguments
):
    patched_loader(make_waveform())
    _, hplus_near, _ = bilby_plugin.nr_time_domain_polarisations(
        TOTAL_MASS, DISTANCE, IOTA, PHASE, **waveform_arguments
    )
    _, hplus_far, _ = bilby_plugin.nr_time_domain_polarisations(
        TOTAL_MASS, 2 * DISTANCE, IOTA, PHASE, **waveform_arguments
    )
    assert np.max(np.abs(hplus_near)) > 0.0
    assert np.allclose(hplus_far, 0.5 * hplus_near, rtol=1e-12, atol=0.0)


def test_t_after_peak_trims_the_record(patched_loader, waveform_arguments):
    patched_loader(make_waveform())
    time_full, _, _ = bilby_plugin.nr_time_domain_polarisations(
        TOTAL_MASS, DISTANCE, IOTA, PHASE, **waveform_arguments
    )
    time_cut, _, _ = bilby_plugin.nr_time_domain_polarisations(
        TOTAL_MASS,
        DISTANCE,
        IOTA,
        PHASE,
        **{**waveform_arguments, "t_after_peak": 100.0},
    )
    assert time_cut[-1] < time_full[-1]
    # 100 M after the peak, in seconds. The cut lands on the source grid (0.5 M)
    # and the resampling then truncates to the last whole 1/srate step.
    geometric_to_seconds = TOTAL_MASS * ut.consts["Msun"]
    assert time_cut[-1] == pytest.approx(
        100.0 * geometric_to_seconds,
        abs=0.5 * geometric_to_seconds + 2.0 / SAMPLING_FREQUENCY,
    )


def test_mode_array_selects_a_subset(patched_loader, waveform_arguments):
    patched_loader(make_waveform())
    _, hplus_all, _ = bilby_plugin.nr_time_domain_polarisations(
        TOTAL_MASS, DISTANCE, IOTA, PHASE, **waveform_arguments
    )
    _, hplus_22, _ = bilby_plugin.nr_time_domain_polarisations(
        TOTAL_MASS,
        DISTANCE,
        IOTA,
        PHASE,
        **{**waveform_arguments, "mode_array": [(2, 2), (2, -2)]},
    )
    # atol=0: the strain is of order 1e-21, so the default atol=1e-8 would call
    # any two of these arrays equal
    assert not np.allclose(hplus_all, hplus_22, atol=0.0)
    peak = np.max(np.abs(hplus_all))
    assert np.max(np.abs(hplus_all - hplus_22)) > 1e-3 * peak


def test_unknown_mode_raises(patched_loader, waveform_arguments):
    patched_loader(make_waveform())
    with pytest.raises(KeyError, match=r"\(5, 5\)"):
        bilby_plugin.nr_time_domain_polarisations(
            TOTAL_MASS,
            DISTANCE,
            IOTA,
            PHASE,
            **{**waveform_arguments, "mode_array": [(5, 5)]},
        )


##############################
# frequency-domain output
##############################


def test_frequency_domain_is_zero_outside_the_band(
    patched_loader, waveform_arguments, frequency_array
):
    patched_loader(make_waveform())
    strain = bilby_plugin.nr_frequency_domain_source_model(
        frequency_array, TOTAL_MASS, DISTANCE, IOTA, PHASE, **waveform_arguments
    )
    outside = (frequency_array < MINIMUM_FREQUENCY) | (
        frequency_array > MAXIMUM_FREQUENCY
    )
    assert np.all(strain["plus"][outside] == 0.0)
    assert np.all(strain["cross"][outside] == 0.0)
    assert np.any(strain["plus"][~outside] != 0.0)


def test_frequency_domain_shape_matches_the_input_grid(
    patched_loader, waveform_arguments, frequency_array
):
    patched_loader(make_waveform())
    strain = bilby_plugin.nr_frequency_domain_source_model(
        frequency_array, TOTAL_MASS, DISTANCE, IOTA, PHASE, **waveform_arguments
    )
    assert strain["plus"].shape == frequency_array.shape
    assert strain["cross"].shape == frequency_array.shape
    assert np.iscomplexobj(strain["plus"])


def test_frequency_domain_raises_on_a_record_shorter_than_the_taper(
    patched_loader, waveform_arguments, frequency_array
):
    # the Stage-1 taper is ~0.51 s here; 300 M at TOTAL_MASS is only ~0.44 s
    patched_loader(make_waveform(u_max=300.0))
    with pytest.raises(ValueError, match="conditioning wants to taper"):
        bilby_plugin.nr_frequency_domain_source_model(
            frequency_array, TOTAL_MASS, DISTANCE, IOTA, PHASE, **waveform_arguments
        )


##############################
# loading, caching, dispatch
##############################


def test_unknown_catalog_raises_a_plain_error():
    with pytest.raises(NotImplementedError) as excinfo:
        bilby_plugin.load_nr_waveform(catalog="nonesuch", ID="0001", path="/tmp")

    message = str(excinfo.value)
    assert "not implemented" in message
    assert "nonesuch" in message
    # the message used to name a maintainer and tell the user to pester them
    assert "Koustav" not in message


def test_clear_waveform_cache_empties_the_cache():
    bilby_plugin._WAVEFORM_CACHE[("sentinel",)] = object()
    assert bilby_plugin._WAVEFORM_CACHE
    bilby_plugin.clear_waveform_cache()
    assert not bilby_plugin._WAVEFORM_CACHE


def test_loader_is_called_once_per_source_model_evaluation(
    patched_loader, waveform_arguments, frequency_array
):
    calls = patched_loader(make_waveform())
    bilby_plugin.nr_frequency_domain_source_model(
        frequency_array, TOTAL_MASS, DISTANCE, IOTA, PHASE, **waveform_arguments
    )
    # once in the FD model for the metadata, once in the TD helper
    assert calls["n"] == 2


##############################
# the per-catalogue wrappers
##############################


@pytest.mark.parametrize(
    "wrapper, catalog",
    [
        (bilby_plugin.sxs_frequency_domain_source_model, "sxs"),
        (bilby_plugin.gra_frequency_domain_source_model, "gra"),
    ],
)
def test_catalogue_wrappers_pin_the_catalog(
    wrapper, catalog, patched_loader, waveform_arguments, frequency_array, monkeypatch
):
    patched_loader(make_waveform())
    seen = {}

    def spy(frequency_array, *args, **kwargs):
        seen.update(kwargs)
        return dict(plus=frequency_array * 0j, cross=frequency_array * 0j)

    monkeypatch.setattr(bilby_plugin, "nr_frequency_domain_source_model", spy)

    arguments = {k: v for k, v in waveform_arguments.items() if k != "catalog"}
    wrapper(frequency_array, TOTAL_MASS, DISTANCE, IOTA, PHASE, **arguments)

    assert seen["catalog"] == catalog


@pytest.mark.parametrize(
    "wrapper",
    [
        bilby_plugin.sxs_frequency_domain_source_model,
        bilby_plugin.gra_frequency_domain_source_model,
    ],
)
def test_catalogue_wrappers_do_not_mutate_the_caller_arguments(
    wrapper, patched_loader, waveform_arguments, frequency_array
):
    patched_loader(make_waveform())
    arguments = {k: v for k, v in waveform_arguments.items() if k != "catalog"}
    before = dict(arguments)

    wrapper(frequency_array, TOTAL_MASS, DISTANCE, IOTA, PHASE, **arguments)

    assert arguments == before
    assert "catalog" not in arguments


@pytest.mark.parametrize(
    "wrapper",
    [
        bilby_plugin.sxs_frequency_domain_source_model,
        bilby_plugin.gra_frequency_domain_source_model,
    ],
)
def test_catalogue_wrappers_keep_the_generic_signature(wrapper):
    """
    bilby introspects the source model to decide which sampled parameters to
    pass, so the wrappers must expose the same parameters as the generic model.
    This is why they are plain functions and not functools.partial objects.
    """
    generic = inspect.signature(bilby_plugin.nr_frequency_domain_source_model)
    assert inspect.signature(wrapper).parameters.keys() == generic.parameters.keys()


@pytest.mark.parametrize(
    "wrapper, catalog",
    [
        (bilby_plugin.sxs_frequency_domain_source_model, "sxs"),
        (bilby_plugin.gra_frequency_domain_source_model, "gra"),
    ],
)
def test_catalogue_wrappers_match_the_generic_model(
    wrapper, catalog, patched_loader, waveform_arguments, frequency_array
):
    patched_loader(make_waveform())
    arguments = {k: v for k, v in waveform_arguments.items() if k != "catalog"}

    from_wrapper = wrapper(
        frequency_array, TOTAL_MASS, DISTANCE, IOTA, PHASE, **arguments
    )
    from_generic = bilby_plugin.nr_frequency_domain_source_model(
        frequency_array,
        TOTAL_MASS,
        DISTANCE,
        IOTA,
        PHASE,
        **{**arguments, "catalog": catalog},
    )

    assert np.array_equal(from_wrapper["plus"], from_generic["plus"])
    assert np.array_equal(from_wrapper["cross"], from_generic["cross"])
