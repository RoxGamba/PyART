"""
Test the Matcher class with a NR waveform
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from PyART import waveform
from PyART.catalogs import sxs
from PyART.analysis.match import Matcher
from PyART.utils import wf_utils as wf_ut

sxs_id = "0180"
nr = sxs.Waveform_SXS(ID=sxs_id, download=True, ignore_deprecation=True)
nr_2 = sxs.Waveform_SXS(ID=sxs_id, download=False, ignore_deprecation=True)
nr.cut(300)
nr_2.cut(300)
M = 100
fmin = 5
fmax = 2048
srate = 8192


def test_self_match_ell_emms():
    """
    Test that the self-match of the single modes
    is 1.0 within numerical accuracy.
    """

    settings = {
        "kind": "single-mode",
        "initial_frequency_mm": fmin,
        "final_frequency_mm": fmax,
        "tlen": len(nr.u),
        "dt": 1 / srate,
        "M": M,
        "resize_factor": 4,
        "modes-or-pol": "modes",
        "pad_end_frac": 0.5,
        "taper_alpha": 0.2,
        "taper_start": 0.05,
        "taper": "sigmoid",
        "debug": False,
    }

    for mode in nr.hlm.keys():
        these_settings = settings.copy()
        these_settings["modes"] = [mode]
        m = Matcher(nr, nr_2, settings=settings)
        match = 1 - m.mismatch
        assert np.isclose(match, 1.0, atol=1e-7)


def test_self_match_pol():
    """
    Test that the self-match of the two polarizations
    is 1.0 within numerical accuracy.
    """

    settings = {
        "kind": "hm",
        "initial_frequency_mm": fmin,
        "final_frequency_mm": fmax,
        "tlen": len(nr.u),
        "dt": 1 / srate,
        "M": M,
        "resize_factor": 4,
        "pad_end_frac": 0.5,
        "taper_alpha": 0.2,
        "taper_start": 0.05,
        "taper": "sigmoid",
        "debug": True,
    }

    def test_self_match_pol_helper(cp, ep):
        """
        Helper function to test self-match for given
        coalescence phase and effective polarization.
        """

        settings_cp = settings.copy()
        settings_cp["coa_phase"] = [cp]
        settings_cp["eff_pols"] = [ep]

        m = Matcher(nr, nr_2, settings=settings_cp)
        match = 1 - m.mismatch
        assert np.isclose(match, 1.0, atol=1e-7)
        print(f"cp: {cp:.2f}, ep: {ep:.2f}, match: {match}")

    for cp in [0, np.pi / 4, np.pi / 2]:
        for ep in [0, np.pi / 4, np.pi / 2]:
            test_self_match_pol_helper(cp, ep)

    pass


def test_skymax_averages_over_the_whole_grid(monkeypatch):
    """
    _compute_mm_skymax loops over coa_phase x eff_pols and accumulates the
    matches in `mms`, but used to return np.average(mm) -- the last scalar --
    throwing the loop away. The result must be the average over the grid.
    """
    fake_matches = [0.90, 0.92, 0.94, 0.98]
    calls = []

    def fake_skymax_match(self, s, wf, inc, psd, modes, **kwargs):
        calls.append(1)
        return fake_matches[len(calls) - 1]

    monkeypatch.setattr(Matcher, "skymax_match", fake_skymax_match)

    settings = {
        "kind": "hm",
        "initial_frequency_mm": fmin,
        "final_frequency_mm": fmax,
        "tlen": len(nr.u),
        "dt": 1 / srate,
        "M": M,
        "resize_factor": 4,
        "pad_end_frac": 0.5,
        "taper_alpha": 0.2,
        "taper_start": 0.05,
        "taper": "sigmoid",
        "debug": False,
        "coa_phase": [0.0, np.pi / 2],
        "eff_pols": [0.0, np.pi / 3],
    }

    m = Matcher(nr, nr_2, settings=settings)

    assert len(calls) == 4, "the grid should be 2 coa_phase x 2 eff_pols"
    # mismatch = 1 - <match>, averaged over the whole grid
    assert m.mismatch == pytest.approx(1 - np.mean(fake_matches))
    # and specifically not just the last entry
    assert m.mismatch != pytest.approx(1 - fake_matches[-1])


def test_skymax_single_point_grid(monkeypatch):
    """With one coa_phase and one eff_pol the average is that single value."""

    def fake_skymax_match(self, s, wf, inc, psd, modes, **kwargs):
        return 0.75

    monkeypatch.setattr(Matcher, "skymax_match", fake_skymax_match)

    settings = {
        "kind": "hm",
        "initial_frequency_mm": fmin,
        "final_frequency_mm": fmax,
        "tlen": len(nr.u),
        "dt": 1 / srate,
        "M": M,
        "resize_factor": 4,
        "pad_end_frac": 0.5,
        "taper_alpha": 0.2,
        "taper_start": 0.05,
        "taper": "sigmoid",
        "debug": False,
        "coa_phase": [0.3],
        "eff_pols": [0.4],
    }

    m = Matcher(nr, nr_2, settings=settings)
    assert m.mismatch == pytest.approx(0.25)


def base_single_mode_settings(**overrides):
    settings = {
        "kind": "single-mode",
        "modes-or-pol": "modes",
        "modes": [(2, 2)],
        "initial_frequency_mm": fmin,
        "final_frequency_mm": fmax,
        "tlen": len(nr.u),
        "dt": 1 / srate,
        "M": M,
        "resize_factor": 4,
        "pad_end_frac": 0.5,
        "taper_alpha": 0.2,
        "taper_start": 0.05,
        "taper": "sigmoid",
        "debug": False,
    }
    settings.update(overrides)
    return settings


@pytest.mark.parametrize("option", ["cut_longer", "cut_second_waveform"])
def test_cut_options_run(option):
    """
    These options had no coverage at all.

    They used to read 'tmrg, _, _, _ = WaveForm.find_max() - WaveForm.u[0]',
    which survives only because find_max returns np.float64 entries: numpy
    broadcasts the tuple and subtracts u[0] from all four, the last three being
    discarded. The merger time is now taken explicitly, which is equivalent for
    a numpy time array and does not depend on that accident.
    """
    wf1 = sxs.Waveform_SXS(ID=sxs_id, download=False, ignore_deprecation=True)
    wf2 = sxs.Waveform_SXS(ID=sxs_id, download=False, ignore_deprecation=True)
    wf1.cut(300)
    wf2.cut(500)  # different start -> non-zero DeltaT

    m = Matcher(wf1, wf2, settings=base_single_mode_settings(**{option: True}))
    assert np.isfinite(m.mismatch)


def test_cut_options_are_mutually_exclusive():
    wf1 = sxs.Waveform_SXS(ID=sxs_id, download=False, ignore_deprecation=True)
    wf2 = sxs.Waveform_SXS(ID=sxs_id, download=False, ignore_deprecation=True)
    wf1.cut(300)
    wf2.cut(300)

    with pytest.raises(RuntimeError, match="cannot be used together"):
        Matcher(
            wf1,
            wf2,
            settings=base_single_mode_settings(
                cut_longer=True, cut_second_waveform=True
            ),
        )


def test_single_mode_with_cached_h2f():
    """
    The output dict referenced h2.delta_t, but h2 only exists when wf2 is in the
    time domain and was not served from the cache -> NameError.
    """
    settings = base_single_mode_settings()

    # first pass: populate the cache the way callers are meant to
    m0 = Matcher(nr, nr_2, settings=settings)
    cache = {"h1f": m0.h1f, "h2f": m0.h2f, "M": M}

    wf1 = sxs.Waveform_SXS(ID=sxs_id, download=False, ignore_deprecation=True)
    wf2 = sxs.Waveform_SXS(ID=sxs_id, download=False, ignore_deprecation=True)
    wf1.cut(300)
    wf2.cut(300)

    m = Matcher(wf1, wf2, settings=base_single_mode_settings(), cache=cache)
    assert np.isfinite(m.mismatch)


##############################
# 'fAM' initial_frequency_mm string parsing
##############################


def test_initial_frequency_mm_fAM_string():
    """
    'initial_frequency_mm' can be given as e.g. '2fAM' (a multiplier of the
    frequency at the amplitude peak) or '2fAM20' (also floored at 20 Hz). This
    used to be parsed with eval(); it is now float(), which is all a plain
    numeric coefficient needs.
    """
    settings = base_single_mode_settings(initial_frequency_mm="1fAM")
    m = Matcher(nr, nr_2, settings=settings)
    assert np.isfinite(m.mismatch)
    # Matcher.__init__ copies settings into self.settings (a separate dict),
    # so the caller's dict is untouched; the resolved value lives on the
    # Matcher instance.
    assert isinstance(m.settings["initial_frequency_mm"], float)
    assert isinstance(settings["initial_frequency_mm"], str)


def test_initial_frequency_mm_fAM_string_with_floor():
    settings = base_single_mode_settings(initial_frequency_mm="1fAM1")
    m = Matcher(nr, nr_2, settings=settings)
    assert np.isfinite(m.mismatch)
    assert isinstance(m.settings["initial_frequency_mm"], float)


def test_initial_frequency_mm_fAM_string_rejects_non_numeric():
    """A malformed multiplier must fail loudly (ValueError), not execute code."""
    settings = base_single_mode_settings(initial_frequency_mm="__import__('os')fAM")
    with pytest.raises(ValueError):
        Matcher(nr, nr_2, settings=settings)


##############################
# _pre_align (extracted from Matcher.__init__ for direct testing: it had no
# dedicated coverage before, only ever exercised incidentally by self-match
# tests where the two waveforms are already identical -- a case where the
# time shift and phase correction it computes are both trivially ~0)
##############################


def _make_chirp_waveform(t0, extra_phase, modes, chirp_rate=0.01):
    """
    A synthetic multi-mode "waveform": each mode m has an amplitude envelope
    that peaks once (a clean merger for find_max) and a phase m*(phi(t) +
    extra_phase) -- extra_phase mimics an arbitrary coalescence-phase choice,
    which _pre_align is meant to remove. t0 shifts the whole thing in time;
    chirp_rate lets two waveforms differ in frequency evolution, not just by
    a constant phase offset (see test_pre_align_shift_setting_moves_reference_time).
    """
    # odd sample count: guarantees a single unambiguous center sample at
    # t_phys=0 for the (symmetric) amplitude peak, so find_max's discrete
    # peak search is not a coin flip between two equally-tall neighbors
    t_phys = np.linspace(-50.0, 50.0, 4001)
    A = np.exp(-(t_phys**2) / (2 * 8.0**2)) + 0.1
    phi = chirp_rate * t_phys**2 + 0.3 * t_phys

    wf = waveform.Waveform()
    wf._u = t0 + t_phys
    wf._t = wf._u.copy()
    wf._domain = "Time"
    wf._hlm = {}
    for ell, emm in modes:
        z = A * np.exp(-1j * emm * (phi + extra_phase))
        wf._hlm[(ell, emm)] = wf_ut.get_multipole_dict(z)
    return wf


def _make_matcher_for_pre_align(modes, pre_align_shift=0.0):
    """A Matcher with just enough state for _pre_align, skipping __init__."""
    m = Matcher.__new__(Matcher)
    m.settings = {"modes": modes, "pre_align_shift": pre_align_shift}
    return m


def test_pre_align_shifts_time_to_coincide_mergers():
    modes = [(2, 2), (3, 3)]
    time_shift = 37.0
    wf1 = _make_chirp_waveform(t0=1000.0, extra_phase=0.0, modes=modes)
    wf2 = _make_chirp_waveform(t0=1000.0 + time_shift, extra_phase=0.6, modes=modes)

    matcher = _make_matcher_for_pre_align(modes)
    matcher._pre_align(wf1, wf2)

    assert np.allclose(wf2.u, wf1.u)


def test_pre_align_corrects_per_mode_phase_offset():
    """
    dphi22, measured from the (2,2) mode alone, must correct every requested
    mode by exp(-1j * dphi22/2 * m): exact for a pure coalescence-phase
    offset (every mode m carries m*extra_phase), which is what this
    synthetic pair has. Both the aligning (2,2) mode and an independent
    (3,3) mode must come back essentially identical to wf1's.
    """
    modes = [(2, 2), (3, 3)]
    extra_phase = 0.6
    wf1 = _make_chirp_waveform(t0=1000.0, extra_phase=0.0, modes=modes)
    wf2 = _make_chirp_waveform(t0=1037.0, extra_phase=extra_phase, modes=modes)

    # sanity: before alignment, the modes genuinely disagree (this is not a
    # vacuous check -- there is a real offset for _pre_align to remove)
    assert not np.allclose(wf2.hlm[(2, 2)]["z"], wf1.hlm[(2, 2)]["z"], atol=1e-3)
    assert not np.allclose(wf2.hlm[(3, 3)]["z"], wf1.hlm[(3, 3)]["z"], atol=1e-3)

    matcher = _make_matcher_for_pre_align(modes)
    matcher._pre_align(wf1, wf2)

    assert np.allclose(wf2.hlm[(2, 2)]["z"], wf1.hlm[(2, 2)]["z"], atol=1e-8)
    assert np.allclose(wf2.hlm[(3, 3)]["z"], wf1.hlm[(3, 3)]["z"], atol=1e-8)


def test_pre_align_shift_setting_moves_reference_time():
    """
    pre_align_shift moves the reference time used to measure dphi22. Give
    wf1/wf2 different chirp rates (not just a constant coalescence-phase
    offset): the true phase difference then genuinely depends on time, so
    measuring dphi22 at a different reference time must change the
    (approximate, single-time-sample) correction applied.
    """
    modes = [(2, 2)]
    wf1 = _make_chirp_waveform(t0=1000.0, extra_phase=0.0, modes=modes, chirp_rate=0.01)

    matcher_a = _make_matcher_for_pre_align(modes, pre_align_shift=0.0)
    matcher_b = _make_matcher_for_pre_align(modes, pre_align_shift=20.0)

    wf2_a = _make_chirp_waveform(
        t0=1000.0, extra_phase=0.6, modes=modes, chirp_rate=0.012
    )
    wf2_b = _make_chirp_waveform(
        t0=1000.0, extra_phase=0.6, modes=modes, chirp_rate=0.012
    )

    matcher_a._pre_align(wf1, wf2_a)
    matcher_b._pre_align(wf1, wf2_b)

    # the two results must differ from each other, even though both start
    # from an identical wf2 -- only the reference time differs.
    assert not np.allclose(wf2_a.hlm[(2, 2)]["z"], wf2_b.hlm[(2, 2)]["z"])
