"""
Test the Matcher class with a NR waveform
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from PyART.catalogs import sxs
from PyART.analysis.match import Matcher

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
