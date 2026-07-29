"""
Tests for the SXS catalog.
"""

from PyART.catalogs import sxs
from PyART.waveform import Waveform
import json
import os
import sys

import numpy as np
import pytest

mode_keys = ["A", "p", "real", "imag", "z"]


def test_sxs():
    """
    Test the SXS download function.
    """
    opts = {
        "ID": "0180",
        "path": "./",
        "download": True,
        "downloads": ["hlm", "metadata", "horizons"],
        "load": ["hlm", "metadata", "horizons"],
        "level": 4,
        "order": 2,
        "nu_rescale": False,
        "ignore_deprecation": True,
    }
    wf = sxs.Waveform_SXS(**opts)

    # check attributes
    assert wf.ID == "0180"
    assert wf.level == 4

    # check that the files were downloaded
    assert os.path.exists("SXS_BBH_0180")
    assert os.path.exists(
        f"SXS_BBH_0180/Lev{wf.level}/rhOverM_Asymptotic_GeometricUnits_CoM.h5"
    )
    assert os.path.exists(f"SXS_BBH_0180/Lev{wf.level}/metadata.json")
    assert os.path.exists(f"SXS_BBH_0180/Lev{wf.level}/Horizons.h5")

    # Check that the colon-named folder the sxs module downloads into was
    # cleaned up, leaving only SXS_BBH_0180. Look in the directory we asked to
    # download into rather than at SXSCACHEDIR: that variable is only set when a
    # download actually happens, so asserting on it made this test pass on a
    # clean tree and fail on every rerun, once the data is already there.
    for fld in os.listdir(opts["path"]):
        assert not fld.startswith(
            "SXS:BBH:0180"
        ), f"Old folder {fld} still exists in {opts['path']}."

    # check that the modes loaded make sense
    for mode in wf.hlm.keys():

        # check ell, emm
        assert mode[0] >= abs(mode[1])
        # check keys
        for key in mode_keys:
            assert key in wf.hlm[mode].keys()
        # check length
        assert len(wf.hlm[mode]["A"]) == len(wf.u)

    # get also order=3
    opts["order"] = 3
    wf = sxs.Waveform_SXS(**opts)

    # check that conversion to LVKNR works
    wf.to_lvk(modes=[(2, 2)])


##############################
# metadata parsing
##############################
#
# The bundled SXS:BBH:0180 carries dimensionless spins and a float
# eccentricity, so it exercises none of the branches below. These tests drive
# load_metadata from synthetic metadata.json files instead.

M1 = 0.6
M2 = 0.4


def base_metadata(**overrides):
    """A minimal SXS-like metadata dict, with the keys load_metadata needs."""
    meta = {
        "reference_mass1": M1,
        "reference_mass2": M2,
        "reference_dimensionless_spin1": [0.0, 0.0, 0.3],
        "reference_dimensionless_spin2": [0.0, 0.0, -0.2],
        "reference_position1": [5.0, 0.0, 0.0],
        "reference_position2": [-5.0, 0.0, 0.0],
        "remnant_mass": 0.95,
        "remnant_dimensionless_spin": [0.0, 0.0, 0.68],
        "alternative_names": ["SXS:BBH:0001", "SXS:BBH:0001"],
        "reference_eccentricity": 1e-4,
        "reference_orbital_frequency": [0.0, 0.0, 0.005],
        "reference_time": 250.0,
        "initial_ADM_energy": 0.99,
        "initial_ADM_linear_momentum": [0.0, 0.0, 0.0],
        "initial_ADM_angular_momentum": [0.0, 0.0, 1.1],
    }
    meta.update(overrides)
    return meta


def make_sxs_stub(tmp_path, meta):
    """A Waveform_SXS pointing at a synthetic metadata.json, without __init__."""
    lev = tmp_path / "Lev4"
    lev.mkdir(parents=True, exist_ok=True)
    (lev / "metadata.json").write_text(json.dumps(meta))

    wf = sxs.Waveform_SXS.__new__(sxs.Waveform_SXS)
    Waveform.__init__(wf)
    wf.sxs_data_path = str(tmp_path)
    wf.level = 4
    wf.ID = "0001"
    wf.src = "BBH"
    wf.order = 2
    wf.ellmax = 4
    wf.nu_rescale = False
    wf.cut_N = None
    wf.cut_U = 0.0
    wf.basename = "rhOverM_Asymptotic_GeometricUnits_CoM.h5"
    return wf


def test_dimensionless_spins_are_used_as_is(tmp_path):
    wf = make_sxs_stub(tmp_path, base_metadata())
    wf.load_metadata()

    assert wf.metadata["chi1z"] == pytest.approx(0.3)
    assert wf.metadata["chi2z"] == pytest.approx(-0.2)


def test_dimensionful_reference_spin_is_normalized_by_mass_squared(tmp_path):
    """
    'reference_spin' is dimensionful and must be divided by the mass squared;
    the 'attempt == "reference_"' check could never match the actual key, so
    these spins were silently left un-normalized.
    """
    S1z, S2z = 0.3 * M1**2, -0.2 * M2**2
    meta = base_metadata(
        reference_spin1=[0.0, 0.0, S1z],
        reference_spin2=[0.0, 0.0, S2z],
    )
    # remove the dimensionless entries so the fallback is used
    del meta["reference_dimensionless_spin1"]
    del meta["reference_dimensionless_spin2"]

    wf = make_sxs_stub(tmp_path, meta)
    wf.load_metadata()

    assert wf.metadata["chi1z"] == pytest.approx(0.3)
    assert wf.metadata["chi2z"] == pytest.approx(-0.2)


def test_initial_dimensionless_spin_fallback(tmp_path):
    meta = base_metadata(
        initial_dimensionless_spin1=[0.0, 0.0, 0.5],
        initial_dimensionless_spin2=[0.0, 0.0, -0.4],
    )
    del meta["reference_dimensionless_spin1"]
    del meta["reference_dimensionless_spin2"]

    wf = make_sxs_stub(tmp_path, meta)
    wf.load_metadata()

    assert wf.metadata["chi1z"] == pytest.approx(0.5)
    assert wf.metadata["chi2z"] == pytest.approx(-0.4)


def test_missing_spin_entries_raise(tmp_path):
    """hS used to be unbound when no attempt matched -> UnboundLocalError."""
    meta = base_metadata()
    del meta["reference_dimensionless_spin1"]
    del meta["reference_dimensionless_spin2"]

    wf = make_sxs_stub(tmp_path, meta)
    with pytest.raises(KeyError, match="No valid spin entry"):
        wf.load_metadata()


@pytest.mark.parametrize(
    "ecc_str, expected",
    [
        ("<1.7e+00", None),  # bound of order unity: no information
        ("<1e-04", 1e-5),  # small bound: effectively circular
        ("<3.2e-03", 1e-5),
    ],
)
def test_string_eccentricity(tmp_path, ecc_str, expected):
    """
    'if "<" in ecc and "e+00"' had a constant truthy second operand, so every
    string eccentricity became None -- including the small bounds that should
    be read as quasi-circular.
    """
    wf = make_sxs_stub(tmp_path, base_metadata(reference_eccentricity=ecc_str))
    wf.load_metadata()

    assert wf.metadata["e0"] == expected


def test_float_eccentricity_is_kept(tmp_path):
    wf = make_sxs_stub(tmp_path, base_metadata(reference_eccentricity=5.11e-05))
    wf.load_metadata()

    assert wf.metadata["e0"] == pytest.approx(5.11e-05)


##############################
# psi4 loading
##############################


def test_load_psi4lm_missing_file_raises_filenotfound(tmp_path):
    """
    nr_psi was only assigned when the file existed, but used unconditionally,
    giving AttributeError instead of naming the missing file.
    """
    wf = make_sxs_stub(tmp_path, base_metadata())
    wf.load_metadata()

    with pytest.raises(FileNotFoundError, match="psi4 file not found"):
        wf.load_psi4lm()


def test_load_psi4lm_without_hlm_raises_runtime(tmp_path):
    """psi4 times are copied from the hlm time array, which must exist."""
    import h5py

    wf = make_sxs_stub(tmp_path, base_metadata())
    wf.load_metadata()

    # a psi4 file that exists, so we get past the FileNotFoundError above
    psi4_name = wf.basename.replace("rhOverM", "rMPsi4")
    fname = tmp_path / "Lev4" / psi4_name
    with h5py.File(fname, "w") as f:
        grp = f.create_group(f"Extrapolated_N{wf.order}.dir")
        t = np.linspace(0.0, 100.0, 50)
        grp["Y_l2_m2.dat"] = np.column_stack((t, np.sin(t), np.cos(t)))

    assert wf.u is None  # hlm never loaded
    with pytest.raises(RuntimeError, match="hlm was never loaded"):
        wf.load_psi4lm()


##############################
# download stdout redirect
##############################


def test_download_simulation_restores_stdout_on_failure(tmp_path, monkeypatch):
    """
    download_simulation redirects stdout/stderr to devnull around the noisy sxs
    calls. Without try/finally, any failure in between left the *whole process*
    writing to devnull, silencing everything afterwards.
    """
    import sxs as sxsmod

    def boom(*args, **kwargs):
        raise RuntimeError("simulated sxs.load failure")

    monkeypatch.setattr(sxsmod, "load", boom)

    wf = sxs.Waveform_SXS.__new__(sxs.Waveform_SXS)
    wf.src = "BBH"
    wf.level = 4
    wf.order = 2

    saved_stdout, saved_stderr = sys.stdout, sys.stderr

    with pytest.raises(RuntimeError, match="simulated sxs.load failure"):
        wf.download_simulation(ID="0001", path=str(tmp_path), extrapolation_order=2)

    assert sys.stdout is saved_stdout, "stdout left redirected to devnull"
    assert sys.stderr is saved_stderr, "stderr left redirected to devnull"


##############################
# horizon loading
##############################


REMNANT_MASS = 0.95
REMNANT_CHI = [0.0, 0.0, 0.68]


def write_horizons(
    tmp_path,
    chi1,
    chi2,
    mass=0.5,
    omega=0.05,
    radius=5.0,
    nt=400,
    remnant=True,
):
    """
    A synthetic Horizons.h5 for a circular orbit in the xy-plane, so that the
    orbital angular momentum points along +z and the spin projections are known
    analytically.

    The columns follow the SXS layout: time first, so ChristodoulouMass.dat and
    DimensionfulInertialSpinMag.dat are (N,2) while CoordCenterInertial.dat and
    chiInertial.dat are (N,4). The spin magnitudes are kept consistent with the
    dimensionless spins, |S| = |chi| m^2.

    AhC, the common horizon, exists only after merger, so it is written on its
    own later and shorter time array. remnant=False omits it, as happens for
    runs that do not merge.
    """
    import h5py

    def add(f, obj, t, chi, x, m):
        chi = np.tile(np.asarray(chi, dtype=float), (len(t), 1))
        grp = f.create_group(obj)
        grp["ChristodoulouMass.dat"] = np.column_stack((t, np.full_like(t, m)))
        grp["CoordCenterInertial.dat"] = np.column_stack((t, x))
        grp["chiInertial.dat"] = np.column_stack((t, chi))
        grp["DimensionfulInertialSpinMag.dat"] = np.column_stack(
            (t, np.linalg.norm(chi, axis=1) * m**2)
        )

    t = np.linspace(0.0, 100.0, nt)
    x1 = np.column_stack(
        (radius * np.cos(omega * t), radius * np.sin(omega * t), np.zeros_like(t))
    )
    x2 = -x1

    fname = tmp_path / "Lev4" / "Horizons.h5"
    fname.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(fname, "w") as f:
        add(f, "AhA.dir", t, chi1, x1, mass)
        add(f, "AhB.dir", t, chi2, x2, mass)
        if remnant:
            t_rem = np.linspace(100.0, 150.0, nt // 4)
            x_rem = np.zeros((len(t_rem), 3))
            add(f, "AhC.dir", t_rem, REMNANT_CHI, x_rem, REMNANT_MASS)
    return t


def test_load_horizon_reads_dimensionless_spin_vector(tmp_path):
    """
    dyn["chi"] must come from chiInertial (the dimensionless spin vector), not
    from DimensionfulInertialSpinMag, which is |S|: dimensionful by m^2 and a
    magnitude, so it carries no direction.
    """
    chi1 = [0.3, 0.0, 0.4]
    chi2 = [0.0, 0.1, -0.2]
    write_horizons(tmp_path, chi1, chi2)

    wf = make_sxs_stub(tmp_path, base_metadata())
    wf.load_horizon()
    d = wf.dyn

    # full 3-vectors, time stripped and kept once in dyn["t"]
    assert np.shape(d["chi1"])[1] == 3
    assert np.shape(d["chi2"])[1] == 3
    assert np.shape(d["x1"])[1] == 3

    assert np.allclose(d["chi1"], chi1)
    assert np.allclose(d["chi2"], chi2)


def test_load_horizon_time_and_masses(tmp_path):
    t = write_horizons(tmp_path, [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], mass=0.4)

    wf = make_sxs_stub(tmp_path, base_metadata())
    wf.load_horizon()
    d = wf.dyn

    assert np.allclose(d["t"], t)
    assert len(d["m1"]) == len(t)
    assert np.allclose(d["m1"], 0.4)
    assert np.allclose(d["m2"], 0.4)


def test_compute_spins_at_tref_circular_orbit(tmp_path):
    """
    For a circular orbit in the xy-plane L is along +z, so the parallel spin
    component is chi_z and the perpendicular one is the in-plane norm.
    """
    chi1 = [0.3, 0.0, 0.4]
    chi2 = [0.0, 0.1, -0.2]
    write_horizons(tmp_path, chi1, chi2)

    wf = make_sxs_stub(tmp_path, base_metadata())
    wf.load_horizon()

    chi1_L, chi1_perp, chi2_L, chi2_perp = wf.compute_spins_at_tref(50.0)

    assert chi1_L == pytest.approx(0.4, abs=1e-8)
    assert chi1_perp == pytest.approx(0.3, abs=1e-8)
    assert chi2_L == pytest.approx(-0.2, abs=1e-8)
    assert chi2_perp == pytest.approx(0.1, abs=1e-8)


def test_load_horizon_spin_magnitudes(tmp_path):
    """
    The dimensionful spin magnitude is loaded alongside the dimensionless spin
    vector; the two are related by chi = S/m^2.
    """
    chi1 = [0.3, 0.0, 0.4]  # |chi1| = 0.5
    chi2 = [0.0, 0.1, -0.2]
    mass = 0.5
    write_horizons(tmp_path, chi1, chi2, mass=mass)

    wf = make_sxs_stub(tmp_path, base_metadata())
    wf.load_horizon()
    d = wf.dyn

    # a magnitude: one scalar per time
    assert np.shape(d["S1_mag"]) == np.shape(d["t"])

    assert np.allclose(d["S1_mag"], np.linalg.norm(chi1) * mass**2)
    assert np.allclose(d["S1_mag"] / d["m1"] ** 2, np.linalg.norm(d["chi1"], axis=1))
    assert np.allclose(d["S2_mag"] / d["m2"] ** 2, np.linalg.norm(d["chi2"], axis=1))


def test_load_horizon_remnant(tmp_path):
    """
    AhC is the common horizon: it forms at merger, so it carries its own time
    array rather than sharing dyn["t"].
    """
    write_horizons(tmp_path, [0.0, 0.0, 0.1], [0.0, 0.0, 0.2])

    wf = make_sxs_stub(tmp_path, base_metadata())
    wf.load_horizon()
    d = wf.dyn

    assert np.allclose(d["m_remnant"], REMNANT_MASS)
    assert np.allclose(d["chi_remnant"], REMNANT_CHI)
    assert np.allclose(
        d["S_remnant_mag"], np.linalg.norm(REMNANT_CHI) * REMNANT_MASS**2
    )
    assert np.shape(d["chi_remnant"])[1] == 3
    assert np.shape(d["x_remnant"])[1] == 3

    # its own, later time array
    assert len(d["t_remnant"]) != len(d["t"])
    assert d["t_remnant"][0] >= d["t"][-1]


def test_load_horizon_without_remnant(tmp_path):
    """Runs that do not merge have no common horizon; that is not an error."""
    write_horizons(tmp_path, [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], remnant=False)

    wf = make_sxs_stub(tmp_path, base_metadata())
    wf.load_horizon()
    d = wf.dyn

    assert "chi1" in d  # the binary is still loaded
    assert "m_remnant" not in d
    assert "chi_remnant" not in d


def test_download_simulation_requires_a_path():
    """
    path is both the download destination and the sxs cache directory. It
    defaults to None, which used to die further down with a cryptic
    "'NoneType' object has no attribute 'endswith'".
    """
    wf = sxs.Waveform_SXS.__new__(sxs.Waveform_SXS)
    wf.src = "BBH"
    wf.level = 4
    wf.order = 2

    with pytest.raises(ValueError, match="needs a path"):
        wf.download_simulation(ID="0001", path=None)
