"""
Test Cataloger bookkeeping that does not require catalog data:
- ID handling in log messages
- merging of the per-process JSONs written by a parallel run
- mm_at_M settings handling
- figure saving in mm_vs_M
"""

import json
import logging

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PyART.catalogs.cataloger import Cataloger


@pytest.fixture(autouse=True)
def no_usetex():
    """
    cataloger.py sets usetex at import; the labels it draws cannot be rendered
    without a LaTeX installation, which is unrelated to what we test here.
    """
    old = matplotlib.rcParams["text.usetex"]
    matplotlib.rc("text", usetex=False)
    yield
    matplotlib.rc("text", usetex=old)


def make_cataloger(**attrs):
    """
    Build a Cataloger without running __init__, which would need real catalog
    data on disk.
    """
    cat = Cataloger.__new__(Cataloger)
    cat.path = "./"
    cat.catalog = "rit"
    cat.verbose = True
    cat.sim_list = []
    cat.data = {}
    cat.json_file = "mismatches.json"
    for key, value in attrs.items():
        setattr(cat, key, value)
    return cat


##############################
# ID handling in logs (#16)
##############################


@pytest.mark.parametrize("ID", ["q1", "0180", "ab", 1362, "SXS:BBH:0180"])
def test_get_Waveform_logs_id_verbatim(ID, caplog):
    """
    IDs may be strings; a format spec of :04 pads short strings on the right
    ('q1' -> 'q100'), reporting an ID that does not exist.
    """
    cat = make_cataloger(catalog="not-a-catalog")

    with caplog.at_level(logging.INFO):
        with pytest.raises(ValueError, match="Unknown catalog"):
            cat.get_Waveform(ID)

    assert f"ID:{ID}" in caplog.text


def test_get_Waveform_logs_short_string_id_without_padding(caplog):
    """Regression: the ':04' spec turned the RIT-style ID 'q1' into 'q100'."""
    cat = make_cataloger(catalog="not-a-catalog")

    with caplog.at_level(logging.INFO):
        with pytest.raises(ValueError):
            cat.get_Waveform("q1")

    assert "q100" not in caplog.text


##############################
# parallel JSON merge (#17)
##############################


def write_json(path, mismatches):
    with open(path, "w") as f:
        json.dump({"mismatches": mismatches}, f)


def test_collect_mismatch_jsons_keeps_all_process_results(tmp_path):
    """
    Each process writes only its own batch; the merge must retain every
    process's results rather than letting later files overwrite earlier ones.
    """
    base = tmp_path / "mismatches.json"
    tmp0 = tmp_path / "mismatches_0.json"
    tmp1 = tmp_path / "mismatches_1.json"

    # process 0 computed simA, process 1 computed simB
    write_json(tmp0, {"simA": {"mm_min": 0.1}})
    write_json(tmp1, {"simB": {"mm_min": 0.2}})

    cat = make_cataloger(json_file=str(base))
    merged = cat.collect_mismatch_jsons([str(tmp0), str(tmp1)])

    assert set(merged["mismatches"]) == {"simA", "simB"}
    assert merged["mismatches"]["simA"]["mm_min"] == 0.1
    assert merged["mismatches"]["simB"]["mm_min"] == 0.2

    # and it is what landed on disk
    with open(base) as f:
        on_disk = json.load(f)
    assert on_disk == merged

    # temporary files are cleaned up
    assert not tmp0.exists()
    assert not tmp1.exists()


def test_collect_mismatch_jsons_does_not_overwrite_existing_entries(tmp_path):
    """
    Entries already in the base file win: 'if key not in json_data' was always
    true (json_data holds 'mismatches', never the sim names), so a stale value
    from a temporary file could clobber a fresh one.
    """
    base = tmp_path / "mismatches.json"
    write_json(base, {"simA": {"mm_min": 999.0}})

    tmp0 = tmp_path / "mismatches_0.json"
    write_json(tmp0, {"simA": {"mm_min": 0.1}, "simB": {"mm_min": 0.2}})

    cat = make_cataloger(json_file=str(base))
    merged = cat.collect_mismatch_jsons([str(tmp0)])

    assert merged["mismatches"]["simA"]["mm_min"] == 999.0
    assert merged["mismatches"]["simB"]["mm_min"] == 0.2


def test_collect_mismatch_jsons_without_existing_base(tmp_path):
    """With no base file, the first temporary file seeds the merge."""
    base = tmp_path / "mismatches.json"
    tmp0 = tmp_path / "mismatches_0.json"
    tmp1 = tmp_path / "mismatches_1.json"
    write_json(tmp0, {"simA": {"mm_min": 0.1}})
    write_json(tmp1, {"simB": {"mm_min": 0.2}})

    cat = make_cataloger(json_file=str(base))
    merged = cat.collect_mismatch_jsons([str(tmp0), str(tmp1)])

    assert set(merged["mismatches"]) == {"simA", "simB"}
    assert base.exists()


##############################
# mm_at_M settings (#18)
##############################


def test_mm_at_M_does_not_mutate_caller_settings(monkeypatch):
    """mm_at_M sets 'M' for its own call; the caller's dict must be untouched."""
    cat = make_cataloger(data={"sim": {"Waveform": object(), "Optimizer": None}})
    monkeypatch.setattr(
        Cataloger, "get_model_waveform", lambda self, name, **kw: object()
    )

    seen = {}

    def fake_matcher(nr, eob, settings=None):
        seen.update(settings)
        return type("M", (), {"mismatch": 0.5})()

    monkeypatch.setattr("PyART.catalogs.cataloger.Matcher", fake_matcher)

    settings = {"kind": "single-mode"}
    mm = cat.mm_at_M("sim", 120.0, mm_settings=settings)

    assert mm == 0.5
    assert seen["M"] == 120.0
    assert "M" not in settings, "mm_at_M mutated the caller's settings dict"


def test_mm_at_M_accepts_default_settings(monkeypatch):
    """mm_settings defaults to None; that used to raise TypeError."""
    cat = make_cataloger(data={"sim": {"Waveform": object(), "Optimizer": None}})
    monkeypatch.setattr(
        Cataloger, "get_model_waveform", lambda self, name, **kw: object()
    )

    seen = {}

    def fake_matcher(nr, eob, settings=None):
        seen.update(settings)
        return type("M", (), {"mismatch": 0.25})()

    monkeypatch.setattr("PyART.catalogs.cataloger.Matcher", fake_matcher)

    mm = cat.mm_at_M("sim", 90.0)

    assert mm == 0.25
    assert seen == {"M": 90.0}


##############################
# mm_vs_M figure saving (#15)
##############################


@pytest.fixture
def cataloger_for_mm_vs_M(monkeypatch, tmp_path):
    """
    A Cataloger whose mismatches are pre-loaded from JSON, so mm_vs_M plots
    without generating any waveform.
    """
    masses = list(np.linspace(100, 200, num=5))
    mm_json = tmp_path / "loaded.json"
    with open(mm_json, "w") as f:
        json.dump(
            {
                "masses": masses,
                "options": {"mm_settings": {}},
                "mismatches": {
                    "sim": {
                        "mm_vs_M": [1e-3] * len(masses),
                        "mm_max": 1e-3,
                        "mm_min": 1e-3,
                    }
                },
            },
            f,
        )

    cat = make_cataloger()
    monkeypatch.setattr(Cataloger, "find_subset", lambda self, ranges=None: ["sim"])
    monkeypatch.setattr(
        Cataloger,
        "get_colors_for_subset",
        lambda self, subset, cmap_var=None, cmap_name=None: {
            "colors": ["C0"],
            "indices": [0],
            "range": (0.0, 1.0),
            "cmap": plt.get_cmap("jet"),
        },
    )
    monkeypatch.setattr(Cataloger, "tex_label_from_key", lambda self, key: key)
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    return cat, str(mm_json), len(masses)


def test_mm_vs_M_saves_user_supplied_figname(cataloger_for_mm_vs_M, tmp_path):
    """A caller-supplied figname must actually be written."""
    cat, mm_json, n = cataloger_for_mm_vs_M
    figname = tmp_path / "myfig.png"

    cat.mm_vs_M(N=n, json_load=mm_json, figname=str(figname))

    assert figname.exists(), "mm_vs_M did not save the user-supplied figname"


def test_mm_vs_M_saves_nothing_when_figname_is_none(
    cataloger_for_mm_vs_M, tmp_path, monkeypatch
):
    """figname=None means 'do not save', matching the json_save convention."""
    cat, mm_json, n = cataloger_for_mm_vs_M
    monkeypatch.chdir(tmp_path)

    cat.mm_vs_M(N=n, json_load=mm_json, figname=None)

    assert (
        list(tmp_path.glob("*.png")) == []
    ), "mm_vs_M saved a figure when figname was None"


def test_mm_vs_M_saves_json_when_requested(cataloger_for_mm_vs_M, tmp_path):
    cat, mm_json, n = cataloger_for_mm_vs_M
    out = tmp_path / "saved.json"

    cat.mm_vs_M(N=n, json_load=mm_json, json_save=str(out))

    assert out.exists()
    with open(out) as f:
        assert "sim" in json.load(f)["mismatches"]
