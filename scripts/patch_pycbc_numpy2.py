"""
Patch pycbc for numpy 2.x compatibility.

pycbc <= 2.10.0 has two numpy 2.x incompatibilities:

  1. events/threshold_cpu.py: np.array(..., copy=False) raises ValueError in
     numpy 2.x when a copy is required; numpy.asarray() preserves the original
     numpy 1.x semantics (copy only if necessary, silently).

  2. filter/matchedfilter.py: numpy.real(x) / numpy.imag(x) dispatch via
     x.real / x.imag. On pycbc Array objects these are plain methods, not
     properties, so numpy.real() returns the bound method instead of the data.
     Replacing with numpy.asarray(x).real / .imag forces conversion via
     Array.__array__ first, which is safe and localized to the affected calls.

Run once after installing pycbc:
    python scripts/patch_pycbc_numpy2.py
"""

import inspect
import re
import shutil
import sys
from pathlib import Path


def _clear_cache(path):
    cache_dir = path.parent / "__pycache__"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"  Cleared __pycache__ in {cache_dir.parent.name}/")


def patch_threshold_cpu():
    try:
        import pycbc.events.threshold_cpu as mod
    except ImportError:
        print("pycbc not found – nothing to patch.", file=sys.stderr)
        sys.exit(1)

    path = Path(inspect.getfile(mod))
    src = path.read_text()
    original = src

    src = src.replace(
        "numpy.array(series.data, copy=False, dtype=numpy.complex64)",
        "numpy.asarray(series.data, dtype=numpy.complex64)",
    )
    src = re.sub(
        r"numpy\.array\(series\.data,\s*copy=False,\s*\n(\s*)dtype=numpy\.complex64\)",
        r"numpy.asarray(series.data,\n\1dtype=numpy.complex64)",
        src,
    )

    if src == original:
        print("threshold_cpu.py: already patched or pattern not found, skipping.")
    else:
        path.write_text(src)
        _clear_cache(path)
        print("threshold_cpu.py: patched (copy=False → asarray).")


def patch_matchedfilter():
    try:
        import pycbc.filter.matchedfilter as mod
    except ImportError:
        print("pycbc not found – nothing to patch.", file=sys.stderr)
        sys.exit(1)

    path = Path(inspect.getfile(mod))
    src = path.read_text()
    original = src

    replacements = [
        ("numpy.real(hplus)", "numpy.asarray(hplus).real"),
        ("numpy.real(hcross)", "numpy.asarray(hcross).real"),
        ("numpy.imag(hplus)", "numpy.asarray(hplus).imag"),
        ("numpy.imag(hcross)", "numpy.asarray(hcross).imag"),
        ("numpy.real(hphccorr)", "numpy.asarray(hphccorr).real"),
        ("numpy.real(hplus_cross_corr)", "numpy.asarray(hplus_cross_corr).real"),
    ]
    for old, new in replacements:
        src = src.replace(old, new)

    if src == original:
        print("matchedfilter.py: already patched or pattern not found, skipping.")
    else:
        path.write_text(src)
        _clear_cache(path)
        n = sum(1 for a, b in zip(original.splitlines(), src.splitlines()) if a != b)
        print(f"matchedfilter.py: patched ({n} lines changed).")


if __name__ == "__main__":
    patch_threshold_cpu()
    patch_matchedfilter()
