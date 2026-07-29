<p align="center">
    <img width="320" height="187.5" alt="a217d0b8-c7d9-410c-987c-3738cb58b0a5" src="https://github.com/user-attachments/assets/072e876e-612a-46fc-9e5f-d6c3c636d24e" />
</p>

<h1 align="center">PyART</h1>
<h3 align="center">Python Analytical Relativity Toolkit</h3>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-GPLv3+-blue.svg" alt="License: GPLv3 or later"></a>
  <a href="https://github.com/RoxGamba/PyART/actions/workflows/tests.yml">
    <img src="https://github.com/RoxGamba/PyART/actions/workflows/tests.yml/badge.svg" alt="Build Status">
  </a>
  <a href="https://github.com/RoxGamba/PyART/actions/workflows/documentation.yml">
    <img src="https://github.com/RoxGamba/PyART/actions/workflows/documentation.yml/badge.svg" alt="Documentation Status">
  </a>
  <a href="https://results.pre-commit.ci/latest/github/RoxGamba/PyART/main">
    <img src="https://results.pre-commit.ci/badge/github/RoxGamba/PyART/main.svg" alt="pre-commit.ci status">
  </a>
  <a href="https://codecov.io/github/RoxGamba/PyART" > 
     <img src="https://codecov.io/github/RoxGamba/PyART/graph/badge.svg?token=R9OV89SSY6"/> 
 </a>
</p>

## Getting started

Install the base library with:
```
pip install .
```
This covers the core `Waveform` interface, analysis tools (matched filtering, eccentricity,
scattering angles, ...), and the PN/BHPT analytic expressions. Optional/soft dependencies (e.g.
`EOBRun_module`, `lalsimulation`) are imported inside `try/except`, so a missing one only disables
the specific model or catalog that needs it, without breaking the rest of the package.

On top of the base install, three extras add optional functionality:

| Extra | Install with | What it gives you |
|---|---|---|
| `catalogs` | `pip install ".[catalogs]"` | Downloading/scraping NR waveform catalogs — SXS (`sxs`, `romspline`) and RIT (`requests`, `beautifulsoup4`). Needed to run the test suite, which exercises SXS/RIT downloads. |
| `models` | `pip install ".[models]"` | Optional analytical/semi-analytical waveform models — TEOBResumS (`teobresums`), IMRPhenomX/X_AS (`phenomxpy`), and SEOBNR (`pyseobnr`). |
| `docs` | `pip install ".[docs]"` | Building the Sphinx documentation locally (`make -C docs html`) — Sphinx, MyST, `furo`, `sphinx-autoapi`, and `seaborn` for the executed tutorial notebooks. Not needed to install or use the library. |

Extras can be combined, e.g.:
```
pip install ".[catalogs,models]"
```

> **Note:** the `models` extra compiles C/Cython extensions (`teobresums`, and `pyseobnr`
> via `pygsl_lite`) linked against the GNU Scientific Library. Install the GSL headers with
> your system package manager *before* installing this extra:
> ```
> apt-get install libgsl-dev   # Debian/Ubuntu
> brew install gsl             # macOS
> ```

> **Note:** PyART depends on `pycbc`, which currently has two incompatibilities with numpy 2.x.
> After installing, apply the one-time patch:
> ```
> python scripts/patch_pycbc_numpy2.py
> ```
> This patches `pycbc`'s `events/threshold_cpu.py` and `filter/matchedfilter.py`
> in-place. It is idempotent and safe to re-run after pycbc upgrades.

## For developers

If you are a developer:

* Install pre-commit hooks (this will automatically format code with black):
    ```
    pip install pre-commit
    pre-commit install
    ```
    
Note: pre-commit.ci is enabled for this repository, which will automatically format code with black on all pull requests.
