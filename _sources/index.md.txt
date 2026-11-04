# PyART — Python Analytical Relativity Toolkit

**PyART** (Python Analytical Relativity Toolkit) provides a unified interface to access and work with numerical relativity (NR) waveform catalogs for compact-object binary mergers. Different NR groups maintain their own data formats and access conventions, which can make comparative or downstream analyses cumbersome. PyART addresses this by offering a lightweight abstraction layer over several major public catalogs.

In addition to catalog access, PyART includes basic tools for waveform comparison—such as time-domain alignment and frequency-domain mismatch calculations—for both single-mode and multi-mode data (and few more).

## Supported Catalogs

- Simulating eXtreme Spacetimes (SXS)  
- Maya 
- CoRe (Computational Relativity)  
- RIT
- GR-Athena  
- ICCUB NR
- SACRA  
- RWZ
- *and others*

```{warning}
PyART is under active development.
The codebase is mostly well-behaved, but some of its dependencies are temperamental, and things may break without warning. Think of it as a promising postdoc — brilliant, but occasionally unpredictable.
```

PyART is currently available only from source. Clone the repository and install the base library with:
```
git clone https://github.com/RoxGamba/PyART.git
cd PyART
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

> **Note:** PyART depends on `pycbc`, which currently has two incompatibilities with numpy 2.x.
> After installing, apply the one-time patch:
> ```
> python scripts/patch_pycbc_numpy2.py
> ```
> This patches `pycbc`'s `events/threshold_cpu.py` and `filter/matchedfilter.py`
> in-place. It is idempotent and safe to re-run after pycbc upgrades.

We promise to make available on PyPI in near future.


```{toctree}
:caption: 'Contents:'
:maxdepth: 2

tutorials/intro_to_waveforms.ipynb
tutorials/catalog_downloads.ipynb
tutorials/phase_alignment.ipynb
tutorials/nr_eob_mismatch.ipynb
tutorials/optimizing_initial_conditions.ipynb
tutorials/iccub_waveform_integration.ipynb
tutorials/scattering_angles.ipynb
tutorials/coordinates_and_twopuncture.ipynb
tutorials/pnpedia.ipynb
```
