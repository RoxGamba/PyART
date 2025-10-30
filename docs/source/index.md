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

PyART is currently available only from source. You can install it as follows:
```
pip install git+https://github.com/RoxGamba/PyART.git
```
We promise to make available on PyPI in near future.


```{toctree}
:caption: 'Contents:'
:maxdepth: 2

tutorials/intro_to_waveforms.ipynb
tutorials/catalog_downloads.ipynb
tutorials/phase_alignment.ipynb
tutorials/mismatch.ipynb
tutorials/nr_eob_mismatch.ipynb
tutorials/optimizing_initial_conditions.ipynb
tutorials/iccub_waveform_integration.ipynb
tutorials/scattering_angles.ipynb
```