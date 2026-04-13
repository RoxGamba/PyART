---

# PyART: A Python Analytical Relativity Toolkit for gravitational-waveforms 

---

tags:

* Python
* gravitational waves
* compact binaries
* numerical relativity
* semi-analytical waveforms

authors:

* name: Simone Albanessi
 orcid:
 affiliation: "1"
* name: Rossella Gamba
 orcid:
 affiliation: "2"
* name: Danilo Chiaramello
 orcid:
 affiliation: "3", "4"
* name: Koustav Chandra
 orcid:
 affiliation: "5"

affiliations:
* name: Friedrich-Schiller-Universit¨at Jena, Theoretisch-Physikalisches Institut, 07743 Jena, Germany
 index: 1
* name: Department of Physics, University of California, Berkeley, CA 94720, USA
 index: 2
* name: Department of Physics, Universit´a degli Studi di Torino, Torino, 10125, Italy
 index: 3
* name: INFN sezione di Torino, Torino, 10125, Italy
 index: 4
* name: Max Planck Institute for Gravitational Physics (Albert Einstein Institute), Potsdam-14476, Germany
 index: 5

date: 13 April 2026
bibliography: references.bib

---

# Summary

`PyART` (Python Analytical Relativity Toolkit) is an open-source Python package that provides a unified framework for gravitational-wave (GW) modeling. It is designed to streamline tasks routinely encountered in the development and validation of waveform models, particularly at the interface between numerical relativity (NR) and semi-analytical approaches.

The package, as such, provides tools for loading, manipulating, and comparing waveforms from heterogeneous sources, including NR catalogs and semi-analytical models. It supports waveform alignment, phase and amplitude comparison, and computing waveform faithfulness. In addition, `PyART` facilitates inspection of binary dynamics and metadata.

...

---

# Statement of Need

Modeled gravitational-wave data analysis relies critically on accurate waveform templates, underpinning templated searches, parameter estimation, and tests of general relativity. The construction and validation of these templates require systematic comparisons between waveforms and dynamical quantities derived from multiple sources, including NR catalogs (e.g., SXS, GRA, RIT) and semi-analytical models.

In practice, such comparisons are technically involved. Different catalogs adopt distinct coordinate conventions, extraction procedures, etc. Moreover, waveform comparison requires careful handling of signal-processing steps, such as windowing, tapering, interpolation, and zero-padding. These operations are error-prone, and standardizing them, as done in `PyART`, improves reproducibility.

Originally developed to ease model calibration and validation efforts, `PyART` has been employed in \citet{}

---

# Functionality

`PyART` is organized into N principal modules, each targeting a distinct aspect:

### `waveform`

The `waveform` module provides the package's core abstraction through the `Waveform` class. This class defines a uniform interface for loading and handling gravitational waveforms, independent of their origin. By encapsulating waveform data and metadata within a consistent structure, the module enables seamless comparison between NR simulations and analytical models, while abstracting away catalog-specific conventions.

---

### `analysis`

The `analysis` module implements routines for quantitative waveform comparison. It includes functionality to compute polarisation and mode-by-mode unfaithfulness, following the standard conventions used in the waveform modeling literature. It also supports systematic sweeps over total mass and frequency ranges, enabling robustness studies of waveform agreement.

Additional tools are provided for time- and phase-aligned waveform alignment, as well as for extracting frequency-domain amplitude and phase. These capabilities are essential for diagnosing discrepancies between models and simulations and for assessing modeling accuracy.

---

### `analytics`


---

### `dynamics`

Implements routines for analyzing binary dynamics associated with GW signals. It supports both NR-derived quantities and semi-analytical (e.g., EOB) dynamics.

Key quantities, such as the orbital frequency ( \omega(t) ), the gravitational-wave phase ( \Phi_{\mathrm{GW}}(t) ), the binding energy, and the angular momentum, can be extracted and compared. These diagnostics are essential for identifying the physical origin of waveform discrepancies and for calibrating analytical models.

---

### `catalog`

The `catalog` module offers a lightweight interface for accessing and managing NR waveform catalogs.


In addition, the module loads NR initial data, such as binary parameters (e.g., mass ratio, spin components, eccentricity), as structured tabular data.

---

# Usage

`PyART` 


Further examples and documentation are provided in the online repository.

# Reproducibility and Availability

`PyART` is fully open source and version-controlled, with documentation and examples provided in the public repository. The package is designed to ensure reproducibility by enforcing consistent conventions for waveform handling and comparison.

---

