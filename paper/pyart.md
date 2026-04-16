
---
title: "PyART: A Python Analytical Relativity Toolkit for Gravitational-Waves from Compact Binaries"
tags:
  - gravitational waves
  - compact binaries
  - numerical relativity
  - semi-analytical waveforms
authors:
  - name: Simone Albanesi
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Rossella Gamba
    orcid: 0000-0000-0000-0000
    affiliation: 2
  - name: Danilo Chiaramello
    orcid: 0000-0000-0000-0000
    affiliation: "3, 4"
  - name: Koustav Chandra
    orcid: 0000-0000-0000-0000
    affiliation: 5
affiliations:
  - name: Friedrich-Schiller-Universität Jena, Theoretisch-Physikalisches Institut, 07743 Jena, Germany
    index: 1
  - name: Department of Physics, University of California, Berkeley, CA 94720, USA
    index: 2
  - name: Department of Physics, Università degli Studi di Torino, Torino, 10125, Italy
    index: 3
  - name: INFN sezione di Torino, Torino, 10125, Italy
    index: 4
  - name: Max Planck Institute for Gravitational Physics (Albert Einstein Institute), Potsdam-14476, Germany
    index: 5

date: 13 April 2026
bibliography: references.bib
---

# Summary

`PyART` (Python Analytical Relativity Toolkit) is an open-source Python package providing a unified framework for gravitational-wave (GW) modeling.
It is designed to streamline tasks routinely encountered in the development and validation of waveform models, particularly at the interface between
numerical relativity (NR) and analytical or semi-analytical approaches. The package provides tools for loading, manipulating, and comparing data
from heterogeneous sources, including NR catalogs, widely used waveform models, and analytical post-Newtonian or post-Minkowskian calculations.

---

# Statement of Need

Modeled gravitational-wave data analysis relies critically on accurate waveform templates, which underpin matched-filter searches, parameter estimation,
and tests of general relativity. The construction and validation of these templates, in turn, require systematic comparisons between waveforms and dynamical
quantities derived from multiple sources, including NR catalogs (e.g., SXS [@Scheel:2025jct], RIT [@Healy:2022wdn], GRA [@Rashti:2024yoc],
Maya [@Ferguson:2023vta], CoRe [@Gonzalez:2022mgo]) and analytical calculations.

In practice, such comparisons are technically involved:
- on the NR side, different catalogs adopt distinct conventions and extraction procedures, and waveform comparison requires careful handling of signal-processing steps such as windowing, tapering, interpolation, and zero-padding.
These operations are error-prone and are typically not discussed in detail in the literature, making it difficult to reproduce results and to compare findings across studies.
- on the analytical side, post-Newtonian and post-Minkowskian calculations are scattered across a considerable body of works, and their numerical implmentation often left to the single researcher.
Beyond the risk of transcription errors, this approach causes significant effort duplication and time waste.

`PyART` addresses these challenges by providing a unified, publicly available toolkit for handling data from simulations, models and analytical calculations,
and for performing systematic comparisons among them. In doing so, it aims to facilitate the development of accurate waveform templates
and to promote transparency and reproducibility in GW astronomy.

---

# State of the Field

Numerous libraries exist for GW data analysis (e.g., ...), waveform generation (e.g., ...),
and the loading of specific NR catalogs (e.g., ...). However, no comprehensive package currently bridges all of these aspects within a single
framework dedicated to the development and validation of waveform models.

The typical waveform modeling workflow combines tools that are often independently developed by different groups.
This fragmentation leads to inefficiency, redundant reimplementation of common routines, and an overall
lack of transparency and reproducibility, making model development unnecessarily time-consuming and error-prone.

Addressing this problem requires expertise spanning NR, analytical relativity, and data analysis. `PyART` builds on established field-specific
libraries — including `pycbc`, `gwpy`, `lalsuite`, `sxs`, `maya`, and `watpy` — as well as more recently developed tools such as
`gweccentricity` [@Shaikh:2023ypz] and [PNPedia](https://github.com/davidtrestini/PNpedia), integrating them into a coherent and accessible framework.

---

# Software Design

`PyART` is designed to be **modular** and **extensible**, in keeping with its goal of streamlining the waveform modeling workflow for compact binaries.
New models, catalogs, anaytical expressions and analysis tools can be incorporated without requiring changes to the underlying data handling and processing infrastructure.

User-friendliness is a core design principle: the package is structured to be intuitive and accessible to researchers who are not (yet) specialists
in waveform modeling. Documentation and worked examples are provided in the online repository via Jupyter notebooks and a comprehensive API reference,
and contributions are encouraged.

## Modules and classes
`PyART` is organized into N principal modules, each targeting a distinct aspect simulation or modeling workflows.

### `waveform`

The `waveform` module provides the package's core abstraction through the `Waveform` class. This class defines a uniform interface for loading and
handling gravitational waveforms, independent of their origin. By encapsulating waveform data, metadata and dynamics within a consistent structure,
the module enables comparison between NR simulations and analytical models, while abstracting away catalog-specific conventions.
Core methods include standard signal-processing operations, such as interpolation and cutting, as well as waveform-specific tools, such as rotations
(TO BE IMPLEMENTED), the calculation of instantaneous fluxes of energy and angular momentum, kick velocity estimates (TO BE IMPLEMENTED) and the
extraction of dynamical quantities.
This class also allows for basic visualization of modes, dynamics and polarisations (TO BE IMPLEMENTED).

### `analysis`

The `analysis` module implements routines for quantitative waveform comparison. It includes functionality to compute polarisation and mode-by-mode unfaithfulness,
following the standard conventions used in the waveform modeling literature. It also supports systematic sweeps over total mass and frequency ranges,
enabling robustness studies of waveform agreement.

Additional tools are provided for time- and phase-aligned waveform alignment, eccentricity estimation (TO BE IMPLEMENTED) via `gweccentricity` [@Shaikh:2023ypz]
and PN formulae  (see below), and calculation of scattering
angles from NR [@Albanesi:2024xus]. These capabilities are essential for diagnosing discrepancies between models and simulations and for assessing modeling accuracy.

---

### `analytic`
The `analytic` module provides tools for manipulation of analytical expressions, with a particular focus on post-Newtonian (PN), post-Minkowskian (PM) and effective-one-body (EOB) calculations.
The basic parent class, `AnalyticalExpression`, provides a lightweight wrapper around `sympy`, allowing for symbolic manipulation of expressions, as well as numerical evaluation.
The class methods include standard operations, such as differentiation, as well as more specialized tools, such as PN order counting and expansion, and conversion to `numpy`-compatible functions.

We also provide:
- coordinate transformation tools, necessary to map ADM to EOB coordinates and vice versa;
- a collection of PN quantities (fluxes and Hamiltonians) currently natively implemented in the package, with plans to expand this collection in the future;
- an interface with the `PNPedia` project, which offers a growing database of PN expressions for various quantities relevant to compact binary dynamics and waveforms;
- an interface with [PostNewtonianSelfForce](https://github.com/BlackHolePerturbationToolkit/PostNewtonianSelfForce) from the Black Hole Perturbation Toolkit (TO BE IMPLEMENTED).

---

### `catalog`

The `catalog` module offers a lightweight interface for accessing and managing NR waveforms from different catalogs.
Each catalog is represented as a `Waveform` subclass, which implements the necessary logic to load (and dowload) waveforms and metadata, and to convert them to the standard `PyART` format. 
This design allows users to access waveforms from multiple sources through a consistent interface, without worrying about catalog-specific details. The module currently supports the SXS, GRA, RIT, Maya and CoRe catalogs, with plans to include additional sources in the future. (TODO: make sure that all catalogs use the same conventions! See patricia's paper, NINJA etc.)

### `models`

The `models` module provides an interface for loading and handling analytical and semi-analytical waveform models. It includes wrappers for commonly used packages, such as
`teobresums`, `pyseobnr`, `lalsuite`, `pycbc` (IMPLEMENT) and `gwsignal` (IMPLEMENT).
User-defined models can be easily added by copying and modifying existing wrappers;
developers of new models are encouraged to contribute their wrappers to the package, adding -- if desired -- support for additional model-specific features.

### `numerical`

The `numerical` module closes the loop between analytical and numerical relativity by providing tools for the setup of NR simulations.
It includes routines for the construction of initial twopunctures data based on semi-analytical models. 
Future developments will include support for the construction of initial data of binary neutron star simulations, as well as black hole
neutron star systems and test-mass configurations.



---

# Research impact statement
`PyART` has been employed in multiple projects, including the validation of waveform models [@Nagar:2024oyk; @Albanesi:2025txj; @Mahesh:2025oaf] and the analysis of new NR simulations [@Rashti:2024yoc; @Shukla:2025kvc]. There are ognoing efforts to use `PyART` within the waveforms LISA working group (Simo give more details if this is true???).

# AI usage disclosure

Generative AI tools were used in the writing of this paper, primarily for language editing and formatting. The scientific content, including the description of the `PyART` package and its functionalities, was developed by the authors with use of AI tools limited to code review, testing, and documentation generation. The authors have reviewed and verified all content to ensure accuracy and integrity, and take full responsibility for the final version of the paper and the `PyART` package. 

# Acknowledgements

We acknowledge contributions from ...

# References
---

