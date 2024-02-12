# PyART

Possible sctructure:
- `waveform.py`    : parent class for waveforms (both EOB and NR)
- `analysis/`      : classes to analyze waveforms and simulations (plots, mismatches, parspace, scattering angles, phasing, ...)
- `analytics/`     : classes for analytic relations and stuff (PN, PM, ...)
- `catalogs/`      : folder with classes to read data from specific nr catalogs & work with them. One .py for each catalog (rit, gra, bam, etk, ...) with catalog-specific functions for data parsing, one parent class to perform typical routines and enforce one single data/metadata format (extrapolation, FFI, ...), could be simulations.py?
- `models/`        : classes to read/generate waveforms with approximants (eob, lal, ...)
- `utils/`         : folder for utilities (integration, derivatives, special functions, ...)
- `examples/`      : example python scripts to work with this library, to be gradually added
