"""
Module for calculating the orbital eccentricity from gravitational waveform data
using various definitions and post-Newtonian (PN) approximations.

Classes
-------
EccentricityCalculator
    Computes the eccentricity from a waveform using PN formulas or external
    amplitude fitting methods.

References
----------
- Eq. 4.8 of https://arxiv.org/pdf/1507.07100.pdf (3PN formula for e_t in Harmonic coordinates)
- https://github.com/anu-gw/gw_eccentricity (AmplitudeFits method)

Dependencies
------------
- numpy
- gw_eccentricity (optional, required for 'gwecc' method)

Example
-------
>>> from eccentricity_calc import EccentricityCalculator
>>> ecc_calc = EccentricityCalculator(h, pars, kind="3PN")
>>> eccentricity = ecc_calc.e
"""
import numpy as np


class EccentricityCalculator:
    """
    Class to compute eccentricity from a waveform using various definitions.
    Parameters
    ----------
    h : object
        Waveform object containing necessary data for eccentricity calculation.
    pars : dict
        Dictionary of parameters required for the calculation (e.g., mass ratio 'q', reference time 'tref_in').
    t : array-like, optional
        Time array associated with the waveform.
    omg : array-like, optional
        Frequency array associated with the waveform.
    kind : str, optional
        String specifying the method for eccentricity calculation. Supported values are "0PN", "1PN", "2PN", "3PN", and "gwecc".
    Attributes
    ----------
    h : object
        Waveform object.
    t : array-like
        Time array.
    omg : array-like
        Frequency array.
    kind : str
        Method for eccentricity calculation.
    pars : dict
        Parameters for calculation.
    e : float or array-like
        Computed eccentricity.
    Methods
    -------
    compute_eccentricity(kind)
        Computes the eccentricity using the specified method.
    _compute_eccentricity_PN_EJ(kind)
        Computes the eccentricity using the PN formula up to 3PN order.
    _compute_eccentricity_gwecc(kind, tref_in=None, method="AmplitudeFits")
        Computes the eccentricity using the gw_eccentricity package.
    """
    def __init__(
        self,
        h,
        pars,
        t=None,
        kind="3PN",
    ):
        """
        Initialize the EccentricityCalculator with waveform data and parameters.
        Parameters
        ----------
        h : object
            Waveform object containing necessary data for eccentricity calculation.
        pars : dict
            Dictionary of parameters required for the calculation (e.g., mass ratio 'q', reference time 'tref_in').
        t : array-like, optional
            Time array associated with the waveform.
        kind : str, optional
            String specifying the method for eccentricity calculation. Supported values are "0PN", "1PN", "2PN", "3PN", and "gwecc".
        """
        self.h = h
        self.t = t
        self.kind = kind
        self.pars = pars

        self.e = self.compute_eccentricity(self.kind)
        pass

    def compute_eccentricity(self, kind):
        """
        Compute the eccentricity based on the specified method.
        Parameters
        ----------
        kind : str
            The type of eccentricity to compute. Supported values are:
            - "PN": Post-Newtonian eccentricity.
            - "gwecc": Gravitational wave eccentricity.
        Returns
        -------
        float or array-like
            The computed eccentricity.

        Raises
        ------
        NotImplementedError
            If the specified kind is not supported.
        Notes
        -----
        Only PN and gwecc eccentricity calculations are currently implemented.
        """
        if kind[1:] == "PN":
            return self._compute_eccentricity_PN_EJ(kind)
        elif kind == "gwecc":
            return self._compute_eccentricity_gwecc(kind, tref_in=self.pars["tref_in"])
        else:
            raise NotImplementedError(
                "Only PN/gwecc eccentricity is implemented for now"
            )

    def _compute_eccentricity_PN_EJ(self, kind):
        """
        Compute eccentricity from the waveform using the 3PN formula
        Eq. 4.8 of https://arxiv.org/pdf/1507.07100.pdf
        e_t in Harmonic coordinates

        Parameters
        ----------
        kind : str
            PN order for the eccentricity calculation. Must be one of:
            "0PN", "1PN", "2PN", "3PN".
        Returns
        -------
        float
            The computed eccentricity at the requested PN order.
        Raises
        ------
        NotImplementedError
            If `kind` is not one of the supported PN orders ("0PN", "1PN", "2PN", "3PN").
        """

        q = self.pars["q"]
        nu = q / (1 + q) ** 2
        nu2 = nu * nu
        nu3 = nu2 * nu
        Pi = np.pi
        Pi2 = Pi * Pi

        Eb, pph = self.h.Eb, self.h.j
        xi = -Eb * pph**2
        e_0PN = 1 - 2 * xi
        e_1PN = -4.0 - 2 * nu + (-1 + 3 * nu) * xi
        e_2PN = (
            (20.0 - 23 * nu) / xi - 22.0 + 60 * nu + 3 * nu2 - (31 * nu + 4 * nu2) * xi
        )
        e_3PN = (
            (-2016 + (5644 - 123 * Pi2) * nu - 252 * nu2) / (12 * xi * xi)
            + (4848 + (-21128 + 369 * Pi2) * nu + 2988 * nu2) / (24 * xi)
            - 20
            + 298 * nu
            - 186 * nu2
            - 4 * nu3
            + (-1 * 30.0 * nu + 283.0 / 4 * nu2 + 5 * nu3) * xi
        )

        if kind == "0PN":
            return np.sqrt(e_0PN)
        elif kind == "1PN":
            return np.sqrt(e_0PN + Eb * e_1PN)
        elif kind == "2PN":
            return np.sqrt(Eb * (Eb * e_2PN + e_1PN) + e_0PN)
        elif kind == "3PN":
            return np.sqrt(Eb * (Eb * (Eb * e_3PN + e_2PN) + e_1PN) + e_0PN)
        else:
            raise NotImplementedError(
                "PN eccentricity is implemented up to 3PN for now"
            )

    def _compute_eccentricity_gwecc(self, kind, tref_in=None, method="AmplitudeFits"):
        """
        Computes the eccentricity of the waveform using the `gw_eccentricity` package.
        Parameters
        ----------
        kind : str
            The type of eccentricity calculation to perform.
        tref_in : float or None, optional
            Reference time for the eccentricity measurement. If None, uses default.
        method : str, optional
            The method to use for amplitude fits. Default is "AmplitudeFits".
        Returns
        -------
        float
            The computed eccentricity value.
        Raises
        ------
        ImportError
            If the `gw_eccentricity` package is not installed.
        Notes
        -----
        This function requires the `gw_eccentricity` package to be installed.
        The waveform data is extracted from the (2, 2) mode.
        The result is also stored in `self.res`.
        """
        try:
            from gw_eccentricity import measure_eccentricity
        except ImportError:
            raise ImportError(
                "To compute the eccentricity from the waveform you need to install `gw_eccentricity`"
            )

        h22 = self.h.hlm[(2, 2)]['z']

        modeDict = {(2, 2): h22}
        dataDict = {"t": self.h.u, "hlm": modeDict}
        res = measure_eccentricity(tref_in=tref_in, dataDict=dataDict, method=method)
        self.res = res
        return res["eccentricity"]
