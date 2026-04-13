import numpy as np
import sympy as sp
import os
import logging
from sympy.parsing.mathematica import parse_mathematica
from .expr import AnalyticExpression, _get_x_exponent, _get_x_power_range

x = sp.symbols("x")


class PNPedia:
    """
    Collection of PN quantities loaded from the PNPedia repository.

    Acts as a dictionary-backed store of :class:`~PyART.analytic.expr.AnalyticExpression`
    objects, one per PN quantity file.  Quantities are parsed on demand from
    the Mathematica-format text files distributed with PNPedia:
        https://github.com/davidtrestini/PNpedia
    """

    def __init__(self, path, dowload=False):
        """
        Initializes the PNPedia class.

        Parameters
        ----------
        path : str
            The path to the local copy of the PNPedia repository.
        dowload : bool, optional
            If True, downloads the PNPedia repository from GitHub. Default is False.
        """
        self.path = path
        if dowload:
            self.download_pnpedia()
        self.__parse_pnpedia()

    def download_pnpedia(self):
        """
        Clone the PNPedia repository from GitHub using git
        to a specified path. This method requires git to be installed on the system and accessible via the command line.
        """
        import subprocess

        # check if the path already exists
        if os.path.exists(self.path):
            logging.info(
                f"PNPedia repository already exists at {self.path}. Skipping download."
            )
            return
        else:
            logging.info("Cloning PNPedia repository from GitHub...")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/davidtrestini/PNpedia.git",
                    self.path,
                ]
            )
            logging.info(f"PNPedia repository cloned to {self.path}")

    def __parse_pnpedia(self):
        """
        Internal method to setup the structure of the PNPedia repository and create a dictionary
        mapping the paths and names of the PN quantities to a dictionary for easy retrieval/evaluation.
        """

        pnpedia_structure = {}
        # Walk through the directory structure and populate the dictionary
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    quantity_name = file[:-4]  # Remove the .txt extension
                    # save the full path to the quantity in the dictionary
                    # use as key the quantity name and directory structure to allow for easy retrieval
                    structure = (
                        root.replace(self.path, "")[1:]
                        .replace("/", "_")
                        .replace(" ", "-")
                        .strip()
                        .lower()
                    )
                    quantity_name = quantity_name.lower()
                    key = quantity_name
                    if structure:
                        key = f"{structure}_{quantity_name}"
                    pnpedia_structure[key] = os.path.join(root, file)

        self.pnpedia_structure = pnpedia_structure
        logging.info("PNPedia structure parsed successfully.")

    # ------------------------------------------------------------------
    # Power-range utilities (delegate to module-level functions from expr)
    # ------------------------------------------------------------------

    def _get_x_power_range(self, expr, x_symbol=None):
        return _get_x_power_range(expr, x_symbol)

    def _get_x_exponent(self, term, x_symbol=None):
        return _get_x_exponent(term, x_symbol)

    def get_pn_quantity(self, name, order, path=None, variable="x"):
        """
        Retrieve a PN quantity from PNPedia based on its name and the desired PN order.

        The PNPedia structure (inside Core post-Newtonian quantities) is:
        - orbit type (Circular, Elliptic, Hyperbolic)
        - Spin (NonSpinning, Spinning)
        - Precession (Precessing, Nonprecessing)
        - Tides (With tidal effects, Without tidal effects)
        - Quantity (Waveform, Fluxes, Constants of motion, Waveform frequencies):
            - Constants of motion (Energy, Angular momentum)
            - Fluxes (Angular momentum flux, Energy flux)
            - Waveform (h_l_m.txt)

        We load the quantities (that are stored as Mathematica-readable text files) using sympy,
        which can parse the Mathematica syntax and convert it to Python expressions.

        Parameters
        ----------
        name : str
            The name of the PN quantity to retrieve,
        order : str
            The desired PN order (e.g., "1", "2", "3", etc.)
        path : str, optional
            The path to the PNPedia file to be loaded. If None, it will be constructed based on the name and order.

        Returns
        -------
        sympy expression
            The requested PN quantity as a sympy expression.
        """

        if path is None:
            # Construct the path based on the name and order
            # Select the right key from the pnpedia_structure dictionary based on the name
            # if there are multiple matches, raise an error and ask the user to specify more details (e.g., orbit type, spin, precession, tides)
            # split the various components of the name and order to construct the key
            name = name.lower()
            name_pieces = name.split("_")

            matches = []
            for key in self.pnpedia_structure.keys():
                key_lower = key.lower()
                key_terms = key_lower.replace("-", "_").split("_")
                for piece in name_pieces:
                    if piece not in key_terms:
                        break
                else:
                    matches.append(key)

            if len(matches) == 0:
                raise ValueError(
                    f"No matches found for quantity name '{name}' in PNPedia structure."
                )
            elif len(matches) > 1:
                raise ValueError(
                    f"Multiple matches found for quantity name '{name}' in PNPedia structure: {matches}. Please specify more details (e.g., orbit type, spin, precession, tides)."
                )
            else:
                path = self.pnpedia_structure[matches[0]]
        else:
            logging.info(f"Loading PN quantity from {path}...")

        with open(path, "r") as file:
            content = file.read()

        # Parse the Mathematica expression using sympy
        content = self.mathematica_to_python_vars(content)
        try:
            pn_quantity = parse_mathematica(content)
        except Exception as e:
            logging.warning(
                "parse_mathematica failed for %s; falling back to sympy.sympify: %s",
                path,
                e,
            )
            content_py = content.replace("[", "(").replace("]", ")").replace("^", "**")
            pn_quantity = sp.sympify(content_py)

        # determine the symbol used for PN counting
        if isinstance(variable, str):
            x_symbol = sp.symbols(variable)
        elif isinstance(variable, sp.Symbol):
            x_symbol = variable
        else:
            raise TypeError("variable must be a string or a sympy.Symbol")

        # handle log terms by replacing them with a dummy function that can
        # be expanded in series
        pn_quantity = pn_quantity.replace(sp.log(x_symbol), sp.symbols("logx0"))

        # Expand to expose separate x powers in each term for correct truncation
        pn_quantity = sp.expand(pn_quantity)

        min_order, max_order = self._get_x_power_range(pn_quantity, x_symbol)
        max_pn_order = max_order - min_order

        requested_pn_order = sp.Rational(order)
        if requested_pn_order > max_pn_order:
            logging.warning(
                f"Requested PN order {order} is higher than the maximum available order {max_pn_order} for this quantity. Returning up to available order {max_pn_order}."
            )

        target_order_power = min_order + requested_pn_order

        truncated_terms = []
        for term in pn_quantity.as_ordered_terms():
            term_power = self._get_x_exponent(term, x_symbol)
            if term_power <= target_order_power:
                truncated_terms.append(term)

        pn_quantity = sum(truncated_terms) if truncated_terms else sp.Integer(0)

        # sub the dummy log function back to log
        result = pn_quantity.replace(sp.symbols("logx0"), sp.log(x_symbol))
        return AnalyticExpression(result)

    def mathematica_to_python_vars(self, expr):
        """
        Convert Mathematica variables in a sympy expression to Python variables.

        Parameters
        ----------
        expr : sympy expression or str
            The sympy expression or raw Mathematica string containing Mathematica variables.

        Returns
        -------
        str
            The expression string with Mathematica identifiers converted to Python-friendly names.
        """

        expr_str = str(expr)

        # Specific, known replacements used in PNPedia files
        var_mapping = {
            # Must come before shorter \[Lambda] etc to avoid partial match
            r"\[Lambda]0'[e]": "lambda0eprime",
            r"\[Lambda]0[e]": "lambda0e",
            r"\[Lambda]0'[et]": "lambda0eprimeEt",
            r"\[Lambda]0[et]": "lambda0eEt",
            r"\[Lambda]0'[Sqrt[1-\[Iota]]]": "lambda0primeSqrt1miota",
            r"\[Lambda]0[Sqrt[1-\[Iota]]]": "lambda0Sqrt1miota",
            r"Derivative[1][\[Lambda]0][Sqrt[1-\[Iota]]]": "Derivative1Lambda0Sqrt1miota",
            r"\[Lambda]0'[Sqrt[1-j]]": "lambda0primeSqrt1mj",
            r"\[Lambda]0[Sqrt[1-j]]": "lambda0Sqrt1mj",
            r"\[CurlyPhi][Sqrt[1-\[Iota]]]": "curlyphiSqrt1miota",
            r"\[CurlyPhi]'[Sqrt[1-\[Iota]]]": "curlyphiPrimeSqrt1miota",
            r"Derivative[1][\[CurlyPhi]][Sqrt[1-\[Iota]]]": "Derivative1CurlyphiSqrt1miota",
            r"\[CurlyPhi][Sqrt[1-j]]": "curlyphiSqrt1mj",
            r"\[CurlyPhi]'[Sqrt[1-j]]": "curlyphiPrimeSqrt1mj",
            r"\[CurlyPhi]$tilde[e]": "curlyphiTildeE",
            r"\[CurlyPhi]$tilde[et]": "curlyphiTildeEt",
            r"\[CurlyPhi]$tilde'[Sqrt[1-\[Iota]]]": "curlyphiTildePrimeSqrt1miota",
            r"\[CurlyPhi]$tilde[Sqrt[1-j]]": "curlyphiTildeSqrt1mj",
            r"\[CurlyPhi]$tilde'[Sqrt[1-j]]": "curlyphiTildePrimeSqrt1mj",
            r"\[CurlyPhi]$tilde[Sqrt[1-\[Iota]]]": "curlyphiTildeSqrt1miota",
            r"\[CurlyPhi]$tilde'[e]": "curlyphiTildePrimeE",
            r"\[Zeta]$tilde[e]": "zetaTildeEt",
            r"\[Zeta]$tilde[et]": "zetaTildeEt",
            r"\[Zeta]$tilde[Sqrt[1-j]]": "zetaTildeSqrt1mj",
            r"\[Zeta]$tilde'[e]": "zetaTildePrimeE",
            r"\[Zeta]$tilde'[et]": "zetaTildePrimeEt",
            r"\[Zeta]$tilde'[Sqrt[1-j]]": "zetaTildePrimeSqrt1mj",
            r"\[Zeta]$tilde[Sqrt[1-\[Iota]]]": "zetaTildeSqrt1miota",
            r"\[Zeta]$tilde'[Sqrt[1-\[Iota]]]": "zetaTildePrimeSqrt1miota",
            r"\[Psi]$tilde[e]": "psiTildeE",
            r"\[Psi]$tilde[et]": "psiTildeEt",
            r"\[Psi]$tilde[Sqrt[1-j]]": "psiTildeSqrt1mj",
            r"\[Kappa]$tilde[e]": "kappaTildeE",
            r"\[Kappa]$tilde'[et]": "kappaTildePrimeEt",
            r"\[Kappa]$tilde[Sqrt[1-j]]": "kappaTildeSqrt1mj",
            r"\[Kappa][Sqrt[1-j]]": "kappaSqrt1mj",
            # Greek letters — plain
            r"\[Nu]": "nu",
            r"\[Delta]": "delta",
            r"\[Epsilon]": "epsilon",
            r"\[Gamma]": "gamma",
            r"\[Iota]": "iota",
            r"\[Theta]": "theta",
            r"\[Phi]": "phi",
            r"\[Zeta]": "zeta",
            r"\[Omega]": "omega",
            r"\[Pi]": "pi",
            r"\[Sigma]": "sigma",
            r"\[Xi]": "xi",
            r"\[Mu]": "mu",
            r"\[Rho]": "rho",
            r"\[Upsilon]": "upsilon",
            r"\[Kappa]": "kappa",
            r"\[Tau]": "tau",
            r"\[Psi]": "psi",
            r"\[Lambda]": "lambda_",
            # Curly variants
            r"\[CurlyEpsilon]": "curlyepsilon",
            r"\[CurlyPhi]": "curlyphi",
            r"\[CurlyTheta]": "curlytheta",
            r"\[CurlyRho]": "curlyrho",
            r"\[CurlyKappa]": "curlykappa",
            # Capital variants
            r"\[CapitalPsi]": "CapitalPsi",
            r"\[CapitalPhi]": "CapitalPhi",
            r"\[CapitalDelta]": "CapitalDelta",
            r"\[CapitalGamma]": "CapitalGamma",
            r"\[CapitalLambda]": "CapitalLambda",
            r"\[CapitalOmega]": "CapitalOmega",
            r"\[CapitalSigma]": "CapitalSigma",
            r"\[CapitalTheta]": "CapitalTheta",
            r"\[CapitalXi]": "CapitalXi",
        }

        # Pre-apply specific mapping in explicit order to avoid partial collisions
        for math_var, py_var in sorted(
            var_mapping.items(), key=lambda item: -len(item[0])
        ):
            expr_str = expr_str.replace(math_var, py_var)

        return expr_str
