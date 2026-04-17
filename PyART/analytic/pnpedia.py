import logging
import os
from tokenize import TokenError

import sympy as sp
from sympy.core.sympify import SympifyError
from sympy.parsing.mathematica import parse_mathematica

from .expr import AnalyticExpression, _get_x_exponent, _get_x_power_range
from .mathematica_parser import MathematicaParser
from .analytic_catalog import AnalyticCatalog


_LOG_X_PLACEHOLDER = sp.Dummy("pnpedia_log_x")

_PNPEDIA_SPECIAL_IDENTIFIER_REPLACEMENTS = (
    # Must come before shorter \[Lambda] etc to avoid partial match.
    (r"\[Lambda]0'[e]", "lambda0eprime"),
    (r"\[Lambda]0[e]", "lambda0e"),
    (r"\[Lambda]0'[et]", "lambda0eprimeEt"),
    (r"\[Lambda]0[et]", "lambda0eEt"),
    (r"\[Lambda]0'[Sqrt[1-\[Iota]]]", "lambda0primeSqrt1miota"),
    (r"\[Lambda]0[Sqrt[1-\[Iota]]]", "lambda0Sqrt1miota"),
    (
        r"Derivative[1][\[Lambda]0][Sqrt[1-\[Iota]]]",
        "Derivative1Lambda0Sqrt1miota",
    ),
    (r"\[Lambda]0'[Sqrt[1-j]]", "lambda0primeSqrt1mj"),
    (r"\[Lambda]0[Sqrt[1-j]]", "lambda0Sqrt1mj"),
    (r"\[CurlyPhi][Sqrt[1-\[Iota]]]", "curlyphiSqrt1miota"),
    (r"\[CurlyPhi]'[Sqrt[1-\[Iota]]]", "curlyphiPrimeSqrt1miota"),
    (
        r"Derivative[1][\[CurlyPhi]][Sqrt[1-\[Iota]]]",
        "Derivative1CurlyphiSqrt1miota",
    ),
    (r"\[CurlyPhi][Sqrt[1-j]]", "curlyphiSqrt1mj"),
    (r"\[CurlyPhi]'[Sqrt[1-j]]", "curlyphiPrimeSqrt1mj"),
    (r"\[CurlyPhi]$tilde[e]", "curlyphiTildeE"),
    (r"\[CurlyPhi]$tilde[et]", "curlyphiTildeEt"),
    (
        r"\[CurlyPhi]$tilde'[Sqrt[1-\[Iota]]]",
        "curlyphiTildePrimeSqrt1miota",
    ),
    (r"\[CurlyPhi]$tilde[Sqrt[1-j]]", "curlyphiTildeSqrt1mj"),
    (r"\[CurlyPhi]$tilde'[Sqrt[1-j]]", "curlyphiTildePrimeSqrt1mj"),
    (r"\[CurlyPhi]$tilde[Sqrt[1-\[Iota]]]", "curlyphiTildeSqrt1miota"),
    (r"\[CurlyPhi]$tilde'[e]", "curlyphiTildePrimeE"),
    (r"\[Zeta]$tilde[e]", "zetaTildeEt"),
    (r"\[Zeta]$tilde[et]", "zetaTildeEt"),
    (r"\[Zeta]$tilde[Sqrt[1-j]]", "zetaTildeSqrt1mj"),
    (r"\[Zeta]$tilde'[e]", "zetaTildePrimeE"),
    (r"\[Zeta]$tilde'[et]", "zetaTildePrimeEt"),
    (r"\[Zeta]$tilde'[Sqrt[1-j]]", "zetaTildePrimeSqrt1mj"),
    (r"\[Zeta]$tilde[Sqrt[1-\[Iota]]]", "zetaTildeSqrt1miota"),
    (r"\[Zeta]$tilde'[Sqrt[1-\[Iota]]]", "zetaTildePrimeSqrt1miota"),
    (r"\[Psi]$tilde[e]", "psiTildeE"),
    (r"\[Psi]$tilde[et]", "psiTildeEt"),
    (r"\[Psi]$tilde[Sqrt[1-j]]", "psiTildeSqrt1mj"),
    (r"\[Kappa]$tilde[e]", "kappaTildeE"),
    (r"\[Kappa]$tilde'[et]", "kappaTildePrimeEt"),
    (r"\[Kappa]$tilde[Sqrt[1-j]]", "kappaTildeSqrt1mj"),
    (r"\[Kappa][Sqrt[1-j]]", "kappaSqrt1mj"),
)

_PNPEDIA_GREEK_IDENTIFIER_REPLACEMENTS = (
    (r"\[Nu]", "nu"),
    (r"\[Delta]", "delta"),
    (r"\[Epsilon]", "epsilon"),
    (r"\[Gamma]", "gamma"),
    (r"\[Iota]", "iota"),
    (r"\[Theta]", "theta"),
    (r"\[Phi]", "phi"),
    (r"\[Zeta]", "zeta"),
    (r"\[Omega]", "omega"),
    (r"\[Pi]", "pi"),
    (r"\[Sigma]", "sigma"),
    (r"\[Xi]", "xi"),
    (r"\[Mu]", "mu"),
    (r"\[Rho]", "rho"),
    (r"\[Upsilon]", "upsilon"),
    (r"\[Kappa]", "kappa"),
    (r"\[Tau]", "tau"),
    (r"\[Psi]", "psi"),
    (r"\[Lambda]", "lambda_"),
)

_PNPEDIA_CURLY_IDENTIFIER_REPLACEMENTS = (
    (r"\[CurlyEpsilon]", "curlyepsilon"),
    (r"\[CurlyPhi]", "curlyphi"),
    (r"\[CurlyTheta]", "curlytheta"),
    (r"\[CurlyRho]", "curlyrho"),
    (r"\[CurlyKappa]", "curlykappa"),
)

_PNPEDIA_CAPITAL_IDENTIFIER_REPLACEMENTS = (
    (r"\[CapitalPsi]", "CapitalPsi"),
    (r"\[CapitalPhi]", "CapitalPhi"),
    (r"\[CapitalDelta]", "CapitalDelta"),
    (r"\[CapitalGamma]", "CapitalGamma"),
    (r"\[CapitalLambda]", "CapitalLambda"),
    (r"\[CapitalOmega]", "CapitalOmega"),
    (r"\[CapitalSigma]", "CapitalSigma"),
    (r"\[CapitalTheta]", "CapitalTheta"),
    (r"\[CapitalXi]", "CapitalXi"),
)


def _build_identifier_replacements(*groups):
    replacements = {}
    for group in groups:
        for source, target in group:
            if source in replacements:
                raise ValueError(
                    f"Duplicate PNPedia identifier replacement for '{source}'"
                )
            replacements[source] = target
    return replacements


_PNPEDIA_IDENTIFIER_REPLACEMENTS = _build_identifier_replacements(
    _PNPEDIA_SPECIAL_IDENTIFIER_REPLACEMENTS,
    _PNPEDIA_GREEK_IDENTIFIER_REPLACEMENTS,
    _PNPEDIA_CURLY_IDENTIFIER_REPLACEMENTS,
    _PNPEDIA_CAPITAL_IDENTIFIER_REPLACEMENTS,
)

_PNPEDIA_PARSE_FALLBACK_EXCEPTIONS = (
    SympifyError,
    SyntaxError,
    TokenError,
    TypeError,
    ValueError,
    NotImplementedError,
)


class PNPedia(AnalyticCatalog):
    """
    Collection of PN quantities loaded from the PNPedia repository.

    Acts as a dictionary-backed store of
    :class:`~PyART.analytic.expr.AnalyticExpression` objects, one per PN
    quantity file. Quantities are parsed on demand from the Mathematica-format
    text files distributed with PNPedia:
        https://github.com/davidtrestini/PNpedia
    """

    structure_name = "PNPedia"

    def __init__(self, path, download=False):
        """
        Initializes the PNPedia class.

        Parameters
        ----------
        path : str
            The path to the local copy of the PNPedia repository.
        download : bool, optional
            If True, downloads the PNPedia repository from GitHub. Default is
            False.
        """
        super().__init__(path)
        self._mathematica_parser = MathematicaParser()
        self._quantity_cache: dict = {}
        if download:
            self.download_pnpedia()
        self.__parse_pnpedia()

    def download_pnpedia(self):
        """
        Clone the PNPedia repository from GitHub using git.

        This method requires git to be installed on the system and accessible
        via the command line.
        """
        import subprocess

        # check if the path already exists
        if os.path.exists(self.path):
            logging.info(
                "PNPedia repository already exists at %s. Skipping download.",
                self.path,
            )
            return

        logging.info("Cloning PNPedia repository from GitHub...")
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/davidtrestini/PNpedia.git",
                self.path,
            ],
            check=True,
        )
        logging.info("PNPedia repository cloned to %s", self.path)

    def __parse_pnpedia(self):
        """
        Build the indexed PNPedia quantity structure for later lookup.
        """

        self._set_index(self.path, (".txt",))
        self.pnpedia_structure = self.indexed_paths
        logging.info("PNPedia structure parsed successfully.")

    def _resolve_quantity_path(self, name, path=None):
        if path is not None:
            resolved_path = self._resolve_existing_path(path, self.path)
            logging.info("Loading PN quantity from %s...", resolved_path)
            return resolved_path

        _, resolved_path = self._resolve_name(
            name,
            ambiguous_hint=(
                "Please specify more details "
                "(e.g., orbit type, spin, precession, tides)."
            ),
        )
        return resolved_path

    def _prepare_quantity_expression(self, content):
        converted_content = self.mathematica_to_python_vars(content)
        return self._mathematica_parser.normalize_source(converted_content)

    def _sympify_quantity_expression(self, prepared_content):
        content_py = (
            prepared_content.replace("[", "(").replace("]", ")").replace("^", "**")
        )
        return sp.sympify(content_py)

    def _parse_quantity_expression(self, content, path):
        prepared_content = self._prepare_quantity_expression(content)
        try:
            return parse_mathematica(prepared_content)
        except _PNPEDIA_PARSE_FALLBACK_EXCEPTIONS as exc:
            logging.warning(
                (
                    "Mathematica parse failed for %s; "
                    "falling back to sympy.sympify: %s"
                ),
                path,
                exc,
            )
            return self._sympify_quantity_expression(prepared_content)

    def _resolve_variable_symbol(self, variable):
        if isinstance(variable, str):
            return sp.symbols(variable)
        if isinstance(variable, sp.Symbol):
            return variable
        raise TypeError("variable must be a string or a sympy.Symbol")

    def _truncate_expression(self, pn_quantity, order, x_symbol):
        # Replace x-dependent logs with a placeholder so expansion and power
        # counting do not discard their multiplicative x dependence.
        pn_quantity = pn_quantity.replace(sp.log(x_symbol), _LOG_X_PLACEHOLDER)

        # Expand multiplicative structure to expose separate x powers in each
        # term for correct truncation without a full symbolic expand.
        pn_quantity = sp.expand_mul(pn_quantity)

        terms = pn_quantity.args if pn_quantity.is_Add else (pn_quantity,)
        term_powers = [_get_x_exponent(term, x_symbol) for term in terms]
        min_order, max_order = _get_x_power_range(pn_quantity, x_symbol)
        max_pn_order = max_order - min_order

        requested_pn_order = sp.Rational(order)
        if requested_pn_order > max_pn_order:
            logging.warning(
                "Requested PN order %s is higher than the maximum available "
                "order %s for this quantity. Returning up to available order "
                "%s.",
                order,
                max_pn_order,
                max_pn_order,
            )
            return pn_quantity.replace(_LOG_X_PLACEHOLDER, sp.log(x_symbol))

        target_order_power = min_order + requested_pn_order
        if target_order_power >= max_order:
            return pn_quantity.replace(_LOG_X_PLACEHOLDER, sp.log(x_symbol))

        truncated_terms = [
            term
            for term, term_power in zip(terms, term_powers)
            if term_power <= target_order_power
        ]
        truncated_quantity = (
            sp.Add(*truncated_terms) if truncated_terms else sp.Integer(0)
        )
        return truncated_quantity.replace(_LOG_X_PLACEHOLDER, sp.log(x_symbol))

    def get_pn_quantity(self, name, order, path=None, variable="x"):
        """
        Retrieve a PN quantity from PNPedia at the requested PN order.

        The PNPedia structure (inside Core post-Newtonian quantities) is:
        - orbit type (Circular, Elliptic, Hyperbolic)
        - Spin (NonSpinning, Spinning)
        - Precession (Precessing, Nonprecessing)
        - Tides (With tidal effects, Without tidal effects)
        - Quantity (
          Waveform, Fluxes, Constants of motion, Waveform frequencies):
            - Constants of motion (Energy, Angular momentum)
            - Fluxes (Angular momentum flux, Energy flux)
            - Waveform (h_l_m.txt)

        Quantities are stored as Mathematica-readable text files.
        They are converted into SymPy expressions on demand.

        Parameters
        ----------
        name : str
            The name of the PN quantity to retrieve,
        order : str
            The desired PN order (e.g., "1", "2", "3", etc.)
        path : str, optional
            The path to the PNPedia file to be loaded. If None, it is resolved
            from the indexed quantity name.

        Returns
        -------
        sympy expression
            The requested PN quantity as a sympy expression.
        """

        resolved_path = self._resolve_quantity_path(name=name, path=path)

        variable_key = variable if isinstance(variable, str) else variable.name
        cache_key = (resolved_path, str(order), variable_key)
        if cache_key in self._quantity_cache:
            return self._quantity_cache[cache_key]

        with open(resolved_path, "r", encoding="utf-8") as file:
            content = file.read()

        pn_quantity = self._parse_quantity_expression(content, resolved_path)
        x_symbol = self._resolve_variable_symbol(variable)
        truncated_quantity = self._truncate_expression(
            pn_quantity,
            order,
            x_symbol,
        )
        result = AnalyticExpression(truncated_quantity)
        self._quantity_cache[cache_key] = result
        return result

    def mathematica_to_python_vars(self, expr):
        """
        Convert Mathematica identifiers to Python-friendly names.

        Parameters
        ----------
        expr : sympy expression or str
            The sympy expression or raw Mathematica string containing
            Mathematica variables.

        Returns
        -------
        str
            The expression string with Mathematica identifiers converted to
            Python-friendly names.
        """

        return self._mathematica_parser.replace_identifiers(
            expr,
            _PNPEDIA_IDENTIFIER_REPLACEMENTS,
        )
