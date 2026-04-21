"""PNPedia catalog access with PNPedia-specific identifier normalization.

The shared Mathematica parser remains library-agnostic. PNPedia-specific
identifier rewrites are defined here and applied before delegating generic
normalization and parsing to ``MathematicaParser``.
"""

import logging
import os
import re
from dataclasses import dataclass
from tokenize import TokenError
from typing import Any

import sympy as sp
from sympy.core.sympify import SympifyError
from sympy.parsing.mathematica import parse_mathematica

from .expr import AnalyticExpression
from .mathematica_parser import MathematicaParser
from .analytic_catalog import AnalyticCatalog

# These replacements encode PNPedia naming conventions only. They stay here so
# the shared Mathematica parser remains generic across analytic sources.

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
    """Combine identifier replacement tables into one validated mapping.

    Parameters
    ----------
    *groups : tuple[tuple[str, str], ...]
        Replacement groups containing ``(source, target)`` pairs.

    Returns
    -------
    dict[str, str]
        Combined mapping from Mathematica identifiers to Python-friendly
        replacements.

    Raises
    ------
    ValueError
        If the same source identifier appears in more than one group.
    """
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


@dataclass(slots=True)
class PNPediaEntry:
    """
    Parsed representation of a PNPedia quantity file and README.md

    Parameters
    ----------
    key: str
        The resolved key corresponding to the selected quantity.
    path: str
        The absolute path to the selected quantity file.
    metadata: dict
        Metadata extracted from the quantity file and associated README.md.
    expr : sympy.Basic
        The parsed symbolic expression for the quantity.
    quantity: AnalyticExpression
        The AnalyticExpression for the quantity corresponding to `expr`.
    """

    key: str
    path: str
    metadata: dict[str, Any]
    expr: sp.Basic
    quantity: AnalyticExpression

    def __getitem__(self, field_name: str) -> Any:
        """Return an entry attribute using mapping-style access.

        Parameters
        ----------
        field_name : str
            Name of the dataclass field to retrieve.

        Returns
        -------
        Any
            Value stored in the requested field.

        Raises
        ------
        KeyError
            If ``field_name`` is not a valid attribute name.
        """
        try:
            return getattr(self, field_name)
        except AttributeError as exc:
            raise KeyError(f"Invalid field name '{field_name}'") from exc


class PNPedia(AnalyticCatalog):
    """
    Collection of PN quantities loaded from the PNPedia repository.

    Acts as a dictionary-backed store of
    :class:`~PyART.analytic.expr.AnalyticExpression` objects, one per PN
    quantity file. Quantities are parsed on demand from the Mathematica-format
    text files distributed with PNPedia.

    PNPedia-specific identifier rewriting is handled in this module before
    the source is passed to the shared
    :class:`~PyART.analytic.mathematica_parser.MathematicaParser`.

    Source files are distributed with PNPedia:
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
        self._entry_cache: dict[str, PNPediaEntry] = {}
        if download:
            self.download_pnpedia()
        self.__parse_pnpedia()

    def download_pnpedia(self):
        """Clone the PNPedia repository into the configured path.

        Returns
        -------
        None
            The repository is cloned when it is not already available.
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
        """Index the available PNPedia quantity files.

        Returns
        -------
        None
            The quantity index is stored on the instance.
        """

        self._set_index(self.path, (".txt",))
        self.pnpedia_structure = self.indexed_paths
        logging.info("PNPedia structure parsed successfully.")

    def _resolve_entry(self, name=None, path=None):
        """Resolve a quantity request to a canonical key and file path.

        Parameters
        ----------
        name : str or None, optional
            Indexed quantity name or tokenized query.
        path : str or None, optional
            Direct path to a quantity file.

        Returns
        -------
        tuple[str, str]
            Canonical quantity key and absolute file path.
        """
        if path is not None:
            resolved_path = self._resolve_existing_path(path, self.path)
            key = self._path_to_key.get(resolved_path)
            if key is None:
                key = os.path.splitext(os.path.basename(resolved_path))[0].lower()
            logging.info("Loading PN quantity from %s...", resolved_path)
            return key, resolved_path

        key, resolved_path = self._resolve_name(
            name,
            ambiguous_hint=(
                "Please specify more details "
                "(e.g., orbit type, spin, precession, tides)."
            ),
        )
        return key, resolved_path

    def _prepare_quantity_expression(self, content):
        """Normalize a raw PNPedia file before symbolic parsing.

        PNPedia-specific identifier substitutions are applied locally first,
        then the shared Mathematica parser performs generic source
        normalization.

        Parameters
        ----------
        content : str
            Raw Mathematica-formatted file content.

        Returns
        -------
        str
            Normalized source string ready for SymPy parsing.
        """
        converted_content = self.mathematica_to_python_vars(content)
        return self._mathematica_parser.normalize_source(converted_content)

    def _sympify_quantity_expression(self, prepared_content):
        """Parse normalized content through a direct ``sympify`` fallback.

        Parameters
        ----------
        prepared_content : str
            Normalized expression source.

        Returns
        -------
        sympy.Expr
            Expression parsed with bracket and power syntax converted to Python
            equivalents.
        """
        content_py = prepared_content.replace("[", "(")
        content_py = content_py.replace("]", ")").replace("^", "**")
        return sp.sympify(content_py)

    def _parse_quantity_expression(self, content, path):
        """Parse a PNPedia quantity with a ``sympify`` fallback path.

        Parameters
        ----------
        content : str
            Raw quantity file content.
        path : str
            Path used for logging and error context.

        Returns
        -------
        sympy.Expr
            Parsed symbolic expression.
        """
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

    def __parse_quantity_readme(self, readme_content, path):
        """
        Parse the README.md associated with a PNPedia quantity file to extract
        metadata such as arXiv references, notation, and endorser information.

        Notation is usually under a "Notations" section, with
        quantities itemized via "*" bullets, as e.g.:
        * ``quantity`` is the ...

        Parameters
        ----------
        readme_content : str
            The content of the README.md file as a string.
        path : str
            The path to the README.md file, used for logging context.

        Returns
        dict
            A dictionary containing extracted metadata fields. Possible keys include
            'arXiv_references', 'notation', and 'endorser'.
        """
        # arXiv references in the form arXiv:xxxx.xxxxx or arXiv:xxxx.xxxxxvN
        arxiv_refs = list(
            dict.fromkeys(re.findall(r"arXiv:\d{4}\.\d{4,5}(?:v\d+)?", readme_content))
        )
        if not arxiv_refs:
            logging.warning("No arXiv references found in README.md at %s", path)

        # Look for notation, in the section starting with "Notations"
        notation = {}
        if "Notations" in readme_content:
            notations_section = readme_content.split("Notations", 1)[1]
            for line in notations_section.splitlines():
                if line.lstrip().startswith("#"):
                    break  # stop at the next section
                stripped_line = line.strip()
                if stripped_line.startswith("*"):
                    bullet = stripped_line.lstrip("*").strip()
                    if " is " in bullet:
                        quantity_name, description = bullet.split(" is ", 1)
                        quantity_name = quantity_name.strip().strip("`")
                        notation[quantity_name] = description.strip()

        # look for endorser, in the section starting with "Endorsers".
        # they are in the form [Endorser Name](website) [[ORCID](orcid_link)]
        # separated by a line break. We just extract the name
        endorsers = []
        if "Endorsers" in readme_content:
            endorsers_section = readme_content.split("Endorsers", 1)[1]
            for line in endorsers_section.splitlines():
                if line.lstrip().startswith("#"):
                    break  # stop at the next section
                # isolate first [Endorser Name](link) match
                match = re.search(r"\[([^\]]+)\]\(", line)
                if match:
                    endorsers.append(match.group(1).strip())
        if not endorsers:
            logging.warning("No endorsers found in README.md at %s", path)

        return {
            "arxiv_references": arxiv_refs,
            "notation": notation,
            "endorsers": endorsers,
        }

    def get_entry(self, name=None, path=None):
        """
        Retrieve a PN quantity from PNPedia.

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

        Inside each folder is also a README.md providing:
        - arXiv references/sources
        - Notation & definitions
        - Endorser

        Parameters
        ----------
        name : str
            The name of the PN quantity to retrieve,
        path : str, optional
            The path to the PNPedia file to be loaded. If None, it is resolved
            from the indexed quantity name.

        Returns
        -------
        PNPediaEntry
            Parsed and cached representation of the requested quantity.
        """
        key, resolved_path = self._resolve_entry(name=name, path=path)
        if resolved_path in self._entry_cache:
            return self._entry_cache[resolved_path]

        # quantity files
        with open(resolved_path, "r", encoding="utf-8") as file:
            content = file.read()
        # readme file
        readme_path = os.path.join(os.path.dirname(resolved_path), "README.md")
        if os.path.isfile(readme_path):
            with open(readme_path, "r", encoding="utf-8") as readme_file:
                readme_content = readme_file.read()
        else:
            logging.warning("README.md not found for %s", resolved_path)
            readme_content = ""

        pn_quantity = self._parse_quantity_expression(content, resolved_path)
        metadata = self.__parse_quantity_readme(readme_content, readme_path)
        quantity = AnalyticExpression(pn_quantity)
        entry = PNPediaEntry(
            key=key,
            path=resolved_path,
            metadata=metadata,
            expr=pn_quantity,
            quantity=quantity,
        )
        self._entry_cache[resolved_path] = entry
        return entry

    def mathematica_to_python_vars(self, expr):
        """
        Convert Mathematica identifiers to Python-friendly names.

        This translation intentionally stays local to PNPedia because the
        shared Mathematica parser should not embed catalog-specific naming
        policy.

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
