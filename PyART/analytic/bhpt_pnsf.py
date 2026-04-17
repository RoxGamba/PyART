import logging
import os
from dataclasses import dataclass
from typing import Any

import sympy as sp

from .expr import AnalyticExpression
from .mathematica_parser import MathematicaParser
from .analytic_catalog import AnalyticCatalog


@dataclass(slots=True)
class BHPTEntry:
    """Parsed representation of a single BHPT series file."""

    key: str
    path: str
    metadata: dict[str, Any]
    assignments: dict[str, str]
    definitions: dict[str, str]
    expr: sp.Basic
    quantity: AnalyticExpression

    def __getitem__(self, field_name: str) -> Any:
        try:
            return getattr(self, field_name)
        except AttributeError as exc:
            raise KeyError(field_name) from exc


class BHPTPN(AnalyticCatalog):
    """
    Interface with the BlackHolePerturbationToolkit (BHPT)
    post-Newtonian self force expressions.
    See:
    https://github.com/BlackHolePerturbationToolkit/PostNewtonianSelfForce
    """

    structure_name = "BHPTPN"

    def __init__(self, path, download=False):
        """
        Initialize the BHPTPN class.

        Parameters
        ----------
        path : str
            The path to the directory where the BHPTPN expressions are stored.
        download : bool, optional
            Whether to download the BHPTPN expressions if they are
            not found in the specified path.
        """
        super().__init__(path)
        if download:
            self.download_bhptpn()
        self._entry_cache = {}
        self._mathematica_parser = MathematicaParser()
        self.__parse_bhptpn()

    def download_bhptpn(self):
        """
        Clone the BHPTPN repository from GitHub to the specified path.
        """
        import subprocess

        # check if the path already exists
        if os.path.exists(self.path):
            logging.info(
                "BHPT repository already exists at %s. Skipping download.",
                self.path,
            )
            return

        logging.info("Cloning BHPT repository to %s...", self.path)
        subprocess.run(
            [
                "git",
                "clone",
                (
                    "https://github.com/"
                    "BlackHolePerturbationToolkit/"
                    "PostNewtonianSelfForce.git"
                ),
                self.path,
            ],
            check=True,
        )
        logging.info("BHPT repository cloned to %s.", self.path)

    def __parse_bhptpn(self):
        """
        Internal method to parse the BHPTPN expressions from the
        files in the specified path.
        """

        if not os.path.isdir(self.path):
            raise FileNotFoundError(f"BHPT path does not exist: {self.path}")

        series_root = os.path.join(self.path, "SeriesData")
        if os.path.isdir(series_root):
            self.series_root = series_root
        else:
            self.series_root = self.path

        self._set_index(self.series_root, (".m",))
        self.bhptpn_structure = self.indexed_paths
        logging.info("BHPT structure parsed successfully.")

    def get_metadata(self, name=None, path=None):
        """Return the metadata association for a resolved BHPT entry."""
        return self.get_entry(name=name, path=path)["metadata"]

    def get_pn_quantity(self, name=None, path=None):
        """Return the parsed BHPT series wrapped as an AnalyticExpression."""
        return self.get_entry(name=name, path=path)["quantity"]

    def get_entry(self, name=None, path=None):
        """
        Resolve, parse, and cache a BHPT Mathematica file.

        The returned :class:`BHPTEntry` contains the resolved key and path,
        parsed
        metadata, raw assignment tables extracted from the Mathematica
        source, the SymPy expression, and the corresponding
        `AnalyticExpression` wrapper.
        """
        key, resolved_path = self._resolve_entry(name=name, path=path)
        if resolved_path in self._entry_cache:
            return self._entry_cache[resolved_path]

        with open(resolved_path, "r", encoding="utf-8") as file:
            raw_content = file.read()

        try:
            parsed_source = self._mathematica_parser.parse_source(raw_content)
        except ValueError as exc:
            raise ValueError(
                f"Failed to parse Mathematica source in {resolved_path}"
            ) from exc

        expression_source = parsed_source.resolve_expression(
            parsed_source.metadata.get("Series")
        )

        if not expression_source:
            raise ValueError(
                "Could not determine BHPT series expression for " f"{resolved_path}"
            )

        try:
            sympy_expr = self._mathematica_parser.parse_expression(expression_source)
        except ValueError as exc:
            raise ValueError(
                f"Failed to parse Mathematica expression in {resolved_path}"
            ) from exc

        entry = BHPTEntry(
            key=key,
            path=resolved_path,
            metadata=parsed_source.metadata,
            assignments=parsed_source.assignments,
            definitions=parsed_source.definitions,
            expr=sympy_expr,
            quantity=AnalyticExpression(sympy_expr),
        )
        self._entry_cache[resolved_path] = entry
        return entry

    def _resolve_entry(self, name=None, path=None):
        """
        Resolve a user query to an indexed BHPT file path and canonical key.

        A direct path lookup wins when `path` is provided, but the resolved
        file must still live inside the configured BHPT tree. Otherwise the
        method performs a normalized token match on the indexed quantity keys.
        """
        if path is not None:
            key, resolved_path = self._resolve_indexed_path(
                path,
                self.series_root,
                self.path,
            )
            return key, resolved_path

        key, resolved_path = self._resolve_name(
            name,
            ambiguous_hint="Please specify more details.",
        )
        return key, resolved_path
