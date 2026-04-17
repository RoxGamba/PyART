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
    """Parsed representation of a single BHPT series file.

    Parameters
    ----------
    key : str
        Canonical catalog key for the entry.
    path : str
        Absolute path to the Mathematica source file.
    metadata : dict[str, Any]
        Metadata association extracted from the file.
    assignments : dict[str, str]
        Top-level Mathematica assignments found in the source.
    definitions : dict[str, str]
        Delayed Mathematica definitions found in the source.
    expr : sympy.Basic
        Parsed SymPy expression for the selected series.
    quantity : AnalyticExpression
        Wrapped analytic expression corresponding to ``expr``.
    """

    key: str
    path: str
    metadata: dict[str, Any]
    assignments: dict[str, str]
    definitions: dict[str, str]
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
            raise KeyError(field_name) from exc


class BHPTPN(AnalyticCatalog):
    """
    Interface to BlackHolePerturbationToolkit self-force series data.

    Parameters
    ----------
    path : str
        Path to the local BHPT repository or extracted data directory.
    download : bool, optional
        If ``True``, clone the repository when it is not already present.

    Notes
    -----
    The expected repository layout follows the
    BlackHolePerturbationToolkit PostNewtonianSelfForce project.
    """

    structure_name = "BHPTPN"

    def __init__(self, path, download=False):
        """Initialize the BHPT catalog wrapper.

        Parameters
        ----------
        path : str
            Path to the directory where BHPT expression files are stored.
        download : bool, optional
            If ``True``, clone the upstream repository before indexing.

        Returns
        -------
        None
            The catalog index and parser cache are initialized on the
            instance.
        """
        super().__init__(path)
        if download:
            self.download_bhptpn()
        self._entry_cache = {}
        self._mathematica_parser = MathematicaParser()
        self.__parse_bhptpn()

    def download_bhptpn(self):
        """Clone the BHPT repository into the configured catalog path.

        Returns
        -------
        None
            The repository is cloned when it is not already available.
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
        """Index the BHPT Mathematica files available on disk.

        Returns
        -------
        None
            The indexed path mapping is stored on the instance.

        Raises
        ------
        FileNotFoundError
            If the configured BHPT path does not exist.
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
        """Return metadata for a resolved BHPT entry.

        Parameters
        ----------
        name : str or None, optional
            Indexed quantity name used for lookup.
        path : str or None, optional
            Direct path to the Mathematica file.

        Returns
        -------
        dict[str, Any]
            Metadata association extracted from the selected entry.
        """
        return self.get_entry(name=name, path=path)["metadata"]

    def get_pn_quantity(self, name=None, path=None):
        """Return a BHPT series as an analytic expression wrapper.

        Parameters
        ----------
        name : str or None, optional
            Indexed quantity name used for lookup.
        path : str or None, optional
            Direct path to the Mathematica file.

        Returns
        -------
        AnalyticExpression
            Parsed BHPT series wrapped for symbolic and numerical use.
        """
        return self.get_entry(name=name, path=path)["quantity"]

    def get_entry(self, name=None, path=None):
        """
        Resolve, parse, and cache a BHPT Mathematica file.

        The returned :class:`BHPTEntry` contains the resolved key and path,
        parsed
        metadata, raw assignment tables extracted from the Mathematica
        source, the SymPy expression, and the corresponding
        `AnalyticExpression` wrapper.

        Parameters
        ----------
        name : str or None, optional
            Indexed quantity name used for lookup.
        path : str or None, optional
            Direct path to a Mathematica file inside the indexed tree.

        Returns
        -------
        BHPTEntry
            Parsed and cached representation of the requested series file.

        Raises
        ------
        ValueError
            If the source or selected expression cannot be parsed.
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

        Parameters
        ----------
        name : str or None, optional
            Indexed quantity name used for lookup.
        path : str or None, optional
            Direct path to a Mathematica file inside the indexed tree.

        Returns
        -------
        tuple[str, str]
            Canonical quantity key and absolute file path.
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
