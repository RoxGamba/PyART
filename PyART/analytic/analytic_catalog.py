"""Shared infrastructure for file-backed analytic catalogs."""

from __future__ import annotations

import os
from collections.abc import Mapping
from types import MappingProxyType


def build_quantity_key(root: str, file_name: str, base_root: str) -> str:
    """Build a normalized lookup key for a catalog quantity file.

    Parameters
    ----------
    root : str
        Directory that contains the quantity file.
    file_name : str
        File name of the quantity entry.
    base_root : str
        Root directory used as the origin for the relative catalog structure.

    Returns
    -------
    str
        Lowercase key composed from the relative directory structure and the
        file stem.
    """
    relative_root = os.path.relpath(root, base_root)
    structure = ""
    if relative_root != ".":
        structure = relative_root.replace(os.sep, "_").replace(" ", "-")
        structure = structure.lower()

    quantity_name = os.path.splitext(file_name)[0].lower()
    return f"{structure}_{quantity_name}" if structure else quantity_name


def index_quantity_files(
    base_root: str,
    suffixes: tuple[str, ...],
) -> dict[str, str]:
    """Index catalog quantity files under a root directory.

    Parameters
    ----------
    base_root : str
        Directory whose descendants should be scanned.
    suffixes : tuple[str, ...]
        File suffixes that identify valid quantity files.

    Returns
    -------
    dict[str, str]
        Mapping from normalized quantity keys to absolute file paths.
    """
    indexed_paths = {}
    for root, dirs, files in os.walk(base_root):
        dirs.sort()
        for file_name in sorted(files):
            if not file_name.endswith(suffixes):
                continue

            key = build_quantity_key(root, file_name, base_root)
            indexed_paths[key] = os.path.abspath(os.path.join(root, file_name))

    return indexed_paths


def normalize_query(text: str) -> str:
    """Normalize user lookup text to the internal key token format.

    Parameters
    ----------
    text : str
        Raw user-provided lookup string.

    Returns
    -------
    str
        Normalized key token string using lowercase underscores.
    """
    return text.lower().replace("-", "_").replace(" ", "_").strip("_")


def resolve_path_in_roots(path: str, *roots: str) -> str:
    """Resolve a path directly or relative to one of several roots.

    Parameters
    ----------
    path : str
        Absolute or relative filesystem path.
    *roots : str
        Candidate base directories used when ``path`` is relative.

    Returns
    -------
    str
        Absolute path resolved from ``path`` and the provided roots.
    """
    if os.path.isabs(path):
        return os.path.abspath(path)

    for root in roots:
        if not root:
            continue

        candidate = os.path.abspath(os.path.join(root, path))
        if os.path.exists(candidate):
            return candidate

    return os.path.abspath(path)


def resolve_query_match(
    name: str,
    indexed_paths: Mapping[str, str],
    structure_name: str,
    ambiguous_hint: str | None = None,
) -> tuple[str, str]:
    """Resolve a user query to a single indexed catalog entry.

    Parameters
    ----------
    name : str
        User-facing quantity name or partial key.
    indexed_paths : Mapping[str, str]
        Mapping of canonical quantity keys to file paths.
    structure_name : str
        Human-readable catalog name used in error messages.
    ambiguous_hint : str or None, optional
        Extra text appended when the query matches more than one entry.

    Returns
    -------
    tuple[str, str]
        Canonical quantity key and resolved file path.

    Raises
    ------
    ValueError
        If ``name`` is missing, matches no entries, or matches multiple
        entries.
    """
    if not name:
        raise ValueError("Either 'name' or 'path' must be provided")

    name_pieces = [piece for piece in normalize_query(name).split("_") if piece]
    matches = []
    for key, path in indexed_paths.items():
        key_terms = normalize_query(key).split("_")
        if all(piece in key_terms for piece in name_pieces):
            matches.append((key, path))

    if not matches:
        raise ValueError(
            f"No matches found for quantity name '{name}' "
            f"in {structure_name} structure."
        )

    if len(matches) > 1:
        message = (
            f"Multiple matches found for quantity name '{name}' "
            f"in {structure_name} structure: "
            f"{[key for key, _ in matches]}."
        )
        if ambiguous_hint:
            message = f"{message} {ambiguous_hint}"
        raise ValueError(message)

    return matches[0]


class AnalyticCatalog:
    """Parent base class for analytic catalogs.

    Parameters
    ----------
    path : str
        Path to the catalog

    Attributes
    ----------
    path : str
        Absolute path to the catalog.
    """

    structure_name = "analytic catalog"

    def __init__(self, path: str):
        """Store the catalog path and initialize lookup state.

        Parameters
        ----------
        path : str
            Path to the catalog on disk.

        Returns
        -------
        None
            This initializer configures internal indexing attributes.
        """
        self.path = os.path.abspath(path)
        self._index_root = self.path
        self._indexed_paths: dict[str, str] = {}
        self._path_to_key: dict[str, str] = {}

    def _set_index(self, base_root: str, suffixes: tuple[str, ...]) -> None:
        """Populate the catalog index from the given root directory.

        Parameters
        ----------
        base_root : str
            Directory to scan for quantity files.
        suffixes : tuple[str, ...]
            File suffixes to include in the index.

        Returns
        -------
        None
            The catalog index is stored on the instance.

        Raises
        ------
        FileNotFoundError
            If ``base_root`` does not exist or is not a directory.
        """
        self._index_root = os.path.abspath(base_root)
        if not os.path.isdir(self._index_root):
            raise FileNotFoundError(
                f"{self.structure_name} catalog root not found: "
                f"'{self._index_root}'"
            )
        self._indexed_paths = index_quantity_files(self._index_root, suffixes)
        self._path_to_key = {path: key for key, path in self._indexed_paths.items()}

    @property
    def indexed_paths(self) -> MappingProxyType[str, str]:
        """Expose the indexed catalog paths as a read-only mapping.

        Returns
        -------
        types.MappingProxyType[str, str]
            Immutable view of the canonical key to path mapping.
        """
        return MappingProxyType(self._indexed_paths)

    def list_quantities(self) -> list[str]:
        """Return the sorted list of indexed quantity keys.

        Returns
        -------
        list[str]
            Sorted canonical keys currently stored in the catalog index.
        """
        return sorted(self._indexed_paths)

    def _resolve_name(
        self,
        name: str,
        ambiguous_hint: str | None = None,
    ) -> tuple[str, str]:
        """Resolve a quantity name against the catalog index.

        Parameters
        ----------
        name : str
            User-facing quantity name or partial key.
        ambiguous_hint : str or None, optional
            Additional guidance appended when multiple matches are found.

        Returns
        -------
        tuple[str, str]
            Canonical key and resolved file path.
        """
        return resolve_query_match(
            name,
            self._indexed_paths,
            self.structure_name,
            ambiguous_hint=ambiguous_hint,
        )

    def _resolve_existing_path(self, path: str, *roots: str) -> str:
        """Resolve a file path and ensure the target exists.

        Parameters
        ----------
        path : str
            Absolute or relative file path.
        *roots : str
            Candidate roots used when ``path`` is relative.

        Returns
        -------
        str
            Absolute path to an existing file.

        Raises
        ------
        FileNotFoundError
            If the resolved path does not exist.
        """
        resolved_path = resolve_path_in_roots(path, *roots)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(
                f"{self.structure_name} quantity file not found: " f"'{resolved_path}'"
            )
        return resolved_path

    def _resolve_indexed_path(self, path: str, *roots: str) -> tuple[str, str]:
        """Resolve a path and verify that it belongs to the catalog index.

        Parameters
        ----------
        path : str
            Absolute or relative file path.
        *roots : str
            Candidate roots used when ``path`` is relative.

        Returns
        -------
        tuple[str, str]
            Canonical quantity key and absolute file path.

        Raises
        ------
        ValueError
            If the resolved path exists outside the indexed tree or does not
            correspond to an indexed quantity.
        """
        resolved_path = resolve_path_in_roots(path, *roots)
        if resolved_path not in self._path_to_key:
            if os.path.exists(resolved_path):
                raise ValueError(
                    f"{self.structure_name} file '{resolved_path}' exists but "
                    f"is outside the indexed root '{self._index_root}'."
                )
            raise ValueError(
                f"No {self.structure_name} quantity found at path '{path}'"
            )
        return self._path_to_key[resolved_path], resolved_path


__all__ = [
    "AnalyticCatalog",
]
