"""Generic helpers for parsing Mathematica source into SymPy expressions.

This module owns Mathematica syntax normalization, structural parsing, and
special-form conversion shared across analytic data sources. Caller-specific
identifier remappings are intentionally supplied by the caller rather than
hard-coded here so the parser stays library-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from tokenize import TokenError
from typing import Any, Mapping

import sympy as sp
from sympy.core.sympify import SympifyError
from sympy.parsing.mathematica import parse_mathematica


@dataclass(slots=True)
class ParsedMathematicaSource:
    """Structured representation of a parsed Mathematica source fragment.

    Parameters
    ----------
    assignments : dict[str, str], optional
        Top-level immediate assignments extracted from the source.
    definitions : dict[str, str], optional
        Top-level delayed definitions extracted from the source.
    metadata : dict[str, Any], optional
        Metadata association extracted from the source.
    last_expression : str or None, optional
        Final top-level expression encountered while parsing.
    """

    assignments: dict[str, str] = field(default_factory=dict)
    definitions: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_expression: str | None = None

    def resolve_expression(self, name: str | None = None) -> str | None:
        """Return the raw expression associated with a symbol or fallback.

        Parameters
        ----------
        name : str or None, optional
            Symbol name to resolve inside the assignment table.

        Returns
        -------
        str or None
            Expression string associated with ``name`` or the last parsed
            expression when ``name`` is not provided.
        """
        if isinstance(name, str):
            return self.assignments.get(name, name)

        if self.last_expression is None:
            return None

        return self.assignments.get(
            self.last_expression,
            self.last_expression,
        )


class MathematicaParser:
    """Parse Mathematica source text and convert symbolic forms to SymPy.

    The parser handles common source normalization steps, extracts metadata
    and assignments from Mathematica files, and converts special symbolic
    constructs such as ``SeriesData`` into explicit SymPy expressions.

    The class is intentionally generic. Library-specific naming conventions
    and identifier rewrites should stay with the caller and be applied through
    the helper methods exposed here.
    """

    def normalize_source(self, text: str) -> str:
        """Normalize raw Mathematica source before structural parsing.

        Parameters
        ----------
        text : str
            Raw Mathematica source text.

        Returns
        -------
        str
            Source with comments removed, box expressions replaced, and common
            escaped identifiers simplified.
        """
        text = self._strip_comments(text)
        text = self._replace_box_expressions(text)
        text = re.sub(r"\\\[([A-Za-z]+)\]", lambda match: match.group(1), text)
        return text.strip()

    def _strip_comments(self, text: str) -> str:
        """Strip Mathematica comments, including nested comment blocks.

        Parameters
        ----------
        text : str
            Raw Mathematica source text.

        Returns
        -------
        str
            Source text with comment blocks removed.

        Raises
        ------
        ValueError
            If the source contains an unmatched Mathematica comment block.
        """
        pieces = []
        comment_depth = 0
        in_string = False
        index = 0

        while index < len(text):
            if comment_depth == 0 and text[index] == '"':
                if not (in_string and index > 0 and text[index - 1] == "\\"):
                    in_string = not in_string
                pieces.append(text[index])
                index += 1
                continue

            if not in_string and text.startswith("(*", index):
                comment_depth += 1
                index += 2
                continue

            if comment_depth > 0:
                if text.startswith("(*", index):
                    comment_depth += 1
                    index += 2
                    continue
                if text.startswith("*)", index):
                    comment_depth -= 1
                    index += 2
                    continue
                index += 1
                continue

            pieces.append(text[index])
            index += 1

        if comment_depth != 0:
            raise ValueError("Unmatched Mathematica comment while parsing source")

        return "".join(pieces)

    def replace_identifiers(
        self,
        text: Any,
        replacements: Mapping[str, str],
    ) -> str:
        """Apply ordered textual identifier replacements to source.

        Parameters
        ----------
        text : Any
            Source object to convert to text before replacement.
        replacements : Mapping[str, str]
            Mapping from source identifiers to replacement identifiers.
            This table is caller-owned so source-specific naming policies stay
            outside the shared parser core.

        Returns
        -------
        str
            Updated text after all replacements have been applied.
        """
        updated_text = str(text)
        for source, target in sorted(
            replacements.items(),
            key=lambda item: -len(item[0]),
        ):
            updated_text = updated_text.replace(source, target)
        return updated_text

    def parse_source(self, text: str) -> ParsedMathematicaSource:
        """
        Extract assignments, delayed definitions, metadata, and the tail
        expression.

        Parameters
        ----------
        text : str
            Raw Mathematica source fragment.

        Returns
        -------
        ParsedMathematicaSource
            Structured representation of the parsed source.
        """
        normalized_source = self.normalize_source(text)

        if self.is_top_level_association(normalized_source):
            return ParsedMathematicaSource(
                metadata=self.parse_association(normalized_source)
            )

        if normalized_source.startswith("Module["):
            module_items = self.extract_module_items(normalized_source)
            return self._parse_source_items(module_items[1:])

        return self._parse_source_items([normalized_source])

    def parse_expression(self, expression_source: str):
        """Parse Mathematica syntax into a normalized SymPy expression.

        Parameters
        ----------
        expression_source : str
            Mathematica expression source to parse.

        Returns
        -------
        sympy.Basic
            Parsed expression after special-form normalization.

        Raises
        ------
        ValueError
            If the expression cannot be parsed by the Mathematica parser.
        """
        expression_source = self._replace_empty_series_data(expression_source)
        try:
            parsed = parse_mathematica(expression_source)
        except (
            SympifyError,
            SyntaxError,
            TokenError,
            TypeError,
            ValueError,
            NotImplementedError,
        ) as exc:
            raise ValueError("Failed to parse Mathematica expression") from exc
        return self._convert_special_forms(parsed)

    def extract_module_items(self, source: str) -> list[str]:
        """Return the ordered expressions stored inside a Mathematica module.

        Parameters
        ----------
        source : str
            Mathematica source string, optionally beginning with ``Module[``.

        Returns
        -------
        list[str]
            Top-level module arguments and body items.

        Raises
        ------
        ValueError
            If a ``Module`` form does not have the expected two arguments.
        """
        source = source.strip()
        if not source.startswith("Module["):
            return [source]

        bracket_start = source.find("[")
        bracket_end = self.find_matching_bracket(source, bracket_start)
        inner = source[bracket_start + 1 : bracket_end].strip()
        module_args = self.split_top_level(inner)
        if len(module_args) != 2:
            raise ValueError("Unexpected Mathematica module structure")

        body_items = self.split_top_level(module_args[1], separator=";")
        return [module_args[0], *body_items]

    def is_top_level_association(self, text: str) -> bool:
        """Return whether a string is a complete top-level association.

        Parameters
        ----------
        text : str
            Source text to inspect.

        Returns
        -------
        bool
            ``True`` when ``text`` begins with ``<|`` and ends with ``|>``.
        """
        text = text.strip()
        return text.startswith("<|") and text.endswith("|>")

    def find_matching_bracket(self, text: str, start_index: int) -> int:
        """Return the index of the closing bracket matching ``start_index``.

        Parameters
        ----------
        text : str
            Source text containing bracketed Mathematica syntax.
        start_index : int
            Index of the opening ``[`` to match.

        Returns
        -------
        int
            Index of the matching closing ``]``.

        Raises
        ------
        ValueError
            If no matching closing bracket is found.
        """
        depth = 0
        in_string = False
        index = start_index
        while index < len(text):
            char = text[index]
            if char == '"':
                in_string = not in_string
            elif not in_string:
                if char == "[":
                    depth += 1
                elif char == "]":
                    depth -= 1
                    if depth == 0:
                        return index
            index += 1

        raise ValueError("Unmatched '[' while parsing Mathematica source")

    def _iter_top_level(self, text: str):
        """
        Yield ``(index, char, at_top)`` for every character in *text*,
        where *at_top* is ``True`` when all bracket/string depths are zero.

        Multi-character tokens ``<|`` and ``|>`` advance the index by 2 and
        yield a single entry with ``char`` set to the two-character token.

        Parameters
        ----------
        text : str
            Source text to scan.

        Returns
        -------
        iterator
            Iterator over ``(index, token, at_top)`` tuples describing the
            top-level scanning state.
        """
        square_depth = 0
        curly_depth = 0
        paren_depth = 0
        assoc_depth = 0
        in_string = False
        index = 0

        while index < len(text):
            if not in_string and text.startswith("<|", index):
                assoc_depth += 1
                yield index, "<|", assoc_depth == 0
                index += 2
                continue
            if not in_string and text.startswith("|>", index):
                assoc_depth -= 1
                yield index, "|>", assoc_depth == 0
                index += 2
                continue

            char = text[index]
            if char == '"':
                in_string = not in_string
                yield index, char, False
            else:
                at_top = (
                    not in_string
                    and square_depth == 0
                    and curly_depth == 0
                    and paren_depth == 0
                    and assoc_depth == 0
                )
                if not in_string:
                    if char == "[":
                        square_depth += 1
                    elif char == "]":
                        square_depth -= 1
                    elif char == "{":
                        curly_depth += 1
                    elif char == "}":
                        curly_depth -= 1
                    elif char == "(":
                        paren_depth += 1
                    elif char == ")":
                        paren_depth -= 1
                yield index, char, at_top
            index += 1

    def split_top_level(self, text: str, separator: str = ",") -> list[str]:
        """Split text on separators that appear only at top level.

        Parameters
        ----------
        text : str
            Source text to split.
        separator : str, optional
            Separator token to honor only at top level.

        Returns
        -------
        list[str]
            Top-level segments with surrounding whitespace removed.
        """
        parts = []
        start = 0

        for index, _char, at_top in self._iter_top_level(text):
            if at_top and text.startswith(separator, index):
                parts.append(text[start:index].strip())
                start = index + len(separator)

        final_part = text[start:].strip()
        if final_part:
            parts.append(final_part)
        return parts

    def find_top_level(self, text: str, token: str) -> int:
        """Find the first top-level occurrence of a token in text.

        Parameters
        ----------
        text : str
            Source text to search.
        token : str
            Token to locate.

        Returns
        -------
        int
            Index of the first top-level occurrence, or ``-1`` when absent.
        """
        for index, _char, at_top in self._iter_top_level(text):
            if at_top and text.startswith(token, index):
                return index
        return -1

    def find_top_level_assignment(self, text: str) -> int:
        """Locate a plain top-level assignment operator in Mathematica code.

        Parameters
        ----------
        text : str
            Source text to search.

        Returns
        -------
        int
            Index of the assignment operator, or ``-1`` when none is found.
        """
        for index, char, at_top in self._iter_top_level(text):
            if at_top and char == "=":
                prev_char = text[index - 1] if index > 0 else ""
                next_char = text[index + 1] if index + 1 < len(text) else ""
                if prev_char not in {":", "-", "<", ">", "="} and next_char != ">":
                    return index
        return -1

    def parse_association(self, text: str) -> dict[str, Any]:
        """Parse a Mathematica association into Python metadata.

        Parameters
        ----------
        text : str
            Mathematica association of the form ``<| ... |>``.

        Returns
        -------
        dict[str, Any]
            Parsed association values converted recursively into Python types.
        """
        inner = text.strip()[2:-2].strip()
        metadata = {}
        for item in self.split_top_level(inner):
            rule_index = self.find_top_level(item, "->")
            if rule_index == -1:
                continue
            key = self.strip_quotes(item[:rule_index].strip())
            value = item[rule_index + 2 :].strip()
            metadata[key] = self.parse_metadata_value(value)
        return metadata

    def parse_metadata_value(self, value: str) -> Any:
        """Parse a metadata value from a Mathematica association.

        Parameters
        ----------
        value : str
            Raw association value.

        Returns
        -------
        Any
            Parsed Python value, preserving strings, lists, and nested
            associations.
        """
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            return self.strip_quotes(value)

        if value.startswith("{") and value.endswith("}"):
            inner = value[1:-1].strip()
            if not inner:
                return []

            items = self.split_top_level(inner)
            if all(self.find_top_level(item, "->") != -1 for item in items):
                parsed = {}
                for item in items:
                    rule_index = self.find_top_level(item, "->")
                    key = self.strip_quotes(item[:rule_index].strip())
                    parsed[key] = self.parse_metadata_value(
                        item[rule_index + 2 :].strip()
                    )
                return parsed

            return [self.parse_metadata_value(item) for item in items]

        return value

    def strip_quotes(self, text: str) -> str:
        """Remove one surrounding pair of double quotes when present.

        Parameters
        ----------
        text : str
            Text to normalize.

        Returns
        -------
        str
            Unquoted text when the input was wrapped in double quotes.
        """
        if text.startswith('"') and text.endswith('"'):
            return text[1:-1]
        return text

    def _parse_source_items(self, items: list[str]) -> ParsedMathematicaSource:
        """Parse a sequence of top-level Mathematica source items.

        Parameters
        ----------
        items : list[str]
            Top-level source items extracted from a file or module body.

        Returns
        -------
        ParsedMathematicaSource
            Structured representation built from the supplied items.
        """
        parsed = ParsedMathematicaSource()
        for item in items:
            if self.is_top_level_association(item):
                parsed.metadata.update(self.parse_association(item))
                continue

            delayed_index = self.find_top_level(item, ":=")
            if delayed_index != -1:
                lhs = item[:delayed_index].strip()
                rhs = item[delayed_index + 2 :].strip()
                parsed.definitions[lhs] = rhs
                continue

            assign_index = self.find_top_level_assignment(item)
            if assign_index != -1:
                lhs = item[:assign_index].strip()
                rhs = item[assign_index + 1 :].strip()
                parsed.assignments[lhs] = rhs
                parsed.last_expression = lhs
                continue

            parsed.last_expression = item.strip()

        return parsed

    def _replace_box_expressions(self, text: str) -> str:
        """Replace notebook box expressions with placeholder tokens.

        Parameters
        ----------
        text : str
            Raw Mathematica source text.

        Returns
        -------
        str
            Source text with box expressions replaced by ``BoxExpr`` tokens.
        """
        pattern = re.compile(r"\\!\\\(\\\*.*?\\\)", re.DOTALL)
        counter = 0
        while True:
            match = pattern.search(text)
            if match is None:
                return text

            placeholder = f"BoxExpr{counter}"
            text = text[: match.start()] + placeholder + text[match.end() :]
            counter += 1

    def _replace_empty_series_data(self, text: str) -> str:
        """Replace empty ``SeriesData`` constructs with explicit zeros.

        Parameters
        ----------
        text : str
            Mathematica expression source.

        Returns
        -------
        str
            Updated source string with empty ``SeriesData`` blocks replaced by
            ``0``.
        """
        search_start = 0
        while True:
            series_index = text.find("SeriesData[", search_start)
            if series_index == -1:
                return text

            bracket_start = text.find("[", series_index)
            bracket_end = self.find_matching_bracket(text, bracket_start)
            inner = text[bracket_start + 1 : bracket_end]
            series_args = self.split_top_level(inner)

            if len(series_args) == 6 and series_args[2].strip() == "{}":
                text = text[:series_index] + "0" + text[bracket_end + 1 :]
                search_start = max(series_index - 1, 0)
                continue

            search_start = series_index + len("SeriesData[")

    def _convert_special_forms(self, expr):
        """Normalize parser-specific symbolic forms into standard SymPy.

        Parameters
        ----------
        expr : object
            Parsed expression tree or literal value.

        Returns
        -------
        object
            Expression with special Mathematica forms converted when needed.
        """
        if not isinstance(expr, sp.Basic):
            return expr

        func_name = getattr(expr.func, "__name__", str(expr.func))
        if func_name == "SeriesData":
            return self._convert_series_data(expr)

        if func_name == "ResummedSeriesData":
            prefactor = self._convert_special_forms(expr.args[0])
            series_expr = self._convert_special_forms(expr.args[1])
            return prefactor * series_expr

        if func_name == "HoldForm":
            return self._convert_special_forms(expr.args[0])

        converted_args = [self._convert_special_forms(arg) for arg in expr.args]
        if not converted_args:
            return self._convert_log_symbol(expr)
        return self._convert_log_symbol(expr.func(*converted_args))

    def _convert_log_symbol(self, expr):
        """Convert placeholder ``LogX`` symbols back into explicit logarithms.

        Parameters
        ----------
        expr : object
            Symbolic object to normalize.

        Returns
        -------
        object
            Converted SymPy logarithm when the symbol matches the placeholder
            pattern, otherwise the input expression.
        """
        if (
            isinstance(expr, sp.Symbol)
            and expr.name.startswith("Log")
            and len(expr.name) > 3
        ):
            return sp.log(sp.Symbol(expr.name[3:]))
        return expr

    def _convert_series_data(self, expr):
        """Convert a Mathematica ``SeriesData`` object into a SymPy series sum.

        Parameters
        ----------
        expr : sympy.Basic
            Parsed ``SeriesData`` expression.

        Returns
        -------
        sympy.Expr
            Explicit SymPy expression equivalent to the series data.
        """
        variable = self._convert_special_forms(expr.args[0])
        point = self._convert_special_forms(expr.args[1])
        coefficients = self._series_coefficients(expr.args[2])
        n_min = sp.Integer(expr.args[3])
        denominator = sp.Integer(expr.args[5])

        result = sp.Integer(0)
        for offset, coefficient in enumerate(coefficients):
            coefficient_expr = self._convert_special_forms(coefficient)
            if coefficient_expr == 0:
                continue

            exponent = sp.Rational(n_min + offset, denominator)
            result += coefficient_expr * self._series_basis(
                variable,
                point,
                exponent,
            )

        return result

    def _series_coefficients(self, coefficient_data):
        """Return the coefficient list encoded inside ``SeriesData``.

        Parameters
        ----------
        coefficient_data : sympy.Basic
            Parsed coefficient payload from a ``SeriesData`` expression.

        Returns
        -------
        list[sympy.Basic]
            Coefficients expanded into a plain Python list.
        """
        if isinstance(coefficient_data, sp.Tuple):
            return list(coefficient_data)

        if isinstance(coefficient_data, sp.Mul):
            tuple_arguments = [
                arg for arg in coefficient_data.args if isinstance(arg, sp.Tuple)
            ]
            if len(tuple_arguments) == 1:
                tuple_argument = tuple_arguments[0]
                multiplier = sp.Mul(
                    *[arg for arg in coefficient_data.args if arg is not tuple_argument]
                )
                return [multiplier * term for term in tuple_argument]

        return [coefficient_data]

    def _series_basis(self, variable, point, exponent):
        """Return the basis monomial associated with a series coefficient.

        Parameters
        ----------
        variable : sympy.Symbol
            Series expansion variable.
        point : sympy.Expr
            Expansion point.
        exponent : sympy.Expr
            Exponent associated with the current coefficient.

        Returns
        -------
        sympy.Expr
            Basis term for the requested series coefficient.
        """
        if point in {sp.Symbol("Infinity"), sp.oo}:
            return variable ** (-exponent)

        if point == 0:
            return variable**exponent
        return (variable - point) ** exponent


__all__ = ["MathematicaParser", "ParsedMathematicaSource"]
