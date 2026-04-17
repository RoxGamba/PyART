"""Reusable helpers for parsing Mathematica source into SymPy expressions."""

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
    """Structured representation of a Mathematica source fragment."""

    assignments: dict[str, str] = field(default_factory=dict)
    definitions: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_expression: str | None = None

    def resolve_expression(self, name: str | None = None) -> str | None:
        """Return the raw expression associated with a symbol or fallback."""
        if isinstance(name, str):
            return self.assignments.get(name, name)

        if self.last_expression is None:
            return None

        return self.assignments.get(
            self.last_expression,
            self.last_expression,
        )


class MathematicaParser:
    """Parse Mathematica source text and convert symbolic forms to SymPy."""

    def normalize_source(self, text: str) -> str:
        """Normalize raw Mathematica source before structural parsing."""
        text = self._strip_comments(text)
        text = self._replace_box_expressions(text)
        text = re.sub(r"\\\[([A-Za-z]+)\]", lambda match: match.group(1), text)
        return text.strip()

    def _strip_comments(self, text: str) -> str:
        """Strip Mathematica comments, including nested comment blocks."""
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
        """Apply ordered textual identifier replacements to source."""
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
        """Parse Mathematica syntax into a normalized SymPy expression."""
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
        """
        Return the ordered expressions stored inside a Mathematica Module.
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
        """Return True when `text` is a complete Mathematica association."""
        text = text.strip()
        return text.startswith("<|") and text.endswith("|>")

    def find_matching_bracket(self, text: str, start_index: int) -> int:
        """Return the index of the closing bracket matching `start_index`."""
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
        """Split `text` on separators that appear only at top level."""
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
        """Find the first top-level occurrence of `token` in `text`."""
        for index, _char, at_top in self._iter_top_level(text):
            if at_top and text.startswith(token, index):
                return index
        return -1

    def find_top_level_assignment(self, text: str) -> int:
        """Locate a plain top-level assignment operator in Mathematica code."""
        for index, char, at_top in self._iter_top_level(text):
            if at_top and char == "=":
                prev_char = text[index - 1] if index > 0 else ""
                next_char = text[index + 1] if index + 1 < len(text) else ""
                if prev_char not in {":", "-", "<", ">", "="} and next_char != ">":
                    return index
        return -1

    def parse_association(self, text: str) -> dict[str, Any]:
        """Parse a Mathematica association `<| ... |>` into Python data."""
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
        """Parse a metadata value from a Mathematica association."""
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
        """Remove a single surrounding pair of double quotes, if present."""
        if text.startswith('"') and text.endswith('"'):
            return text[1:-1]
        return text

    def _parse_source_items(self, items: list[str]) -> ParsedMathematicaSource:
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
        if (
            isinstance(expr, sp.Symbol)
            and expr.name.startswith("Log")
            and len(expr.name) > 3
        ):
            return sp.log(sp.Symbol(expr.name[3:]))
        return expr

    def _convert_series_data(self, expr):
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
        if point in {sp.Symbol("Infinity"), sp.oo}:
            return variable ** (-exponent)

        if point == 0:
            return variable**exponent
        return (variable - point) ** exponent


__all__ = ["MathematicaParser", "ParsedMathematicaSource"]
