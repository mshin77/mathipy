"""Mathematical content analysis and domain classification."""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

from mathipy.cognitive_load import inflections, plural
from mathipy.utils import consume_phrases, extract_numbers, extract_variables, strip_option_labels

logger = logging.getLogger(__name__)


_unit_abbreviations = {
    "km": "kilometer", "cm": "centimeter", "mm": "millimeter", "m": "meter",
    "kg": "kilogram", "mg": "milligram", "g": "gram",
    "lb": "pound", "lbs": "pound", "oz": "ounce",
    "ml": "milliliter", "l": "liter",
    "hr": "hour", "hrs": "hour", "min": "minute", "sec": "second",
    "ft": "foot", "in": "inch", "yd": "yard", "mi": "mile",
    "sq": "square", "cu": "cubic",
}

_unit_abbrev_re = re.compile(
    r"(?<![A-Za-z])(?<=\d)\s?(" + "|".join(sorted(set(_unit_abbreviations) - {"in"}, key=len, reverse=True))
    + r")(?![A-Za-z])", re.IGNORECASE)
_inch_follow = r"\.|\s*(?:by\b|x\b|×|[^\w\s]|$)"
_inch_re = re.compile(rf"(?<=\d)\s?in\b(?={_inch_follow})", re.IGNORECASE)
_inch_pair_re = re.compile(
    rf"(?<=\d)\s?in\.?\s*(?:by\b|x\b|×)\s*\d+(?:\.\d+)?\s?in\b(?!{_inch_follow})", re.IGNORECASE)


_decimal_notation = re.compile(r"(?<![\w.])\d*\.\d+")
_verb_terms = {"add", "subtract", "multiply", "divide", "solve", "simplify", "factor",
               "measure", "convert", "differentiate", "integrate", "graph"}
_no_plural = {"mean", "data", "per", "for every", "elapsed"}
_mean_verb = re.compile(r"\b(?:do|does|did|will|would|could|can|might|may)"
                        r"(?:\s+(?:this|that|it|these|those|he|she|they))?\s+mean\b")
_even_adverb = re.compile(r"(?<!\bis )(?<!\bare )(?<!\bbe )\beven\b(?!\s+(?:numbers?|integers?)\b)")
_count_words = r"\d[\d,.]*|per|many|few|several|one|two|three|four|five|six|seven|eight|nine|ten|twelve"
_time_word = re.compile(r"(\b(?:" + _count_words + r")\s+)?\b(?:second|day|week|month|year)s?\b")
_clock = r"(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?"
# A clock time needs a time cue: "at/by/until 10:30", a clock that "shows" or "reads"
# it, "9:15 to 10:00", "between 3:15 and 4:00", or am/pm; "about 1:20" and
# "from 4:12" stay ratios.
_clock_time = re.compile(
    rf"\b(?:at|by|until|till|before|after|shows?|showed|reads?|time\s+is)\s+{_clock}"
    rf"(?:\s*(?:to|until|and|-|–)\s*{_clock})?\b"
    rf"|\b{_clock}\s*(?:to|until|-|–)\s*{_clock}\b"
    rf"|\bbetween\s+{_clock}\s+and\s+{_clock}\b"
    rf"|\b{_clock}\s*(?:[ap]\.?m\b\.?|o'clock)", re.I)
# A pair named as a scale or ratio right beside it, on the same line, stays a ratio
# even after a time cue: "built at 1:24 scale", "The map scale shows 1:50", "the ratio
# 10:15 to 2:30". A ratio word before "at/by/until..." does not count ("read the scale
# at 10:30" is a time), and a pair followed by am/pm or o'clock is always a time.
_ratio_word = r"(?:scale|ratios?|proportions?|odds)"
_ratio_before = re.compile(rf"\b{_ratio_word}[ \t]+(?:(?:is|was|of|shows?|reads?)[ \t]+)?$",
                           re.I)
_ratio_after = re.compile(rf"[ \t]*{_ratio_word}\b", re.I)
_am_pm = re.compile(r"[ap]\.?m\b|o'clock", re.I)
_am_pm_after = re.compile(r"[ \t]*(?:[ap]\.?m\b|o'clock)", re.I)
_time_preposition = re.compile(r"(?:at|by|until|till|before|after|between)\b", re.I)


def _strip_clock_times(text: str) -> str:
    def strip(m):
        if _am_pm.search(m.group(0)) or _am_pm_after.match(text, m.end()):
            return " "
        named_after = _ratio_after.match(text, m.end())
        named_before = (not _time_preposition.match(m.group(0))
                        and _ratio_before.search(text, 0, m.start()))
        return m.group(0) if named_after or named_before else " "
    return _clock_time.sub(strip, text)


def _drop_everyday_senses(text: str) -> str:
    """Lowercased text with mean as a verb, adverbial even, and uncounted time words blanked."""
    text = _mean_verb.sub(" | ", text)
    text = _even_adverb.sub(" | ", text)
    return _time_word.sub(lambda m: m.group(0) if m.group(1) else " | ", text)


def _match_unit_abbreviations(text: str) -> dict[str, int]:
    """Counts of measurement units written as abbreviations after a number."""
    counts: dict[str, int] = {}
    for abbrev in _unit_abbrev_re.findall(text):
        unit = _unit_abbreviations[abbrev.lower()]
        counts[unit] = counts.get(unit, 0) + 1
    inches = len(_inch_re.findall(text)) + len(_inch_pair_re.findall(text))
    counts.update({"inch": inches} if inches else {})
    return counts


class MathContentAnalyzer:
    """Analyze math content and classify by Common Core State Standards domain."""

    def __init__(self):
        self._init_patterns()
        self._init_vocabulary()

    def _init_patterns(self):
        self.patterns = {
            "addition": re.compile(r"\d+\s*\+\s*\d+"),
            "subtraction": re.compile(r"\d+\s*[-−]\s*\d+"),
            "multiplication": re.compile(r"\d+\s*[×*·]\s*\d+"),
            "division": re.compile(r"\d+\s*÷\s*\d+"),
            "variable": re.compile(r"(?<![A-Za-z'’_])[a-zA-Z](?![A-Za-z'’_])"),
            "equation": re.compile(r"(?<![<>!=])=(?!=)"),
            "inequality": re.compile(r"\S+\s*[<>≤≥]=?\s*\S+"),
            "exponent": re.compile(r"(?:\w+|\))\^[\w\d{}]+|\w+\*\*[\w\d{}]+"),
            "function": re.compile(r"\b[a-zA-Z]+\([^)]+\)"),
            "polynomial": re.compile(
                r"(?<![A-Za-z])[a-z]\^?\d*\s*[+\-−]\s*\d*[a-z]\^?\d*(?![A-Za-z])"),
            "fraction": re.compile(r"\d+/\d+|\\frac\{\d+\}\{\d+\}"),
            "decimal": re.compile(r"\d+\.\d+"),
            "percentage": re.compile(r"\d+\.?\d*\s*(?:%|percent\b)", re.I),
            "ratio": re.compile(
                r"\d+\s*:\s*\d+"
                r"|\bratios?\s+of\s+\d+\s+to\s+\d+"
                r"|\b\d+\s+to\s+\d+\s+ratio"
                r"|\bfor\s+every\s+\d+", re.I),
            "scientific_notation": re.compile(r"\d+\.?\d*\s*[×x]\s*10\^[-−]?\d+"),
            "derivative": re.compile(r"d/dx|(?<![A-Za-z])[fgh]['′](?![a-z])|\\frac\{d\}\{dx\}"),
            "integral": re.compile(r"∫|\\int"),
            "limit": re.compile(r"\\lim|lim_"),
            "summation": re.compile(r"∑|\\sum"),
        }

        self.symbols = {
            "+": "addition", "-": "subtraction", "−": "subtraction", "×": "multiplication",
            "*": "multiplication", "·": "multiplication", "÷": "division",
            "/": "division", "=": "equals", "<": "less_than",
            ">": "greater_than", "≤": "less_equal", "≥": "greater_equal",
            "≠": "not_equal", "≈": "approximately", "√": "square_root",
            "∑": "summation", "∫": "integral", "π": "pi", "∞": "infinity",
            "(": "grouping", ")": "grouping",
            "[": "grouping", "]": "grouping",
            "{": "grouping", "}": "grouping",
            "^": "exponent", "%": "percent", "°": "degree",
            "±": "plus_minus", "∠": "angle", "⊥": "perpendicular",
            "∥": "parallel", "≅": "congruent", "△": "triangle",
        }

    def _init_vocabulary(self):
        self.domains = {
            "arithmetic": {
                "add", "subtract", "multiply", "divide", "sum", "difference",
                "product", "quotient", "remainder", "factor", "multiple",
                "even", "odd", "prime", "composite", "digit", "place value",
            },
            "algebra": {
                "variable", "coefficient", "term", "expression", "equation",
                "inequality", "solve", "simplify", "factor", "polynomial",
                "linear", "quadratic", "function", "slope", "intercept",
            },
            "geometry": {
                "point", "line", "ray", "segment", "angle", "triangle",
                "rectangle", "square", "circle", "polygon", "area",
                "perimeter", "volume", "parallel", "perpendicular", "congruent",
                "symmetry", "symmetric", "line of symmetry", "transformation",
                "rotation", "reflection", "translation", "dilation",
                "similar figure", "similar triangle", "scale drawing",
            },
            "statistics": {
                "mean", "median", "mode", "range", "data", "graph", "chart",
                "probability", "outcome", "sample", "population", "distribution",
                "standard deviation", "variance", "correlation",
            },
            "calculus": {
                "derivative", "integral", "differentiate", "integrate",
                "antiderivative", "asymptote", "tangent line",
                "instantaneous rate", "optimization", "convergence",
                "diverges", "converges", "limit of the", "approaches infinity",
            },
            "fractions": {
                "fraction", "numerator", "denominator", "mixed number",
                "improper", "equivalent", "simplify", "common denominator",
                "decimal", "percent", "ratio", "proportion",
            },
            "measurement": {
                "measure", "capacity", "weight", "mass", "temperature",
                "unit", "convert", "conversion", "equivalent measure",
                "inch", "foot", "feet", "yard", "mile",
                "centimeter", "meter", "kilometer", "millimeter",
                "gram", "kilogram", "liter", "milliliter",
                "ounce", "pound", "ton", "cup", "pint", "quart", "gallon",
                "second", "minute", "hour", "day", "week", "month", "year",
                "degree", "elapsed", "ruler", "clock", "thermometer",
            },
            "ratio_proportion": {
                "ratio", "rate", "unit rate", "proportion", "proportional",
                "percent", "percentage", "scale factor", "equivalent ratio",
                "per", "for every",
            },
        }

        self.all_terms: set[str] = set()
        for terms in self.domains.values():
            self.all_terms.update(terms)
        self._form_to_term = {
            form: term
            for term in sorted(self.all_terms)
            for form in (inflections(term) if term in _verb_terms
                         else {term} if term in _no_plural else {term, plural(term)})
        }

    def analyze(self, text: str) -> dict[str, Any]:
        """Analyze math content in the given text.

        Args:
            text: Input text to analyze.

        Returns:
            Dictionary with ``pattern_matches``, ``symbol_counts``, ``numbers``,
            ``vocabulary``, ``domain_classification``, and ``math_density``."""
        if not text or not text.strip():
            return self._empty_analysis()

        text_lower = text.lower()
        content = strip_option_labels(text)
        pattern_matches = self._match_patterns(content)
        symbol_counts = self._count_symbols(text)
        numbers = extract_numbers(text)
        term_matches = self._match_vocabulary(text_lower)
        domain = self._classify_domain(text_lower, pattern_matches, term_matches)

        word_count = len(content.split())
        return {
            "pattern_matches": pattern_matches,
            "symbol_counts": symbol_counts,
            "total_math_symbols": sum(symbol_counts.values()),
            "unique_symbol_types": len(symbol_counts),
            "numbers": {
                "count": len(numbers),
                "values": numbers[:20],
                "range": max(numbers) - min(numbers) if numbers else 0,
                "has_negative": any(n < 0 for n in numbers),
                "has_decimal": bool(_decimal_notation.search(text)),
            },
            "vocabulary": {
                "math_terms": sorted(term_matches),
                "term_count": sum(term_matches.values()),
                "unique_terms": len(term_matches),
            },
            "domain_classification": domain,
            "math_density": sum(pattern_matches.values()) / word_count if word_count else 0,
        }

    def _match_patterns(self, text: str) -> dict[str, int]:
        untimed = _strip_clock_times(text)  # "at 10:30" is a time, not a ratio
        counts = {n: len(m) for n, p in self.patterns.items()
                  if n != "variable" and (m := p.findall(untimed if n == "ratio" else text))}
        variables = set(extract_variables(text))
        hits = sum(1 for v in self.patterns["variable"].findall(text) if v in variables)
        return {**counts, **({"variable": hits} if hits else {})}

    def _count_symbols(self, text: str) -> dict[str, int]:
        text = text.replace("<=", "≤").replace(">=", "≥").replace("!=", "≠")
        counts = Counter(self.symbols[c] for c in text if c in self.symbols)
        return dict(counts)

    def _match_vocabulary(self, text: str) -> dict[str, int]:
        matched: dict[str, int] = {}
        for form, hits in consume_phrases(_drop_everyday_senses(text), self._form_to_term).items():
            matched[self._form_to_term[form]] = matched.get(self._form_to_term[form], 0) + hits
        for unit, count in _match_unit_abbreviations(text).items():
            matched[unit] = matched.get(unit, 0) + count
        return matched

    def _classify_domain(
        self,
        text: str,
        patterns: dict[str, int],
        terms: dict[str, int],
    ) -> dict[str, Any]:
        domain_scores: dict[str, float] = {
            domain: sum(terms[t] for t in vocab if t in terms)
            for domain, vocab in self.domains.items()
        }

        if patterns.get("derivative") or patterns.get("integral"):
            domain_scores["calculus"] = domain_scores.get("calculus", 0) + 1

        if patterns.get("fraction"):
            domain_scores["fractions"] = domain_scores.get("fractions", 0) + 1

        if patterns.get("equation") or patterns.get("variable"):
            domain_scores["algebra"] = domain_scores.get("algebra", 0) + 1

        total = sum(domain_scores.values())
        primary = max(domain_scores, key=domain_scores.get) if total else "unknown"

        return {
            "primary": primary,
            "confidence": domain_scores.get(primary, 0) / total if total else 0,
            "scores": domain_scores,
            "secondary": sorted(
                domain_scores.keys(),
                key=lambda k: domain_scores[k],
                reverse=True,
            )[1:3] if total else [],
        }

    def _empty_analysis(self) -> dict[str, Any]:
        return {
            "pattern_matches": {},
            "symbol_counts": {},
            "total_math_symbols": 0,
            "unique_symbol_types": 0,
            "numbers": {
                "count": 0, "values": [], "range": 0,
                "has_negative": False, "has_decimal": False,
            },
            "vocabulary": {"math_terms": [], "term_count": 0, "unique_terms": 0},
            "domain_classification": {
                "primary": "unknown", "confidence": 0,
                "scores": dict.fromkeys(self.domains, 0), "secondary": [],
            },
            "math_density": 0,
        }
