"""Mathematical content analysis and domain classification."""

import re
import logging
from collections import Counter
from typing import Any, Dict, List, Set, Union

from mathipy.utils import extract_numbers

logger = logging.getLogger(__name__)


class MathContentAnalyzer:
    """Analyze math content and classify by Common Core State Standards domain.

    Detects math patterns (equations, fractions, operations), counts symbols,
    extracts numbers and vocabulary, and classifies the primary math domain
    (arithmetic, algebra, geometry, statistics, calculus, fractions).
    """

    def __init__(self):
        self._init_patterns()
        self._init_vocabulary()

    def _init_patterns(self):
        self.patterns = {
            "addition": re.compile(r"\d+\s*\+\s*\d+"),
            "subtraction": re.compile(r"\d+\s*[-−]\s*\d+"),
            "multiplication": re.compile(r"\d+\s*[×*·]\s*\d+"),
            "division": re.compile(r"\d+\s*[÷/]\s*\d+"),
            "variable": re.compile(r"\b[a-zA-Z]\b(?!\w)"),
            "equation": re.compile(r"[^=]+=\s*[^=]+"),
            "inequality": re.compile(r"[^<>=]+\s*[<>≤≥]\s*[^<>=]+"),
            "exponent": re.compile(r"\w+\^[\w\d{}]+|\w+\*\*[\w\d{}]+"),
            "function": re.compile(r"\b[a-zA-Z]+\([^)]+\)"),
            "polynomial": re.compile(r"[a-z]\^?\d*\s*[+\-]\s*[a-z]\^?\d*"),
            "fraction": re.compile(r"\d+/\d+|\\frac\{\d+\}\{\d+\}"),
            "decimal": re.compile(r"\d+\.\d+"),
            "percentage": re.compile(r"\d+\.?\d*%"),
            "ratio": re.compile(r"\d+:\d+"),
            "scientific_notation": re.compile(r"\d+\.?\d*\s*[×x]\s*10\^[-]?\d+"),
            "derivative": re.compile(r"d/dx|f'|\\frac\{d\}\{dx\}"),
            "integral": re.compile(r"∫|\\int"),
            "limit": re.compile(r"\\lim|lim_"),
            "summation": re.compile(r"∑|\\sum"),
        }

        self.symbols = {
            "+": "addition", "-": "subtraction", "×": "multiplication",
            "*": "multiplication", "·": "multiplication", "÷": "division",
            "/": "division", "=": "equals", "<": "less_than",
            ">": "greater_than", "≤": "less_equal", "≥": "greater_equal",
            "≠": "not_equal", "≈": "approximately", "√": "square_root",
            "∑": "summation", "∫": "integral", "π": "pi", "∞": "infinity",
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
            },
            "statistics": {
                "mean", "median", "mode", "range", "data", "graph", "chart",
                "probability", "outcome", "sample", "population", "distribution",
                "standard deviation", "variance", "correlation",
            },
            "calculus": {
                "limit", "derivative", "integral", "differentiate", "integrate",
                "continuous", "rate of change", "maximum", "minimum",
                "optimization", "series", "convergence",
            },
            "fractions": {
                "fraction", "numerator", "denominator", "mixed number",
                "improper", "equivalent", "simplify", "common denominator",
                "decimal", "percent", "ratio", "proportion",
            },
        }

        self.all_terms: Set[str] = set()
        for terms in self.domains.values():
            self.all_terms.update(terms)

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze math content in the given text.

        Args:
            text: Input text to analyze.

        Returns:
            Dictionary with ``pattern_matches``, ``symbol_counts``, ``numbers``,
            ``vocabulary``, ``domain_classification``, and ``math_density``.
        """
        if not text or not text.strip():
            return self._empty_analysis()

        text_lower = text.lower()
        pattern_matches = self._match_patterns(text)
        symbol_counts = self._count_symbols(text)
        numbers = extract_numbers(text)
        term_matches = self._match_vocabulary(text_lower)
        domain = self._classify_domain(text_lower, pattern_matches, term_matches)

        word_count = len(text.split())
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
                "has_decimal": any(isinstance(n, float) and n != int(n) for n in numbers),
            },
            "vocabulary": {
                "math_terms": list(term_matches.keys()),
                "term_count": sum(term_matches.values()),
                "unique_terms": len(term_matches),
            },
            "domain_classification": domain,
            "math_density": sum(pattern_matches.values()) / word_count if word_count else 0,
        }

    def _match_patterns(self, text: str) -> Dict[str, int]:
        matches = {}
        for name, pattern in self.patterns.items():
            found = pattern.findall(text)
            if found:
                matches[name] = len(found)
        return matches

    def _count_symbols(self, text: str) -> Dict[str, int]:
        counts = Counter()
        for char in text:
            if char in self.symbols:
                counts[self.symbols[char]] += 1
        return dict(counts)

    def _match_vocabulary(self, text: str) -> Dict[str, int]:
        matches = {}
        for term in self.all_terms:
            count = text.count(term)
            if count > 0:
                matches[term] = count
        return matches

    def _classify_domain(
        self,
        text: str,
        patterns: Dict[str, int],
        terms: Dict[str, int],
    ) -> Dict[str, Any]:
        domain_scores: Dict[str, float] = {}

        for domain, vocab in self.domains.items():
            score = 0
            for term in vocab:
                if term in terms:
                    score += terms[term]
            domain_scores[domain] = score

        if patterns.get("derivative") or patterns.get("integral"):
            domain_scores["calculus"] = domain_scores.get("calculus", 0) + 5

        if patterns.get("fraction"):
            domain_scores["fractions"] = domain_scores.get("fractions", 0) + 3

        if patterns.get("equation") or patterns.get("variable"):
            domain_scores["algebra"] = domain_scores.get("algebra", 0) + 2

        primary = max(domain_scores, key=domain_scores.get) if domain_scores else "unknown"
        total = sum(domain_scores.values()) or 1

        return {
            "primary": primary,
            "confidence": domain_scores.get(primary, 0) / total,
            "scores": domain_scores,
            "secondary": sorted(
                domain_scores.keys(),
                key=lambda k: domain_scores[k],
                reverse=True,
            )[1:3] if len(domain_scores) > 1 else [],
        }

    def _empty_analysis(self) -> Dict[str, Any]:
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
                "primary": "unknown", "confidence": 0, "scores": {}, "secondary": [],
            },
            "math_density": 0,
        }
