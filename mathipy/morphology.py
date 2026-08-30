"""Morphological structure of mathematics vocabulary.

Nominalization, Greek and Latin roots, and derivational suffixes carry a
distinct load from word length: a nominalized process is harder to parse than
the verb it came from, and a technical suffix marks a term as domain-specific.
"""

import re

numeric_prefix_words = {
    "unit": 1, "unique": 1, "unify": 1, "unicycle": 1,
    "monomial": 1, "monotonic": 1,
    "binary": 2, "bisect": 2, "bisector": 2, "bimodal": 2, "binomial": 2,
    "biweekly": 2, "bicycle": 2,
    "triangle": 3, "triangular": 3, "trisect": 3, "trinomial": 3,
    "triple": 3, "tripled": 3, "tripling": 3, "trio": 3, "tricycle": 3,
    "quadrilateral": 4, "quadrant": 4, "quadruple": 4, "quadratic": 4,
    "quadrangle": 4, "tetrahedron": 4,
    "pentagon": 5, "pentagonal": 5, "pentomino": 5,
    "hexagon": 6, "hexagonal": 6, "hexahedron": 6,
    "heptagon": 7, "heptagonal": 7,
    "octagon": 8, "octagonal": 8, "octahedron": 8, "octant": 8,
    "nonagon": 9,
    "decagon": 10, "decimal": 10, "decade": 10, "decimeter": 10,
    "dodecagon": 12, "dozen": 12,
    "century": 100, "centimeter": 100, "percent": 100, "percentage": 100,
    "millennium": 1000, "millimeter": 1000, "kilometer": 1000, "kilogram": 1000,
}

metric_prefix_words = {
    "millimeter": 0.001, "milliliter": 0.001, "milligram": 0.001,
    "centimeter": 0.01, "centiliter": 0.01,
    "decimeter": 0.1, "deciliter": 0.1,
    "decameter": 10.0, "hectometer": 100.0,
    "kilometer": 1000.0, "kilogram": 1000.0, "kiloliter": 1000.0,
}

_geometric_suffix = re.compile(
    r"\b\w{3,}(?:gons?|gonal|hedrons?|hedra|laterals?)\b", re.I)

_nominalization = re.compile(r"\b\w{4,}(?:tions?|sions?|ments?|ities|ity|ness)\b", re.I)
_not_nominalizations = {
    "nation", "nations", "million", "millions", "billion", "billions",
    "lion", "lions", "onion", "onions", "station", "stations",
    "cushion", "cushions", "fashion", "fashions", "mention", "mentions",
    "city", "cities", "quantity", "quantities", "entity", "entities",
    "moment", "moments", "element", "elements", "comment", "comments",
    "witness", "business",
}

_word = re.compile(r"[a-z]+", re.I)


def morphology_features(text: str) -> dict[str, float]:
    """Counts of meaning-bearing morphemes below the word."""
    text = text or ""
    words = [w.lower() for w in _word.findall(text)]

    def _lookup(word, table):
        if word in table:
            return table[word]
        if word.endswith("s") and word[:-1] in table:
            return table[word[:-1]]
        return None

    numeric = [v for w in words if (v := _lookup(w, numeric_prefix_words)) is not None]
    metric = [v for w in words if (v := _lookup(w, metric_prefix_words)) is not None]
    geometric = _geometric_suffix.findall(text)
    nominal = [m for m in _nominalization.findall(text)
               if m.lower() not in _not_nominalizations]

    return {
        "morph_nominalization_count": len(nominal),
        "morph_nominalization_per_100w": round(100 * len(nominal) / len(words), 3)
        if words else 0.0,
        "morph_numeric_prefix_count": len(numeric),
        "morph_numeric_prefix_max": max(numeric, default=0),
        "morph_metric_prefix_count": len(metric),
        "morph_geometric_suffix_count": len(geometric),
    }
