"""Relations the wording of an item encodes."""

import re
from collections import Counter

from mathipy.utils import consume_phrases

relational_terms: dict[str, list[str]] = {

    "rate": ["per", "for every", "for each", "apiece"],

}

number_words = ["zero", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten", "eleven", "twelve", "twenty", "thirty",
                "forty", "fifty", "hundred", "thousand"]

ordinal_words = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh",
                 "eighth", "ninth", "tenth", "eleventh", "twelfth", "twentieth",
                 "last", "next"]

fraction_words = ["half", "halves", "third", "fourth", "quarter", "fifth",
                  "sixth", "seventh", "eighth", "ninth", "tenth", "twelfth", "hundredth"]

polysemous_words = ["table", "mean", "product", "power", "base", "root", "plane", "right",
            "odd", "even", "volume", "face", "degree", "times", "difference",
            "expression", "operation", "rational", "irrational", "natural", "real",
            "positive", "negative", "order", "range", "mode", "factor", "term",
            "prime", "point", "line", "set", "value", "figure", "solution"]

_comparatives = ["less", "fewer", "more", "greater", "smaller", "larger",
                 "shorter", "longer", "taller", "heavier", "lighter",
                 "faster", "slower", "older", "younger"]

_comparison = re.compile(
    r"\b(?:" + "|".join(_comparatives) + r")\b(?:\s+\w+){0,3}?\s+than\b", re.I)

_comparison_question = re.compile(
    r"\bhow\s+(?:many|much)\s+(?:" + "|".join(_comparatives) + r")\b", re.I)

_multiplicative = re.compile(
    r"\b(?:\d+(?:\.\d+)?|" + "|".join(number_words) + r"|several|many)\s+times\b"
    r"|\btimes\s+as\s+\w+\b"
    r"|\btimes\s+(?:larger|greater|smaller|longer|shorter|more|less)\b"
    r"|\b(?:twice|double|triple|quadruple|half)\s+(?:as|the|that|of)\b", re.I)

_division = re.compile(
    r"\bdivid(?:e|es|ed|ing|ing)\b|\bdivision\b|\bquotient\b"
    r"|\b(?:shared?|split|distribute[ds]?)\b(?:\s+\w+){0,4}?\s+"
    r"(?:equally|evenly|among|between|into)\b"
    r"|\bhow many (?:groups|sets|bags|boxes|piles)\b"
    r"|\beach\s+\w+\s+(?:gets|receives|has)\b", re.I)

_quantity_before = re.compile(
    r"\b(?:\d+(?:\.\d+)?|" + "|".join(number_words) + r")\b"
    r"(?:\s+\w+){0,2}?\s+"
    r"(?:" + "|".join(_comparatives) + r")\b(?:\s+\w+){0,3}?\s+than\b", re.I)
_part_words = ["fraction", "percent", "percentage", "part", "portion"] + fraction_words
_partitive = re.compile(
    r"\b(?:" + "|".join(_part_words) + r")s?\s+of\b"
    r"|(?:%|\b\d+/\d+)\s+of\b"
    r"|\bout\s+of\b"
    r"|\b(?:\d+|" + "|".join(number_words) + r")\s+of\s+(?:the|these|those)\b(?!\s+following)",
    re.I)
_distribution = re.compile(
    r"(?<!\bfor )\b(?:each|every)\s+(?:\w+\s+){0,4}?"
    r"(?:has|have|gets|costs?|holds?|contains?|weighs?|receives?|is|are)\b"
    r"|\bin\s+(?:each|every)\b", re.I)
_time_second = re.compile(
    r"\b(?:\d+(?:\.\d+)?|per|" + "|".join(number_words) + r")\s+second\b|\bsecond\s+per\b")
_ordinal_suffix = re.compile(r"\b\d+(?:st|nd|rd|th)\b", re.I)
_slash_fraction = re.compile(r"\b\d+\s*/\s*\d+\b")
_nominal = re.compile(r"\b(?:number|room|page|problem|item|question|bus|route|"
                      r"channel|line)\s+\d+\b", re.I)
_integer = re.compile(r"\b\d{1,2}:\d{2}\b|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b")
_word_fraction = re.compile(
    r"\b(?:" + "|".join(number_words) + r")[\s-]+(?:"
    + "|".join(sorted(fraction_words, key=len, reverse=True)) + r")s?\b")


def _count(text: str, phrases: list[str]) -> int:
    lowered = text.lower()
    return sum(len(re.findall(r"\b" + re.escape(p) + r"\b", lowered)) for p in phrases)


def relational_features(text: str) -> dict[str, int]:
    """Counts of the relations the wording encodes, and of the marked comparative,
    whose surface order reverses the order the operation needs."""
    text = text or ""
    hits = consume_phrases(text, [t for terms in relational_terms.values() for t in terms])
    counts = {f"rel_{name}": sum(hits.get(t, 0) for t in terms)
              for name, terms in relational_terms.items()}
    counts["rel_partitive"] = len(_partitive.findall(text))
    counts["rel_distribution"] = len(_distribution.findall(text))
    counts["rel_comparison"] = (len(_comparison.findall(text))
                                + len(_comparison_question.findall(text)))
    counts["rel_multiplicative"] = len(_multiplicative.findall(text))
    counts["rel_division"] = len(_division.findall(text))
    counts["rel_order_reversed"] = len(_quantity_before.findall(text))
    counts["rel_total"] = sum(v for k, v in counts.items() if k != "rel_order_reversed")
    return counts


def number_features(text: str) -> dict[str, int]:
    """Numbers split by what they do: count, index, name a part, or identify (Woodin et al., 2024)."""
    text = text or ""
    lowered = text.lower()

    nominal = len(_nominal.findall(text))
    ordinal = (len(_ordinal_suffix.findall(text))
               + _count(_time_second.sub(" | ", _word_fraction.sub(" | ", lowered)), ordinal_words))
    fraction = len(_slash_fraction.findall(text)) + len(_word_fraction.findall(lowered))

    digits = [m.group(0) for m in _integer.finditer(text)]
    consumed = nominal + len(_slash_fraction.findall(text)) * 2
    cardinal = max(0, len(digits) - consumed)

    values = [float(d.replace(",", "")) for d in digits if "." not in d and ":" not in d]
    return {"num_cardinal": cardinal,
            "num_ordinal": ordinal,
            "num_fraction": fraction,
            "num_nominal": nominal,
            "num_round": sum(1 for v in values if v and (v % 10 == 0 or v % 25 == 0)),
            "num_total": cardinal + ordinal + fraction + nominal}


def polysemy_features(text: str) -> dict[str, int]:
    """Everyday words carrying a distinct mathematical sense."""
    lowered = (text or "").lower()
    found = Counter()
    for word in polysemous_words:
        hits = len(re.findall(r"\b" + word + r"s?\b", lowered))
        if hits:
            found[word] = hits
    return {"polysemy_count": sum(found.values()), "polysemy_unique": len(found)}


def register_features(text: str) -> dict[str, int]:
    """Every register measure for one item."""
    return {**relational_features(text), **number_features(text),
            **polysemy_features(text)}
