"""References from item text into the image.

A deictic reference ("shown above") tells a reader to look at the figure; a
label reference ("angle A") names an element inside it. Both mark text that
cannot be read on its own, which is a different demand from text that merely
mentions a picture.
"""

import re
from collections import Counter

_display_nouns = ("figure", "graph", "diagram", "table", "chart", "picture",
                  "drawing", "model", "number line", "grid", "spinner", "map",
                  "shape", "solid", "box", "image", "photograph")

deictic_phrases = [
    "shown", "above", "below", "pictured", "drawn", "plotted", "graphed",
    "shaded", "this figure", "this graph", "this diagram", "this table",
    "in the figure", "in the graph", "in the diagram", "in the box",
    "on the grid",
] + [f"the {n}" for n in _display_nouns]

_following_display = re.compile(
    r"\bfollowing\s+(?:" + "|".join(_display_nouns) + r")s?\b", re.I)

_label_elements = ["point", "angle", "side", "triangle", "quadrilateral",
                   "line", "segment", "vertex", "vertices", "circle", "arc", "ray",
                   "square", "rectangle", "polygon", "face", "edge", "base",
                   "radius", "diameter", "chord", "diagonal"]

_display_parts = ["axis", "axes", "gridline", "tick mark", "legend",
                  "bar", "slice", "sector", "wedge", "cell", "data point",
                  "curve", "plotted point", "coordinate", "interval"]

_marked_elements = ["shaded", "unshaded", "circled", "marked", "labeled",
                    "labelled", "highlighted", "dotted", "starred"]

_not_a_series = ("answer", "question", "step", "figure", "table", "graph",
                 "chart", "part", "page", "item", "grade", "section", "example",
                 "add", "find", "subtract", "multiply", "divide", "use", "draw",
                 "write", "explain", "show", "circle", "solve", "estimate",
                 "compare", "round", "count", "give", "list", "name", "place",
                 "select", "choose", "mark", "label", "complete", "there", "each",
                 "what", "which", "how", "the", "this", "that", "these", "those")

_label_ref = re.compile(
    r"\b(?i:" + "|".join(_label_elements) + r")\s+[A-Z]{1,4}\b"
    r"|\b(?i:the|each|every|this|that)\s+(?:\w+\s+){0,2}(?i:"
    + "|".join(_display_parts) + r")\b"
    r"|\b(?i:" + "|".join(_marked_elements) + r")\b"
    r"|\b(?!(?i:" + "|".join(_not_a_series) + r")\b)[A-Z][a-z]{2,}\s+(?:[A-Z]\b|\d\b)")


def deictic_features(text: str) -> dict[str, int]:
    """Counts of linguistic pointers to the image."""
    lowered = (text or "").lower()
    found = Counter()
    for phrase in deictic_phrases:
        hits = len(re.findall(r"\b" + re.escape(phrase) + r"\b", lowered))
        if hits:
            found[phrase] = hits
    following = len(_following_display.findall(lowered))
    if following:
        found["following <display>"] = following
    return {"deictic_count": sum(found.values()), "deictic_unique": len(found)}


def label_features(text: str) -> dict[str, int]:
    """Counts of references to a specific labeled element inside the image."""
    return {"label_count": len(_label_ref.findall(text or ""))}


def crossmodal_features(text: str) -> dict[str, int]:
    """Every cross-modal measure for one item."""
    return {**deictic_features(text), **label_features(text)}
