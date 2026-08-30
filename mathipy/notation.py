"""Normalize math notation from OCR and language-model output.

The same expression arrives as LaTeX, as hybrid ASCII, or with the backslash
stripped. Without normalization a tokenizer counts them as different things.
Fractions stay as ``a/b`` rather than becoming a rendered form.
"""

import re

_frac = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
_sqrt_latex = re.compile(r"\\sqrt\{([^{}]+)\}")
_caret_braces = re.compile(r"\^\{(\w+)\}")
_caret_parens = re.compile(r"\^\((\w+)\)")
_times = re.compile(r"\\times")
_cdot = re.compile(r"\\cdot")
_leq = re.compile(r"\\leq|\u2264")
_geq = re.compile(r"\\geq|\u2265")
_neq = re.compile(r"\\neq|\u2260")
_pm = re.compile(r"\\pm|\u00b1")
_stray_backslash_word = re.compile(r"\\([a-zA-Z]+)")
_alnum = re.compile(r"[0-9A-Za-z]")
OPERATORS = "×÷<>"


def normalize_math_notation(text: str) -> str:
    """Converts LaTeX and hybrid-ASCII math markup to one plain-text grammar."""
    if not text:
        return text or ""
    out = text
    out = _frac.sub(lambda m: f"{m.group(1)}/{m.group(2)}", out)
    out = _sqrt_latex.sub(lambda m: f"sqrt({m.group(1)})", out)
    out = _caret_braces.sub(lambda m: f"^{m.group(1)}", out)
    out = _caret_parens.sub(lambda m: f"^{m.group(1)}", out)
    out = _times.sub("*", out)
    out = _cdot.sub("*", out)
    out = _leq.sub("<=", out)
    out = _geq.sub(">=", out)
    out = _neq.sub("!=", out)
    out = _pm.sub("+/-", out)
    out = _stray_backslash_word.sub(lambda m: m.group(1), out)
    return out


def insert_operators(text: str, reference: str, operators: str = OPERATORS) -> str:
    """Inserts operators the reference carries and the text lacks."""
    wanted = set(operators)
    here = [i for i, ch in enumerate(text) if _alnum.match(ch)]
    there = [i for i, ch in enumerate(reference) if _alnum.match(ch)]
    if not here or len(here) != len(there):
        return text
    if [text[i] for i in here] != [reference[i] for i in there]:
        return text

    out, at_here, at_there = [], 0, 0
    for k in range(len(here) + 1):
        stop_here = here[k] if k < len(here) else len(text)
        stop_there = there[k] if k < len(there) else len(reference)
        gap, other = text[at_here:stop_here], reference[at_there:stop_there]
        if (wanted & (set(other) - set(gap))) and "\n" not in gap:
            out.append(re.sub(r"\s+", "", other))
        else:
            out.append(gap)
        if k < len(here):
            out.append(text[stop_here])
            at_here, at_there = stop_here + 1, stop_there + 1

    merged = "".join(out)
    strip = lambda s: "".join(ch for ch in s if _alnum.match(ch))
    if strip(merged) != strip(text) or (set(merged) - set(text)) - wanted:
        return text
    return merged
