"""Normalize math notation from OCR and language-model output."""

import re

_frac = re.compile(r"(\d?)\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}")
_frac_digits = re.compile(r"\\([dt]?frac)\s*(\d)\s*(\d)")  # \frac12 is \frac{1}{2}
_text_command = re.compile(
    r"\\(?:text|textrm|textbf|textit|mathrm|mathbf|mathit|operatorname)\s*\{([^{}]*)\}")
_escaped = re.compile(r"\\([$&#_{}])")
_spacing = re.compile(r"\\[,;:! ]|\\quad\b|\\qquad\b")
_percent = re.compile(r"\\%")
_sqrt_latex = re.compile(r"\\sqrt\{([^{}]+)\}")
_root_latex = re.compile(r"\\sqrt\[([^\[\]]+)\]\{([^{}]+)\}")
_caret_braces = re.compile(r"\^\{(\d+|[A-Za-z])\}")
_caret_group = re.compile(r"\^\{([^{}]+)\}")
_caret_parens = re.compile(r"\^\((\d+|[A-Za-z])\)")
_times = re.compile(r"\\times(?![a-zA-Z])")
_cdot = re.compile(r"\\cdot(?![a-zA-Z])")
_dots = re.compile(r"\\[lc]?dots(?![a-zA-Z])")
_pmod = re.compile(r"\\pmod\{([^{}]+)\}")
_leq = re.compile(r"\\leq?(?![a-zA-Z])|\u2264")
_geq = re.compile(r"\\geq?(?![a-zA-Z])|\u2265")
_neq = re.compile(r"\\neq?(?![a-zA-Z])|\u2260")
_lt = re.compile(r"\\lt(?![a-zA-Z])")
_gt = re.compile(r"\\gt(?![a-zA-Z])")
_div = re.compile(r"\\div(?![a-zA-Z])")
_delimiter_size = re.compile(r"\\(?:left|right)(?![a-zA-Z])")
_pm = re.compile(r"\\pm(?![a-zA-Z])|\u00b1")
_stray_backslash_word = re.compile(r"\\([a-zA-Z]+)")
_alnum = re.compile(r"[0-9A-Za-z]")
OPERATORS = "×÷<>"


_atom = r"(?:\d+(?:\.\d+)?|[A-Za-z]|\\[A-Za-z]+|[A-Za-z]+\([^()]*\)|\([^()]*\))"
_simple = re.compile(rf"[+-]?{_atom}(?:\^{_atom})?")


def _group(part: str) -> str:
    """Parenthesize a numerator or denominator that is more than one number, letter, or
    power, so a/b keeps its meaning: ``\\frac{1}{2x}`` reads (1)/(2x), not 1/2*x."""
    part = part.strip()
    return part if _simple.fullmatch(part) else f"({part})"


def _fraction(m: re.Match) -> str:
    whole = m.group(1)
    frac = f"{_group(m.group(2))}/{_group(m.group(3))}"
    after = m.string[m.end():m.end() + 1]
    if after == "^" or (after.isalnum() or after == "(") and not whole:
        frac = f"({frac})"  # (a/b)^2 and (1/2)x, not a/b^2 or 1/2x
    if whole:
        return f"{whole} {frac}"
    # keep \frac{1}{2}\frac{3}{4} from fusing into 1/23/4
    return f" {frac}" if m.start() and m.string[m.start() - 1] in "})" else frac


def normalize_math_notation(text: str) -> str:
    """Converts LaTeX and hybrid-ASCII math markup to one plain-text grammar."""
    if not text:
        return text or ""
    out = _frac_digits.sub(r"\\\1{\2}{\3}", text)
    out = _spacing.sub(" ", out)
    # Operator commands go first, while the command after them is still LaTeX: once
    # \frac{c}{d} in \cdot\frac{c}{d} becomes c/d, \cdot no longer ends at a non-letter.
    out = _pmod.sub(lambda m: f"(mod {m.group(1)})", out)
    out = _dots.sub("...", out)
    out = _times.sub("*", out)
    out = _cdot.sub("*", out)
    out = _leq.sub("<=", out)
    out = _geq.sub(">=", out)
    out = _neq.sub("!=", out)
    out = _lt.sub("<", out)
    out = _gt.sub(">", out)
    out = _div.sub("÷", out)
    out = _delimiter_size.sub("", out)
    out = _pm.sub("+/-", out)
    previous = None
    while out != previous:
        previous = out
        out = _text_command.sub(lambda m: m.group(1), out)
        out = _pmod.sub(lambda m: f"(mod {m.group(1)})", out)  # once inner braces are gone
        out = _root_latex.sub(lambda m: f"root({m.group(2)}, {m.group(1)})", out)
        out = _sqrt_latex.sub(lambda m: f"sqrt({m.group(1)})", out)
        out = _caret_braces.sub(lambda m: f"^{m.group(1)}", out)
        out = _caret_group.sub(lambda m: f"^({m.group(1)})", out)
        out = _frac.sub(_fraction, out)
    out = _percent.sub("%", out)
    out = _escaped.sub(r"\1", out)
    out = _caret_parens.sub(lambda m: f"^{m.group(1)}", out)
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

    def strip(s):
        return "".join(ch for ch in s if _alnum.match(ch))

    if strip(merged) != strip(text) or (set(merged) - set(text)) - wanted:
        return text
    return merged
