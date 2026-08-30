"""Replace identifying information in transcripts and written responses."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

patterns = {
    "email": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
    "phone": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "url": r"\bhttps?://\S+|\bwww\.\S+",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "student_id": r"\b(?:id|student\s*(?:id|number))\s*[:#]?\s*\d{4,}\b",
    "date": r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
}

_titles = r"(?:Mr|Mrs|Ms|Miss|Dr|Prof|Professor|Coach|Principal)"
_title_name = rf"\b{_titles}\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"

_greetings = r"(?:Hi|Hello|Hey|Welcome|Thanks|Thank you|Good job|Well done|Great job)"
_vocative = rf"\b{_greetings},?\s+([A-Z][a-z]{{2,}})\b"
_vocative_skip = {"Thanks", "Thank", "Hello", "Hi", "Hey", "Great", "Good", "Well",
                  "Yes", "Okay", "Sure", "Right", "Correct", "Nice", "Perfect",
                  "Awesome", "Excellent", "You", "Your", "That", "This", "What",
                  "How", "Why", "Can", "Could", "Would", "Let", "Now", "So"}

replacements = {
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "url": "[URL]",
    "ssn": "[SSN]",
    "student_id": "[ID]",
    "date": "[DATE]",
    "name": "[NAME]",
}


def _mask_names(text: str, names: Sequence[str]) -> tuple[str, int]:
    count = 0
    for name in sorted(set(names), key=len, reverse=True):
        if not name.strip():
            continue
        pattern = re.compile(rf"\b{re.escape(name.strip())}\b", re.IGNORECASE)
        text, n = pattern.subn(replacements["name"], text)
        count += n
    return text, count


def _vocative_names(text: str) -> list[str]:
    return [m.group(1) for m in re.finditer(_vocative, text)
            if m.group(1) not in _vocative_skip]


def deidentify(
    text: str,
    names: Sequence[str] = (),
    keep: Sequence[str] = (),
    detect_vocatives: bool = True,
) -> dict[str, Any]:
    """Mask identifiers in text, returning the result and per-category counts."""
    counts = {}
    cleaned = text

    found = list(names)
    if detect_vocatives:
        found += _vocative_names(text)
    cleaned, counts["name"] = _mask_names(cleaned, found)

    titled = re.compile(_title_name)
    cleaned, n = titled.subn(replacements["name"], cleaned)
    counts["name"] += n

    for label, pattern in patterns.items():
        if label in keep:
            counts[label] = 0
            continue
        cleaned, n = re.subn(pattern, replacements[label], cleaned, flags=re.IGNORECASE)
        counts[label] = n

    return {"text": cleaned, "counts": counts, "total": sum(counts.values())}


def deidentify_turns(
    turns: Sequence[dict[str, Any]],
    keep: Sequence[str] = (),
    pseudonymize_speakers: bool = True,
) -> list[dict[str, Any]]:
    """Mask identifiers across turns, using speaker labels as names to remove."""
    labels = [t.get("speaker", "") for t in turns]
    aliases = {}
    for label in labels:
        if label and label not in aliases:
            aliases[label] = f"Speaker {len(aliases) + 1}"

    cleaned = []
    for turn in turns:
        result = deidentify(turn.get("text", ""), names=labels, keep=keep)
        row = dict(turn)
        row["text"] = result["text"]
        row["deidentified"] = result["counts"]
        if pseudonymize_speakers:
            row["speaker"] = aliases.get(turn.get("speaker", ""), turn.get("speaker", ""))
        cleaned.append(row)
    return cleaned


def scan(text: str, names: Sequence[str] = ()) -> dict[str, Any]:
    """Report identifiers found without modifying the text."""
    found = {}
    for label, pattern in patterns.items():
        found[label] = len(re.findall(pattern, text, flags=re.IGNORECASE))
    found["name"] = len(re.findall(_title_name, text))
    found["name"] += len(_vocative_names(text))
    for name in set(names):
        if name.strip():
            found["name"] += len(re.findall(rf"\b{re.escape(name.strip())}\b", text, re.IGNORECASE))

    total = sum(found.values())
    capitalized = {w for w in re.findall(r"\b[A-Z][a-z]{2,}\b", text)} - _vocative_skip
    if total:
        verdict = "identifiers_present"
    elif capitalized:
        verdict = "no_pattern_match_review_needed"
    else:
        verdict = "no_pattern_match"

    return {"found": found, "total": total,
            "unmatched_capitalized": sorted(capitalized)[:20],
            "verdict": verdict}
