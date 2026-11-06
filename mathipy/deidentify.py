"""Replace identifying information in transcripts and written responses."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

patterns = {
    "email": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
    "phone": r"(?<![\d.,])(?:\b|(?<![\w(+])(?=[(+]))(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}"
             r"[-.\s]?\d{4}\b(?![.,]?\d)",
    "url": r"\bhttps?://\S+|\bwww\.\S+",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "student_id": r"\b(?:(?:id|student\s*(?:id|number|no\.?))(?:\s*(?:number|no\.?))?"
                  r"\s*[:#]*\s*\d{4,}"
                  r"|student\s*[:#]?\s*\d{5,})\b",
    "date": r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}"
            r"(?:st|nd|rd|th)?,?\s+\d{4}\b"
            r"|\b(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/(?:19|20)\d{2}\b"
            r"|\b(?:0?[1-9]|1[0-2])-(?:0?[1-9]|[12]\d|3[01])-(?:19|20)\d{2}\b"
            r"|\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b",
}
_titles = r"(?:Mr|Mrs|Ms|Miss|Dr|Prof|Professor|Coach)"
# A capitalized word in Latin or Cyrillic script, with inner capitals, apostrophes, or
# hyphens: McDonald, O'Neil, García, Smith-Jones, Ольга. Greek capitals are left out
# because in math text they are symbols (Δx, ΔABC), not names.
_name_word = r"[A-ZÀ-ÖØ-ÞĀ-ſЀ-Я](?:[^\W\d_]|['’-](?=[^\W\d_]))+"
_particle = r"(?:(?:de|del|della|der|den|di|da|dos|du|la|le|van|von|bin|al|el|ibn)\s+)*"
_title_name = rf"\b{_titles}\.?\s+{_particle}{_name_word}(?:\s+{_particle}{_name_word})?"

_greeting_words = ("hi", "hello", "hey", "welcome", "thanks", "thank you", "good job",
                   "well done", "great job", "good morning", "good afternoon")
_greetings = "(?:" + "|".join(f"[{w[0].upper()}{w[0]}]{w[1:]}" for w in _greeting_words) + ")"
_vocative = rf"(?<!\w){_greetings}\b,?\s+({_name_word})"
_vocative_skip = {"Thanks", "Thank", "Hello", "Hi", "Hey", "Great", "Good", "Well",
                  "Yes", "Okay", "Sure", "Right", "Correct", "Nice", "Perfect",
                  "Awesome", "Excellent", "You", "Your", "That", "This", "What",
                  "How", "Why", "Can", "Could", "Would", "Let", "Now", "So",
                  "Everyone", "Everybody", "All", "Class", "Team", "Guys", "Folks",
                  "Friends", "Kids", "Boys", "Girls", "Students", "There", "Again",
                  "Back", "Today", "Ladies", "Gentlemen",
                  "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
                  "Sunday", "January", "February", "March", "September", "October",
                  "November", "December", "Math", "Science", "English", "History"}
# Months and subjects that are also given names (April, June, Art...) stay maskable.

_role_words = {"teacher", "teachers", "student", "students", "instructor",
               "tutor", "unknown", "speaker", "class", "group", "all", "interviewer",
               "researcher", "observer", "parent", "principal"}

# Names that are also everyday words: as a speaker label, only "Mark" is masked, not "mark".
_word_names = {"mark", "will", "grace", "hope", "joy", "faith", "rose", "may", "june",
               "april", "bill", "art", "pat", "frank", "grant", "dawn", "rich", "ray",
               "rob", "chase", "drew", "dean", "page", "reed", "wade", "ivy", "iris",
               "lily", "penny", "ruby", "amber", "cliff", "summer", "max", "sue", "jean",
               "gene", "glen", "eve"}

replacements = {
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "url": "[URL]",
    "ssn": "[SSN]",
    "student_id": "[ID]",
    "date": "[DATE]",
    "name": "[NAME]",
}


def _name_patterns(names: Sequence[str], exact: Sequence[str] = ()) -> list[re.Pattern]:
    """A pattern per full name (any case) and per capitalized part of a multi-word
    name, so "Maria" alone is masked but the verb "will" is not for "Will Smith".
    ``exact`` names, picked up after a greeting, match only as written, so
    "Hello Mark" does not mask the verb "mark"."""
    full, parts = set(), set()
    for name in names:
        name = name.strip()
        if not name:
            continue
        full.add(name)
        words = re.findall(r"[^\W\d_](?:[^\W\d_]|['’-])+", name)
        if len(words) > 1:
            parts.update(w[0].upper() + w[1:] for w in words
                         if w.lower() not in _role_words and not re.fullmatch(_titles, w))
    lowered = {n.lower() for n in full}
    forms = ([(n, re.IGNORECASE) for n in full]
             + [(p, 0) for p in (parts | set(exact)) if p.lower() not in lowered])
    forms.sort(key=lambda f: len(f[0]), reverse=True)
    return [re.compile(rf"\b{re.escape(n)}\b", flags) for n, flags in forms]


def _mask_names(text: str, names: Sequence[str], exact: Sequence[str] = ()
                ) -> tuple[str, int]:
    count = 0
    for pattern in _name_patterns(names, exact):
        text, n = pattern.subn(replacements["name"], text)
        count += n
    return text, count


_code_label = re.compile(r"[A-Z]\d*|[A-Z]{1,2}\d+")


def _is_name_label(label: str | None) -> bool:
    label = (label or "").strip()
    words = set(re.findall(r"[a-z]{2,}", label.lower()))  # "Student A", "Group 1" are codes
    return bool(label) and not _code_label.fullmatch(label) and not words <= _role_words


def _in_number_list(m: re.Match) -> bool:
    """A space-only "phone number" inside a run of numbers ("345 678 1234 9") is a list;
    one written with "-", "." or "(" is a phone number wherever it sits."""
    if re.search(r"[-.()]", m.group(0)):
        return False
    before, after = m.string[max(0, m.start() - 2):m.start()], m.string[m.end():m.end() + 2]
    return bool(re.fullmatch(r"\d ", before) or re.fullmatch(r" \d", after))


def _opens_clause(text: str, start: int) -> bool:
    """A greeting opens a sentence or clause ("Okay, thanks Maria", "(laughs) Hello
    Maria"), so "well done Friday" after a word in mid-sentence is not one."""
    before = text[:start].rstrip(" \t ")
    return not before or before[-1] in "\r\n" or not before[-1].isalnum()


def _vocative_names(text: str) -> list[str]:
    return [m.group(1) for m in re.finditer(_vocative, text)
            if m.group(1) not in _vocative_skip and _opens_clause(text, m.start())]


def deidentify(
    text: str,
    names: Sequence[str] = (),
    keep: Sequence[str] = (),
    detect_vocatives: bool = True,
    greeted: Sequence[str] = (),
) -> dict[str, Any]:
    """Mask identifiers in text, returning the result and per-category counts.

    Emails, URLs, dates, and numbers are masked before names, so a name inside an
    address, or a greeted name that is also a month ("Hello May"), does not break
    them apart. ``greeted`` adds names found after greetings elsewhere,
    such as in other turns of the same transcript."""
    counts = {}
    cleaned = text or ""

    def replace(label, m):
        if label == "phone" and (_in_number_list(m) or m.group(0).isdigit()):
            return m.group(0)
        counts[label] += 1
        return replacements[label]

    def mask(labels):
        nonlocal cleaned
        for label in labels:
            counts[label] = 0
            if label not in keep:
                cleaned = re.sub(patterns[label], lambda m, label=label: replace(label, m),
                                 cleaned, flags=re.IGNORECASE)

    mask(patterns)
    if "name" in keep:
        counts["name"] = 0
    else:
        exact = list(greeted) + (_vocative_names(cleaned) if detect_vocatives else [])
        cleaned, n = re.subn(_title_name, replacements["name"], cleaned)
        cleaned, counts["name"] = _mask_names(cleaned, names, exact)
        counts["name"] += n

    counts = {label: counts[label] for label in ("name", *patterns)}
    return {"text": cleaned, "counts": counts, "total": sum(counts.values())}


def deidentify_turns(
    turns: Sequence[dict[str, Any]],
    keep: Sequence[str] = (),
    pseudonymize_speakers: bool = True,
) -> list[dict[str, Any]]:
    """Mask identifiers across turns, using speaker labels, and names greeted in any
    turn, as names to remove from every turn."""
    labels = [t.get("speaker", "") for t in turns]
    named = [label.strip() for label in labels if _is_name_label(label)]
    names = [n for n in named if n.lower() not in _word_names]
    greeted = ([n[0].upper() + n[1:] for n in named if n.lower() in _word_names]
               + [n for t in turns for n in _vocative_names(t.get("text") or "")])
    aliases = {}
    for label in labels:
        if label and label not in aliases:
            aliases[label] = f"Speaker {len(aliases) + 1}"

    cleaned = []
    for turn in turns:
        result = deidentify(turn.get("text") or "", names=names, keep=keep, greeted=greeted)
        row = dict(turn)
        row["text"] = result["text"]
        row["deidentified"] = result["counts"]
        if pseudonymize_speakers:
            row["speaker"] = aliases.get(turn.get("speaker", ""), turn.get("speaker", ""))
        cleaned.append(row)
    return cleaned


def scan(text: str, names: Sequence[str] = ()) -> dict[str, Any]:
    """Report identifiers found without modifying the text."""
    text = text or ""
    found = deidentify(text, names=names)["counts"]

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
