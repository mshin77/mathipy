"""Speaker-turn segmentation and turn-level measures for math dialogue."""

from __future__ import annotations

import csv
import re
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

transcript_formats = ("plain", "vtt", "srt", "csv")

_speaker_line = re.compile(
    r"^\s*(?:[\[(\-–]\s*)?([^\W\d_][^:\n\])]{0,39})[\])]?(?!(?<=\d):\d):(?!//)\s*(.*)$")
_timestamp = re.compile(r"\d{1,2}:\d{2}(:\d{2})?[.,]?\d*\s*-->")
_stopwords = set(
    "a an the of to in for is are and or on with it its that this you i we he "
    "she they be was were do does did so but not no yes okay right".split()
)


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower())
            if len(w) > 2 and w not in _stopwords}


_clock = r"\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?"
_lead_clock = re.compile(rf"^\s*(?:\[{_clock}\]|\({_clock}\)|{_clock})\s*")
_label_clock = re.compile(
    rf"^(\s*[^\W\d_][^:\n\[\]()]{{0,39}}?)\s*(?:\[{_clock}\]|\({_clock}\))\s*:")


def _untimed(line: str) -> str:
    """Drop a timestamp before a label ("[00:01:23] Teacher: ...") or between a label
    and its colon ("Teacher (00:01:23): ..."); a line that is not then labeled is kept."""
    stripped = _label_clock.sub(r"\1:", _lead_clock.sub("", line, count=1), count=1)
    return stripped if _speaker_line.match(stripped) else line


def _from_plain(lines: Sequence[str], unlabeled: str | None = None) -> list[tuple[str, str]]:
    turns = []
    for line in lines:
        line = _untimed(line)
        match = _speaker_line.match(line)
        if match:
            turns.append([match.group(1).strip(), match.group(2).strip()])
        elif turns and line.strip():
            turns[-1][1] = f"{turns[-1][1]} {line.strip()}".strip()
        elif unlabeled and line.strip():
            turns.append([unlabeled, line.strip()])
    return [(s, t) for s, t in turns if t]


_voice = re.compile(r"^<v(?:\.[^\s>]*)?\s+([^>]+)>(.*)$")
_cue_tag = re.compile(r"</?[^>]+>")


_vtt_block = re.compile(r"^(?:NOTE|STYLE|REGION)(?:\s|$)")
_dash_turn = re.compile(r"^[-–]\s+")  # "- Yes." marks a new speaker; "-3" is a number


def _from_captions(lines: Sequence[str], continue_speaker: bool = False
                   ) -> list[tuple[str, str]]:
    """Read cues as turns; each labeled line or "- " line starts a turn.

    ``lines`` keep their blank lines, which end cues and NOTE/STYLE/REGION blocks; a
    line between a blank line and a timing line is a cue identifier and is dropped.
    An unlabeled cue is "Unknown" unless ``continue_speaker`` credits it to the
    previous speaker, for captions that name a speaker only when the speaker changes."""
    lines = [ln.strip() for ln in lines]
    timed = any(_timestamp.search(ln) for ln in lines)
    cues, cue, seen, skipping = [], [], False, False
    for i, line in enumerate(lines):
        if _timestamp.search(line):
            if seen:
                cues.append(cue)
            cue, seen, skipping = [], True, False
        elif not line:
            skipping = False
        elif _vtt_block.match(line) and (i == 0 or not lines[i - 1]):
            skipping = True
        elif skipping or (timed and not seen):
            continue
        elif (i + 1 < len(lines) and _timestamp.search(lines[i + 1])
              and (i == 0 or not lines[i - 1] or line.isdigit())):
            continue
        else:
            cue.append(line)
    cues.append(cue)

    turns = []
    for cue in cues:
        for j, line in enumerate(cue):
            voice = _voice.match(line)
            match = None if voice else _speaker_line.match(line)
            if voice:
                speaker, text = voice.group(1).strip(), voice.group(2)
            elif match:
                speaker, text = match.group(1).strip(), match.group(2)
            elif _dash_turn.match(line):
                speaker, text = "Unknown", _dash_turn.sub("", line, count=1)
            elif turns and (j > 0 or (continue_speaker and turns[-1][0] != "Unknown")):
                turns[-1][1] = f"{turns[-1][1]} {_cue_tag.sub('', line).strip()}".strip()
                continue
            else:
                speaker, text = "Unknown", line
            turns.append([speaker, _cue_tag.sub("", text).strip()])
    return [(s, t) for s, t in turns if t]


def _from_csv(path: Path, speaker_key: str, text_key: str) -> list[tuple[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return [((r.get(speaker_key) or "").strip(), (r.get(text_key) or "").strip())
            for r in rows if (r.get(text_key) or "").strip()]


def segment_turns(
    source: str | Path | None = None,
    transcript_format: str = "plain",
    speaker_key: str = "speaker",
    text_key: str = "text",
    text: str | None = None,
    separator: str | None = None,
    strip_pattern: str | None = None,
    continue_speaker: bool = False,
) -> list[dict[str, Any]]:
    """Split a transcript into speaker turns, from a file or an in-memory string.

    For ``vtt`` and ``srt``, an unlabeled cue is an "Unknown" turn; set
    ``continue_speaker`` to credit it to the previous speaker instead."""
    if transcript_format not in transcript_formats:
        raise ValueError(f"transcript_format must be one of {transcript_formats}")
    if text is None and source is None:
        raise ValueError("provide either source or text")
    if text is not None and transcript_format == "csv":
        raise ValueError("csv format requires a file path, not text")

    def parse(body: str) -> list[tuple[str, str]]:
        body = body.lstrip("\ufeff")
        body = re.sub(strip_pattern, "", body) if strip_pattern else body
        lines = [ln.strip() for ln in (body.split(separator) if separator else body.splitlines())]
        if transcript_format == "plain":
            pairs = _from_plain([ln for ln in lines if ln])
        else:
            pairs = _from_captions(lines, continue_speaker)
        if not pairs and any(lines):
            warnings.warn("no speaker turns found in a non-empty transcript; "
                          "check the speaker label format", stacklevel=3)
        return pairs

    if text is not None:
        pairs = parse(text)
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if transcript_format == "csv":
            pairs = _from_csv(path, speaker_key, text_key)
        else:
            pairs = parse(path.read_text(encoding="utf-8"))

    return [{"turn": i, "speaker": s, "text": t, "word_count": len(t.split())}
            for i, (s, t) in enumerate(pairs)]


def turn_measures(turns: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return talk share, turn length, and cross-speaker uptake."""
    if not turns:
        return {"turns": 0, "speakers": 0, "talk_share": {}, "turn_counts": {},
                "mean_turn_words": 0.0, "uptake_mean": 0.0}

    words = {}
    counts = {}
    for t in turns:
        words[t["speaker"]] = words.get(t["speaker"], 0) + t["word_count"]
        counts[t["speaker"]] = counts.get(t["speaker"], 0) + 1
    total = sum(words.values()) or 1

    uptakes = []
    for prev, curr in zip(turns, turns[1:]):
        if prev["speaker"] == curr["speaker"]:
            continue
        base = _tokens(prev["text"])
        if base:
            uptakes.append(len(_tokens(curr["text"]) & base) / len(base))

    return {
        "turns": len(turns),
        "speakers": len(words),
        "talk_share": {s: round(w / total, 4) for s, w in words.items()},
        "turn_counts": counts,
        "mean_turn_words": round(sum(t["word_count"] for t in turns) / len(turns), 2),
        "uptake_mean": round(sum(uptakes) / len(uptakes), 4) if uptakes else 0.0,
    }


def check_speakers(turns: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Flag speaker labels likely to be transcription artifacts."""
    labels = {}
    for t in turns:
        labels.setdefault(t["speaker"], []).append(t["word_count"])

    singletons = [s for s, w in labels.items() if len(w) == 1]
    near = [(a, b) for a in labels for b in labels
            if a < b and a.lower().replace(" ", "") == b.lower().replace(" ", "")]
    consecutive = sum(1 for p, c in zip(turns, turns[1:]) if p["speaker"] == c["speaker"])

    return {
        "speakers": sorted(labels),
        "single_turn_speakers": sorted(singletons),
        "near_duplicate_labels": near,
        "consecutive_same_speaker": consecutive,
        "verdict": "review" if singletons or near else "clean",
    }
