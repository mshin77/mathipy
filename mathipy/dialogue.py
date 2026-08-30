"""Speaker-turn segmentation and turn-level measures for math dialogue."""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

transcript_formats = ("plain", "vtt", "srt", "csv")

_speaker_line = re.compile(r"^\s*([^:\n]{1,40}):\s*(.*)$")
_timestamp = re.compile(r"\d{1,2}:\d{2}(:\d{2})?[.,]?\d*\s*-->")
_stopwords = set(
    "a an the of to in for is are and or on with it its that this you i we he "
    "she they be was were do does did so but not no yes okay right".split()
)


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower())
            if len(w) > 2 and w not in _stopwords}


def _from_plain(lines: Sequence[str]) -> list[tuple[str, str]]:
    turns = []
    for line in lines:
        match = _speaker_line.match(line)
        if match:
            turns.append([match.group(1).strip(), match.group(2).strip()])
        elif turns and line.strip():
            turns[-1][1] += " " + line.strip()
    return [(s, t) for s, t in turns if t]


def _from_captions(lines: Sequence[str]) -> list[tuple[str, str]]:
    body = [ln for ln in lines
            if ln.strip() and not _timestamp.search(ln)
            and not ln.strip().isdigit() and ln.strip() != "WEBVTT"]
    return _from_plain(body)


def _from_csv(path: Path, speaker_key: str, text_key: str) -> list[tuple[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return [(r.get(speaker_key, "").strip(), r.get(text_key, "").strip())
            for r in rows if r.get(text_key, "").strip()]


def segment_turns(
    source: str | Path | None = None,
    transcript_format: str = "plain",
    speaker_key: str = "speaker",
    text_key: str = "text",
    text: str | None = None,
    separator: str | None = None,
    strip_pattern: str | None = None,
) -> list[dict[str, Any]]:
    """Split a transcript into speaker turns, from a file or an in-memory string.

    Pass ``text`` to segment a string directly. ``separator`` splits turns when they
    are not newline-delimited, and ``strip_pattern`` removes inline annotations.
    """
    if transcript_format not in transcript_formats:
        raise ValueError(f"transcript_format must be one of {transcript_formats}")
    if text is None and source is None:
        raise ValueError("provide either source or text")

    if text is not None:
        if transcript_format == "csv":
            raise ValueError("csv format requires a file path, not text")
        body = re.sub(strip_pattern, "", text) if strip_pattern else text
        lines = body.split(separator) if separator else body.splitlines()
        lines = [ln.strip() for ln in lines if ln.strip()]
        pairs = _from_plain(lines) if transcript_format == "plain" else _from_captions(lines)
        return [{"turn": i, "speaker": s, "text": t, "word_count": len(t.split())}
                for i, (s, t) in enumerate(pairs)]

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if transcript_format == "csv":
        pairs = _from_csv(path, speaker_key, text_key)
    else:
        raw = path.read_text(encoding="utf-8")
        if strip_pattern:
            raw = re.sub(strip_pattern, "", raw)
        lines = raw.split(separator) if separator else raw.splitlines()
        lines = [ln.strip() for ln in lines if ln.strip()]
        pairs = _from_plain(lines) if transcript_format == "plain" else _from_captions(lines)

    return [{"turn": i, "speaker": s, "text": t, "word_count": len(t.split())}
            for i, (s, t) in enumerate(pairs)]


def turn_measures(turns: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return talk share, turn length, and cross-speaker uptake."""
    if not turns:
        return {"turns": 0, "speakers": 0, "talk_share": {}, "mean_turn_words": 0.0,
                "uptake_mean": 0.0}

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
