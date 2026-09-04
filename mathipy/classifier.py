"""Visual model classification for math assessment images."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from mathipy._api import VisionAPIClient
from mathipy.visual import (
    visual_function_definitions,
    visual_functions,
    visual_model_definitions,
    visual_models,
)

logger = logging.getLogger(__name__)

_type_lines = "\n".join(f"- {m}: {visual_model_definitions[m]}" for m in visual_models)
_function_lines = "\n".join(f"- {f}: {visual_function_definitions[f]}"
                            for f in visual_functions)

_example = json.dumps(
    {m: (m == "bar_graph" or m == "table") for m in visual_models}
    | {"primary": "bar_graph", "function": "essential",
       "figure_box": [0.08, 0.21, 0.74, 0.66], "option_boxes": []},
    indent=None,
)

classify_system_prompt = (
    "You are an expert classifier for K-12 math assessment visual representations."
)

classify_user_prompt = f"""For this math assessment item image, identify which visual representations are present.
Return a JSON object with boolean values for each type, plus a "primary" field for the most prominent type,
plus a "function" field for the instructional role of the image:
{_function_lines}
Return "no_visual" for "function" exactly when "primary" is text_only, that is
when the image carries no figure at all: a screenshot of prose, an answer interface,
or a bare fragment of notation. An image that is itself a figure, including a cropped
figure filling the frame, always takes essential or decorative even when "figure_box"
is null. Never pair text_only with essential or decorative, and never pair a figure
type with no_visual.

Types:
{_type_lines}

Also return "figure_box": the rectangle containing the mathematical figure, as
[left, top, right, bottom] in fractions of image width and height, where 0,0 is
the top-left corner. The box must contain the figure together with its own
title, axis labels, tick labels, legend and callout labels, and must NOT contain
the item's prose, the answer options, or any response control (buttons, entry
boxes, radio buttons). Return "figure_box": null when the image carries no
figure - when it is a screenshot of text, an answer interface, or a fragment of
notation on a text baseline.

When the answer choices are themselves graphics rather than text, return
"option_boxes": a list of one rectangle per choice, in the same coordinate
form. Four small graphs offered as four choices are four option boxes, not one
figure. Return an empty list when the choices are text.

Example response, showing the required format only. Its values are not a
recommended answer:
{_example}

Return ONLY valid JSON, nothing else."""


def _normalize_label(value: Any) -> str:
    """Fold a returned label to the spelling the typology uses.

    The model returns "Bar Graph", "bar-graph" and "bar_graph" for the same
    thing across calls. An exact membership test sent the first two to
    "other", so the same image classified twice could land in two different
    columns.
    """
    if not isinstance(value, str):
        return ""
    return re.sub(r"[\s-]+", "_", value.strip().lower())


_MIN_BOX_AREA = 0.01
_MAX_BOX_AREA = 0.98


def _parse_box(value: Any) -> list[float] | None:
    """Validate a returned figure box, or None if it cannot be trusted.

    The box is the one output that silently changes every downstream pixel
    measure, so a malformed one has to be refused rather than clipped into
    something plausible. Coordinates are fractions of width and height, which
    keeps the box independent of the resolution the item was exported at.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        left, top, right, bottom = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if not all(0.0 <= v <= 1.0 for v in (left, top, right, bottom)):
        return None
    if right <= left or bottom <= top:
        return None
    if not _MIN_BOX_AREA <= (right - left) * (bottom - top) <= _MAX_BOX_AREA:
        return None
    return [left, top, right, bottom]


def _parse_boxes(value: Any) -> list[list[float]]:
    """Validate a list of option-choice boxes, dropping any that fail.

    Answer choices that are graphics are several regions, not one. A single
    rectangle drawn round all of them would enclose the choice labels, the
    selection controls and the space between, which is the chrome the box
    exists to exclude.
    """
    if not isinstance(value, (list, tuple)):
        return []
    return [box for box in (_parse_box(v) for v in value) if box]


def _build_user_prompt(item_text: str | None = None) -> str:
    if item_text is None:
        return classify_user_prompt
    return (classify_user_prompt
            + f'\n\nItem text (judge the "function" field against it):\n{item_text}')


class VisualModelClassifier(VisionAPIClient):
    """Classify which visual model types appear in an assessment image.

    Returns boolean flags per type, the primary type, the instructional
    function of the image, and a model count.

    Requires ``pip install mathipy[ocr]`` and a ``GEMINI_API_KEY`` or
    ``OPENAI_API_KEY`` in the ``.env`` file.
    """

    def classify(self, source: str | Path | bytes, votes: int = 1,
                 item_text: str | None = None) -> dict[str, Any]:
        """Classify visual models present in the image.

        Args:
            source: Image file path, URL, or bytes.
            votes: Number of independent classification calls; results are
                merged by majority (flags) and mode (primary, function).
            item_text: Item text. The function label compares image content
                against it; without it the label rests on the image alone.

        Returns:
            Dict with a boolean per model type, ``"primary"`` (str),
            ``"function"`` (str), and ``"model_count"`` (int).
        """
        image_b64, mime_type = self._prepare_image(source)

        results = [self._classify_once(image_b64, mime_type, item_text)
                   for _ in range(max(1, votes))]
        merged = results[0] if len(results) == 1 else self._merge_votes(results)
        return merged | self.provenance() | {"votes": max(1, votes)}

    def _classify_once(self, image_b64: str, mime_type: str,
                       item_text: str | None = None) -> dict[str, Any]:
        call = self._call_gemini if self.provider == "gemini" else self._call_openai
        raw = call(image_b64, mime_type,
                   system_prompt=classify_system_prompt,
                   user_prompt=_build_user_prompt(item_text),
                   json_output=True)
        return self._parse_classify_response(raw)

    @staticmethod
    def text_only_result() -> dict[str, Any]:
        """Return a classification result for an item carrying no image.

        ``text_only`` is set here rather than left False. Leaving it False
        while naming it as primary contradicted the type's purpose: the
        no-visual group flag it exists to drive read 0 for exactly the items
        it was meant to identify.
        """
        result = {m: False for m in visual_models}
        result["text_only"] = True
        result.update({"primary": "text_only", "function": "no_visual",
                       "model_count": 1, "figure_box": None, "option_boxes": [],
                       "parsed": True, "status": "ok"})
        return result

    @staticmethod
    def fallback_result(status: str = "unparseable") -> dict[str, Any]:
        """Return an all-False classification for failed or unparseable calls.

        ``parsed`` is False and ``primary`` is None rather than "other". A
        failed call previously returned primary="other", making it byte-
        identical to a real classification of an unlisted visual - so every
        failure entered a difficulty model as a legitimate "other" item with
        no way to tell the two apart.

        Args:
            status: Why no classification is carried. "unparseable" means the
                response could not be read; "empty" means it was read and named
                nothing, which is a verdict about the image rather than a
                failure of the call. Retrying is worthwhile for the first and
                pointless for the second, and ``parsed`` alone cannot tell
                them apart.
        """
        result = {m: False for m in visual_models}
        result.update({"primary": None, "function": None, "model_count": 0,
                       "figure_box": None, "option_boxes": [], "parsed": False,
                       "status": status})
        return result

    @staticmethod
    def _merge_votes(results: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge repeat calls, dropping failed ones before the vote.

        A failed call carries no opinion, so letting it vote would pull every
        flag toward False and let ``None`` win the primary field on a 2-1
        split. Only parsed calls are counted, and the merge itself fails if
        none survived.
        """
        usable = [r for r in results if r.get("parsed", True)]
        if not usable:
            statuses = {r.get("status", "unparseable") for r in results}
            return VisualModelClassifier.fallback_result(
                "empty" if statuses == {"empty"} else "unparseable")

        n = len(usable)
        entry: dict[str, Any] = {
            m: sum(r[m] for r in usable) * 2 > n for m in visual_models
        }
        for field in ("primary", "function"):
            entry[field] = Counter(r[field] for r in usable).most_common(1)[0][0]
        entry["model_count"] = sum(entry[m] for m in visual_models)
        boxes = [r["figure_box"] for r in usable if r.get("figure_box")]
        entry["figure_box"] = ([sorted(c)[len(c) // 2] for c in zip(*boxes)]
                               if len(boxes) * 2 > n else None)
        entry["option_boxes"] = max((r.get("option_boxes") or [] for r in usable),
                                    key=len, default=[])
        entry["parsed"] = True
        entry["status"] = "ok"
        return VisualModelClassifier._reconcile(entry)

    @staticmethod
    def _reconcile(entry: dict[str, Any]) -> dict[str, Any]:
        """Make ``primary`` consistent with the flags it is drawn from.

        The model can name a primary type it did not flag True, or flag types
        and then name one outside the typology. Either way the primary column
        and the flag columns disagree about the same image, and a model using
        both reads a contradiction. The named type wins - it is the more
        considered judgment - and its flag is raised to match.
        """
        primary = entry.get("primary")
        if primary in visual_models and not entry[primary]:
            entry[primary] = True
            entry["model_count"] = sum(entry[m] for m in visual_models)
        elif primary == "other" and entry["model_count"] == 0:
            entry["text_only"] = False
        return entry

    @staticmethod
    def _strip_code_fence(raw: str) -> str:
        """Remove a markdown fence without assuming it is well formed."""
        cleaned = re.sub(r"^```(?:json|JSON)?[ \t]*\r?\n?", "", raw.strip())
        return re.sub(r"\r?\n?```\s*$", "", cleaned).strip()

    @staticmethod
    def _parse_classify_response(raw: str) -> dict[str, Any]:
        cleaned = VisualModelClassifier._strip_code_fence(raw or "")

        parsed = None
        start = cleaned.find("{")
        if start >= 0:
            decoder = json.JSONDecoder()
            try:
                parsed, _ = decoder.raw_decode(cleaned[start:])
            except json.JSONDecodeError:
                parsed = None
        if not isinstance(parsed, dict):
            logger.warning("Unparseable classify response; the call is worth retrying")
            return VisualModelClassifier.fallback_result("unparseable")

        entry: dict[str, Any] = {m: bool(parsed.get(m, False)) for m in visual_models}

        primary = _normalize_label(parsed.get("primary"))
        if primary and primary not in visual_models:
            logger.warning("unrecognised primary %r; recording as unclassified", primary)
            primary = None

        raw_function = parsed.get("function")
        function = _normalize_label(raw_function)
        if function in (None, "null"):
            function = "no_visual" if primary == "text_only" else None
        elif function not in visual_functions:
            function = "unknown"
        if primary == "text_only" and function in ("essential", "decorative"):
            function = "no_visual"
        elif primary and primary != "text_only" and function == "no_visual":
            function = "unknown"

        box = _parse_box(parsed.get("figure_box"))
        option_boxes = _parse_boxes(parsed.get("option_boxes"))

        if not (primary or any(entry.values()) or box or option_boxes):
            logger.info("Classify response named no visual model; the image carries none")
            return VisualModelClassifier.fallback_result("empty")

        if primary == "text_only":
            function = None

        entry.update({
            "primary": primary or None,
            "function": function,
            "model_count": sum(entry[m] for m in visual_models),
            "figure_box": box,
            "option_boxes": option_boxes,
            "parsed": True,
            "status": "ok",
        })
        return VisualModelClassifier._reconcile(entry)
