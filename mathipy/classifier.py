"""Visual model classification for math assessment images."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from mathipy._api import VisionAPIClient
from mathipy.visual import visual_models

logger = logging.getLogger(__name__)

_type_list = ", ".join(visual_models)

_example = json.dumps(
    {m: (m == "bar_graph" or m == "table") for m in visual_models}
    | {"primary": "bar_graph"},
    indent=None,
)

classify_system_prompt = (
    "You are an expert classifier for K-12 math assessment visual representations."
)

classify_user_prompt = f"""For this math assessment item image, identify which visual representations are present.
Return a JSON object with boolean values for each type, plus a "primary" field for the most prominent type.
Types: {_type_list}

Example response:
{_example}

Return ONLY valid JSON, nothing else."""


class VisualModelClassifier(VisionAPIClient):
    """Classify which of the 20 visual model types appear in an assessment image.

    Returns boolean flags per type, the primary type, and a model count.

    Requires ``pip install mathipy[ocr]`` and a ``GEMINI_API_KEY`` or
    ``OPENAI_API_KEY`` in your ``.env`` file.
    """

    def classify(self, source: str | Path | bytes) -> dict[str, Any]:
        """Classify visual models present in the image.

        Args:
            source: Image file path, URL, or bytes.

        Returns:
            Dict with a boolean per model type, ``"primary"`` (str),
            and ``"model_count"`` (int).
        """
        image_b64, mime_type = self._prepare_image(source)

        if self.provider == "gemini":
            raw = self._call_gemini(
                image_b64, mime_type,
                system_prompt=classify_system_prompt,
                user_prompt=classify_user_prompt,
            )
        else:
            raw = self._call_openai(
                image_b64, mime_type,
                system_prompt=classify_system_prompt,
                user_prompt=classify_user_prompt,
            )

        return self._parse_classify_response(raw)

    @staticmethod
    def text_only_result() -> dict[str, Any]:
        """Return a classification result for text-only items (no image)."""
        result = {m: False for m in visual_models}
        result["primary"] = "text_only"
        result["model_count"] = 0
        return result

    @staticmethod
    def _parse_classify_response(raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip()

        json_match = re.search(r"\{[\s\S]*\}", cleaned)
        if not json_match:
            logger.warning("No JSON found in classify response; returning all-False")
            result = {m: False for m in visual_models}
            result["primary"] = "other"
            result["model_count"] = 0
            return result

        parsed = json.loads(json_match.group())

        entry: dict[str, Any] = {}
        for m in visual_models:
            entry[m] = bool(parsed.get(m, False))

        primary = parsed.get("primary", "other")
        if primary not in visual_models and primary != "text_only":
            primary = "other"
        entry["primary"] = primary
        entry["model_count"] = sum(entry[m] for m in visual_models)
        return entry
