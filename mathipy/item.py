"""Item-level feature extraction orchestrating all mathipy analyzers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mathipy.cognitive_load import CognitiveLoadEstimator
from mathipy.math_content import MathContentAnalyzer
from mathipy.readability import ReadabilityAnalyzer
from mathipy.utils import safe_get
from mathipy.visual import VisualFeatureExtractor

logger = logging.getLogger(__name__)


class ItemFeatureExtractor:
    """Run all four analyzers and return a flat ``prefix_*`` dict.

    Readability, math-content, and cognitive-load features come from
    ``text``; visual features are extracted per image and aggregated to
    the item level (mean for continuous, sum for counts, max for
    dimensions, mode for complexity level).
    """

    def __init__(self):
        self._read = ReadabilityAnalyzer()
        self._math = MathContentAnalyzer()
        self._cog = CognitiveLoadEstimator()
        self._vis = VisualFeatureExtractor()

    def extract(
        self,
        text: str,
        image_paths: list[str | Path] | None = None,
    ) -> dict[str, Any]:
        """Extract all features for one item.

        Args:
            text: Concatenated item text (already collapsed across images).
            image_paths: Optional list of image file paths for visual features.

        Returns:
            Flat dict with ``readability_*``, ``math_*``, ``cognitive_*``,
            and ``visual_*`` keys.
        """
        row: dict[str, Any] = {}

        # readability
        rd = self._read.analyze(text)
        row.update({
            "readability_flesch_kincaid_grade": rd.get("flesch_kincaid_grade"),
            "readability_avg_grade_level": rd.get("average_grade_level"),
            "readability_flesch_reading_ease": rd.get("flesch_reading_ease"),
            "readability_gunning_fog": rd.get("gunning_fog"),
            "readability_smog_index": rd.get("smog_index"),
            "readability_coleman_liau": rd.get("coleman_liau_index"),
            "readability_ari": rd.get("automated_readability_index"),
            "readability_dale_chall": rd.get("dale_chall_readability"),
            "readability_low_confidence": int(rd.get("low_confidence", False)),
        })

        # math content
        mc = self._math.analyze(text)
        row.update({
            "math_density": mc.get("math_density"),
            "math_total_symbols": mc.get("total_math_symbols"),
            "math_unique_symbol_types": mc.get("unique_symbol_types"),
            "math_numbers_count": safe_get(mc, "numbers", "count"),
            "math_numbers_range": safe_get(mc, "numbers", "range"),
            "math_has_negative": int(safe_get(mc, "numbers", "has_negative", default=False)),
            "math_has_decimal": int(safe_get(mc, "numbers", "has_decimal", default=False)),
            "math_vocab_unique_terms": safe_get(mc, "vocabulary", "unique_terms"),
            "math_vocab_term_count": safe_get(mc, "vocabulary", "term_count"),
            "math_domain_primary": safe_get(mc, "domain_classification", "primary"),
            "math_domain_confidence": safe_get(mc, "domain_classification", "confidence"),
        })
        patterns = mc.get("pattern_matches", {})
        for pname in ["equation", "fraction", "variable", "percentage", "ratio", "exponent"]:
            row[f"math_pat_{pname}"] = patterns.get(pname, 0)

        # cognitive load
        cl = self._cog.estimate(
            text, readability_grade=row.get("readability_flesch_kincaid_grade")
        )
        row.update({
            "cognitive_intrinsic": cl.get("intrinsic_cognitive_load"),
            "cognitive_extraneous": cl.get("extraneous_cognitive_load"),
            "cognitive_germane": cl.get("germane_cognitive_load"),
            "cognitive_total": cl.get("total_cognitive_load"),
            "cognitive_numeric_elements": cl.get("numeric_elements"),
            "cognitive_variable_count": cl.get("variable_count"),
            "cognitive_operation_count": cl.get("operation_count"),
        })

        # visual features
        if image_paths:
            flat_list = []
            for p in image_paths:
                path = Path(p)
                if not path.exists():
                    continue
                try:
                    flat_list.append(self._vis.extract_flat(str(path)))
                except Exception as e:
                    logger.warning(f"Visual extraction failed for {path}: {e}")
            row.update(VisualFeatureExtractor.aggregate_visual_features(flat_list))
        else:
            row.update(VisualFeatureExtractor.aggregate_visual_features([]))

        return row
