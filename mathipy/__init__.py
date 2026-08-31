"""mathipy - Multimodal item feature extraction for K-12 math assessment."""

__version__ = "0.4.6"
__author__ = "Mikyung Shin"
__email__ = "shin.mikyung@gmail.com"

from mathipy.classifier import VisualModelClassifier
from mathipy.cognitive_load import CognitiveLoadEstimator
from mathipy.cohesion import (
    cohesion_features,
    connective_density,
    lexical_diversity,
    lexical_overlap,
    pronoun_density,
)
from mathipy.deidentify import deidentify, deidentify_turns, scan
from mathipy.dialogue import check_speakers, segment_turns, turn_measures
from mathipy.documents import (body_paragraphs, check_alignment, paragraph_images,
                               paragraph_text, segment_docx)
from mathipy.item import ItemFeatureExtractor, MultimodalAnalyzer
from mathipy.crossmodal import crossmodal_features, deictic_features, label_features
from mathipy.features import (
    composite_features,
    dependent_sets,
    drop_components,
    drop_composites,
    near_duplicates,
)
from mathipy.fractions import fraction_features
from mathipy.geometry import classify_shapes
from mathipy.math_content import MathContentAnalyzer
from mathipy.morphology import morphology_features
from mathipy.notation import normalize_math_notation, insert_operators
from mathipy.symbolic import channel_pairs, symbolic_features
from mathipy.register import (
    homonym_features,
    number_features,
    register_features,
    relational_features,
)
from mathipy.ocr import MultimodalOCR
from mathipy.readability import ReadabilityAnalyzer
from mathipy.utils import compute_interrater_reliability, safe_get
from mathipy.validation import (
    disagreements,
    ocr_rubric,
    score_agreement,
    stratified_sample,
    visual_rubric,
    write_coding_sheets,
    write_label_studio,
)
from mathipy.visual import (
    VisualFeatureExtractor,
    flags_by_group,
    flags_by_sign,
    group_names,
    sign_names,
    visual_function_definitions,
    visual_functions,
    visual_model_definitions,
    visual_model_groups,
    visual_model_info,
    visual_model_signs,
)

__all__ = [
    "ReadabilityAnalyzer",
    "MathContentAnalyzer",
    "CognitiveLoadEstimator",
    "VisualFeatureExtractor",
    "MultimodalOCR",
    "VisualModelClassifier",
    "MultimodalAnalyzer",
    "ItemFeatureExtractor",
    "flags_by_group",
    "flags_by_sign",
    "group_names",
    "sign_names",
    "visual_function_definitions",
    "visual_functions",
    "visual_model_definitions",
    "visual_model_groups",
    "visual_model_info",
    "visual_model_signs",
    "safe_get",
    "compute_interrater_reliability",
    "segment_docx",
    "check_alignment",
    "body_paragraphs",
    "paragraph_images",
    "paragraph_text",
    "deidentify",
    "deidentify_turns",
    "scan",
    "segment_turns",
    "turn_measures",
    "check_speakers",
    "cohesion_features",
    "connective_density",
    "lexical_overlap",
    "lexical_diversity",
    "pronoun_density",
    "crossmodal_features",
    "deictic_features",
    "label_features",
    "normalize_math_notation",
    "insert_operators",
    "morphology_features",
    "fraction_features",
    "composite_features",
    "near_duplicates",
    "drop_composites",
    "drop_components",
    "dependent_sets",
    "symbolic_features",
    "channel_pairs",
    "classify_shapes",
    "register_features",
    "relational_features",
    "number_features",
    "homonym_features",
    "stratified_sample",
    "write_coding_sheets",
    "write_label_studio",
    "score_agreement",
    "disagreements",
    "ocr_rubric",
    "visual_rubric",
]
