"""Redundancy structure of the extracted feature set."""

composite_features = {
    "num_total": (
        "num_cardinal", "num_ordinal", "num_fraction", "num_nominal",
    ),
    "rel_total": (
        "rel_comparison", "rel_rate", "rel_multiplicative", "rel_partitive",
        "rel_distribution", "rel_division",
    ),
    "sym_total": (
        "sym_addition", "sym_subtraction", "sym_multiplicative",
        "sym_division", "sym_comparison", "sym_equality", "sym_exponent",
    ),
    "cohesion_connective_total": (
        "cohesion_connective_additive", "cohesion_connective_causal",
        "cohesion_connective_temporal", "cohesion_connective_conditional",
        "cohesion_connective_adversative", "cohesion_connective_clarifying",
    ),
    "shape_quad_quadrilateral": (
        "shape_quad_trapezoid", "shape_quad_kite", "shape_quad_irregular",
    ),
    "shape_classified_count": (
        "shape_filled_count", "shape_outline_count",
    ),
    "readability_avg_grade_level": (
        "readability_flesch_kincaid_grade", "readability_gunning_fog",
        "readability_smog_index",
    ),
}

compositional_sets = {
    "frequency_bands": (
        "visual_low_freq_ratio", "visual_mid_freq_ratio", "visual_high_freq_ratio",
    ),
}

near_duplicates = (
    (
        "cognitive_operation_count", "math_total_symbols", 0.9999,
        "cognitive_load counts the ASCII operator set +-*/^=<> while "
        "math_content counts a wider symbol table that contains it; in "
        "ordinary math prose the wider table adds few extra hits",
    ),
)


def drop_composites(columns):
    """Columns with every composite removed, keeping its components."""
    return [c for c in columns if c not in composite_features]


def drop_components(columns):
    """Columns with the components removed, keeping each composite whose
    components were all present."""
    absorbed = {
        part
        for composite, parts in composite_features.items()
        if composite in columns and all(p in columns for p in parts)
        for part in parts
    }
    return [c for c in columns if c not in absorbed]


def dependent_sets(columns):
    """Every composite in ``columns`` whose components are also present."""
    return {
        composite: parts
        for composite, parts in composite_features.items()
        if composite in columns and all(p in columns for p in parts)
    }


def saturated_sets(columns):
    """Compositional sets in ``columns`` that are present in full."""
    return {
        name: parts
        for name, parts in compositional_sets.items()
        if all(p in columns for p in parts)
    }
