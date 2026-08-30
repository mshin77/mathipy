"""Redundancy structure of the extracted feature set.

A composite is the exact sum of its components; carrying both makes a
correlation matrix singular. The dependencies are declared here so an analysis
can drop them programmatically rather than from a hand-maintained list.

``compositional_sets`` records shares that sum to a constant, where no single
column is the redundant one. ``near_duplicates`` records pairs that overlap in
practice rather than by algebra.
"""

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
    """Columns with every composite removed, keeping its components.

    Use when a correlation matrix is about to be formed - factor analysis,
    clustering, or any shrinkage prior over the full set.
    """
    return [c for c in columns if c not in composite_features]


def drop_components(columns):
    """Columns with the components removed, keeping each composite whose
    components were all present.

    The opposite reduction: use when the total is the quantity of interest
    and the breakdown is not.
    """
    absorbed = {
        part
        for composite, parts in composite_features.items()
        if composite in columns and all(p in columns for p in parts)
        for part in parts
    }
    return [c for c in columns if c not in absorbed]


def dependent_sets(columns):
    """Every composite in ``columns`` whose components are also present.

    Reports the linear dependencies actually realised in a given column set,
    so a caller can see what would make the matrix singular before it does.
    """
    return {
        composite: parts
        for composite, parts in composite_features.items()
        if composite in columns and all(p in columns for p in parts)
    }


def saturated_sets(columns):
    """Compositional sets in ``columns`` that are present in full.

    A set of shares summing to a constant is singular when every member is
    included, and no single column can be named as the redundant one - which
    is why these are reported separately from ``composite_features`` rather
    than folded into it. The caller chooses which part to leave out.
    """
    return {
        name: parts
        for name, parts in compositional_sets.items()
        if all(p in columns for p in parts)
    }
