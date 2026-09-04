"""Geometric shape subtypes from contour analysis.

Shape category is too coarse for difficulty: van Hiele (1986) levels and
the prototype phenomenon (Hershkowitz, 1989) predict recognition cost
rises with irregularity and atypical orientation, so subtype is the
signal. Deterministic image processing, no API call.
"""

from __future__ import annotations

import collections
import math
from pathlib import Path

import numpy as np

from mathipy._api import _optional_import

cv2, cv2_available = _optional_import("cv2", "opencv-python-headless")

_SIDE_TOL = 0.05       # relative tolerance for "equal length" (triangles, quadrilaterals)

_ANGLE_TOL_DEG = 7.0

_WORK_SCALE = 1000

_POLYGON_SIDE_TOL = 0.20
_MIN_AREA_SHARE = 1e-4
_MIN_CONTOUR_AREA_FLOOR = 40

_empty = {
    "shape_triangle_equilateral": 0, "shape_triangle_isosceles": 0,
    "shape_triangle_scalene": 0, "shape_triangle_right": 0,
    "shape_quad_square": 0, "shape_quad_rectangle": 0, "shape_quad_rhombus": 0,
    "shape_quad_parallelogram": 0, "shape_quad_trapezoid": 0,
    "shape_quad_kite": 0, "shape_quad_quadrilateral": 0, "shape_quad_irregular": 0,
    "shape_polygon_pentagon": 0, "shape_polygon_hexagon": 0,
    "shape_polygon_heptagon": 0, "shape_polygon_octagon": 0,
    "shape_polygon_regular_other": 0, "shape_polygon_irregular": 0,
    "shape_circle": 0, "shape_ellipse": 0,
    "shape_filled_count": 0, "shape_outline_count": 0,
    "shape_mean_fill_ratio": 0.0, "shape_partition_count": 0,
    "shape_shaded_partition_count": 0,
}

_FILL_THRESHOLD = 0.5

_ERODE_SHARE = 0.12
_ERODE_FLOOR = 9

_OWN_INTERIOR_SHARE = 0.75

_GRIDLINE_SHARE = 0.6

_LATTICE_TOL = 0.25

_POLYGON_NAMES = {5: "shape_polygon_pentagon", 6: "shape_polygon_hexagon",
                  7: "shape_polygon_heptagon", 8: "shape_polygon_octagon"}

_QUAD_ANCESTORS = {
    "shape_quad_square": ["shape_quad_rectangle", "shape_quad_rhombus",
                          "shape_quad_parallelogram", "shape_quad_trapezoid",
                          "shape_quad_quadrilateral"],
    "shape_quad_rectangle": ["shape_quad_parallelogram", "shape_quad_trapezoid",
                             "shape_quad_quadrilateral"],
    "shape_quad_rhombus": ["shape_quad_parallelogram", "shape_quad_trapezoid",
                           "shape_quad_quadrilateral"],
    "shape_quad_parallelogram": ["shape_quad_trapezoid", "shape_quad_quadrilateral"],
    "shape_quad_trapezoid": ["shape_quad_quadrilateral"],
    "shape_quad_kite": ["shape_quad_quadrilateral"],
    "shape_quad_irregular": ["shape_quad_quadrilateral"],
}


def _resize_to_work_scale(gray: np.ndarray) -> np.ndarray:
    """Cap the working resolution so an area threshold means one thing.

    Downsampling only. Interpolating a small figure up adds no detail and
    costs accuracy where it matters most here: a circle outline enlarged to
    work scale loses enough boundary smoothness to be reclassified as an
    ellipse. Small images keep their own pixels, and ``_min_area`` carries the
    scale adjustment for them instead.
    """
    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest <= _WORK_SCALE:
        return gray
    factor = _WORK_SCALE / longest
    return cv2.resize(gray, (max(1, round(w * factor)), max(1, round(h * factor))),
                      interpolation=cv2.INTER_AREA)


def _min_area(gray: np.ndarray) -> float:
    """Smallest contour worth classifying, as a share of the frame.

    A fixed 100 px² is 0.026% of a full-page screenshot and 0.1% of a small
    figure, so the same constant admitted text glyphs on one image and
    rejected real grid cells on another.
    """
    h, w = gray.shape[:2]
    return max(_MIN_CONTOUR_AREA_FLOOR, _MIN_AREA_SHARE * h * w)


def _depth(hierarchy: np.ndarray, index: int) -> int:
    """How many contours enclose this one."""
    depth = 0
    parent = hierarchy[index][3]
    while parent != -1:
        depth += 1
        parent = hierarchy[parent][3]
    return depth


def _side_lengths(pts: np.ndarray) -> list[float]:
    n = len(pts)
    return [float(np.linalg.norm(pts[i] - pts[(i + 1) % n])) for i in range(n)]


def _interior_angles(pts: np.ndarray) -> list[float]:
    n = len(pts)
    angles = []
    for i in range(n):
        a, b, c = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
        v1, v2 = a - b, c - b
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        angles.append(math.degrees(math.acos(np.clip(cos_theta, -1.0, 1.0))))
    return angles


def _near(a: float, b: float, rel_tol: float = _SIDE_TOL) -> bool:
    return abs(a - b) <= rel_tol * max(a, b, 1e-9)


def _near_deg(a: float, b: float, tol: float = _ANGLE_TOL_DEG) -> bool:
    """Compare two angles on an absolute scale.

    Angles were compared with the relative ``_near``, which at 108 degrees
    admits a 16-degree difference and at 20 degrees admits three. Measurement
    error does not scale with the size of the angle, so the relative form
    judged small angles strictly and large ones loosely - the wrong way round.
    """
    return abs(a - b) <= tol


def _classify_triangle(sides: list[float],
                       angles: list[float]) -> tuple[str, bool]:
    s = sorted(sides)
    is_right = any(abs(a - 90) <= _ANGLE_TOL_DEG for a in angles)
    if _near(s[0], s[2]) and _near(s[1], s[2]):
        shape = "shape_triangle_equilateral"
    elif _near(s[0], s[1]) or _near(s[1], s[2]) or _near(s[0], s[2]):
        shape = "shape_triangle_isosceles"
    else:
        shape = "shape_triangle_scalene"
    return shape, is_right


def _opposite_sides_parallel(pts: np.ndarray, i: int) -> bool:
    """Whether side i and its opposite side (i+2) are parallel: the sine of
    the angle between their direction vectors, from the cross product, is
    near zero. This is the actual definition of a parallel side pair -
    equal opposite angles (a parallelogram property) is a different thing
    and does not detect a general (non-parallelogram) trapezoid."""
    n = len(pts)
    v1 = pts[(i + 1) % n] - pts[i]
    v2 = pts[(i + 3) % n] - pts[(i + 2) % n]
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-9:
        return False
    sin_angle = abs(v1[0] * v2[1] - v1[1] * v2[0]) / norm
    return sin_angle < 0.05


def _classify_quadrilateral(pts: np.ndarray, sides: list[float],
                            angles: list[float]) -> str:
    both_pairs_parallel = (_opposite_sides_parallel(pts, 0)
                           and _opposite_sides_parallel(pts, 1))
    one_pair_parallel = (_opposite_sides_parallel(pts, 0)
                         or _opposite_sides_parallel(pts, 1))
    all_sides_equal = all(_near(sides[i], sides[0]) for i in range(4))
    opp_sides_equal = _near(sides[0], sides[2]) and _near(sides[1], sides[3])
    all_angles_right = all(_near_deg(a, 90) for a in angles)
    adjacent_pairs_equal = (_near(sides[0], sides[1]) and _near(sides[2], sides[3])) or \
                           (_near(sides[1], sides[2]) and _near(sides[3], sides[0]))

    if all_sides_equal and both_pairs_parallel and all_angles_right:
        return "shape_quad_square"
    if opp_sides_equal and both_pairs_parallel and all_angles_right:
        return "shape_quad_rectangle"
    if all_sides_equal and both_pairs_parallel:
        return "shape_quad_rhombus"
    if opp_sides_equal and both_pairs_parallel:
        return "shape_quad_parallelogram"
    if adjacent_pairs_equal and not opp_sides_equal:
        return "shape_quad_kite"
    if one_pair_parallel:
        return "shape_quad_trapezoid"
    return "shape_quad_irregular"


def _classify_contour(contour: np.ndarray, min_area: float) -> tuple[str, str | None] | None:
    area = cv2.contourArea(contour)
    if area < min_area:
        return None
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    pts = approx.reshape(-1, 2).astype(float)
    n = len(pts)

    is_stable = len(cv2.approxPolyDP(contour, 0.01 * perimeter, True)) <= n + 1

    if n == 3 and is_stable:
        sides = _side_lengths(pts)
        angles = _interior_angles(pts)
        shape, is_right = _classify_triangle(sides, angles)
        return (shape, "shape_triangle_right") if is_right else (shape, None)

    if n == 4 and is_stable:
        sides = _side_lengths(pts)
        angles = _interior_angles(pts)
        return (_classify_quadrilateral(pts, sides, angles), None)

    if 5 <= n <= 10 and is_stable:
        sides = _side_lengths(pts)
        angles = _interior_angles(pts)
        regular = all(_near(s, sides[0], _POLYGON_SIDE_TOL) for s in sides) and \
            all(_near_deg(a, angles[0]) for a in angles)
        return (_POLYGON_NAMES.get(n, "shape_polygon_regular_other") if regular
                else "shape_polygon_irregular", None)

    circularity = 4 * math.pi * area / (perimeter * perimeter)
    if circularity > 0.85:
        return ("shape_circle", None)
    if circularity > 0.6:
        return ("shape_ellipse", None)
    return ("shape_polygon_irregular", None) if n > 4 else None


def _interior_mask(binary_shape: tuple[int, int], contour: np.ndarray) -> np.ndarray:
    """The region a contour encloses, with its own stroke band removed.

    The band is eroded by a fraction of the shape's own size rather than a
    fixed number of pixels, which is what makes the fill ratio comparable
    across shapes of different sizes.
    """
    mask = np.zeros(binary_shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    _, _, w, h = cv2.boundingRect(contour)
    k = max(_ERODE_FLOOR, int(_ERODE_SHARE * min(w, h))) | 1
    return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def _fill_ratio(binary: np.ndarray, contour: np.ndarray) -> float:
    """Share of a contour's interior that is drawn in.

    The interior excludes the stroke band; measured against the full enclosed
    area the ratio is a function of size rather than of fill.
    """
    mask = _interior_mask(binary.shape, contour)
    enclosed = int(np.count_nonzero(mask))
    if enclosed == 0:
        return 0.0
    return float(np.count_nonzero(cv2.bitwise_and(binary, mask)) / enclosed)


def _lattice(binary: np.ndarray, contour: np.ndarray) -> tuple[int, int] | None:
    """Row and column counts of a ruled grid inside a contour, or None.

    Gridlines are found as peaks in the projection profile of the stroke mask
    rather than as connected components. Counting components cannot work on a
    shaded model: a shaded cell is contiguous with the lines around it, so
    blob counting returned one region for ten shaded cells and for fifty.
    A projection profile is unaffected, because a line that is invisible where
    it crosses a shaded cell is still drawn across the rest of the figure.
    """
    x, y, w, h = cv2.boundingRect(contour)
    if w < 12 or h < 12:
        return None
    inside = binary[y:y + h, x:x + w]

    def divisions(profile: np.ndarray, span: int, extent: int) -> int:
        hits = np.flatnonzero(profile >= _GRIDLINE_SHARE * span)
        if hits.size == 0:
            return 0
        breaks = np.insert(np.diff(hits) > 1, 0, True)
        starts = hits[breaks]
        ends = np.append(hits[np.append(breaks[1:], True)], hits[-1])[:starts.size]
        widths = ends - starts + 1
        thin = starts[widths <= max(3, int(np.median(widths)) * 2)]
        starts = thin if thin.size >= 3 else starts
        if starts.size < 3:
            return 0
        gap = float(np.median(np.diff(starts)))
        return int(round(extent / gap)) if gap >= 2 else 0

    rows = divisions((inside > 0).sum(axis=1), w, h)
    cols = divisions((inside > 0).sum(axis=0), h, w)
    return (rows, cols) if rows >= 2 and cols >= 2 else None


def _shaded_cells(binary: np.ndarray, contour: np.ndarray,
                  grid: tuple[int, int]) -> int:
    """Cells of a ruled grid whose interior is filled."""
    nrow, ncol = grid
    x, y, w, h = cv2.boundingRect(contour)
    inside = binary[y:y + h, x:x + w]
    ys = np.linspace(0, h, nrow + 1).round().astype(int)
    xs = np.linspace(0, w, ncol + 1).round().astype(int)
    pad_y = max(1, h // (nrow * 4)); pad_x = max(1, w // (ncol * 4))
    filled = 0
    for r in range(nrow):
        for c in range(ncol):
            cell = inside[ys[r] + pad_y:ys[r + 1] - pad_y,
                          xs[c] + pad_x:xs[c + 1] - pad_x]
            if cell.size and np.count_nonzero(cell) / cell.size >= _FILL_THRESHOLD:
                filled += 1
    return filled


def _shaded_regions(binary: np.ndarray, contour: np.ndarray, min_area: float) -> int:
    """Number of solidly shaded sub-regions inside a partitioned figure."""
    mask = _interior_mask(binary.shape, contour)
    inside = cv2.bitwise_and(binary, mask)
    _, _, w, h = cv2.boundingRect(contour)
    k = max(3, int(0.04 * min(w, h)) | 1)
    blobs = cv2.morphologyEx(inside, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    count, _, stats, _ = cv2.connectedComponentsWithStats(blobs, connectivity=4)
    interior = max(1, int(np.count_nonzero(mask)))
    return sum(1 for i in range(1, count)
               if min_area <= stats[i, cv2.CC_STAT_AREA]
               < _OWN_INTERIOR_SHARE * interior)


def classify_shapes(image_source: str | Path | np.ndarray) -> dict[str, int]:
    """Counts of geometric shape subtypes found via contour analysis.

    Returns an all-zero dict (same keys) if OpenCV is unavailable or no
    shapes are found, so output always has a stable column set.
    """
    if not cv2_available:
        return dict(_empty)

    if isinstance(image_source, np.ndarray):
        image = image_source
    elif isinstance(image_source, (bytes, bytearray)):
        image = cv2.imdecode(np.frombuffer(bytes(image_source), dtype=np.uint8),
                             cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(str(image_source))
    if image is None:
        return dict(_empty)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = _resize_to_work_scale(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return dict(_empty)
    hierarchy = hierarchy[0]
    areas = [cv2.contourArea(c) for c in contours]
    min_area = _min_area(gray)

    counts = dict(_empty)
    holes = collections.Counter()
    for index in range(len(contours)):
        parent = hierarchy[index][3]
        if parent != -1 and _depth(hierarchy, index) % 2 == 1:
            holes[parent] += 1
    fills = []
    for index, contour in enumerate(contours):
        parent = hierarchy[index][3]
        if _depth(hierarchy, index) % 2 == 1:
            if (areas[index] >= min_area
                    and areas[index] < _OWN_INTERIOR_SHARE * areas[parent]):
                counts["shape_partition_count"] += 1
            continue

        result = _classify_contour(contour, min_area)
        if result is None:
            continue
        primary, secondary = result
        counts[primary] += 1
        for ancestor in _QUAD_ANCESTORS.get(primary, ()):
            counts[ancestor] += 1
        if secondary:
            counts[secondary] += 1

        ratio = _fill_ratio(binary, contour)
        fills.append(ratio)
        counts["shape_filled_count" if ratio >= _FILL_THRESHOLD
               else "shape_outline_count"] += 1
        grid = _lattice(binary, contour)
        shaded = _shaded_cells(binary, contour, grid) if grid else 0
        if grid and abs(shaded / (grid[0] * grid[1]) - ratio) <= _LATTICE_TOL:
            counts["shape_partition_count"] += max(0, grid[0] * grid[1] - holes[index])
            counts["shape_shaded_partition_count"] += shaded
        else:
            regions = _shaded_regions(binary, contour, min_area)
            counts["shape_shaded_partition_count"] += regions
            counts["shape_partition_count"] += regions

    if fills:
        counts["shape_mean_fill_ratio"] = round(sum(fills) / len(fills), 4)
    return counts
