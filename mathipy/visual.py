"""Visual feature extraction from assessment images.

Types are grouped by the multiple-representations framework (Lesh et al.,
1987), with the data display family following Friel et al. (2001) and
instructional function adapting Levin et al. (1987). ``text_only`` marks an
item exported
as an image that carries no figure, so ``visual_group_no_visual`` is the
figure indicator.

Pixel measures read whatever region they are given. Over a whole frame they
read ink on the page, including the item's interface and text; ``crop_to_box``
narrows to the figure and reports which region was measured.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mathipy._api import _optional_import
from mathipy.utils import safe_get

_WORK_SCALE = 1000

_MAX_UPSCALE = 2.0
_THIN_STRIP_PX = 64

_CANNY_THRESHOLDS = (50, 150)

_HOUGH_VOTES = 80

_TRIM_MARGIN = 8

logger = logging.getLogger(__name__)

_pil_image_mod, pillow_available = _optional_import("PIL.Image", "pillow")
Image = _pil_image_mod
cv2, cv2_available = _optional_import("cv2", "opencv-python-headless")

visual_models = [
    "number_line", "coordinate_plane", "function_graph", "bar_graph", "line_graph",
    "pictograph", "dot_plot", "histogram", "circle_graph", "scatter_plot",
    "box_plot", "stem_leaf_plot",
    "table", "geometric_figure", "solid_figure", "net_diagram", "area_model", "array",
    "tape_diagram", "fraction_model", "pattern_visual", "tree_diagram",
    "venn_diagram", "measurement_tool",
    "picture", "text_only", "other",
]

visual_functions = ["essential", "representational", "decorative"]

visual_model_definitions = {
    "number_line": "a line marked with ordered numeric values or tick marks",
    "coordinate_plane": "a grid with two perpendicular axes for plotting points or shapes, with no single function curve emphasized",
    "function_graph": "a line or curve on axes representing a function or equation; choose over line_graph when the line is a mathematical relationship rather than observed data",
    "bar_graph": "separated rectangular bars whose lengths represent category values",
    "line_graph": "data points connected by segments over an ordered axis such as time; choose over function_graph when points are observed data values",
    "pictograph": "repeated icons or pictures representing counts",
    "dot_plot": "dots stacked above a number line showing frequency (line plot)",
    "histogram": "adjacent touching bars over numeric intervals; choose over bar_graph when bars touch and the axis is binned numbers",
    "circle_graph": "a circle divided into sectors showing parts of a whole (pie chart)",
    "scatter_plot": "unconnected points on two numeric axes showing association",
    "box_plot": "a box spanning quartiles with a median line and whiskers along a numeric axis; choose over number_line when a quartile box is drawn on the axis, and over bar_graph always",
    "stem_leaf_plot": "a two-column display of stems with ordered leaf digits; choose over table whenever the display separates stems from leaves, however tabular it looks",
    "table": "rows and columns of values or labels without pictorial encoding",
    "geometric_figure": "a two-dimensional shape, angle, or line construction",
    "solid_figure": "a three-dimensional object drawn with depth or perspective",
    "net_diagram": "a flattened, unfolded pattern of a three-dimensional figure; choose over solid_figure when unfolded",
    "area_model": "a rectangle partitioned into regions representing products or parts; choose over array when the display is a partitioned region rather than countable objects",
    "array": "discrete objects arranged in equal rows and columns; choose over area_model when the elements are countable objects",
    "tape_diagram": "segmented strips or side-by-side bars representing quantities and their relationship (strip or bar model)",
    "fraction_model": "a shape or set partitioned with parts shaded or marked to show a fraction",
    "pattern_visual": "a repeating or growing sequence of shapes or objects",
    "tree_diagram": "branching lines showing outcomes, factors, or hierarchies; choose over geometric_figure when the lines branch to organise possibilities rather than forming a shape",
    "venn_diagram": "overlapping circles showing set relationships; choose over geometric_figure when the circles overlap to show membership rather than being the object of study",
    "measurement_tool": "a depicted ruler, protractor, clock, scale, thermometer, or similar instrument; choose over number_line when the marked scale belongs to a drawn instrument rather than a bare line",
    "picture": "a photograph or illustration of real-world objects that carries no mathematical encoding; choose a more specific type whenever the image encodes mathematical structure",
    "text_only": "the image carries only the item's own text, equation, table of answer choices or response box, with no diagram, graph, figure or other visual model; choose this whenever nothing in the image would be lost by reading the text alone",
    "other": "a mathematical visual not covered by any listed type",
}

visual_model_groups = {
    "number_line": "number_quantity", "coordinate_plane": "number_quantity",
    "function_graph": "number_quantity",
    "area_model": "part_whole_model", "array": "part_whole_model",
    "tape_diagram": "part_whole_model", "fraction_model": "part_whole_model",
    "bar_graph": "data_display", "line_graph": "data_display",
    "pictograph": "data_display", "dot_plot": "data_display",
    "histogram": "data_display", "circle_graph": "data_display",
    "scatter_plot": "data_display", "box_plot": "data_display",
    "stem_leaf_plot": "data_display",
    "geometric_figure": "geometric", "solid_figure": "geometric",
    "net_diagram": "geometric", "measurement_tool": "geometric",
    "table": "organizational", "tree_diagram": "organizational",
    "venn_diagram": "organizational",
    "pattern_visual": "other_visual", "picture": "other_visual",
    "text_only": "no_visual", "other": "other_visual",
}

visual_model_info = [
    ("number_line", "number_quantity", "NBT, NF, NS, G, EE, F", "K-12"),
    ("coordinate_plane", "number_quantity", "G, EE, F", "5-12"),
    ("function_graph", "number_quantity", "F, EE", "8-12"),
    ("area_model", "part_whole_model", "OA, NF, NBT", "K-5"),
    ("array", "part_whole_model", "OA, NBT", "K-3"),
    ("tape_diagram", "part_whole_model", "OA, NF", "K-7"),
    ("fraction_model", "part_whole_model", "NF", "3-5"),
    ("bar_graph", "data_display", "MD (K-5), SP (6-12)", "K-12"),
    ("line_graph", "data_display", "MD (K-5), SP (6-12)", "4-12"),
    ("pictograph", "data_display", "MD", "K-3"),
    ("dot_plot", "data_display", "MD (K-5), SP (6-12)", "3-12"),
    ("histogram", "data_display", "SP", "6-12"),
    ("circle_graph", "data_display", "SP", "6-12"),
    ("scatter_plot", "data_display", "SP", "8-12"),
    ("box_plot", "data_display", "SP", "6-12"),
    ("stem_leaf_plot", "data_display", "SP (legacy)", "4-12"),
    ("geometric_figure", "geometric", "G", "K-12"),
    ("solid_figure", "geometric", "G", "K-12"),
    ("net_diagram", "geometric", "G", "6-8"),
    ("measurement_tool", "geometric", "MD, G", "K-8"),
    ("table", "organizational", "All domains", "K-12"),
    ("tree_diagram", "organizational", "SP, OA", "6-12"),
    ("venn_diagram", "organizational", "OA, SP", "2-12"),
    ("pattern_visual", "other_visual", "OA", "K-5"),
    ("picture", "other_visual", "Various", "K-12"),
    ("text_only", "no_visual", "Various", "K-12"),
    ("other", "other_visual", "Various", "K-12"),
]

group_names = sorted(set(visual_model_groups.values()))

visual_model_signs = {
    "geometric_figure": "icon", "solid_figure": "icon", "net_diagram": "icon",
    "area_model": "icon", "array": "icon", "fraction_model": "icon",
    "tape_diagram": "icon", "pictograph": "icon", "measurement_tool": "icon",
    "picture": "icon", "pattern_visual": "icon",
    "number_line": "index", "coordinate_plane": "index", "dot_plot": "index",
    "scatter_plot": "index", "line_graph": "index", "function_graph": "index",
    "bar_graph": "index", "histogram": "index", "box_plot": "index",
    "circle_graph": "index", "tree_diagram": "index",
    "table": "symbol", "stem_leaf_plot": "symbol", "venn_diagram": "symbol",
    "other": "symbol",
    "text_only": "none",
}

sign_names = sorted(set(visual_model_signs.values()))


def _is_set(value: Any) -> bool:
    """Read a flag that may have round-tripped through CSV or JSON as text.

    Plain truthiness read "0", "false" and "no" as True, silently inverting
    every negative flag that reached this function as a string.
    """
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "t", "y"}
    return bool(value)


def trim_margin(image: np.ndarray | bytes, margin: int = _TRIM_MARGIN
                ) -> np.ndarray | bytes:
    """Crop an image to its ink, leaving a fixed margin.

    Capture margin is not a property of the item, but every ratio computed over
    the frame divides by it: the same figure captured with 0, 60 and 200 pixels
    of white around it measured .0624, .0496 and .0360. After trimming, all
    three measure .0652. Trimming makes the capture reproducible without
    re-capturing anything, and unlike a manual crop it is deterministic, so a
    reader can regenerate the numbers.
    """
    if not cv2_available:
        return image
    if isinstance(image, (bytes, bytearray)):
        decoded = cv2.imdecode(np.frombuffer(bytes(image), np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            return image
        image = decoded
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    rows, cols = np.where(ink > 0)
    if not len(cols):
        return image
    h, w = image.shape[:2]
    y0, y1 = max(0, rows.min() - margin), min(h, rows.max() + margin + 1)
    x0, x1 = max(0, cols.min() - margin), min(w, cols.max() + margin + 1)
    return image[y0:y1, x0:x1]


def crop_to_box(image: np.ndarray | bytes, box: Sequence[float] | None
                ) -> tuple[np.ndarray | bytes, str]:
    """Crop to a figure box; return the image and the region measured.

    ``box`` holds fractions of width and height, so it survives a change of
    export resolution. Scope is ``"figure"`` when the crop applied and
    ``"frame"`` otherwise; the two are not comparable, so it is returned
    rather than assumed.
    """
    if box is None or not cv2_available:
        return image, "frame"
    if isinstance(image, (bytes, bytearray)):
        decoded = cv2.imdecode(np.frombuffer(bytes(image), np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            return image, "frame"
        image = decoded
    h, w = image.shape[:2]
    left, top, right, bottom = box
    x0, x1 = int(round(left * w)), int(round(right * w))
    y0, y1 = int(round(top * h)), int(round(bottom * h))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return image, "frame"
    return image[y0:y1, x0:x1], "figure"


def flags_by_group(entry: dict[str, Any]) -> dict[str, int]:
    """Collapse per-type boolean flags into one flag per representation family."""
    hits = dict.fromkeys(group_names, False)
    for m in visual_models:
        if _is_set(entry.get(m)):
            hits[visual_model_groups[m]] = True
    return {f"visual_group_{g}": int(v) for g, v in hits.items()}


def flags_by_sign(entry: dict[str, Any]) -> dict[str, int]:
    """Collapse per-type boolean flags into one flag per Peircean sign class.

    The coarsest of the three collapses. Where the type flags are too sparse
    to estimate - most fire on under 3% of items - this asks whether the kind
    of sign matters at all.

    ``visual_sign_none`` is the no-figure class and is the sign-side twin of
    ``visual_group_no_visual``; the three real classes are read against it.
    """
    hits = dict.fromkeys(sign_names, False)
    for m in visual_models:
        if _is_set(entry.get(m)):
            hits[visual_model_signs[m]] = True
    return {f"visual_sign_{s}": int(v) for s, v in hits.items()}


class VisualFeatureExtractor:
    """Extract visual complexity features from assessment images.

    Analyzes dimensions, pixel statistics, edge metrics, structural elements
    (lines, circles, shapes), frequency domain, and overall complexity score.

    Requires ``pip install mathipy[vision]`` for full features.
    """

    def __init__(self):
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        if not pillow_available:
            logger.warning("Image loading requires Pillow")
        if not cv2_available:
            logger.warning("Advanced features require OpenCV")

    def extract(self, image_source: str | Path | bytes | np.ndarray) -> dict[str, Any]:
        """Extract visual features from an image.

        Args:
            image_source: File path, Path object, raw image bytes, or numpy array.

        Returns:
            Dictionary with ``dimensions``, ``pixel_statistics``, ``edge_metrics``,
            ``structural_elements``, ``frequency_domain``, and ``complexity_score``.
        """
        image = self._load_image(image_source)
        if image is None:
            return self._empty_features()

        features = {
            "dimensions": self._extract_dimensions(image),
            "pixel_statistics": self._extract_pixel_stats(image),
        }

        if cv2_available:
            gray = self._to_gray(image)
            work = self._resize_to_work_scale(gray)
            features["edge_metrics"] = self._extract_edge_metrics(work)
            features["structural_elements"] = self._extract_structural_elements(work)
            features["frequency_domain"] = self._extract_frequency_features(work)

        return features

    vis_map = {
        "visual_contrast": ("pixel_statistics", "std"),
        "visual_mean": ("pixel_statistics", "mean"),
        "visual_edge_ratio": ("edge_metrics", "edge_ratio"),
        "visual_aspect_ratio": ("dimensions", "aspect_ratio"),
        "visual_high_freq_ratio": ("frequency_domain", "high_freq_ratio"),
        "visual_mid_freq_ratio": ("frequency_domain", "mid_freq_ratio"),
        "visual_total_shapes": ("structural_elements", "total_shapes"),
        "visual_line_count": ("structural_elements", "line_count"),
        "visual_circle_count": ("structural_elements", "circle_count"),
        "visual_width": ("dimensions", "width"),
        "visual_height": ("dimensions", "height"),
    }
    vis_sum_keys = {"visual_total_shapes", "visual_line_count", "visual_circle_count"}
    vis_max_keys = {"visual_width", "visual_height"}

    def extract_flat(self, image_source: str | Path | bytes | np.ndarray) -> dict[str, Any]:
        """One flat ``visual_*`` dict for an image.

        Keys are those in ``vis_map``. Every key is None when the image
        cannot be read or when OpenCV is absent, rather than 0, which would
        be indistinguishable from a real measurement.
        """
        nested = self.extract(image_source)
        return {key: safe_get(nested, *path, default=None)
                for key, path in self.vis_map.items()}

    @classmethod
    def aggregate_visual_features(
        cls, flat_list: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """One item-level dict from several per-image dicts.

        Area-weighted mean for continuous metrics, sum for counts, max for
        dimensions, plus ``visual_image_count``. Weighting by area keeps a
        6px inline fragment from moving an item as far as its screenshot;
        aspect ratio is recomputed from the aggregated dimensions.
        """
        if not flat_list:
            return {k: None for k in cls.vis_map} | {"visual_image_count": 0}

        agg: dict[str, Any] = {}
        for k in cls.vis_map:
            pairs = [(v, cls._pixels(f)) for f in flat_list
                     if (v := f.get(k)) is not None]
            if not pairs:
                agg[k] = None
            elif k in cls.vis_sum_keys:
                agg[k] = sum(v for v, _ in pairs)
            elif k in cls.vis_max_keys:
                agg[k] = max(v for v, _ in pairs)
            else:
                weight = sum(a for _, a in pairs)
                agg[k] = (sum(v * a for v, a in pairs) / weight if weight
                          else sum(v for v, _ in pairs) / len(pairs))

        w, h = agg["visual_width"], agg["visual_height"]
        agg["visual_aspect_ratio"] = round(w / h, 3) if w and h else None
        agg["visual_image_count"] = len(flat_list)
        return agg

    @staticmethod
    def _pixels(flat: dict[str, Any]) -> float:
        """Pixel area of one extracted image, used to weight the item mean."""
        w, h = flat.get("visual_width"), flat.get("visual_height")
        return float(w * h) if w and h else 0.0

    def _load_image(self, source: str | Path | bytes | np.ndarray) -> np.ndarray | None:
        if isinstance(source, np.ndarray):
            return source

        if not pillow_available and not cv2_available:
            logger.error("Cannot load image - no image library available")
            return None

        if isinstance(source, (bytes, bytearray)):
            data = np.frombuffer(bytes(source), dtype=np.uint8)
            if cv2_available:
                return cv2.imdecode(data, cv2.IMREAD_COLOR)
            pil_image = Image.open(io.BytesIO(bytes(source)))
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            return np.array(pil_image)[:, :, ::-1]

        path = Path(source) if isinstance(source, str) else source
        if not path.exists():
            logger.error(f"Image not found: {path}")
            return None

        if cv2_available:
            image = cv2.imread(str(path))
            if image is None:
                logger.error(f"Image unreadable: {path}")
            return image
        elif pillow_available:
            pil_image = Image.open(path)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            return np.array(pil_image)[:, :, ::-1]  # RGB to BGR

        return None

    def _extract_dimensions(self, image: np.ndarray) -> dict[str, Any]:
        h, w = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1

        return {
            "width": w,
            "height": h,
            "aspect_ratio": round(w / h, 3) if h > 0 else 0.0,
            "total_pixels": w * h,
            "channels": channels,
        }

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        """One greyscale definition for every metric in this class.

        ``_extract_pixel_stats`` used an unweighted channel mean while the cv2
        path used luma weights, so ``visual_mean`` and ``visual_edge_ratio``
        described the same image on two different luminance scales. An alpha
        plane was averaged in as if it were a colour, which shifted the mean of
        any RGBA image toward opaque white.
        """
        if image.ndim == 2:
            return image if image.dtype == np.uint8 else image.astype(np.uint8)
        image = image[:, :, :3]
        if cv2_available:
            return cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_BGR2GRAY)
        return np.mean(image, axis=2).astype(np.uint8)

    def _resize_to_work_scale(self, gray: np.ndarray) -> np.ndarray:
        """Put every image on one pixel scale before counting anything.

        Edge, line and circle counts are counts of pixels crossing a fixed
        threshold, so the same figure exported at two resolutions produced
        counts differing several-fold and an ``edge_ratio`` that falls as
        1/scale by construction. Export resolution then entered a difficulty
        model as if it were visual complexity. Dimensions are still reported
        from the original image; only the counting runs on this copy.
        """
        h, w = gray.shape[:2]
        longest = max(h, w)
        if longest == 0:
            return gray
        factor = _WORK_SCALE / longest
        if min(h, w) < _THIN_STRIP_PX:
            factor = min(factor, _MAX_UPSCALE)
        if factor == 1:
            return gray
        interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
        return cv2.resize(gray, (max(1, round(w * factor)), max(1, round(h * factor))),
                          interpolation=interp)

    def _extract_pixel_stats(self, image: np.ndarray) -> dict[str, float]:
        gray = self._to_gray(image)

        return {
            "mean": float(np.mean(gray)),
            "std": float(np.std(gray)),
            "min": float(np.min(gray)),
            "max": float(np.max(gray)),
            "median": float(np.median(gray)),
        }

    def _extract_edge_metrics(self, gray: np.ndarray) -> dict[str, float]:
        edges = cv2.Canny(gray, *_CANNY_THRESHOLDS)

        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.size

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        return {
            "edge_pixels": int(edge_pixels),
            "edge_ratio": float(edge_pixels / total_pixels),
            "sobel_mean": float(np.mean(sobel_mag)),
            "sobel_max": float(np.max(sobel_mag)),
            "laplacian_mean": float(np.mean(np.abs(laplacian))),
            "laplacian_std": float(np.std(laplacian)),
        }

    def _extract_structural_elements(self, gray: np.ndarray) -> dict[str, Any]:
        """Count lines, circles and closed shapes on the work-scale image.

        ``line_count`` counts Hough segments over a Canny edge map, and Canny
        returns both sides of every stroke, so a drawn line yields two
        segments. The count is a stable multiple of the true line count, not
        the true line count.
        """
        h, w = gray.shape[:2]
        edges = cv2.Canny(gray, *_CANNY_THRESHOLDS)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, _HOUGH_VOTES,
            minLineLength=max(20, min(h, w) // 10), maxLineGap=10
        )

        max_radius = min(h, w) // 2
        circles = None if max_radius < 2 else cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, minDist=max(1, min(h, w) // 8),
            param1=_CANNY_THRESHOLDS[1], param2=40,
            minRadius=min(max(8, min(h, w) // 40), max_radius - 1),
            maxRadius=max_radius
        )

        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        shapes = {"triangles": 0, "rectangles": 0, "circles": 0, "polygons": 0}
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                vertices = len(approx)

                if vertices == 3:
                    shapes["triangles"] += 1
                elif vertices == 4:
                    shapes["rectangles"] += 1
                elif vertices > 6 and perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.8:
                        shapes["circles"] += 1
                    else:
                        shapes["polygons"] += 1
                else:
                    shapes["polygons"] += 1

        return {
            "line_count": len(lines) if lines is not None else 0,
            "circle_count": len(circles[0]) if circles is not None else 0,
            "contour_count": len(contours),
            "shapes": shapes,
            "total_shapes": sum(shapes.values()),
        }

    def _extract_frequency_features(self, gray: np.ndarray) -> dict[str, float]:
        f_transform = np.fft.fft2(gray.astype(float))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2

        total_energy = np.sum(magnitude)

        r_low = min(rows, cols) // 8
        y, x = np.ogrid[:rows, :cols]
        mask_low = np.sqrt((x - ccol)**2 + (y - crow)**2) <= r_low
        low_energy = np.sum(magnitude * mask_low)

        r_mid = min(rows, cols) // 4
        mask_mid = np.sqrt((x - ccol)**2 + (y - crow)**2) <= r_mid
        mid_energy = np.sum(magnitude * (mask_mid & ~mask_low))

        high_energy = total_energy - low_energy - mid_energy

        return {
            "total_energy": float(total_energy),
            "low_freq_ratio": float(low_energy / total_energy) if total_energy > 0 else 0,
            "mid_freq_ratio": float(mid_energy / total_energy) if total_energy > 0 else 0,
            "high_freq_ratio": float(high_energy / total_energy) if total_energy > 0 else 0,
        }

    def _empty_features(self) -> dict[str, Any]:
        return {
            "dimensions": {},
            "pixel_statistics": {},
            "edge_metrics": {},
            "structural_elements": {},
            "frequency_domain": {},
        }
