"""Visual feature extraction from assessment images."""

from __future__ import annotations

import io
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
    "number_line", "coordinate_plane", "bar_graph", "line_graph",
    "picture_graph", "dot_plot", "histogram", "circle_graph", "scatter_plot",
    "box_plot", "stem_leaf_plot",
    "table", "expression_frame", "polygon", "circle_figure", "angle_figure",
    "solid_figure", "net_diagram", "area_model", "array",
    "tape_diagram", "pattern_visual", "tree_diagram",
    "venn_diagram", "measurement_tool",
    "picture", "text_only",
]

visual_functions = ["no_visual", "essential", "non_essential"]

visual_function_definitions = {
    "no_visual": "the image carries no figure at all: prose, notation, answer choices or response controls only, including a page scan whose only graphics are the answer choices; use this exactly when the type is text_only",
    "essential": "the student must read from the figure a value, relation, or configuration that neither the text nor any text answer choice states, so removing the figure leaves the item unsolvable; this includes a point, scale, partition, or count marked in the image that the text does not give, an instrument drawn against a figure whose reading supplies the value, and an answer-choice image whose choice exists only as a graphic and is not also printed as text",
    "non_essential": "every value and relation needed is stated in the text or in text answer choices, so the item can be answered from the text alone; this covers a figure that restates what the text or the choices already state, a figure that depicts none of the mathematics the item asks about, a blank response workspace (grid, axes, number line, expression frame, response chart, or drag cards printed with the candidate numbers) that only gives a place to build or record the answer, and a generic instrument or manipulative sheet supplied as supplementary material, even when the item tells the student to use it; a workspace is essential only when it carries a marked value, point, or scale the text does not state",
}

visual_model_definitions = {
    "number_line": "a line marked with ordered numeric values or tick marks (number line diagram), including a blank one given as a response workspace; includes a double number line, which is two such lines in parallel; one of the three visual fraction models, with area_model and tape_diagram; the count scale of a data display and the axes of a coordinate plane belong to that display rather than to number_line, and a scale on a drawn instrument is measurement_tool",
    "coordinate_plane": "a coordinate system of two perpendicular axes on which points, shapes, lines, or curves are plotted, including the graph of a function or equation, a distribution curve over a scaled axis, and a blank coordinate grid given as a response workspace; choose line_graph or scatter_plot only when the plotted points are observed data that do not all fall on one exact rule, and keep coordinate_plane when every point of a series lies exactly on one line or smooth curve, joined or not, as in a fee or rate schedule; a square grid with no axes is grid paper and takes the type of the figure drawn or marked on it; the complex plane is one of these",
    "bar_graph": "separated rectangular bars whose lengths represent category values",
    "line_graph": "observed or recorded data points joined by segments over an ordered axis such as time (time graph); choose over coordinate_plane when the changes from point to point are irregular, as in measured or reported values, and choose coordinate_plane when every point of a series lies exactly on one line or smooth curve",
    "picture_graph": "repeated identical icons or pictures set in labeled categories, where the number of icons in a category, read with any key in which one icon stands for several objects, gives that category's count (picture graph); choose over table when the icons sit in the rows of a ruled chart, choose picture when each category shows one drawing with its value printed as a label, and choose table when the marks are tally marks or numerals rather than pictures",
    "dot_plot": "dots stacked above a number line showing frequency; the Standards use 'line plot' through grade 5 and 'dot plot' from grade 6",
    "histogram": "adjacent touching bars over numeric intervals; choose over bar_graph when bars touch and the axis is binned numbers",
    "circle_graph": "a circle divided into sectors whose labels carry categories, values, or outcomes — a pie chart, or a spinner whose sectors are labeled with outcomes; choose area_model when the sectors are equal and carry no labels other than shading, since the count of equal parts is then the content, and circle_figure when the circle is studied as a shape rather than divided into sectors",
    "scatter_plot": "unconnected points of observed bivariate data plotted against two labeled axes, with or without a fitted line or curve drawn through them, whether or not the horizontal axis is numeric; choose line_graph when the points are joined by segments, and coordinate_plane when the points lie exactly on one rule rather than scattering about it; a residual plot is one of these",
    "box_plot": "a box spanning quartiles with a median line and whiskers along a numeric axis; choose over number_line when a quartile box is drawn on the axis, and over bar_graph always",
    "stem_leaf_plot": "a two-column display of stems with ordered leaf digits; choose over table whenever the display separates stems from leaves, however tabular it looks",
    "table": "rows and columns of values or labels without pictorial encoding, in which an entry's row and column together tell what it means, and a picture used only as a row or column heading counts as a label; includes two-way tables, ratio tables, function tables, tally charts, and a matrix; choose picture_graph when repeated icons stand for counts in the rows, and choose text_only when the ruled grid holds only the item's own statements beside answer buttons, when the image is an unruled list of numbers or ordered pairs, or when it is an empty frame with headings and no entries",
    "polygon": "a closed plane figure bounded by straight sides, including a composite region made of joined straight-sided shapes and a schematic or scale drawing of a real object, such as a cross-section, floor plan, or staircase profile, whose labeled lengths enter the computation; choose area_model when the rectangle is partitioned to show a product or parts of a whole, array when the figure is rows and columns of separate objects, coordinate_plane when the figure is drawn on axes, pattern_visual when it repeats or grows in a sequence, solid_figure when it is drawn with depth, circle_figure when a curve rather than straight sides bounds the region, angle_figure when the lines leave the region open, measurement_tool when a graduated instrument is shown against the figure, and picture when the shapes are labeled tiles, cards, or cutouts handled as objects rather than studied as figures; a workspace whose marked points are the vertices of a figure to be constructed, or blank grid paper on which a figure is to be drawn, takes the type of that figure",
    "expression_frame": "an arithmetic expression or vertical computation drawn with empty boxes to be filled, with or without number or digit cards supplied for placement; choose table when the boxes form labeled rows and columns of values",
    "circle_figure": "a circle or circular region studied as a geometric shape, for its size or position or with radius, diameter, arc, chord, or central angle marked; choose area_model when the circle is partitioned into parts of a whole, circle_graph when the sectors carry data values, venn_diagram when circles overlap to show set membership, polygon when straight sides rather than a curve bound the region, and picture when plain circles stand for counted objects such as counters, coins, or marbles",
    "angle_figure": "an angle, ray, line, or line segment construction where no closed region is the object of study; choose polygon when the lines close a region and measurement_tool when a protractor or ruler is shown against the figure; marked points alone are not a construction, and a workspace whose points are the vertices of a figure to be constructed takes the type of that figure",
    "solid_figure": "a three-dimensional object drawn with depth or perspective, including a cross-section of one",
    "net_diagram": "a flattened, unfolded pattern of a three-dimensional figure; choose over solid_figure when unfolded",
    "area_model": "a drawn region — rectangle, circle, or other shape — partitioned into equal or labeled parts to show a product or parts of a whole, including a fraction circle and a spinner or circle cut into equal sectors that carry no labels other than shading; choose over array when the display is a partitioned region rather than countable objects, and over polygon or circle_figure whenever the partition is what the question is about; blank grid paper with no region drawn on it is not an area_model; one of the three visual fraction models, with tape_diagram and number_line",
    "array": "identical discrete objects arranged in at least two equal rows and two equal columns, so that the count is rows times columns; choose over area_model when the elements are countable objects, and choose picture when the objects form a single row or a scattered set, or when each object is distinguished by its own label, as with numbered cards set out in rows, so that the grid layout carries no count",
    "tape_diagram": "one or more rectangular strips drawn as lengths, split into segments or set side by side for comparison, whose lengths represent quantities and their relationship (strip or bar model, fraction strip); choose text_only when boxed letters or numbers sit on a text line as a sentence or equation rather than as lengths of tape, expression_frame when the boxes are empty slots to be filled, and number_line when the strip carries a numeric scale of tick marks; one of the three visual fraction models, with area_model and number_line",
    "pattern_visual": "a repeating or growing sequence of two or more shapes, objects, or figures shown in order, whose repeated unit or rule of growth the item asks about, whether the elements are geometric shapes or realistic drawings; choose over picture when the drawing shows the repeated unit of a sequence to be extended; an answer-choice image showing a single candidate element is typed by what that element is, such as circle_figure or angle_figure, never by the sequence it would complete",
    "tree_diagram": "branching lines showing outcomes, factors, or hierarchies; choose over polygon and angle_figure when the lines branch to organize possibilities rather than forming a shape",
    "venn_diagram": "overlapping circles showing set relationships; choose over circle_figure when the circles overlap to show membership rather than being the object of study",
    "measurement_tool": "a drawn instrument bearing a graduated scale or dial from which a value is read — ruler, protractor, thermometer, clock face, speedometer, weighing scale, or pan balance — including a printed instrument sheet supplied with the item; choose over number_line when the scale belongs to a drawn instrument rather than a bare line, and over polygon, circle_figure, and angle_figure when the instrument is shown against a figure to be measured, even when the instrument fills only a small part of the image; choose picture for a device that displays digits rather than a scale, such as a calculator; a scale drawing takes the type of the figure it draws, its scale key being a legend rather than an instrument; choose the shape's own type when the instrument's outline rather than its reading is the object of study",
    "picture": "a drawing or photograph of real-world objects, or of plain counters standing for objects, in which any quantity is carried by a label, caption, or price tag attached to an object or by how many objects are drawn, rather than by a scale, a partition, or measured geometry; includes a map of locations or routes and a sheet of cards, tiles, or cutouts supplied as manipulatives; choose picture_graph when repeated icons in labeled categories give counts, array when identical objects fill equal rows and columns whose product is the count, measurement_tool when an object bears a graduated scale to be read, area_model when an object is partitioned into parts of a whole, solid_figure when the object is drawn as a geometric solid to be measured, pattern_visual when the drawing repeats a unit to be extended, and polygon or angle_figure when a schematic line drawing of an object carries lengths or angles that enter the computation",
    "text_only": "the image carries only the item's own text, equation, unruled list of numbers or ordered pairs, answer choices, or response controls — entry boxes, radio buttons, a grid of statements with selection buttons, or an empty frame with headings and no entries — with no diagram, graph, figure, or other visual model; choose this whenever nothing in the image would be lost by reading the text alone, but give a blank number line, coordinate grid, or expression frame offered as a workspace its own type, since its scale or slots are part of the representation",
}

visual_model_groups = {
    "polygon": "figural", "circle_figure": "figural", "angle_figure": "figural",
    "solid_figure": "figural", "net_diagram": "figural",
    "number_line": "quantity_model", "tape_diagram": "quantity_model",
    "area_model": "quantity_model", "array": "quantity_model",
    "coordinate_plane": "coordinate",
    "bar_graph": "data_display", "line_graph": "data_display",
    "picture_graph": "data_display", "dot_plot": "data_display",
    "histogram": "data_display", "circle_graph": "data_display",
    "scatter_plot": "data_display", "box_plot": "data_display",
    "stem_leaf_plot": "data_display",
    "table": "symbolic", "expression_frame": "symbolic",
    "picture": "depictive", "measurement_tool": "depictive",
    "pattern_visual": "sequential",
    "tree_diagram": "relational", "venn_diagram": "relational",
    "text_only": "no_visual",
}

group_names = sorted(set(visual_model_groups.values()))

visual_model_signs = {
    "expression_frame": "symbol",
    "polygon": "icon", "circle_figure": "icon", "angle_figure": "icon",
    "solid_figure": "icon", "net_diagram": "icon",
    "area_model": "icon", "array": "icon",
    "tape_diagram": "icon", "picture_graph": "icon", "measurement_tool": "icon",
    "picture": "icon", "pattern_visual": "icon",
    "number_line": "index", "coordinate_plane": "index", "dot_plot": "index",
    "scatter_plot": "index", "line_graph": "index",
    "bar_graph": "index", "histogram": "index", "box_plot": "index",
    "circle_graph": "index", "tree_diagram": "index",
    "table": "symbol", "stem_leaf_plot": "symbol", "venn_diagram": "symbol",
    "text_only": "none",
}

sign_names = sorted(set(visual_model_signs.values()))

visual_model_representations = {
    "number_line": "quantity_model", "tape_diagram": "quantity_model",
    "area_model": "quantity_model", "array": "quantity_model",
    "table": "tabular",
    "coordinate_plane": "coordinate_graph", "line_graph": "coordinate_graph",
    "scatter_plot": "coordinate_graph",
    "bar_graph": "data_display", "picture_graph": "data_display",
    "dot_plot": "data_display", "histogram": "data_display",
    "circle_graph": "data_display", "box_plot": "data_display",
    "stem_leaf_plot": "data_display",
    "polygon": "geometric_figure", "circle_figure": "geometric_figure",
    "angle_figure": "geometric_figure", "solid_figure": "geometric_figure",
    "net_diagram": "geometric_figure",
    "measurement_tool": "measurement_tool",
    "picture": "pictorial",
    "expression_frame": "other", "pattern_visual": "other",
    "tree_diagram": "other", "venn_diagram": "other",
    "text_only": "no_visual",
}

representation_names = sorted(set(visual_model_representations.values()))

visual_subtypes = {
    "number_line": ["double_number_line"],
    "table": ["ratio_table", "two_way_table", "function_table", "data_table"],
    "circle_graph": ["pie_chart", "spinner"],
    "area_model": ["fraction_circle", "grid_model"],
    "tape_diagram": ["fraction_strip"],
    "measurement_tool": ["ruler", "protractor", "clock", "thermometer",
                         "dial_scale", "pan_balance"],
    "tree_diagram": ["probability_tree", "factor_tree"],
}


def _is_set(value: Any) -> bool:
    """Read a flag that may have round-tripped through CSV or JSON as text."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "t", "y"}
    return bool(value)


def trim_margin(image: np.ndarray | bytes, margin: int = _TRIM_MARGIN
                ) -> np.ndarray | bytes:
    """Crop an image to its ink, leaving a fixed margin."""
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
    """Crop to a figure box; return the image and the region measured."""
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


def flags_by_representation(entry: dict[str, Any], prefix: str = "visual_representation") -> dict[str, int]:
    """Collapse per-type flags into representation categories."""
    hits = dict.fromkeys(representation_names, False)
    for m in visual_models:
        if _is_set(entry.get(m)):
            hits[visual_model_representations[m]] = True
    return {f"{prefix}_{f}": int(v) for f, v in hits.items()}


def flags_by_sign(entry: dict[str, Any]) -> dict[str, int]:
    """Collapse per-type boolean flags into one flag per Peircean sign class."""
    hits = dict.fromkeys(sign_names, False)
    for m in visual_models:
        if _is_set(entry.get(m)):
            hits[visual_model_signs[m]] = True
    return {f"visual_sign_{s}": int(v) for s, v in hits.items()}


class VisualFeatureExtractor:
    """Extract visual complexity features from assessment images."""

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
            ``structural_elements``, and ``frequency_domain``."""
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
        """One flat ``visual_*`` dict for an image."""
        nested = self.extract(image_source)
        return {key: safe_get(nested, *path, default=None)
                for key, path in self.vis_map.items()}

    @classmethod
    def aggregate_visual_features(
        cls, flat_list: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """One item-level dict from several per-image dicts."""
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
        """One greyscale definition for every metric in this class."""
        if image.ndim == 2:
            return image if image.dtype == np.uint8 else image.astype(np.uint8)
        image = image[:, :, :3]
        if cv2_available:
            return cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_BGR2GRAY)
        return np.mean(image, axis=2).astype(np.uint8)

    def _resize_to_work_scale(self, gray: np.ndarray) -> np.ndarray:
        """Put every image on one pixel scale before counting anything."""
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
        """Count lines, circles and closed shapes on the work-scale image."""
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
