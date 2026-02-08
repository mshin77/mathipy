"""Visual feature extraction from assessment images."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    logger.warning("Pillow not available - install with: pip install pillow")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - install with: pip install opencv-python-headless")

VISUAL_MODELS = [
    "number_line", "coordinate_plane", "bar_graph", "line_graph",
    "pictograph", "dot_plot", "histogram", "circle_graph", "scatter_plot",
    "table", "geometric_figure", "solid_figure", "area_model", "array",
    "tape_diagram", "fraction_model", "pattern_visual", "tree_diagram",
    "picture", "other",
]


class VisualFeatureExtractor:
    """Extract visual complexity features from assessment images.

    Analyzes dimensions, pixel statistics, edge metrics, structural elements
    (lines, circles, shapes), frequency domain, and overall complexity score.

    Requires ``pip install mathipy[vision]`` for full features.
    """

    def __init__(self):
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        if not PILLOW_AVAILABLE:
            logger.warning("Image loading requires Pillow")
        if not CV2_AVAILABLE:
            logger.warning("Advanced features require OpenCV")

    def extract(self, image_source: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """Extract visual features from an image.

        Args:
            image_source: File path, Path object, or numpy array.

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

        if CV2_AVAILABLE:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            features["edge_metrics"] = self._extract_edge_metrics(gray)
            features["structural_elements"] = self._extract_structural_elements(gray)
            features["frequency_domain"] = self._extract_frequency_features(gray)

        features["complexity_score"] = self._calculate_complexity(features)

        return features

    def _load_image(self, source: Union[str, Path, np.ndarray]) -> Optional[np.ndarray]:
        if isinstance(source, np.ndarray):
            return source

        if not PILLOW_AVAILABLE and not CV2_AVAILABLE:
            logger.error("Cannot load image - no image library available")
            return None

        path = Path(source) if isinstance(source, str) else source
        if not path.exists():
            logger.error(f"Image not found: {path}")
            return None

        if CV2_AVAILABLE:
            return cv2.imread(str(path))
        elif PILLOW_AVAILABLE:
            pil_image = Image.open(path)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            return np.array(pil_image)[:, :, ::-1]  # RGB to BGR

        return None

    def _extract_dimensions(self, image: np.ndarray) -> Dict[str, Any]:
        h, w = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1

        return {
            "width": w,
            "height": h,
            "aspect_ratio": round(w / h, 3) if h > 0 else 0.0,
            "total_pixels": w * h,
            "channels": channels,
        }

    def _extract_pixel_stats(self, image: np.ndarray) -> Dict[str, float]:
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        return {
            "mean": float(np.mean(gray)),
            "std": float(np.std(gray)),
            "min": float(np.min(gray)),
            "max": float(np.max(gray)),
            "median": float(np.median(gray)),
            "contrast": float(np.max(gray) - np.min(gray)),
        }

    def _extract_edge_metrics(self, gray: np.ndarray) -> Dict[str, float]:
        if not CV2_AVAILABLE:
            return {}

        v = np.median(gray)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(gray, lower, upper)

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

    def _extract_structural_elements(self, gray: np.ndarray) -> Dict[str, Any]:
        if not CV2_AVAILABLE:
            return {}

        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 100,
            minLineLength=50, maxLineGap=10
        )

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30,
            minRadius=10, maxRadius=100
        )

        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
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

    def _extract_frequency_features(self, gray: np.ndarray) -> Dict[str, float]:
        if not CV2_AVAILABLE:
            return {}

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

    def _calculate_complexity(self, features: Dict[str, Any]) -> Dict[str, Any]:
        scores = []

        edge = features.get("edge_metrics", {})
        if edge:
            edge_score = min(1.0, edge.get("edge_ratio", 0) * 10)
            scores.append(edge_score)

        struct = features.get("structural_elements", {})
        if struct:
            shape_score = min(1.0, struct.get("total_shapes", 0) / 20)
            scores.append(shape_score)

        freq = features.get("frequency_domain", {})
        if freq:
            high_freq = freq.get("high_freq_ratio", 0)
            scores.append(min(1.0, high_freq * 2))

        overall = sum(scores) / len(scores) if scores else 0.5

        return {
            "overall": overall,
            "level": "low" if overall < 0.3 else "medium" if overall < 0.6 else "high",
            "component_scores": {
                "edge": scores[0] if len(scores) > 0 else 0,
                "shape": scores[1] if len(scores) > 1 else 0,
                "frequency": scores[2] if len(scores) > 2 else 0,
            },
        }

    def _empty_features(self) -> Dict[str, Any]:
        return {
            "dimensions": {},
            "pixel_statistics": {},
            "edge_metrics": {},
            "structural_elements": {},
            "frequency_domain": {},
            "complexity_score": {"overall": 0, "level": "unknown", "component_scores": {}},
        }
