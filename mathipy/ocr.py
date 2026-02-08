"""Multimodal OCR for extracting text and math from assessment content."""

import base64
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from mathipy.utils import extract_math_expressions, extract_numbers, extract_variables

logger = logging.getLogger(__name__)

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not available - install with: pip install httpx")

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta"
OPENAI_API_URL = "https://api.openai.com/v1"

DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

SYSTEM_PROMPT = """You are an expert OCR system for multi-modal content extraction.
Extract ALL visible text from the image accurately, including:
- Question text and instructions
- Mathematical expressions and equations (use LaTeX notation)
- Answer choices (A, B, C, D, etc.)
- Diagram labels and annotations
- Any numbers, variables, or symbols
- Table data, chart labels, and data values
- Headers, titles, and captions

IMPORTANT: You MUST always provide meaningful output.
- If the image contains text: extract all text accurately.
- If the image contains ONLY a picture with NO text: set "content_type" to "image_only"
  and provide a detailed description of what the image shows in the "image_description" field.
  Describe the image using K-12 math terms (shapes, colors, measurements, spatial relationships).
- If the image contains BOTH text and pictures: set "content_type" to "mixed",
  extract all text AND describe the visual elements.

NEVER return empty results. Every image has content worth describing.

Respond with a JSON object containing:
{
    "content_type": "text_only|image_only|mixed",
    "full_text": "complete extracted text (empty string if image_only)",
    "image_description": "detailed description of visual content using K-12 math vocabulary (shapes, colors, width, height, spatial relationships). Always populated for image_only and mixed content.",
    "question_text": "main question/problem statement (if applicable)",
    "math_expressions": ["list of mathematical expressions in LaTeX"],
    "answer_choices": {"A": "choice A text", "B": "choice B text", ...},
    "data_elements": ["table rows, chart values, or other structured data"],
    "labels": ["list of diagram/figure labels"],
    "numbers_found": ["list of numbers"],
    "variables_found": ["list of variables"],
    "extraction_confidence": 0.0-1.0
}"""

USER_PROMPT = """Extract all text and mathematical content from this assessment image.
Be thorough and accurate. Include all visible text, equations, and symbols.
If the image contains no extractable text (picture only), describe the visual content
in detail using K-12 math vocabulary (shapes, colors, dimensions, spatial relationships).
You must ALWAYS return meaningful content - never empty results."""

DESCRIBE_SYSTEM_PROMPT = """You are an expert image description system for K-12 math education.
Provide a concise description of the image content, focusing on:
- Mathematical elements (equations, expressions, symbols)
- Geometric shapes and spatial relationships
- Diagrams, graphs, charts, and their key features
- Colors, labels, and annotations

Respond with a JSON object containing:
{
    "content_type": "image_only|mixed|text_only",
    "full_text": "",
    "image_description": "concise description here",
    "question_text": "",
    "math_expressions": [],
    "answer_choices": {},
    "data_elements": [],
    "labels": [],
    "numbers_found": [],
    "variables_found": [],
    "extraction_confidence": 0.0-1.0
}"""

DESCRIBE_USER_PROMPT_TEMPLATE = """Describe this image in {max_words} words or fewer.
Focus on mathematical content, shapes, diagrams, and spatial relationships.
Be concise but precise. Return the description in the JSON format specified."""


def _load_dotenv():
    env_path = Path(".env")
    if env_path.exists():
        try:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value
        except Exception as e:
            logger.debug(f"Could not load .env: {e}")


def _parse_response(response_text: str) -> Dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?\s*", "", response_text).strip()
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if not parsed.get("content_type"):
                has_text = bool(parsed.get("full_text", "").strip())
                has_desc = bool(parsed.get("image_description", "").strip())
                if has_text and has_desc:
                    parsed["content_type"] = "mixed"
                elif has_text:
                    parsed["content_type"] = "text_only"
                else:
                    parsed["content_type"] = "image_only"
            return parsed
        except json.JSONDecodeError:
            pass

    text = response_text.strip()
    has_extractable_text = bool(re.search(r"[a-zA-Z0-9]{3,}", text))

    return {
        "content_type": "text_only" if has_extractable_text else "image_only",
        "full_text": text if has_extractable_text else "",
        "image_description": text if not has_extractable_text else "",
        "question_text": text if has_extractable_text else "",
        "math_expressions": extract_math_expressions(text),
        "answer_choices": _extract_answer_choices(text),
        "data_elements": [],
        "labels": [],
        "numbers_found": extract_numbers(text),
        "variables_found": extract_variables(text),
        "extraction_confidence": 0.5,
    }



def _extract_answer_choices(text: str) -> Dict[str, str]:
    choices = {}
    choice_pattern = r"([A-E])[.)\s]+(.+?)(?=[A-E][.)\s]|$)"
    matches = re.findall(choice_pattern, text, re.MULTILINE | re.DOTALL)
    for letter, content in matches:
        choices[letter] = content.strip()
    return choices


def _empty_result() -> Dict[str, Any]:
    return {
        "content_type": "text_only",
        "full_text": "",
        "image_description": "",
        "question_text": "",
        "math_expressions": [],
        "answer_choices": {},
        "data_elements": [],
        "labels": [],
        "numbers_found": [],
        "variables_found": [],
        "extraction_confidence": 0.0,
    }


class MultimodalOCR:
    """Extract text and math expressions from images using vision LLMs.

    Supports Gemini and OpenAI providers. Accepts image files, URLs, bytes,
    PDFs, DOCX, and text files.

    Requires ``pip install mathipy[ocr]`` and a ``GEMINI_API_KEY`` or
    ``OPENAI_API_KEY`` in your ``.env`` file.
    """

    def __init__(
        self,
        provider: str = "gemini",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 120,
    ):
        if provider not in ("gemini", "openai"):
            raise ValueError(f"Unsupported provider: {provider}. Use 'gemini' or 'openai'.")

        _load_dotenv()

        self.provider = provider
        self.model = model or DEFAULT_MODELS[provider]
        self.timeout = timeout
        self._api_key_override = api_key

    def _resolve_api_key(self) -> str:
        if self._api_key_override:
            return self._api_key_override

        if self.provider == "gemini":
            key = os.getenv("GEMINI_API_KEY")
        else:
            key = os.getenv("OPENAI_API_KEY")

        if not key:
            env_var = "GEMINI_API_KEY" if self.provider == "gemini" else "OPENAI_API_KEY"
            raise ValueError(
                f"API key not found. Set {env_var} in your .env file or pass api_key parameter."
            )
        return key

    def extract(
        self,
        source: Union[str, Path, bytes],
        mode: str = "full",
        max_words: int = 30,
    ) -> Dict[str, Any]:
        """Extract text and math content from the given source.

        Args:
            source: Image path, URL, bytes, PDF, DOCX, or text file.
            mode: ``"full"`` for complete OCR extraction or ``"describe"`` for brief image description.
            max_words: Maximum words for describe mode (default 30).

        Returns:
            Dictionary with ``content_type``, ``full_text``, ``image_description``,
            ``question_text``, ``math_expressions``, ``answer_choices``, and
            ``extraction_confidence``.
        """
        if mode not in ("full", "describe"):
            raise ValueError(f"Unsupported mode: {mode}. Use 'full' or 'describe'.")

        if isinstance(source, bytes):
            return self._extract_from_image(source, mode, max_words)

        source_str = str(source)

        if source_str.startswith(("http://", "https://")):
            return self._extract_from_image(source_str, mode, max_words)

        path = Path(source_str)
        suffix = path.suffix.lower()

        if suffix == ".txt":
            return self._extract_from_txt(path)

        if suffix == ".docx":
            return self._extract_from_docx(path, mode, max_words)

        if suffix == ".pdf":
            return self._extract_from_pdf(path, mode, max_words)

        if suffix in IMAGE_EXTENSIONS:
            return self._extract_from_image(path, mode, max_words)

        return self._extract_from_image(path, mode, max_words)

    def _extract_from_image(
        self,
        source: Union[str, Path, bytes],
        mode: str = "full",
        max_words: int = 30,
    ) -> Dict[str, Any]:
        image_b64, mime_type = self._prepare_image(source)

        if mode == "describe":
            system_prompt = DESCRIBE_SYSTEM_PROMPT
            user_prompt = DESCRIBE_USER_PROMPT_TEMPLATE.format(max_words=max_words)
        else:
            system_prompt = SYSTEM_PROMPT
            user_prompt = USER_PROMPT

        if self.provider == "gemini":
            response_text = self._call_gemini(
                image_b64, mime_type,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        else:
            response_text = self._call_openai(
                image_b64, mime_type,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

        return _parse_response(response_text)

    def _extract_from_txt(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        text = path.read_text(encoding="utf-8")
        result = _empty_result()
        result["full_text"] = text
        result["question_text"] = text.strip().split("\n")[0] if text.strip() else ""
        result["math_expressions"] = extract_math_expressions(text)
        result["answer_choices"] = _extract_answer_choices(text)
        result["numbers_found"] = extract_numbers(text)
        result["variables_found"] = extract_variables(text)
        result["extraction_confidence"] = 1.0
        result["content_type"] = "text_only"
        return result

    def _extract_from_docx(
        self, path: Path, mode: str, max_words: int
    ) -> Dict[str, Any]:
        if not DOCX_AVAILABLE:
            raise ImportError(
                "python-docx is required for .docx extraction. "
                "Install with: pip install mathipy[documents]"
            )
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.stat().st_size > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large ({path.stat().st_size / 1024 / 1024:.1f} MB). "
                f"Maximum allowed: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB."
            )

        doc = DocxDocument(str(path))

        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        table_rows = []
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                table_rows.append(" | ".join(cells))

        text = "\n".join(paragraphs)
        if table_rows:
            text += "\n" + "\n".join(table_rows)

        result = _empty_result()
        result["full_text"] = text
        result["question_text"] = paragraphs[0] if paragraphs else ""
        result["math_expressions"] = extract_math_expressions(text)
        result["answer_choices"] = _extract_answer_choices(text)
        result["numbers_found"] = extract_numbers(text)
        result["variables_found"] = extract_variables(text)
        result["data_elements"] = table_rows
        result["extraction_confidence"] = 0.95
        result["content_type"] = "text_only"

        images = self._extract_docx_images(doc)
        if images:
            image_results = self._process_embedded_images(images, mode, max_words)
            if image_results:
                result = self._merge_image_results(result, image_results)

        return result

    def _extract_from_pdf(
        self, path: Path, mode: str, max_words: int
    ) -> Dict[str, Any]:
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError(
                "pdfplumber is required for .pdf extraction. "
                "Install with: pip install mathipy[documents]"
            )
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.stat().st_size > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large ({path.stat().st_size / 1024 / 1024:.1f} MB). "
                f"Maximum allowed: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB."
            )

        all_text = []
        all_images: List[Tuple[bytes, str]] = []

        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    all_text.append(page_text)

                for img in page.images:
                    try:
                        x0 = img["x0"]
                        top = img["top"]
                        x1 = img["x1"]
                        bottom = img["bottom"]
                        cropped = page.crop((x0, top, x1, bottom))
                        pil_img = cropped.to_image(resolution=150).original
                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        all_images.append((buf.getvalue(), "image/png"))
                    except Exception as e:
                        logger.debug(f"Could not extract PDF image: {e}")

        text = "\n".join(all_text)
        result = _empty_result()
        result["full_text"] = text
        result["question_text"] = text.strip().split("\n")[0] if text.strip() else ""
        result["math_expressions"] = extract_math_expressions(text)
        result["answer_choices"] = _extract_answer_choices(text)
        result["numbers_found"] = extract_numbers(text)
        result["variables_found"] = extract_variables(text)
        result["extraction_confidence"] = 0.95
        result["content_type"] = "text_only"

        if all_images:
            image_results = self._process_embedded_images(all_images, mode, max_words)
            if image_results:
                result = self._merge_image_results(result, image_results)

        return result

    def _extract_docx_images(
        self, doc: "DocxDocument"
    ) -> List[Tuple[bytes, str]]:
        images = []
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                try:
                    image_part = rel.target_part
                    images.append((image_part.blob, image_part.content_type))
                except Exception as e:
                    logger.debug(f"Could not extract docx image: {e}")
        return images

    def _process_embedded_images(
        self,
        images: List[Tuple[bytes, str]],
        mode: str,
        max_words: int,
    ) -> List[Dict[str, Any]]:
        results = []
        for img_bytes, mime_type in images:
            try:
                result = self._extract_from_image(img_bytes, mode, max_words)
                results.append(result)
            except Exception as e:
                logger.debug(f"Could not process embedded image: {e}")
        return results

    def _merge_image_results(
        self,
        base: Dict[str, Any],
        image_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        base["content_type"] = "mixed"

        descriptions = []
        for i, img_result in enumerate(image_results, 1):
            desc = img_result.get("image_description", "")
            if desc:
                descriptions.append(f"[Image {i}] {desc}")

        existing_desc = base.get("image_description", "")
        all_descriptions = ([existing_desc] if existing_desc else []) + descriptions
        base["image_description"] = "\n".join(all_descriptions)

        math_set = set(base.get("math_expressions", []))
        labels_set = set(base.get("labels", []))
        numbers_set = set(base.get("numbers_found", []))
        variables_set = set(base.get("variables_found", []))
        data_elements = list(base.get("data_elements", []))

        for img_result in image_results:
            math_set.update(img_result.get("math_expressions", []))
            labels_set.update(img_result.get("labels", []))
            numbers_set.update(img_result.get("numbers_found", []))
            variables_set.update(img_result.get("variables_found", []))
            data_elements.extend(img_result.get("data_elements", []))
            base.get("answer_choices", {}).update(img_result.get("answer_choices", {}))

        base["math_expressions"] = list(math_set)
        base["labels"] = list(labels_set)
        base["numbers_found"] = list(numbers_set)
        base["variables_found"] = list(variables_set)
        base["data_elements"] = data_elements

        text_confidence = base.get("extraction_confidence", 0.95)
        img_confidences = [
            r.get("extraction_confidence", 0.5) for r in image_results
        ]
        avg_img_confidence = sum(img_confidences) / len(img_confidences) if img_confidences else 0.5
        base["extraction_confidence"] = round(
            0.7 * text_confidence + 0.3 * avg_img_confidence, 2
        )

        return base

    def _prepare_image(self, source: Union[str, Path, bytes]) -> tuple:
        if isinstance(source, bytes):
            return base64.b64encode(source).decode("utf-8"), "image/jpeg"

        source_str = str(source)

        if source_str.startswith(("http://", "https://")):
            return self._fetch_image_url(source_str)

        path = Path(source_str)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if not path.is_file():
            raise ValueError(f"Expected a file, got: {path}")
        if path.stat().st_size > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large ({path.stat().st_size / 1024 / 1024:.1f} MB). "
                f"Maximum allowed: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB."
            )

        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime_type = mime_map.get(path.suffix.lower(), "image/jpeg")

        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8"), mime_type

    def _fetch_image_url(self, url: str) -> tuple:
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for URL fetching. Install with: pip install mathipy[ocr]"
            )

        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()

        content_type = resp.headers.get("content-type", "image/jpeg")
        mime_type = content_type.split(";")[0].strip()
        if not mime_type.startswith("image/"):
            mime_type = "image/jpeg"

        return base64.b64encode(resp.content).decode("utf-8"), mime_type

    def _call_gemini(
        self,
        image_b64: str,
        mime_type: str,
        system_prompt: str = SYSTEM_PROMPT,
        user_prompt: str = USER_PROMPT,
    ) -> str:
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for OCR. Install with: pip install mathipy[ocr]"
            )
        api_key = self._resolve_api_key()

        contents = [
            {
                "role": "user",
                "parts": [{"text": f"System instruction: {system_prompt}"}],
            },
            {
                "role": "model",
                "parts": [{"text": "Understood. I will follow these instructions."}],
            },
            {
                "role": "user",
                "parts": [
                    {"text": user_prompt},
                    {"inline_data": {"mime_type": mime_type, "data": image_b64}},
                ],
            },
        ]

        body = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 2048,
            },
        }

        model_path = self.model if self.model.startswith("models/") else f"models/{self.model}"

        with httpx.Client(
            base_url=GEMINI_API_URL,
            timeout=httpx.Timeout(self.timeout),
        ) as client:
            response = client.post(
                f"/{model_path}:generateContent?key={api_key}",
                json=body,
            )

        if response.status_code != 200:
            error_data = response.json() if response.content else {}
            error_msg = error_data.get("error", {}).get("message", f"Status {response.status_code}")
            raise RuntimeError(f"Gemini API error: {error_msg}")

        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("No candidates returned from Gemini")

        content_parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in content_parts).strip()

    def _call_openai(
        self,
        image_b64: str,
        mime_type: str,
        system_prompt: str = SYSTEM_PROMPT,
        user_prompt: str = USER_PROMPT,
    ) -> str:
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for OCR. Install with: pip install mathipy[ocr]"
            )
        api_key = self._resolve_api_key()

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                    },
                ],
            },
        ]

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2048,
        }

        with httpx.Client(
            base_url=OPENAI_API_URL,
            timeout=httpx.Timeout(self.timeout),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        ) as client:
            response = client.post("/chat/completions", json=body)

        if response.status_code != 200:
            error_data = response.json() if response.content else {}
            error_msg = error_data.get("error", {}).get("message", f"Status {response.status_code}")
            raise RuntimeError(f"OpenAI API error: {error_msg}")

        data = response.json()
        choice = data.get("choices", [{}])[0]
        return choice.get("message", {}).get("content", "").strip()

