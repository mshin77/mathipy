"""Base class for vision API clients (Gemini / OpenAI)."""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import httpx
    httpx_available = True
except ImportError:
    httpx_available = False

gemini_api_url = "https://generativelanguage.googleapis.com/v1beta"
openai_api_url = "https://api.openai.com/v1"

default_models = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o",
}

image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}

max_file_size = 20 * 1024 * 1024  # 20 MB


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
            logger.warning(f"Could not load .env: {e}")


class VisionAPIClient:
    """Shared base for classes that call Gemini or OpenAI vision endpoints."""

    def __init__(
        self,
        provider: str = "gemini",
        model: str | None = None,
        api_key: str | None = None,
        timeout: int = 120,
    ):
        if provider not in ("gemini", "openai"):
            raise ValueError(f"Unsupported provider: {provider}. Use 'gemini' or 'openai'.")

        _load_dotenv()

        self.provider = provider
        self.model = model or default_models[provider]
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

    def _prepare_image(self, source: str | Path | bytes) -> tuple:
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
        if path.stat().st_size > max_file_size:
            raise ValueError(
                f"File too large ({path.stat().st_size / 1024 / 1024:.1f} MB). "
                f"Maximum allowed: {max_file_size / 1024 / 1024:.0f} MB."
            )

        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif",
            ".webp": "image/webp", ".bmp": "image/bmp",
        }
        mime_type = mime_map.get(path.suffix.lower(), "image/jpeg")

        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8"), mime_type

    def _fetch_image_url(self, url: str) -> tuple:
        if not httpx_available:
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
        system_prompt: str = "",
        user_prompt: str = "",
    ) -> str:
        if not httpx_available:
            raise ImportError("httpx is required. Install with: pip install mathipy[ocr]")
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
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048},
        }

        model_path = self.model if self.model.startswith("models/") else f"models/{self.model}"

        with httpx.Client(
            base_url=gemini_api_url,
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
        system_prompt: str = "",
        user_prompt: str = "",
    ) -> str:
        if not httpx_available:
            raise ImportError("httpx is required. Install with: pip install mathipy[ocr]")
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
            base_url=openai_api_url,
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
