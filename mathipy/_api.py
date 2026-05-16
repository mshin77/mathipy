"""Base class for vision API clients (Gemini / OpenAI)."""

from __future__ import annotations

import base64
import importlib
import ipaddress
import logging
import os
import re
import socket
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _optional_import(name: str, install_hint: str | None = None):
    try:
        return importlib.import_module(name), True
    except ImportError:
        if install_hint:
            logger.warning(f"{name} not available - install with: pip install {install_hint}")
        return None, False


httpx, httpx_available = _optional_import("httpx")

gemini_api_url = "https://generativelanguage.googleapis.com/v1beta"
openai_api_url = "https://api.openai.com/v1"

default_models = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o",
}

image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}

max_file_size = 20 * 1024 * 1024  # 20 MB

_secret_pattern = re.compile(
    r'AIza[A-Za-z0-9_-]{30,}|sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}'
)


def _sanitize_error(msg: str) -> str:
    return _secret_pattern.sub("***", msg)


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
                f"API key not found. Set {env_var} in the .env file or pass api_key parameter."
            )
        return key

    def _prepare_image(self, source: str | Path | bytes) -> tuple:
        if isinstance(source, bytes):
            return base64.b64encode(source).decode("utf-8"), "image/jpeg"

        source_str = str(source)

        if source_str.startswith("http://"):
            raise ValueError("Only https:// URLs are allowed for image fetch")
        if source_str.startswith("https://"):
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

    def _validate_image_url(self, url: str) -> None:
        # reject internal/private/loopback addresses
        parsed = urlparse(url)
        if parsed.scheme != "https":
            raise ValueError(f"Only https:// URLs allowed, got scheme: {parsed.scheme}")
        if not parsed.hostname:
            raise ValueError("URL has no hostname")
        try:
            addr_info = socket.getaddrinfo(parsed.hostname, None)
        except socket.gaierror as e:
            raise ValueError(f"Cannot resolve host: {e}") from None
        for _af, _kind, _proto, _canon, sockaddr in addr_info:
            ip = ipaddress.ip_address(sockaddr[0])
            if (ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
                raise ValueError(f"URL resolves to disallowed address: {ip}")

    def _fetch_image_url(self, url: str) -> tuple:
        if not httpx_available:
            raise ImportError(
                "httpx is required for URL fetching. Install with: pip install mathipy[ocr]"
            )
        self._validate_image_url(url)
        with httpx.Client(timeout=self.timeout, follow_redirects=False) as client:
            resp = client.get(url)
            resp.raise_for_status()
        declared = int(resp.headers.get("content-length", "0") or 0)
        if declared > max_file_size or len(resp.content) > max_file_size:
            raise ValueError(
                f"Image too large (max {max_file_size // 1024 // 1024} MB)"
            )
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
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
        ) as client:
            response = client.post(
                f"/{model_path}:generateContent",
                json=body,
            )

        if response.status_code != 200:
            error_data = response.json() if response.content else {}
            error_msg = error_data.get("error", {}).get("message", f"Status {response.status_code}")
            raise RuntimeError(f"Gemini API error: {_sanitize_error(error_msg)}")

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
            raise RuntimeError(f"OpenAI API error: {_sanitize_error(error_msg)}")

        data = response.json()
        choice = data.get("choices", [{}])[0]
        return choice.get("message", {}).get("content", "").strip()
