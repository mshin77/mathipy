"""Base class for vision API clients (Gemini / OpenAI)."""

from __future__ import annotations

import base64
import importlib
import io
import ipaddress
import logging
import os
import re
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _optional_import(name: str, install_hint: str | None = None):
    try:
        return importlib.import_module(name), True
    except ImportError:
        if install_hint:
            logger.debug(f"{name} not available - install with: pip install {install_hint}")
        return None, False


httpx, httpx_available = _optional_import("httpx")
anthropic_sdk, anthropic_available = _optional_import("anthropic", "anthropic")

gemini_api_url = "https://generativelanguage.googleapis.com/v1beta"
openai_api_url = "https://api.openai.com/v1"
anthropic_api_url = "https://api.anthropic.com"

default_models = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-2024-08-06",
    "anthropic": "claude-opus-5-5",
}
default_urls = {"gemini": gemini_api_url, "openai": openai_api_url, "anthropic": anthropic_api_url}
api_key_vars = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
# models that refuse sampling settings take None for both
default_sampling = {"gemini": (0.0, 0), "openai": (0.0, 0), "anthropic": (None, None)}
_unset = object()

image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
_nat64 = ipaddress.ip_network("64:ff9b::/96")

max_file_size = 20 * 1024 * 1024  # 20 MB
min_image_px = 512
max_image_px = 2048

_secret_pattern = re.compile(
    r'AIza[A-Za-z0-9_-]{30,}|sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}'
)


_RETRY_STATUS = {429, 500, 502, 503, 504}
_UNAVAILABLE_STATUS = _RETRY_STATUS | {401, 403, 404}
_REFUSAL_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION",
                    "content_filter"}


class ProviderUnavailableError(RuntimeError):
    """A call that failed for reasons other than the provider declining the request."""


_RETRY_ATTEMPTS = 3
_RETRY_AFTER_STATUS = {429, 503}
_RETRY_AFTER_CAP = 30


def _retry_wait(response, attempt):
    header = (response.headers.get("retry-after", "")
              if response.status_code in _RETRY_AFTER_STATUS else "").strip()
    return min(int(header), _RETRY_AFTER_CAP) if header.isdigit() else 2 ** attempt


def _post_retrying(client, path, body):
    """Retry a rate limit or a gateway failure; a 4xx is a request problem and stands."""
    for attempt in range(_RETRY_ATTEMPTS):
        last = attempt + 1 == _RETRY_ATTEMPTS
        try:
            response = client.post(path, json=body)
        except httpx.TransportError as err:
            if last:
                raise ProviderUnavailableError(f"transport failure: {type(err).__name__}") from err
            time.sleep(2 ** attempt)
            continue
        if response.status_code not in _RETRY_STATUS or last:
            return response
        time.sleep(_retry_wait(response, attempt))


def _sanitize_error(msg: str) -> str:
    return _secret_pattern.sub("***", msg)


def _unavailable(response) -> bool:
    return (response.status_code in _UNAVAILABLE_STATUS
            or (response.status_code == 400 and "api key" in _error_message(response).lower()))


def _error_message(response) -> str:
    try:
        err = response.json().get("error")
    except (ValueError, AttributeError):
        err = None
    msg = err.get("message") if isinstance(err, dict) else err
    return _sanitize_error(str(msg or f"Status {response.status_code}"))


def _json_body(response, provider: str) -> dict:
    try:
        data = response.json()
    except ValueError as err:
        raise ProviderUnavailableError(f"{provider} API error: unreadable response body") from err
    if not isinstance(data, dict):
        raise ProviderUnavailableError(f"{provider} API error: unexpected response body")
    return data


def _dotenv_value(raw: str) -> str:
    """The value of a .env line: the text inside matching quotes, otherwise the text
    before an inline `` #`` comment."""
    raw = raw.strip()
    quoted = re.fullmatch(r"""(['"])(.*?)\1(?:\s+#.*)?""", raw)
    if quoted:
        return quoted.group(2)
    return re.split(r"\s+#", raw, maxsplit=1)[0].strip()


def _load_dotenv():
    env_path = Path(".env")
    if env_path.exists():
        try:
            with open(env_path, encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = re.sub(r"^export\s+", "", key.strip())
                        value = _dotenv_value(value)
                        if key and key not in os.environ:
                            os.environ[key] = value
        except Exception as e:
            logger.warning(f"Could not load .env: {e}")


def _upscale_small(data: bytes) -> tuple[bytes, str]:
    """Enlarge an image whose short side falls below ``min_image_px``, and re-encode
    formats the vision APIs reject as PNG. Returns the bytes and their MIME type, or
    "" for the type when the image cannot be decoded."""
    try:
        from PIL import Image
    except ImportError:
        return data, ""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception:
        return data, ""

    fmt = {"PNG": "image/png", "JPEG": "image/jpeg", "GIF": "image/gif",
           "WEBP": "image/webp"}.get((image.format or "").upper())
    try:
        from PIL import ImageOps
        image = ImageOps.exif_transpose(image)  # a re-encoded copy carries no EXIF rotation
    except Exception:
        pass
    short_side, long_side = min(image.size), max(image.size)
    scale = min(min_image_px / short_side, max_image_px / long_side) if short_side else 1
    upscale = 0 < short_side < min_image_px and scale > 1
    has_alpha = (image.mode in ("RGBA", "LA", "PA")
                 or (image.mode == "P" and "transparency" in image.info))
    if fmt and not upscale and not has_alpha:
        return data, fmt
    # Formats the vision APIs reject (TIFF, BMP, ...) are re-encoded as PNG, shrunk
    # toward max_image_px on the long side (an uncompressed scan can be very large)
    # but never below min_image_px on the short side. An accepted format re-encoded
    # only to flatten its alpha keeps its size.
    if fmt and not upscale:
        scale = 1
    elif not upscale:
        scale = max(min(1, max_image_px / long_side), min(1, min_image_px / short_side)) \
            if short_side else 1
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    if has_alpha:
        rgba = image.convert("RGBA")
        image = Image.new("RGB", rgba.size, "white")
        image.paste(rgba, mask=rgba.getchannel("A"))
    elif image.mode.startswith("I;16") or image.mode in ("I", "F"):
        # convert("RGB") clips these to 0-255; rescale so a 16-bit scan keeps its contrast.
        import numpy as np
        values = np.nan_to_num(np.asarray(image, dtype=np.float64))
        peak = float(values.max()) if values.size else 0.0
        if image.mode == "F":
            top = 1.0 if peak <= 1.0 else max(peak, 255.0)
        else:
            top = 65535.0 if image.mode.startswith("I;16") or peak > 255 else 255.0
        image = Image.fromarray((np.clip(values, 0, top) * 255.0 / top).round().astype(np.uint8))
    elif image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.resize(size, Image.LANCZOS).save(buffer, format="PNG")
    logger.debug("upscaled %s to %s before the vision call", image.size, size)
    return buffer.getvalue(), "image/png"


class VisionAPIClient:
    """Shared base for classes that call Gemini or OpenAI vision endpoints."""

    def __init__(
        self,
        provider: str = "gemini",
        model: str | None = None,
        api_key: str | None = None,
        timeout: int = 120,
        base_url: str | None = None,
        temperature: float | None = _unset,
        seed: int | None = _unset,
    ):
        if provider not in default_models:
            raise ValueError(f"Unsupported provider: {provider}. Use one of {sorted(default_models)}.")

        _load_dotenv()

        self.provider = provider
        self.model = model or default_models[provider]
        self.timeout = timeout
        self.temperature = default_sampling[provider][0] if temperature is _unset else temperature
        self.seed = default_sampling[provider][1] if seed is _unset else seed
        self.served: dict[str, str | None] = {}
        self._api_key_override = api_key
        self.base_url = base_url or default_urls[provider]

    def provenance(self) -> dict[str, str]:
        """Return the provider, model, sampling settings, and UTC timestamp."""
        return {
            "provider": self.provider,
            "model": self.model,
            "served_model": self.served.get("model"),
            "system_fingerprint": self.served.get("fingerprint"),
            "temperature": self.temperature,
            "seed": self.seed,
            "base_url": self.base_url,
            "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def _resolve_api_key(self) -> str:
        if self._api_key_override:
            return self._api_key_override

        env_var = api_key_vars[self.provider]
        key = os.getenv(env_var)
        if not key:
            raise ValueError(
                f"API key not found. Set {env_var} in the .env file or pass api_key parameter."
            )
        return key

    def _prepare_image(self, source: str | Path | bytes) -> tuple:
        if isinstance(source, bytes):
            data, mime_type = _upscale_small(source)
            return base64.b64encode(data).decode("utf-8"), mime_type or "image/jpeg"

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
            data, detected_mime = _upscale_small(f.read())
        mime_type = detected_mime or mime_type
        return base64.b64encode(data).decode("utf-8"), mime_type

    def _validate_image_url(self, url: str) -> str:
        """Check every address the host resolves to and return the first, so the fetch
        connects to an address that was checked rather than resolving the name again."""
        parsed = urlparse(url)
        if parsed.scheme != "https":
            raise ValueError(f"Only https:// URLs allowed, got scheme: {parsed.scheme}")
        if not parsed.hostname:
            raise ValueError("URL has no hostname")
        try:
            addr_info = socket.getaddrinfo(parsed.hostname, None)
        except socket.gaierror as e:
            raise ValueError(f"Cannot resolve host: {e}") from None
        addresses = [ipaddress.ip_address(info[4][0].split("%")[0]) for info in addr_info]
        if not addresses:
            raise ValueError("Cannot resolve host: no addresses")
        for address in addresses:
            ip = address
            if ip.version == 6 and ip.ipv4_mapped:
                ip = ip.ipv4_mapped
            if ip.version == 6 and ip in _nat64:
                ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
            if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                    or ip.is_multicast or ip.is_unspecified or not ip.is_global):
                raise ValueError(f"URL resolves to disallowed address: {ip}")
        return str(addresses[0])

    def _fetch_image_url(self, url: str) -> tuple:
        if not httpx_available:
            raise ImportError(
                "httpx is required for URL fetching. Install with: pip install mathipy[ocr]"
            )
        address = self._validate_image_url(url)
        request_url, headers, extensions = url, {}, {}
        if address:
            parsed = urlparse(url)
            host = f"[{address}]" if ":" in address else address
            netloc = f"{host}:{parsed.port}" if parsed.port else host
            request_url = parsed._replace(netloc=netloc).geturl()
            headers = {"Host": parsed.netloc.rsplit("@", 1)[-1]}
            extensions = {"sni_hostname": parsed.hostname}
        too_large = ValueError(f"Image too large (max {max_file_size // 1024 // 1024} MB)")
        with httpx.Client(timeout=self.timeout, follow_redirects=False) as client:
            with client.stream("GET", request_url, headers=headers,
                               extensions=extensions) as resp:
                resp.raise_for_status()
                if int(resp.headers.get("content-length", "0") or 0) > max_file_size:
                    raise too_large
                chunks, size = [], 0
                for chunk in resp.iter_bytes():
                    size += len(chunk)
                    if size > max_file_size:
                        raise too_large
                    chunks.append(chunk)
        content, detected = _upscale_small(b"".join(chunks))
        mime_type = resp.headers.get("content-type", "").split(";")[0].strip()
        mime_type = detected or (mime_type if mime_type.startswith("image/") else "image/jpeg")
        return base64.b64encode(content).decode("utf-8"), mime_type

    def _call_gemini(
        self,
        image_b64: str,
        mime_type: str,
        system_prompt: str = "",
        user_prompt: str = "",
        json_output: bool = False,
        max_output_tokens: int = 8192,
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
            "generationConfig": {"maxOutputTokens": max_output_tokens}
                                | ({"temperature": self.temperature} if self.temperature is not None else {})
                                | ({"seed": self.seed} if self.seed is not None else {}),
        }
        if json_output:
            body["generationConfig"]["responseMimeType"] = "application/json"

        model_path = self.model if self.model.startswith("models/") else f"models/{self.model}"

        with httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
        ) as client:
            response = _post_retrying(client, f"/{model_path}:generateContent", body)

        if response.status_code != 200:
            error = ProviderUnavailableError if _unavailable(response) else RuntimeError
            raise error(f"Gemini API error: {_error_message(response)}")

        data = _json_body(response, "Gemini")
        self.served = {"model": data.get("modelVersion"), "fingerprint": None}
        blocked = (data.get("promptFeedback") or {}).get("blockReason")
        candidates = data.get("candidates")
        candidates = candidates if isinstance(candidates, list) else []
        if blocked or not candidates:
            raise RuntimeError(f"Gemini API error: no candidates ({blocked or 'empty'})")

        candidate = candidates[0] if isinstance(candidates[0], dict) else {}
        content = candidate.get("content")
        content_parts = content.get("parts") if isinstance(content, dict) else None
        content_parts = content_parts if isinstance(content_parts, list) else []
        text = "".join(p.get("text") or "" for p in content_parts if isinstance(p, dict)).strip()
        if not text:
            reason = candidate.get("finishReason") or "no content"
            error = RuntimeError if reason in _REFUSAL_REASONS else ProviderUnavailableError
            raise error(f"Gemini API error: empty reply ({reason})")
        return text

    def _call_openai(
        self,
        image_b64: str,
        mime_type: str,
        system_prompt: str = "",
        user_prompt: str = "",
        json_output: bool = False,
        max_output_tokens: int = 8192,
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
            # api.openai.com takes max_completion_tokens; compatible servers take max_tokens
            ("max_completion_tokens" if self.base_url == openai_api_url else "max_tokens"): max_output_tokens,
        } | ({"temperature": self.temperature} if self.temperature is not None else {})           | ({"seed": self.seed} if self.seed is not None else {})
        if json_output:
            body["response_format"] = {"type": "json_object"}

        with httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        ) as client:
            response = _post_retrying(client, "/chat/completions", body)

        if response.status_code != 200:
            error = ProviderUnavailableError if _unavailable(response) else RuntimeError
            raise error(f"OpenAI API error: {_error_message(response)}")

        data = _json_body(response, "OpenAI")
        self.served = {"model": data.get("model"), "fingerprint": data.get("system_fingerprint")}
        choices = data.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else {}
        choice = choice if isinstance(choice, dict) else {}
        message = choice.get("message")
        message = message if isinstance(message, dict) else {}
        content = message.get("content")
        if isinstance(content, list):  # some compatible servers return content parts
            content = "".join(part.get("text") or "" for part in content
                              if isinstance(part, dict))
        if not isinstance(content, str):
            content = ""
        if not content:
            reason = message.get("refusal") or choice.get("finish_reason") or "no content"
            refused = message.get("refusal") or reason in _REFUSAL_REASONS
            raise (RuntimeError if refused else ProviderUnavailableError)(
                f"OpenAI API error: empty reply ({reason})")
        return content.strip()

    def _call_anthropic(
        self,
        image_b64: str,
        mime_type: str,
        system_prompt: str = "",
        user_prompt: str = "",
        json_output: bool = False,
        max_output_tokens: int = 8192,
    ) -> str:
        if not anthropic_available:
            raise ImportError("anthropic is required. Install with: pip install mathipy[ocr]")
        sdk = anthropic_sdk
        client = sdk.Anthropic(api_key=self._resolve_api_key(), base_url=self.base_url,
                               timeout=self.timeout, max_retries=_RETRY_ATTEMPTS)
        content = [{"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": image_b64}},
                   {"type": "text", "text": user_prompt}]
        try:
            response = client.messages.create(
                model=self.model, max_tokens=max_output_tokens, system=system_prompt,
                messages=[{"role": "user", "content": content}],
                **({"temperature": self.temperature} if self.temperature is not None else {}))
        except (sdk.APIConnectionError, sdk.RateLimitError, sdk.InternalServerError,
                getattr(sdk, "OverloadedError", sdk.InternalServerError),  # 529
                sdk.AuthenticationError, sdk.PermissionDeniedError, sdk.NotFoundError) as err:
            raise ProviderUnavailableError(f"Anthropic API error: {_sanitize_error(str(err))}") from err
        except sdk.APIStatusError as err:
            raise RuntimeError(f"Anthropic API error: {_sanitize_error(str(err))}") from err

        self.served = {"model": response.model, "fingerprint": None}
        if response.stop_reason == "refusal":
            raise RuntimeError("Anthropic API error: empty reply (refusal)")
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if not text:
            raise ProviderUnavailableError(f"Anthropic API error: empty reply ({response.stop_reason})")
        return text

    def _call(self, *args, **kwargs) -> str:
        """Send one vision request to the configured provider."""
        return {"gemini": self._call_gemini, "openai": self._call_openai,
                "anthropic": self._call_anthropic}[self.provider](*args, **kwargs)
