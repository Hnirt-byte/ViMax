from __future__ import annotations

import asyncio
import json
from pathlib import PurePosixPath
from typing import Any, Callable, Sequence
from urllib.parse import urlparse

import aiohttp

from interfaces.image_output import ImageOutput
from utils.image import image_path_to_b64
from utils.rate_limiter import RateLimiter


DEFAULT_BASE_URL = "https://apihub.agnes-ai.com/v1"
DEFAULT_MODEL = "agnes-image-2.1-flash"


class AgnesImageAPIError(RuntimeError):
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self.payload = payload
        super().__init__(f"Agnes image generation failed with HTTP {status_code}: {payload}")


class AgnesImageProvider:
    """Async Agnes Image provider for the OpenAI-compatible Images API."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        request_timeout_seconds: float = 60.0,
        max_retries: int = 3,
        retry_base_delay_seconds: float = 1.0,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.request_timeout_seconds = max(1.0, request_timeout_seconds)
        self.max_retries = max(1, max_retries)
        self.retry_base_delay_seconds = max(0.0, retry_base_delay_seconds)
        self.rate_limiter = rate_limiter

    async def generate_single_image(
        self,
        prompt: str,
        reference_image_paths: Sequence[str] | None = None,
        size: str | None = None,
        **kwargs: Any,
    ) -> ImageOutput:
        if not self.api_key:
            raise ValueError("Agnes image API key is required")
        if self.rate_limiter is not None:
            await self.rate_limiter.acquire()

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "n": 1,
            "size": size or "1024x1024",
        }
        references = [_resolve_image_reference(path) for path in reference_image_paths or []]
        if references:
            payload["extra_body"] = {
                "response_format": "url",
                "image": references,
            }
        progress = kwargs.get("progress")
        _emit_progress(progress, "image_generation", f"Generating image with {self.model}", {"model": self.model, "reference_count": len(references)})

        for attempt in range(1, self.max_retries + 1):
            try:
                status, response = await _post_json(
                    f"{self.base_url}/images/generations",
                    headers=self._headers(),
                    payload=payload,
                    timeout=aiohttp.ClientTimeout(total=self.request_timeout_seconds),
                )
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt >= self.max_retries:
                    raise
                await self._retry_after_transient_error(progress, attempt, exc)
                continue

            if status >= 400:
                error = AgnesImageAPIError(status, response)
                if not _is_retryable_agnes_image_error(error) or attempt >= self.max_retries:
                    raise error
                await self._retry_after_transient_error(progress, attempt, error)
                continue

            result = _image_output_from_response(response)
            _emit_progress(progress, "image_completed", "Agnes image generation completed", {"model": self.model, "format": result.fmt})
            return result

        raise RuntimeError("Agnes image generation exhausted retries without a result")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _retry_after_transient_error(
        self,
        progress: Callable[[str, str, dict[str, Any]], None] | None,
        attempt: int,
        error: BaseException,
    ) -> None:
        delay = self.retry_base_delay_seconds * (2 ** (attempt - 1))
        _emit_progress(
            progress,
            "image_retry",
            f"Agnes image request failed transiently; retrying after {delay:g}s",
            {"model": self.model, "attempt": attempt, "max_attempts": self.max_retries, "error": str(error)},
        )
        await asyncio.sleep(delay)


def _image_output_from_response(response: Any) -> ImageOutput:
    data = response.get("data") if isinstance(response, dict) else None
    item = data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else None
    if item is None:
        raise ValueError(f"Agnes image response missing data[0]: {response}")
    url = item.get("url")
    if isinstance(url, str) and url:
        return ImageOutput(fmt="url", ext=_extension_from_url(url), data=url)
    b64_json = item.get("b64_json")
    if isinstance(b64_json, str) and b64_json:
        return ImageOutput(fmt="b64", ext="png", data=_strip_data_uri(b64_json))
    raise ValueError(f"Agnes image response missing data[0].url or data[0].b64_json: {response}")


def _is_retryable_agnes_image_error(exc: BaseException) -> bool:
    if isinstance(exc, AgnesImageAPIError):
        return exc.status_code == 429 or exc.status_code >= 500
    return isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError))


def _resolve_image_reference(reference: str) -> str:
    if reference.startswith(("http://", "https://", "data:")):
        return reference
    return image_path_to_b64(reference, mime=True)


def _strip_data_uri(value: str) -> str:
    return value.split(",", 1)[1] if value.startswith("data:") and "," in value else value


def _extension_from_url(url: str) -> str:
    suffix = PurePosixPath(urlparse(url).path).suffix.lower().lstrip(".")
    return suffix if suffix in {"png", "jpg", "jpeg", "webp"} else "png"


def _emit_progress(progress: Callable[[str, str, dict[str, Any]], None] | None, stage: str, message: str, metadata: dict[str, Any]) -> None:
    if progress is not None:
        progress(stage, message, metadata)


async def _post_json(
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: aiohttp.ClientTimeout,
) -> tuple[int, Any]:
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, json=payload) as response:
            text = await response.text()
            try:
                body = json.loads(text)
            except json.JSONDecodeError:
                body = {"message": text}
            return response.status, body
