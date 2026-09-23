from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Sequence
from urllib.parse import urlencode

import aiohttp

from interfaces.video_output import VideoOutput
from utils.image import image_path_to_b64
from utils.rate_limiter import RateLimiter


DEFAULT_BASE_URL = "https://apihub.agnes-ai.com/v1"
DEFAULT_MODEL = "agnes-video-2.5-flash"


class AgnesVideoAPIError(RuntimeError):
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self.payload = payload
        super().__init__(f"Agnes video generation failed with HTTP {status_code}: {payload}")


class AgnesVideoProvider:
    """Async Agnes Video provider for submit, polling, and media download."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        default_seconds: int = 5,
        default_aspect_ratio: str = "16:9",
        default_resolution: str = "720P",
        request_timeout_seconds: float = 60.0,
        poll_timeout_seconds: float = 900.0,
        poll_interval_seconds: float = 5.0,
        max_retries: int = 3,
        retry_base_delay_seconds: float = 1.0,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.default_seconds = default_seconds
        self.default_aspect_ratio = default_aspect_ratio
        self.default_resolution = default_resolution

        self.request_timeout_seconds = max(1.0, request_timeout_seconds)
        self.poll_timeout_seconds = max(1.0, poll_timeout_seconds)
        self.poll_interval_seconds = max(0.0, poll_interval_seconds)
        self.max_retries = max(1, max_retries)
        self.retry_base_delay_seconds = max(0.0, retry_base_delay_seconds)
        self.rate_limiter = rate_limiter

    async def generate_single_video(
        self,
        prompt: str = "",
        reference_image_paths: Sequence[str] | None = None,
        aspect_ratio: str | None = None,
        seconds: int | None = None,
        **kwargs: Any,
    ) -> VideoOutput:
        if not self.api_key:
            raise ValueError("Agnes video API key is required")
        references = [_resolve_image_reference(path) for path in reference_image_paths or []]
        if len(references) > 2:
            raise ValueError("Agnes video supports at most two reference images")

        progress = kwargs.get("progress")
        payload = {
            "model": self.model,
            "prompt": prompt,
            "mode": "text",
            "seconds": str(seconds or self.default_seconds),
            "aspect_ratio": aspect_ratio or self.default_aspect_ratio,
            "size": kwargs.get("resolution", self.default_resolution),
            "n": 1,
        }
        if kwargs.get("seed") is not None:
            payload["seed"] = kwargs["seed"]
        if len(references) == 1:
            payload["mode"] = "img2video"
            payload["first_frame"] = references[0]
        elif len(references) == 2:
            payload["mode"] = "keyframe"
            payload["first_frame"] = references[0]
            payload["last_frame"] = references[1]
        _emit_progress(progress, "video_create", f"Creating Agnes video task with {self.model}", {"model": self.model})
        _, response = await self._request_with_retry(
            lambda: self._post_json(f"{self.base_url}/videos", payload),
            progress=progress,
            operation_name="create",
        )
        task_id = _task_id_from_response(response)
        _emit_progress(progress, "video_task_created", "Agnes video task created", {"model": self.model, "task_id": task_id})

        deadline = asyncio.get_running_loop().time() + self.poll_timeout_seconds
        while asyncio.get_running_loop().time() < deadline:
            _, result = await self._request_with_retry(
                lambda: self._get_json(self._status_url(task_id)),
                progress=progress,
                operation_name="poll",
            )
            task_status = _status_from_response(result)
            _emit_progress(progress, "video_status", f"Agnes video status: {task_status}", {"model": self.model, "task_id": task_id, "status": task_status})
            if task_status.lower() in {"completed", "success", "succeeded", "done", "finished"}:
                video_url = _video_url_from_response(result)
                _emit_progress(progress, "video_download_start", "Downloading Agnes video output", {"model": self.model, "task_id": task_id})
                _, video_bytes = await self._request_with_retry(
                    lambda: self._get_bytes(video_url),
                    progress=progress,
                    operation_name="download",
                )
                _emit_progress(progress, "video_completed", "Agnes video generation completed", {"model": self.model, "task_id": task_id})
                return VideoOutput(fmt="bytes", ext="mp4", data=video_bytes)
            if task_status.lower() in {"failed", "error", "cancelled", "canceled", "expired"}:
                raise RuntimeError(f"Agnes video task {task_id} failed: {result}")
            await asyncio.sleep(self.poll_interval_seconds)

        raise TimeoutError(f"Agnes video task {task_id} timed out after {self.poll_timeout_seconds:g}s")

    async def _post_json(self, url: str, payload: dict[str, Any]) -> tuple[int, Any]:
        if self.rate_limiter is not None:
            await self.rate_limiter.acquire()
        return await _post_json(url, headers=self._headers(), payload=payload, timeout=self._timeout())

    async def _get_json(self, url: str) -> tuple[int, Any]:
        if self.rate_limiter is not None:
            await self.rate_limiter.acquire()
        return await _get_json(url, headers=self._headers(), timeout=self._timeout())

    async def _get_bytes(self, url: str) -> tuple[int, bytes]:
        return await _get_bytes(url, headers={}, timeout=self._timeout())

    async def _request_with_retry(
        self,
        request: Callable[[], Any],
        *,
        progress: Callable[[str, str, dict[str, Any]], None] | None,
        operation_name: str,
    ) -> tuple[int, Any]:
        for attempt in range(1, self.max_retries + 1):
            try:
                status, payload = await request()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt >= self.max_retries:
                    raise
                await self._retry_after_transient_error(progress, operation_name, attempt, exc)
                continue
            if status < 400:
                return status, payload
            error = AgnesVideoAPIError(status, payload)
            if not _is_retryable_agnes_video_error(error) or attempt >= self.max_retries:
                raise error
            await self._retry_after_transient_error(progress, operation_name, attempt, error)
        raise RuntimeError(f"Agnes video {operation_name} exhausted retries")

    async def _retry_after_transient_error(
        self,
        progress: Callable[[str, str, dict[str, Any]], None] | None,
        operation: str,
        attempt: int,
        error: BaseException,
    ) -> None:
        delay = self.retry_base_delay_seconds * (2 ** (attempt - 1))
        _emit_progress(
            progress,
            "video_retry",
            f"Agnes video {operation} failed transiently; retrying after {delay:g}s",
            {"model": self.model, "operation": operation, "attempt": attempt, "max_attempts": self.max_retries, "error": str(error)},
        )
        await asyncio.sleep(delay)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _timeout(self) -> aiohttp.ClientTimeout:
        return aiohttp.ClientTimeout(total=self.request_timeout_seconds)

    def _status_url(self, video_id: str) -> str:
        api_root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        return f"{api_root}/agnesapi?{urlencode({'video_id': video_id, 'model_name': self.model})}"


def _task_id_from_response(response: Any) -> str:
    if not isinstance(response, dict):
        raise ValueError(f"Agnes video create response must be an object: {response}")
    raw_data = response.get("data")
    data: dict[str, Any] = raw_data if isinstance(raw_data, dict) else {}
    task_id = response.get("video_id") or response.get("task_id") or response.get("id") or data.get("video_id") or data.get("task_id") or data.get("id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"Agnes video create response missing video_id, task_id, or id: {response}")
    return task_id


def _status_from_response(response: Any) -> str:
    if not isinstance(response, dict):
        raise ValueError(f"Agnes video status response missing status: {response}")
    raw_data = response.get("data")
    data: dict[str, Any] = raw_data if isinstance(raw_data, dict) else {}
    status = response.get("status") or data.get("status")
    if not isinstance(status, str):
        raise ValueError(f"Agnes video status response missing status: {response}")
    return status


def _video_url_from_response(response: Any) -> str:
    if not isinstance(response, dict):
        raise ValueError(f"Agnes video completion response must be an object: {response}")
    raw_data = response.get("data")
    data: dict[str, Any] = raw_data if isinstance(raw_data, dict) else {}
    for value in (response.get("video_url"), response.get("url"), response.get("output_url"), data.get("video_url"), data.get("url")):
        if isinstance(value, str) and value:
            return value
    raise ValueError(f"Agnes video completion response missing video URL: {response}")


def _is_retryable_agnes_video_error(exc: BaseException) -> bool:
    if isinstance(exc, AgnesVideoAPIError):
        return exc.status_code == 429 or exc.status_code >= 500
    return isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError))


def _resolve_image_reference(reference: str) -> str:
    if reference.startswith(("http://", "https://", "data:")):
        return reference
    return image_path_to_b64(reference, mime=True)


def _emit_progress(progress: Callable[[str, str, dict[str, Any]], None] | None, stage: str, message: str, metadata: dict[str, Any]) -> None:
    if progress is not None:
        progress(stage, message, metadata)


async def _post_json(url: str, *, headers: dict[str, str], payload: dict[str, Any], timeout: aiohttp.ClientTimeout) -> tuple[int, Any]:
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, json=payload) as response:
            return response.status, await _response_json(response)


async def _get_json(url: str, *, headers: dict[str, str], timeout: aiohttp.ClientTimeout) -> tuple[int, Any]:
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, headers=headers) as response:
            return response.status, await _response_json(response)


async def _get_bytes(url: str, *, headers: dict[str, str], timeout: aiohttp.ClientTimeout) -> tuple[int, bytes]:
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, headers=headers) as response:
            return response.status, await response.read()


async def _response_json(response: aiohttp.ClientResponse) -> Any:
    text = await response.text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"message": text}
