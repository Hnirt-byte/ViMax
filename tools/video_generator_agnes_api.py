from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Sequence
from urllib.parse import urlencode

import aiohttp

from interfaces.video_output import VideoOutput
from utils.image import image_path_to_b64
from utils.image_reference import public_source_url_for_image
from utils.rate_limiter import RateLimiter


DEFAULT_BASE_URL = "https://apihub.agnes-ai.com/v1"
DEFAULT_MODEL = "agnes-video-2.5-flash"
DEFAULT_PAID_FALLBACK_MODEL = "agnes-video-2.5"
# Deprecated compatibility path. New reference jobs use the configured 2.5 model.
DEFAULT_REFERENCE_MODEL = "agnes-video-v2.0"
logger = logging.getLogger(__name__)


class AgnesVideoAPIError(RuntimeError):
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self.payload = payload
        super().__init__(f"Agnes video generation failed with HTTP {status_code}: {payload}")


class AgnesVideoProvider:
    """Async Agnes Video provider; v2.0 reference support is explicit legacy compatibility."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        reference_model: str = DEFAULT_REFERENCE_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        default_seconds: int = 5,
        default_aspect_ratio: str = "16:9",
        default_resolution: str = "720P",
        request_timeout_seconds: float = 60.0,
        poll_timeout_seconds: float = 900.0,
        poll_interval_seconds: float = 10.0,
        max_retries: int = 3,
        retry_base_delay_seconds: float = 1.0,
        queue_max_attempts: int = 3,
        queue_retry_base_delay_seconds: float = 30.0,
        queue_retry_max_delay_seconds: float = 300.0,
        queue_retry_multiplier: float = 2.0,
        allow_paid_video_fallback: bool = False,
        paid_fallback_model: str = DEFAULT_PAID_FALLBACK_MODEL,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.reference_model = reference_model  # Deprecated v2.0 compatibility model.
        self.base_url = base_url.rstrip("/")
        self.default_seconds = default_seconds
        self.default_aspect_ratio = default_aspect_ratio
        self.default_resolution = default_resolution

        self.request_timeout_seconds = max(1.0, request_timeout_seconds)
        self.poll_timeout_seconds = max(1.0, poll_timeout_seconds)
        self.poll_interval_seconds = max(0.0, poll_interval_seconds)
        self.max_retries = max(1, max_retries)
        self.retry_base_delay_seconds = max(0.0, retry_base_delay_seconds)
        self.queue_max_attempts = max(1, queue_max_attempts)
        self.queue_retry_base_delay_seconds = max(0.0, queue_retry_base_delay_seconds)
        self.queue_retry_max_delay_seconds = max(self.queue_retry_base_delay_seconds, queue_retry_max_delay_seconds)
        self.queue_retry_multiplier = max(1.0, queue_retry_multiplier)
        self.allow_paid_video_fallback = allow_paid_video_fallback
        self.paid_fallback_model = paid_fallback_model.strip() or DEFAULT_PAID_FALLBACK_MODEL
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
        references = list(reference_image_paths or [])
        if len(references) > 2:
            raise ValueError("Agnes video supports at most two reference images")

        progress = kwargs.get("progress")
        duration = seconds or self.default_seconds
        ratio = aspect_ratio or self.default_aspect_ratio
        resolution = kwargs.get("resolution", self.default_resolution)
        model = self.reference_model if references and self.model == self.reference_model else self.model
        payload = self._create_payload(
            model=model,
            prompt=prompt,
            references=references,
            aspect_ratio=ratio,
            seconds=duration,
            resolution=resolution,
            seed=kwargs.get("seed"),
            negative_prompt=kwargs.get("negative_prompt"),
        )
        try:
            _, response = await self._create_task(model=model, payload=payload, progress=progress)
        except AgnesVideoAPIError as exc:
            if not self._can_use_paid_fallback(model, exc):
                raise
            fallback_model = self.paid_fallback_model
            logger.warning(
                "Agnes Video Flash queue remained full after %d submit attempts; using explicitly enabled paid fallback model %s",
                self.queue_max_attempts,
                fallback_model,
            )
            _emit_progress(
                progress,
                "video_paid_fallback",
                f"Agnes Video Flash queue remained full; using explicitly enabled paid fallback {fallback_model}",
                {"from_model": model, "to_model": fallback_model, "reason": "video_queue_full"},
            )
            model = fallback_model
            payload = self._create_payload(
                model=model,
                prompt=prompt,
                references=references,
                aspect_ratio=ratio,
                seconds=duration,
                resolution=resolution,
                seed=kwargs.get("seed"),
                negative_prompt=kwargs.get("negative_prompt"),
            )
            _, response = await self._create_task(model=model, payload=payload, progress=progress)
        task_id = _task_id_from_response(response)
        _emit_progress(progress, "video_task_created", "Agnes video task created", {"model": model, "task_id": task_id})

        deadline = asyncio.get_running_loop().time() + self.poll_timeout_seconds
        while asyncio.get_running_loop().time() < deadline:
            _, result = await self._request_with_retry(
                lambda: self._get_json(self._status_url(task_id, model)),
                progress=progress,
                operation_name="poll",
                model=model,
            )
            task_status = _status_from_response(result)
            _emit_progress(progress, "video_status", f"Agnes video status: {task_status}", {"model": model, "task_id": task_id, "status": task_status})
            if task_status.lower() in {"completed", "success", "succeeded", "done", "finished"}:
                video_url = _video_url_from_response(result)
                _emit_progress(progress, "video_download_start", "Downloading Agnes video output", {"model": model, "task_id": task_id})
                _, video_bytes = await self._request_with_retry(
                    lambda: self._get_bytes(video_url),
                    progress=progress,
                    operation_name="download",
                    model=model,
                )
                _emit_progress(progress, "video_completed", "Agnes video generation completed", {"model": model, "task_id": task_id})
                return VideoOutput(fmt="bytes", ext="mp4", data=video_bytes)
            if task_status.lower() in {"failed", "error", "cancelled", "canceled", "expired"}:
                raise RuntimeError(f"Agnes video task {task_id} failed: {result}")
            await asyncio.sleep(self.poll_interval_seconds)

        raise TimeoutError(f"Agnes video task {task_id} timed out after {self.poll_timeout_seconds:g}s")

    def _create_payload(
        self,
        *,
        model: str,
        prompt: str,
        references: Sequence[str],
        aspect_ratio: str,
        seconds: int,
        resolution: str,
        seed: Any,
        negative_prompt: Any,
    ) -> dict[str, Any]:
        if references:
            if model == self.reference_model:
                return _build_v2_reference_payload(
                    model=model,
                    prompt=prompt,
                    references=references,
                    aspect_ratio=aspect_ratio,
                    seconds=seconds,
                    resolution=resolution,
                    seed=seed,
                    negative_prompt=negative_prompt,
                )
            return _build_25_keyframe_payload(
                model=model,
                prompt=prompt,
                references=references,
                aspect_ratio=aspect_ratio,
                seconds=seconds,
                resolution=resolution,
                seed=seed,
            )
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "mode": "text",
            "seconds": str(seconds),
            "aspect_ratio": aspect_ratio,
            "size": resolution,
            "n": 1,
        }
        if seed is not None:
            payload["seed"] = seed
        return payload

    async def _create_task(
        self,
        *,
        model: str,
        payload: dict[str, Any],
        progress: Callable[[str, str, dict[str, Any]], None] | None,
    ) -> tuple[int, Any]:
        logger.info("Creating Agnes video task with selected model %s", model)
        _emit_progress(progress, "video_create", f"Creating Agnes video task with {model}", {"model": model})
        return await self._request_with_retry(
            lambda: self._post_json(f"{self.base_url}/videos", payload),
            progress=progress,
            operation_name="create",
            model=model,
        )

    def _can_use_paid_fallback(self, model: str, exc: AgnesVideoAPIError) -> bool:
        return (
            self.allow_paid_video_fallback
            and _is_video_queue_full_error(exc)
            and model.endswith("-flash")
            and self.paid_fallback_model != model
        )

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
        model: str,
    ) -> tuple[int, Any]:
        retry_attempt = 1
        queue_attempt = 1
        while True:
            try:
                status, payload = await request()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if retry_attempt >= self.max_retries:
                    raise
                await self._retry_after_transient_error(progress, operation_name, retry_attempt, exc, model)
                retry_attempt += 1
                continue
            if status < 400:
                return status, payload
            error = AgnesVideoAPIError(status, payload)
            if operation_name == "create" and _is_video_queue_full_error(error):
                if queue_attempt >= self.queue_max_attempts:
                    raise error
                await self._retry_after_queue_full(progress, queue_attempt, model)
                queue_attempt += 1
                continue
            if not _is_retryable_agnes_video_error(error) or retry_attempt >= self.max_retries:
                raise error
            await self._retry_after_transient_error(progress, operation_name, retry_attempt, error, model)
            retry_attempt += 1

    async def _retry_after_transient_error(
        self,
        progress: Callable[[str, str, dict[str, Any]], None] | None,
        operation: str,
        attempt: int,
        error: BaseException,
        model: str,
    ) -> None:
        delay = self.retry_base_delay_seconds * (2 ** (attempt - 1))
        _emit_progress(
            progress,
            "video_retry",
            f"Agnes video {operation} failed transiently; retrying after {delay:g}s",
            {"model": model, "operation": operation, "attempt": attempt, "max_attempts": self.max_retries, "error": str(error)},
        )
        await asyncio.sleep(delay)

    async def _retry_after_queue_full(
        self,
        progress: Callable[[str, str, dict[str, Any]], None] | None,
        queue_attempt: int,
        model: str,
    ) -> None:
        delay = min(
            self.queue_retry_max_delay_seconds,
            self.queue_retry_base_delay_seconds * (self.queue_retry_multiplier ** (queue_attempt - 1)),
        )
        logger.warning(
            "Agnes video queue full; waiting %.1fs before submit retry %d/%d",
            delay,
            queue_attempt + 1,
            self.queue_max_attempts,
        )
        _emit_progress(
            progress,
            "video_queue_wait",
            f"Agnes video queue is full; retrying submit after {delay:g}s",
            {
                "model": model,
                "operation": "create",
                "error_code": "video_queue_full",
                "queue_attempt": queue_attempt,
                "max_attempts": self.queue_max_attempts,
                "delay_seconds": delay,
            },
        )
        await asyncio.sleep(delay)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _timeout(self) -> aiohttp.ClientTimeout:
        return aiohttp.ClientTimeout(total=self.request_timeout_seconds)

    def _status_url(self, video_id: str, model: str) -> str:
        api_root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        return f"{api_root}/agnesapi?{urlencode({'video_id': video_id, 'model_name': model})}"


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
    raw_metadata = response.get("metadata")
    metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
    raw_data_metadata = data.get("metadata")
    data_metadata: dict[str, Any] = raw_data_metadata if isinstance(raw_data_metadata, dict) else {}
    for value in (response.get("video_url"), response.get("url"), response.get("output_url"), data.get("video_url"), data.get("url"), metadata.get("url"), data_metadata.get("url")):
        if isinstance(value, str) and value:
            return value
    raise ValueError(f"Agnes video completion response missing video URL: {response}")


def _is_retryable_agnes_video_error(exc: BaseException) -> bool:
    if isinstance(exc, AgnesVideoAPIError):
        return exc.status_code == 429 or exc.status_code >= 500
    return isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError))


def _is_video_queue_full_error(exc: BaseException) -> bool:
    return (
        isinstance(exc, AgnesVideoAPIError)
        and exc.status_code == 503
        and isinstance(exc.payload, dict)
        and exc.payload.get("code") == "video_queue_full"
    )


def _build_v2_reference_payload(
    *,
    model: str,
    prompt: str,
    references: Sequence[str],
    aspect_ratio: str,
    seconds: int,
    resolution: str,
    seed: Any,
    negative_prompt: Any,
) -> dict[str, Any]:
    width, height = _v2_dimensions(aspect_ratio, resolution)
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_frames": _v2_num_frames(seconds),
        "frame_rate": 24,
    }
    if len(references) == 1:
        payload["image"] = _public_image_url(references[0], model)
        payload["mode"] = "ti2vid"
    else:
        payload["extra_body"] = {"image": [_public_image_url(reference, model) for reference in references], "mode": "keyframes"}
    if seed is not None:
        payload["seed"] = seed
    if isinstance(negative_prompt, str) and negative_prompt:
        payload["negative_prompt"] = negative_prompt
    return payload


def _build_25_keyframe_payload(
    *,
    model: str,
    prompt: str,
    references: Sequence[str],
    aspect_ratio: str,
    seconds: int,
    resolution: str,
    seed: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "mode": "keyframe",
        "seconds": str(seconds),
        "size": resolution,
        "aspect_ratio": aspect_ratio,
        "first_frame": _public_image_url(references[0], model),
        "n": 1,
    }
    if len(references) == 2:
        payload["last_frame"] = _public_image_url(references[1], model)
    if seed is not None:
        payload["seed"] = seed
    return payload


def _public_image_url(reference: str, model: str) -> str:
    source_url = public_source_url_for_image(reference)
    if source_url is not None:
        return source_url
    raise ValueError(
        f"Agnes Video {model} image references must be publicly accessible http(s) URLs; "
        "local files require a persisted source_url sidecar"
    )


def _v2_num_frames(seconds: int) -> int:
    target = max(9, round(seconds * 24))
    return min(441, max(9, round((target - 1) / 8) * 8 + 1))


def _v2_dimensions(aspect_ratio: str, resolution: str) -> tuple[int, int]:
    normalized_resolution = str(resolution).upper().removesuffix("P")
    try:
        short_edge = int(normalized_resolution)
    except ValueError as exc:
        raise ValueError(f"Unsupported Agnes Video v2.0 resolution: {resolution}") from exc
    if short_edge not in {480, 720, 1080}:
        raise ValueError(f"Unsupported Agnes Video v2.0 resolution: {resolution}")
    ratios = {"16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4)}
    if aspect_ratio not in ratios:
        raise ValueError(f"Unsupported Agnes Video v2.0 aspect ratio: {aspect_ratio}")
    horizontal, vertical = ratios[aspect_ratio]
    if horizontal >= vertical:
        return round(short_edge * horizontal / vertical), short_edge
    return short_edge, round(short_edge * vertical / horizontal)


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
