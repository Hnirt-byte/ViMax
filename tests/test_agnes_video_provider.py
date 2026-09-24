import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from agent_runtime.vimax_adapters import _build_video_generator
from interfaces.video_output import VideoOutput
from tools.video_generator_agnes_api import AgnesVideoProvider


class AgnesVideoProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_generates_short_text_video_and_downloads_result(self):
        captured = {}
        limiter = AsyncMock()
        progress = []

        async def fake_post(url, *, headers, payload, timeout):
            captured.update(url=url, headers=headers, payload=payload, timeout=timeout)
            return 200, {"video_id": "task-1", "status": "queued"}

        async def fake_get(url, *, headers, timeout):
            self.assertEqual(url, "https://apihub.agnes-ai.com/agnesapi?video_id=task-1&model_name=agnes-video-2.5-flash")
            return 200, {"status": "completed", "video_url": "https://cdn.example.test/result.mp4"}

        async def fake_download(url, *, headers, timeout):
            self.assertEqual(url, "https://cdn.example.test/result.mp4")
            return 200, b"video-bytes"

        provider = AgnesVideoProvider(
            api_key="test-key",
            rate_limiter=limiter,
            request_timeout_seconds=12,
            poll_interval_seconds=0,
        )
        with patch("tools.video_generator_agnes_api._post_json", fake_post), \
             patch("tools.video_generator_agnes_api._get_json", fake_get), \
             patch("tools.video_generator_agnes_api._get_bytes", fake_download):
            output = await provider.generate_single_video(
                prompt="A blue paper airplane glides across a cloudless sky.",
                seconds=5,
                progress=lambda stage, message, metadata: progress.append((stage, message, metadata)),
            )

        self.assertIsInstance(output, VideoOutput)
        self.assertEqual(captured["url"], "https://apihub.agnes-ai.com/v1/videos")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(captured["payload"], {
            "model": "agnes-video-2.5-flash",
            "prompt": "A blue paper airplane glides across a cloudless sky.",
            "mode": "text",
            "seconds": "5",
            "aspect_ratio": "16:9",
            "size": "720P",
            "n": 1,
        })
        self.assertEqual(captured["timeout"].total, 12)
        self.assertEqual(output.fmt, "bytes")
        self.assertEqual(output.ext, "mp4")
        self.assertEqual(output.data, b"video-bytes")
        self.assertEqual([event[0] for event in progress], [
            "video_create",
            "video_task_created",
            "video_status",
            "video_download_start",
            "video_completed",
        ])
        self.assertGreaterEqual(limiter.acquire.await_count, 2)

    async def test_polls_queued_task_until_completion(self):
        post = AsyncMock(return_value=(200, {"id": "task-2"}))
        get = AsyncMock(side_effect=[
            (200, {"status": "queued"}),
            (200, {"status": "processing"}),
            (200, {"status": "completed", "url": "https://cdn.example.test/result.mp4"}),
        ])
        download = AsyncMock(return_value=(200, b"video"))
        provider = AgnesVideoProvider(api_key="test-key", poll_interval_seconds=0)
        with patch("tools.video_generator_agnes_api._post_json", post), \
             patch("tools.video_generator_agnes_api._get_json", get), \
             patch("tools.video_generator_agnes_api._get_bytes", download):
            await provider.generate_single_video("A kite flies over a meadow.")

        self.assertEqual(get.await_count, 3)

    async def test_supports_nested_agnes_response_envelopes(self):
        post = AsyncMock(return_value=(200, {"data": {"video_id": "task-nested"}}))
        get = AsyncMock(return_value=(200, {"data": {"status": "succeeded", "url": "https://cdn.example.test/nested.mp4"}}))
        download = AsyncMock(return_value=(200, b"nested-video"))
        provider = AgnesVideoProvider(api_key="test-key", poll_interval_seconds=0)
        with patch("tools.video_generator_agnes_api._post_json", post), \
             patch("tools.video_generator_agnes_api._get_json", get), \
             patch("tools.video_generator_agnes_api._get_bytes", download):
            output = await provider.generate_single_video("A boat crosses a calm lake.")

        self.assertEqual(output.data, b"nested-video")

    async def test_maps_one_image_to_i2v_and_two_images_to_keyframes(self):
        payloads = []

        async def fake_post(url, *, headers, payload, timeout):
            payloads.append(payload)
            return 200, {"video_id": f"task-{len(payloads)}"}

        async def fake_get(url, *, headers, timeout):
            return 200, {"status": "completed", "metadata": {"url": "https://cdn.example.test/result.mp4"}}

        async def fake_download(url, *, headers, timeout):
            return 200, b"video"

        first = "https://images.example.test/first.png"
        last = "https://images.example.test/last.png"
        provider = AgnesVideoProvider(api_key="test-key", poll_interval_seconds=0)
        with patch("tools.video_generator_agnes_api._post_json", fake_post), \
             patch("tools.video_generator_agnes_api._get_json", fake_get), \
             patch("tools.video_generator_agnes_api._get_bytes", fake_download):
            await provider.generate_single_video("A bird takes flight.", [first])
            await provider.generate_single_video("The bird reaches a distant tree.", [first, last])

        self.assertEqual(payloads[0], {
            "model": "agnes-video-2.5-flash",
            "prompt": "A bird takes flight.",
            "mode": "keyframe",
            "seconds": "5",
            "size": "720P",
            "aspect_ratio": "16:9",
            "first_frame": first,
            "n": 1,
        })
        self.assertEqual(payloads[1]["model"], "agnes-video-2.5-flash")
        self.assertEqual(payloads[1]["mode"], "keyframe")
        self.assertEqual(payloads[1]["first_frame"], first)
        self.assertEqual(payloads[1]["last_frame"], last)

    async def test_rejects_missing_api_key_without_network_request(self):
        post = AsyncMock()
        provider = AgnesVideoProvider(api_key="")
        with patch("tools.video_generator_agnes_api._post_json", post):
            with self.assertRaisesRegex(ValueError, "API key"):
                await provider.generate_single_video("A quiet river.")
        post.assert_not_awaited()

    async def test_retries_429_then_succeeds(self):
        post = AsyncMock(side_effect=[
            (429, {"error": {"message": "rate limited"}}),
            (200, {"task_id": "task-429"}),
        ])
        get = AsyncMock(return_value=(200, {"status": "completed", "video_url": "https://cdn.example.test/result.mp4"}))
        download = AsyncMock(return_value=(200, b"video"))
        provider = AgnesVideoProvider(api_key="test-key", max_retries=2, retry_base_delay_seconds=0, poll_interval_seconds=0)
        with patch("tools.video_generator_agnes_api._post_json", post), \
             patch("tools.video_generator_agnes_api._get_json", get), \
             patch("tools.video_generator_agnes_api._get_bytes", download):
            await provider.generate_single_video("A lantern floats above water.")
        self.assertEqual(post.await_count, 2)

    async def test_retries_5xx_then_succeeds(self):
        post = AsyncMock(side_effect=[
            (503, {"error": {"message": "unavailable"}}),
            (200, {"task_id": "task-503"}),
        ])
        get = AsyncMock(return_value=(200, {"status": "completed", "video_url": "https://cdn.example.test/result.mp4"}))
        download = AsyncMock(return_value=(200, b"video"))
        provider = AgnesVideoProvider(api_key="test-key", max_retries=2, retry_base_delay_seconds=0, poll_interval_seconds=0)
        with patch("tools.video_generator_agnes_api._post_json", post), \
             patch("tools.video_generator_agnes_api._get_json", get), \
             patch("tools.video_generator_agnes_api._get_bytes", download):
            await provider.generate_single_video("A lantern floats above water.")
        self.assertEqual(post.await_count, 2)

    async def test_retries_timeout_then_raises_timeout(self):
        post = AsyncMock(side_effect=asyncio.TimeoutError)
        provider = AgnesVideoProvider(api_key="test-key", max_retries=2, retry_base_delay_seconds=0)
        with patch("tools.video_generator_agnes_api._post_json", post):
            with self.assertRaises(asyncio.TimeoutError):
                await provider.generate_single_video("A lantern floats above water.")
        self.assertEqual(post.await_count, 2)

    async def test_raises_when_remote_job_fails(self):
        post = AsyncMock(return_value=(200, {"task_id": "task-failed"}))
        get = AsyncMock(return_value=(200, {"status": "failed", "error": "generation rejected"}))
        provider = AgnesVideoProvider(api_key="test-key", poll_interval_seconds=0)
        with patch("tools.video_generator_agnes_api._post_json", post), \
             patch("tools.video_generator_agnes_api._get_json", get):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                await provider.generate_single_video("A lantern floats above water.")

    async def test_rejects_invalid_create_response_without_retry(self):
        post = AsyncMock(return_value=(200, {"status": "queued"}))
        provider = AgnesVideoProvider(api_key="test-key")
        with patch("tools.video_generator_agnes_api._post_json", post):
            with self.assertRaisesRegex(ValueError, "missing video_id"):
                await provider.generate_single_video("A lantern floats above water.")
        self.assertEqual(post.await_count, 1)

    def test_agent_factory_selects_agnes_from_video_base_url(self):
        with patch("agent_runtime.vimax_adapters.video_api_key", return_value="test-key"), \
             patch("agent_runtime.vimax_adapters.video_model", return_value="agnes-video-2.5-flash"), \
             patch("agent_runtime.vimax_adapters.video_base_url", return_value="https://apihub.agnes-ai.com/v1"), \
             patch("agent_runtime.vimax_adapters.video_provider", return_value="agnes"):
            provider = _build_video_generator()

        self.assertIsInstance(provider, AgnesVideoProvider)
        self.assertEqual(getattr(provider, "model"), "agnes-video-2.5-flash")


if __name__ == "__main__":
    unittest.main()
