import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from agent_runtime.vimax_adapters import _build_video_generator
from interfaces.video_output import VideoOutput
from tools.video_generator_agnes_api import AgnesVideoAPIError, AgnesVideoProvider


class AgnesVideoProviderTests(unittest.IsolatedAsyncioTestCase):
    def test_default_poll_interval_is_conservative(self):
        provider = AgnesVideoProvider(api_key="test-key")
        self.assertGreaterEqual(provider.poll_interval_seconds, 10)

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

    async def test_persists_task_id_before_polling(self):
        created = []

        async def fake_get(url, *, headers, timeout):
            self.assertEqual(created, [("task-persisted", "agnes-video-2.5-flash")])
            return 200, {"status": "completed", "url": "https://cdn.example.test/result.mp4"}

        provider = AgnesVideoProvider(api_key="test-key", poll_interval_seconds=0)
        with patch("tools.video_generator_agnes_api._post_json", AsyncMock(return_value=(200, {"video_id": "task-persisted"}))), \
             patch("tools.video_generator_agnes_api._get_json", fake_get), \
             patch("tools.video_generator_agnes_api._get_bytes", AsyncMock(return_value=(200, b"video"))):
            await provider.generate_single_video(
                "A lantern floats above water.",
                task_created_callback=lambda video_id, model: created.append((video_id, model)),
            )

    async def test_polls_existing_video_id_without_submitting_again(self):
        provider = AgnesVideoProvider(api_key="test-key", poll_interval_seconds=0)
        with patch("tools.video_generator_agnes_api._post_json", AsyncMock(side_effect=AssertionError("must not submit"))), \
             patch("tools.video_generator_agnes_api._get_json", AsyncMock(return_value=(200, {"status": "completed", "url": "https://cdn.example.test/result.mp4"}))), \
             patch("tools.video_generator_agnes_api._get_bytes", AsyncMock(return_value=(200, b"video"))):
            output = await provider.poll_existing_task("task-existing", "agnes-video-2.5-flash")

        self.assertEqual(output.data, b"video")

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

    async def test_transmits_explicit_portrait_render_settings(self):
        post = AsyncMock(return_value=(200, {"video_id": "task-portrait"}))
        get = AsyncMock(return_value=(200, {"status": "completed", "url": "https://cdn.example.test/result.mp4"}))
        download = AsyncMock(return_value=(200, b"video"))
        provider = AgnesVideoProvider(api_key="test-key", poll_interval_seconds=0)
        with patch("tools.video_generator_agnes_api._post_json", post), \
             patch("tools.video_generator_agnes_api._get_json", get), \
             patch("tools.video_generator_agnes_api._get_bytes", download):
            await provider.generate_single_video(
                "A red paper airplane rises through a warm sunset sky.",
                ["https://images.example.test/portrait-first-frame.png"],
                aspect_ratio="9:16",
                resolution="720P",
                seconds=5,
            )

        payload = post.await_args.kwargs["payload"]
        self.assertEqual(payload["aspect_ratio"], "9:16")
        self.assertEqual(payload["size"], "720P")
        self.assertEqual(payload["seconds"], "5")

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

    async def test_waits_for_video_queue_capacity_before_successful_submit(self):
        post = AsyncMock(side_effect=[
            (503, {"code": "video_queue_full", "message": "video queue is full"}),
            (503, {"code": "video_queue_full", "message": "video queue is full"}),
            (200, {"video_id": "task-queue"}),
        ])
        get = AsyncMock(return_value=(200, {"status": "completed", "url": "https://cdn.example.test/result.mp4"}))
        download = AsyncMock(return_value=(200, b"video"))
        sleeps = AsyncMock()
        events = []
        provider = AgnesVideoProvider(
            api_key="test-key",
            max_retries=1,
            queue_max_attempts=3,
            queue_retry_base_delay_seconds=30,
            queue_retry_max_delay_seconds=90,
            poll_interval_seconds=0,
        )
        with patch("tools.video_generator_agnes_api._post_json", post), \
             patch("tools.video_generator_agnes_api._get_json", get), \
             patch("tools.video_generator_agnes_api._get_bytes", download), \
             patch("tools.video_generator_agnes_api.asyncio.sleep", sleeps):
            await provider.generate_single_video(
                "A lantern floats above water.",
                progress=lambda stage, message, metadata: events.append((stage, metadata)),
            )

        self.assertEqual(post.await_count, 3)
        self.assertEqual([call.args[0] for call in sleeps.await_args_list], [30, 60])
        queue_events = [metadata for stage, metadata in events if stage == "video_queue_wait"]
        self.assertEqual([event["delay_seconds"] for event in queue_events], [30, 60])
        self.assertTrue(all(event["error_code"] == "video_queue_full" for event in queue_events))

    async def test_stops_after_configured_video_queue_attempt_limit(self):
        post = AsyncMock(return_value=(503, {"code": "video_queue_full", "message": "video queue is full"}))
        sleeps = AsyncMock()
        provider = AgnesVideoProvider(
            api_key="test-key",
            max_retries=1,
            queue_max_attempts=3,
            queue_retry_base_delay_seconds=30,
            poll_interval_seconds=0,
        )
        with patch("tools.video_generator_agnes_api._post_json", post), \
             patch("tools.video_generator_agnes_api.asyncio.sleep", sleeps):
            with self.assertRaisesRegex(AgnesVideoAPIError, "video_queue_full") as raised:
                await provider.generate_single_video("A lantern floats above water.")

        self.assertEqual(raised.exception.queue_attempts, 3)
        self.assertEqual(post.await_count, 3)
        self.assertEqual([call.args[0] for call in sleeps.await_args_list], [30, 60])

    async def test_does_not_resubmit_after_video_id_is_obtained(self):
        post = AsyncMock(return_value=(200, {"video_id": "task-once"}))
        get = AsyncMock(side_effect=[
            (429, {"error": {"message": "status rate limited"}}),
            (200, {"status": "completed", "url": "https://cdn.example.test/result.mp4"}),
        ])
        download = AsyncMock(return_value=(200, b"video"))
        sleeps = AsyncMock()
        provider = AgnesVideoProvider(
            api_key="test-key",
            max_retries=2,
            retry_base_delay_seconds=0,
            queue_max_attempts=3,
            poll_interval_seconds=0,
        )
        with patch("tools.video_generator_agnes_api._post_json", post), \
             patch("tools.video_generator_agnes_api._get_json", get), \
             patch("tools.video_generator_agnes_api._get_bytes", download), \
             patch("tools.video_generator_agnes_api.asyncio.sleep", sleeps):
            await provider.generate_single_video("A lantern floats above water.")

        self.assertEqual(post.await_count, 1)
        self.assertEqual(get.await_count, 2)

    async def test_flash_success_does_not_use_paid_fallback(self):
        post = AsyncMock(return_value=(200, {"video_id": "flash-task"}))
        get = AsyncMock(return_value=(200, {"status": "completed", "url": "https://cdn.example.test/result.mp4"}))
        download = AsyncMock(return_value=(200, b"video"))
        events = []
        provider = AgnesVideoProvider(
            api_key="test-key",
            allow_paid_video_fallback=True,
            paid_fallback_model="agnes-video-2.5",
            poll_interval_seconds=0,
        )
        with patch("tools.video_generator_agnes_api._post_json", post), \
             patch("tools.video_generator_agnes_api._get_json", get), \
             patch("tools.video_generator_agnes_api._get_bytes", download):
            await provider.generate_single_video(
                "A red paper airplane glides through a clear sky.",
                ["https://images.example.test/scene-0-first-frame.png"],
                progress=lambda stage, message, metadata: events.append((stage, metadata)),
            )

        self.assertEqual(post.await_count, 1)
        self.assertEqual(post.await_args.kwargs["payload"]["model"], "agnes-video-2.5-flash")
        self.assertNotIn("video_paid_fallback", [stage for stage, _ in events])

    async def test_flash_queue_full_refuses_paid_fallback_when_disabled(self):
        post = AsyncMock(return_value=(503, {"code": "video_queue_full", "message": "video queue is full"}))
        provider = AgnesVideoProvider(
            api_key="test-key",
            allow_paid_video_fallback=False,
            paid_fallback_model="agnes-video-2.5",
            queue_max_attempts=1,
        )
        with patch("tools.video_generator_agnes_api._post_json", post):
            with self.assertRaisesRegex(AgnesVideoAPIError, "video_queue_full"):
                await provider.generate_single_video(
                    "A red paper airplane glides through a clear sky.",
                    ["https://images.example.test/scene-0-first-frame.png"],
                )

        self.assertEqual(post.await_count, 1)
        self.assertEqual(post.await_args.kwargs["payload"]["model"], "agnes-video-2.5-flash")

    async def test_flash_queue_full_uses_explicit_paid_fallback_with_same_first_frame(self):
        post = AsyncMock(side_effect=[
            (503, {"code": "video_queue_full", "message": "video queue is full"}),
            (200, {"video_id": "standard-task"}),
        ])
        get = AsyncMock(return_value=(200, {"status": "completed", "url": "https://cdn.example.test/result.mp4"}))
        download = AsyncMock(return_value=(200, b"video"))
        events = []
        provider = AgnesVideoProvider(
            api_key="test-key",
            allow_paid_video_fallback=True,
            paid_fallback_model="agnes-video-2.5",
            queue_max_attempts=1,
            poll_interval_seconds=0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            first_frame = Path(tmp) / "scene-0-first-frame.png"
            first_frame.write_bytes(b"cached-image")
            Path(f"{first_frame}.source.json").write_text(
                json.dumps({"source_url": "https://images.example.test/scene-0-first-frame.png"}),
                encoding="utf-8",
            )
            with patch("tools.video_generator_agnes_api._post_json", post), \
                 patch("tools.video_generator_agnes_api._get_json", get), \
                 patch("tools.video_generator_agnes_api._get_bytes", download):
                await provider.generate_single_video(
                    "A red paper airplane glides through a clear sky.",
                    [str(first_frame)],
                    progress=lambda stage, message, metadata: events.append((stage, metadata)),
                )

        self.assertEqual(post.await_count, 2)
        flash_payload, standard_payload = [call.kwargs["payload"] for call in post.await_args_list]
        self.assertEqual(flash_payload["model"], "agnes-video-2.5-flash")
        self.assertEqual(standard_payload["model"], "agnes-video-2.5")
        self.assertEqual(flash_payload["first_frame"], "https://images.example.test/scene-0-first-frame.png")
        self.assertEqual(standard_payload["first_frame"], "https://images.example.test/scene-0-first-frame.png")
        create_models = [metadata["model"] for stage, metadata in events if stage == "video_create"]
        self.assertEqual(create_models, ["agnes-video-2.5-flash", "agnes-video-2.5"])
        fallback_events = [metadata for stage, metadata in events if stage == "video_paid_fallback"]
        self.assertEqual(fallback_events, [{
            "from_model": "agnes-video-2.5-flash",
            "to_model": "agnes-video-2.5",
            "reason": "video_queue_full",
        }])

    async def test_paid_fallback_never_submits_after_flash_video_id(self):
        post = AsyncMock(return_value=(200, {"video_id": "flash-task"}))
        get = AsyncMock(side_effect=[
            (429, {"error": {"message": "status rate limited"}}),
            (200, {"status": "completed", "url": "https://cdn.example.test/result.mp4"}),
        ])
        download = AsyncMock(return_value=(200, b"video"))
        provider = AgnesVideoProvider(
            api_key="test-key",
            allow_paid_video_fallback=True,
            paid_fallback_model="agnes-video-2.5",
            max_retries=2,
            retry_base_delay_seconds=0,
            poll_interval_seconds=0,
        )
        with patch("tools.video_generator_agnes_api._post_json", post), \
             patch("tools.video_generator_agnes_api._get_json", get), \
             patch("tools.video_generator_agnes_api._get_bytes", download), \
             patch("tools.video_generator_agnes_api.asyncio.sleep", AsyncMock()):
            await provider.generate_single_video("A red paper airplane glides through a clear sky.")

        self.assertEqual(post.await_count, 1)
        self.assertEqual(post.await_args.kwargs["payload"]["model"], "agnes-video-2.5-flash")

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
             patch("agent_runtime.vimax_adapters.video_allow_paid_fallback", return_value=True), \
             patch("agent_runtime.vimax_adapters.video_paid_fallback_model", return_value="agnes-video-2.5"), \
             patch("agent_runtime.vimax_adapters.video_provider", return_value="agnes"):
            provider = _build_video_generator()

        self.assertIsInstance(provider, AgnesVideoProvider)
        self.assertEqual(getattr(provider, "model"), "agnes-video-2.5-flash")
        self.assertTrue(getattr(provider, "allow_paid_video_fallback"))
        self.assertEqual(getattr(provider, "paid_fallback_model"), "agnes-video-2.5")


if __name__ == "__main__":
    unittest.main()
