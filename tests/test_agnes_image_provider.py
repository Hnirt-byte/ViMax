import asyncio
import base64
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image

from agent_runtime.vimax_adapters import _build_image_generator
from tools.image_generator_agnes_api import AgnesImageProvider


class AgnesImageProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_generates_image_from_prompt_with_current_agnes_images_endpoint(self):
        captured = {}
        limiter = AsyncMock()
        progress = []

        async def fake_post(url, *, headers, payload, timeout):
            captured.update(url=url, headers=headers, payload=payload, timeout=timeout)
            return 200, {"data": [{"url": "https://cdn.example.test/image.webp"}]}

        provider = AgnesImageProvider(
            api_key="test-key",
            rate_limiter=limiter,
            request_timeout_seconds=12,
        )
        with patch("tools.image_generator_agnes_api._post_json", fake_post):
            result = await provider.generate_single_image(
                "a cinematic alpine lake",
                size="1536x1024",
                progress=lambda stage, message, metadata: progress.append((stage, message, metadata)),
            )

        self.assertEqual(captured["url"], "https://apihub.agnes-ai.com/v1/images/generations")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(captured["payload"], {
            "model": "agnes-image-2.1-flash",
            "prompt": "a cinematic alpine lake",
            "n": 1,
            "size": "1536x1024",
        })
        self.assertEqual(captured["timeout"].total, 12)
        limiter.acquire.assert_awaited_once()
        self.assertEqual(result.fmt, "url")
        self.assertEqual(result.ext, "webp")
        self.assertEqual(result.data, "https://cdn.example.test/image.webp")
        self.assertEqual([item[0] for item in progress], ["image_generation", "image_completed"])

    async def test_supports_local_and_url_image_references_and_base64_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference_path = Path(tmp) / "reference.png"
            Image.new("RGB", (16, 9), "red").save(reference_path)
            encoded = _encoded_png()
            post = AsyncMock(return_value=(200, {"data": [{"b64_json": f"data:image/png;base64,{encoded}"}]}))
            provider = AgnesImageProvider(api_key="test-key")
            with patch("tools.image_generator_agnes_api._post_json", post):
                result = await provider.generate_single_image(
                    "restyle this image",
                    reference_image_paths=[str(reference_path), "https://images.example.test/reference.jpg"],
                )

        payload = post.await_args.kwargs["payload"]
        self.assertEqual(payload["extra_body"]["response_format"], "url")
        self.assertTrue(payload["extra_body"]["image"][0].startswith("data:image/png;base64,"))
        self.assertEqual(payload["extra_body"]["image"][1], "https://images.example.test/reference.jpg")
        self.assertEqual(result.fmt, "b64")
        self.assertEqual(result.data, encoded)

    async def test_transmits_a_single_image_reference_as_an_array(self):
        post = AsyncMock(return_value=(200, {"data": [{"url": "https://cdn.example.test/image.png"}]}))
        provider = AgnesImageProvider(api_key="test-key")
        with patch("tools.image_generator_agnes_api._post_json", post):
            await provider.generate_single_image(
                "restyle this image",
                reference_image_paths=["https://images.example.test/reference.jpg"],
            )

        self.assertEqual(
            post.await_args.kwargs["payload"]["extra_body"]["image"],
            ["https://images.example.test/reference.jpg"],
        )

    async def test_rejects_missing_api_key_without_network_request(self):
        post = AsyncMock()
        provider = AgnesImageProvider(api_key="")
        with patch("tools.image_generator_agnes_api._post_json", post):
            with self.assertRaisesRegex(ValueError, "API key"):
                await provider.generate_single_image("a skyline")
        post.assert_not_awaited()

    async def test_retries_429_then_succeeds(self):
        post = AsyncMock(side_effect=[
            (429, {"error": {"message": "rate limited"}}),
            (200, {"data": [{"url": "https://cdn.example.test/image.png"}]}),
        ])
        provider = AgnesImageProvider(api_key="test-key", max_retries=2, retry_base_delay_seconds=0)
        with patch("tools.image_generator_agnes_api._post_json", post):
            result = await provider.generate_single_image("a lighthouse")
        self.assertEqual(post.await_count, 2)
        self.assertEqual(result.fmt, "url")

    async def test_retries_5xx_then_succeeds(self):
        post = AsyncMock(side_effect=[
            (503, {"error": {"message": "unavailable"}}),
            (200, {"data": [{"url": "https://cdn.example.test/image.png"}]}),
        ])
        provider = AgnesImageProvider(api_key="test-key", max_retries=2, retry_base_delay_seconds=0)
        with patch("tools.image_generator_agnes_api._post_json", post):
            await provider.generate_single_image("a lighthouse")
        self.assertEqual(post.await_count, 2)

    async def test_retries_timeout_then_raises_timeout(self):
        post = AsyncMock(side_effect=asyncio.TimeoutError)
        provider = AgnesImageProvider(api_key="test-key", max_retries=2, retry_base_delay_seconds=0)
        with patch("tools.image_generator_agnes_api._post_json", post):
            with self.assertRaises(asyncio.TimeoutError):
                await provider.generate_single_image("a lighthouse")
        self.assertEqual(post.await_count, 2)

    async def test_rejects_invalid_response_without_retry(self):
        post = AsyncMock(return_value=(200, {"data": [{}]}))
        provider = AgnesImageProvider(api_key="test-key")
        with patch("tools.image_generator_agnes_api._post_json", post):
            with self.assertRaisesRegex(ValueError, r"url or data\[0\].b64_json"):
                await provider.generate_single_image("a lighthouse")
        self.assertEqual(post.await_count, 1)

    def test_agent_factory_selects_agnes_from_image_base_url(self):
        with patch("agent_runtime.vimax_adapters.image_api_key", return_value="test-key"), \
             patch("agent_runtime.vimax_adapters.image_model", return_value="agnes-image-2.1-flash"), \
             patch("agent_runtime.vimax_adapters.image_base_url", return_value="https://apihub.agnes-ai.com/v1"):
            provider = _build_image_generator()

        self.assertIsInstance(provider, AgnesImageProvider)
        self.assertEqual(provider.model, "agnes-image-2.1-flash")


def _encoded_png() -> str:
    buffer = BytesIO()
    Image.new("RGB", (16, 9), "blue").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


if __name__ == "__main__":
    unittest.main()
