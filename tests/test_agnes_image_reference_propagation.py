import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image

from interfaces.image_output import ImageOutput
from tools.video_generator_agnes_api import AgnesVideoProvider


class AgnesImageReferencePropagationTests(unittest.IsolatedAsyncioTestCase):
    def test_image_output_keeps_local_file_and_public_source_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "first_frame.png"
            output = ImageOutput(
                fmt="pil",
                ext="png",
                data=Image.new("RGB", (9, 16), "red"),
                source_url="https://images.example.test/agnes-first-frame.png",
            )

            output.save(str(image_path))

            self.assertTrue(image_path.exists())
            metadata = json.loads(Path(f"{image_path}.source.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata, {"source_url": "https://images.example.test/agnes-first-frame.png"})

    async def test_agnes_25_uses_public_url_preserved_by_first_frame_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            first_frame = Path(tmp) / "first_frame.png"
            ImageOutput(
                fmt="pil",
                ext="png",
                data=Image.new("RGB", (9, 16), "red"),
                source_url="https://images.example.test/agnes-first-frame.png",
            ).save(str(first_frame))
            post = AsyncMock(return_value=(200, {"video_id": "task-1"}))
            get = AsyncMock(return_value=(200, {"status": "completed", "url": "https://cdn.example.test/video.mp4"}))
            download = AsyncMock(return_value=(200, b"video"))
            provider = AgnesVideoProvider(api_key="test-key", model="agnes-video-2.5-flash", poll_interval_seconds=0)

            with patch("tools.video_generator_agnes_api._post_json", post), \
                 patch("tools.video_generator_agnes_api._get_json", get), \
                 patch("tools.video_generator_agnes_api._get_bytes", download):
                await provider.generate_single_video("A red airplane glides.", [str(first_frame)])

            payload = post.await_args.kwargs["payload"]
            self.assertEqual(payload["model"], "agnes-video-2.5-flash")
            self.assertEqual(payload["first_frame"], "https://images.example.test/agnes-first-frame.png")

    async def test_direct_public_url_has_priority_over_artifact_lookup(self):
        post = AsyncMock(return_value=(200, {"video_id": "task-1"}))
        get = AsyncMock(return_value=(200, {"status": "completed", "url": "https://cdn.example.test/video.mp4"}))
        download = AsyncMock(return_value=(200, b"video"))
        provider = AgnesVideoProvider(api_key="test-key", model="agnes-video-2.5-flash", poll_interval_seconds=0)

        with patch("tools.video_generator_agnes_api._post_json", post), \
             patch("tools.video_generator_agnes_api._get_json", get), \
             patch("tools.video_generator_agnes_api._get_bytes", download):
            await provider.generate_single_video("A red airplane glides.", ["https://images.example.test/direct.png"])

        self.assertEqual(post.await_args.kwargs["payload"]["first_frame"], "https://images.example.test/direct.png")

    async def test_local_file_without_public_url_fails_before_submit_with_active_model_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            first_frame = Path(tmp) / "first_frame.png"
            Image.new("RGB", (9, 16), "red").save(first_frame)
            post = AsyncMock()
            provider = AgnesVideoProvider(api_key="test-key", model="agnes-video-2.5-flash")

            with patch("tools.video_generator_agnes_api._post_json", post):
                with self.assertRaisesRegex(ValueError, r"Agnes Video agnes-video-2\.5-flash image references must be publicly accessible http\(s\) URLs"):
                    await provider.generate_single_video("A red airplane glides.", [str(first_frame)])

            post.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
