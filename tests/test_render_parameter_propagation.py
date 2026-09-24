import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from interfaces import Camera, CharacterInScene, ShotDescription
from pipelines.idea2video_pipeline import Idea2VideoPipeline
from pipelines.script2video_pipeline import Script2VideoPipeline


class _SavedOutput:
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"mock-output")


def _single_shot() -> ShotDescription:
    return ShotDescription(
        idx=0,
        is_last=True,
        cam_idx=0,
        visual_desc="A red paper airplane over a clear sky.",
        variation_type="small",
        variation_reason="One stable clip per scene.",
        ff_desc="A red paper airplane in portrait composition.",
        ff_vis_char_idxs=[],
        lf_desc="A red paper airplane in portrait composition.",
        lf_vis_char_idxs=[],
        motion_desc="The red paper airplane glides upward.",
        audio_desc="Soft wind.",
    )


class RenderParameterPropagationTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_frame_uses_explicit_portrait_image_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_generator = MagicMock()
            image_generator.generate_single_image = AsyncMock(return_value=_SavedOutput())
            pipeline = Script2VideoPipeline(
                chat_model=MagicMock(),
                image_generator=image_generator,
                video_generator=MagicMock(),
                working_dir=tmp,
            )
            pipeline.reference_image_selector = MagicMock(
                select_reference_images_and_generate_prompt=AsyncMock(return_value={
                    "reference_image_path_and_text_pairs": [],
                    "text_prompt": "portrait first frame",
                })
            )
            pipeline.frame_events = {0: {"first_frame": asyncio.Event()}}
            (Path(tmp) / "shots" / "0").mkdir(parents=True)

            await pipeline.generate_frames_for_single_camera(
                camera=Camera(idx=0, active_shot_idxs=[0]),
                shot_descriptions=[_single_shot()],
                characters=[],
                character_portraits_registry={},
                priority_shot_idxs=[],
                image_size="720x1280",
            )

            self.assertEqual(image_generator.generate_single_image.await_args.kwargs["size"], "720x1280")

    async def test_main_video_uses_explicit_agnes_render_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_generator = MagicMock()
            video_generator.generate_single_video = AsyncMock(return_value=_SavedOutput())
            pipeline = Script2VideoPipeline(
                chat_model=MagicMock(),
                image_generator=MagicMock(),
                video_generator=video_generator,
                working_dir=tmp,
            )
            pipeline.frame_events = {0: {"first_frame": asyncio.Event()}}
            pipeline.frame_events[0]["first_frame"].set()
            (Path(tmp) / "shots" / "0").mkdir(parents=True)

            await pipeline.generate_video_for_single_shot(
                _single_shot(),
                render_options={"aspect_ratio": "9:16", "resolution": "720P", "seconds": 5},
            )

            kwargs = video_generator.generate_single_video.await_args.kwargs
            self.assertEqual(kwargs["aspect_ratio"], "9:16")
            self.assertEqual(kwargs["resolution"], "720P")
            self.assertEqual(kwargs["seconds"], 5)

    async def test_single_shared_front_reference_uses_one_image_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_generator = MagicMock()
            image_generator.generate_single_image = AsyncMock(return_value=_SavedOutput())
            pipeline = Idea2VideoPipeline(
                chat_model=MagicMock(),
                image_generator=image_generator,
                video_generator=MagicMock(),
                working_dir=tmp,
            )
            airplane = CharacterInScene(
                idx=0,
                identifier_in_scene="Red paper airplane",
                is_visible=True,
                static_features="Matte red folded paper airplane with one white center crease, no logo or text.",
                dynamic_features=None,
            )

            registry = await pipeline.generate_character_portraits(
                characters=[airplane],
                character_portraits_registry=None,
                style="cinematic",
                reference_views=["front"],
                image_size="720x1280",
            )

            self.assertEqual(image_generator.generate_single_image.await_count, 1)
            self.assertEqual(image_generator.generate_single_image.await_args.kwargs["size"], "720x1280")
            self.assertEqual(set(registry["Red paper airplane"]), {"front"})


if __name__ == "__main__":
    unittest.main()
