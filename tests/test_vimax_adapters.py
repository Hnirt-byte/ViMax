import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from interfaces import Camera, CharacterInScene, ShotBriefDescription, ShotDescription
from agent_runtime.session_index import SessionIndex
from agent_runtime.vimax_adapters import ViMaxAdapters, _persist_waiting_video_capacity_state
from agent_runtime.tools import ToolRuntimeContext
from pipelines.idea2video_pipeline import Idea2VideoPipeline
from pipelines.script2video_pipeline import Script2VideoPipeline
from tools.video_generator_agnes_api import AgnesVideoAPIError, AgnesVideoProvider


class FakeIdeaPipeline:
    def __init__(self, chat_model, image_generator, video_generator, working_dir):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

    async def develop_story(self, idea, user_requirement, quiet=False):
        path = self.working_dir / "story.txt"
        path.write_text("story", encoding="utf-8")
        return "story"

    async def extract_characters(self, story, quiet=False):
        chars = [CharacterInScene(idx=0, identifier_in_scene="Cat", is_visible=True, static_features="black cat", dynamic_features="helmet")]
        (self.working_dir / "characters.json").write_text(json.dumps([c.model_dump() for c in chars]), encoding="utf-8")
        return chars

    async def write_script_based_on_story(self, story, user_requirement, quiet=False):
        script = [{"scene": "cat jumps"}]
        (self.working_dir / "script.json").write_text(json.dumps(script), encoding="utf-8")
        return script




class HangingIdeaPipeline(FakeIdeaPipeline):
    async def develop_story(self, idea, user_requirement, quiet=False):
        await asyncio.sleep(10)
        return "story"



class FakeRevisionModel:
    async def ainvoke(self, prompt):
        return SimpleNamespace(content='[{"idx": 0, "description": "more oppressive"}]')


class FailRenderIdeaPipeline(FakeIdeaPipeline):
    async def __call__(self, idea, user_requirement, style, quiet=False):
        raise RuntimeError("render failed")


class FailRender403IdeaPipeline(FakeIdeaPipeline):
    async def __call__(self, idea, user_requirement, style, quiet=False):
        raise RuntimeError("OpenRouter video create failed with HTTP 403: {'error': {'message': 'Key limit exceeded (total limit). Manage it using token sk-short', 'code': 403}}")


class NoisyRenderIdeaPipeline(FakeIdeaPipeline):
    async def __call__(self, idea, user_requirement, style, quiet=False):
        print("NOISE_FROM_RENDER_PIPELINE")
        final = self.working_dir / "final_video.mp4"
        final.write_text("video", encoding="utf-8")
        return str(final)


class SceneRenderIdeaPipeline(FakeIdeaPipeline):
    portrait_calls = []

    async def generate_character_portraits(self, characters, character_portraits_registry, style, reference_views=None, image_size=None, progress=None):
        self.__class__.portrait_calls.append({"working_dir": self.working_dir, "characters": characters, "style": style, "reference_views": reference_views, "image_size": image_size})
        return character_portraits_registry or {}


class SceneRenderScriptPipeline:
    render_calls = []

    def __init__(self, chat_model, image_generator, video_generator, working_dir):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

    async def __call__(self, script, user_requirement, style, characters=None, character_portraits_registry=None, quiet=False, progress=None, render_options=None):
        self.__class__.render_calls.append({
            "working_dir": self.working_dir,
            "script": script,
            "user_requirement": user_requirement,
            "style": style,
            "characters": characters,
            "character_portraits_registry": character_portraits_registry,
            "render_options": render_options,
        })
        final = self.working_dir / "final_video.mp4"
        final.write_text("scene-video", encoding="utf-8")
        return str(final)


class QueueFullSceneRenderScriptPipeline(SceneRenderScriptPipeline):
    async def __call__(self, script, user_requirement, style, characters=None, character_portraits_registry=None, quiet=False, progress=None, render_options=None):
        shot_dir = self.working_dir / "shots" / "0"
        shot_dir.mkdir(parents=True, exist_ok=True)
        (shot_dir / "first_frame.png").write_bytes(b"cached-frame")
        (shot_dir / "video_request.json").write_text(json.dumps({
            "schema_version": 1,
            "shot_idx": 0,
            "provider": "AgnesVideoProvider",
            "model": "agnes-video-2.5-flash",
            "prompt": "cached prompt",
            "reference_image_paths": ["shots/0/first_frame.png"],
            "parameters": {"aspect_ratio": "9:16", "resolution": "720P", "seconds": 5},
            "output_path": "shots/0/video.mp4",
            "status": "submitting",
            "last_attempt": None,
            "submit_attempts": 0,
            "video_id": None,
        }), encoding="utf-8")
        raise AgnesVideoAPIError(503, {"code": "video_queue_full", "message": "queue full"})


class FakeScriptPipeline:
    def __init__(self, chat_model, image_generator, video_generator, working_dir):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

    async def plan_text_artifacts(self, script, user_requirement, style, characters=None, progress=None, quiet=False):
        if progress:
            progress("design_storyboard", "Designing storyboard", {})
            progress("decompose_shots", "Decomposing shot visual descriptions", {"shot_count": 1})
            progress("construct_camera_tree", "Constructing camera tree", {"shot_count": 1})
        (self.working_dir / "storyboard.json").write_text("[]", encoding="utf-8")
        (self.working_dir / "camera_tree.json").write_text("[]", encoding="utf-8")
        shot_dir = self.working_dir / "shots" / "0"
        shot_dir.mkdir(parents=True, exist_ok=True)
        (shot_dir / "shot_description.json").write_text("{}", encoding="utf-8")
        if characters:
            (self.working_dir / "characters.json").write_text(json.dumps([c.model_dump() for c in characters]), encoding="utf-8")
        return {}




class FailingScriptPipeline(FakeScriptPipeline):
    async def plan_text_artifacts(self, script, user_requirement, style, characters=None, progress=None, quiet=False):
        if progress:
            progress("design_storyboard", "Designing storyboard", {})
        raise RuntimeError("storyboard failed")


class FakeInitChatModel:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return object()


class Script2VideoPlanningProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_plan_text_artifacts_emits_progress_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Script2VideoPipeline(chat_model=object(), image_generator=object(), video_generator=object(), working_dir=tmp)
            chars = [CharacterInScene(idx=0, identifier_in_scene="Cat", is_visible=True, static_features="black cat", dynamic_features="helmet")]
            storyboard = [ShotBriefDescription(idx=0, is_last=True, cam_idx=0, visual_desc="cat jumps", audio_desc="wind")]
            shot = ShotDescription(idx=0, is_last=True, cam_idx=0, visual_desc="cat jumps", variation_type="small", variation_reason="simple motion", ff_desc="cat starts", ff_vis_char_idxs=[0], lf_desc="cat lands", lf_vis_char_idxs=[0], motion_desc="cat jumps", audio_desc="wind")
            camera = [Camera(idx=0, active_shot_idxs=[0])]

            async def design_storyboard(script, characters, user_requirement, quiet=False):
                return storyboard

            async def decompose_visual_descriptions(shot_brief_descriptions, characters, quiet=False):
                return [shot]

            async def construct_camera_tree(shot_descriptions, quiet=False):
                return camera

            pipeline.design_storyboard = design_storyboard
            pipeline.decompose_visual_descriptions = decompose_visual_descriptions
            pipeline.construct_camera_tree = construct_camera_tree
            events = []
            await pipeline.plan_text_artifacts("script", "req", "style", characters=chars, progress=lambda stage, message, metadata=None: events.append(stage))
            self.assertEqual(events, ["extract_characters", "design_storyboard", "decompose_shots", "construct_camera_tree"])


    async def test_idea_pipeline_quiet_suppresses_text_planning_prints(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Idea2VideoPipeline(chat_model=object(), image_generator=object(), video_generator=object(), working_dir=tmp)

            async def develop_story(idea, user_requirement):
                return "story"

            pipeline.screenwriter = SimpleNamespace(develop_story=develop_story)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = await pipeline.develop_story("idea", "req", quiet=True)
            self.assertEqual(result, "story")
            self.assertEqual(stdout.getvalue(), "")


class ViMaxAdapterTests(unittest.IsolatedAsyncioTestCase):
    def test_build_chat_model_uses_bounded_init_chat_model_kwargs(self):
        fake = FakeInitChatModel()
        with patch.dict("os.environ", {
            "VIMAX_LLM_API_KEY": "test-key",
            "VIMAX_LLM_MODEL": "test-model",
            "VIMAX_LLM_BASE_URL": "https://example.invalid/v1",
            "VIMAX_LLM_REQUEST_TIMEOUT_SECONDS": "12",
            "VIMAX_NARRATIVE_MAX_TOKENS": "1234",
        }), patch("agent_runtime.vimax_adapters.init_chat_model", fake):
            from agent_runtime.vimax_adapters import _build_chat_model

            _build_chat_model()

        self.assertEqual(fake.calls[0]["model"], "test-model")
        self.assertEqual(fake.calls[0]["base_url"], "https://example.invalid/v1")
        self.assertEqual(fake.calls[0]["timeout"], 12.0)
        self.assertEqual(fake.calls[0]["max_retries"], 0)
        self.assertEqual(fake.calls[0]["max_completion_tokens"], 1234)


    async def test_narrative_planning_uses_text_only_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", FakeIdeaPipeline), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", FakeScriptPipeline):
                result = await adapter.vimax_narrative_planning({"idea": "moon cat", "user_requirement": "short", "style": "anime"})
            self.assertTrue(result.ok)
            payload = json.loads(result.content)
            self.assertTrue(payload["ready_for_render"])
            root = Path(tmp) / payload["working_dir"]
            self.assertTrue((root / "idea2video" / "scene_0" / "storyboard.json").exists())
            self.assertTrue((root / "idea2video" / "scene_0" / "camera_tree.json").exists())
            self.assertTrue((root / "idea2video" / "scene_0" / "shots" / "0" / "shot_description.json").exists())
            self.assertFalse((root / "script2video" / "storyboard.json").exists())
            self.assertFalse((root / "script2video" / "final_video.mp4").exists())


    async def test_script_mode_persists_source_script_for_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            adapter = ViMaxAdapters(Path(tmp), index)
            script = "A red ball rolls across a white table."
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", FakeScriptPipeline):
                result = await adapter.vimax_narrative_planning({"script": script, "user_requirement": "one shot"})
            self.assertTrue(result.ok)
            payload = json.loads(result.content)
            root = Path(tmp) / payload["working_dir"]
            self.assertEqual((root / "script2video" / "script.txt").read_text(encoding="utf-8"), script)
            self.assertEqual(index.artifact_checklist(payload["session_id"])["script2video/script.txt"], True)
            from agent_runtime.vimax_adapters import _load_script_text
            self.assertEqual(_load_script_text(root), script)


    async def test_narrative_planning_forwards_pipeline_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            adapter = ViMaxAdapters(Path(tmp), index)
            events = []
            runtime = ToolRuntimeContext("vimax_narrative_planning", "vimax_narrative_planning", turn_id="turn-test", progress_callback=events.append)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", FakeIdeaPipeline), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", FakeScriptPipeline):
                result = await adapter.vimax_narrative_planning({"idea": "moon cat"}, runtime)
            self.assertTrue(result.ok)
            stages = [event["progress"]["stage"] for event in events if event.get("type") == "tool_progress"]
            self.assertIn("initializing_llm", stages)
            self.assertIn("develop_story", stages)
            self.assertIn("design_storyboard", stages)
            self.assertIn("decompose_shots", stages)
            self.assertIn("construct_camera_tree", stages)


    async def test_plan_scene_failure_marks_session_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", FakeIdeaPipeline), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", FailingScriptPipeline):
                result = await adapter.vimax_narrative_planning({"idea": "moon cat"})
            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "recoverable_planning_step_failed")
            self.assertTrue(result.metadata["retryable"])
            session = index.active()
            self.assertEqual(session["stage"], "error")
            self.assertIn("storyboard failed", session["summary"])


    async def test_narrative_planning_timeout_marks_session_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch.dict("os.environ", {"VIMAX_NARRATIVE_STEP_TIMEOUT_SECONDS": "0.01"}), \
                 patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", HangingIdeaPipeline):
                result = await adapter.vimax_narrative_planning({"idea": "moon cat"})
            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "recoverable_planning_step_failed")
            session = index.active()
            self.assertIsNotNone(session)
            self.assertEqual(session["stage"], "error")
            self.assertIn("timed out", session["summary"])



    async def test_active_session_without_new_input_continues_existing_idea(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="moon cat", user_requirement="short", style="anime")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()),                  patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", FakeIdeaPipeline),                  patch("agent_runtime.vimax_adapters.Script2VideoPipeline", FakeScriptPipeline):
                result = await adapter.vimax_narrative_planning({})
            self.assertTrue(result.ok)
            payload = json.loads(result.content)
            self.assertEqual(payload["session_id"], record["session_id"])
            self.assertEqual(index.active()["session_id"], record["session_id"])


    async def test_active_session_continuation_preserves_existing_style(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="moon cat", user_requirement="short", style="anime")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()),                  patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", FakeIdeaPipeline),                  patch("agent_runtime.vimax_adapters.Script2VideoPipeline", FakeScriptPipeline):
                result = await adapter.vimax_narrative_planning({"session_id": record["session_id"]})
            self.assertTrue(result.ok)
            self.assertEqual(index.get(record["session_id"])["style"], "anime")

    async def test_new_idea_creates_new_session_instead_of_reusing_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", FakeIdeaPipeline), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", FakeScriptPipeline):
                first = await adapter.vimax_narrative_planning({"idea": "moon cat"})
                second = await adapter.vimax_narrative_planning({"idea": "ocean robot"})
            self.assertNotEqual(json.loads(first.content)["session_id"], json.loads(second.content)["session_id"])

    async def test_new_idea_initializes_named_empty_active_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            empty = index.create(project_name="00")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", FakeIdeaPipeline), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", FakeScriptPipeline):
                result = await adapter.vimax_narrative_planning({"idea": "moon cat"})
            self.assertTrue(result.ok)
            payload = json.loads(result.content)
            self.assertEqual(payload["session_id"], empty["session_id"])
            self.assertEqual(index.active()["project_name"], "00")
            self.assertEqual(index.active()["idea"], "moon cat")
            self.assertEqual(len(index.load()["sessions"]), 1)


    async def test_explicit_session_with_different_idea_creates_new_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            old = index.create(idea="old cat")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", FakeIdeaPipeline), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", FakeScriptPipeline):
                result = await adapter.vimax_narrative_planning({"session_id": old["session_id"], "idea": "new robot"})
            self.assertTrue(result.ok)
            payload = json.loads(result.content)
            self.assertNotEqual(payload["session_id"], old["session_id"])
            self.assertEqual(index.get(payload["session_id"])["idea"], "new robot")

    async def test_revision_mode_rewrites_existing_artifact_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="x")
            target = Path(tmp) / record["working_dir"] / "idea2video" / "scene_0" / "storyboard.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('[{"idx": 0, "description": "calm"}]', encoding="utf-8")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=FakeRevisionModel()):
                result = await adapter.vimax_narrative_planning({"revision_target": "idea2video/scene_0/storyboard.json", "revision_instruction": "make it oppressive"})
            self.assertTrue(result.ok)
            self.assertIn("more oppressive", target.read_text(encoding="utf-8"))
            self.assertTrue((Path(tmp) / ".vimax" / "logs" / "revisions.jsonl").exists())
            self.assertTrue(index.get(record["session_id"])["stale"]["final_video"])


    async def test_revision_missing_instruction_marks_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="x")
            target = Path(tmp) / record["working_dir"] / "idea2video" / "scene_0" / "storyboard.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('[]', encoding="utf-8")
            adapter = ViMaxAdapters(Path(tmp), index)
            result = await adapter.vimax_narrative_planning({"revision_target": "idea2video/scene_0/storyboard.json"})
            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "missing_revision_instruction")
            self.assertEqual(index.get(record["session_id"])["stage"], "error")


    async def test_revision_missing_target_marks_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="x")
            adapter = ViMaxAdapters(Path(tmp), index)
            result = await adapter.vimax_narrative_planning({"revision_target": "idea2video/scene_0/missing.json", "revision_instruction": "change it"})
            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "dependency_missing")
            self.assertEqual(index.get(record["session_id"])["stage"], "error")

    async def test_render_setup_failure_marks_session_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="x")
            root = Path(tmp) / record["working_dir"] / "idea2video"
            (root / "scene_0" / "shots" / "0").mkdir(parents=True, exist_ok=True)
            (root / "story.txt").write_text("story", encoding="utf-8")
            (root / "characters.json").write_text("[]", encoding="utf-8")
            (root / "script.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "storyboard.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "camera_tree.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "shots" / "0" / "shot_description.json").write_text("{}", encoding="utf-8")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", side_effect=RuntimeError("missing key")):
                result = await adapter.vimax_render_video({})
            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "render_failed")
            self.assertIn("missing key", result.content)
            self.assertEqual(index.get(record["session_id"])["stage"], "error")

    async def test_render_failure_marks_session_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="x")
            root = Path(tmp) / record["working_dir"] / "idea2video"
            (root / "scene_0" / "shots" / "0").mkdir(parents=True, exist_ok=True)
            (root / "story.txt").write_text("story", encoding="utf-8")
            (root / "characters.json").write_text("[]", encoding="utf-8")
            (root / "script.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "storyboard.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "camera_tree.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "shots" / "0" / "shot_description.json").write_text("{}", encoding="utf-8")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", FailRenderIdeaPipeline):
                result = await adapter.vimax_render_video({})
            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "render_failed")
            self.assertIn("render failed", result.content)
            self.assertEqual(index.get(record["session_id"])["stage"], "error")
            status_path = Path(tmp) / record["working_dir"] / "render_status.json"
            events_path = Path(tmp) / record["working_dir"] / "render_events.jsonl"
            self.assertTrue(status_path.exists())
            self.assertTrue(events_path.exists())
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "error")
            self.assertEqual(status["error_type"], "render_failed")

    async def test_render_403_key_limit_is_non_retryable_and_sanitized(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="x")
            root = Path(tmp) / record["working_dir"] / "idea2video"
            (root / "scene_0" / "shots" / "0").mkdir(parents=True, exist_ok=True)
            (root / "story.txt").write_text("story", encoding="utf-8")
            (root / "characters.json").write_text("[]", encoding="utf-8")
            (root / "script.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "storyboard.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "camera_tree.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "shots" / "0" / "shot_description.json").write_text("{}", encoding="utf-8")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", FailRender403IdeaPipeline):
                result = await adapter.vimax_render_video({})
            self.assertFalse(result.ok)
            self.assertFalse(result.metadata["retryable"])
            self.assertIn("<redacted>", result.metadata["error"])
            self.assertNotIn("sk-short", result.metadata["error"])
            status = json.loads((Path(tmp) / record["working_dir"] / "render_status.json").read_text(encoding="utf-8"))
            self.assertFalse(status["retryable"])
            self.assertNotIn("sk-short", status["error"])


    async def test_render_pipeline_stdout_is_suppressed(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="x", style="anime")
            root = Path(tmp) / record["working_dir"] / "idea2video"
            (root / "scene_0" / "shots" / "0").mkdir(parents=True, exist_ok=True)
            (root / "story.txt").write_text("story", encoding="utf-8")
            (root / "characters.json").write_text("[]", encoding="utf-8")
            (root / "script.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "storyboard.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "camera_tree.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "shots" / "0" / "shot_description.json").write_text("{}", encoding="utf-8")
            adapter = ViMaxAdapters(Path(tmp), index)
            stdout = io.StringIO()
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()),                  patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()),                  patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()),                  patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", NoisyRenderIdeaPipeline),                  contextlib.redirect_stdout(stdout):
                result = await adapter.vimax_render_video({})
            self.assertTrue(result.ok)
            self.assertNotIn("NOISE_FROM_RENDER_PIPELINE", stdout.getvalue())

    async def test_render_dependency_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            index.create(idea="x")
            adapter = ViMaxAdapters(Path(tmp), index)
            result = await adapter.vimax_render_video({})
            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "dependency_missing")

    async def test_render_scene_only_uses_selected_idea_scene_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="paper airplane", user_requirement="vertical", style="cinematic")
            idea_dir = Path(tmp) / record["working_dir"] / "idea2video"
            idea_dir.mkdir(parents=True, exist_ok=True)
            (idea_dir / "story.txt").write_text("story", encoding="utf-8")
            (idea_dir / "characters.json").write_text("[]", encoding="utf-8")
            (idea_dir / "script.json").write_text(json.dumps(["scene zero", "scene one"]), encoding="utf-8")
            for scene_name in ("scene_0", "scene_1"):
                scene_dir = idea_dir / scene_name / "shots" / "0"
                scene_dir.mkdir(parents=True, exist_ok=True)
                (scene_dir.parent.parent / "storyboard.json").write_text("[]", encoding="utf-8")
                (scene_dir.parent.parent / "camera_tree.json").write_text("[]", encoding="utf-8")
                (scene_dir / "shot_description.json").write_text("{}", encoding="utf-8")
            SceneRenderIdeaPipeline.portrait_calls = []
            SceneRenderScriptPipeline.render_calls = []
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", SceneRenderIdeaPipeline), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", SceneRenderScriptPipeline):
                result = await adapter.vimax_render_scene({"session_id": record["session_id"], "scene_id": "scene_1"})
            self.assertTrue(result.ok)
            self.assertEqual(result.metadata["scene_id"], "scene_1")
            self.assertEqual(len(SceneRenderScriptPipeline.render_calls), 1)
            self.assertEqual(SceneRenderScriptPipeline.render_calls[0]["working_dir"], idea_dir / "scene_1")
            self.assertEqual(SceneRenderScriptPipeline.render_calls[0]["script"], "scene one")
            self.assertTrue((idea_dir / "scene_1" / "final_video.mp4").exists())
            self.assertFalse((idea_dir / "scene_0" / "final_video.mp4").exists())

    def test_waiting_state_keeps_completed_jobs_and_updates_only_saturated_shot(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = Path(tmp) / "scene_1"
            for shot_idx, status, attempts in ((10, "submitting", 0), (2, "completed", 1)):
                shot_dir = scene_dir / "shots" / str(shot_idx)
                shot_dir.mkdir(parents=True)
                (shot_dir / "video_request.json").write_text(json.dumps({
                    "schema_version": 1,
                    "shot_idx": shot_idx,
                    "provider": "AgnesVideoProvider",
                    "model": "agnes-video-2.5-flash",
                    "prompt": f"shot {shot_idx}",
                    "reference_image_paths": [f"shots/{shot_idx}/first_frame.png"],
                    "parameters": {"seconds": 5},
                    "output_path": f"shots/{shot_idx}/video.mp4",
                    "status": status,
                    "last_attempt": None,
                    "submit_attempts": attempts,
                    "video_id": None,
                }), encoding="utf-8")
            error = AgnesVideoAPIError(503, {"code": "video_queue_full"})
            error.queue_attempts = 3
            error.shot_idx = 10

            state = _persist_waiting_video_capacity_state(
                session_id="session-a", scene_id="scene_1", scene_dir=scene_dir, error=error,
            )

            self.assertEqual([job["shot_idx"] for job in state["jobs"]], [2, 10])
            completed, waiting = state["jobs"]
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["submit_attempts"], 1)
            self.assertEqual(waiting["status"], "waiting_for_video_capacity")
            self.assertEqual(waiting["submit_attempts"], 3)

    def test_waiting_state_rejects_symlinked_shot_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_dir = root / "scene_1"
            external_shot = root / "outside" / "0"
            external_shot.mkdir(parents=True)
            request_path = external_shot / "video_request.json"
            original = {
                "schema_version": 1, "shot_idx": 0, "provider": "AgnesVideoProvider",
                "model": "agnes-video-2.5-flash", "prompt": "cached", "reference_image_paths": [],
                "parameters": {}, "output_path": "shots/0/video.mp4", "status": "submitting",
                "last_attempt": None, "submit_attempts": 0, "video_id": None,
            }
            request_path.write_text(json.dumps(original), encoding="utf-8")
            (scene_dir / "shots").mkdir(parents=True)
            (scene_dir / "shots" / "0").symlink_to(external_shot, target_is_directory=True)
            error = AgnesVideoAPIError(503, {"code": "video_queue_full"})
            error.shot_idx = 0

            with self.assertRaisesRegex(ValueError, "escapes the scene directory"):
                _persist_waiting_video_capacity_state(
                    session_id="session-a", scene_id="scene_1", scene_dir=scene_dir, error=error,
                )

            self.assertEqual(json.loads(request_path.read_text(encoding="utf-8")), original)

    async def test_render_scene_queue_full_returns_waiting_state_with_persisted_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="paper airplane", user_requirement="vertical", style="cinematic")
            idea_dir = Path(tmp) / record["working_dir"] / "idea2video"
            scene_dir = idea_dir / "scene_1" / "shots" / "0"
            scene_dir.mkdir(parents=True, exist_ok=True)
            (idea_dir / "characters.json").write_text("[]", encoding="utf-8")
            (idea_dir / "script.json").write_text(json.dumps(["scene zero", "scene one"]), encoding="utf-8")
            (scene_dir.parent.parent / "storyboard.json").write_text("[]", encoding="utf-8")
            (scene_dir.parent.parent / "camera_tree.json").write_text("[]", encoding="utf-8")
            (scene_dir / "shot_description.json").write_text("{}", encoding="utf-8")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", SceneRenderIdeaPipeline), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", QueueFullSceneRenderScriptPipeline):
                result = await adapter.vimax_render_scene({"session_id": record["session_id"], "scene_id": "scene_1"})

            self.assertTrue(result.ok)
            self.assertEqual(result.metadata["status"], "waiting_for_video_capacity")
            self.assertEqual(index.get(record["session_id"])["stage"], "waiting_for_video_capacity")
            state = json.loads((idea_dir / "scene_1" / "waiting_video_capacity.json").read_text(encoding="utf-8"))
            self.assertEqual(state["session_id"], record["session_id"])
            self.assertEqual(state["scene_id"], "scene_1")
            self.assertEqual(state["status"], "waiting_for_video_capacity")
            self.assertEqual(state["jobs"][0]["model"], "agnes-video-2.5-flash")
            self.assertEqual(state["jobs"][0]["reference_image_paths"], ["shots/0/first_frame.png"])
            self.assertIsNone(state["jobs"][0]["video_id"])

    async def test_resume_rejects_malformed_waiting_state_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="paper airplane", user_requirement="vertical", style="cinematic")
            scene_dir = Path(tmp) / record["working_dir"] / "idea2video" / "scene_0"
            scene_dir.mkdir(parents=True)
            (scene_dir / "waiting_video_capacity.json").write_text("{not-json", encoding="utf-8")
            adapter = ViMaxAdapters(Path(tmp), index)

            result = await adapter.vimax_resume_waiting_scene({"session_id": record["session_id"], "scene_id": "scene_0"})

            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "invalid_waiting_state")

    async def test_resume_rejects_incomplete_expected_shot_set_before_provider_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="paper airplane")
            scene_dir = Path(tmp) / record["working_dir"] / "idea2video" / "scene_1"
            scene_dir.mkdir(parents=True)
            job = {
                "schema_version": 1, "shot_idx": 0, "provider": "AgnesVideoProvider",
                "model": "agnes-video-2.5-flash", "prompt": "cached", "reference_image_paths": [],
                "parameters": {}, "output_path": "shots/0/video.mp4", "status": "waiting_for_video_capacity",
                "last_attempt": None, "submit_attempts": 1, "video_id": None,
            }
            state = {
                "schema_version": 1, "session_id": record["session_id"], "scene_id": "scene_1",
                "expected_shot_indices": [0, 1], "jobs": [job],
            }
            (scene_dir / "waiting_video_capacity.json").write_text(json.dumps(state), encoding="utf-8")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_video_generator", side_effect=AssertionError("provider must not initialize")):
                result = await adapter.vimax_resume_waiting_scene({"session_id": record["session_id"], "scene_id": "scene_1"})

            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "incomplete_waiting_state")

    async def test_resume_never_resubmits_submission_outcome_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="paper airplane")
            scene_dir = Path(tmp) / record["working_dir"] / "idea2video" / "scene_1"
            shot_dir = scene_dir / "shots" / "0"
            shot_dir.mkdir(parents=True)
            job = {
                "schema_version": 1, "shot_idx": 0, "provider": "AgnesVideoProvider",
                "model": "agnes-video-2.5-flash", "prompt": "cached", "reference_image_paths": [],
                "parameters": {}, "output_path": "shots/0/video.mp4", "status": "submission_outcome_unknown",
                "last_attempt": "2026-09-25T09:00:00", "submit_attempts": 1, "video_id": None,
            }
            (shot_dir / "video_request.json").write_text(json.dumps(job), encoding="utf-8")
            state = {
                "schema_version": 1, "session_id": record["session_id"], "scene_id": "scene_1",
                "expected_shot_indices": [0], "jobs": [job],
            }
            (scene_dir / "waiting_video_capacity.json").write_text(json.dumps(state), encoding="utf-8")
            provider = AgnesVideoProvider(api_key="test-key", model="agnes-video-2.5-flash")
            provider.generate_single_video = AsyncMock(side_effect=AssertionError("must not resubmit"))
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_video_generator", return_value=provider):
                first = await adapter.vimax_resume_waiting_scene({"session_id": record["session_id"], "scene_id": "scene_1"})
                second = await adapter.vimax_resume_waiting_scene({"session_id": record["session_id"], "scene_id": "scene_1"})

            self.assertFalse(first.ok)
            self.assertFalse(second.ok)
            provider.generate_single_video.assert_not_awaited()

    async def test_resume_processes_all_waiting_shots_before_assembly(self):
        class SavedVideo:
            def save(self, path):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_bytes(b"resumed-video")

        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="paper airplane")
            scene_dir = Path(tmp) / record["working_dir"] / "idea2video" / "scene_1"
            jobs = []
            for shot_idx in (0, 1, 2):
                shot_dir = scene_dir / "shots" / str(shot_idx)
                shot_dir.mkdir(parents=True)
                (shot_dir / "first_frame.png").write_bytes(b"cached-frame")
                job = {
                    "schema_version": 1, "shot_idx": shot_idx, "provider": "AgnesVideoProvider",
                    "model": "agnes-video-2.5-flash", "prompt": f"cached {shot_idx}",
                    "reference_image_paths": [f"shots/{shot_idx}/first_frame.png"],
                    "parameters": {"seconds": 5}, "output_path": f"shots/{shot_idx}/video.mp4",
                    "status": "waiting_for_video_capacity", "last_attempt": None,
                    "submit_attempts": 1, "video_id": None,
                }
                (shot_dir / "video_request.json").write_text(json.dumps(job), encoding="utf-8")
                jobs.append(job)
            state = {
                "schema_version": 1, "session_id": record["session_id"], "scene_id": "scene_1",
                "expected_shot_indices": [0, 1, 2], "jobs": jobs,
            }
            (scene_dir / "waiting_video_capacity.json").write_text(json.dumps(state), encoding="utf-8")
            provider = AgnesVideoProvider(api_key="test-key", model="agnes-video-2.5-flash")

            async def generate(**kwargs):
                kwargs["task_created_callback"](f"task-{kwargs['prompt'][-1]}", "agnes-video-2.5-flash")
                return SavedVideo()

            provider.generate_single_video = AsyncMock(side_effect=generate)
            adapter = ViMaxAdapters(Path(tmp), index)
            def concatenate(clips, output):
                Path(output).write_bytes(b"final-video")
            with patch("agent_runtime.vimax_adapters._build_video_generator", return_value=provider), \
                 patch("agent_runtime.vimax_adapters.concatenate_video_files", side_effect=concatenate):
                result = await adapter.vimax_resume_waiting_scene({"session_id": record["session_id"], "scene_id": "scene_1"})

            self.assertTrue(result.ok)
            self.assertEqual(provider.generate_single_video.await_count, 3)
            self.assertTrue((scene_dir / "final_video.mp4").exists())
            persisted = json.loads((scene_dir / "waiting_video_capacity.json").read_text(encoding="utf-8"))
            self.assertEqual([job["status"] for job in persisted["jobs"]], ["completed", "completed", "completed"])

    async def test_resume_waiting_scene_polls_persisted_video_id_without_resubmit(self):
        class SavedVideo:
            def save(self, path):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_bytes(b"resumed-video")

        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="paper airplane")
            scene_dir = Path(tmp) / record["working_dir"] / "idea2video" / "scene_1"
            shot_dir = scene_dir / "shots" / "0"
            shot_dir.mkdir(parents=True)
            (shot_dir / "first_frame.png").write_bytes(b"cached-frame")
            request = {
                "schema_version": 1,
                "shot_idx": 0,
                "provider": "AgnesVideoProvider",
                "model": "agnes-video-2.5",
                "prompt": "cached prompt",
                "reference_image_paths": ["shots/0/first_frame.png"],
                "parameters": {"aspect_ratio": "9:16", "resolution": "720P", "seconds": 5},
                "output_path": "shots/0/video.mp4",
                "status": "polling",
                "last_attempt": "2026-09-25T09:00:00",
                "submit_attempts": 3,
                "video_id": "already-submitted",
            }
            (shot_dir / "video_request.json").write_text(json.dumps(request), encoding="utf-8")
            state = {
                "schema_version": 1,
                "session_id": record["session_id"],
                "scene_id": "scene_1",
                "provider": "AgnesVideoProvider",
                "model": "agnes-video-2.5",
                "parameters": request["parameters"],
                "reference_image_paths": request["reference_image_paths"],
                "status": "waiting_for_video_capacity",
                "last_attempt": request["last_attempt"],
                "submit_attempts": 3,
                "video_id": "already-submitted",
                "expected_shot_indices": [0],
                "jobs": [request],
            }
            (scene_dir / "waiting_video_capacity.json").write_text(json.dumps(state), encoding="utf-8")
            provider = AgnesVideoProvider(api_key="test-key", model="agnes-video-2.5-flash")
            provider.poll_existing_task = AsyncMock(return_value=SavedVideo())
            provider.generate_single_video = AsyncMock(side_effect=AssertionError("must not resubmit"))
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", side_effect=AssertionError("chat must not initialize")), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", side_effect=AssertionError("image must not initialize")), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=provider):
                result = await adapter.vimax_resume_waiting_scene({"session_id": record["session_id"], "scene_id": "scene_1"})

            self.assertTrue(result.ok)
            provider.poll_existing_task.assert_awaited_once_with("already-submitted", "agnes-video-2.5", progress=None)
            provider.generate_single_video.assert_not_awaited()
            self.assertTrue((scene_dir / "shots" / "0" / "video.mp4").exists())
            self.assertTrue((scene_dir / "final_video.mp4").exists())

    async def test_render_scene_reuses_existing_scene_video_without_initializing_providers(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="paper airplane")
            scene_dir = Path(tmp) / record["working_dir"] / "idea2video" / "scene_0"
            scene_dir.mkdir(parents=True, exist_ok=True)
            (scene_dir / "final_video.mp4").write_text("cached-scene-video", encoding="utf-8")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", side_effect=AssertionError("chat model must not initialize")), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", side_effect=AssertionError("image provider must not initialize")), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", side_effect=AssertionError("video provider must not initialize")):
                result = await adapter.vimax_render_scene({"session_id": record["session_id"], "scene_id": "scene_0"})
            self.assertTrue(result.ok)
            self.assertTrue(result.metadata["render_cached"])
            self.assertFalse(result.metadata["render_started"])
            self.assertEqual(result.metadata["scene_video_path"], f"{record['working_dir']}/idea2video/scene_0/final_video.mp4")

    async def test_render_scene_forwards_render_plan_to_portraits_and_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="paper airplane", user_requirement="vertical", style="cinematic")
            idea_dir = Path(tmp) / record["working_dir"] / "idea2video"
            scene_dir = idea_dir / "scene_0" / "shots" / "0"
            scene_dir.mkdir(parents=True, exist_ok=True)
            (idea_dir / "characters.json").write_text("[]", encoding="utf-8")
            (idea_dir / "script.json").write_text(json.dumps(["scene zero"]), encoding="utf-8")
            (scene_dir.parent.parent / "storyboard.json").write_text("[]", encoding="utf-8")
            (scene_dir.parent.parent / "camera_tree.json").write_text("[]", encoding="utf-8")
            (scene_dir / "shot_description.json").write_text("{}", encoding="utf-8")
            (idea_dir / "render_plan.json").write_text(json.dumps({
                "target": {
                    "aspect_ratio": "9:16",
                    "resolution": "720P",
                    "duration_seconds_per_scene": 5,
                    "image_size": "720x1280",
                },
                "shared_character_reference_views": ["front"],
            }), encoding="utf-8")
            SceneRenderIdeaPipeline.portrait_calls = []
            SceneRenderScriptPipeline.render_calls = []
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", SceneRenderIdeaPipeline), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", SceneRenderScriptPipeline):
                result = await adapter.vimax_render_scene({"session_id": record["session_id"], "scene_id": "scene_0"})
            self.assertTrue(result.ok)
            self.assertEqual(SceneRenderIdeaPipeline.portrait_calls[0]["reference_views"], ["front"])
            self.assertEqual(SceneRenderIdeaPipeline.portrait_calls[0]["image_size"], "720x1280")
            self.assertEqual(SceneRenderScriptPipeline.render_calls[0]["render_options"], {
                "aspect_ratio": "9:16",
                "resolution": "720P",
                "seconds": 5,
                "image_size": "720x1280",
            })
