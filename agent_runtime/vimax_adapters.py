from __future__ import annotations

import asyncio
from datetime import datetime
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from tenacity import RetryError

from interfaces import CharacterInScene
from agents.event_extractor import EventExtractor
from agents.global_information_planner import GlobalInformationPlanner
from agents.novel_compressor import NovelCompressor
from agents.scene_extractor import SceneExtractor
from pipelines.novel2movie_pipeline import Novel2MoviePipeline
from pipelines.idea2video_pipeline import Idea2VideoPipeline
from pipelines.script2video_pipeline import Script2VideoPipeline
from tools.image_generator_agnes_api import AgnesImageProvider
from tools.image_generator_nanobanana_yunwu_api import ImageGeneratorNanobananaYunwuAPI
from tools.image_generator_openrouter_api import ImageGeneratorOpenRouterAPI
from tools.reranker_bge_silicon_api import RerankerBgeSiliconapi
from tools.video_generator_agnes_api import AgnesVideoAPIError, AgnesVideoProvider, VideoOutput, _is_video_queue_full_error, _status_from_response, _video_url_from_response
from tools.video_generator_openrouter_api import VideoGeneratorOpenRouterAPI
from tools.video_generator_veo_yunwu_api import VideoGeneratorVeoYunwuAPI
from utils.video import concatenate_video_files

from .config import api_provider_from_base_url, embedding_api_key, embedding_base_url, embedding_model, embedding_model_provider, image_api_key, image_base_url, image_model, llm_api_key, llm_base_url, llm_model, llm_model_provider, reranker_api_key, reranker_base_url, reranker_model, video_allow_paid_fallback, video_api_key, video_base_url, video_model, video_paid_fallback_model, video_provider
from .models import ToolResult
from .tools import ToolArgumentSchema, ToolRuntimeContext, ToolSpec


class _AutomationTaskCreated(RuntimeError):
    """Internal sentinel: persist the remote id, then return to the short cron pass."""


class _UnavailableGenerator:
    async def generate_single_image(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Image generator is not available in narrative planning mode")

    async def generate_single_video(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Video generator is not available in narrative planning mode")


def build_vimax_adapter_specs(workspace_root: str | Path, session_index: Any) -> list[ToolSpec]:
    adapter = ViMaxAdapters(Path(workspace_root), session_index)
    return [
        ToolSpec(
            name="vimax_narrative_planning",
            description=(
                "Create or revise ViMax structured text artifacts for the active session. "
                "Idea mode writes story, characters, script, and scene-level storyboard/shot_decomposition/camera_tree under idea2video/scene_<idx>/. "
                "Script mode writes characters, storyboard, shot_decomposition, and camera_tree under script2video/. "
                "Pass the active session_id from prompt context when the user is working in the selected project. An empty active session is initialized in place; a different source on a non-empty session creates a new session instead of overwriting existing artifacts. If idea/script/revision_target are omitted and the active session has an idea, continue that session and fill missing structured text artifacts. "
                "It does not generate keyframes, video clips, or final video. Call this before revising storyboard/shots when those artifacts do not exist."
            ),
            handler=adapter.vimax_narrative_planning,
            schema={
                "session_id": ToolArgumentSchema(str, required=False, default=""),
                "idea": ToolArgumentSchema(str, required=False, default=""),
                "script": ToolArgumentSchema(str, required=False, default=""),
                "user_requirement": ToolArgumentSchema(str, required=False, default=""),
                "style": ToolArgumentSchema(str, required=False, default=""),
                "revision_target": ToolArgumentSchema(str, required=False, default=""),
                "revision_instruction": ToolArgumentSchema(str, required=False, default=""),
            },
        ),
        ToolSpec(
            name="vimax_novel_planning",
            description=(
                "Create ViMax structured text artifacts from a novel or novel excerpt. "
                "This writes novel2video/novel, events, relevant_chunks, scenes, and global_information text artifacts. "
                "Use this when the user provides long prose, a novel excerpt, or asks for novel-to-video planning. Pass the active session_id when the user is working in a selected empty project. "
                "It does not generate character portraits, scene videos, or final video."
            ),
            handler=adapter.vimax_novel_planning,
            schema={
                "session_id": ToolArgumentSchema(str, required=False, default=""),
                "novel_text": ToolArgumentSchema(str, required=True),
                "user_requirement": ToolArgumentSchema(str, required=False, default=""),
                "style": ToolArgumentSchema(str, required=False, default=""),
            },
        ),
        ToolSpec(
            name="vimax_render_video",
            description=(
                "Render keyframes, video clips, and final video for the active ViMax session. "
                "This checks that structured text artifacts exist before rendering and reports missing dependencies instead of pretending render started."
            ),
            handler=adapter.vimax_render_video,
            schema={
                "session_id": ToolArgumentSchema(str, required=False, default=""),
                "mode": ToolArgumentSchema(str, required=False, default="foreground"),
                "force": ToolArgumentSchema(bool, required=False, default=False),
            },
        ),
        ToolSpec(
            name="vimax_render_scene",
            description=(
                "Render exactly one planned idea2video scene without rendering or concatenating other scenes. "
                "The selected scene reuses its existing storyboard, shot descriptions, camera tree, frames, clips, and scene final_video.mp4 cache under idea2video/scene_<n>/. "
                "Use only for an idea2video session after narrative planning; scene_id must be an exact value such as scene_0."
            ),
            handler=adapter.vimax_render_scene,
            schema={
                "session_id": ToolArgumentSchema(str, required=False, default=""),
                "scene_id": ToolArgumentSchema(str, required=True),
            },
        ),
        ToolSpec(
            name="vimax_resume_waiting_scene",
            description=(
                "Resume exactly one idea2video scene previously saved as waiting_for_video_capacity. "
                "It reuses persisted video parameters and references, performs no image generation, and polls an existing video_id without submitting a duplicate job."
            ),
            handler=adapter.vimax_resume_waiting_scene,
            schema={
                "session_id": ToolArgumentSchema(str, required=True),
                "scene_id": ToolArgumentSchema(str, required=True),
            },
        ),
    ]


class ViMaxAdapters:
    def __init__(self, workspace_root: Path, session_index: Any) -> None:
        self.workspace_root = workspace_root.resolve()
        self.session_index = session_index

    async def vimax_narrative_planning(self, args: dict[str, Any], runtime: ToolRuntimeContext | None = None) -> ToolResult:
        idea = str(args.get("idea", "") or "").strip()
        script = str(args.get("script", "") or "").strip()
        user_requirement = str(args.get("user_requirement", "") or "").strip()
        requested_style = str(args.get("style", "") or "").strip()
        style = requested_style
        session = self._resolve_session(str(args.get("session_id", "") or ""), idea=idea, script=script, user_requirement=user_requirement, style=requested_style)
        session_id = session["session_id"]
        working_dir = self.session_index.working_dir(session_id)
        idea_dir = working_dir / "idea2video"
        script_dir = working_dir / "script2video"
        idea_dir.mkdir(parents=True, exist_ok=True)
        script_dir.mkdir(parents=True, exist_ok=True)

        if not idea and not script:
            revision_target = str(args.get("revision_target") or "").strip()
            if revision_target:
                return await self._revise_narrative_artifact(session_id, working_dir, revision_target, str(args.get("revision_instruction") or "").strip(), runtime)
            session_idea = str(session.get("idea") or "").strip()
            if session_idea:
                idea = session_idea
                user_requirement = user_requirement or str(session.get("user_requirement") or "").strip()
                style = requested_style or str(session.get("style") or "").strip() or "Cinematic, coherent, 16:9"
            else:
                return ToolResult("vimax_narrative_planning", False, "Provide `idea`, `script`, a revision target, or an active session with an existing idea for narrative planning.", {"error_type": "missing_input", "session_id": session_id})

        style = style or str(session.get("style") or "").strip() or "Cinematic, coherent, 16:9"
        self._update_session_metadata(session_id, idea="", user_requirement="", style=style)

        try:
            self.session_index.update_stage(session_id, "narrative_planning", "Generating structured text artifacts")
            if runtime:
                runtime.emit_progress("Starting narrative planning", stage="starting", metadata={"session_id": session_id})
                await asyncio.sleep(0)
            generated_before = self.session_index.artifact_checklist(session_id)
            if runtime:
                runtime.emit_progress("Initializing bounded chat model", stage="initializing_llm", metadata={"session_id": session_id, "timeout_seconds": _llm_request_timeout_seconds(), "max_tokens": _narrative_max_tokens()})
                await asyncio.sleep(0)
            chat_model = _build_chat_model()
            if runtime:
                runtime.emit_progress("Bounded chat model initialized", stage="chat_model_ready", metadata={"session_id": session_id})
                await asyncio.sleep(0)
            dummy = _UnavailableGenerator()
            # Do not globally redirect stdout/stderr while the JSONL CLI is streaming events.
            # The adapter exposes pipeline progress through explicit tool_progress events instead.
            if idea:
                idea_pipeline = Idea2VideoPipeline(chat_model=chat_model, image_generator=dummy, video_generator=dummy, working_dir=str(idea_dir))
                if runtime:
                    runtime.emit_progress("Idea pipeline initialized", stage="idea_pipeline_ready", metadata={"session_id": session_id})
                    await asyncio.sleep(0)
                story = await _run_planning_step(
                    "Developing story from user idea",
                    "develop_story",
                    idea_pipeline.develop_story(idea=idea, user_requirement=user_requirement, quiet=True),
                    runtime,
                    {"session_id": session_id},
                )
                characters = await _run_planning_step(
                    "Extracting characters from story",
                    "extract_characters",
                    idea_pipeline.extract_characters(story=story, quiet=True),
                    runtime,
                    {"session_id": session_id},
                )
                scene_scripts = await _run_planning_step(
                    "Writing scene scripts from story",
                    "write_script",
                    idea_pipeline.write_script_based_on_story(story=story, user_requirement=user_requirement, quiet=True),
                    runtime,
                    {"session_id": session_id},
                )
                for idx, scene_script in enumerate(scene_scripts if isinstance(scene_scripts, list) else [scene_scripts]):
                    scene_dir = idea_dir / f"scene_{idx}"
                    scene_text = scene_script if isinstance(scene_script, str) else json.dumps(scene_script, ensure_ascii=False, indent=2)
                    script_pipeline = Script2VideoPipeline(chat_model=chat_model, image_generator=dummy, video_generator=dummy, working_dir=str(scene_dir))
                    await _run_planning_step(
                        f"Planning scene {idx} storyboard and shots",
                        "plan_scene",
                        script_pipeline.plan_text_artifacts(script=scene_text, user_requirement=user_requirement, style=style, characters=characters, progress=_pipeline_progress(runtime, session_id, scene_index=idx), quiet=True),
                        runtime,
                        {"session_id": session_id, "scene_index": idx},
                    )
            else:
                (script_dir / "script.txt").write_text(script, encoding="utf-8")
                script_pipeline = Script2VideoPipeline(chat_model=chat_model, image_generator=dummy, video_generator=dummy, working_dir=str(script_dir))
                if runtime:
                    runtime.emit_progress("Script pipeline initialized", stage="script_pipeline_ready", metadata={"session_id": session_id})
                    await asyncio.sleep(0)
                await _run_planning_step(
                    "Planning storyboard and shots from provided script",
                    "plan_script",
                    script_pipeline.plan_text_artifacts(script=script, user_requirement=user_requirement, style=style, progress=_pipeline_progress(runtime, session_id), quiet=True),
                    runtime,
                    {"session_id": session_id},
                )
        except Exception as exc:
            self.session_index.update_stage(session_id, "error", f"Narrative planning failed: {exc}")
            checklist = self.session_index.artifact_checklist(session_id)
            payload = {
                "session_id": session_id,
                "working_dir": str(working_dir.relative_to(self.workspace_root)),
                "error_type": "recoverable_planning_step_failed",
                "retryable": True,
                "error": str(exc),
                "present": [path for path, present in checklist.items() if present],
                "missing": [path for path, present in checklist.items() if not present],
            }
            if runtime:
                runtime.emit_progress("Narrative planning failed; partial artifacts were kept", stage="planning_failed", metadata=payload)
            return ToolResult("vimax_narrative_planning", False, f"Narrative planning failed: {exc}", payload)

        checklist = self.session_index.artifact_checklist(session_id)
        generated = [path for path, present in checklist.items() if present and not generated_before.get(path)]
        reused = [path for path, present in checklist.items() if present and generated_before.get(path)]
        ready_for_render = _ready_for_render(checklist)
        self.session_index.update_stage(session_id, "narrative_planned", "Structured text planning complete" if ready_for_render else "Structured text planning partially complete")
        if runtime:
            runtime.emit_progress("Narrative planning complete", stage="completed", metadata={"ready_for_render": ready_for_render})
        payload = {
            "session_id": session_id,
            "working_dir": str(working_dir.relative_to(self.workspace_root)),
            "generated": generated,
            "reused": reused,
            "missing": [path for path, present in checklist.items() if not present],
            "ready_for_render": ready_for_render,
        }
        return ToolResult("vimax_narrative_planning", True, json.dumps(payload, ensure_ascii=False, indent=2), payload)

    async def _revise_narrative_artifact(self, session_id: str, working_dir: Path, revision_target: str, revision_instruction: str, runtime: ToolRuntimeContext | None = None) -> ToolResult:
        if not revision_instruction:
            self.session_index.update_stage(session_id, "error", "Revision failed: missing revision_instruction")
            return ToolResult("vimax_narrative_planning", False, "revision_instruction is required when revision_target is provided.", {"error_type": "missing_revision_instruction", "session_id": session_id, "revision_target": revision_target})
        try:
            target_path = _resolve_artifact_path(working_dir, revision_target)
        except ValueError as exc:
            self.session_index.update_stage(session_id, "error", f"Revision failed: {exc}")
            return ToolResult("vimax_narrative_planning", False, str(exc), {"error_type": "invalid_revision_target", "session_id": session_id, "revision_target": revision_target})
        if not target_path.exists():
            self.session_index.update_stage(session_id, "error", f"Revision failed: target does not exist: {revision_target}")
            return ToolResult("vimax_narrative_planning", False, f"Revision target does not exist: {revision_target}", {"error_type": "dependency_missing", "session_id": session_id, "revision_target": revision_target})
        try:
            self.session_index.update_stage(session_id, "narrative_planning", "Revising structured text artifact")
            if runtime:
                runtime.emit_progress("Revising structured text artifact", stage="revising", metadata={"session_id": session_id, "revision_target": revision_target})
            chat_model = _build_chat_model()
            before = target_path.read_text(encoding="utf-8")
            revised = await _revise_artifact_with_llm(chat_model, target_path.relative_to(working_dir).as_posix(), before, revision_instruction)
            if target_path.suffix == ".json":
                try:
                    revised_payload = json.loads(revised)
                except json.JSONDecodeError as exc:
                    self.session_index.update_stage(session_id, "error", f"Revision failed: invalid JSON output: {exc}")
                    return ToolResult("vimax_narrative_planning", False, f"Revision output was not valid JSON: {exc}", {"error_type": "invalid_revision_json", "session_id": session_id, "revision_target": revision_target})
                revised = json.dumps(revised_payload, ensure_ascii=False, indent=2)
            target_path.write_text(revised, encoding="utf-8")
        except Exception as exc:
            self.session_index.update_stage(session_id, "error", f"Revision failed: {exc}")
            raise

        stale = _stale_keys_for_revision(target_path.relative_to(working_dir).as_posix())
        if stale:
            self.session_index.mark_stale(session_id, stale)
        self.session_index.append_log("revisions", {"session_id": session_id, "target": target_path.relative_to(working_dir).as_posix(), "instruction": revision_instruction, "stale": stale, "before_preview": before[:500], "after_preview": revised[:500]})
        checklist = self.session_index.artifact_checklist(session_id)
        ready_for_render = _ready_for_render(checklist)
        self.session_index.update_stage(session_id, "narrative_planned" if ready_for_render else "narrative_planning", "Revised structured text artifact")
        payload = {
            "session_id": session_id,
            "working_dir": str(working_dir.relative_to(self.workspace_root)),
            "generated": [],
            "reused": [path for path, present in checklist.items() if present],
            "revised": [target_path.relative_to(working_dir).as_posix()],
            "missing": [path for path, present in checklist.items() if not present],
            "stale": stale,
            "ready_for_render": ready_for_render,
            "revision_target": target_path.relative_to(working_dir).as_posix(),
        }
        return ToolResult("vimax_narrative_planning", True, json.dumps(payload, ensure_ascii=False, indent=2), payload)

    async def vimax_novel_planning(self, args: dict[str, Any], runtime: ToolRuntimeContext | None = None) -> ToolResult:
        novel_text = str(args.get("novel_text", "") or "").strip()
        user_requirement = str(args.get("user_requirement", "") or "").strip()
        style = str(args.get("style", "") or "").strip() or "Cinematic, coherent, 16:9"
        if not novel_text:
            return ToolResult("vimax_novel_planning", False, "novel_text is required for novel planning.", {"error_type": "missing_input"})

        session_id_arg = str(args.get("session_id", "") or "").strip()
        session = self._resolve_session(session_id_arg, idea=novel_text, script="", user_requirement=user_requirement, style=style)
        session_id = session["session_id"]
        working_dir = self.session_index.working_dir(session_id)
        novel_dir = working_dir / "novel2video"
        novel_dir.mkdir(parents=True, exist_ok=True)
        generated_before = self.session_index.artifact_checklist(session_id)

        try:
            self.session_index.update_stage(session_id, "novel_planning", "Generating novel structured text artifacts")
            if runtime:
                runtime.emit_progress("Starting novel planning", stage="starting", metadata={"session_id": session_id})
                await asyncio.sleep(0)
            pipeline = _build_novel_pipeline(novel_dir)
            await _run_planning_step(
                "Planning novel structured text artifacts",
                "novel_plan_text_artifacts",
                pipeline.plan_text_artifacts(
                    novel_text=novel_text,
                    user_requirement=user_requirement,
                    style=style,
                    progress=_pipeline_progress(runtime, session_id),
                    quiet=True,
                ),
                runtime,
                {"session_id": session_id},
            )
        except Exception as exc:
            self.session_index.update_stage(session_id, "error", f"Novel planning failed: {exc}")
            return ToolResult("vimax_novel_planning", False, str(exc), {"error_type": "exception", "session_id": session_id})

        checklist = self.session_index.artifact_checklist(session_id)
        generated = [path for path, present in checklist.items() if path.startswith("novel2video/") and present and not generated_before.get(path)]
        reused = [path for path, present in checklist.items() if path.startswith("novel2video/") and present and generated_before.get(path)]
        missing = [path for path, present in checklist.items() if path.startswith("novel2video/") and not present]
        ready = _novel_text_ready(checklist)
        self.session_index.update_stage(session_id, "novel_planned" if ready else "novel_planning", "Novel structured text planning complete" if ready else "Novel structured text planning partially complete")
        if runtime:
            runtime.emit_progress("Novel planning complete", stage="completed", metadata={"session_id": session_id, "ready_for_scene_render": False})
        payload = {
            "session_id": session_id,
            "working_dir": str(working_dir.relative_to(self.workspace_root)),
            "generated": generated,
            "reused": reused,
            "missing": missing,
            "ready_for_scene_render": False,
        }
        return ToolResult("vimax_novel_planning", True, json.dumps(payload, ensure_ascii=False, indent=2), payload)

    async def vimax_render_video(self, args: dict[str, Any], runtime: ToolRuntimeContext | None = None) -> ToolResult:
        session_id = str(args.get("session_id", "") or "").strip()
        session = self.session_index.get(session_id) if session_id else self.session_index.active()
        if session is None:
            return ToolResult("vimax_render_video", False, "No active session to render.", {"error_type": "missing_session"})
        session_id = session["session_id"]
        checklist = self.session_index.artifact_checklist(session_id)
        missing = _missing_render_dependencies(checklist)
        working_dir = self.session_index.working_dir(session_id)
        if missing:
            payload = {"error_type": "dependency_missing", "missing": missing, "session_id": session_id}
            _write_render_status(working_dir, status="dependency_missing", payload=payload)
            return ToolResult("vimax_render_video", False, f"Dependency missing: {', '.join(missing)}", payload)

        self.session_index.update_stage(session_id, "rendering", "Rendering video artifacts")
        _write_render_status(working_dir, status="rendering", payload={"session_id": session_id, "render_started": True, "render_completed": False})
        try:
            chat_model = _build_chat_model()
            image_generator = _build_image_generator()
            video_generator = _build_video_generator()
            if runtime:
                runtime.emit_progress("Starting video render", stage="rendering", metadata={"session_id": session_id})
            if _idea_mode_ready(checklist):
                idea_pipeline = Idea2VideoPipeline(chat_model=chat_model, image_generator=image_generator, video_generator=video_generator, working_dir=str(working_dir / "idea2video"))
                with _suppress_pipeline_output():
                    final_video = await idea_pipeline(idea=str(session.get("idea", "")), user_requirement=str(session.get("user_requirement", "")), style=str(session.get("style", "")), quiet=True)
                self.session_index.update_stage(session_id, "rendered", "Final video rendered")
                payload = {"session_id": session_id, "render_mode": "idea2video", "render_started": True, "render_completed": True, "final_video_path": str(Path(final_video).relative_to(self.workspace_root)), "missing": []}
                _write_render_status(working_dir, status="rendered", payload=payload)
                return ToolResult("vimax_render_video", True, json.dumps(payload, ensure_ascii=False, indent=2), payload)
            if _script_mode_ready(checklist):
                script_dir = working_dir / "script2video"
                script_text = _load_script_text(working_dir)
                characters = _load_characters(script_dir / "characters.json")
                pipeline = Script2VideoPipeline(chat_model=chat_model, image_generator=image_generator, video_generator=video_generator, working_dir=str(script_dir))
                with _suppress_pipeline_output():
                    final_video = await pipeline(script=script_text, user_requirement=str(session.get("user_requirement", "")), style=str(session.get("style", "")), characters=characters, quiet=True, progress=_pipeline_progress(runtime, session_id))
                self.session_index.update_stage(session_id, "rendered", "Final video rendered")
                payload = {"session_id": session_id, "render_mode": "script2video", "render_started": True, "render_completed": True, "final_video_path": str(Path(final_video).relative_to(self.workspace_root)), "missing": []}
                _write_render_status(working_dir, status="rendered", payload=payload)
                return ToolResult("vimax_render_video", True, json.dumps(payload, ensure_ascii=False, indent=2), payload)
            if _novel_mode_ready(checklist):
                novel_dir = working_dir / "novel2video"
                pipeline = _build_novel_render_pipeline(novel_dir, chat_model, image_generator, video_generator)
                with _suppress_pipeline_output():
                    render_result = await pipeline.render_video_artifacts(style=str(session.get("style", "")), user_requirement=str(session.get("user_requirement", "")), quiet=True, progress=_pipeline_progress(runtime, session_id))
                scene_videos_dir = Path(render_result["scene_videos_dir"])
                self.session_index.update_stage(session_id, "novel_scene_rendered", "Novel scene videos rendered")
                payload = {
                    "session_id": session_id,
                    "render_mode": "novel2video",
                    "render_started": True,
                    "render_completed": True,
                    "scene_render_completed": True,
                    "final_video_path": None,
                    "scene_videos_dir": str(scene_videos_dir.relative_to(self.workspace_root)),
                    "scene_video_dirs": [str(Path(path).relative_to(self.workspace_root)) for path in render_result.get("scene_video_dirs", [])],
                    "scene_count": render_result.get("scene_count", 0),
                    "missing": [],
                }
                _write_render_status(working_dir, status="rendered", payload=payload)
                return ToolResult("vimax_render_video", True, json.dumps(payload, ensure_ascii=False, indent=2), payload)
        except Exception as exc:
            unwrapped = _unwrap_retry_error(exc)
            error_text = _sanitize_error_text(str(unwrapped))
            wrapped_error_text = _sanitize_error_text(str(exc))
            self.session_index.update_stage(session_id, "error", f"Render failed: {error_text}")
            checklist = self.session_index.artifact_checklist(session_id)
            payload = {
                "error_type": "render_failed",
                "retryable": _is_retryable_render_error(unwrapped),
                "session_id": session_id,
                "error": error_text,
                "wrapped_error": wrapped_error_text,
                "present": [path for path, present in checklist.items() if present],
                "missing": [path for path, present in checklist.items() if not present],
            }
            _write_render_status(working_dir, status="error", payload=payload)
            if runtime:
                runtime.emit_progress("Render failed; partial artifacts were kept", stage="render_failed", metadata=payload)
            return ToolResult("vimax_render_video", False, f"Render failed: {error_text}", payload)
        payload = {"error_type": "dependency_missing", "session_id": session_id}
        _write_render_status(working_dir, status="dependency_missing", payload=payload)
        return ToolResult("vimax_render_video", False, "No render mode matched current session.", payload)

    async def vimax_render_scene(self, args: dict[str, Any], runtime: ToolRuntimeContext | None = None) -> ToolResult:
        session_id = str(args.get("session_id", "") or "").strip()
        scene_id = str(args.get("scene_id", "") or "").strip()
        session = self.session_index.get(session_id) if session_id else self.session_index.active()
        if session is None:
            return ToolResult("vimax_render_scene", False, "No active session to render.", {"error_type": "missing_session"})
        if not _is_valid_idea_scene_id(scene_id):
            return ToolResult(
                "vimax_render_scene",
                False,
                "scene_id must use the exact format scene_<n>.",
                {"error_type": "invalid_scene_id", "session_id": session["session_id"], "scene_id": scene_id},
            )

        session_id = session["session_id"]
        working_dir = self.session_index.working_dir(session_id)
        idea_dir = working_dir / "idea2video"
        scene_dir = idea_dir / scene_id
        final_video_path = scene_dir / "final_video.mp4"
        if final_video_path.exists():
            self.session_index.update_stage(session_id, "scene_rendered", f"Scene {scene_id} render cache reused")
            payload = {
                "session_id": session_id,
                "scene_id": scene_id,
                "render_mode": "idea2video_scene",
                "render_started": False,
                "render_completed": True,
                "render_cached": True,
                "scene_video_path": str(final_video_path.relative_to(self.workspace_root)),
                "missing": [],
            }
            _write_render_status(scene_dir, status="rendered", payload=payload)
            if runtime:
                runtime.emit_progress("Selected scene already rendered; cache reused", stage="scene_render_cached", metadata={"session_id": session_id, "scene_id": scene_id})
            return ToolResult("vimax_render_scene", True, json.dumps(payload, ensure_ascii=False, indent=2), payload)

        missing = _missing_idea_scene_dependencies(idea_dir, scene_dir)
        if missing:
            payload = {
                "error_type": "dependency_missing",
                "session_id": session_id,
                "scene_id": scene_id,
                "missing": missing,
            }
            _write_render_status(scene_dir, status="dependency_missing", payload=payload)
            return ToolResult("vimax_render_scene", False, f"Dependency missing: {', '.join(missing)}", payload)

        try:
            scene_script = _load_idea_scene_script(idea_dir / "script.json", scene_id)
            characters_path = scene_dir / "characters.json"
            if not characters_path.exists():
                characters_path = idea_dir / "characters.json"
            characters = _load_characters(characters_path)
            render_options, reference_views = _load_idea_render_plan(idea_dir / "render_plan.json")

            self.session_index.update_stage(session_id, "scene_rendering", f"Rendering {scene_id}")
            payload = {
                "session_id": session_id,
                "scene_id": scene_id,
                "render_mode": "idea2video_scene",
                "render_started": True,
                "render_completed": False,
            }
            _write_render_status(scene_dir, status="rendering", payload=payload)
            if runtime:
                runtime.emit_progress("Starting selected scene render", stage="scene_rendering", metadata={"session_id": session_id, "scene_id": scene_id})

            chat_model = _build_chat_model()
            image_generator = _build_image_generator()
            video_generator = _build_video_generator()
            portrait_pipeline = Idea2VideoPipeline(
                chat_model=chat_model,
                image_generator=image_generator,
                video_generator=video_generator,
                working_dir=str(idea_dir),
            )
            with _suppress_pipeline_output():
                character_portraits_registry = await portrait_pipeline.generate_character_portraits(
                    characters=characters,
                    character_portraits_registry=None,
                    style=str(session.get("style", "")),
                    reference_views=reference_views,
                    image_size=render_options.get("image_size"),
                )
                pipeline = Script2VideoPipeline(
                    chat_model=chat_model,
                    image_generator=image_generator,
                    video_generator=video_generator,
                    working_dir=str(scene_dir),
                )
                final_video = await pipeline(
                    script=scene_script,
                    user_requirement=str(session.get("user_requirement", "")),
                    style=str(session.get("style", "")),
                    characters=characters,
                    character_portraits_registry=character_portraits_registry,
                    quiet=True,
                    progress=_pipeline_progress(runtime, session_id, scene_index=_idea_scene_index(scene_id)),
                    render_options=render_options,
                )

            self.session_index.update_stage(session_id, "scene_rendered", f"Rendered {scene_id}")
            payload = {
                "session_id": session_id,
                "scene_id": scene_id,
                "render_mode": "idea2video_scene",
                "render_started": True,
                "render_completed": True,
                "render_cached": False,
                "scene_video_path": str(Path(final_video).relative_to(self.workspace_root)),
                "missing": [],
            }
            _write_render_status(scene_dir, status="rendered", payload=payload)
            return ToolResult("vimax_render_scene", True, json.dumps(payload, ensure_ascii=False, indent=2), payload)
        except Exception as exc:
            unwrapped = _unwrap_retry_error(exc)
            if _is_video_queue_full_error(unwrapped):
                try:
                    payload = _persist_waiting_video_capacity_state(
                        session_id=session_id,
                        scene_id=scene_id,
                        scene_dir=scene_dir,
                        error=unwrapped,
                    )
                    self.session_index.update_stage(session_id, "waiting_for_video_capacity", f"Waiting for Agnes video capacity for {scene_id}")
                    _write_render_status(scene_dir, status="waiting_for_video_capacity", payload=payload)
                    if runtime:
                        runtime.emit_progress("Agnes video queue is full; scene saved for resume", stage="waiting_for_video_capacity", metadata=payload)
                    return ToolResult("vimax_render_scene", True, json.dumps(payload, ensure_ascii=False, indent=2), payload)
                except Exception as persist_exc:
                    error_text = _sanitize_error_text(str(_unwrap_retry_error(persist_exc)))
                    payload = {
                        "error_type": "waiting_state_persistence_failed",
                        "session_id": session_id,
                        "scene_id": scene_id,
                        "error": error_text,
                    }
                    _write_render_status(scene_dir, status="error", payload=payload)
                    return ToolResult("vimax_render_scene", False, f"Could not persist waiting video state: {error_text}", payload)
            error_text = _sanitize_error_text(str(unwrapped))
            self.session_index.update_stage(session_id, "error", f"Scene {scene_id} render failed: {error_text}")
            payload = {
                "error_type": "render_failed",
                "retryable": _is_retryable_render_error(unwrapped),
                "session_id": session_id,
                "scene_id": scene_id,
                "error": error_text,
            }
            _write_render_status(scene_dir, status="error", payload=payload)
            if runtime:
                runtime.emit_progress("Selected scene render failed; partial artifacts were kept", stage="scene_render_failed", metadata=payload)
            return ToolResult("vimax_render_scene", False, f"Render failed: {error_text}", payload)

    async def vimax_resume_waiting_scene(self, args: dict[str, Any], runtime: ToolRuntimeContext | None = None) -> ToolResult:
        session_id = str(args.get("session_id", "") or "").strip()
        scene_id = str(args.get("scene_id", "") or "").strip()
        session = self.session_index.get(session_id) if session_id else None
        if session is None:
            return ToolResult("vimax_resume_waiting_scene", False, "A valid session_id is required.", {"error_type": "missing_session", "session_id": session_id})
        if not _is_valid_idea_scene_id(scene_id):
            return ToolResult("vimax_resume_waiting_scene", False, "scene_id must use the exact format scene_<n>.", {"error_type": "invalid_scene_id", "session_id": session_id, "scene_id": scene_id})
        scene_dir = self.session_index.working_dir(session_id) / "idea2video" / scene_id
        try:
            state_path = _resolve_scene_artifact_path(scene_dir, "waiting_video_capacity.json")
            if not state_path.exists():
                return ToolResult("vimax_resume_waiting_scene", False, "No waiting video-capacity state exists for this scene.", {"error_type": "waiting_state_missing", "session_id": session_id, "scene_id": scene_id})
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict) or state.get("schema_version") != 1 or state.get("session_id") != session_id or state.get("scene_id") != scene_id:
                return ToolResult("vimax_resume_waiting_scene", False, "Waiting state does not match the requested session and scene.", {"error_type": "invalid_waiting_state", "session_id": session_id, "scene_id": scene_id})
            jobs = state.get("jobs")
            if not isinstance(jobs, list) or not jobs:
                return ToolResult("vimax_resume_waiting_scene", False, "Waiting state has no resumable video jobs.", {"error_type": "invalid_waiting_state", "session_id": session_id, "scene_id": scene_id})
            jobs = sorted(jobs, key=lambda job: job["shot_idx"] if isinstance(job, dict) and isinstance(job.get("shot_idx"), int) else -1)
            shot_indices = [job.get("shot_idx") for job in jobs if isinstance(job, dict)]
            if len(shot_indices) != len(jobs) or any(not isinstance(shot_idx, int) or shot_idx < 0 for shot_idx in shot_indices) or len(set(shot_indices)) != len(shot_indices):
                return ToolResult("vimax_resume_waiting_scene", False, "Waiting state contains invalid or duplicate shot indices.", {"error_type": "invalid_waiting_state", "session_id": session_id, "scene_id": scene_id})
            expected_shot_indices = state.get("expected_shot_indices")
            if not isinstance(expected_shot_indices, list) or any(not isinstance(shot_idx, int) or shot_idx < 0 for shot_idx in expected_shot_indices) or len(set(expected_shot_indices)) != len(expected_shot_indices) or sorted(expected_shot_indices) != shot_indices:
                return ToolResult("vimax_resume_waiting_scene", False, "Waiting state is missing one or more planned video jobs.", {"error_type": "incomplete_waiting_state", "session_id": session_id, "scene_id": scene_id})
            video_generator = _build_video_generator()
            if not isinstance(video_generator, AgnesVideoProvider):
                return ToolResult("vimax_resume_waiting_scene", False, "Waiting scene requires AgnesVideoProvider.", {"error_type": "provider_mismatch", "session_id": session_id, "scene_id": scene_id})
        except Exception as exc:
            error_text = _sanitize_error_text(str(_unwrap_retry_error(exc)))
            return ToolResult("vimax_resume_waiting_scene", False, f"Resume state could not be loaded: {error_text}", {"error_type": "invalid_waiting_state", "session_id": session_id, "scene_id": scene_id, "error": error_text})

        def persist(status: str) -> None:
            timestamp = datetime.now().isoformat(timespec="seconds")
            for job in jobs:
                if not isinstance(job, dict):
                    raise ValueError("Invalid waiting video job")
                shot_idx = job.get("shot_idx")
                if not isinstance(shot_idx, int) or shot_idx < 0:
                    raise ValueError("Waiting video job missing shot_idx")
                request_path = _resolve_scene_artifact_path(scene_dir, f"shots/{shot_idx}/video_request.json")
                _atomic_write_json(request_path, {"schema_version": 1, **job})
            primary = next((job for job in jobs if job.get("status") != "completed"), jobs[0])
            state.update({
                "provider": primary.get("provider"), "model": primary.get("model"), "parameters": primary.get("parameters"),
                "reference_image_paths": primary.get("reference_image_paths"), "status": status, "last_attempt": timestamp,
                "submit_attempts": sum(int(job.get("submit_attempts", 0)) for job in jobs), "video_id": primary.get("video_id"), "jobs": jobs,
            })
            _atomic_write_json(state_path, state)

        try:
            for job in jobs:
                if not isinstance(job, dict) or job.get("status") == "completed":
                    continue
                shot_idx = job["shot_idx"]
                model = job.get("model")
                if not isinstance(model, str):
                    raise RuntimeError("Persisted video job is missing its model")
                if job.get("provider") != "AgnesVideoProvider":
                    raise RuntimeError("Persisted video provider does not match AgnesVideoProvider")
                output_rel = job.get("output_path")
                if not isinstance(output_rel, str):
                    raise ValueError("Waiting video job missing output_path")
                output_path = _resolve_scene_artifact_path(scene_dir, output_rel)
                video_id = job.get("video_id")
                if isinstance(video_id, str) and video_id:
                    output = await video_generator.poll_existing_task(video_id, model, progress=None)
                else:
                    if job.get("status") != "waiting_for_video_capacity":
                        return ToolResult("vimax_resume_waiting_scene", False, "Persisted video job is not eligible for a new submission.", {"error_type": "invalid_waiting_job_status", "session_id": session_id, "scene_id": scene_id, "shot_idx": shot_idx})
                    if model != video_generator.model:
                        raise RuntimeError("Persisted video model does not match the configured Agnes provider model")
                    raw_refs = job.get("reference_image_paths")
                    if not isinstance(raw_refs, list):
                        raise ValueError("Waiting video job missing reference_image_paths")
                    refs = [_resolve_scene_artifact_path(scene_dir, str(ref)) for ref in raw_refs]
                    if not refs or not all(path.exists() for path in refs):
                        raise RuntimeError("Persisted video references are missing")
                    job.update({"status": "submitting", "last_attempt": datetime.now().isoformat(timespec="seconds")})
                    persist("waiting_for_video_capacity")
                    def task_created(task_id: str, selected_model: str, current_job: dict[str, Any] = job) -> None:
                        current_job.update({"video_id": task_id, "model": selected_model, "status": "polling", "submit_attempts": int(current_job.get("submit_attempts", 0)) + 1})
                        persist("waiting_for_video_capacity")
                    try:
                        output = await video_generator.generate_single_video(
                            prompt=str(job.get("prompt") or ""), reference_image_paths=[str(path) for path in refs],
                            task_created_callback=task_created, **dict(job.get("parameters") or {}),
                        )
                    except AgnesVideoAPIError as exc:
                        exc.shot_idx = shot_idx
                        raise
                output.save(str(output_path))
                job["status"] = "completed"
            if any(job.get("status") != "completed" for job in jobs):
                raise RuntimeError("Cannot assemble a scene until every persisted video job is completed")
            final_path = _resolve_scene_artifact_path(scene_dir, "final_video.mp4")
            clips = [_resolve_scene_artifact_path(scene_dir, str(job["output_path"])) for job in jobs]
            if not all(path.exists() for path in clips):
                raise RuntimeError("Cannot assemble a scene because one or more completed video clips are missing")
            if len(clips) == 1:
                shutil.copyfile(clips[0], final_path)
            else:
                concatenate_video_files([str(path) for path in clips], str(final_path))
            persist("rendered")
            self.session_index.update_stage(session_id, "scene_rendered", f"Resumed and rendered {scene_id}")
            payload = {"session_id": session_id, "scene_id": scene_id, "status": "rendered", "render_completed": True, "scene_video_path": str(final_path.relative_to(self.workspace_root))}
            _write_render_status(scene_dir, status="rendered", payload=payload)
            return ToolResult("vimax_resume_waiting_scene", True, json.dumps(payload, ensure_ascii=False, indent=2), payload)
        except Exception as exc:
            unwrapped = _unwrap_retry_error(exc)
            if _is_video_queue_full_error(unwrapped):
                failed_shot_idx = getattr(unwrapped, "shot_idx", None)
                if not isinstance(failed_shot_idx, int):
                    return ToolResult("vimax_resume_waiting_scene", False, "Resume could not identify the saturated video job safely.", {"error_type": "resume_failed", "session_id": session_id, "scene_id": scene_id})
                queue_attempts = max(1, int(getattr(unwrapped, "queue_attempts", 1)))
                failed_job = next((job for job in jobs if job.get("shot_idx") == failed_shot_idx), None)
                if not isinstance(failed_job, dict):
                    return ToolResult("vimax_resume_waiting_scene", False, "Resume could not identify the saturated video job safely.", {"error_type": "resume_failed", "session_id": session_id, "scene_id": scene_id})
                failed_job.update({
                    "status": "waiting_for_video_capacity",
                    "last_error_code": "video_queue_full",
                    "last_attempt": datetime.now().isoformat(timespec="seconds"),
                    "submit_attempts": int(failed_job.get("submit_attempts", 0)) + queue_attempts,
                })
                try:
                    persist("waiting_for_video_capacity")
                except Exception as persist_exc:
                    error_text = _sanitize_error_text(str(_unwrap_retry_error(persist_exc)))
                    return ToolResult("vimax_resume_waiting_scene", False, f"Could not persist waiting video state: {error_text}", {"error_type": "waiting_state_persistence_failed", "session_id": session_id, "scene_id": scene_id, "error": error_text})
                self.session_index.update_stage(session_id, "waiting_for_video_capacity", f"Waiting for Agnes video capacity for {scene_id}")
                payload = {"session_id": session_id, "scene_id": scene_id, "status": "waiting_for_video_capacity", "render_completed": False, "jobs": jobs}
                _write_render_status(scene_dir, status="waiting_for_video_capacity", payload=payload)
                return ToolResult("vimax_resume_waiting_scene", True, json.dumps(payload, ensure_ascii=False, indent=2), payload)
            error_text = _sanitize_error_text(str(unwrapped))
            return ToolResult("vimax_resume_waiting_scene", False, f"Resume failed: {error_text}", {"error_type": "resume_failed", "session_id": session_id, "scene_id": scene_id, "error": error_text})

    async def vimax_resume_waiting_scene_once(self, args: dict[str, Any], runtime: ToolRuntimeContext | None = None) -> ToolResult:
        """Perform at most one Agnes create or one status poll, without queue backoff."""
        session_id = str(args.get("session_id", "") or "").strip()
        scene_id = str(args.get("scene_id", "") or "").strip()
        session = self.session_index.get(session_id) if session_id else None
        if session is None or not _is_valid_idea_scene_id(scene_id):
            return ToolResult("vimax_resume_waiting_scene_once", False, "A valid session_id and exact scene_id are required.", {"error_type": "invalid_resume_target", "session_id": session_id, "scene_id": scene_id})
        scene_dir = self.session_index.working_dir(session_id) / "idea2video" / scene_id
        try:
            state_path = _resolve_scene_artifact_path(scene_dir, "waiting_video_capacity.json")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict) or state.get("schema_version") != 1 or state.get("session_id") != session_id or state.get("scene_id") != scene_id:
                raise ValueError("state identity mismatch")
            jobs = state.get("jobs")
            expected = state.get("expected_shot_indices")
            if state.get("status") not in {"waiting_for_video_capacity", "polling"} or not isinstance(jobs, list) or not jobs or not isinstance(expected, list):
                raise ValueError("state is not maintainable")
            jobs = sorted(jobs, key=lambda item: item.get("shot_idx", -1) if isinstance(item, dict) else -1)
            shot_ids = [job.get("shot_idx") for job in jobs if isinstance(job, dict)]
            if len(shot_ids) != len(jobs) or any(not isinstance(idx, int) or isinstance(idx, bool) or idx < 0 for idx in shot_ids) or sorted(expected) != shot_ids or len(set(shot_ids)) != len(shot_ids):
                raise ValueError("invalid expected shots")
            for job in jobs:
                shot_idx = job["shot_idx"]
                manifest_path = _resolve_scene_artifact_path(scene_dir, f"shots/{shot_idx}/video_request.json")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest != {"schema_version": 1, **job}:
                    raise ValueError("manifest diverges from waiting state")
                status = job.get("status")
                video_id = job.get("video_id")
                if status not in {"waiting_for_video_capacity", "polling", "completed"}:
                    raise ValueError("unsupported job status")
                if video_id is not None and (not isinstance(video_id, str) or not video_id.strip()):
                    raise ValueError("invalid video_id")
                if status == "waiting_for_video_capacity" and video_id is not None:
                    raise ValueError("waiting job has video_id")
                if status in {"polling", "completed"} and video_id is None:
                    raise ValueError("submitted job has no video_id")
                if job.get("provider") != "AgnesVideoProvider" or not isinstance(job.get("model"), str) or not job["model"].strip():
                    raise ValueError("persisted provider/model is invalid")
                if not isinstance(job.get("prompt"), str):
                    raise ValueError("persisted prompt is invalid")
                parameters = job.get("parameters")
                if not isinstance(parameters, dict) or any(not isinstance(key, str) for key in parameters) or set(parameters) - {"aspect_ratio", "seconds", "resolution", "seed", "negative_prompt"}:
                    raise ValueError("persisted video parameters are invalid")
                if "seconds" in parameters and (not isinstance(parameters["seconds"], int) or isinstance(parameters["seconds"], bool) or parameters["seconds"] <= 0):
                    raise ValueError("persisted video duration is invalid")
                if "aspect_ratio" in parameters and (not isinstance(parameters["aspect_ratio"], str) or not parameters["aspect_ratio"].strip()):
                    raise ValueError("persisted video aspect ratio is invalid")
                if "resolution" in parameters and (not isinstance(parameters["resolution"], str) or not parameters["resolution"].strip()):
                    raise ValueError("persisted video resolution is invalid")
                references = job.get("reference_image_paths")
                if not isinstance(references, list) or len(references) > 2 or (status != "completed" and not references):
                    raise ValueError("persisted video references are invalid")
                for reference in references:
                    if not isinstance(reference, str) or not reference:
                        raise ValueError("persisted video reference path is invalid")
                    reference_path = _resolve_scene_artifact_path(scene_dir, reference)
                    if not reference_path.exists() or not reference_path.is_file():
                        raise ValueError("persisted video references are missing")
                output_path = job.get("output_path")
                if not isinstance(output_path, str) or not output_path:
                    raise ValueError("persisted video output path is invalid")
                output = _resolve_scene_artifact_path(scene_dir, output_path)
                if status == "completed" and (not output.exists() or not output.is_file()):
                    raise ValueError("completed job is missing its video artifact")
                if status != "completed" and output.exists():
                    raise ValueError("unexpected video artifact before completion")
            final_path = _resolve_scene_artifact_path(scene_dir, "final_video.mp4")
            if final_path.exists():
                raise ValueError("final video already exists for nonterminal waiting state")
        except Exception:
            return ToolResult("vimax_resume_waiting_scene_once", False, "Persisted scene state is ambiguous; no submission was made.", {"error_type": "invalid_waiting_state", "session_id": session_id, "scene_id": scene_id})

        try:
            video_generator = _build_video_generator()
        except Exception as exc:
            lowered = str(_unwrap_retry_error(exc)).lower()
            error_type = "missing_credentials" if "vimax_video_api_key" in lowered or "video api key" in lowered else "provider_error"
            message = "Video credentials are unavailable; no submission was made." if error_type == "missing_credentials" else "Video provider initialization failed; no submission was made."
            return ToolResult("vimax_resume_waiting_scene_once", False, message, {"error_type": error_type, "session_id": session_id, "scene_id": scene_id})
        if not isinstance(video_generator, AgnesVideoProvider):
            return ToolResult("vimax_resume_waiting_scene_once", False, "Video provider is incompatible with this persisted Agnes scene; no submission was made.", {"error_type": "provider_error", "session_id": session_id, "scene_id": scene_id})

        # Cron mode must never inherit the interactive queue retry/fallback policy.
        video_generator.queue_max_attempts = 1
        video_generator.max_retries = 1
        video_generator.allow_paid_video_fallback = False

        def persist(status: str) -> None:
            timestamp = datetime.now().isoformat(timespec="seconds")
            for job in jobs:
                _atomic_write_json(_resolve_scene_artifact_path(scene_dir, f"shots/{job['shot_idx']}/video_request.json"), {"schema_version": 1, **job})
            primary = next((job for job in jobs if job.get("status") != "completed"), jobs[0])
            state.update({
                "provider": primary.get("provider"), "model": primary.get("model"), "parameters": primary.get("parameters"),
                "reference_image_paths": primary.get("reference_image_paths"), "status": status, "last_attempt": timestamp,
                "submit_attempts": sum(int(job.get("submit_attempts", 0)) for job in jobs), "video_id": primary.get("video_id"), "jobs": jobs,
            })
            _atomic_write_json(state_path, state)

        async def finalize_if_complete() -> ToolResult | None:
            if any(job.get("status") != "completed" for job in jobs):
                return None
            clips = [_resolve_scene_artifact_path(scene_dir, str(job["output_path"])) for job in jobs]
            if not all(path.exists() and path.is_file() for path in clips):
                return ToolResult("vimax_resume_waiting_scene_once", False, "Completed clip evidence is incomplete; no overwrite was made.", {"error_type": "incomplete_completed_scene", "session_id": session_id, "scene_id": scene_id})
            if len(clips) == 1:
                shutil.copyfile(clips[0], final_path)
            else:
                concatenate_video_files([str(path) for path in clips], str(final_path))
            persist("rendered")
            self.session_index.update_stage(session_id, "scene_rendered", f"Resumed and rendered {scene_id}")
            payload = {"session_id": session_id, "scene_id": scene_id, "status": "rendered", "render_mode": "idea2video_scene", "render_completed": True, "scene_video_path": str(final_path.relative_to(self.workspace_root)), "missing": []}
            _write_render_status(scene_dir, status="rendered", payload=payload)
            return ToolResult("vimax_resume_waiting_scene_once", True, json.dumps(payload, ensure_ascii=False, indent=2), payload)

        job = next((item for item in jobs if item.get("status") != "completed"), None)
        if not isinstance(job, dict):
            return (await finalize_if_complete()) or ToolResult("vimax_resume_waiting_scene_once", False, "No resumable job exists.", {"error_type": "invalid_waiting_state", "session_id": session_id, "scene_id": scene_id})
        shot_idx = job["shot_idx"]
        model = str(job["model"])
        if model != video_generator.model:
            return ToolResult("vimax_resume_waiting_scene_once", False, "Configured video provider does not match the persisted job model; no submission was made.", {"error_type": "provider_error", "session_id": session_id, "scene_id": scene_id, "shot_idx": shot_idx})
        video_id = job.get("video_id")
        if isinstance(video_id, str) and video_id:
            try:
                _, response = await video_generator._request_with_retry(lambda: video_generator._get_json(video_generator._status_url(video_id, model)), progress=None, operation_name="poll", model=model)
                remote_status = _status_from_response(response).lower()
                if remote_status in {"completed", "success", "succeeded", "done", "finished"}:
                    video_url = _video_url_from_response(response)
                    _, data = await video_generator._request_with_retry(lambda: video_generator._get_bytes(video_url), progress=None, operation_name="download", model=model)
                    VideoOutput(fmt="bytes", ext="mp4", data=data).save(str(_resolve_scene_artifact_path(scene_dir, str(job["output_path"]))))
                    job["status"] = "completed"
                    terminal = await finalize_if_complete()
                    if terminal is not None:
                        return terminal
                elif remote_status in {"failed", "error", "cancelled", "canceled", "expired"}:
                    job["status"] = "failed"
                    persist("failed")
                    return ToolResult("vimax_resume_waiting_scene_once", False, "Remote video task failed; no resubmission was made.", {"error_type": "remote_video_failed", "session_id": session_id, "scene_id": scene_id, "shot_idx": shot_idx})
                job["status"] = "polling"
                persist("polling")
                return ToolResult("vimax_resume_waiting_scene_once", True, "Remote task remains pending.", {"session_id": session_id, "scene_id": scene_id, "status": "polling", "render_completed": False})
            except Exception:
                persist("polling")
                return ToolResult("vimax_resume_waiting_scene_once", False, "Remote task polling failed; its video id was preserved.", {"error_type": "poll_failed", "session_id": session_id, "scene_id": scene_id, "shot_idx": shot_idx})

        if job.get("status") != "waiting_for_video_capacity" or model != video_generator.model:
            return ToolResult("vimax_resume_waiting_scene_once", False, "Job is not eligible for a new Flash submission.", {"error_type": "invalid_waiting_job_status", "session_id": session_id, "scene_id": scene_id, "shot_idx": shot_idx})
        try:
            refs = [_resolve_scene_artifact_path(scene_dir, str(path)) for path in list(job.get("reference_image_paths") or [])]
            if not refs or not all(path.exists() and path.is_file() for path in refs):
                raise ValueError("persisted video references are missing")
            job.update({"status": "submitting", "last_attempt": datetime.now().isoformat(timespec="seconds"), "submit_attempts": int(job.get("submit_attempts", 0)) + 1})
            persist("submitting")
            def task_created(task_id: str, selected_model: str) -> None:
                job.update({"video_id": task_id, "model": selected_model, "status": "polling"})
                persist("polling")
                raise _AutomationTaskCreated()
            await video_generator.generate_single_video(prompt=str(job.get("prompt") or ""), reference_image_paths=[str(path) for path in refs], task_created_callback=task_created, **dict(job.get("parameters") or {}))
            job["status"] = "submission_outcome_unknown"
            persist("submission_outcome_unknown")
            return ToolResult("vimax_resume_waiting_scene_once", False, "Submission outcome is ambiguous; no automatic resubmission will occur.", {"error_type": "ambiguous_video_submission", "session_id": session_id, "scene_id": scene_id, "shot_idx": shot_idx})
        except _AutomationTaskCreated:
            self.session_index.update_stage(session_id, "video_polling", f"Agnes accepted {scene_id}; waiting for completion")
            return ToolResult("vimax_resume_waiting_scene_once", True, "Remote video id persisted; polling is deferred to a future pass.", {"session_id": session_id, "scene_id": scene_id, "status": "polling", "render_completed": False})
        except AgnesVideoAPIError as exc:
            if _is_video_queue_full_error(exc):
                job.update({"status": "waiting_for_video_capacity", "last_error_code": "video_queue_full"})
                persist("waiting_for_video_capacity")
                self.session_index.update_stage(session_id, "waiting_for_video_capacity", f"Waiting for Agnes video capacity for {scene_id}")
                return ToolResult("vimax_resume_waiting_scene_once", True, "Agnes queue remains full.", {"session_id": session_id, "scene_id": scene_id, "status": "waiting_for_video_capacity", "render_completed": False})
            job["status"] = "submission_outcome_unknown"
            persist("submission_outcome_unknown")
            return ToolResult("vimax_resume_waiting_scene_once", False, "Submission failed ambiguously; no automatic resubmission will occur.", {"error_type": "ambiguous_video_submission", "session_id": session_id, "scene_id": scene_id, "shot_idx": shot_idx})
        except Exception:
            job["status"] = "submission_outcome_unknown"
            persist("submission_outcome_unknown")
            return ToolResult("vimax_resume_waiting_scene_once", False, "Submission failed ambiguously; no automatic resubmission will occur.", {"error_type": "ambiguous_video_submission", "session_id": session_id, "scene_id": scene_id, "shot_idx": shot_idx})

    def _resolve_session(self, session_id: str, *, idea: str, script: str, user_requirement: str, style: str) -> dict[str, Any]:
        requested_source = idea or script
        if session_id:
            session = self.session_index.get(session_id)
            if session is None:
                session = self.session_index.create(idea=requested_source, user_requirement=user_requirement, style=style, session_id=session_id)
            elif requested_source and _is_new_source_for_session(session, requested_source):
                session = self.session_index.create(idea=requested_source, user_requirement=user_requirement, style=style)
            else:
                self.session_index.set_active(session_id)
        else:
            if requested_source:
                active = self.session_index.active()
                if active is not None and self._session_is_empty(active):
                    session = self.session_index.set_active(active["session_id"])
                else:
                    session = self.session_index.create(idea=requested_source, user_requirement=user_requirement, style=style)
            else:
                session = self.session_index.active() or self.session_index.create(idea=requested_source, user_requirement=user_requirement, style=style)
        self._update_session_metadata(session["session_id"], idea=requested_source, user_requirement=user_requirement, style=style)
        return self.session_index.get(session["session_id"]) or session

    def _session_is_empty(self, session: dict[str, Any]) -> bool:
        if str(session.get("idea") or "").strip():
            return False
        session_id = str(session.get("session_id") or "").strip()
        if not session_id:
            return False
        return not any(self.session_index.artifact_checklist(session_id).values())

    def _update_session_metadata(self, session_id: str, *, idea: str, user_requirement: str, style: str) -> None:
        data = self.session_index.load()
        record = data.get("sessions", {}).get(session_id)
        if not isinstance(record, dict):
            return
        if idea and not record.get("idea"):
            record["idea"] = idea
        if user_requirement:
            record["user_requirement"] = user_requirement
        if style:
            record["style"] = style
        self.session_index.save(data)


class _DiscardStream:
    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass


_PIPELINE_OUTPUT_SINK = _DiscardStream()


@contextmanager
def _suppress_pipeline_output():
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        with redirect_stdout(_PIPELINE_OUTPUT_SINK), redirect_stderr(_PIPELINE_OUTPUT_SINK):
            yield
    finally:
        logging.disable(previous_disable_level)


def _narrative_step_timeout_seconds() -> float:
    raw = os.environ.get("VIMAX_NARRATIVE_STEP_TIMEOUT_SECONDS", "900")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 900.0


async def _run_planning_step(
    message: str,
    stage: str,
    awaitable: Any,
    runtime: ToolRuntimeContext | None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    timeout_seconds = _narrative_step_timeout_seconds()
    event_metadata = dict(metadata or {})
    event_metadata["timeout_seconds"] = timeout_seconds
    if runtime:
        runtime.emit_progress(message, stage=stage, metadata=event_metadata)
        await asyncio.sleep(0)
    try:
        with _suppress_pipeline_output():
            if timeout_seconds <= 0:
                return await awaitable
            return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(f"{message} timed out after {timeout_seconds:g}s") from exc
    except Exception as exc:
        raise RuntimeError(f"{message} failed: {exc}") from exc


def _is_new_source_for_session(session: dict[str, Any], requested_source: str) -> bool:
    current = str(session.get("idea") or "").strip()
    requested = requested_source.strip()
    if not current or not requested:
        return False
    return current != requested


def _llm_request_timeout_seconds() -> float:
    raw = os.environ.get("VIMAX_LLM_REQUEST_TIMEOUT_SECONDS", "300")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 300.0


def _narrative_max_tokens() -> int:
    raw = os.environ.get("VIMAX_NARRATIVE_MAX_TOKENS", "4096")
    try:
        return max(256, int(raw))
    except ValueError:
        return 4096


def _pipeline_progress(runtime: ToolRuntimeContext | None, session_id: str, *, scene_index: int | None = None):
    if runtime is None:
        return None

    def emit(stage: str, message: str, metadata: dict[str, Any] | None = None) -> None:
        payload = dict(metadata or {})
        payload["session_id"] = session_id
        if scene_index is not None:
            payload["scene_index"] = scene_index
        runtime.emit_progress(message, stage=stage, metadata=payload)

    return emit


def _build_chat_model() -> Any:
    api_key = llm_api_key()
    if not api_key:
        raise RuntimeError("VIMAX_LLM_API_KEY or configs/agent.local.yaml llm.api_key is required for narrative planning")
    return init_chat_model(
        model=llm_model(),
        model_provider=llm_model_provider(),
        api_key=api_key,
        base_url=llm_base_url(),
        timeout=_llm_request_timeout_seconds(),
        max_retries=0,
        max_completion_tokens=_narrative_max_tokens(),
    )


def _build_image_generator() -> AgnesImageProvider | ImageGeneratorNanobananaYunwuAPI | ImageGeneratorOpenRouterAPI:
    api_key = image_api_key()
    if not api_key:
        raise RuntimeError("VIMAX_IMAGE_API_KEY, VIMAX_LLM_API_KEY, or configs/agent.local.yaml image/llm api_key is required for image generation")
    model = image_model()
    base_url = image_base_url()
    provider = api_provider_from_base_url(base_url)
    if provider == "openrouter":
        return ImageGeneratorOpenRouterAPI(api_key=api_key, model=model, base_url=base_url)
    if provider == "agnes":
        return AgnesImageProvider(api_key=api_key, model=model, base_url=base_url)
    return ImageGeneratorNanobananaYunwuAPI(api_key=api_key, model=model, base_url=base_url)


def _build_video_generator() -> Any:
    api_key = video_api_key()
    if not api_key:
        raise RuntimeError("VIMAX_VIDEO_API_KEY, VIMAX_LLM_API_KEY, or configs/agent.local.yaml video/llm api_key is required for video generation")
    model = video_model()
    base_url = video_base_url()
    provider = video_provider().strip().lower()
    if provider == "openrouter":
        return VideoGeneratorOpenRouterAPI(api_key=api_key, model=model, base_url=base_url)
    if provider == "agnes":
        return AgnesVideoProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            allow_paid_video_fallback=video_allow_paid_fallback(),
            paid_fallback_model=video_paid_fallback_model(),
        )
    if provider == "yunwu":
        return VideoGeneratorVeoYunwuAPI(api_key=api_key, t2v_model=model, ff2v_model=model, base_url=base_url)
    raise RuntimeError(f"Unsupported video base_url for automatic provider matching: {base_url}")


class _IdentityRewriter:
    async def __call__(self, prompt: str) -> str:
        return prompt


def _build_embedding_model() -> Any:
    api_key = embedding_api_key()
    base_url = embedding_base_url()
    provider = embedding_model_provider().strip().lower()
    if not api_key or not base_url:
        raise RuntimeError("VIMAX_EMBEDDING_API_KEY or configs/agent.local.yaml embedding api_key/base_url is required for novel planning")
    if provider != "openai":
        raise RuntimeError(f"Unsupported embedding model_provider: {provider}")
    return OpenAIEmbeddings(model=embedding_model(), api_key=api_key, base_url=base_url)


def _build_reranker() -> RerankerBgeSiliconapi:
    api_key = reranker_api_key()
    base_url = reranker_base_url()
    if not api_key or not base_url:
        raise RuntimeError("VIMAX_RERANKER_API_KEY or configs/agent.local.yaml reranker api_key/base_url is required for novel planning")
    return RerankerBgeSiliconapi(api_key=api_key, base_url=base_url, model=reranker_model())


def _build_novel_pipeline(working_dir: Path) -> Novel2MoviePipeline:
    api_key = llm_api_key()
    if not api_key:
        raise RuntimeError("VIMAX_LLM_API_KEY or configs/agent.local.yaml llm.api_key is required for novel planning")
    base_url = llm_base_url()
    model = llm_model()
    dummy = _UnavailableGenerator()
    return Novel2MoviePipeline(
        novel_compressor=NovelCompressor(api_key=api_key, base_url=base_url, chat_model=model),
        event_extractor=EventExtractor(api_key=api_key, base_url=base_url, chat_model=model),
        embeddings=_build_embedding_model(),
        rerank_model=_build_reranker(),
        scene_extractor=SceneExtractor(api_key=api_key, base_url=base_url, chat_model=model),
        global_information_planner=GlobalInformationPlanner(api_key=api_key, base_url=base_url, chat_model=model),
        image_generator=dummy,
        rewriter=_IdentityRewriter(),
        script2video_pipeline=dummy,
        working_dir=str(working_dir),
    )


def _build_novel_render_pipeline(working_dir: Path, chat_model: Any, image_generator: Any, video_generator: Any) -> Novel2MoviePipeline:
    api_key = llm_api_key()
    if not api_key:
        raise RuntimeError("VIMAX_LLM_API_KEY or configs/agent.local.yaml llm.api_key is required for novel rendering")
    base_url = llm_base_url()
    model = llm_model()
    script_pipeline = Script2VideoPipeline(chat_model=chat_model, image_generator=image_generator, video_generator=video_generator, working_dir=str(working_dir / "videos"))
    return Novel2MoviePipeline(
        novel_compressor=NovelCompressor(api_key=api_key, base_url=base_url, chat_model=model),
        event_extractor=EventExtractor(api_key=api_key, base_url=base_url, chat_model=model),
        embeddings=_build_embedding_model(),
        rerank_model=_build_reranker(),
        scene_extractor=SceneExtractor(api_key=api_key, base_url=base_url, chat_model=model),
        global_information_planner=GlobalInformationPlanner(api_key=api_key, base_url=base_url, chat_model=model),
        image_generator=image_generator,
        rewriter=_IdentityRewriter(),
        script2video_pipeline=script_pipeline,
        working_dir=str(working_dir),
    )


def _unwrap_retry_error(exc: Exception) -> Exception:
    if isinstance(exc, RetryError):
        try:
            return exc.last_attempt.exception() or exc
        except Exception:
            return exc
    return exc


def _is_retryable_render_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if isinstance(exc, AttributeError):
        return False
    if "http 403" in text or "key limit exceeded" in text or "quota" in text:
        return False
    return True


def _sanitize_error_text(text: str) -> str:
    sanitized = text
    for marker in ("workspaces/default/keys/",):
        if marker in sanitized:
            prefix, rest = sanitized.split(marker, 1)
            key_id = []
            for char in rest:
                if char.isalnum() or char in "-_":
                    key_id.append(char)
                    continue
                break
            sanitized = prefix + marker + "<redacted>" + rest[len(key_id):]
    if "sk-" in sanitized:
        prefix, rest = sanitized.split("sk-", 1)
        token = []
        for char in rest:
            if char.isalnum() or char in "-_":
                token.append(char)
                continue
            break
        sanitized = prefix + "sk-<redacted>" + rest[len(token):]
    return sanitized


def _resolve_scene_artifact_path(scene_dir: Path, relative_path: str) -> Path:
    candidate = (scene_dir / relative_path).resolve()
    try:
        candidate.relative_to(scene_dir.resolve())
    except ValueError as exc:
        raise ValueError("Persisted scene artifact path escapes the scene directory") from exc
    return candidate


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _persist_waiting_video_capacity_state(
    *,
    session_id: str,
    scene_id: str,
    scene_dir: Path,
    error: BaseException,
) -> dict[str, Any]:
    timestamp = datetime.now().isoformat(timespec="seconds")
    queue_attempts = max(1, int(getattr(error, "queue_attempts", 1)))
    jobs: list[dict[str, Any]] = []
    entries: list[tuple[int, Path, dict[str, Any]]] = []
    for request_path in (scene_dir / "shots").glob("*/video_request.json"):
        safe_path = _resolve_scene_artifact_path(scene_dir, str(request_path.relative_to(scene_dir)))
        raw = json.loads(safe_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid persisted video request: {safe_path}")
        shot_idx = raw.get("shot_idx")
        if not isinstance(shot_idx, int) or shot_idx < 0:
            raise ValueError(f"Invalid shot_idx in {safe_path}")
        entries.append((shot_idx, safe_path, raw))
    failed_shot_idx = getattr(error, "shot_idx", None)
    if not isinstance(failed_shot_idx, int):
        incomplete_shots = [
            shot_idx for shot_idx, _, raw in entries
            if raw.get("status") != "completed" and not raw.get("video_id")
        ]
        if len(incomplete_shots) != 1:
            raise RuntimeError("Agnes video queue was full but the failing shot was not identified")
        failed_shot_idx = incomplete_shots[0]
    for shot_idx, request_path, raw in sorted(entries, key=lambda item: item[0]):
        prior_attempts = raw.get("submit_attempts", 0)
        if not isinstance(prior_attempts, int) or prior_attempts < 0:
            raise ValueError(f"Invalid submit_attempts in {request_path}")
        job = dict(raw)
        if shot_idx == failed_shot_idx:
            job.update({
                "status": "waiting_for_video_capacity",
                "last_attempt": timestamp,
                "submit_attempts": prior_attempts + queue_attempts,
                "last_error_code": "video_queue_full",
            })
        elif job.get("status") == "pending_video_submit":
            job["status"] = "waiting_for_video_capacity"
        _atomic_write_json(request_path, job)
        jobs.append(job)
    if not jobs:
        raise RuntimeError("Agnes video queue was full but no persisted scene video request was found")
    failed_job = next(job for job in jobs if job["shot_idx"] == failed_shot_idx)
    state = {
        "schema_version": 1,
        "session_id": session_id,
        "scene_id": scene_id,
        "provider": failed_job["provider"],
        "model": failed_job["model"],
        "parameters": failed_job["parameters"],
        "reference_image_paths": failed_job["reference_image_paths"],
        "status": "waiting_for_video_capacity",
        "last_attempt": timestamp,
        "submit_attempts": sum(job["submit_attempts"] for job in jobs),
        "video_id": failed_job.get("video_id"),
        "expected_shot_indices": [job["shot_idx"] for job in jobs],
        "jobs": jobs,
    }
    _atomic_write_json(_resolve_scene_artifact_path(scene_dir, "waiting_video_capacity.json"), state)
    return state


def _write_render_status(working_dir: Path, *, status: str, payload: dict[str, Any]) -> None:
    working_dir.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        **payload,
    }
    (working_dir / "render_status.json").write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")
    with (working_dir / "render_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _write_characters_if_missing(path: Path, characters: list[CharacterInScene]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([character.model_dump() for character in characters], ensure_ascii=False, indent=2), encoding="utf-8")


def _load_characters(path: Path) -> list[CharacterInScene]:
    return [CharacterInScene.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def _load_script_text(working_dir: Path) -> str:
    script_text = working_dir / "script2video" / "script.txt"
    if script_text.exists():
        return script_text.read_text(encoding="utf-8")
    idea_script = working_dir / "idea2video" / "script.json"
    if idea_script.exists():
        payload = json.loads(idea_script.read_text(encoding="utf-8"))
        return json.dumps(payload, ensure_ascii=False, indent=2) if not isinstance(payload, str) else payload
    story = working_dir / "idea2video" / "story.txt"
    if story.exists():
        return story.read_text(encoding="utf-8")
    return ""


def _load_idea_render_plan(render_plan_path: Path) -> tuple[dict[str, Any], list[str] | None]:
    """Load optional scene render controls without changing legacy sessions."""
    if not render_plan_path.exists():
        return {}, None
    try:
        raw = json.loads(render_plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid render_plan.json: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("render_plan.json must contain an object")
    target = raw.get("target", {})
    if not isinstance(target, dict):
        raise ValueError("render_plan.json target must contain an object")

    options: dict[str, Any] = {}
    field_mapping = {
        "aspect_ratio": "aspect_ratio",
        "resolution": "resolution",
        "duration_seconds_per_scene": "seconds",
        "image_size": "image_size",
    }
    for plan_key, option_key in field_mapping.items():
        value = target.get(plan_key)
        if value is not None and str(value).strip():
            options[option_key] = value
    if "seconds" in options:
        try:
            options["seconds"] = int(options["seconds"])
        except (TypeError, ValueError) as exc:
            raise ValueError("render_plan.json duration_seconds_per_scene must be an integer") from exc
        if options["seconds"] <= 0:
            raise ValueError("render_plan.json duration_seconds_per_scene must be positive")

    requested_views = raw.get("shared_character_reference_views")
    if requested_views is None:
        return options, None
    if not isinstance(requested_views, list) or not requested_views:
        raise ValueError("render_plan.json shared_character_reference_views must be a non-empty list")
    allowed_views = {"front", "side", "back"}
    reference_views = [str(view) for view in requested_views]
    if any(view not in allowed_views for view in reference_views) or len(set(reference_views)) != len(reference_views):
        raise ValueError("render_plan.json shared_character_reference_views must contain unique front, side, and/or back values")
    return options, reference_views


def _is_valid_idea_scene_id(scene_id: str) -> bool:
    if not scene_id.startswith("scene_"):
        return False
    suffix = scene_id.removeprefix("scene_")
    return suffix.isdigit() and f"scene_{int(suffix)}" == scene_id


def _idea_scene_index(scene_id: str) -> int:
    if not _is_valid_idea_scene_id(scene_id):
        raise ValueError(f"Invalid idea scene_id: {scene_id}")
    return int(scene_id.removeprefix("scene_"))


def _missing_idea_scene_dependencies(idea_dir: Path, scene_dir: Path) -> list[str]:
    missing: list[str] = []
    if not (idea_dir / "script.json").exists():
        missing.append("idea2video/script.json")
    if not (scene_dir / "characters.json").exists() and not (idea_dir / "characters.json").exists():
        missing.append(f"{scene_dir.name}/characters.json or idea2video/characters.json")
    if not (scene_dir / "storyboard.json").exists():
        missing.append(f"{scene_dir.name}/storyboard.json")
    if not (scene_dir / "camera_tree.json").exists():
        missing.append(f"{scene_dir.name}/camera_tree.json")
    if not any((scene_dir / "shots").glob("*/shot_description.json")):
        missing.append(f"{scene_dir.name}/shots/*/shot_description.json")
    return missing


def _load_idea_scene_script(script_path: Path, scene_id: str) -> str:
    scripts = json.loads(script_path.read_text(encoding="utf-8"))
    if not isinstance(scripts, list):
        raise ValueError("idea2video/script.json must contain a scene list")
    scene_index = _idea_scene_index(scene_id)
    if scene_index >= len(scripts):
        raise ValueError(f"No script entry for {scene_id}")
    scene_script = scripts[scene_index]
    return scene_script if isinstance(scene_script, str) else json.dumps(scene_script, ensure_ascii=False, indent=2)


def _resolve_artifact_path(working_dir: Path, revision_target: str) -> Path:
    rel = Path(revision_target)
    if rel.is_absolute():
        raise ValueError(f"revision_target must be relative to session working_dir: {revision_target}")
    path = (working_dir / rel).resolve()
    if path != working_dir and working_dir not in path.parents:
        raise ValueError(f"revision_target escapes session working_dir: {revision_target}")
    return path


async def _revise_artifact_with_llm(chat_model: Any, target: str, current_text: str, instruction: str) -> str:
    prompt = (
        "Revise this ViMax structured artifact exactly as requested. "
        "Return only the complete replacement file content, with no Markdown fences or explanation. "
        "If the file is JSON, preserve valid JSON and the existing schema shape.\n\n"
        f"Target: {target}\n"
        f"Revision instruction: {instruction}\n\n"
        "Current file content:\n"
        f"{current_text}"
    )
    if hasattr(chat_model, "ainvoke"):
        response = await chat_model.ainvoke(prompt)
    elif hasattr(chat_model, "invoke"):
        response = chat_model.invoke(prompt)
    else:
        raise RuntimeError("chat_model does not support invoke/ainvoke for revision mode")
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    return _strip_markdown_fences(str(content).strip())


def _strip_markdown_fences(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _stale_keys_for_revision(target: str) -> list[str]:
    if "storyboard.json" in target:
        return ["shot_descriptions", "camera_tree", "frames", "clips", "final_video"]
    if "shot_description.json" in target:
        return ["frames", "clips", "final_video"]
    if "camera_tree.json" in target:
        return ["frames", "clips", "final_video"]
    if target.endswith("script.json") or target.endswith("story.txt"):
        return ["storyboard", "shot_descriptions", "camera_tree", "frames", "clips", "final_video"]
    if target.endswith("characters.json"):
        return ["storyboard", "shot_descriptions", "frames", "clips", "final_video"]
    return ["frames", "clips", "final_video"]


def _ready_for_render(checklist: dict[str, bool]) -> bool:
    return _idea_mode_ready(checklist) or _script_mode_ready(checklist) or _novel_mode_ready(checklist)


def _missing_render_dependencies(checklist: dict[str, bool]) -> list[str]:
    if _ready_for_render(checklist):
        return []
    idea_required = ["idea2video/story.txt", "idea2video/characters.json", "idea2video/script.json", "idea2video/scene_*/storyboard.json", "idea2video/scene_*/shots/*/shot_description.json", "idea2video/scene_*/camera_tree.json"]
    script_required = ["script2video/script.txt", "script2video/characters.json", "script2video/storyboard.json", "script2video/shots/*/shot_description.json", "script2video/camera_tree.json"]
    novel_required = ["novel2video/novel/novel_compressed.txt", "novel2video/events/event_*.json", "novel2video/relevant_chunks/event_*", "novel2video/scenes/event_*/scene_*.json", "novel2video/global_information/characters/event_level/*.json", "novel2video/global_information/characters/novel_level/*.json"]
    return [f"idea mode: {path}" for path in idea_required if not checklist.get(path)] + [f"script mode: {path}" for path in script_required if not checklist.get(path)] + [f"novel mode: {path}" for path in novel_required if not checklist.get(path)]


def _idea_mode_ready(checklist: dict[str, bool]) -> bool:
    return bool(checklist.get("idea2video/story.txt") and checklist.get("idea2video/characters.json") and checklist.get("idea2video/script.json") and checklist.get("idea2video/scene_*/storyboard.json") and checklist.get("idea2video/scene_*/shots/*/shot_description.json") and checklist.get("idea2video/scene_*/camera_tree.json"))


def _novel_text_ready(checklist: dict[str, bool]) -> bool:
    return _novel_mode_ready(checklist)


def _novel_mode_ready(checklist: dict[str, bool]) -> bool:
    return bool(checklist.get("novel2video/novel/novel_compressed.txt") and checklist.get("novel2video/events/event_*.json") and checklist.get("novel2video/relevant_chunks/event_*") and checklist.get("novel2video/scenes/event_*/scene_*.json") and checklist.get("novel2video/global_information/characters/event_level/*.json") and checklist.get("novel2video/global_information/characters/novel_level/*.json"))


def _script_mode_ready(checklist: dict[str, bool]) -> bool:
    return bool(checklist.get("script2video/script.txt") and checklist.get("script2video/characters.json") and checklist.get("script2video/storyboard.json") and checklist.get("script2video/shots/*/shot_description.json") and checklist.get("script2video/camera_tree.json"))
