import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .action_parser import extract_action
from .camera_actions import VALID_ACTIONS, apply_action
from .camera_state import (
    camera_distance,
    get_camera_state,
    load_camera_state_json,
    save_camera_state_json,
    set_camera_state,
)
from .chatgpt_client import ask_chatgpt
from .prompt_writer import write_llm_prompt
from .spatial_knowledge import (
    build_target_spatial_context,
    extract_spatial_diagnosis,
    load_simple_spatial_knowledge,
)
from .volume_scene import build_isosurface_pipeline, load_raw_volume, save_screenshot

MIN_CAMERA_DISTANCE = 1e-3


class CameraReasoningSession:
    def __init__(
        self,
        raw_path: str,
        dimensions: tuple,
        scalar_type: str = "uint8",
        isovalue: float = 80,
        output_dir: str = "output",
        target_description: str = "No target description provided.",
        target_image_path: Optional[str] = None,
        simple_spatial_knowledge_path: Optional[str] = None,
    ):
        self.raw_path = raw_path
        self.dimensions = dimensions
        self.scalar_type = scalar_type
        self.isovalue = isovalue
        self.output_dir = Path(output_dir)
        self.target_description = target_description
        self.target_image_path = target_image_path

        self._actor = None
        self._renderer = None
        self._render_window = None
        self._image_data = None

        self._step = 0
        self._action_history: List[Dict] = []
        self._camera_state_stack: List[Dict] = []

        # Optional lightweight spatial-knowledge spec (see camera_reasoning/spatial_knowledge.py).
        # Fully backward compatible: if no path is given, or loading fails, self._spatial_context
        # stays None and the prompt/response format is unchanged from before this feature existed.
        self._spatial_data: Optional[dict] = None
        self._spatial_context: Optional[str] = None
        self._last_spatial_diagnosis: dict = {}  # last parsed SPATIAL_DIAGNOSIS block, for future strict checks
        if simple_spatial_knowledge_path:
            self._spatial_data = load_simple_spatial_knowledge(simple_spatial_knowledge_path)
            if self._spatial_data:
                self._spatial_context = build_target_spatial_context(
                    self._spatial_data, self.target_description
                )
                print("[spatial_knowledge] Spatial context for this session:")
                print(self._spatial_context)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self):
        """Load data, build VTK scene, and reset the camera."""
        self._make_output_dirs()
        self._image_data = load_raw_volume(self.raw_path, self.dimensions, self.scalar_type)
        self._actor, self._renderer, self._render_window = build_isosurface_pipeline(
            self._image_data, self.isovalue
        )
        self._renderer.ResetCamera()
        print(f"Scene initialized. Isovalue={self.isovalue}, dims={self.dimensions}.")

    def render_and_save(self) -> str:
        """Render the current scene and save screenshots. Returns the latest screenshot path."""
        self._require_initialized()
        self._render_window.Render()
        return self._save_screenshot(action_name=None)

    def write_llm_prompt(self) -> str:
        """Write the LLM prompt file and return its text content."""
        self._require_initialized()
        return write_llm_prompt(
            output_dir=str(self.output_dir),
            camera_state=get_camera_state(self._renderer.GetActiveCamera()),
            action_history=self._action_history,
            target_description=self.target_description,
            screenshot_path=str(self.output_dir / "screenshots" / "latest.png"),
            target_image_path=self.target_image_path,
            spatial_context=self._spatial_context,
        )

    def ask_chatgpt(self, prompt: Optional[str] = None, model: Optional[str] = None) -> str:
        """Send a prompt + the current screenshot to the OpenAI API and return the raw reply.

        Pass `prompt` to experiment with custom wording instead of the auto-generated one
        (e.g. write_llm_prompt(), edit the text, then hand it in here). The response is not
        applied automatically — pass it to process_chatgpt_response() when you're ready.
        """
        self._require_initialized()
        if prompt is None:
            prompt = self.write_llm_prompt()
        return ask_chatgpt(
            prompt=prompt,
            screenshot_path=str(self.output_dir / "screenshots" / "latest.png"),
            target_image_path=self.target_image_path,
            model=model,
        )

    def ask_chatgpt_and_process(self, prompt: Optional[str] = None, model: Optional[str] = None) -> str:
        """Convenience wrapper: ask_chatgpt() followed by process_chatgpt_response(). Returns the applied action."""
        response = self.ask_chatgpt(prompt=prompt, model=model)
        print(response)
        return self.process_chatgpt_response(response)

    def process_chatgpt_response(self, response: str) -> str:
        """Parse a pasted ChatGPT response, apply the action, and advance the loop. Returns the action name."""
        self._require_initialized()

        if self._spatial_context:
            self._log_spatial_diagnosis(response)

        action = extract_action(response)
        print(f"Extracted action: {action}")

        if action == "STOP":
            print("STOP received — alignment marked as complete.")
            self._append_history(action)
            self._save_action_history()
            return action

        if action == "UNDO_LAST":
            self._do_undo()
        else:
            self._apply_and_advance(action)
        return action

    def reset_camera(self):
        """Hard-reset camera to fit the scene (destroys manual alignment)."""
        self._require_initialized()
        self._renderer.ResetCamera()
        self._renderer.ResetCameraClippingRange()
        print("Camera hard-reset to fit scene.")

    def load_camera_state(self, path: str):
        """Restore camera from a saved JSON file."""
        self._require_initialized()
        state = load_camera_state_json(path)
        set_camera_state(self._renderer.GetActiveCamera(), state)
        self._renderer.ResetCameraClippingRange()
        print(f"Camera state loaded from {path}.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_spatial_diagnosis(self, response: str):
        """Parse and log the required SPATIAL_DIAGNOSIS block. Never raises: a missing or
        malformed block only produces a warning, it doesn't block action application.

        self._last_spatial_diagnosis is kept around so a future caller can add strict
        enforcement (e.g. reject the action if likely_issue_type is missing/inconsistent)
        without having to change how the response is parsed.
        """
        diagnosis = extract_spatial_diagnosis(response)
        self._last_spatial_diagnosis = diagnosis

        if not diagnosis:
            print(
                "[spatial_knowledge] Missing SPATIAL_DIAGNOSIS block; "
                "model may not have followed the required diagnostic format."
            )
            return

        raw_obs = diagnosis.get("raw_visual_observation", {})
        diag = diagnosis.get("diagnosis", {})
        update = diagnosis.get("minimal_update", {})

        print("[spatial_knowledge] Parsed SPATIAL_DIAGNOSIS:")
        print(f"  target_view: {diagnosis.get('target_view', '?')}")
        print(f"  visible_surface: {raw_obs.get('visible_surface', '?')}")
        print(f"  visible_parts: {raw_obs.get('visible_parts', '?')}")
        print(f"  dominant_side: {raw_obs.get('dominant_side', '?')}")
        print(f"  closest_region: {raw_obs.get('closest_region', '?')}")
        print(f"  side_depth: {raw_obs.get('side_depth', '?')}")
        print(f"  image_roll: {raw_obs.get('image_roll', '?')}")
        print(f"  current_view_estimate: {diag.get('current_view_estimate', '?')}")
        print(f"  likely_issue_type: {diag.get('likely_issue_type', '?')}")
        print(f"  recommended_update: {update.get('recommended_update', '?')}")

    def _require_initialized(self):
        if self._renderer is None:
            raise RuntimeError("Call session.initialize() first.")

    def _make_output_dirs(self):
        for subdir in ("screenshots", "camera_states"):
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _apply_and_advance(self, action: str):
        camera = self._renderer.GetActiveCamera()
        prev_state = get_camera_state(camera)
        self._camera_state_stack.append(prev_state)

        apply_action(action, camera, self._renderer)

        new_state = get_camera_state(camera)
        if camera_distance(new_state) < MIN_CAMERA_DISTANCE:
            print("WARNING: camera too close to focal point — reverting.")
            set_camera_state(camera, prev_state)
            self._camera_state_stack.pop()
            return

        self._step += 1
        self._append_history(action)

        self._render_window.Render()
        self._save_screenshot(action_name=action)
        self._save_camera_state(action_name=action)
        self._save_action_history()
        self.write_llm_prompt()
        print(f"Step {self._step}: {action} applied.")

    def _do_undo(self):
        if not self._camera_state_stack:
            print("Nothing to undo.")
            return
        camera = self._renderer.GetActiveCamera()
        prev_state = self._camera_state_stack.pop()
        set_camera_state(camera, prev_state)
        self._renderer.ResetCameraClippingRange()

        self._step += 1
        self._append_history("UNDO_LAST")
        self._render_window.Render()
        self._save_screenshot(action_name="UNDO_LAST")
        self._save_camera_state(action_name="UNDO_LAST")
        self._save_action_history()
        self.write_llm_prompt()
        print(f"Step {self._step}: UNDO_LAST applied.")

    def _save_screenshot(self, action_name: Optional[str]) -> str:
        latest = str(self.output_dir / "screenshots" / "latest.png")
        if action_name is None:
            step_name = f"step_{self._step:03d}.png"
        else:
            step_name = f"step_{self._step:03d}_{action_name}.png"
        step_path = str(self.output_dir / "screenshots" / step_name)

        save_screenshot(self._render_window, latest)
        shutil.copy(latest, step_path)
        return latest

    def _save_camera_state(self, action_name: Optional[str]):
        camera = self._renderer.GetActiveCamera()
        state = get_camera_state(camera)
        latest = self.output_dir / "camera_states" / "latest_camera.json"
        save_camera_state_json(state, latest)
        if action_name:
            step_path = (
                self.output_dir / "camera_states" / f"step_{self._step:03d}_{action_name}.json"
            )
            save_camera_state_json(state, step_path)

    def _append_history(self, action: str):
        self._action_history.append({"step": self._step, "action": action})

    def _save_action_history(self):
        path = self.output_dir / "action_history.json"
        with open(path, "w") as f:
            json.dump(self._action_history, f, indent=2)
