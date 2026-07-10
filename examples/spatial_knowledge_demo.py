"""
Small demo for the spatial-knowledge feature as it's actually wired up today.

Shows, without needing VTK or a live API call, that:
  1. the JSON spec loads correctly,
  2. dump_spatial_knowledge_json() returns the full spec verbatim (this is what
     CameraReasoningSession.__init__ actually uses as spatial_context now — see
     camera_reasoning/session.py), not a narrowed/summarized subset,
  3. write_llm_prompt() asks the model for free-form "Visual observation" and
     "Camera position inference" sections — no forced JSON schema — ahead of the
     final action line, and
  4. a hypothetical model response using that free-text format parses correctly
     with extract_diagnosis_sections(), alongside the existing "Next action:"
     contract (so action_parser is unaffected).

build_target_spatial_context() / build_simple_spatial_context() (the older
narrowed/generic summarizers) and extract_structured_fields() are kept only for
backward compatibility — see their own quick check at the bottom.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from camera_reasoning.action_parser import extract_action
from camera_reasoning.prompt_writer import write_llm_prompt
from camera_reasoning.spatial_knowledge import (
    build_simple_spatial_context,
    build_target_spatial_context,
    dump_spatial_knowledge_json,
    extract_diagnosis_sections,
    extract_structured_fields,
    load_simple_spatial_knowledge,
)

SIMPLE_SPATIAL_KNOWLEDGE_PATH = "data/simple_foot_spatial_spec.json"

SAMPLE_TARGET = (
    "Show the foot bones in dorsal top-down view with toes at the bottom and ankle at the top."
)

# 1. Load the JSON spec.
data = load_simple_spatial_knowledge(SIMPLE_SPATIAL_KNOWLEDGE_PATH)
assert data is not None, "expected the spec to load"

# 2. Dump the full spec verbatim (this is what CameraReasoningSession uses).
context = dump_spatial_knowledge_json(data)
print("=== Full JSON spatial context (first 300 chars) ===")
print(context[:300] + "...")
for name in data["canonical_views"]:
    assert f'"{name}"' in context  # ALL six views present, nothing narrowed
print()

# 3. Build one real prompt using the existing prompt_writer, with the spatial context injected.
fake_camera_state = {
    "position": [0.0, 0.0, 800.0],
    "focal_point": [0.0, 0.0, 0.0],
    "view_up": [0.0, 1.0, 0.0],
    "view_angle": 30.0,
    "clipping_range": [500.0, 1100.0],
}
prompt = write_llm_prompt(
    output_dir="output",
    camera_state=fake_camera_state,
    action_history=[],
    target_description=SAMPLE_TARGET,
    screenshot_path="output/screenshots/latest.png",
    spatial_context=context,
)
print("=== Prompt asks for free-form diagnosis sections (no forced JSON schema) ===")
assert "Visual observation:" in prompt
assert "Camera position inference:" in prompt
assert "SPATIAL_DIAGNOSIS" not in prompt  # the old strict-JSON gate is gone
print("OK — prompt requires observation + position inference before the final action line.")
print()

# 4. Simulate a hypothetical model response and confirm both the free-text sections
#    and the existing action-parsing contract work side by side.
sample_response = """
Visual observation:
The image shows the dorsal (top) surface of the foot with toes near the bottom and
the ankle region near the top. There is a small amount of side depth visible.

Camera position inference:
The camera is close to the DorsalTopDown position but slightly oblique rather than
perfectly perpendicular to the dorsal surface.

Reasoning:
Since the target is DorsalTopDown and the current view is mostly correct but oblique,
a small elevation adjustment should straighten it out.

Next action:
ELEVATION_UP_FINE

Expected visual change:
Side depth will decrease and the view will look more directly top-down.
"""

sections = extract_diagnosis_sections(sample_response)
print("=== Parsed diagnosis sections ===")
print(f"  visual_observation: {sections['visual_observation']}")
print(f"  camera_position_inference: {sections['camera_position_inference']}")

assert "toes near the bottom" in sections["visual_observation"]
assert "DorsalTopDown" in sections["camera_position_inference"]

action = extract_action(sample_response)
print(f"\nExtracted action (existing contract, unaffected): {action}")
assert action == "ELEVATION_UP_FINE"

# --- Backward-compat check: the older narrowed/generic helpers still work standalone. ---
target_specific_context = build_target_spatial_context(data, SAMPLE_TARGET)
assert "Target view contract (resolved from target description): DorsalTopDown" in target_specific_context
legacy_context = build_simple_spatial_context(data)
assert "Canonical views (pick exactly one for the target):" in legacy_context
legacy_fields = extract_structured_fields("target_view: DorsalTopDown\nlikely_issue_type: too_oblique")
assert legacy_fields["target_view"] == "DorsalTopDown"
print("\nBackward-compat OK: build_target_spatial_context() / build_simple_spatial_context() / "
      "extract_structured_fields() still work standalone, even though the live session no longer uses them.")

print("\nAll checks passed.")
