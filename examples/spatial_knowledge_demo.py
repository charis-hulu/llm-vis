"""
Small demo for the target-specific spatial-knowledge diagnostic rubric.

Shows, without needing VTK or a live API call, that:
  1. the JSON spec loads correctly,
  2. build_target_spatial_context() narrows the canonical-view section down to just
     the one view resolved from the target description (this is what session.py
     actually uses now — see camera_reasoning/session.py __init__),
  3. write_llm_prompt() requires a SPATIAL_DIAGNOSIS JSON block ahead of the final
     action line, and
  4. a hypothetical model response using that block parses correctly with
     extract_spatial_diagnosis(), alongside the existing "Next action:" contract
     (so action_parser is unaffected).

The older build_simple_spatial_context() / extract_structured_fields() functions
are kept only for backward compatibility — see their own quick check at the bottom.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from camera_reasoning.action_parser import extract_action
from camera_reasoning.prompt_writer import write_llm_prompt
from camera_reasoning.spatial_knowledge import (
    build_simple_spatial_context,
    build_target_spatial_context,
    extract_spatial_diagnosis,
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

# 2. Build the target-specific context (this is what CameraReasoningSession uses).
context = build_target_spatial_context(data, SAMPLE_TARGET)
print("=== Target-specific spatial context ===")
print(context)
assert "Target view contract (resolved from target description): DorsalTopDown" in context
assert "PlantarBottomUp" not in context  # only the resolved view's contract is included
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
print("=== Prompt requires a SPATIAL_DIAGNOSIS block ===")
assert "SPATIAL_DIAGNOSIS:" in prompt
assert "DorsalTopDown" in prompt
assert "correct_view, wrong_opposite_view, too_oblique" in prompt
print("OK — prompt requires the JSON diagnostic block before the final action line.")
print()

# 4. Simulate a hypothetical model response and confirm both the SPATIAL_DIAGNOSIS block
#    and the existing action-parsing contract work side by side.
sample_response = """
SPATIAL_DIAGNOSIS:
```json
{
  "target_view": "DorsalTopDown",
  "raw_visual_observation": {
    "visible_surface": "mostly dorsal",
    "visible_parts": ["toes", "metatarsals", "ankle_region"],
    "dominant_side": "neither",
    "closest_region": "unclear",
    "side_depth": "moderate",
    "image_roll": "toes appear bottom",
    "confidence": 0.72
  },
  "target_contract_check": {
    "should_see_present": ["toes", "metatarsals", "ankle_region"],
    "should_see_missing": [],
    "should_not_dominate_violations": ["strong side depth"],
    "evidence_for_target": ["dorsal surface appears dominant"],
    "evidence_against_target": ["side depth is still moderate"]
  },
  "diagnosis": {
    "current_view_estimate": "close to dorsal top-down but still oblique",
    "likely_issue_type": "too_oblique"
  },
  "minimal_update": {
    "recommended_update": "reduce oblique tilt and align the camera more directly with the dorsal-to-plantar axis",
    "minimal_change_rationale": "only the elevation angle needs adjusting; roll and main view are already correct"
  }
}
```

Next action:
ELEVATION_UP_FINE

Expected visual change:
Side depth will decrease and the view will look more directly top-down.
"""

diagnosis = extract_spatial_diagnosis(sample_response)
print("=== Parsed SPATIAL_DIAGNOSIS ===")
print(f"  target_view: {diagnosis['target_view']}")
print(f"  likely_issue_type: {diagnosis['diagnosis']['likely_issue_type']}")
print(f"  recommended_update: {diagnosis['minimal_update']['recommended_update']}")

assert diagnosis["target_view"] == "DorsalTopDown"
assert diagnosis["diagnosis"]["likely_issue_type"] == "too_oblique"

action = extract_action(sample_response)
print(f"\nExtracted action (existing contract, unaffected): {action}")
assert action == "ELEVATION_UP_FINE"

# --- Backward-compat check: the older generic (non-target-specific) helpers still work. ---
legacy_context = build_simple_spatial_context(data)
assert "Canonical views (pick exactly one for the target):" in legacy_context
legacy_fields = extract_structured_fields("target_view: DorsalTopDown\nlikely_issue_type: too_oblique")
assert legacy_fields["target_view"] == "DorsalTopDown"
print("\nBackward-compat OK: build_simple_spatial_context() / extract_structured_fields() still work.")

print("\nAll checks passed.")
