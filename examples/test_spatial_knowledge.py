"""
Small standalone tests for the target-specific spatial-knowledge diagnostic rubric
(camera_reasoning/spatial_knowledge.py). No pytest in this project — plain asserts,
run directly: .venv/bin/python examples/test_spatial_knowledge.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from camera_reasoning import CameraReasoningSession
from camera_reasoning.spatial_knowledge import (
    build_target_spatial_context,
    extract_spatial_diagnosis,
    load_simple_spatial_knowledge,
    resolve_target_view,
)

SIMPLE_SPATIAL_KNOWLEDGE_PATH = "data/simple_foot_spatial_spec.json"

data = load_simple_spatial_knowledge(SIMPLE_SPATIAL_KNOWLEDGE_PATH)
assert data is not None, "expected the spec to load"


def test_resolve_target_view():
    assert resolve_target_view(data, "Give me a dorsal top-down view") == "DorsalTopDown"
    assert resolve_target_view(data, "Show the sole of the foot") == "PlantarBottomUp"
    assert resolve_target_view(data, "big toe side / hallux view please") == "MedialSide"
    assert resolve_target_view(data, "little toe side view") == "LateralSide"
    assert resolve_target_view(data, "view from the toe end") == "DistalToeEnd"
    assert resolve_target_view(data, "view from the ankle end") == "ProximalAnkleEnd"
    # exact canonical view name wins outright
    assert resolve_target_view(data, "Please render LateralSide exactly") == "LateralSide"
    # nothing matches
    assert resolve_target_view(data, "make it look nice") is None
    print("test_resolve_target_view: OK")


def test_build_target_spatial_context_includes_only_target_view():
    context = build_target_spatial_context(data, "dorsal top-down view of the foot")
    assert "Target view contract (resolved from target description): DorsalTopDown" in context
    # the other five canonical view names should NOT appear as their own contract sections
    for other in ("PlantarBottomUp", "MedialSide", "LateralSide", "DistalToeEnd", "ProximalAnkleEnd"):
        assert f"contract (resolved from target description): {other}" not in context
    print("test_build_target_spatial_context_includes_only_target_view: OK")


def test_build_target_spatial_context_unresolved_lists_all_views():
    context = build_target_spatial_context(data, "make it look nice")
    assert "could not be resolved automatically" in context
    for name in data["canonical_views"]:
        assert name in context
    print("test_build_target_spatial_context_unresolved_lists_all_views: OK")


def test_build_target_spatial_context_includes_roll_warning():
    context = build_target_spatial_context(data, "dorsal top-down view")
    assert "image-space roll/orientation" in context
    assert "not proof of anatomical view" in context or "not" in context
    print("test_build_target_spatial_context_includes_roll_warning: OK")


VALID_DIAGNOSIS_RESPONSE = """
SPATIAL_DIAGNOSIS:
```json
{
  "target_view": "DorsalTopDown",
  "raw_visual_observation": {
    "visible_surface": "mostly dorsal",
    "visible_parts": ["toes", "metatarsals"],
    "dominant_side": "neither",
    "closest_region": "unclear",
    "side_depth": "moderate",
    "image_roll": "toes appear bottom",
    "confidence": 0.7
  },
  "target_contract_check": {
    "should_see_present": ["toes", "metatarsals"],
    "should_see_missing": [],
    "should_not_dominate_violations": ["strong side depth"],
    "evidence_for_target": ["dorsal surface visible"],
    "evidence_against_target": ["side depth present"]
  },
  "diagnosis": {
    "current_view_estimate": "close but oblique",
    "likely_issue_type": "too_oblique"
  },
  "minimal_update": {
    "recommended_update": "reduce oblique tilt",
    "minimal_change_rationale": "only elevation needs adjusting"
  }
}
```

Next action:
ELEVATION_UP_FINE
"""


def test_extract_spatial_diagnosis_valid():
    diagnosis = extract_spatial_diagnosis(VALID_DIAGNOSIS_RESPONSE)
    assert diagnosis["target_view"] == "DorsalTopDown"
    assert diagnosis["diagnosis"]["likely_issue_type"] == "too_oblique"
    assert diagnosis["raw_visual_observation"]["side_depth"] == "moderate"
    print("test_extract_spatial_diagnosis_valid: OK")


def test_extract_spatial_diagnosis_missing_or_invalid():
    assert extract_spatial_diagnosis("no diagnosis block here, just text") == {}
    broken = "SPATIAL_DIAGNOSIS:\n```json\n{ this is not valid json \n```\nNext action:\nSTOP"
    assert extract_spatial_diagnosis(broken) == {}
    print("test_extract_spatial_diagnosis_missing_or_invalid: OK")


def test_session_logging_does_not_crash_on_missing_diagnosis():
    session = CameraReasoningSession(
        raw_path="data/foot_256x256x256_uint8.raw",
        dimensions=(256, 256, 256),
        scalar_type="uint8",
        isovalue=80,
        output_dir="output",
        target_description="dorsal top-down view of the foot",
        simple_spatial_knowledge_path=SIMPLE_SPATIAL_KNOWLEDGE_PATH,
    )
    session.initialize()
    session.render_and_save()
    # No SPATIAL_DIAGNOSIS block at all — should warn, not raise, and still parse the action.
    action = session.process_chatgpt_response("Next action:\nSTOP")
    assert action == "STOP"
    assert session._last_spatial_diagnosis == {}
    print("test_session_logging_does_not_crash_on_missing_diagnosis: OK")


if __name__ == "__main__":
    test_resolve_target_view()
    test_build_target_spatial_context_includes_only_target_view()
    test_build_target_spatial_context_unresolved_lists_all_views()
    test_build_target_spatial_context_includes_roll_warning()
    test_extract_spatial_diagnosis_valid()
    test_extract_spatial_diagnosis_missing_or_invalid()
    test_session_logging_does_not_crash_on_missing_diagnosis()
    print("\nAll tests passed.")
