"""
Small standalone tests for the spatial-knowledge helpers (camera_reasoning/spatial_knowledge.py).
No pytest in this project — plain asserts, run directly:
  .venv/bin/python examples/test_spatial_knowledge.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from camera_reasoning import CameraReasoningSession
from camera_reasoning.spatial_knowledge import (
    build_target_spatial_context,
    dump_spatial_knowledge_json,
    extract_diagnosis_sections,
    load_simple_spatial_knowledge,
    resolve_target_view,
)

SIMPLE_SPATIAL_KNOWLEDGE_PATH = "data/simple_foot_spatial_spec.json"

data = load_simple_spatial_knowledge(SIMPLE_SPATIAL_KNOWLEDGE_PATH)
assert data is not None, "expected the spec to load"


def test_resolve_target_view():
    # resolve_target_view() derives all matching keywords from the JSON itself (view
    # names, camera_from labels, main_parts names/synonyms) — nothing is hardcoded for
    # this (or any) object, so a phrase like "sole" only resolves if it's literally
    # derivable from the spec's own fields; "bottom" is, since it's a camelCase word
    # in "PlantarBottomUp" itself.
    assert resolve_target_view(data, "Give me a dorsal top-down view") == "DorsalTopDown"
    assert resolve_target_view(data, "Show the bottom of the foot") == "PlantarBottomUp"
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
    # The roll/orientation-vs-anatomical-view warning comes entirely from the JSON's
    # own image_space_notes — nothing about it is hardcoded in the code.
    context = build_target_spatial_context(data, "dorsal top-down view")
    assert "Image-space vs. anatomical-view notes:" in context
    assert data["image_space_notes"]["examples"][0] in context
    print("test_build_target_spatial_context_includes_roll_warning: OK")


def test_dump_spatial_knowledge_json_is_full_and_verbatim():
    dumped = dump_spatial_knowledge_json(data)
    # every canonical view present, not just a resolved subset — this is the live
    # session's actual context source now (see CameraReasoningSession.__init__).
    for name in data["canonical_views"]:
        assert f'"{name}"' in dumped
    assert '"should_not_dominate"' in dumped  # raw JSON keys, not summarized prose
    print("test_dump_spatial_knowledge_json_is_full_and_verbatim: OK")


FREE_FORM_RESPONSE = """
Visual observation:
The image shows the dorsal (top) surface of the foot with toes near the bottom and
the ankle region near the top. There is a small amount of side depth visible.

Camera position inference:
Based on the spatial knowledge above, the camera is close to the DorsalTopDown
position but slightly oblique rather than perfectly perpendicular to the dorsal surface.

Reasoning:
Since the target is DorsalTopDown and the current view is mostly correct but oblique,
a small elevation adjustment should straighten it out.

Next action:
ELEVATION_UP_FINE

Expected visual change:
Side depth will decrease and the view will look more directly top-down.
"""


def test_extract_diagnosis_sections_valid():
    sections = extract_diagnosis_sections(FREE_FORM_RESPONSE)
    assert "toes near the bottom" in sections["visual_observation"]
    assert "DorsalTopDown" in sections["camera_position_inference"]
    # sections must not bleed into each other or into Reasoning/Next action
    assert "Reasoning" not in sections["camera_position_inference"]
    assert "ELEVATION_UP_FINE" not in sections["camera_position_inference"]
    print("test_extract_diagnosis_sections_valid: OK")


def test_extract_diagnosis_sections_missing():
    assert extract_diagnosis_sections("Next action:\nSTOP") == {}
    print("test_extract_diagnosis_sections_missing: OK")


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
    # No "Visual observation"/"Camera position inference" sections at all — must not crash,
    # and (unlike the old strict-JSON gate) the action still applies since there's no longer
    # a validation gate blocking it — diagnosis sections are logged, not enforced.
    action = session.process_chatgpt_response("Next action:\nSTOP")
    assert action == "STOP"
    print("test_session_logging_does_not_crash_on_missing_diagnosis: OK")


if __name__ == "__main__":
    test_resolve_target_view()
    test_build_target_spatial_context_includes_only_target_view()
    test_build_target_spatial_context_unresolved_lists_all_views()
    test_build_target_spatial_context_includes_roll_warning()
    test_dump_spatial_knowledge_json_is_full_and_verbatim()
    test_extract_diagnosis_sections_valid()
    test_extract_diagnosis_sections_missing()
    test_session_logging_does_not_crash_on_missing_diagnosis()
    print("\nAll tests passed.")
