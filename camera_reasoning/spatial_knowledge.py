"""
Lightweight support for an optional "simple spatial knowledge" JSON spec (e.g.
data/simple_foot_spatial_spec.json) that gives the LLM object-space grounding
(anatomical terms, canonical views, should_see/should_not_dominate checks) so it
doesn't confuse image-space layout (e.g. "toes at the bottom") with the actual
camera view. This is intentionally a flat, human-readable spec, not an ontology.

Everything here is optional: if no path is given, or the file is missing/invalid,
callers get None back and the rest of the pipeline behaves exactly as before.
"""
import json
import re
from pathlib import Path
from typing import Optional


def load_simple_spatial_knowledge(path: str) -> Optional[dict]:
    """Load a simple spatial-knowledge JSON spec. Returns None if unavailable or invalid."""
    file_path = Path(path)
    if not file_path.exists():
        print(f"[spatial_knowledge] No file at {path!r} — continuing without spatial context.")
        return None

    try:
        data = json.loads(file_path.read_text())
    except json.JSONDecodeError as e:
        print(f"[spatial_knowledge] {path!r} is not valid JSON ({e}) — continuing without spatial context.")
        return None

    print(f"[spatial_knowledge] Loaded {data.get('name', path)!r} (version={data.get('version', '?')}).")
    return data


_VIEW_KEYWORDS = {
    "DorsalTopDown": ["dorsal", "top down", "top-down", "topdown", "top view"],
    "PlantarBottomUp": ["plantar", "bottom", "sole"],
    "MedialSide": ["medial", "big toe side", "hallux"],
    "LateralSide": ["lateral", "little toe side", "fifth toe side"],
    "DistalToeEnd": ["distal", "toe end", "from toes"],
    "ProximalAnkleEnd": ["proximal", "ankle end", "heel end", "from ankle"],
}


def resolve_target_view(data: dict, target_description: str) -> Optional[str]:
    """Map a free-text target description to one canonical view name, if possible.

    Deterministic, two-pass matching against data["canonical_views"]:
      1. An exact canonical view name appearing anywhere in the description wins
         outright (case-insensitive), e.g. "...render a DorsalTopDown view".
      2. Otherwise, fall back to simple lowercase keyword matching (_VIEW_KEYWORDS).

    Returns None if nothing matches — callers should then ask the model to pick
    one of the canonical views itself instead of assuming a target view.
    """
    views = data.get("canonical_views", {})
    if not views or not target_description:
        return None

    text = target_description.lower()

    for view_name in views:
        if view_name.lower() in text:
            return view_name

    for view_name, keywords in _VIEW_KEYWORDS.items():
        if view_name not in views:
            continue
        if any(keyword in text for keyword in keywords):
            return view_name

    return None


def build_target_spatial_context(data: dict, target_description: str) -> str:
    """Target-specific spatial context: same object/terms/facts/rules as
    build_simple_spatial_context(), but the canonical-view section is narrowed to
    exactly the resolved target view's contract instead of listing all six — so the
    model gets one concrete should_see/should_not_dominate checklist to diagnose
    against, rather than a wall of view options to sift through itself.

    Falls back to listing all canonical views (asking the model to pick exactly
    one) if resolve_target_view() can't resolve the target description.
    """
    lines = [f"Object: {data.get('object', '?')}"]

    terms = data.get("anatomical_terms", {})
    if terms:
        lines.append("Terms: " + "; ".join(f"{k}={v}" for k, v in terms.items()))

    parts = data.get("main_parts", {})
    if parts:
        lines.append(
            "Main parts: "
            + "; ".join(f"{name} ({info.get('location', '?')})" for name, info in parts.items())
        )

    facts = data.get("stable_object_space_facts", [])
    if facts:
        lines.append("Stable facts: " + " ".join(facts))

    views = data.get("canonical_views", {})
    target_view = resolve_target_view(data, target_description)

    if target_view and target_view in views:
        view = views[target_view]
        lines.append(f"Target view contract (resolved from target description): {target_view}")
        lines.append(f"  meaning: {view.get('meaning', '')}")
        lines.append(f"  camera_from: {view.get('camera_from', '?')}")
        lines.append(f"  looks_toward: {view.get('looks_toward', '?')}")
        lines.append(f"  should_see: {', '.join(view.get('should_see', []))}")
        lines.append(f"  should_not_dominate: {', '.join(view.get('should_not_dominate', []))}")
        lines.append(f"  simple_check: {view.get('simple_check', '')}")
    elif views:
        lines.append(
            "Target view could not be resolved automatically from the target description. "
            "Choose exactly one of the following canonical views yourself:"
        )
        for name, view in views.items():
            lines.append(f"  - {name}: {view.get('meaning', '')}")

    notes = data.get("image_space_notes", {}).get("examples", [])
    if notes:
        lines.append("Image-space vs. anatomical-view notes: " + " ".join(notes))

    rules = data.get("minimal_reasoning_rules", [])
    if rules:
        lines.append(
            "Reasoning rules: "
            + " ".join(f"{r.get('rule')} — {r.get('instruction')}" for r in rules)
        )

    lines.append(
        "Do not treat image-space roll/orientation (e.g. toes at bottom/top/left/right) as "
        "proof of anatomical view by itself — that is screen layout, not object-space view."
    )

    return "\n".join(lines)


def build_simple_spatial_context(data: dict) -> str:
    """Compact, human-readable summary of a simple spatial-knowledge spec for prompt injection.

    Deliberately a short prose/bullet summary, not a dump of the raw JSON.
    """
    lines = []

    lines.append(f"Object: {data.get('object', '?')}")

    terms = data.get("anatomical_terms", {})
    if terms:
        lines.append("Terms: " + "; ".join(f"{k}={v}" for k, v in terms.items()))

    parts = data.get("main_parts", {})
    if parts:
        lines.append(
            "Main parts: "
            + "; ".join(f"{name} ({info.get('location', '?')})" for name, info in parts.items())
        )

    facts = data.get("stable_object_space_facts", [])
    if facts:
        lines.append("Stable facts: " + " ".join(facts))

    views = data.get("canonical_views", {})
    if views:
        lines.append("Canonical views (pick exactly one for the target):")
        for name, view in views.items():
            should_see = ", ".join(view.get("should_see", []))
            should_not = ", ".join(view.get("should_not_dominate", []))
            lines.append(
                f"  - {name}: {view.get('meaning', '')} "
                f"Should see: {should_see}. Should NOT dominate: {should_not}."
            )

    notes = data.get("image_space_notes", {}).get("examples", [])
    if notes:
        lines.append("Image-space vs. anatomical-view notes: " + " ".join(notes))

    rules = data.get("minimal_reasoning_rules", [])
    if rules:
        lines.append(
            "Reasoning rules: "
            + " ".join(f"{r.get('rule')} — {r.get('instruction')}" for r in rules)
        )

    return "\n".join(lines)


STRUCTURED_FIELDS = [
    "target_view",
    "current_view_estimate",
    "key_mismatches",
    "likely_issue_type",
    "recommended_update",
    "minimal_change_rationale",
]


def extract_structured_fields(response: str) -> dict:
    """Best-effort extraction of the optional structured spatial-reasoning fields
    (target_view, current_view_estimate, key_mismatches, likely_issue_type,
    recommended_update, minimal_change_rationale) from a response, for logging only.

    Fields that aren't present are simply omitted — this never raises and never
    blocks the main action-parsing flow.
    """
    fields = {}
    for name in STRUCTURED_FIELDS:
        match = re.search(rf"{name}\s*:\s*(.+)", response, re.IGNORECASE)
        if match:
            fields[name] = match.group(1).strip()
    return fields


# Allowed values for diagnosis.likely_issue_type in the SPATIAL_DIAGNOSIS block. Not
# enforced by extract_spatial_diagnosis() (that would risk raising on an odd-but-useful
# model answer) — this is just the documented vocabulary the prompt asks the model to use.
LIKELY_ISSUE_TYPES = [
    "correct_view",
    "wrong_opposite_view",
    "too_oblique",
    "roll_only",
    "wrong_surface",
    "wrong_end_view",
    "unclear_low_confidence",
]


def extract_spatial_diagnosis(response: str) -> dict:
    """Parse the required SPATIAL_DIAGNOSIS JSON block out of a model response.

    Expects:
        SPATIAL_DIAGNOSIS:
        ```json
        { ... }
        ```
    (a fenced code block, "json" tag optional, immediately after the label). Returns
    {} if the block is missing or isn't valid JSON — this is best-effort, for logging
    and (later) validation, and must never raise or block the action-parsing flow.
    """
    match = re.search(
        r"SPATIAL_DIAGNOSIS\s*:\s*```(?:json)?\s*(.*?)```",
        response,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return {}

    try:
        data = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return {}

    return data if isinstance(data, dict) else {}
