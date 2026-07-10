"""
Lightweight support for an optional "simple spatial knowledge" JSON spec that gives
the LLM object-space grounding (spatial terms, canonical views, should_see/
should_not_dominate checks) so it doesn't confuse image-space layout (where
something appears on screen) with the actual camera view. This is intentionally a
flat, human-readable spec, not an ontology.

Nothing in this module hardcodes any particular object or vocabulary — every
domain-specific term is read from whichever JSON file is passed in. Different specs
for entirely different objects work with the exact same code, as long as they follow
the same shape (canonical_views with camera_from/looks_toward, main_parts, etc.).

Everything here is optional: if no path is given, or the file is missing/invalid,
callers get None back and the rest of the pipeline behaves exactly as before.
"""
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def dump_spatial_knowledge_json(data: dict) -> str:
    """Return the full spatial-knowledge JSON, pretty-printed, verbatim.

    Unlike build_target_spatial_context()/build_simple_spatial_context() (which summarize
    into prose and, for the target-specific version, narrow to just the resolved view),
    this dumps the entire spec exactly as loaded — every canonical view, every field —
    with nothing filtered out or reworded. Use this when you want the model to see the
    complete spec rather than a curated subset.
    """
    return json.dumps(data, indent=2)


def _camel_case_words(name: str) -> List[str]:
    """Split a CamelCase identifier like 'FrontTopView' into ['Front', 'Top', 'View']."""
    return re.findall(r"[A-Z][a-z0-9]*", name)


def _normalize_phrase(text: str) -> str:
    """Lowercase and collapse hyphens/underscores/slashes to spaces, so JSON-derived
    phrases (e.g. 'left_side_panel') match differently-punctuated free text (e.g.
    'left side panel', 'left-side panel') the same way.
    """
    return re.sub(r"[-_/]+", " ", str(text).lower()).strip()


def _shared_camel_words(views: dict) -> set:
    """Words shared by more than one canonical view's CamelCase name (e.g. a common
    word appearing in two different view names) provide no discriminating power for
    keyword matching and are excluded from single-word candidates (they're still
    used in multi-word phrases, where they're unambiguous).
    """
    counts: Dict[str, int] = {}
    for view_name in views:
        for word in set(w.lower() for w in _camel_case_words(view_name)):
            counts[word] = counts.get(word, 0) + 1
    return {word for word, count in counts.items() if count > 1}


def _derive_view_phrases(data: dict, view_name: str, view: dict, shared_words: set) -> List[str]:
    """Candidate matching phrases for `view_name`, derived entirely from the JSON
    spec — no hardcoded domain vocabulary. Sources: the view name's own words (and
    their joined form), its camera_from label, and any main_parts (key name +
    also_called synonyms) located on that same side/end.
    """
    parts = data.get("main_parts", {})
    words = _camel_case_words(view_name)
    phrases = [_normalize_phrase(w) for w in words if w.lower() not in shared_words]
    phrases.append(_normalize_phrase(" ".join(words)))

    camera_from = view.get("camera_from")
    if camera_from:
        label = _normalize_phrase(camera_from)
        phrases.append(label)
        for part_key, part_info in parts.items():
            if not isinstance(part_info, dict):
                continue
            if label and label in _normalize_phrase(part_info.get("location", "")):
                phrases.append(_normalize_phrase(part_key))
                for synonym in part_info.get("also_called", []):
                    phrases.append(_normalize_phrase(synonym))

    return [p for p in phrases if p]


def resolve_target_view(data: dict, target_description: str) -> Optional[str]:
    """Map a free-text target description to one canonical view name, if possible.

    Deterministic, two-pass matching against data["canonical_views"] — every keyword
    is derived from the JSON spec itself (view names, camera_from labels, main_parts
    names/synonyms); nothing is hardcoded for any particular object or domain.
      1. An exact canonical view name appearing anywhere in the description wins
         outright (case-insensitive), e.g. "...render a FrontTopView view".
      2. Otherwise, match against phrases derived from the view's own name, its
         camera_from label, and any main_parts located on that side/end.

    Returns None if nothing matches — callers should then ask the model to pick
    one of the canonical views itself instead of assuming a target view.
    """
    views = data.get("canonical_views", {})
    if not views or not target_description:
        return None

    text = _normalize_phrase(target_description)

    for view_name in views:
        if view_name.lower() in text:
            return view_name

    shared_words = _shared_camel_words(views)
    for view_name, view in views.items():
        phrases = _derive_view_phrases(data, view_name, view, shared_words)
        if any(phrase in text for phrase in phrases):
            return view_name

    return None


def _append_list_section(lines: List[str], title: str, values: Any, indent: str = "  ") -> None:
    """Append a compact text section from a list or string value."""
    if not values:
        return
    if isinstance(values, str):
        lines.append(f"{title}: {values}")
        return
    if isinstance(values, list):
        lines.append(f"{title}:")
        for value in values:
            lines.append(f"{indent}- {value}")


def _append_optional_view_guidance(lines: List[str], view: Optional[dict]) -> None:
    """Render any optional per-view guidance fields the JSON itself provides
    (decision_rules, negative_rules, common_mistakes, counterexamples). Purely
    data-driven — only emits something if the loaded spec actually has it.
    """
    if not view:
        return
    _append_list_section(lines, "JSON decision_rules", view.get("decision_rules"))
    _append_list_section(lines, "JSON negative_rules", view.get("negative_rules"))
    _append_list_section(lines, "JSON common_mistakes", view.get("common_mistakes"))
    _append_list_section(lines, "JSON counterexamples", view.get("counterexamples"))


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
        _append_optional_view_guidance(lines, view)
    elif views:
        lines.append(
            "Target view could not be resolved automatically from the target description. "
            "Choose exactly one of the following canonical views yourself:"
        )
        for name, view in views.items():
            lines.append(f"  - {name}: {view.get('meaning', '')} camera_from={view.get('camera_from', '?')}.")

    notes = data.get("image_space_notes", {}).get("examples", [])
    if notes:
        lines.append("Image-space vs. anatomical-view notes: " + " ".join(notes))

    rules = data.get("minimal_reasoning_rules", [])
    if rules:
        lines.append(
            "Reasoning rules from JSON: "
            + " ".join(f"{r.get('rule')} — {r.get('instruction')}" for r in rules)
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


_AXIS_STRING_TO_VECTOR = {
    "+x": (1.0, 0.0, 0.0),
    "x+": (1.0, 0.0, 0.0),
    "x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "x-": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "y+": (0.0, 1.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "y-": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "z+": (0.0, 0.0, 1.0),
    "z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
    "z-": (0.0, 0.0, -1.0),
}


def _normalize_vector(vec: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
    norm = math.sqrt(sum(float(v) * float(v) for v in vec))
    if norm <= 1e-12:
        return None
    return tuple(float(v) / norm for v in vec)


def _dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _parse_axis_vector(value: Any) -> Optional[Tuple[float, float, float]]:
    """Accept [x,y,z], '+x', '-z', or {'axis': 'z', 'sign': -1}."""
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return _normalize_vector((float(value[0]), float(value[1]), float(value[2])))
        except (TypeError, ValueError):
            return None

    if isinstance(value, str):
        key = value.strip().lower().replace(" ", "")
        return _AXIS_STRING_TO_VECTOR.get(key)

    if isinstance(value, dict):
        axis = str(value.get("axis", "")).lower().strip()
        sign = float(value.get("sign", 1.0))
        if axis in {"x", "y", "z"}:
            base = _AXIS_STRING_TO_VECTOR[axis]
            return tuple(v if sign >= 0 else -v for v in base)

    return None


def _load_anatomical_axis_map(data: Optional[dict]) -> Dict[str, Tuple[float, float, float]]:
    """Load optional object-space axis mapping from the JSON spec.

    Supported JSON shapes (whatever labels the spec defines — not hardcoded):
      "object_axes": {"front": "+z", "back": "-z", ...}
      "axis_mapping": {"top": [0,0,1], ...}
      "anatomical_axes": {"left": {"axis":"z", "sign":1}, ...}

    If absent, returns {} and the pipeline simply skips deterministic camera hints.
    """
    if not data:
        return {}
    raw_axes = data.get("object_axes") or data.get("axis_mapping") or data.get("anatomical_axes") or {}
    if not isinstance(raw_axes, dict):
        return {}

    axes: Dict[str, Tuple[float, float, float]] = {}
    for label, raw_value in raw_axes.items():
        vec = _parse_axis_vector(raw_value)
        if vec is not None:
            axes[str(label).lower()] = vec
    return axes


def build_camera_view_hint(data: Optional[dict], camera_state: Optional[dict]) -> Optional[str]:
    """Return a deterministic object-axis hint from camera position - focal point.

    This does NOT replace visual diagnosis. It gives the LLM an anchor when the JSON spec
    provides object_axes/axis_mapping/anatomical_axes, so the model does not rely only on
    visual guessing. If no axis map exists, returns None.
    """
    axes = _load_anatomical_axis_map(data)
    if not axes or not camera_state:
        return None

    try:
        pos = tuple(float(v) for v in camera_state["position"])
        fp = tuple(float(v) for v in camera_state["focal_point"])
    except (KeyError, TypeError, ValueError):
        return None

    camera_from = _normalize_vector(tuple(p - f for p, f in zip(pos, fp)))
    if camera_from is None:
        return None

    scores = sorted(
        ((label, _dot(camera_from, axis_vec)) for label, axis_vec in axes.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    if not scores:
        return None

    # Which canonical view (if any) has this label as its camera_from — derived from
    # the loaded JSON, not hardcoded.
    label_to_view = {
        str(view.get("camera_from", "")).lower(): name
        for name, view in (data or {}).get("canonical_views", {}).items()
        if view.get("camera_from")
    }

    primary_label, primary_score = scores[0]
    secondary = [f"{label}={score:.3f}" for label, score in scores[1:3]]
    primary_view = label_to_view.get(primary_label, primary_label)
    return (
        "Deterministic camera-side estimate from current camera vector: "
        f"primary={primary_label} ({primary_view}), score={primary_score:.3f}; "
        f"next_best={', '.join(secondary)}. "
        "Use this as a weak anchor, but still verify against the screenshot."
    )


STRUCTURED_FIELDS = [
    "target_view",
    "current_view_inclination",
    "current_view_estimate",
    "key_mismatches",
    "likely_issue_type",
    "recommended_update",
    "minimal_change_rationale",
]


def extract_structured_fields(response: str) -> dict:
    """Best-effort extraction of single-line "field: value" fields (e.g.
    current_view_inclination, the only one the current prompt actually asks for —
    the others are legacy/optional) from a response, for logging only.

    Fields that aren't present are simply omitted — this never raises and never
    blocks the main action-parsing flow.
    """
    fields = {}
    for name in STRUCTURED_FIELDS:
        match = re.search(rf"{name}\s*:\s*(.+)", response, re.IGNORECASE)
        if match:
            fields[name] = match.group(1).strip()
    return fields


# Section headers recognized in the free-form diagnosis response, in the order the
# prompt asks for them. Used by extract_diagnosis_sections() to know where each
# section ends (at the start of the next known header, or end of string).
DIAGNOSIS_SECTION_HEADERS = [
    "Visual observation",
    "Visual diagnosis",
    "Camera position inference",
    "current_view_inclination",
    "Reasoning",
    "Next action",
    "Expected visual change",
]

# "Visual observation" (spatial_context prompt format) and "Visual diagnosis" (the
# older/no-spatial_context format) serve the same role — whichever the model was
# actually asked for, both are stored under the same "visual_observation" key so
# callers don't need to know which prompt format produced the response.
_VISUAL_OBSERVATION_ALIASES = ("Visual observation", "Visual diagnosis")


def extract_diagnosis_sections(response: str) -> dict:
    """Best-effort extraction of the free-form "Visual observation"/"Visual
    diagnosis" and "Camera position inference" sections from a response, for
    logging only.

    No forced JSON schema here — the model just writes free text under each labeled
    header. Each section runs until the next recognized header or end of string.
    Missing sections are simply omitted; this never raises and never blocks the
    action-parsing flow.
    """
    header_pattern = "|".join(re.escape(h) for h in DIAGNOSIS_SECTION_HEADERS)
    sections = {}

    for name in _VISUAL_OBSERVATION_ALIASES:
        match = re.search(
            rf"{re.escape(name)}\s*:\s*\n?(.*?)(?=\n\s*(?:{header_pattern})\s*:|\Z)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            text = match.group(1).strip()
            if text:
                sections["visual_observation"] = text
                break

    match = re.search(
        rf"Camera position inference\s*:\s*\n?(.*?)(?=\n\s*(?:{header_pattern})\s*:|\Z)",
        response,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        text = match.group(1).strip()
        if text:
            sections["camera_position_inference"] = text

    return sections
