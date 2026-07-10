"""
Frozen-reference-embedding alignment: select the most visually similar reference
view ONCE at the start (via image embeddings), freeze it as the alignment target,
then run the normal iterative camera-reasoning loop always comparing against that
same frozen target — never re-retrieving a new nearest view mid-loop.

This module is purely additive. It reuses, unmodified:
  - image_view_matcher.ImageViewMatcher       (existing image embedding index/retrieval)
  - text_view_matcher.TextViewMatcher         (existing text embedding index/retrieval)
  - reference_views/reference_views_metadata.json (existing camera metadata)
  - reference_views/view_descriptions.json    (existing per-view descriptions)
  - chatgpt_client.ask_chatgpt                (existing AI/vision model interface)
  - camera_actions.VALID_ACTIONS/ACTION_DESCRIPTIONS (existing camera action schema)
  - camera_state.get_camera_state             (existing camera metadata reader)
  - CameraReasoningSession.process_chatgpt_response (existing action-application path —
    fed a synthetic "Next action:\\n<ACTION>" response, so screenshot/camera-state/
    history bookkeeping all go through the exact same code as the normal loop)

session.py, prompt_writer.py, spatial_knowledge.py, action_parser.py, and
reference_views.py are all untouched.
"""
import json
from pathlib import Path
from typing import Optional

from .camera_actions import ACTION_DESCRIPTIONS, VALID_ACTIONS
from .camera_state import get_camera_state
from .chatgpt_client import ask_chatgpt
from .image_view_matcher import ImageViewMatcher
from .text_view_matcher import TextViewMatcher
from .view_description_generator import _extract_json_object  # reuse existing JSON-extraction logic

DEFAULT_REFERENCES_DIR = "reference_views"
DEFAULT_CACHE_PATH = "reference_embeddings.pkl"
DEFAULT_TEXT_CACHE_PATH = "reference_text_embeddings.pkl"

ANALYSIS_PROMPT_TEMPLATE = """You are analyzing a rendered image of foot bones.

The current camera is trying to align with this fixed reference target:

Fixed reference target view:
{alignment_target_view_name}

Fixed reference target description:
{alignment_target_description}

Known reference views (view_descriptions.json, verbatim — for grounding anatomical
vocabulary ONLY; the current screenshot is not necessarily any of these views; use
this only to understand what each label typically looks like, so you don't misuse
a label like "lateral" for a view that doesn't actually match it):
```json
{all_view_descriptions_json}
```

Your task:
1. Describe the current screenshot.
2. Compare the current screenshot against the fixed reference target.
3. Identify what needs to change visually/spatially so the current screenshot becomes closer to the fixed reference target.

Return only valid JSON with this schema:

{{
  "current_description": {{
    "short_description": "...",
    "primary_orientation": "...",
    "visible_surfaces": ["..."]
  }},
  "fixed_reference_view_name": "...",
  "difference_from_fixed_reference": {{
    "summary": "...",
    "shared_surfaces": ["..."],
    "extra_or_stronger_surfaces_in_current": ["..."],
    "missing_or_weaker_surfaces_in_current": ["..."],
    "orientation_shift": "...",
    "alignment_quality": "poor | partial | close | aligned",
    "confidence": 0.0
  }}
}}

Definitions:
- dorsal = top side of the foot
- plantar = bottom or sole side of the foot
- medial = big-toe side of the foot
- lateral = little-toe side of the foot
- distal = toe-end direction
- proximal = ankle or heel-end direction

Rules:
- The fixed reference target does not change during the loop.
- Use the fixed reference description as context, but describe the current screenshot based on the actual image.
- Do not assume the current screenshot is exactly the same as the fixed reference.
- If the current screenshot looks almost identical to the fixed reference, say the difference is small and set alignment_quality to "aligned".
- Use the "Known reference views" list only to calibrate what each anatomical label looks like. Do not label the current screenshot as one of those views unless it actually matches — judge from the actual pixels, not from assuming a label.
- primary_orientation should be lowercase.
- For oblique views, use hyphenated labels such as dorsal-medial or plantar-proximal.
- visible_surfaces should be lowercase anatomical labels.
- Do not output camera movement yet.
- Return JSON only.
"""

REASONING_PROMPT_TEMPLATE = """You are controlling a VTK camera for a rendered foot-bone object.

The current view is being aligned toward a fixed reference target.

Fixed reference target:
{alignment_target_reference}

Current view analysis:
{current_view_analysis}

Current camera metadata:
{current_camera_metadata}

Allowed camera actions (this is the existing camera action schema used by this
project — action_type MUST be exactly one of these names, do not invent new ones):
{action_list}

Decide the next camera movement that would make the current screenshot closer to the fixed reference target.

Return only valid JSON using this schema:

{{
  "reasoning": "...",
  "current_orientation_summary": "...",
  "reference_orientation_summary": "...",
  "needed_change": "...",
  "camera_action": {{
    "action_type": "...",
    "parameters": {{}}
  }},
  "confidence": 0.0
}}

Important rules:
- action_type must be exactly one of the allowed camera actions listed above.
- The reasoning should explicitly explain how the movement reduces the difference between the current screenshot and the fixed reference target.
- The fixed reference target must remain the same across all iterations.
- Do not retrieve a new nearest reference view inside the loop.
- If the current view is already aligned with the fixed reference target, choose STOP.
- Prefer small incremental camera movements (the *_FINE or *_MEDIUM actions) unless a larger movement is clearly needed.
- Return JSON only.
"""


def _fallback_description(view_name: str) -> dict:
    """Minimal description used when no generated description is available for a
    view — keeps the pipeline running instead of crashing on missing data.
    """
    return {
        "view_name": view_name,
        "short_description": f"Reference view named {view_name!r}; no generated description available.",
        "primary_orientation": view_name.lower(),
        "visible_surfaces": [],
    }


def _safe_stop_action(reason: str) -> dict:
    """Safe no-op result used whenever the model output can't be trusted — reuses
    the existing STOP action rather than inventing a new one.
    """
    return {
        "reasoning": f"Falling back to STOP: {reason}.",
        "current_orientation_summary": "",
        "reference_orientation_summary": "",
        "needed_change": "",
        "camera_action": {"action_type": "STOP", "parameters": {}},
        "confidence": 0.0,
        "error": reason,
    }


# ------------------------------------------------------------------
# Step 1 (called ONCE): retrieve + freeze the alignment target
# ------------------------------------------------------------------

def initialize_alignment_target(
    initial_image_path: str,
    references_dir: str = DEFAULT_REFERENCES_DIR,
    metadata_path: Optional[str] = None,
    descriptions_path: Optional[str] = None,
    cache_path: Optional[str] = DEFAULT_CACHE_PATH,
    model: Optional[str] = None,
) -> dict:
    """Compute the initial image's embedding, retrieve the single most similar
    reference view, load its existing metadata + description, and return the
    frozen alignment target dict. Call this exactly once, before the loop.
    """
    print(f"[frozen_reference_alignment] Initial image path: {initial_image_path}")

    references_dir_path = Path(references_dir)
    metadata_file = Path(metadata_path) if metadata_path else references_dir_path / "reference_views_metadata.json"
    descriptions_file = Path(descriptions_path) if descriptions_path else references_dir_path / "view_descriptions.json"

    if not references_dir_path.exists():
        raise FileNotFoundError(
            f"[frozen_reference_alignment] ERROR: reference embedding database not found: "
            f"{references_dir_path}. Generate reference views first "
            f"(see camera_reasoning/reference_views.py) before using frozen-reference alignment."
        )

    # --- Retrieval (existing embedding index/matcher, reused unmodified) ---
    matcher = ImageViewMatcher(reference_dir=str(references_dir_path), cache_path=cache_path)
    matcher.build_index()
    match_result = matcher.match(initial_image_path, top_k=1)

    best_view = match_result["best_view"]
    similarity = match_result["view_scores"][0]["max_similarity"] if match_result["view_scores"] else None
    reference_image_path = next(
        (m["reference_image"] for m in match_result["top_matches"] if m["view_label"] == best_view), None
    )

    # --- Existing camera metadata for that view ---
    metadata_entry = {}
    if metadata_file.exists():
        all_metadata = json.loads(metadata_file.read_text())
        metadata_entry = next((m for m in all_metadata if m.get("view_name") == best_view), {}) or {}
        if not reference_image_path:
            reference_image_path = metadata_entry.get("image_path")
    else:
        print(
            f"[frozen_reference_alignment] WARNING: reference metadata file not found at "
            f"{metadata_file}; alignment target will have no camera metadata."
        )

    # --- Existing view_descriptions.json, dumped verbatim (same philosophy as
    # dump_spatial_knowledge_json: no filtering/reformatting) for vocabulary
    # grounding — loaded once here, frozen alongside the target itself ---
    description = None
    all_view_descriptions = []
    if descriptions_file.exists():
        all_view_descriptions = json.loads(descriptions_file.read_text())
        description = next((d for d in all_view_descriptions if d.get("view_name") == best_view), None)
    else:
        print(
            f"[frozen_reference_alignment] ERROR: view description file not found at "
            f"{descriptions_file}. Falling back to minimal per-view descriptions "
            "(see camera_reasoning/view_description_generator.py to generate real ones)."
        )

    if description is None:
        print(
            f"[frozen_reference_alignment] WARNING: no description found for {best_view!r}; "
            "using a fallback minimal description."
        )
        description = _fallback_description(best_view)

    alignment_target_reference = {
        "view_name": best_view,
        "similarity": similarity,
        "image_path": reference_image_path,
        "metadata": metadata_entry,
        "description": description,
        "all_view_descriptions": all_view_descriptions,
    }

    print(f"[frozen_reference_alignment] Frozen alignment target view: {best_view}")
    print(f"[frozen_reference_alignment] Initial similarity: {similarity}")
    print(f"[frozen_reference_alignment] Frozen reference image path: {reference_image_path}")
    print(f"[frozen_reference_alignment] Frozen reference description: {description}")
    print(f"[frozen_reference_alignment] Loaded {len(all_view_descriptions)} view descriptions (view_descriptions.json) for grounding.")

    return alignment_target_reference


# ------------------------------------------------------------------
# Step 2 (called every iteration): analyze current vs. frozen target
# ------------------------------------------------------------------

def analyze_current_view_against_frozen_reference(
    current_image_path: str,
    alignment_target_reference: dict,
    model: Optional[str] = None,
) -> dict:
    """Ask the existing AI/vision model to describe the current screenshot and
    compare it against the frozen alignment_target_reference. Never re-retrieves
    a reference view — the target passed in is used exactly as given.
    """
    description = alignment_target_reference.get("description") or _fallback_description(
        alignment_target_reference.get("view_name", "unknown")
    )
    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        alignment_target_view_name=alignment_target_reference.get("view_name", "unknown"),
        alignment_target_description=json.dumps(description, indent=2),
        all_view_descriptions_json=json.dumps(alignment_target_reference.get("all_view_descriptions", []), indent=2),
    )

    response_text = ask_chatgpt(
        prompt=prompt,
        screenshot_path=current_image_path,
        target_image_path=alignment_target_reference.get("image_path"),
        model=model,
    )

    parsed = _extract_json_object(response_text)
    if parsed is None:
        print("[frozen_reference_alignment] WARNING: analysis response was not valid JSON; storing raw response.")
        return {
            "current_image_path": current_image_path,
            "error": "invalid_json",
            "raw_response": response_text,
        }

    # Bookkeeping field only — not part of the model's JSON schema — so the next
    # step can still show the model the actual current screenshot.
    parsed["current_image_path"] = current_image_path
    return parsed


# ------------------------------------------------------------------
# Step 3 (called every iteration): decide the camera action
# ------------------------------------------------------------------

def reason_camera_action_to_match_frozen_reference(
    current_view_analysis: dict,
    alignment_target_reference: dict,
    current_camera_metadata: dict,
    model: Optional[str] = None,
) -> dict:
    """Ask the existing AI/model interface to pick exactly one action from the
    existing camera action schema (VALID_ACTIONS) that moves the camera closer to
    the frozen alignment_target_reference.
    """
    if current_view_analysis.get("error"):
        print(
            "[frozen_reference_alignment] WARNING: current_view_analysis had an error "
            "(no usable current-view analysis); returning safe STOP action."
        )
        return _safe_stop_action(reason="upstream analysis failed to parse")

    # Exclude all_view_descriptions here — that's grounding context for the
    # analysis step only; keep this prompt scoped to the target + this iteration.
    target_for_prompt = {k: v for k, v in alignment_target_reference.items() if k != "all_view_descriptions"}

    action_list = "\n".join(f"  - {name}: {ACTION_DESCRIPTIONS[name]}" for name in sorted(VALID_ACTIONS))
    prompt = REASONING_PROMPT_TEMPLATE.format(
        alignment_target_reference=json.dumps(target_for_prompt, indent=2, default=str),
        current_view_analysis=json.dumps(current_view_analysis, indent=2, default=str),
        current_camera_metadata=json.dumps(current_camera_metadata, indent=2, default=str),
        action_list=action_list,
    )

    current_image_path = current_view_analysis.get("current_image_path")
    response_text = ask_chatgpt(
        prompt=prompt,
        screenshot_path=current_image_path or alignment_target_reference.get("image_path"),
        target_image_path=alignment_target_reference.get("image_path"),
        model=model,
    )

    parsed = _extract_json_object(response_text)
    if parsed is None:
        print(
            "[frozen_reference_alignment] WARNING: camera reasoning response was not valid JSON; "
            "returning safe STOP action."
        )
        result = _safe_stop_action(reason="model output was not valid JSON")
        result["raw_response"] = response_text
        return result

    action_type = (parsed.get("camera_action") or {}).get("action_type")
    if action_type not in VALID_ACTIONS:
        print(
            f"[frozen_reference_alignment] WARNING: model returned unsupported action_type "
            f"{action_type!r}; falling back to STOP."
        )
        parsed["camera_action"] = {"action_type": "STOP", "parameters": {}}

    return parsed


# ------------------------------------------------------------------
# Verification (called every iteration): cross-check the LLM's own observation
# against two independent embedding-based rankings. Purely diagnostic — never
# blocks or overrides the camera-reasoning step.
# ------------------------------------------------------------------

def verify_observation_against_embeddings(
    current_image_path: str,
    current_description_text: str,
    image_matcher: ImageViewMatcher,
    text_matcher: TextViewMatcher,
    top_k: int = 5,
) -> dict:
    """Cross-check the LLM's own visual-observation text (from
    analyze_current_view_against_frozen_reference()'s current_description) against
    two independent embedding-based top-k rankings over the reference-view database:
      1. image embedding of the actual current screenshot (image_matcher)
      2. text embedding of the LLM's own description (text_matcher)

    Both matchers are passed in already-built (see run_frozen_reference_alignment_loop)
    so the reference database is only encoded once per run, not once per iteration —
    only the query (current screenshot / current description) is re-encoded each time.

    This does not select or freeze anything — it's a diagnostic sanity check, logged
    and saved in the trace, so you can spot cases where the LLM's own words disagree
    with what the embeddings say about the same screenshot.
    """
    image_result = image_matcher.match(current_image_path, top_k=top_k)
    text_result = text_matcher.match(current_description_text, top_k=top_k)

    image_top_views = [m["view_label"] for m in image_result["top_matches"]]
    text_top_views = [m["view_label"] for m in text_result["top_matches"]]

    return {
        "image_top_k": image_result["top_matches"],
        "text_top_k": text_result["top_matches"],
        "image_best_view": image_result["best_view"],
        "text_best_view": text_result["best_view"],
        "top1_agree": image_result["best_view"] == text_result["best_view"],
        "overlap_count": len(set(image_top_views) & set(text_top_views)),
    }


# ------------------------------------------------------------------
# Trace saving
# ------------------------------------------------------------------

def save_iteration_trace(
    trace_dir: str,
    iteration: int,
    current_image_path: str,
    alignment_target_reference: dict,
    current_view_analysis: dict,
    camera_reasoning_output: dict,
    applied_action: Optional[str],
    observation_verification: Optional[dict] = None,
) -> str:
    trace = {
        "iteration": iteration,
        "current_image_path": current_image_path,
        "frozen_alignment_target": {
            "view_name": alignment_target_reference.get("view_name"),
            "initial_similarity": alignment_target_reference.get("similarity"),
            "image_path": alignment_target_reference.get("image_path"),
            "description": alignment_target_reference.get("description"),
        },
        "current_view_analysis": {
            "current_description": current_view_analysis.get("current_description"),
            "difference_from_fixed_reference": current_view_analysis.get("difference_from_fixed_reference"),
        },
        "observation_verification": observation_verification,
        "camera_reasoning": camera_reasoning_output,
        "camera_action": applied_action,
    }
    out_dir = Path(trace_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / f"trace_{iteration:03d}.json"
    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2, default=str)
    return str(trace_path)


# ------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------

def run_frozen_reference_alignment_loop(
    session,
    num_iterations: int = 5,
    references_dir: str = DEFAULT_REFERENCES_DIR,
    trace_dir: Optional[str] = None,
    model: Optional[str] = None,
    dry_run: bool = False,
    verify_observations: bool = True,
    verify_top_k: int = 5,
) -> dict:
    """Run the frozen-reference-embedding alignment loop against an already-
    `.initialize()`d CameraReasoningSession.

    initialize_alignment_target() is called exactly once, before the loop. Every
    iteration reuses the SAME alignment_target_reference — it is never re-retrieved.
    Camera actions are applied via session.process_chatgpt_response() (the existing
    action-application path), fed a synthetic "Next action:\\n<ACTION>" string, so
    screenshot/camera-state/history bookkeeping all reuse the existing code exactly.

    If verify_observations is True (default), every iteration also cross-checks the
    LLM's own "current_description" against top-verify_top_k image-embedding and
    text-embedding rankings over the reference database (verify_observation_against_
    embeddings()) — purely diagnostic, logged and saved to the trace, never gates
    the camera action.
    """
    initial_image_path = session.render_and_save()
    alignment_target_reference = initialize_alignment_target(
        initial_image_path, references_dir=references_dir, model=model
    )

    # Built once, reused every iteration — only the query (current screenshot /
    # current description) is re-encoded per iteration, not the reference database.
    image_matcher = None
    text_matcher = None
    if verify_observations:
        image_matcher = ImageViewMatcher(reference_dir=references_dir, cache_path=DEFAULT_CACHE_PATH)
        image_matcher.build_index()
        text_matcher = TextViewMatcher(
            descriptions_path=str(Path(references_dir) / "view_descriptions.json"),
            cache_path=DEFAULT_TEXT_CACHE_PATH,
        )
        text_matcher.build_index()

    if trace_dir is None:
        trace_dir = str(Path(session.output_dir) / "traces")

    applied_actions = []
    iteration_count = 1 if dry_run else num_iterations

    for i in range(iteration_count):
        current_image_path = str(Path(session.output_dir) / "screenshots" / "latest.png")

        # flush=True on every print in this loop: Jupyter/ipykernel can otherwise
        # buffer stdout and only flush it to the cell output once the whole loop
        # finishes, instead of streaming each iteration's output as it happens.
        print(f"\n[frozen_reference_alignment] Iteration: {i}", flush=True)
        print(f"[frozen_reference_alignment] Current image path: {current_image_path}", flush=True)
        print(f"[frozen_reference_alignment] Frozen alignment target view: {alignment_target_reference['view_name']}", flush=True)

        current_view_analysis = analyze_current_view_against_frozen_reference(
            current_image_path, alignment_target_reference, model=model
        )
        print(f"[frozen_reference_alignment] Generated current description: {current_view_analysis.get('current_description')}", flush=True)
        print(f"[frozen_reference_alignment] Difference from frozen reference: {current_view_analysis.get('difference_from_fixed_reference')}", flush=True)

        observation_verification = None
        if verify_observations:
            observation_text = (current_view_analysis.get("current_description") or {}).get("short_description")
            if observation_text:
                observation_verification = verify_observation_against_embeddings(
                    current_image_path, observation_text, image_matcher, text_matcher, top_k=verify_top_k
                )
                print("[frozen_reference_alignment] Verification — top-3 similar image views:", flush=True)
                for m in observation_verification["image_top_k"][:3]:
                    print(f"    {m['rank']}. {m['view_label']:<20s} similarity={m['similarity']:.4f}", flush=True)
                print("[frozen_reference_alignment] Verification — top-3 similar description views:", flush=True)
                for m in observation_verification["text_top_k"][:3]:
                    print(f"    {m['rank']}. {m['view_label']:<20s} similarity={m['similarity']:.4f}", flush=True)
                print(
                    f"[frozen_reference_alignment] Verification — image_best={observation_verification['image_best_view']} "
                    f"text_best={observation_verification['text_best_view']} "
                    f"agree={observation_verification['top1_agree']} "
                    f"overlap={observation_verification['overlap_count']}/{verify_top_k}",
                    flush=True,
                )
            else:
                print("[frozen_reference_alignment] Verification skipped — no observation text available.", flush=True)

        # Existing camera_state.py reader — session's own VTK camera, untouched.
        current_camera_metadata = get_camera_state(session._renderer.GetActiveCamera())

        camera_reasoning_output = reason_camera_action_to_match_frozen_reference(
            current_view_analysis, alignment_target_reference, current_camera_metadata, model=model
        )
        print(f"[frozen_reference_alignment] Camera reasoning output: {camera_reasoning_output}", flush=True)

        action_type = (camera_reasoning_output.get("camera_action") or {}).get("action_type", "STOP")
        print(f"[frozen_reference_alignment] Camera action: {action_type}", flush=True)

        applied_action = None
        if dry_run:
            print("[frozen_reference_alignment] Dry run — camera action NOT applied.", flush=True)
        else:
            # Reuse the existing action-application path exactly (action_parser +
            # apply_action + screenshot/camera-state/history bookkeeping) by handing
            # it a synthetic response in the same format the normal loop produces.
            applied_action = session.process_chatgpt_response(f"Next action:\n{action_type}")
            applied_actions.append(applied_action)

        if trace_dir:
            trace_path = save_iteration_trace(
                trace_dir, i, current_image_path, alignment_target_reference,
                current_view_analysis, camera_reasoning_output, applied_action,
                observation_verification=observation_verification,
            )
            print(f"[frozen_reference_alignment] Saved trace: {trace_path}", flush=True)

        if not dry_run and applied_action == "STOP":
            break

    return {
        "alignment_target_reference": alignment_target_reference,
        "applied_actions": applied_actions,
        "dry_run": dry_run,
    }


def run_camera_reasoning_pipeline(
    session,
    num_iterations: int = 5,
    use_frozen_reference_alignment: bool = False,
    references_dir: str = DEFAULT_REFERENCES_DIR,
    trace_dir: Optional[str] = None,
    model: Optional[str] = None,
    dry_run: bool = False,
    verify_observations: bool = True,
    verify_top_k: int = 5,
) -> dict:
    """Single entry point that switches between the new frozen-reference-embedding
    alignment loop and the existing (completely unmodified) text-based reasoning
    loop, controlled by `use_frozen_reference_alignment`.

    use_frozen_reference_alignment=False (default): preserves old behavior exactly
    — delegates to session.ask_chatgpt_and_process(), unchanged.
    """
    if not use_frozen_reference_alignment:
        actions = []
        for _ in range(num_iterations):
            action = session.ask_chatgpt_and_process(model=model)
            actions.append(action)
            if action == "STOP":
                break
        return {"mode": "text_based", "applied_actions": actions}

    result = run_frozen_reference_alignment_loop(
        session,
        num_iterations=num_iterations,
        references_dir=references_dir,
        trace_dir=trace_dir,
        model=model,
        dry_run=dry_run,
        verify_observations=verify_observations,
        verify_top_k=verify_top_k,
    )
    result["mode"] = "frozen_reference_alignment"
    return result
