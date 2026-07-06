from pathlib import Path
from typing import Dict, List, Optional

from .camera_actions import ACTION_DESCRIPTIONS, VALID_ACTIONS


def write_llm_prompt(
    output_dir: str,
    camera_state: dict,
    action_history: List[Dict],
    target_description: str,
    screenshot_path: str,
    target_image_path: Optional[str] = None,
    spatial_context: Optional[str] = None,
) -> str:
    lines = []

    lines += [
        "You are ChatGPT, acting as a camera-reasoning assistant for a 3D VTK scene.",
        "",
        "Your job is to compare the current render with the target view description and",
        "choose exactly one next camera action.",
        "",
        "Rules:",
        "- You are NOT allowed to invent raw camera coordinates.",
        "- You are NOT allowed to output arbitrary angles.",
        "- You are NOT allowed to generate VTK code.",
        "- You MUST choose exactly one action from the allowed action list below.",
        "",
    ]

    if spatial_context:
        lines += [
            "Spatial knowledge (target-specific — source of truth for view reasoning,",
            "takes priority over guesswork):",
            spatial_context,
            "",
            "Important: anatomical view and image orientation are NOT the same thing.",
            "\"Toes at the bottom/top/left/right\" is image-space information (screen roll/",
            "orientation); it does not by itself prove the anatomical view. The anatomical",
            "view depends on which side the camera is actually on (dorsal/plantar/medial/",
            "lateral/distal/proximal), not on where things land on screen.",
            "",
            "You must answer in exactly this order:",
            "  1. A SPATIAL_DIAGNOSIS JSON block (required format shown below).",
            "  2. A brief Reasoning note, only if it adds something the JSON doesn't already say.",
            "  3. The final action line, in the existing action format.",
            "",
            "Before answering, work through these steps:",
            "  1. Observe the CURRENT screenshot only — record what you actually see first,",
            "     before deciding whether it matches the target.",
            "  2. Compare those observations against the target view contract's should_see",
            "     and should_not_dominate lists above.",
            "  3. Classify the issue as exactly one of:",
            "       correct_view, wrong_opposite_view, too_oblique, roll_only,",
            "       wrong_surface, wrong_end_view, unclear_low_confidence",
            "  4. Choose the smallest camera/view update that fixes it.",
            "  5. Do not change rendering style or isovalue. Do not invent anatomy that isn't",
            "     in the spatial knowledge above.",
            "",
        ]

    lines += [
        "Target view:",
        target_description,
        "",
    ]

    lines += [
        "Current render screenshot:",
        str(screenshot_path),
    ]
    if target_image_path and Path(target_image_path).exists():
        lines += [f"Target image: {target_image_path}"]
    lines.append("")

    pos = camera_state["position"]
    fp  = camera_state["focal_point"]
    up  = camera_state["view_up"]
    va  = camera_state["view_angle"]
    cr  = camera_state["clipping_range"]
    import numpy as np
    dist = float(np.linalg.norm(np.array(pos) - np.array(fp)))

    lines += [
        "Current camera:",
        f"  position:       [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]",
        f"  focal_point:    [{fp[0]:.3f}, {fp[1]:.3f}, {fp[2]:.3f}]",
        f"  view_up:        [{up[0]:.3f}, {up[1]:.3f}, {up[2]:.3f}]",
        f"  view_angle:     {va:.2f}",
        f"  clipping_range: [{cr[0]:.3f}, {cr[1]:.3f}]",
        f"  distance:       {dist:.3f}",
        "",
    ]

    if action_history:
        lines.append("Action history:")
        for i, entry in enumerate(action_history, 1):
            lines.append(f"  {i}. {entry['action']}")
        lines.append("")
    else:
        lines += ["Action history:", "  (none yet)", ""]

    lines.append("Allowed actions:")
    for name in sorted(VALID_ACTIONS):
        lines.append(f"  - {name}: {ACTION_DESCRIPTIONS[name]}")
    lines.append("")

    lines += ["Return exactly this format:", ""]

    if spatial_context:
        # Require a single fenced JSON block (parsed by extract_spatial_diagnosis()) that
        # forces the model to separate raw observation, contract check, and diagnosis
        # BEFORE it ever gets to pick an action — instead of a generic free-text diagnosis
        # that gets rationalized into a canonical view after the fact.
        lines += [
            "SPATIAL_DIAGNOSIS:",
            "```json",
            "{",
            '  "target_view": "DorsalTopDown",',
            '  "raw_visual_observation": {',
            '    "visible_surface": "mostly dorsal",',
            '    "visible_parts": ["toes", "metatarsals", "ankle_region"],',
            '    "dominant_side": "neither",',
            '    "closest_region": "unclear",',
            '    "side_depth": "moderate",',
            '    "image_roll": "toes appear bottom",',
            '    "confidence": 0.72',
            "  },",
            '  "target_contract_check": {',
            '    "should_see_present": ["toes", "metatarsals", "ankle_region"],',
            '    "should_see_missing": [],',
            '    "should_not_dominate_violations": ["strong side depth"],',
            '    "evidence_for_target": ["dorsal surface appears dominant"],',
            '    "evidence_against_target": ["side depth is still moderate"]',
            "  },",
            '  "diagnosis": {',
            '    "current_view_estimate": "close to dorsal top-down but still oblique",',
            '    "likely_issue_type": "too_oblique"',
            "  },",
            '  "minimal_update": {',
            '    "recommended_update": "reduce side perspective and move closer to true dorsal top-down",',
            '    "minimal_change_rationale": "the target surface is mostly correct, but side depth is too strong"',
            "  }",
            "}",
            "```",
            "(values above are an example only — replace them with your actual observations)",
            "",
            "Reasoning (brief, optional — skip if the JSON above already says it):",
            "...",
        ]
    else:
        lines += [
            "Visual diagnosis:",
            "...",
            "",
            "Reasoning:",
            "...",
        ]

    lines += [
        "",
        "Next action:",
        "ACTION_NAME",
        "",
        "Expected visual change:",
        "...",
    ]

    prompt = "\n".join(lines)
    out_path = Path(output_dir) / "llm_prompt.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(prompt)
    return prompt
