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
    camera_view_hint: Optional[str] = None,
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
            "Spatial knowledge (full JSON spec, verbatim — source of truth for view",
            "reasoning, takes priority over guesswork). Read canonical_views below and",
            "pick the one matching the target description; use its should_see/",
            "should_not_dominate fields as your diagnostic checklist.",
            "```json",
            spatial_context,
            "```",
            "",
            "Important: anatomical/spatial view and image orientation are NOT the same thing.",
            "Where something appears on screen (top/bottom/left/right of the image) is",
            "image-space information — it does not by itself prove which side/end of the",
            "object the camera is actually on. See image_space_notes in the spatial",
            "knowledge above for this object's specific examples.",
            "",
        ]
        if camera_view_hint:
            lines += [
                "Camera-vector hint (deterministic, from current camera position - focal point):",
                camera_view_hint,
                "Use this as an anchor only if it agrees with the screenshot; do not ignore the image.",
                "",
            ]

        lines += [
            "Before answering, work through these steps:",
            "  1. Describe what you actually see in the current screenshot.",
            "  2. Using the spatial knowledge above, infer the camera's current anatomical",
            "     position (which side/end of the object it is facing).",
            "  3. Decide which single canonical view (by its exact name in canonical_views",
            "     above) the current observation most inclines toward — even if it's an",
            "     imperfect/oblique match, pick the closest one.",
            "  4. Compare that inferred position/view against the target view.",
            "  5. Choose the smallest camera action that moves toward the target.",
            "  6. Do not change rendering style or isovalue. Do not invent anatomy that isn't",
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
    fp = camera_state["focal_point"]
    up = camera_state["view_up"]
    va = camera_state["view_angle"]
    cr = camera_state["clipping_range"]
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
        lines += [
            "Visual observation:",
            "<describe what you actually see in the current screenshot>",
            "",
            "Camera position inference:",
            "<using the spatial knowledge JSON above, infer which anatomical side/end the",
            "camera is currently facing, and how that compares to the target view>",
            "",
            "current_view_inclination: <exact canonical_views name the current observation",
            "most inclines toward, e.g. one of the names from canonical_views above>",
            "",
            "Reasoning:",
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
