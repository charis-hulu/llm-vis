from pathlib import Path
from .camera_actions import ACTION_DESCRIPTIONS, VALID_ACTIONS


def write_llm_prompt(
    output_dir: str,
    camera_state: dict,
    action_history: list[dict],
    target_description: str,
    screenshot_path: str,
    target_image_path: str | None = None,
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

    lines += [
        "Return exactly this format:",
        "",
        "Visual diagnosis:",
        "...",
        "",
        "Reasoning:",
        "...",
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
