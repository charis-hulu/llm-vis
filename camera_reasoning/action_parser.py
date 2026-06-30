import re
from .camera_actions import VALID_ACTIONS


class ActionParseError(ValueError):
    pass


def extract_action(response: str) -> str:
    """Extract exactly one valid action from a ChatGPT response string."""
    # Primary: look for line(s) immediately after "Next action:"
    match = re.search(r"Next action:\s*\n\s*([A-Z_]+)", response, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip().upper()
        if candidate in VALID_ACTIONS:
            return candidate

    # Fallback: scan entire response for any valid action token
    found = set()
    for token in re.findall(r"\b([A-Z][A-Z_]+)\b", response):
        if token in VALID_ACTIONS:
            found.add(token)

    if len(found) == 0:
        valid_list = "\n  ".join(sorted(VALID_ACTIONS))
        raise ActionParseError(
            f"No valid action found in the response.\n"
            f"Valid actions are:\n  {valid_list}"
        )

    if len(found) > 1:
        raise ActionParseError(
            f"Multiple valid actions found: {', '.join(sorted(found))}.\n"
            f"Please paste a response that contains exactly one action name."
        )

    return found.pop()
