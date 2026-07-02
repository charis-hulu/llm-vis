import re
from .camera_actions import VALID_ACTIONS


class ActionParseError(ValueError):
    pass


PLACEHOLDER_MARKER = "PASTE CHATGPT RESPONSE HERE"


def extract_action(response: str) -> str:
    """Extract exactly one valid action from a ChatGPT response string."""
    if PLACEHOLDER_MARKER in response.upper():
        raise ActionParseError(
            "chatgpt_response still contains the placeholder text — "
            "paste ChatGPT's actual reply (with a 'Next action:' line) before running this cell."
        )

    # Primary: look for the token immediately after "Next action:", whether
    # it's on the same line or the line below (allow markdown bold, etc.)
    # Action names may contain digits (e.g. ROLL_CW_90), so [A-Z_0-9] is needed.
    match = re.search(
        r"Next action:\**\s*\n?\s*\**\s*([A-Z][A-Z_0-9]+)", response, re.IGNORECASE
    )
    if match:
        candidate = match.group(1).strip().upper()
        if candidate in VALID_ACTIONS:
            return candidate

    # Fallback: scan the response for any valid action token, but ignore the
    # "Allowed actions:" list (format "  - ACTION_NAME: description") which
    # gets echoed back verbatim if the full prompt was pasted alongside the
    # reply — otherwise every action in that list is picked up as a match.
    scan_text = re.sub(r"(?m)^\s*-\s*[A-Z][A-Z_0-9]+:.*$", "", response)

    found = set()
    for token in re.findall(r"\b([A-Z][A-Z_0-9]+)\b", scan_text):
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
