"""
Standalone script equivalent of the Jupyter notebook.
Run this to step through the manual ChatGPT camera reasoning loop interactively.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from camera_reasoning import CameraReasoningSession

session = CameraReasoningSession(
    raw_path="data/foot_256x256x256_uint8.raw",
    dimensions=(256, 256, 256),
    scalar_type="uint8",
    isovalue=80,
    output_dir="output",
    target_description=(
        "Top view of the foot. Big toe on the right, little toe on the left, "
        "toes near the top, heel near the bottom."
    ),
)

session.initialize()
session.render_and_save()
session.write_llm_prompt()

print("\n--- LLM PROMPT ---")
print(open("output/llm_prompt.txt").read())
print("\nScreenshot saved to: output/screenshots/latest.png")
print("\nSend the prompt and screenshot to ChatGPT, then paste the response below.")

while True:
    print("\n--- Paste ChatGPT response (end with a line containing only 'END') ---")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    response = "\n".join(lines)

    try:
        session.process_chatgpt_response(response)
    except Exception as e:
        print(f"Error: {e}")
        continue

    print("\n--- LLM PROMPT for next iteration ---")
    print(open("output/llm_prompt.txt").read())
    print("Screenshot: output/screenshots/latest.png")

    if any(e["action"] == "STOP" for e in session._action_history[-1:]):
        print("\nAlignment complete.")
        break
