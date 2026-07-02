"""
Replay a sequence of camera actions in a live VTK window to sanity-check whether
the sequence chosen by the LLM actually makes sense.

Defaults to replaying output/action_history.json (the last run). Edit ACTION_SEQUENCE
below to test a different sequence by hand.

Press any key in the render window to apply the next action. Each step is also
saved to output/replay_screenshots/ so you can review the sequence afterward.
"""
import json
import sys
from pathlib import Path

import vtk

sys.path.insert(0, str(Path(__file__).parent.parent))

from camera_reasoning.camera_actions import apply_action
from camera_reasoning.volume_scene import build_isosurface_pipeline, load_raw_volume, save_screenshot

RAW_PATH = "data/foot_256x256x256_uint8.raw"
DIMENSIONS = (256, 256, 256)
ISOVALUE = 80

history_path = Path("output/action_history.json")
if history_path.exists():
    ACTION_SEQUENCE = [entry["action"] for entry in json.loads(history_path.read_text())]
else:
    ACTION_SEQUENCE = [
        "ELEVATION_UP_COARSE",
        "AZIMUTH_RIGHT_MEDIUM",
        "ZOOM_IN",
    ]

print("Replaying sequence:")
for i, action in enumerate(ACTION_SEQUENCE, 1):
    print(f"  {i}. {action}")

image = load_raw_volume(RAW_PATH, DIMENSIONS)
actor, renderer, render_window = build_isosurface_pipeline(image, ISOVALUE)
render_window.SetOffScreenRendering(0)  # we want an on-screen interactive window here
renderer.ResetCamera()

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)
interactor.Initialize()
render_window.Render()

save_screenshot(render_window, "output/replay_screenshots/step_000_initial.png")
print(f"\nStep 0 (initial view) saved. Press any key in the render window to apply "
      f"step 1: {ACTION_SEQUENCE[0] if ACTION_SEQUENCE else '(sequence empty)'}")

state = {"i": 0}


def advance(obj, event):
    i = state["i"]
    if i >= len(ACTION_SEQUENCE):
        return
    action = ACTION_SEQUENCE[i]
    print(f"Step {i + 1}/{len(ACTION_SEQUENCE)}: applying {action}")
    apply_action(action, renderer.GetActiveCamera(), renderer)
    render_window.Render()
    save_screenshot(render_window, f"output/replay_screenshots/step_{i + 1:03d}_{action}.png")
    state["i"] += 1
    if state["i"] >= len(ACTION_SEQUENCE):
        print("Sequence complete. Close the window when done.")


interactor.AddObserver("KeyPressEvent", advance)
interactor.Start()
