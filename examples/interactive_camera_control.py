"""
Interactively choose a camera action via keyboard and see its effect immediately
in a live VTK window. Useful for building intuition about what each action
actually does before trusting an LLM to pick them.

Input is handled entirely by VTK's own interactor (KeyPressEvent), not Python's
input() — the render window needs its native event loop (interactor.Start())
running continuously to stay visible and responsive, which isn't compatible
with blocking on input() in the same thread.

Press Escape to quit, Backspace to undo. See the printed key legend for actions.
Mouse-driven free camera orbit/pan/zoom is intentionally disabled (vtkInteractorStyleUser)
so it can't interfere with the key bindings — use the listed actions to move the camera.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import vtk

from camera_reasoning.camera_actions import VALID_ACTIONS, apply_action
from camera_reasoning.camera_state import get_camera_state, set_camera_state
from camera_reasoning.volume_scene import build_isosurface_pipeline, load_raw_volume

RAW_PATH = "data/foot_256x256x256_uint8.raw"
DIMENSIONS = (256, 256, 256)
ISOVALUE = 80

ACTIONS = sorted(a for a in VALID_ACTIONS if a not in ("STOP", "UNDO_LAST"))
# Avoid VTK's own reserved interactor shortcuts, which can fire independently of our
# KeyPressEvent observer: q/e=quit, r=reset camera, w/s=wireframe/surface, p=pick,
# 3=stereo, f=fly-to, u=user event, j/t=joystick/trackball toggle.
KEYS = "124567890abcdghiklmnovxyz"[: len(ACTIONS)]
KEY_TO_ACTION = dict(zip(KEYS, ACTIONS))


def print_legend():
    print("\nKey bindings:")
    for key, action in KEY_TO_ACTION.items():
        print(f"  [{key}] {action}")
    print("  [BackSpace] undo last action")
    print("  [Escape] quit")


def print_camera_state(renderer):
    state = get_camera_state(renderer.GetActiveCamera())
    pos = state["position"]
    fp = state["focal_point"]
    print(f"  position=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]  "
          f"focal_point=[{fp[0]:.1f}, {fp[1]:.1f}, {fp[2]:.1f}]")


def main():
    image = load_raw_volume(RAW_PATH, DIMENSIONS)
    actor, renderer, render_window = build_isosurface_pipeline(image, ISOVALUE)
    render_window.SetOffScreenRendering(0)  # on-screen so you can watch it live
    renderer.ResetCamera()

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    # vtkInteractorStyleTrackballCamera (the default) has hardcoded shortcuts of its
    # own (q/e=quit, r=reset camera, w/s=wireframe/surface, p=pick, 3=stereo, ...)
    # that fire alongside our KeyPressEvent observer and collide with our bindings.
    # vtkInteractorStyleUser has no default key or mouse behavior, so only our
    # own key bindings apply.
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleUser())

    history = []
    camera_stack = []

    def on_key_press(obj, event):
        key = obj.GetKeySym()

        if key == "Escape":
            obj.TerminateApp()
            return

        if key == "BackSpace":
            if not camera_stack:
                print("Nothing to undo.")
                return
            set_camera_state(renderer.GetActiveCamera(), camera_stack.pop())
            renderer.ResetCameraClippingRange()
            render_window.Render()
            history.append("UNDO_LAST")
            print("Undid last action.")
            print_camera_state(renderer)
            return

        action = KEY_TO_ACTION.get(key)
        if action is None:
            return

        camera = renderer.GetActiveCamera()
        camera_stack.append(get_camera_state(camera))
        apply_action(action, camera, renderer)
        render_window.Render()
        history.append(action)
        print(f"Applied: {action}")
        print_camera_state(renderer)

    interactor.AddObserver("KeyPressEvent", on_key_press)
    interactor.Initialize()
    render_window.Render()

    print_legend()
    print_camera_state(renderer)
    print("\nClick the render window to give it keyboard focus, then press a key.")

    interactor.Start()

    print("\nSession action history:", history)


if __name__ == "__main__":
    main()
