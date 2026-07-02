import json
import numpy as np
from pathlib import Path
from typing import Union


def get_camera_state(camera) -> dict:
    return {
        "position":      list(camera.GetPosition()),
        "focal_point":   list(camera.GetFocalPoint()),
        "view_up":       list(camera.GetViewUp()),
        "view_angle":    camera.GetViewAngle(),
        "clipping_range": list(camera.GetClippingRange()),
    }


def set_camera_state(camera, state: dict):
    camera.SetPosition(*state["position"])
    camera.SetFocalPoint(*state["focal_point"])
    camera.SetViewUp(*state["view_up"])
    camera.SetViewAngle(state["view_angle"])
    camera.SetClippingRange(*state["clipping_range"])


def camera_distance(state: dict) -> float:
    pos = np.array(state["position"])
    fp  = np.array(state["focal_point"])
    return float(np.linalg.norm(pos - fp))


def save_camera_state_json(state: dict, path: Union[str, Path]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_camera_state_json(path: Union[str, Path]) -> dict:
    with open(path) as f:
        return json.load(f)
