"""
Generate the 18-view reference image bank for the foot dataset.

Before relying on the output: calibrate INITIAL_DORSAL_CAMERA in
camera_reasoning/reference_views.py against your actual dataset (the numbers here
are just carried over from the existing ResetCamera() defaults as a starting
point), then inspect reference_views/*.png and adjust VIEW_ANGLES signs there if
Medial/Lateral or Distal/Proximal come out swapped.

Run with: .venv/bin/python examples/generate_reference_views.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from camera_reasoning.reference_views import generate_reference_views

generate_reference_views(
    raw_path="data/foot_256x256x256_uint8.raw",
    dimensions=(256, 256, 256),
    scalar_type="uint8",
    isovalue=80,
    output_dir="reference_views",
)
