"""
Run the camera-reasoning pipeline against the foot dataset, optionally using
frozen-reference-embedding alignment instead of the existing text-based loop.

Examples:
  # One-shot check: retrieve + freeze target, analyze, reason, print result, apply nothing.
  .venv/bin/python examples/run_frozen_reference_alignment.py --use-frozen-reference-alignment --dry-run

  # Real run: freeze the target once, then iterate up to 5 times, applying actions.
  .venv/bin/python examples/run_frozen_reference_alignment.py --use-frozen-reference-alignment --iterations 5

  # Old behavior, unchanged (no frozen-reference alignment):
  .venv/bin/python examples/run_frozen_reference_alignment.py --iterations 5
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from camera_reasoning import CameraReasoningSession
from camera_reasoning.frozen_reference_alignment import run_camera_reasoning_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-frozen-reference-alignment", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--references", default="reference_views")
    args = parser.parse_args()

    session = CameraReasoningSession(
        raw_path="data/foot_256x256x256_uint8.raw",
        dimensions=(256, 256, 256),
        scalar_type="uint8",
        isovalue=80,
        output_dir="output",
        target_description="Align with the closest matching reference view.",
    )
    session.initialize()

    result = run_camera_reasoning_pipeline(
        session,
        num_iterations=args.iterations,
        use_frozen_reference_alignment=args.use_frozen_reference_alignment,
        references_dir=args.references,
        dry_run=args.dry_run,
    )

    print("\n=== FINAL RESULT ===")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
