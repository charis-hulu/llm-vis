"""
Quick standalone check: can the configured AI endpoint generate images (not just read them)?

Uses the same .env config (OPENAI_API_KEY / AI_URL / AI_MODEL) as the camera reasoning
loop. Tries the OpenAI images.generate API, saves the result if it works.
"""
import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")

api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("AI_URL") or None
model = os.environ.get("AI_MODEL") or "gpt-image-1"

client = OpenAI(api_key=api_key, base_url=base_url)

print(f"Requesting image generation from model={model!r} base_url={base_url!r} ...")

try:
    result = client.images.generate(
        model=model,
        prompt="A simple red cube on a white background, studio lighting.",
        n=1,
    )
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    raise SystemExit(1)

image = result.data[0]
out_path = Path(__file__).parent / "test_image_generation_output.png"

if getattr(image, "b64_json", None):
    out_path.write_bytes(base64.b64decode(image.b64_json))
    print(f"Saved image to {out_path}")
elif getattr(image, "url", None):
    print(f"Image available at URL: {image.url}")
else:
    print(f"Unexpected response shape: {result}")
