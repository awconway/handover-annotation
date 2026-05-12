#!/usr/bin/env python3
"""Recreate a source image with GPT Image 2 at 4K landscape size."""

from __future__ import annotations

import argparse
import base64
import os
import sys
import urllib.request
from pathlib import Path

from openai import OpenAI


DEFAULT_PROMPT = """
Recreate the supplied slide as a polished 4K landscape presentation image.
Preserve the source slide's content, clinical/scientific style, visual hierarchy,
card layout, colors, and overall composition while improving sharpness, spacing,
legibility, and projection quality.

GLOBAL HEADER LOCK:
Canvas is exactly 3840 x 2160.
Every slide must use the same fixed header geometry:
- Kicker text starts at x=108, y=128.
- Kicker font: uppercase sans serif, 48 px, letter-spaced, color #1f61b5.
- Kicker underline starts at x=108, y=210.
- Underline size: width 520 px, height 8 px.
- Underline color: #1f61b5.
- Main title starts at x=108, y=318.
- Main title font: elegant serif similar to Georgia, color #07122f.
- Do not move, resize, recolor, restyle, or reinterpret the kicker, underline, or title block.
- Keep all slide-specific content outside the locked header area.

Use the same pale grey-blue clinical presentation style, white rounded panels,
thin #c9d9eb borders, dark navy #07122f text, and muted slate-blue #456d9d accents.
Do not add logos, watermarks, patient-identifiable photos, nurse caps, scrapbook
textures, torn paper, sticky tape, dark mode, purple gradients, or saturated red labels.
""".strip()
DEFAULT_SIZE = "3840x2160"
DEFAULT_QUALITY = "high"
DEFAULT_MODEL = "gpt-image-2"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def decode_image_response(result) -> bytes:
    item = result.data[0]
    b64_json = getattr(item, "b64_json", None)
    if b64_json:
        return base64.b64decode(b64_json)

    url = getattr(item, "url", None)
    if url:
        with urllib.request.urlopen(url) as response:
            return response.read()

    raise RuntimeError("Image response did not include b64_json or url output.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Use gpt-image-2 to recreate an input image as a 4K PNG."
    )
    parser.add_argument("image", type=Path, help="Input image to send as a reference")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--size", default=DEFAULT_SIZE)
    parser.add_argument("--quality", default=DEFAULT_QUALITY)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/tpch-rounds-slide-deck-scientific-4k"),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    load_dotenv(Path.cwd() / ".env")

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY was not found in the environment or .env.", file=sys.stderr)
        return 2

    image_path = args.image
    if not image_path.is_absolute():
        image_path = Path.cwd() / image_path
    image_path = image_path.resolve()
    if not image_path.exists():
        print(f"Input image does not exist: {image_path}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / f"{image_path.stem}-4k.png"

    client = OpenAI()
    print(
        f"Submitting {image_path} to {args.model} "
        f"(size={args.size}, quality={args.quality})"
    )

    with image_path.open("rb") as image_file:
        result = client.images.edit(
            model=args.model,
            image=[image_file],
            prompt=args.prompt,
            size=args.size,
            quality=args.quality,
            output_format="png",
        )

    output_path.write_bytes(decode_image_response(result))
    print(f"Wrote {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
