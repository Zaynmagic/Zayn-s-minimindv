# Make colored-square images and chat rows for quick checks.

import argparse
import json
from pathlib import Path
from PIL import Image


def generate(path):
    path = Path(path)
    (path / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for name, color in (
        ("red", (255, 0, 0)),
        ("green", (0, 255, 0)),
        ("blue", (0, 0, 255)),
        ("yellow", (255, 255, 0)),
    ):
        relative = f"images/{name}.png"
        Image.new("RGB", (32, 32), color).save(path / relative)
        rows.append(
            {
                "image": relative,
                "conversations": [
                    {"role": "user", "content": "<image>\nDescribe the color."},
                    {"role": "assistant", "content": f"The square is {name}."},
                ],
            }
        )
    for stage in ("align", "sft"):
        (path / f"{stage}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic")
    generate(parser.parse_args().output)
