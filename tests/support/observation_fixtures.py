"""A minimal on-disk public observation for offline delivery tests."""

from __future__ import annotations

import hashlib
import json

from PIL import Image


def observation_fixture(root, cameras, *, sequence=1):
    """Write one ``ora-observation.v1`` with a small lossless RGB per camera."""
    source = root / f"source-{sequence}"
    source.mkdir()
    records = {}
    for index, name in enumerate(cameras):
        directory = source / name
        directory.mkdir()
        path = directory / "rgb.png"
        Image.new("RGB", (12, 10), (sequence, index * 30, 100)).save(path)
        records[name] = {
            "width": 12, "height": 10,
            "files": {"rgb_lossless_png": {
                "path": f"{name}/rgb.png",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }},
        }
    observation = {
        "schema": "ora-observation.v1",
        "observation_id": f"observation-{sequence}", "sequence": sequence,
        "captured_at_unix_s": 123.0 + sequence,
        "cameras": records, "proprioception": {"end_effectors": {}},
    }
    (source / "observation.json").write_text(json.dumps(observation))
    return observation, source
