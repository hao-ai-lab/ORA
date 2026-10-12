"""Isolated simulator worker for rebuildable LIBERO gallery previews."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from .robustness import RobustnessEnvironment
from .session import _prepare_pinned_libero_imports


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_cached_image(
    path: Path, expected_sha256: str | None, *, width: int, height: int,
) -> bool:
    if not path.is_file():
        return False
    if expected_sha256 and _sha256(path) != expected_sha256:
        return False
    try:
        with Image.open(path) as image:
            return image.mode == "RGB" and image.size == (width, height)
    except OSError:
        return False


def render_batch(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Render one deterministic batch in the configured simulator process."""

    specs = payload.get("specs")
    render = payload.get("render")
    if not isinstance(specs, list) or not isinstance(render, Mapping):
        raise RuntimeError("preview worker input has no specs or render contract")
    width = int(render["width"])
    height = int(render["height"])
    seed = int(render["seed"])
    libero_pro_root = Path(str(payload["libero_pro_root"])).resolve()
    data_root = Path(str(payload["data_root"])).resolve()
    output_root = Path(str(payload["output_root"])).resolve()

    with tempfile.TemporaryDirectory(
        prefix="libero-preview-runtime-",
    ) as runtime_name:
        _prepare_pinned_libero_imports(
            libero_pro_root=libero_pro_root,
            data_root=data_root,
            runtime_config_root=Path(runtime_name),
        )
        from libero.libero.envs import OffScreenRenderEnv

        results = []
        for spec in specs:
            print(
                f"LIBERO preview worker: {spec.get('key')}",
                file=sys.stderr,
                flush=True,
            )
            output = Path(str(spec["output_path"])).resolve()
            if output_root not in output.parents:
                raise RuntimeError(f"preview output escaped cache: {output}")
            if not _valid_cached_image(
                output, spec.get("expected_image_sha256"),
                width=width, height=height,
            ):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                environment: Any = OffScreenRenderEnv(
                    bddl_file_name=spec["bddl_path"],
                    robots=list(render["robots"]),
                    controller=str(render["controller"]),
                    camera_names=[str(render["camera"])],
                    camera_heights=height,
                    camera_widths=width,
                    camera_depths=False,
                    horizon=10,
                )
                if spec.get("robustness_case") is not None:
                    environment = RobustnessEnvironment(
                        environment,
                        case=str(spec["robustness_case"]),
                        suite=str(spec["runtime_suite"]),
                        task_name=str(spec["task_name"]),
                        bddl_path=Path(str(spec["bddl_path"])),
                        seed=seed,
                    )
                try:
                    environment.reset()
                    states = torch.load(
                        spec["init_path"], map_location="cpu", weights_only=False,
                    )
                    if len(states) <= seed:
                        raise RuntimeError(
                            f"preview init state is missing: {spec['key']}"
                        )
                    observation = environment.set_init_state(states[seed])
                    rgb = np.asarray(
                        observation[f"{render['camera']}_image"]
                    )[::-1]
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                    if rgb.shape != (height, width, 3):
                        raise RuntimeError(f"invalid preview image shape: {rgb.shape}")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    temporary = output.with_name(
                        f".{output.name}.{os.getpid()}.tmp.png"
                    )
                    Image.fromarray(rgb, mode="RGB").save(
                        temporary, format="PNG", optimize=True,
                    )
                    os.replace(temporary, output)
                finally:
                    environment.close()
            record = {
                key: spec[key]
                for key in (
                    "key", "layer", "family", "suite", "task_id",
                    "task_name", "perturbation", "instruction", "bddl_source",
                    "init_source", "bddl_sha256", "init_sha256",
                    "preview_identity_sha256", "image",
                )
            }
            for optional in ("runtime_suite", "robustness_case"):
                if spec.get(optional) is not None:
                    record[optional] = spec[optional]
            record.update({
                "seed": seed,
                "width": width,
                "height": height,
                "image_sha256": _sha256(output),
                "bytes": output.stat().st_size,
            })
            results.append(record)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    results = render_batch(payload)
    temporary = args.result.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
