"""Source-checked intermediate grasp views shared by Native and managed clients.

These are historical post-close views. They never enter the current camera
bundle or replace the final observation. The final frame is delivered by the
ordinary observation path, including after a failed lift.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from .action_views import PostActionObservationError


def materialize_skill_observations(
    *, receipt: Mapping[str, Any] | None, run_root: Path, workspace: Path,
    camera_names: Sequence[str],
) -> list[dict[str, Any]]:
    if not isinstance(receipt, Mapping):
        return []
    record = receipt.get("receipt", receipt)
    if not isinstance(record, Mapping) or record.get("schema") != "ora.manipulation-skill-sequence-receipt.v1":
        return []
    skill = record.get("skill_receipt", {})
    evidence = skill.get("observation_evidence", [])
    if not evidence:
        return []
    if (record.get("skill") != "grasp_execute" or "lift" not in record.get("committed_sequence", [])
            or not isinstance(evidence, list) or len(evidence) != 1):
        raise PostActionObservationError("invalid intermediate grasp observation binding")
    run_root, workspace = run_root.resolve(), workspace.resolve()
    if not workspace.is_relative_to(run_root):
        raise PostActionObservationError("skill view workspace escapes run")
    source = evidence[0]
    root = Path(source["observation_root"]).resolve()
    if not root.is_relative_to(run_root) or source.get("role") != "post_close_before_lift":
        raise PostActionObservationError("skill observation source escapes its authorized run or role")
    raw = (root / "observation.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != source.get("observation_sha256"):
        raise PostActionObservationError("skill observation manifest hash mismatch")
    observation = json.loads(raw)
    if observation.get("observation_id") != source.get("observation_id"):
        raise PostActionObservationError("skill observation identity mismatch")
    images = []
    # At most two baseline views. The ordinary current cameras provide the
    # other side of the comparison and keep their original identity/count.
    available = [name for name in camera_names if name in observation.get("cameras", {})]
    # Maintained dual-arm cameras carry their public arm prefix; prefer that
    # view over an unrelated arm or a second external camera. Single-arm
    # bundles already put the external and wrist views first.
    arm_views = [name for name in available if name.startswith(str(source.get("arm")) + "_")]
    selected = list(dict.fromkeys(available[:1] + arm_views[:1] + available))[:2]
    for index, name in enumerate(selected):
        files = observation["cameras"][name]["files"]
        ref = files.get("rgb_lossless_png") or files["rgb"]
        path = (root / ref["path"]).resolve()
        if not path.is_relative_to(root):
            raise PostActionObservationError("skill RGB source escapes observation")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != ref.get("sha256"):
            raise PostActionObservationError("skill RGB hash mismatch")
        destination = (workspace / "inputs/skill-observations" / source["observation_sha256"]).resolve()
        if not destination.is_relative_to(workspace):
            raise PostActionObservationError("materialized skill RGB destination escapes workspace")
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / f"{index:02d}-{digest}.png"
        if target.exists() and target.read_bytes() != data:
            raise PostActionObservationError("immutable skill RGB artifact drift")
        if not target.exists():
            target.write_bytes(data)
            target.chmod(0o444)
        with Image.open(target) as image:
            width, height = image.size
        images.append({"role": source["role"], "camera": name,
            "observation_id": source["observation_id"], "sha256": digest,
            "observation_sha256": source["observation_sha256"],
            "width": width, "height": height,
            "path": target.relative_to(workspace).as_posix(),
            "run_path": target.relative_to(run_root).as_posix()})
    return images


def native_blocks(images: Sequence[Mapping[str, Any]], *, run_root: Path) -> list[dict[str, Any]]:
    from ora.agent.evidence.trajectory import image_data_url

    blocks = []
    for item in images:
        blocks.extend([
            {"type": "text", "text": (
                f"GRASP BASELINE · after close, before lift · {item['camera']} · "
                "historical intermediate frame; compare with the current action-result view"
            )},
            {"type": "image_url", "image_url": {"url": image_data_url(run_root / item["run_path"])}},
        ])
    return blocks


def resolve_skill_images(context: Mapping[str, Any], *, root: Path, path_key: str = "path") -> tuple[Path, ...]:
    """Verify materialized historical images before a managed backend reads them."""
    from .action_views import _safe_file

    images = context.get("skill_observations", [])
    if not isinstance(images, list) or len(images) > 2:
        raise PostActionObservationError("invalid skill observation image count")
    paths = []
    for item in images:
        if item.get("role") != "post_close_before_lift":
            raise PostActionObservationError("invalid skill observation image role")
        path = _safe_file(root, item[path_key])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item.get("sha256"):
            raise PostActionObservationError("materialized skill image hash mismatch")
        paths.append(path)
    return tuple(paths)
