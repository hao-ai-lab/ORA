"""Shared SAM3 text-grounding and Contact-GraspNet loopback services."""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path
from typing import Any, Mapping

from ora.agent.model_services.resident import (
    GraspService,
    SegmentService,
    _decode_png,
    _encode_npy,
    _request_id,
    _response_service_identity,
)
from ora.agent.model_services.identity import (
    grounding_grasp_adapter_sha256,
)


class GroundingSegmentService(SegmentService):
    """One SAM3 instance serving both point prompts and text grounding."""

    def __init__(self, *, source_root: Path, checkpoint: Path) -> None:
        super().__init__(source_root=source_root, checkpoint=checkpoint)
        self.adapter_sha256 = grounding_grasp_adapter_sha256()

    def _warmup(self) -> float:
        import numpy as np
        from PIL import Image

        image = np.full((128, 128, 3), 32, dtype=np.uint8)
        image[40:92, 48:84] = [220, 48, 32]
        source = Image.fromarray(image)
        started = time.perf_counter()
        self._predict(source, (64.0, 64.0))
        with self.torch.inference_mode(), self.torch.autocast(
            "cuda", dtype=self.torch.bfloat16,
        ):
            state = self.processor.set_image(source)
            self.processor.set_text_prompt("red object", state)
        self.torch.cuda.synchronize()
        return (time.perf_counter() - started) * 1000

    def health(self) -> dict[str, Any]:
        value = super().health()
        value["capabilities"] = ["segment_point", "ground_text", "segment_points_v1"]
        return value

    def infer_ground(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np

        if payload.get("schema") != "ora.model-service.ground-text-request.v1":
            raise ValueError("text-grounding request schema mismatch")
        text = payload.get("text")
        maximum = payload.get("max_candidates")
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 120:
            raise ValueError("text must contain 1..120 non-whitespace characters")
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or not 1 <= maximum <= 3
        ):
            raise ValueError("max_candidates must be an integer in [1,3]")
        image = _decode_png(payload.get("image_png_base64"))
        with self.torch.inference_mode(), self.torch.autocast(
            "cuda", dtype=self.torch.bfloat16,
        ):
            state = self.processor.set_image(image)
            state = self.processor.set_text_prompt(text.strip(), state)
        self.torch.cuda.synchronize()
        masks = state["masks"].detach().cpu().numpy().astype(np.uint8)
        scores = state["scores"].detach().float().cpu().numpy().reshape(-1)
        boxes = state["boxes"].detach().float().cpu().numpy().reshape(-1, 4)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim == 2:
            masks = masks[None]
        if not (masks.ndim == 3 and len(masks) == len(scores) == len(boxes)):
            raise RuntimeError("SAM3 returned unaligned text-grounding outputs")
        order = np.argsort(-scores, kind="stable")[:maximum]
        masks, scores, boxes = masks[order], scores[order], boxes[order]
        self.requests += 1
        return {
            "schema": "ora.model-service.ground-text.v1",
            "request_id": _request_id(),
            "model": dict(self.model_identity),
            "masks_npy_base64": _encode_npy(masks),
            "scores_npy_base64": _encode_npy(scores),
            "boxes_xyxy_npy_base64": _encode_npy(boxes),
        }


def _serve_segmentation(service: GroundingSegmentService, *, host: str, port: int) -> None:
    import uvicorn
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="Robot Harness segmentation service")
    lock = asyncio.Lock()

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return service.health()

    async def run_inference(method: str, payload: dict[str, Any]) -> dict[str, Any]:
        queued = time.perf_counter()
        try:
            async with lock:
                started = time.perf_counter()
                result = await asyncio.to_thread(getattr(service, method), payload)
                finished = time.perf_counter()
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result["metrics"] = {
            "queue_ms": (started - queued) * 1000,
            "inference_ms": (finished - started) * 1000,
            "total_ms": (finished - queued) * 1000,
        }
        result["service_identity"] = _response_service_identity(service)
        return result

    @app.post("/v1/segment-point")
    async def segment_point(payload: dict[str, Any]) -> dict[str, Any]:
        return await run_inference("infer", payload)

    @app.post("/v1/segment-points")
    async def segment_points(payload: dict[str, Any]) -> dict[str, Any]:
        return await run_inference("infer_batch", payload)

    @app.post("/v1/ground-text")
    async def ground_text(payload: dict[str, Any]) -> dict[str, Any]:
        return await run_inference("infer_ground", payload)

    uvicorn.run(app, host=host, port=port, workers=1, access_log=False)


def _serve_grasp(service: GraspService, *, host: str, port: int) -> None:
    import uvicorn
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="Robot Harness grasp-proposal service")
    lock = asyncio.Lock()

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return service.health()

    @app.post("/v1/grasp-proposals")
    async def grasp_proposals(payload: dict[str, Any]) -> dict[str, Any]:
        queued = time.perf_counter()
        try:
            async with lock:
                started = time.perf_counter()
                result = await asyncio.to_thread(service.infer, payload)
                finished = time.perf_counter()
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result["metrics"] = {
            "queue_ms": (started - queued) * 1000,
            "inference_ms": (finished - started) * 1000,
            "total_ms": (finished - queued) * 1000,
        }
        result["service_identity"] = _response_service_identity(service)
        return result

    uvicorn.run(app, host=host, port=port, workers=1, access_log=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", required=True, choices=("segmentation", "grasp_proposals"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("model services must bind loopback only")
    if args.service == "segmentation":
        service = GroundingSegmentService(
            source_root=args.source_root, checkpoint=args.checkpoint,
        )
        _serve_segmentation(service, host=args.host, port=args.port)
        return
    if args.config is None:
        raise RuntimeError("grasp_proposals requires --config")
    service = GraspService(
        source_root=args.source_root, checkpoint=args.checkpoint,
        config=args.config,
    )
    service.adapter_sha256 = grounding_grasp_adapter_sha256()
    _serve_grasp(service, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
