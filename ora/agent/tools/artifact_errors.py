"""Typed failures of target references, the artifact store and model services.

The segmented-target loader, the grasp proposers and the SAM3 and
Contact-GraspNet clients used to raise a bare ``ValueError`` for every
failure, and each tool turned that into one code. A malformed ``target_ref``,
a store whose index no longer matched its manifest and a model service that
had restarted under a new instance identity all reached the policy as the same
argument refusal (EXP11B census D3, D4, B4), so a Harness fault blamed the
argument and a refusal the policy could fix looked like infrastructure.

Every class here still subclasses ``ValueError``: a caller that catches
``ValueError`` keeps working, and a caller that can tell them apart raises the
envelope that says who failed. ``agent/tools/target_refusals.py`` and
``agent/tools/service_refusals.py`` map them onto existing codes only; no new
code enters ``agent/error_classes.py``.
"""

from __future__ import annotations

from typing import Any


class TargetRefMalformed(ValueError):
    """The value is not a ``target-`` reference at all."""


class TargetRefUnknown(ValueError):
    """A well-formed reference that no segmented target in this run carries."""


class ArtifactIntegrityError(ValueError):
    """A Harness-written artifact is missing, malformed or outside its root."""

    code = "artifact_integrity_fault"


class ArtifactHashMismatch(ArtifactIntegrityError):
    """A Harness-written artifact no longer matches its recorded hash."""

    code = "artifact_hash_mismatch"


class TargetInsufficient(ValueError):
    """The target's own pixels or depth cannot support the requested proposal.

    Deterministic for one ``target_ref``: the same reference on the same
    observation fails the same way, so only a different reference can succeed.
    """


#: ``transient``: the service did not answer (connection, timeout, 5xx).
#: ``permanent``: it answered as a different or misconfigured service (instance,
#: model or adapter identity drift, a malformed or unready health record); no
#: later call in this episode can match the bound identity again.
#: ``request``: it refused the request content with a 4xx status.
SERVICE_FAILURES = ("transient", "permanent", "request")


class ModelServiceError(ValueError):
    """One model-service call failed, with the class of failure and its detail."""

    def __init__(
        self, message: str, *, service: str, failure: str,
        status: int | None = None, detail: Any = None,
    ) -> None:
        if failure not in SERVICE_FAILURES:
            raise ValueError(f"unknown model-service failure class: {failure}")
        super().__init__(message)
        self.service = service
        self.failure = failure
        self.status = status
        self.detail = detail


def service_identity_drift(message: str, *, service: str) -> ModelServiceError:
    """A bound service answered with another identity or an unready health record."""

    return ModelServiceError(message, service=service, failure="permanent")


def http_failure(exc: Exception, *, service: str, what: str) -> ModelServiceError:
    """Classify one httpx failure, keeping the service's own ``detail`` text.

    ``str(HTTPStatusError)`` keeps only the status line and a documentation
    link; the 422 body that names the refused field was being dropped.
    """

    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    detail = None
    if response is not None:
        try:
            body = response.json()
        except Exception:  # noqa: BLE001 - any undecodable body keeps detail None
            body = None
        detail = body.get("detail") if isinstance(body, dict) else None
    failure = (
        "request" if isinstance(status, int) and 400 <= status < 500 else "transient"
    )
    text = f"{service} {what} failed"
    if isinstance(status, int):
        text += f" with HTTP {status}"
    if detail is not None:
        text += f": {detail}"
    elif not isinstance(status, int):
        text += f": {type(exc).__name__}"
    return ModelServiceError(
        text, service=service, failure=failure,
        status=status if isinstance(status, int) else None, detail=detail,
    )


__all__ = [
    "ArtifactHashMismatch", "ArtifactIntegrityError", "ModelServiceError",
    "SERVICE_FAILURES", "TargetInsufficient", "TargetRefMalformed",
    "TargetRefUnknown", "http_failure", "service_identity_drift",
]
