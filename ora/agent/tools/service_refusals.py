"""The envelope for a failed SAM3 or Contact-GraspNet backed tool call.

``vision_segment``, ``vision_ground`` and ``grasp_plan`` turned every backend
``ValueError`` into one retryable code (``segmentation_failed``,
``grounding_failed``, ``grasp_proposal_failed``). A restarted service whose
instance identity can never match the bound one again, a 422 naming the field
the service refused, and a target with too few depth pixels all looked like a
transient hiccup: the first loops forever, the second
loses the service's own ``detail`` text, the third is retried unchanged.

The classes come from ``agent/tools/artifact_errors.py``; the codes are
existing ones:

* a permanent service fault -- ``internal_tool_error``, not retryable
  (infrastructure, so it is abort evidence);
* a store fault -- its artifact code, not retryable (infrastructure);
* a 4xx refusal of the request content -- ``invalid_arguments`` with the
  service's ``detail``;
* a target the proposer cannot use -- the tool's own code, ``field=target_ref``,
  and only a different reference can succeed;
* a transient failure -- the tool's own code, retryable, with the service and
  status. It is not classed as infrastructure: that would need a new code in
  ``agent/error_classes.py``, which changes the generated v1 ``finish_episode``
  text.
"""

from __future__ import annotations

from typing import Any

from .artifact_errors import (
    ArtifactIntegrityError, ModelServiceError, TargetInsufficient,
)
from .robot import ToolError


def refusal(
    error: ValueError, *, code: str, target_ref: Any = None,
) -> ToolError:
    """Map one backend failure of a tool whose own failure code is ``code``."""

    if isinstance(error, ArtifactIntegrityError):
        return ToolError(str(error), code=error.code, retryable=False)
    if isinstance(error, ModelServiceError):
        service = {"service": error.service, "failure": error.failure}
        if error.status is not None:
            service["status"] = error.status
        if error.failure == "permanent":
            return ToolError(
                str(error), code="internal_tool_error", retryable=False,
                constraints=service,
            )
        if error.failure == "request":
            return ToolError(
                str(error), code="invalid_arguments",
                constraints={**service, "detail": error.detail},
            )
        return ToolError(str(error), code=code, retryable=True, constraints=service)
    if isinstance(error, TargetInsufficient):
        return ToolError(
            str(error), code=code, retryable=True, field="target_ref",
            received=target_ref, retry_requires_changed_arguments=True,
        )
    return ToolError(str(error), code=code, retryable=True)


__all__ = ["refusal"]
