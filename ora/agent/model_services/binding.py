"""Host-side endpoint selection shared by preflight and policy tool clients."""

from __future__ import annotations

import os
from typing import Any, Mapping


def service_endpoint(
    binding: Mapping[str, Any] | None, variable: str, default: str,
) -> str:
    """Use an explicit override, then the bound endpoint, then an unbound default.

    Client identity validation still rejects overrides that disagree with a
    bound service. Reading this function never contacts or starts a service.
    """

    if variable in os.environ:
        return os.environ[variable]
    if binding is not None and "endpoint" in binding:
        return str(binding["endpoint"])
    return default


def service_client_arguments(
    binding: Mapping[str, Any] | None, variable: str, default: str,
) -> dict[str, Any]:
    """Keep endpoint selection and expected identity paired in every constructor."""
    return {"endpoint": service_endpoint(binding, variable, default),
            "expected_binding": binding}
