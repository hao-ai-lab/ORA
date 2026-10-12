"""The ``move_to`` tool of the semantic pose primitives.

A mixin of ``core.ManipulationToolRuntime``; it holds methods only, no
dataclass fields.
"""

from __future__ import annotations

from typing import Any, Mapping



class MoveToRuntimeMixin:
    """The ``move_to`` semantic pose primitive dispatch."""

    def _move_to(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return self._motion_move_to_primitive_v2(arguments)
