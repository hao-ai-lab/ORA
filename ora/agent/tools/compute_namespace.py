# Copyright contributors to the Robot Harness Runtime project.
"""The names a restricted geometry block may reach, and nothing else.

Kept as explicit tables rather than module objects: an attribute the policy
reaches for is either listed here or refused, and adding one is a visible edit.
Three refusals in one batch were names this file could have carried and did not
-- `np.column_stack` for a rotation built from its columns, `float` for a
hand-written quaternion conversion, and an `import` for an `array` that `np`
already provides -- so each cost the policy a turn and taught it nothing.
"""

from __future__ import annotations

from typing import Any, Mapping

import math

import numpy as np
from scipy.spatial.transform import Rotation

from .robot import ToolError


class _Table:
    """An attribute facade over an explicit mapping. Never a module object."""

    __slots__ = ("_name", "_entries")

    def __init__(self, name: str, entries: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_entries", dict(entries))

    def __getattr__(self, attr: str) -> Any:
        entries = object.__getattribute__(self, "_entries")
        if attr.startswith("_") or attr not in entries:
            name = object.__getattribute__(self, "_name")
            raise AttributeError(f"{name}.{attr} is not available in this evaluator")
        return entries[attr]

    def __setattr__(self, attr: str, value: Any) -> None:
        raise AttributeError("the evaluator namespace is read-only")

    def __dir__(self) -> list[str]:
        return sorted(object.__getattribute__(self, "_entries"))


_LINALG = _Table("np.linalg", {
    "norm": np.linalg.norm, "solve": np.linalg.solve,
    "lstsq": np.linalg.lstsq, "inv": np.linalg.inv, "det": np.linalg.det,
    # A surface normal from measured points is a principal axis of their
    # scatter, and blocks reached for `eigh` on `M.T @ M` to get it. Both are
    # pure numeric decompositions of an array the block already holds.
    "eigh": np.linalg.eigh, "svd": np.linalg.svd,
})

_NUMPY = _Table("np", {
    "array": np.array, "asarray": np.asarray,
    "dot": np.dot, "cross": np.cross, "outer": np.outer,
    "sqrt": np.sqrt, "abs": np.abs, "sign": np.sign,
    "sum": np.sum, "mean": np.mean, "median": np.median,
    "min": np.min, "max": np.max, "argmin": np.argmin, "argmax": np.argmax,
    "minimum": np.minimum, "maximum": np.maximum,
    "round": np.round, "clip": np.clip,
    # The element-wise conditional. `if` is not admitted, so this and
    # `sign` are how a block fixes the sign of a direction it just derived.
    "where": np.where,
    "concatenate": np.concatenate, "stack": np.stack,
    # Reached for and refused by live blocks building a rotation from
    # its columns; binding them is cheaper than teaching around them.
    "column_stack": np.column_stack, "vstack": np.vstack, "hstack": np.hstack,
    "transpose": np.transpose, "float64": np.float64,
    "eye": np.eye, "zeros": np.zeros, "ones": np.ones,
    "arccos": np.arccos, "arcsin": np.arcsin, "arctan2": np.arctan2,
    "cos": np.cos, "sin": np.sin, "tan": np.tan,
    "degrees": np.degrees, "radians": np.radians,
    "linalg": _LINALG, "pi": float(np.pi),
})

_ROTATION = _Table("Rotation", {
    "from_quat": Rotation.from_quat, "from_matrix": Rotation.from_matrix,
    "from_rotvec": Rotation.from_rotvec, "from_euler": Rotation.from_euler,
    "identity": Rotation.identity,
})


def _vector(value: Any, *, size: int = 3) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"expected a finite {size}-element vector")
    return vector


def _unit(value: Any, *, size: int = 3) -> np.ndarray:
    vector = _vector(value, size=size)
    scale = float(np.max(np.abs(vector)))
    if scale == 0:
        raise ValueError("a zero vector has no direction; supply a nonzero measured direction")
    scaled = vector / scale
    return scaled / np.linalg.norm(scaled)


def _angle_deg(a: Any, b: Any) -> float:
    first, second = _unit(a), _unit(b)
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(first, second)),
                                      np.dot(first, second))))


def _rotate(q_xyzw: Any, vector: Any) -> np.ndarray:
    return Rotation.from_quat(_unit(q_xyzw, size=4)).apply(_vector(vector))


def _transform_point(translation: Any, q_xyzw: Any, point: Any) -> np.ndarray:
    return _vector(translation) + _rotate(q_xyzw, point)


_GEOMETRY = _Table("geom", {
    "unit": _unit, "angle_deg": _angle_deg, "rotate": _rotate,
    "transform_point": _transform_point,
})

_MATH = _Table("math", {name: getattr(math, name) for name in (
    "sqrt", "sin", "cos", "tan", "asin", "acos", "atan2", "degrees", "radians",
    "hypot", "floor", "ceil", "fabs", "isfinite", "pi",
)})


def _namespace() -> dict[str, Any]:
    """A fresh evaluation namespace. Built per call; never reused."""

    # float and int appear in every hand-written quaternion conversion and
    # are pure numeric conversions of values the block already holds. abs,
    # max, min and round are the same thing one step further: a policy that
    # writes numeric Python writes them by reflex, and five live refusals
    # in recorded runs were nothing but their absence. They are deliberately not
    # added to RESERVED_NAMES, so a block or binding that already uses one of
    # these words as its own name keeps working.
    return {"__builtins__": {}, "np": _NUMPY, "Rotation": _ROTATION,
            "pi": float(np.pi), "float": float, "int": int,
            "abs": abs, "max": max, "min": min, "round": round,
            "geom": _GEOMETRY, "math": _MATH}
