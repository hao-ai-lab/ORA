"""The SDK a composed tool body may call, and nothing else.

Every name a body can reach is listed in :data:`HOST`. Three kinds exist
(see :mod:`.starlark_runtime`):

* **actions** (``sdk.gripper``, ``sdk.move``) are primitive robot actions. Only
  ``execute()`` may call them; each one suspends the body and is dispatched by
  the trusted executor through the Supervisor exactly like the matching
  atomic tool. A body never addresses a backend: :func:`lower_motion_v3`
  turns a request into the ``ora-motion-action.v3`` envelope every
  maintained profile already executes.
* ``sdk.call(stage=..., arguments=...)`` runs the composed tool a declared
  ``tool`` stage pins as one step: its ``plan()`` on the latest observation,
  then its actions dispatched as that stage's attempts; the body gets back one
  outcome summarizing the nested receipt. Nesting is at most
  :data:`~.declaration.MAX_NESTING_DEPTH` deep.
* **queries** read deployable public state and are journaled, so a replay
  never reads the world twice. ``sdk.read(tool, arguments)`` calls one of the
  policy's read tools (:data:`READ_TOOLS`) with the same validation and result
  the policy would get: in ``plan()`` on the current observation, in
  ``execute()`` on a fresh capture taken after the last dispatched action, so
  a body can re-measure mid-sequence. ``sdk.robot_state`` reads one arm;
  ``sdk.grasp.target`` and ``sdk.grasp.precommit_probe`` (call time) wrap the
  closed-skill candidate commitment and the read-only pre-commit motion probe;
  ``sdk.capture`` (execution time) captures an unpublished observation under a
  role; and **check queries** (``sdk.check.grasp``) measure a declared
  read-only stage. A check query's result is trusted host evidence: the
  executor records it as that stage's outcome, so a body decides flow but never
  authors a measurement it reports.
* **pure** functions are deterministic data constructors: the declaration
  header (``composed_tool``, ``stage`` ...), the results a body returns
  (``sdk.commit``, ``sdk.complete`` ...) and vector/pose arithmetic, because
  Starlark has no numpy.

Action outcomes come back to the body as ``{"status", "reason",
"measurements"}`` with ``status`` one of :data:`ACTION_OUTCOMES`; the body
branches on that and on nothing it could not observe.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ora.contracts.action import MOTION_ACTION_V3_SCHEMA, validate_runtime_action

from .declaration import ARM_PROFILES, DECLARATION_CONSTRUCTORS, STAGE_PRIMITIVES  # noqa: F401 - re-exported
from .starlark_runtime import HostKind

INTERACTIONS = ("free_space", "contact_approach", "precision")
GRIPPER_DIRECTIVES = ("open", "close")
#: What a dispatched primitive action can come back as. ``terminated`` is not
#: here: the executor ends the body itself when the episode reaches its
#: terminal boundary, so a body never sees it.
ACTION_OUTCOMES = ("executed", "not_reached", "unavailable", "refused")

RESULT_KINDS = ("commit", "refuse", "complete", "abort")
_CODE_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")
_MAX_QUANTITY_CHARS = 64
_MAX_UNIT_CHARS = 16
MAX_MEASUREMENT_VECTOR = 16


class SdkCallError(ValueError):
    """A body called an SDK function with arguments it does not accept."""


# ---- action signatures ---------------------------------------------------------------------------
# The Python signatures below are the call contract of each action; a request
# is bound against them before validation, so positional and keyword spellings
# normalize to one form.


def _gripper_signature(arm, directive, *, stage):  # noqa: ANN001, ANN202 - signature only
    raise NotImplementedError


def _move_signature(arm, pose, *, interaction, stage):  # noqa: ANN001, ANN202 - signature only
    raise NotImplementedError


def _call_signature(arguments=None, *, stage):  # noqa: ANN001, ANN202 - signature only
    raise NotImplementedError


ACTION_SIGNATURES: dict[str, Callable[..., Any]] = {
    "sdk.gripper": _gripper_signature,
    "sdk.move": _move_signature,
    "sdk.call": _call_signature,
}
#: The primitive each action symbol drives; a declared stage names one of these
#: (or ``check`` for a read-only stage).
ACTION_PRIMITIVES = {"sdk.gripper": "gripper", "sdk.move": "move", "sdk.call": "tool"}
#: The policy read tools a body may call through ``sdk.read``; their results are what the policy would get.
READ_TOOLS = ("observe", "mark_point", "geometry_query", "compute", "vision_segment", "vision_ground",
              "grasp_plan", "align_plan")


def _robot_state_signature(arm):  # noqa: ANN001, ANN202 - signature only
    raise NotImplementedError


def _grasp_target_signature(candidate_id):  # noqa: ANN001, ANN202 - signature only
    raise NotImplementedError


def _precommit_probe_signature(target):  # noqa: ANN001, ANN202 - signature only
    raise NotImplementedError


def _capture_signature(role):  # noqa: ANN001, ANN202 - signature only
    raise NotImplementedError


def _check_grasp_signature(target, *, stage, hold=False):  # noqa: ANN001, ANN202 - signature only
    raise NotImplementedError


def _ask_signature(question, *, images=(), values=None):  # noqa: ANN001, ANN202 - signature only
    raise NotImplementedError


def _read_signature(tool, arguments=None):  # noqa: ANN001, ANN202 - signature only
    raise NotImplementedError


QUERY_SIGNATURES: dict[str, Callable[..., Any]] = {
    "sdk.robot_state": _robot_state_signature,
    "sdk.grasp.target": _grasp_target_signature,
    "sdk.grasp.precommit_probe": _precommit_probe_signature,
    "sdk.capture": _capture_signature,
    "sdk.check.grasp": _check_grasp_signature,
    "sdk.ask": _ask_signature,
    "sdk.read": _read_signature,
}
#: Queries only plan() may use, and queries only execute() may use.
PLAN_QUERIES = frozenset({"sdk.grasp.target", "sdk.grasp.precommit_probe"})
EXECUTE_QUERIES = frozenset({"sdk.capture", "sdk.check.grasp", "sdk.ask"})
#: Check queries: their result is the outcome of the read-only stage they name.
CHECK_QUERIES = frozenset({"sdk.check.grasp"})
CAPTURE_ROLES = ("post_close_before_lift",)


def bind_call(symbol: str, signature: Callable[..., Any], args: Sequence[Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        bound = inspect.signature(signature).bind(*args, **kwargs)
    except TypeError as exc:
        raise SdkCallError(f"{symbol}: {exc}") from None
    return dict(bound.arguments)


def validate_action_request(
    symbol: str,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    arms: Sequence[str],
    stages: Mapping[str, str],
) -> dict[str, Any]:
    """Normalize one suspended action into ``{primitive, arm, stage, ...}``.

    ``stages`` maps each declared stage to its primitive; an action must name a
    declared stage of the same primitive.
    """

    if symbol not in ACTION_SIGNATURES:
        raise SdkCallError(f"{symbol} is not an SDK action")
    call = bind_call(symbol, ACTION_SIGNATURES[symbol], args, kwargs)
    primitive = ACTION_PRIMITIVES[symbol]
    stage = call["stage"]
    if stage not in stages:
        raise SdkCallError(f"{symbol}: stage {stage!r} is not declared")
    if stages[stage] != primitive:
        raise SdkCallError(f"{symbol}: stage {stage!r} is declared for {stages[stage]}, not {primitive}")
    if primitive == "tool":
        arguments = call.get("arguments")
        if arguments is not None and not isinstance(arguments, Mapping):
            raise SdkCallError(f"{symbol}: arguments must be a dict of the nested tool's inputs")
        return {"primitive": "tool", "stage": stage, "arguments": dict(arguments or {})}
    arm = call["arm"]
    if arm not in arms:
        raise SdkCallError(f"{symbol}: arm {arm!r} is not one of {list(arms)}")
    request: dict[str, Any] = {"primitive": primitive, "stage": stage, "arm": arm}
    if primitive == "gripper":
        if call["directive"] not in GRIPPER_DIRECTIVES:
            raise SdkCallError(f"{symbol}: directive must be one of {list(GRIPPER_DIRECTIVES)}")
        request["directive"] = call["directive"]
    else:
        if call["interaction"] not in INTERACTIONS:
            raise SdkCallError(f"{symbol}: interaction must be one of {list(INTERACTIONS)}")
        request["pose"] = _pose(call["pose"], f"{symbol} pose")
        request["interaction"] = call["interaction"]
    return request


def lower_motion_v3(
    request: Mapping[str, Any], *, action_id: str, observation_id: str, observation_sha256: str,
) -> dict[str, Any]:
    """The validated motion v3 envelope that performs one normalized request."""

    arm = request["arm"]
    if request["primitive"] == "gripper":
        action, arguments = "set_gripper", {"targets": {arm: request["directive"]}}
    else:
        action = "move_pose"
        arguments = {"targets": {arm: request["pose"]}, "interactions": {arm: request["interaction"]}}
    return validate_runtime_action({
        "schema": MOTION_ACTION_V3_SCHEMA,
        "observation_id": observation_id,
        "action_id": action_id,
        "action": action,
        "arguments": arguments,
        "bound_observation_sha256": observation_sha256,
    })


#: The largest measurements the trusted executor attaches to one dispatched
#: action of each primitive (orchestrator/action_executor.py: endpoint_residual
#: and RoboDojo motion_progress on a pose stage; gripper_stage.py: per-arm
#: gripper_outcome on a gripper stage), at their widest shapes. The static
#: receipt-budget check sizes a worst-case receipt from these.
WORST_CASE_OUTCOME_MEASUREMENTS: dict[str, list[dict[str, Any]]] = {
    "move": [
        {"quantity": "endpoint_residual", "value": {
            "reason": "endpoint_residual_exceeded_on_the_commanded_arm",
            "errors_by_arm": {
                arm: {"translation_m": 0.0123456789, "orientation_rad": 0.0123456789, "reached": False}
                for arm in ("left", "right")
            },
        }},
        {"quantity": "motion_progress", "value": {
            arm: {"commanded_translation_m": 0.123456789, "achieved_translation_m": 0.123456789,
                  "fraction": 0.123456789, "stopped_reason": "planner_rejected_the_remaining_segment"}
            for arm in ("left", "right")
        }},
    ],
    "gripper": [
        {"quantity": "gripper_outcome", "value": {
            arm: {"directive": "close", "applied": 0.0, "measured_opening": 0.0123456789,
                  "outcome": "gripper_stalled_before_target", "stall_ticks": 123}
            for arm in ("left", "right")
        }},
    ],
    # executor: the summary of a nested composed tool, one per tool stage attempt.
    "tool": [
        {"quantity": "nested_tool", "value": {
            "tool": "t" * 40, "version": "c99", "status": "aborted", "stop_reason": "s" * 64,
            "dispatched_actions": 24, "stages": {f"stage_{index:02d}": "not_reached" for index in range(8)},
        }},
    ],
    "check": [],
}


# ---- pure results ---------------------------------------------------------------------------------


def _code(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or not set(value) <= _CODE_CHARS or not value[0].isalpha():
        raise SdkCallError(f"{where} must be a lowercase snake_case code, got {value!r}")
    return value


def measure(stage: Any, quantity: Any, value: Any, unit: Any = None) -> dict[str, Any]:
    """One named measurement the body attributes to a declared stage."""

    if not isinstance(stage, str) or not stage:
        raise SdkCallError("sdk.measure: stage must be a declared stage name")
    quantity = _code(quantity, "sdk.measure quantity")
    if len(quantity) > _MAX_QUANTITY_CHARS:
        raise SdkCallError(f"sdk.measure: quantity longer than {_MAX_QUANTITY_CHARS} characters")
    if not (value is None or isinstance(value, (bool, int, float, str)) or _is_vector(value)):
        raise SdkCallError("sdk.measure: value must be a scalar, a string or a list of numbers")
    if isinstance(value, list) and len(value) > MAX_MEASUREMENT_VECTOR:
        raise SdkCallError(f"sdk.measure: a vector value has at most {MAX_MEASUREMENT_VECTOR} numbers")
    if isinstance(value, str) and len(value) > _MAX_QUANTITY_CHARS:
        raise SdkCallError(f"sdk.measure: string values are codes of at most {_MAX_QUANTITY_CHARS} characters")
    if unit is not None and (not isinstance(unit, str) or not unit or len(unit) > _MAX_UNIT_CHARS):
        raise SdkCallError("sdk.measure: unit must be a short string or None")
    item = {"kind": "measurement", "stage": stage, "quantity": quantity, "value": value}
    if unit is not None:
        item["unit"] = unit
    return item


def _measurements(items: Any, where: str) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        raise SdkCallError(f"{where}: measurements must be a list of sdk.measure results")
    for item in items:
        if not isinstance(item, Mapping) or item.get("kind") != "measurement":
            raise SdkCallError(f"{where}: measurements must come from sdk.measure")
    return [dict(item) for item in items]


def commit(evidence: Any = None, **data: Any) -> dict[str, Any]:
    """plan() result: the committed targets ``execute()`` receives as ``c``.

    ``evidence`` maps header-declared evidence names (``composed_tool(evidence=
    [...])``) to plain data a plan-time query returned; each is echoed in the
    immediate result and in the receipt under its own name.
    """

    if evidence is not None and not isinstance(evidence, Mapping):
        raise SdkCallError("sdk.commit: evidence must be a dict of declared names")
    return {"kind": "commit", "data": dict(data), "evidence": dict(evidence or {})}


def refuse(
    code: Any, message: Any, field: Any = None, retryable: Any = True, constraints: Any = None,
) -> dict[str, Any]:
    """plan() result: a structured refusal; no action is submitted.

    ``constraints`` is plain data a host query already returned (for example a
    probe's planning summary); it travels unchanged in the ToolError.
    """

    if not isinstance(message, str) or not message.strip():
        raise SdkCallError("sdk.refuse: message must be non-empty text")
    if field is not None and not isinstance(field, str):
        raise SdkCallError("sdk.refuse: field must be an input name or None")
    if not isinstance(retryable, bool):
        raise SdkCallError("sdk.refuse: retryable must be a bool")
    if constraints is not None and not isinstance(constraints, Mapping):
        raise SdkCallError("sdk.refuse: constraints must be a dict or None")
    return {
        "kind": "refuse", "code": _code(code, "sdk.refuse code"), "message": message.strip(),
        "field": field, "retryable": retryable, "constraints": dict(constraints or {}),
    }


def complete(measurements: Any = None) -> dict[str, Any]:
    """execute() result: the body finished its sequence."""

    return {"kind": "complete", "measurements": _measurements(measurements or [], "sdk.complete")}


def abort(code: Any, measurements: Any = None) -> dict[str, Any]:
    """execute() result: the body stopped early; ``code`` becomes the receipt's stop reason."""

    return {
        "kind": "abort", "code": _code(code, "sdk.abort code"),
        "measurements": _measurements(measurements or [], "sdk.abort"),
    }


def check_hold(verification: Any, judgment: Any = None) -> dict[str, Any]:
    """``sdk.check.hold``: the fused hold verdict of a verify block (and a model judgement)."""

    from .hold import hold_verdict

    if verification is not None and not isinstance(verification, Mapping):
        raise SdkCallError("sdk.check.hold: verification must be a verify block or None")
    if judgment is not None and not isinstance(judgment, Mapping):
        raise SdkCallError("sdk.check.hold: judgment must be an sdk.ask answer or None")
    return hold_verdict(verification, judgment)


# ---- pure arithmetic ------------------------------------------------------------------------------


def _is_vector(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
    )


def _vector(value: Any, where: str, length: int | None = None) -> list[float]:
    if not _is_vector(value) or (length is not None and len(value) != length):
        size = f" of length {length}" if length is not None else ""
        raise SdkCallError(f"{where} must be a list{size} of numbers")
    out = [float(item) for item in value]
    if not all(math.isfinite(item) for item in out):
        raise SdkCallError(f"{where} must be finite")
    return out


def _pose(value: Any, where: str) -> dict[str, list[float]]:
    if not isinstance(value, Mapping) or set(value) != {"position_m", "orientation_xyzw"}:
        raise SdkCallError(f"{where} must be {{position_m, orientation_xyzw}}")
    position = _vector(value["position_m"], f"{where}.position_m", 3)
    orientation = _vector(value["orientation_xyzw"], f"{where}.orientation_xyzw", 4)
    norm = math.sqrt(sum(item * item for item in orientation))
    if not 0.98 <= norm <= 1.02:
        raise SdkCallError(f"{where}.orientation_xyzw must be a unit quaternion")
    return {"position_m": position, "orientation_xyzw": [item / norm for item in orientation]}


def vec_add(a: Any, b: Any) -> list[float]:
    left, right = _vector(a, "sdk.vec.add a"), _vector(b, "sdk.vec.add b")
    if len(left) != len(right):
        raise SdkCallError("sdk.vec.add: lengths differ")
    return [x + y for x, y in zip(left, right)]


def vec_sub(a: Any, b: Any) -> list[float]:
    left, right = _vector(a, "sdk.vec.sub a"), _vector(b, "sdk.vec.sub b")
    if len(left) != len(right):
        raise SdkCallError("sdk.vec.sub: lengths differ")
    return [x - y for x, y in zip(left, right)]


def vec_scale(a: Any, k: Any) -> list[float]:
    if isinstance(k, bool) or not isinstance(k, (int, float)) or not math.isfinite(k):
        raise SdkCallError("sdk.vec.scale: k must be a finite number")
    return [x * float(k) for x in _vector(a, "sdk.vec.scale a")]


def vec_norm(a: Any) -> float:
    return math.sqrt(sum(x * x for x in _vector(a, "sdk.vec.norm a")))


def vec_dot(a: Any, b: Any) -> float:
    left, right = _vector(a, "sdk.vec.dot a"), _vector(b, "sdk.vec.dot b")
    if len(left) != len(right):
        raise SdkCallError("sdk.vec.dot: lengths differ")
    return sum(x * y for x, y in zip(left, right))


def vec_cross(a: Any, b: Any) -> list[float]:
    x1, y1, z1 = _vector(a, "sdk.vec.cross a", 3)
    x2, y2, z2 = _vector(b, "sdk.vec.cross b", 3)
    return [y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - y1 * x2]


def _unit_quaternion(value: Any, where: str) -> list[float]:
    q = _vector(value, where, 4)
    norm = math.sqrt(sum(item * item for item in q))
    if not 0.98 <= norm <= 1.02:
        raise SdkCallError(f"{where} must be a unit quaternion")
    return [item / norm for item in q]


def quat_mul(a: Any, b: Any) -> list[float]:
    """The rotation ``a`` after ``b`` (Hamilton product, xyzw)."""

    x1, y1, z1, w1 = _unit_quaternion(a, "sdk.quat.mul a")
    x2, y2, z2, w2 = _unit_quaternion(b, "sdk.quat.mul b")
    return [w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2]


def quat_axis_angle(axis: Any, angle_rad: Any) -> list[float]:
    """The rotation of ``angle_rad`` about the world-frame ``axis`` (xyzw)."""

    vector = _vector(axis, "sdk.quat.axis_angle axis", 3)
    norm = math.sqrt(sum(item * item for item in vector))
    if norm < 1e-9 or isinstance(angle_rad, bool) or not isinstance(angle_rad, (int, float)) \
            or not math.isfinite(angle_rad):
        raise SdkCallError("sdk.quat.axis_angle needs a nonzero axis and a finite angle")
    half = float(angle_rad) / 2.0
    return [item / norm * math.sin(half) for item in vector] + [math.cos(half)]


def quat_rotate(q: Any, v: Any) -> list[float]:
    """``v`` rotated by ``q``."""

    x, y, z, w = _unit_quaternion(q, "sdk.quat.rotate q")
    vx, vy, vz = _vector(v, "sdk.quat.rotate v", 3)
    # v' = v + 2w (q x v) + 2 q x (q x v), with q the vector part.
    cx, cy, cz = y * vz - z * vy, z * vx - x * vz, x * vy - y * vx
    ex, ey, ez = y * cz - z * cy, z * cx - x * cz, x * cy - y * cx
    return [vx + 2 * (w * cx + ex), vy + 2 * (w * cy + ey), vz + 2 * (w * cz + ez)]


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SdkCallError(f"{where} must be a finite number")
    return float(value)


def math_sqrt(x: Any) -> float:
    value = _number(x, "sdk.math.sqrt x")
    if value < 0:
        raise SdkCallError("sdk.math.sqrt of a negative number")
    return math.sqrt(value)


def math_atan2(y: Any, x: Any) -> float:
    return math.atan2(_number(y, "sdk.math.atan2 y"), _number(x, "sdk.math.atan2 x"))


def math_acos(x: Any) -> float:
    return math.acos(max(-1.0, min(1.0, _number(x, "sdk.math.acos x"))))


def pose_translate(pose: Any, offset_m: Any) -> dict[str, list[float]]:
    """The same orientation at ``position + offset_m`` (world frame, metres)."""

    base = _pose(pose, "sdk.pose.translate pose")
    offset = _vector(offset_m, "sdk.pose.translate offset_m", 3)
    return {"position_m": vec_add(base["position_m"], offset), "orientation_xyzw": base["orientation_xyzw"]}


PURE: dict[str, Callable[..., Any]] = {
    **DECLARATION_CONSTRUCTORS,
    "sdk.commit": commit,
    "sdk.refuse": refuse,
    "sdk.complete": complete,
    "sdk.abort": abort,
    "sdk.measure": measure,
    "sdk.vec.add": vec_add,
    "sdk.vec.sub": vec_sub,
    "sdk.vec.scale": vec_scale,
    "sdk.vec.norm": vec_norm,
    "sdk.pose.translate": pose_translate,
    "sdk.vec.dot": vec_dot,
    "sdk.vec.cross": vec_cross,
    "sdk.quat.mul": quat_mul,
    "sdk.quat.axis_angle": quat_axis_angle,
    "sdk.quat.rotate": quat_rotate,
    "sdk.math.sqrt": math_sqrt,
    "sdk.math.sin": lambda x: math.sin(_number(x, "sdk.math.sin x")),
    "sdk.math.cos": lambda x: math.cos(_number(x, "sdk.math.cos x")),
    "sdk.math.atan2": math_atan2,
    "sdk.math.acos": math_acos,
    "sdk.check.hold": check_hold,
}

HOST: dict[str, HostKind] = {
    **{symbol: "action" for symbol in ACTION_SIGNATURES},
    **{symbol: "query" for symbol in QUERY_SIGNATURES},
    **{symbol: "pure" for symbol in PURE},
}
