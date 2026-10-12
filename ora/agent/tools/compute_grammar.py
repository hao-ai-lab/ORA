"""The restricted evaluator's grammar: one allow-list, read by the evaluator and published.

``compute.validate_source`` refuses whatever these tables do not admit, and
``grammar()`` renders the same tables as the statement of supported constructs
that ``compute_grammar_from_evaluator_v1`` appends to the published ``compute``
description, so the two cannot drift. EXP11A g016: GPT-6 Astra wrote
``p[2] = ...`` (7x), ``p[2] += ...`` (1x) and ``np.diag`` (1x) against a
description that named none of them; the refusals named an AST class.

No numpy import here: composition reads this module.
"""

from __future__ import annotations

import ast

MAX_SOURCE_BYTES = 4096
MAX_STATEMENTS = 24
MAX_OUTPUTS = 12
MAX_VALUE_ELEMENTS = 64
MAX_POW_EXPONENT = 8
# A published surface states a bound the schema can carry. The evaluator had no
# binding ceiling because nothing published it; a name-per-entry input shape can
# be given one, and a refusal here is cheaper than a namespace the block cannot
# keep straight.
MAX_BINDINGS = 16
CHILD_DATA_LIMIT_BYTES = 512 * 1024 * 1024
CHILD_CPU_SECONDS = 2
WALL_TIMEOUT_S = 5.0
WORKER_START_TIMEOUT_S = 30.0

# Names the evaluator binds itself; a block may not shadow them.
RESERVED_NAMES = frozenset({"np", "Rotation", "pi", "float", "int", "geom", "math"})


# --------------------------------------------------------------------------
# Layer 1 — AST allowlist
# --------------------------------------------------------------------------

_ALLOWED_NODES: frozenset[type[ast.AST]] = frozenset(
    {
        ast.Module, ast.Assign, ast.Expr,
        ast.Name, ast.Load, ast.Store, ast.Constant,
        ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
        ast.Call, ast.keyword, ast.Attribute,
        ast.Subscript, ast.Slice, ast.List, ast.Tuple,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
        ast.Pow, ast.MatMult,
        ast.UAdd, ast.USub, ast.Not,
        ast.And, ast.Or,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        # `n if n[2] > 0 else -n` is how a block fixes the sign of a normal it
        # just derived. It is a pure expression with no branch to take and no
        # name bound conditionally, unlike the `if` statement below it, and
        # refusing it sent live blocks back for a turn to rediscover
        # `np.sign`. The statement stays refused; this is its replacement.
        ast.IfExp,
    }
)

# Rejected explicitly so the error names the construct instead of saying
# "unsupported node", which a policy cannot act on.
_NAMED_REJECTIONS: dict[type[ast.AST], str] = {
    ast.Import: "import", ast.ImportFrom: "import",
    ast.FunctionDef: "function definition", ast.AsyncFunctionDef: "function definition",
    ast.Lambda: "lambda", ast.ClassDef: "class definition",
    ast.For: "for loop", ast.AsyncFor: "for loop", ast.While: "while loop",
    ast.ListComp: "comprehension", ast.SetComp: "comprehension",
    ast.DictComp: "comprehension", ast.GeneratorExp: "comprehension",
    ast.With: "with block", ast.AsyncWith: "with block",
    ast.Try: "try block", ast.Raise: "raise",
    ast.Global: "global", ast.Nonlocal: "nonlocal", ast.Delete: "del",
    ast.Assert: "assert", ast.Yield: "yield", ast.YieldFrom: "yield",
    ast.Await: "await", ast.JoinedStr: "f-string", ast.NamedExpr: "walrus assignment",
    ast.Starred: "star unpacking", ast.Dict: "dict literal", ast.Set: "set literal",
    ast.If: "if statement",
    ast.AugAssign: "augmented assignment (x += v)",
}

#: What to write instead, for refusals the generic advice does not answer.
#: A block that only chooses between two numbers has admitted spellings, and
#: one refused without them spends a turn rediscovering `np.sign`.
_REJECTION_RECOVERY: dict[type[ast.AST], str] = {
    ast.If: "Write the branch as one expression: `a if condition else b`, "
            "np.where(condition, a, b), or arithmetic such as v * np.sign(v[2]).",
    ast.AugAssign: "Assign the new value to a name: `x = x + v`.",
}

#: The refusal for ``x[i] = v`` and ``x.a = v``. A block builds a changed array
#: as a new value; in-place element assignment stays out because it silently
#: truncates a float into an integer array (``p = np.array([1, 2, 3]);
#: p[2] = 0.5`` leaves 0) and makes the base both an input and an output.
TARGET_RECOVERY = (
    "Build the changed value as a new array, e.g. `q = np.array([p[0], p[1], z])` "
    "or `q = np.where(np.arange(3) == 2, z, p)`."
)


#: Attributes reachable on a *value* the block has already produced, as opposed
#: to the table entries reached through ``np`` and ``Rotation``.
#:
#: This is an allowlist for the same reason layer 2 is one. Rejecting only
#: dunder attributes leaves every other attribute of every reachable object in
#: play, and two of those break stated contracts: ``ndarray.tofile``,
#: ``ndarray.dump`` and ``ndarray.ctypes`` are file and memory access reached
#: from an instance rather than from the module table, and
#: ``Rotation.identity().random()`` reaches the classmethod through the instance
#: and returns a different answer every call. An evaluator whose receipt must be
#: re-executable offline cannot expose a random draw, and a denylist of one
#: prefix cannot keep up with a dependency's attribute surface.
#:
#: Blocking those three by empty builtins alone is not enough: numpy only fails
#: them because its internal import raises ``KeyError: '__import__'``, which is
#: an accident of the dependency rather than a decision here, and it reaches the
#: policy as a message it cannot act on.
_INSTANCE_ATTRIBUTES: frozenset[str] = frozenset({
    # numeric ndarray shape and value access
    "T", "shape", "ndim", "size", "real", "imag",
    "reshape", "ravel", "flatten", "transpose", "squeeze", "copy",
    "tolist", "item", "astype", "diagonal", "trace",
    "sum", "mean", "min", "max", "argmin", "argmax", "prod",
    "std", "var", "cumsum", "dot", "round", "clip", "conj",
    "any", "all", "sort", "argsort", "nonzero", "take", "repeat",
    # Rotation: the deterministic conversions and compositions
    "apply", "inv", "magnitude", "approx_equal",
    "as_quat", "as_matrix", "as_rotvec", "as_euler", "as_mrp", "as_davenport",
})


#: How each admitted node reads in the published grammar: (group, token). Every
#: admitted node has an entry (a test enforces it), so adding a node without
#: saying so fails. Group order is the published order.
_NODE_LABELS: dict[type[ast.AST], tuple[str, str]] = {
    ast.Assign: ("statements", "assignment to a plain name or a tuple of plain names (x = ..., a, b = ...)"),
    ast.Expr: ("statements", "a bare expression (the last one is returned as result)"),
    ast.Add: ("operators", "+"), ast.Sub: ("operators", "-"), ast.Mult: ("operators", "*"),
    ast.Div: ("operators", "/"), ast.FloorDiv: ("operators", "//"), ast.Mod: ("operators", "%"),
    ast.Pow: ("operators", "**"), ast.MatMult: ("operators", "@"),
    ast.UAdd: ("operators", "unary +"), ast.USub: ("operators", "unary -"), ast.Not: ("operators", "not"),
    ast.And: ("operators", "and"), ast.Or: ("operators", "or"),
    ast.Eq: ("operators", "=="), ast.NotEq: ("operators", "!="), ast.Lt: ("operators", "<"),
    ast.LtE: ("operators", "<="), ast.Gt: ("operators", ">"), ast.GtE: ("operators", ">="),
    ast.IfExp: ("forms", "a if condition else b"),
    ast.Call: ("forms", "calls to listed functions"), ast.keyword: ("forms", "keyword arguments"),
    ast.Attribute: ("forms", "listed attributes"), ast.Subscript: ("forms", "indexing x[i], x[i, j]"),
    ast.Slice: ("forms", "slices x[a:b]"), ast.List: ("forms", "list literals"),
    ast.Tuple: ("forms", "tuple literals"), ast.Name: ("forms", "names"),
    ast.Constant: ("forms", "number, boolean and short string literals"),
    ast.BinOp: ("", ""), ast.UnaryOp: ("", ""), ast.BoolOp: ("", ""), ast.Compare: ("", ""),
    ast.Module: ("", ""), ast.Load: ("", ""), ast.Store: ("", ""),
}


def grammar() -> dict[str, list[str] | int]:
    """The evaluator's admitted and refused constructs, as published values."""

    groups: dict[str, list[str]] = {"statements": [], "operators": [], "forms": []}
    for kind in sorted(_ALLOWED_NODES, key=lambda node: node.__name__):
        group, token = _NODE_LABELS[kind]
        if group:
            groups[group].append(token)
    return {
        **{name: sorted(tokens) for name, tokens in groups.items()},
        "not_available": sorted({*_NAMED_REJECTIONS.values(), "subscript or attribute assignment (x[i] = v)"}),
        "reserved_names": sorted(RESERVED_NAMES),
        "value_attributes": sorted(_INSTANCE_ATTRIBUTES),
        "maximum_absolute_exponent": MAX_POW_EXPONENT,
    }


def grammar_text() -> str:
    """The published statement of supported constructs, generated from ``grammar()``."""

    value = grammar()
    return (
        "Evaluator grammar (generated from the evaluator's own allow-list). "
        f"Statements: {'; '.join(value['statements'])}. "
        f"Operators: {' '.join(value['operators'])}. "
        f"Forms: {', '.join(value['forms'])}. "
        f"Not available: {', '.join(value['not_available'])}. "
        f"Reserved names, never assigned or bound: {', '.join(value['reserved_names'])}. "
        f"Attributes of a value the block produced: {', '.join(value['value_attributes'])}. "
        f"The exponent of ** is a numeric literal, sign allowed, of magnitude at most "
        f"{value['maximum_absolute_exponent']}."
    )
