# Compute usage v2 with routing v1

Compute evaluates numbers already supplied by the policy. It does not obtain
new observations or execute robot commands. Routing v1 requires a call before
using a newly calculated number to choose or parameterize an action, including
a single offset or angle. It preserves direct copying of valid tool values,
reuse while inputs and assumptions are unchanged, and selection of constants
that need no calculation. Image, depth and calibration queries belong to
`geometry_query`; named current-pixel measurements belong to `mark_point`.
The exact model-visible description is the `compute` entry in
[`tool_routing_v1/shared-help.json`](../../ora/policy/tool_surfaces/tool_routing_v1/shared-help.json).
The evaluator, input schema and output schema retain their v2 behavior. That
file also appends a four-sentence copy of the rule to `move_pose`, `move_to`,
`rotate_pitch` and `rotate_wrist`; the current LIBERO, RoboTwin and Dojo
surfaces replace that copy with one sentence pointing to
`compute`'s Use when, which stays the rule's only owner
([reference](../reference/public-interfaces.md#tool-description-consistency)).

## Prebound names

No import statement is needed: the evaluator already provides `np`, `Rotation`,
`math`, `geom`, `pi`, `float` and `int`. These are restricted numeric facades;
the full NumPy/SciPy/math module environment is not exposed. Inputs in `bindings`
are automatically available as variables; numeric lists become arrays.

| Operation | Expression |
|---|---|
| Distance between points | `np.linalg.norm(b - a)` |
| Degrees to radians | `math.radians(angle)` |
| Normalize a nonzero direction | `geom.unit(v)` |
| Unsigned angle, degrees | `geom.angle_deg(a, b)` |
| Rotate a vector, quaternion xyzw | `geom.rotate(q, v)` |
| Transform a point | `geom.transform_point(t, q, p)` |
| Euler angles to quaternion | `Rotation.from_euler('xyz', angles, degrees=True).as_quat()` |

The point transform is `t + Rotation(q).apply(p)`. The caller supplies frame
identities and consistent units. Normalization scales before taking a norm to
avoid overflow/underflow; a zero vector is rejected because it has no direction.
No helper turns an assumed geometric relationship into a measured one.

## Complete call

```json
{
  "bindings": [
    {"name": "a", "value": [0.1, 0.2, 0.3]},
    {"name": "b", "value": [0.4, 0.6, 0.3]}
  ],
  "expressions": "distance_m = np.linalg.norm(b - a)",
  "outputs": ["distance_m"]
}
```

The numeric result is `distance_m = 0.5`. Replace these illustrative values with
actual inputs. Optional `from` on a binding records a source label; it does not
verify that source. Each call starts fresh, so pass inputs again on later calls.

`expressions` is one JSON string, with line breaks encoded as `\n`. Do not put
Markdown fences, import, def, print or return statements inside it. `outputs`
selects final variable names; a final bare expression is named `result`.
Omitting the selection retains the existing automatic numeric-variable output.
After JSON decoding, statement separators must be real newlines or semicolons;
literal backslash-n characters between Python statements cause a syntax error.
Correct that source and retry instead of substituting a mental answer.

## Limit errors

- Output count/size: select only the final values. A temporary array may be
  larger than a returned value when it stays within execution memory bounds;
  unselected values are not serialized. Alternatively return a slice or summary.
- Unknown name/import: use a prebound name or supply the value in `bindings`.
- Non-finite number: check zero division, magnitude and units. For vector
  directions and angles, prefer the scaled helpers.
- Source/time/memory limit: shorten the expression with the prebound operators
  or split independent calculations. Repeating identical input does not help.

The unchanged bounds are 4096 source bytes, 24 statements, 16 published input
bindings, 12 returned values, 64 elements and two dimensions per returned value,
512 MiB child data limit, two CPU seconds and five seconds for the child-result
wait. Selection does not bypass these bounds for a returned value. No arbitrary
imports, filesystem/network access or persistent notebook state are available.

## Validation and activation

The two model-visible JSON examples live in
`ora/policy/tool_surfaces/compute_v2/examples.json`; tests execute the
exact Help examples through every applicable surface. Helper math, output
selection and isolation tests run offline in the actual evaluator worker.
These checks prove delivery/execution contracts, not fewer model errors or
better task success. Historical runs retain their original source.

All eight applicable catalog selectors activate the current routing generation with Compute v2
helpers and output selection. Core and prompt bundle selections retain B2's
existing defaults; the separate Compute-first prompt treatment stays on its
source branch. The Dojo v4 consumer and source-refreshed Panda routing
bindings are included; no unrelated Compute-first prompt treatment is
selected.
