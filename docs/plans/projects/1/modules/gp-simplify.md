# `gp/simplify.py`

Status: todo (step 4, with `gp/engine.py`). Spec: [implementation-backbone.md](../implementation-backbone.md).

## Purpose

Turn evolved DEAP trees into something a person can read and copy from. It supports the distillation workflow for program (b) and the "pretty-printed if/else code" output of (a).

## Proposed API

```python
simplify(tree) -> tree                        # semantics-preserving rewrites
statements(tree) -> int                       # same counting rule as the engine's limit
to_python(tree, features, scaler) -> str      # if/elif/else source with raw-unit thresholds
motifs(hall_of_fame) -> pd.DataFrame          # feature, direction, threshold range, count
```

## Rewrites in `simplify`

- `if_then_else(c, x, x)` becomes `x`; constant conditions collapse.
- Double negation removal, and `not_` pushed into the comparison (`not_(lt(a, b))` becomes `ge(a, b)`).
- Repeated comparisons on the same feature merge when the result is equivalent.

## Design decisions

- Rewrites must preserve predictions on every input. Property test: random trees give the same output before and after on random data.
- `statements` is imported by the engine for its limit, so there is one counting rule in the whole codebase.
- `to_python` maps scaled thresholds back to raw units with the fold's scaler. Rounding to meaningful values is left to the person, because it is a judgment about the domain.
- `motifs` counts how often each feature, comparison direction and threshold range appears across hall-of-fame programs. It feeds `viz.feature_freq` and the structure choice for (b).

## Open points

- This module is optional for the user's own program (b). It is the support the plan calls for, not a replacement for writing `model.py` by hand.
