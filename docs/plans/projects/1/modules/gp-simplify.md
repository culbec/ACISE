# `gp/simplify.py`

Status: done (step 4b). Tests: in `projects/1/tests/test_engine.py`. Spec: [implementation-backbone.md](../implementation-backbone.md).

## Purpose

Turn evolved DEAP trees into something a person can read and copy from. It supports the distillation workflow for program (b) and the "pretty-printed if/else code" output of (a).

## API

```python
simplify(tree, pset) -> gp.PrimitiveTree                  # semantics-preserving rewrites
to_python(tree, pset, prep, name="predict") -> str        # if/elif/else source over raw units
motifs(trees) -> pd.DataFrame                             # feature, op, count, thr_min, thr_median, thr_max
```

`statements` comes from `gp/engine.py`. No logging needed here.

## How `simplify` works

1. Read the prefix list into nested tuples `(name, *children)`; leaves are feature names, `True`/`False`, or floats.
2. Rewrite bottom-up until nothing changes.
3. Print the nested form as a DEAP expression string and rebuild with `gp.PrimitiveTree.from_string(s, pset)`, which parses typed float constants (checked on DEAP 1.4). The result compiles with the engine's `predict` as is.

## Rewrites

- `if_then_else(c, x, x)` -> `x`; `if_then_else(True, a, b)` -> `a`; `if_then_else(False, a, b)` -> `b`.
- `if_then_else(c, True, False)` -> `c`; `if_then_else(c, False, True)` -> `not_(c)`.
- `not_(not_(x))` -> `x`; `not_` folded into comparisons (`lt` <-> `ge`, `le` <-> `gt`); `not_(True)` -> `False`.
- Constant on the left is flipped so the feature is on the left: `lt(c, f)` -> `gt(f, c)`, `le(c, f)` -> `ge(f, c)` (exact, ties included).
- `and_` / `or_` with a constant child collapse; `and_(x, x)` / `or_(x, x)` -> `x`.
- Comparisons between two constants are evaluated.
- Same feature, same op: `and_(ge(f, a), ge(f, b))` -> `ge(f, max(a, b))`, `or_` -> `min`; mirrored for `gt`, `lt`, `le`.

## `to_python`

- `if_then_else` becomes `if` / `else`; an `if_then_else` in the else branch becomes `elif`. Leaves become `return True` / `return False`.
- `and_`, `or_`, `not_` inside a condition stay inline (`and`, `or`, `not`); outside a condition they become `return <expr>`.
- A comparison of feature `f` with scaled threshold `t` prints `build["f"] >= raw`, `raw = t * data_range_ + data_min_` from the fold's `MinMaxScaler` (`prep[-1]`). Feature-vs-feature comparisons print the scaled form of both sides.
- Thresholds print with `.6g`. Rounding to meaningful values is left to the person.
- The function's docstring states its statement count.

## `motifs`

One row per (feature, op) across the given trees, after the same feature-on-the-left flip as `simplify` (`lt(c, f)` counts as `gt`). Columns: `count`, `thr_min`, `thr_median`, `thr_max` in scaled units. `motifs(trees).groupby("feature")["count"].sum()` feeds `viz.feature_freq_performers`.

Skipped: counting recurring whole sub-trees. Thresholds differ between runs, so exact sub-trees rarely repeat; add it with threshold binning if the (feature, op) table is not enough.

## Verification (`tests/test_simplify.py`)

- Property test: 200 random trees (`genHalfAndHalf`, fixed seed) give the same `predict` output before and after `simplify` on random data, and never have more statements after.
- One hand-written case per rewrite.
- `to_python` output, `exec`-ed on raw rows, matches `predict` on the scaled rows (integer-valued raw data, so no row sits on a printed threshold).
- `motifs` on two hand-made trees gives the expected counts and ranges.

## Open points

- This module is optional for the user's own program (b). It is the support the plan calls for, not a replacement for writing `model.py` by hand.
