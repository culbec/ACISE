# `solutions/flow_control/model.py`

Status: owned by the user (step 5). Spec: [implementation-backbone.md](../implementation-backbone.md). Method: the main plan's *(b) Hand-written flow-control program*.

## Purpose

The hand-written program of part (b): at most `N` if/elif statements over the selected features, distilled from the GP output and domain knowledge.

## What Claude provides

Nothing in this file. The user writes it. The supporting pieces are elsewhere:

- `gp/simplify.py`: simplification, pretty-printing, and recurring-structure counts.
- `training/cv.py`: runs the program on every fold. The program only needs to fit the `fit_predict` shape.
- `metrics`, `stats`: identical scoring and comparison as for (a).

## Contract the CV loop expects

```python
def predict(build, thr) -> bool        # True = predict failure; at most N if/elif statements
THRESHOLDS: dict[str, ...]             # named parameters with raw-unit defaults and search grids
```

- The structure is fixed after it is validated on the dev segment. Only `thr` is tuned, per fold, on that fold's train.
- The statement count is reported with the results.
- Reasoning for each domain-knowledge rule goes in the 04_handwritten notebook, not in this file.

## Open points

- A per-fold threshold grid-search helper (quantiles of the raw feature, scored with `metrics`). Say if you want it written, or if you prefer to write it yourself.
