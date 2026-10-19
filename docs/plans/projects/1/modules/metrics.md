# `metrics/metrics.py`

Status: done (step 3a). Spec: [implementation-backbone.md](../implementation-backbone.md). Tests: `projects/1/tests/test_metrics.py`.

## Purpose

One shared definition of "how good is a prediction", used by feature-selection checks, the CV loop, the GP fitness, and the final results. Used for both (a) and (b), so they are scored identically.

## API

```python
confusion(y_true, y_pred) -> Counts                   # NamedTuple tp, fp, tn, fn
rates(counts) -> tuple[float, float]                  # (tpr, fpr)
mcc(counts) -> float
scores(y_true, y_pred) -> dict[str, float]            # precision, recall, f1, mcc, accuracy, tpr, fpr, tp, fp, tn, fn
domain(y_true, y_pred) -> dict[str, float]            # pct_builds_saved, pct_failures_caught (0-100)

# baselines -> bool array. Deterministic ones take (split); seeded ones (split, seed). run_cv picks by its `seeds` argument
parrot(split)              # reads split.raw_test["last_label"]; all-pass on T2 (column dropped)
majority(split)            # majority class of split.y_fit; a tie predicts pass
```

The positive class is *failed*. Predicting `True` means "will fail".

## Design decisions

- **Reported scores use `sklearn.metrics`** (`zero_division=0`). sklearn has no FPR scorer, so FPR comes from the counts.
- **GP fitness path stays numpy:** `confusion`, `rates` and `mcc` work from one `tp/fp/tn/fn` tuple. Measured on a 2,500-row fold: ~6 µs vs ~1.9 ms for `matthews_corrcoef`. At ~50k evaluations per GP run, sklearn would add ~95 s per run. A test checks the fast MCC against sklearn.
- Division by zero returns `0.0`, never NaN, because NaN would break GP selection.
- `scores` also returns the raw counts, so the CV results frame carries the confusion matrix.
- `pct_builds_saved` = share of builds predicted pass. `pct_failures_caught` = recall on the fail class. Both are percentages (0-100); every other score is a fraction.
- Accuracy is reported but the docs and notebook explain why it misleads under imbalance.
- The multi-objective pair for NSGA-II is `(tpr, fpr)` from `rates`, as in Saidani et al.
- **No decision-tree baseline.** The plan's depth-limited tree was dropped: the project exposes how GP and the flow-control program are used, not whether they beat standard ML. Parrot and majority stay as trivial reference points.
- **Baselines read `split` by duck typing** (`X_fit`, `y_fit`, `X_test`, `raw_test`), so `metrics` does not import `training`. `raw_test` must hold the unscaled `ds.X` test rows with all columns: Parrot needs `last_label`, which is not in the selected list (on T1 atlasdb it ranks 30th), and scaling could map a constant train column to 0.

## Verification

- Hand-computed confusion matrix and scores (MCC = 1/6 case).
- Edge cases: all-pass prediction, all-fail prediction, no failures in the fold. None return NaN.
- Fast `mcc` matches sklearn on random data.
- Baselines: Parrot on T1/T2 frames, majority including the tie, a one-split tree that separates 1-D data.
- Run only this file (`uv run --with pytest pytest tests/test_metrics.py`). The full suite includes `test_features.py`, which is memory-heavy.
