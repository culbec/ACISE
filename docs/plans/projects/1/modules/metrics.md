# `metrics/metrics.py`

Status: todo (step 3a). Spec: [implementation-backbone.md](../implementation-backbone.md).

## Purpose

One shared definition of "how good is a prediction", used by feature-selection checks, the CV loop, the GP fitness, and the final results. Used for both (a) and (b), so they are scored identically.

## Proposed API

```python
confusion(y_true, y_pred) -> ConfusionCounts          # tp, fp, tn, fn
rates(counts) -> Rates                                # tpr, fpr
scores(y_true, y_pred) -> dict[str, float]            # precision, recall, f1, mcc, accuracy, tpr, fpr
domain(y_true, y_pred) -> dict[str, float]            # pct_builds_saved, pct_failures_caught
baselines: parrot(last_label), majority(y_train), shallow_tree(X, y, features, depth)
```

The positive class is *failed*. Predicting `True` means "will fail".

## Design decisions

- **Pure numpy, vectorised.** The GP calls the fitness thousands of times per run, so the TPR/FPR/MCC path computes from one `tp/fp/tn/fn` tuple and does not go through sklearn.
- Division by zero (no positives in a fold, or no predicted positives) returns `0.0` for the score and is logged once. It never returns NaN, because NaN would break GP selection.
- `pct_builds_saved` = share of builds predicted pass. `pct_failures_caught` = recall on the fail class.
- Accuracy is reported but the docs and notebook explain why it misleads under imbalance.
- The multi-objective pair for NSGA-II is `(tpr, fpr)` from `rates`, as in Saidani et al.
- `shallow_tree` uses the same selected features and `N`-bounded depth as the reference point in the plan.

## Verification

- Hand-computed confusion matrices, including the all-pass and all-fail edge cases.
- Cross-check each score against `sklearn.metrics` on random data.
