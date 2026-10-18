# `training/cv.py`

Status: todo (step 3b). Spec: [implementation-backbone.md](../implementation-backbone.md). Protocol: the main plan's *Evaluation protocol*.

## Purpose

Run the time-series cross-validation loop, assign folds, and make the train / validation / test splits, with no dependence on how the model is built. GP, program (b), and the baselines all plug into the same loop.

## Proposed API

```python
folds(ds: Dataset, n_splits: int = 5) -> list[Fold]

run_cv(ds, features, fit_predict, n_splits=5, seeds=range(10)) -> pd.DataFrame
```

- `Fold` has index arrays `train`, `val`, `test` and its fold number.
- `fit_predict(train, val, test, seed) -> Result` is supplied by the caller. It receives scaled and imputed arrays and returns test predictions, plus an optional payload (hall of fame, convergence stats).
- `run_cv` returns a tidy frame: fold, seed, and every metric from `metrics`.

## Split logic

1. `ds` rows are already chronological, and rows before `dev_end` are the dev segment.
2. `TimeSeriesSplit(n_splits)` runs over the rows after `dev`. The dev rows are prepended to every fold's train.
3. In each fold the last 20% of train becomes the inner validation split (early stopping, picking the final program).
4. Each test fold is used once.
5. If a test fold has fewer than about 10 failures, warn and suggest `n_splits=3`.

## Design decisions

- **Scaling and imputation are fitted here, per fold, on that fold's train.** Min-max scaler and median imputer are fitted on train, then applied to val and test. The selected feature list comes from the frozen `Selection`.
- Seeds are an argument, so the same loop serves deterministic models (one seed) and GP (ten seeds).
- Parallelism over folds and seeds can be added behind a `n_jobs` argument once a GP run is timed. It is not in the first version.
- The loop never sees labels from a later fold when it fits anything.

## Verification

- Property tests on the splits: train indices all precede test indices, no overlap among train/val/test, dev always inside train, validation is the tail of train.
- Scaler fitted on train only: a spike in test data does not change the scaler.
