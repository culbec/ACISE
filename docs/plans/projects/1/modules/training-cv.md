# `training/cv.py`

Status: done (step 3b). Spec: [implementation-backbone.md](../implementation-backbone.md). Protocol: the main plan's *Evaluation protocol*. Tests: `projects/1/tests/test_cv.py`.

## Purpose

Run the time-series cross-validation loop, assign folds, and make the fit / validation / test splits, with no dependence on how the model is built. GP, program (b), and the baselines all plug into the same loop.

## API

```python
folds(ds, n_splits=5, val_frac=0.2) -> list[Fold]        # Fold(k, fit, val, test), property train = fit + val
prepare(ds, features, fold) -> Split                    # imputation + scaling fitted on fold.fit
run_cv(ds, features, fit_predict, method, n_splits=5, seeds=None) -> (DataFrame, payloads)
```

- `Fold` holds row positions in `ds.X`. `fold.train` (fit + val) is what `analysis.features.drift` expects: `drift(ds, [f.train for f in folds(ds)], ...)` gives the same folds as the feature-selection notebook.
- `Split` holds `X_fit`, `y_fit`, `X_val`, `y_val`, `X_test` (selected features, scaled), `raw_test` (unscaled `ds.X` test rows, all columns, for Parrot), and `prep` (the fitted pipeline, so program (b) can map raw-unit thresholds to scaled space).
- `fit_predict` returns a bool array (True = will fail) or `Result(pred, payload)`. Without `seeds` it is called as `fit_predict(split)` once per fold (Parrot, majority, program (b)). With `seeds` it is called as `fit_predict(split, seed)` per seed (GP).
- `run_cv` returns a wide frame, one row per fold × seed: `method, fold, seed, n_test, n_fail`, then every key of `metrics.scores` and `metrics.domain`. `seed` is NaN for unseeded models. Payloads are keyed by `(fold, seed)`.

## Split logic

1. One project per run. A pooled `Dataset` raises: its rows are not in one time order.
2. Dev rows must be a prefix of the rows (checked).
3. `TimeSeriesSplit(n_splits)` runs over the rows after dev. Dev is prepended to every train.
4. The last `val_frac` (20%) of train is the validation split.
5. A test fold with fewer than 10 failures raises a `UserWarning` suggesting `n_splits=3`.

## Design decisions

- **Imputation and scaling are fitted on `fit` only**, not on fit + val, so validation is scored like test. `SimpleImputer(median, keep_empty_features=True)` then `MinMaxScaler`, with pandas output so column names survive. `keep_empty_features` stops an all-NaN column on fit from shifting the columns.
- Test values outside the fit range stay outside [0, 1]. They are not clipped: comparisons still work.
- `n_jobs` is not implemented. Add it after timing one GP run.

## Check on real data (atlasdb, T2, frozen 10 features)

| Fold | fit | val | test | test failures |
|---|---|---|---|---|
| 0 | 573 | 143 | 284 | 70 |
| 1 | 800 | 200 | 284 | 77 |
| 2 | 1028 | 256 | 284 | 61 |
| 3 | 1255 | 313 | 284 | 62 |
| 4 | 1482 | 370 | 284 | 81 |

All folds are above the 10-failure threshold. The majority baseline gives MCC 0 and accuracy 0.72–0.79 on every fold.

## Verification

- Fold properties: fit < val < test in time, dev inside fit, train contiguous with val as its tail, expanding window.
- Pooled dataset rejected; few-failure warning raised.
- A spike in a test row does not change the scaled fit rows, and is not clipped; fit rows have no NaN after imputation.
- `run_cv` with an unseeded baseline (5 rows, no payloads) and a seeded model returning `Result` (10 rows, payloads by `(fold, seed)`).
