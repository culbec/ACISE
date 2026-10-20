# `training/cv.py`

Status: done (step 3b). Spec: [implementation-backbone.md](../implementation-backbone.md). Protocol: the main plan's *Evaluation protocol*. Tests: `projects/1/tests/test_cv.py`.

## Purpose

Run the time-series cross-validation loop, assign folds, and make the fit / validation / test splits, with no dependence on how the model is built. GP, program (b), and the baselines all plug into the same loop.

## API

```python
folds(ds, n_splits=5, val_frac=0.2) -> list[Fold]        # Fold(k, fit, val, test), property train = fit + val
prepare(ds, features, fold) -> Split                    # imputation + quantile scaling fitted on fold.fit
run_cv(ds, features, fit_predict, method, n_splits=5, seeds=None) -> (DataFrame, payloads)
```

- `Fold` holds row positions in `ds.X`. `fold.train` (fit + val) is what `analysis.features.drift` expects: `drift(ds, [f.train for f in folds(ds)], ...)` gives the same folds as the feature-selection notebook.
- `Split` holds `X_fit`, `y_fit`, `X_val`, `y_val`, `X_test` (selected features, scaled), `raw_test` (unscaled `ds.X` test rows, all columns, for Parrot), and `prep` (the fitted pipeline, so program (b) can map raw-unit thresholds to scaled space and back).
- `fit_predict` returns a bool array (True = will fail) or `Result(pred, payload)`. Without `seeds` it is called as `fit_predict(split)` once per fold (Parrot, majority, program (b)). With `seeds` it is called as `fit_predict(split, seed)` per seed (GP).
- `run_cv` returns a wide frame, one row per fold × seed: `method, fold, seed, n_test, n_fail`, then every key of `metrics.scores` and `metrics.domain`. `seed` is NaN for unseeded models. Payloads are keyed by `(fold, seed)`.

## Split logic

1. One project per run. A pooled `Dataset` raises: its rows are not in one time order.
2. Dev rows must be a prefix of the rows (checked).
3. `TimeSeriesSplit(n_splits)` runs over the rows after dev. Dev is prepended to every train.
4. The last `val_frac` (20%) of train is the validation split.
5. A test fold with fewer than 10 failures raises a `UserWarning` suggesting `n_splits=3`.

## Design decisions

- **Imputation and scaling are fitted on `fit` only**, not on fit + val, so validation is scored like test. `SimpleImputer(median, keep_empty_features=True)` then `QuantileTransformer(n_quantiles=min(200, len(fit)))`, with pandas output so column names survive. `keep_empty_features` stops an all-NaN column on fit from shifting the columns.
- **Quantile, not min-max.** A scaled value is the row's percentile among the fit rows. Min-max left 90% of `time_interval` below 0.085 and of `src_churn` / `import_change` below 0.04, so the GP's U(0, 1) thresholds rarely landed where the data varies: 92% of the best programs used `time_interval`, 24% were one statement. Check (atlasdb, folds 0/2/4, SO, 5 seeds): median val MCC 0.121 -> 0.151, test MCC 0.198 -> 0.246, statements 1 -> 4-6. Val rose on every fold, so the choice does not rest on test. The notebook repeats the check on the dev split only.
- Val and test values outside the fit range are clipped to 0 or 1. Harmless: thresholds lie in [0, 1].
- **`to_python`** (`gp/simplify.py`): a threshold maps back to an exact raw-unit threshold (the transform jumps at tied knots and is linear between them, so the boundary is a knot or a linear inverse inside one knot interval). A feature-vs-feature comparison has no raw form; it prints as `pct(a) op pct(b)` with the knots `Q` and `pct` embedded in the output. Checked: generated source equals `predict` on 200 random synthetic and 300 random real (atlasdb, last fold) programs.
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
- A spike in a test row does not change the scaled fit rows, and is clipped to 1; fit rows have no NaN after imputation.
- `run_cv` with an unseeded baseline (5 rows, no payloads) and a seeded model returning `Result` (10 rows, payloads by `(fold, seed)`).

## Pooled CV (dropped)

Status: dropped on 2026-10-20, not implemented. Kept below as a record.

**Decision.** With quantile scaling, the atlasdb learning curve (last fold, SO, 5 seeds) flattens: test MCC mean 0.129, 0.158, 0.178, 0.184 at 370, 741, 1,111, 1,482 fit rows. The last step (+0.006) is far inside the seed spread (std 0.06-0.10). Under min-max the curve still rose (0.05, 0.11, 0.15, 0.17); most of that rise was the scaler wasting small fit sets, not missing data. Pooling would add about 1.9x rows for an expected gain near 0.01 MCC, with three more scalers, a per-project readout and atlasdb still dominating the pool. It would also not touch the main spread, which is across folds (SO fold medians 0.05-0.29, folds 1 and 3 weak in every run): drift over time, not row count. Final CV (atlasdb, 10 seeds): SO median MCC 0.185 (fold-median mean 0.172), MO 0.155 (0.180, 200 generations); Majority and Parrot 0.

Original plan, unchanged: Pool: `atlasdb`, `gradle-baseline`, `cassandra`, `rskj`, `spring-cloud-aws` (the most balanced pass/fail on T2). Motivation: the atlasdb learning curve (last fold, 5 seeds, min-max scaling) has a test MCC mean that still rises at 1,482 fit rows (0.05, 0.11, 0.15, 0.17) and a seed spread that narrows. Val MCC at 1,482 rows (0.13) is below test (0.17): 370 val rows are a noisy pick of the final program, and pooling enlarges val too. More rows are expected to cut variance; the ceiling stays near the programs' own fit MCC (~0.2-0.25). Order: quantile scaling first (done), rerun atlasdb, then pool.

### Pool size (T2)

| Project | Builds | Failures | Dev builds / failures | Test rows per fold / failures |
|---|---|---|---|---|
| atlasdb | 2136 | 597 | 427 / 145 | 284 / ~75 |
| rskj | 845 | 125 | 169 / 8 | 112 / ~20 |
| spring-cloud-aws | 418 | 72 | 83 / 9 | 55 / ~10 |
| gradle-baseline | 378 | 64 | 75 / 4 | 50 / ~10 |
| cassandra | 313 | 81 | 62 / 23 | 41 / ~10 |
| **pool** | 4090 | 939 | 816 / 189 | 542 / ~125 |

atlasdb is 52% of the rows, 64% of the failures and 77% of the dev failures, so pooled selection and pooled scores lean on it. Per-project scores are reported for that reason.

### Changes

1. **`data/load.py`**: `load_project(name: str | list[str] | None, all=False)`; a list stacks those projects. `preprocess` already filters T2 and cuts dev per project.
2. **`training/cv.py`, `folds`**: a pooled `Dataset` is no longer rejected. Per project: dev rows, then `TimeSeriesSplit(n_splits)` over that project's rows after dev, train = its dev + its train part, val = the last `val_frac` of that train. Fold `k` concatenates fold `k` of every project. Each project keeps its own time order; projects are independent repositories, so no cross-project ordering is assumed. The few-failure warning checks the pooled test fold; per-project test counts go to `DEBUG`.
3. **`training/cv.py`, `dev_split`**: head/tail cut per project (today a pooled dev tail would be only the last project).
4. **`training/cv.py`, `prepare`**: one imputer + quantile pipeline per project, fitted on that project's fit rows (`time_interval` medians range from 39 to 2,956 across the pool, so one global scaler would make every threshold mean something different per project). A scaled value is then the percentile within its own project, the same meaning in every project, and one outlier in a small project cannot set its range. `Split.prep` becomes `dict[str, Pipeline]` (project -> pipeline); single-project data gives a one-entry dict. `to_python(tree, pset, prep[project])` prints a program in one project's raw units.
5. **`training/cv.py`, `run_cv`**: a `project` column. Every fold x seed gets one row with `project = "all"`, and on pooled data one more row per project (scores on that project's test rows). `stats.summary` groups by `method, project` when the column is there; `per_fold`, `paired_wilcoxon` and `position_per_fold` are called on `results[results.project == "all"]` (or one project).
6. **`analysis/features.py`**: no code change. `select_features(pooled_ds)` runs on the pooled dev rows and saves `selected_features_T2_pooled.json`, with the project list in `meta`. The relevance and redundancy statistics are rank based, but pooled ranks still mix project scales; this is a known limitation, noted in the notebook. Fix if it matters: run selection on per-project quantile-scaled dev rows. The frozen list and N may differ from atlasdb's 10.
7. **`gp/engine.py`**: no change; it only sees `Split`.
8. **Readout**: atlasdb's per-project folds in the pooled run are the same rows as in the single-project run. So the `project == "atlasdb"` rows of the pooled CV and `cv_T2_atlasdb.csv` are paired per fold: "pooled-trained vs atlasdb-only, on atlasdb test rows" answers whether more data helped.

### Caveats

- Dev failures of gradle-baseline, rskj and spring-cloud-aws are 4, 8 and 9, so their 20% dev validation tail holds about 1-2 failures each. Tuning and the program (b) dev runs stay driven by atlasdb.
- With 5 folds the Wilcoxon signed-rank test has a minimum two-sided p of 0.0625: never significant at 0.05, pooled or not. Report A12 over seeds as the effect size next to it.

### Cost

Pooled fit rows are about 1.9x atlasdb's, so a run takes about 15-35 s; the CV (5 folds x 10 seeds x 2 variants) about 40 min.

### Verification (`tests/test_cv.py`)

- Two synthetic projects with different scales: every fold has rows of both; within a project fit < val < test in time and dev is inside fit; the scaled fit rows of each project lie in [0, 1].
- `dev_split` on pooled data has validation rows from every project.
- `run_cv` on pooled data gives `1 + n_projects` rows per fold x seed, and the `"all"` row's counts equal the sum of the per-project rows.
- atlasdb's fit / val / test rows in each pooled fold equal its single-project folds (mapped through the pooled row positions).
