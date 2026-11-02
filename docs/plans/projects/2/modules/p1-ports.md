# Project 1 ports (step 0)

Status: todo. Spec: [implementation-backbone.md](../implementation-backbone.md). Tests: copied with each module and adapted (`test_metrics.py`, `test_stats.py`, `test_features.py`, `test_cv.py`).

## Purpose

Bring the finished, tested project 1 utilities into `acise_p2_cidt` (D2: copies, no imports from `acise_p1_cigp`). Change only what the new dataset needs.

## Why few changes are needed

`data/build.py` always writes the **same column names** whatever the repository (`created_at`, `failed`, `prev_failed`, …). So the column names in the ports can stay module constants: the repository-specific part is the build step, not these modules. `schema.json` carries what changes per repository (provenance, nominal and dropped columns), and `data/load.py` copies it into `ds.meta`.

## Per-module changes

| Module | Change |
|---|---|
| `utils/logging/log.py` | `ROOT = "acise_p2_cidt"`. Nothing else. |
| `stats/tests.py` | None. |
| `metrics/metrics.py` | Import `PREV_LABEL_COL` from `acise_p2_cidt.data.load` (`"prev_failed"`, already 1 = failed). `parrot()` becomes `raw_test[PREV_LABEL_COL].astype(bool)`; it still predicts all-pass on T2, where the column is dropped. Remove the `PASS_LABEL` import. |
| `analysis/features.py` | `NOMINAL` is read from `ds.meta["nominal"]` instead of a constant. Rename `project` to `repo` in `load_features` / output file names (`selected_features_<target>_<repo>.json`). Output dir from the project 2 package root. |
| `training/cv.py` | `prepare()` and `dev_split()` take `prep: Callable[[int], Pipeline] = impute_only`, which builds the pipeline from the fit size. `impute_only` = `SimpleImputer(strategy="median", keep_empty_features=True)`; the project 1 quantile pipeline stays available as `impute_quantile`. Trees do not change under monotone transforms, so scaling is skipped and thresholds stay in raw units. The one-project check in `folds()` reads `ds.meta["repo"]`. `Split.raw_fit` / `raw_val` keep their meaning. |
| `viz/plots.py` | `FIG_DIR` from the project 2 root. Add: `runs_per_week(df)` (bars of runs plus a failure-rate line), `failed_job_share(df)`, `tree_icicle(tree, feature_names)` (Plotly icicle of the top levels: split text, samples, failure rate per node), `validation_curve(df, param)`, `retest_vs_cv(cv, retest)`. Importances reuse `scores_bar`. |

## Verification

- The copied tests pass after the renames, with the project 1 fixtures replaced by small project 2 frames where a column name changed.
- `test_cv.py` adds: with `impute_only`, `split.X_test` equals the raw test rows wherever no value was missing.
- `test_metrics.py`: Parrot on a frame with `prev_failed` gives `prev_failed` as bool; without the column, all-pass.
- New plot helpers are checked in the notebooks only (Plotly output), as in project 1.
