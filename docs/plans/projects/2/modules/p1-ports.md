# Project 1 ports (step 0)

Status: done (2026-11-02). Spec: [implementation-backbone.md](../implementation-backbone.md). Tests: copied with each module and adapted (`test_metrics.py`, `test_stats.py`, `test_features.py`, `test_cv.py`).

> **Superseded in part (2026-11-07)** by the [multi-repo backbone](../../../../specs/project-2/multi-repo-analysis/implementation-backbone.md): `analysis/features.py` files keyed by `dataset` instead of `owner__name` (Sec. 3.2, done, phase 1); `training/cv.py` per-source rows and `meta["dataset"]` check (Sec. 7, phase 6). The other ports are unchanged. Where the two differ, the multi-repo backbone wins.

## Purpose

Bring the finished, tested project 1 utilities into `acise_p2_cidt` (D2: copies, no imports from `acise_p1_cigp`). Change only what the new dataset needs.

## Why few changes are needed

`data/build.py` always writes the **same column names** whatever the repository (`created_at`, `failed`, `prev_failed`, ...). So the column names in the ports can stay module constants: the repository-specific part is the build step, not these modules. `schema.json` carries what changes per repository (provenance, nominal and dropped columns), and `data/load.py` copies it into `ds.meta`.

## Per-module changes

| Module | Change |
|---|---|
| `utils/logging/log.py` | `ROOT = "acise_p2_cidt"`. Nothing else. |
| `stats/tests.py` | Remove the GP seed dimension: `seed` leaves `ID_COLS`, `per_fold()` is a plain pivot (one row per fold and method), and `position_per_fold()` (program (b) against the GP seed spread) is removed. |
| `metrics/metrics.py` | Import `PREV_LABEL_COL` from `acise_p2_cidt.data.load` (`"prev_failed"`, already 1 = failed). `parrot()` becomes `raw_test[PREV_LABEL_COL].astype(bool)`; it still predicts all-pass on T2, where the column is dropped. Remove the `PASS_LABEL` import. Remove the GP-only `rates()` (NSGA-II objective pair) and fast `mcc()` (GP fitness); `domain()` computes recall itself. |
| `analysis/features.py` | `NOMINAL` is read from `ds.meta["nominal"]` instead of a constant. Rename `project` to `repo` in `load_features` / output file names (`selected_features_<target>_<repo>.json`). Output dir from the project 2 package root. |
| `training/cv.py` | Preprocessing is median imputation only (`SimpleImputer(strategy="median", keep_empty_features=True)`), fitted on the fit rows; `Split.prep` holds the fitted imputer. Trees do not change under monotone transforms, so the GP quantile scaling is removed and thresholds stay in raw units. `Split.raw_fit` / `raw_val` (raw inputs for program (b)) are removed. `run_cv()` has no seeds: CART is deterministic, so one run per fold; payloads are keyed by fold. The one-project check in `folds()` reads `ds.meta["repo"]`. |
| `viz/plots.py` | `FIG_DIR` from the project 2 root. Add: `runs_per_week(df)` (bars of runs plus a failure-rate line), `failed_job_share(df)`, `tree_icicle(tree, feature_names)` (Plotly icicle of the top levels: split text, samples, failure rate per node), `validation_curve(df, param)`, `retest_vs_cv(cv, retest)`. Importances reuse `scores_bar`. Remove the GP plots (`convergence`, `pareto`, `size_vs_fitness`, `feature_freq_performers`, `learning_curve` over seeds). `label_balance()` counts per label for one repository, optionally per group (`by`). `tradeoff()` sweeps the decision threshold instead of the GP `alpha`. |

## Verification

- The copied tests pass after the renames, with the project 1 fixtures replaced by small project 2 frames where a column name changed.
- `test_cv.py` adds: `split.X_test` equals the raw test rows wherever no value was missing, and missing values get the fit-row median.
- `test_metrics.py`: Parrot on a frame with `prev_failed` gives `prev_failed` as bool; without the column, all-pass.
- New plot helpers are checked in the notebooks only (Plotly output), as in project 1.

## Implementation notes

Deviations from the table above, decided while porting:

- `data/load.py` holds only the constants and the `Dataset` dataclass (`order` replaces `groups`), which the ports import. `available_repos` and `load` come with module 4.
- `parrot()` uses `raw_test["prev_failed"] == 1`, not `astype(bool)`: a NaN (no earlier run) must predict pass, and `astype(bool)` turns NaN into `True`.
- The one-repository check in `folds()` raises when `ds.meta["repo"]` is missing or empty.
- `analysis/features.py` drops the schema's nominal columns before `clean()` (`_without_nominal`), so `clean()` no longer needs them. Repository names map to file names as `owner/name` -> `owner__name`, like `data/processed`.
- `tradeoff()` takes `point_name` instead of the hard-coded `(b)`.
- `failed_job_share(df)` expects one row per failed job with a `category` column; the job-name-to-category mapping is done in the EDA notebook.
- `tree_icicle` colours nodes by P(fail) as `predict_proba` gives it (the weighted class share), which differs from the raw failure rate when `class_weight="balanced"`.
- `test_features.py` adds a save/load round-trip; `test_metrics.py` adds the NaN case to the Parrot test and drops the fast-MCC check; `test_stats.py` uses an sklearn tree and a threshold rule as stand-ins instead of GP and program (b).
