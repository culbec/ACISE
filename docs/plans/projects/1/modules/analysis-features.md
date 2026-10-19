# `analysis/features.py`

Status: done (step 2). Spec: [implementation-backbone.md](../implementation-backbone.md). Method: the main plan's *Feature selection* section. Tests: `projects/1/tests/test_features.py`.

## Purpose

Choose the feature list and the statement budget `N` once, from the development segment only. Both are then frozen for every CV fold and shared by (a) and (b). The headline target is **T2** (first failures); T1 is run for completeness.

## API

```python
select_features(ds: Dataset, n_max=20, min_freq=0.6, n_boot=20, rho=0.8, seed=0) -> Selection
drift(ds: Dataset, trains: list[np.ndarray], selected: list[str], n_max=20, rho=0.8, seed=0) -> pd.DataFrame
```

`Selection` holds:

| Field | Content |
|---|---|
| `features` | frozen list, in relevance order; `N = Selection.size` |
| `scores` | feature × statistic, plus `score` (mean percentile rank) |
| `stability` | selection frequency over the block subsamples, per family |
| `importance` | random-forest permutation importance (embedded check) |
| `disagreements` | embedded top-5 features the filters dropped |
| `meta` | target, project (`pooled` for several), dev rows/failures, columns dropped in `clean`, parameters |

`Selection.save(out_dir)` writes `results/features/selected_features_<target>_<project>.json`.

Each stage is its own function so notebooks can plot the intermediate results:

1. `clean(X)`: drop the nominal codes (see below), columns with more than 30% missing, near-constant columns (`VarianceThreshold` 1e-3 on min-max scaled data) and binary columns whose minority value is under 1%. Median-impute the rest.
2. `relevance(X, y)`: features × statistic. Numeric: ANOVA F and |Kendall τ|. Binary: chi-square. All: mutual information (on binary columns this is the information gain). `score` is the mean **percentile** rank over the statistics that apply, so numeric and binary features are comparable.
3. `prune_redundant(X, ranking, rho=0.8)`: walk the ranking and keep a feature only if its |Spearman ρ| with every kept feature is at most 0.8.
4. `stability(X, y, rep, ...)`: repeat 1–3 on `n_boot` block subsamples; count how often each feature is in the top `n_max`.
5. **Threshold:** the final list is the pruned features with stability ≥ `min_freq` (default 0.6), in relevance order, capped at `n_max`.
6. `embedded_check(X, y, selected)`: random forest (`class_weight="balanced"`) fitted on the first 70% of dev; permutation importance (ROC AUC) on the last 30%. Returns the disagreements so the notebook can record a decision. It does not change the list.
7. `drift(ds, trains, selected)`: diagnostic only (see below).

## Design decisions

- **Dev slice only.** Input is `X[ds.dev_mask]`; the function has no access to later rows. The dev slice stays at the start of each project, so it comes before every test fold.
- **Drift check instead of moving the dev slice.** Spreading dev over time would cover more history, but early test folds would then be scored with a feature list chosen partly from builds after them. Instead, `drift` re-runs steps 1–3 on each fold's train (positions in `ds.X`, supplied by `training/cv.py`) and reports, per frozen feature and fold, whether the feature or a near-duplicate (|ρ| > `rho` on that train) is in the fold's top `n_max`. It never changes the frozen list.
- **Nominal codes are excluded** (`change_type`, `action_changed`, `day_of_week`, `time_of_day`). The GP compares with `lt`/`ge`, and thresholds on unordered codes mean nothing.
- **Binary is decided by the value set** ({0, 1}), not by `nunique == 2`: on a small dev slice a count column such as `last_fail_gap_avg` can take only two values.
- **Rank-based statistics.** Churn and count columns are heavily skewed (skew up to ~24), so Pearson is not used.
- **Subsampling, not bootstrap.** Deviation from the plan ("bootstrap resamples of train"): each resample keeps 80% of contiguous blocks (length n^(1/3)), drawn **without replacement** and left in time order. Blocks keep the dependence between neighbouring builds. Drawing with replacement duplicates rows, which inflates the k-NN mutual-information estimate for continuous columns; on atlasdb this made `fail_ratio_pr_inc`, `file_fail_prob_avg` and `consec_fail_builds_avg` win every resample.
- **Family credit in stability.** Near-duplicates swap the top slot between resamples (about 50/50), which would split the family's frequency and drop it under any threshold. A pick is credited to the full-dev representative it duplicates (|ρ| > `rho`).
- **History aggregates are candidates like any other.** `fail_ratio_*`, `consec_fail_builds_*`, `file_fail_prob_*` and `last_fail_gap_avg` are computed by RavenBuild only from builds strictly before the current one (`temp_df = build_history_df.iloc[:truncate_idx]`), so they do not leak. The threshold decides whether they stay.
- **Scaling and imputation** inside the selection are local and not returned; `training/cv.py` re-fits its own per fold.
- **T1 and T2** run separately. On T1, `last_label` is kept; the notebook reports with and without it (`dataclasses.replace(ds, X=ds.X.drop(columns="last_label"))`).

## Data notes (from the RavenBuild source)

- **Inverted names.** RavenBuild's `result == 1` means **pass**. `fail_ratio_re`, `fail_ratio_com_pr`, `fail_ratio_com_re` and `consec_fail_builds_*` therefore measure *passes*. `fail_ratio_pr` is a fail ratio (it uses `1 - mean`), and `file_fail_prob_*` uses `result == 0` (fail). Read GP rules with this in mind.
- **Copied columns.** `get_open_source_features.py` appends `import_deleted_line` to `class_changed` and `met_sig_modified` too; the computed `class_changed_line` (`parser_diff[14]`) and `met_sig_modified_line` (`parser_diff[15]`) are discarded. Both columns equal `import_deleted` on all 11,252 rows. A real class-rename signal would be useful (a renamed class also deletes imports), but the CSVs do not contain it, and recomputing it needs the GumTree diffs from the pickled raw files, which are not loaded. Pruning keeps one of the three. Whichever name is kept, it is interpreted as `import_deleted`.

## Results (frozen, atlasdb only, seed 42)

Notebook: `projects/1/notebook/feature_selection.ipynb`. Files: `results/features/selected_features_{T1,T2}_atlasdb.json`.

| Target | Dev rows | Dev failures | N |
|---|---|---|---|
| T2 (headline) | 427 | 145 | **10** |
| T1 | 631 | 229 | 18 |

- **T2 frozen list:** `time_interval`, `pr_duration`, `has_old_change`, `file_fail_prob_max`, `met_changed`, `is_same_counter`, `src_churn`, `config_files`, `import_change`, `fail_ratio_pr_inc`.
- **Drift drop (T2):** `min_freq = 0.6` gave 14 features. The 4 lost in ≥ 3 drift folds were removed after selection and are recorded in `meta.dropped_due_to_drift`: `log_src_files`, `class_deleted`, `prev_testchurn`, `line_deleted`. They were dropped after selection, not before, so their near-duplicates (e.g. `pr_src_files` for `log_src_files`) cannot take their place.
- **Embedded disagreements (T2):** `log_test_files` (family of `prev_testchurn`, now dropped) and `fail_ratio_pr` (family of `fail_ratio_pr_inc`). `line_added` (stability 0.35) is not added.
- **Seed sensitivity:** with seed 0, the families are mostly the same, but the representative changes (e.g. `src_files`/`src_churn`, `file_fail_prob_avg`/`_max`, `met_sig_modified`/`import_change`). Interpret GP and (b) rules by family.
- **T1:** `last_label` ranks 30th with stability 0, although it is a candidate in every subsample. Parrot MCC on dev is 0.075, so the plan's expectation that T1 is mostly Parrot does not hold on atlasdb.
- Runtime about 75 s per target, mostly the random forest and the 20 subsamples.

## Verification

- Synthetic data with informative, duplicate, constant, rare-binary, nominal and noise columns: informative ones are kept, only one of each duplicate pair is kept, and constant, rare and nominal columns are dropped. The duplicate's picks are credited to its representative, and every selected feature meets `min_freq`.
- Determinism with a fixed seed.
- `drift` marks the informative features as present on every fold.

## Open points

- Only atlasdb is analyzed. The other projects' dev slices have too few first failures (4–9) for a stable selection.
- Plots for this module live in `viz/plots.py` (`scores_bar` with percentile ranks, `stability_bar`, `corr_heatmap`, `drift_heatmap`).
