# `data/load.py`

Status: todo (step 1). Spec: [implementation-backbone.md](../implementation-backbone.md).

## Purpose

Turn one project's RavenBuild feature CSV into a clean, time-ordered DataFrame plus a binary target, with no information from the future and nothing fitted on data.

## Proposed API

```python
load_project(name: str, target: Literal["T1", "T2"]) -> Dataset
```

`Dataset` is a small frozen dataclass:

- `X: pd.DataFrame`: features only, in chronological order.
- `y: pd.Series`: `failed = 1`, `passed = 0`, same index as `X`.
- `dev_mask: np.ndarray`: boolean, True on the dev segment (first ~20% of each project).
- `meta`: project name, target, row counts, failure counts.

## Steps, in order

1. Read `data/raw/circleci_repos/features/<name>.csv`. Read only the CSVs, never the `.txt` pickles.
2. Sort by `build_nums`, then drop it.
3. Drop leakage columns: `build_duration`.
4. Remap `now_label` to `y` after the label-direction check (below).
5. T1: keep all rows. T2: keep rows with `last_label = pass`, then drop `last_label` because it is constant.
6. Compute `dev_mask` on the filtered frame (after the T2 filter), so dev is 20% of the rows that are actually modelled.

## Not done here

Imputation, scaling, variance filtering, and feature selection. They are fitted on data, so doing them at load would leak test rows into train. They happen in `analysis/features.py` (dev segment) and `training/cv.py` (per fold).

## Checks before it is finished

- **Label direction.** Join with `outcome` in `circleci_java_builds.csv` and confirm whether `1` means pass. Check testcontainers-java (98% `1`), fresco, and cbioportal.
- **Previous-build columns.** Confirm `pr_*` and `log_*` describe the previous build. If any column describes the current build, drop it and note it in the doc.
- Unknown project name raises a clear error that lists the available projects.

## Verification

Small test on a fixture CSV: sort order, dropped columns, label remap, T2 filter, `dev_mask`.

## Open points

- Pooled loading across projects is not planned. Add it only if cross-project runs are wanted.
- Is `data/raw/` populated? Needed for the label-direction check.
