# `validate/live.py`

Status: done (step 8). Spec: [implementation-backbone.md](../implementation-backbone.md). Tests: `projects/2/tests/test_live.py`.

> **Superseded in part (2026-11-07)** by the [multi-repo backbone](../../../../specs/project-2/multi-repo-analysis/implementation-backbone.md): §3.2 (done, phase 1: `train_final(dataset)`, `model_path(dataset, ...)`, `--dataset` and `--model` required), §9.1 multi-target sidecar, `retest` per source, `predict_run(slug)`, `--record` (phase 8). Where the two differ, the multi-repo backbone wins.

## Purpose

The validation the assignment asks for (D4): freeze one model at collection date `C`, re-test it unchanged on runs that happened after `C`, and explain single live predictions.

## API

```python
MODEL_DIR = <projects/2/results/model>

train_final(repo: str, target: Target, features: list[str], choice: Choice) -> pathlib.Path
    # fit on all rows up to C; writes cart_<repo>_<target>.skops + .json sidecar
load_model(path: pathlib.Path) -> tuple[CART, dict]          # model, sidecar
retest(path: pathlib.Path) -> pd.DataFrame                   # metrics on rows created after C, plus baselines
predict_run(path: pathlib.Path, ref: str) -> dict            # ref = run id or SHA
main()   # CLIs: `train-final`, `retest`, `predict-run`
```

## Sidecar (`.json` next to the `.skops` file)

`repo`, `target`, `features`, `params`, `alpha`, `threshold`, `collected_at` (C), `cutoff`, `n_train`, `n_val`, `train_fail_rate`, `medians`, `trained_at`, code version (`git describe --always --dirty`), and the SHA-256 of the `.skops` file. `load_model` checks the hash before loading.

## Design decisions

- **Training data for the final model.** `load(repo, target, until=C)`. Imputation medians are computed on these rows and stored in the sidecar, so new rows are imputed with training statistics, not their own.
- **Retest flow.** `collect(t, since=C)` adds new windows to the same cache; `build(t)` rebuilds the whole table (old + new), so the new rows' history features see the old runs and earlier new runs, never later ones (the build's normal rule). Rows with `created_at > C` are scored by the loaded model **as saved**. Parrot and majority are scored on the same rows. The output frame has the same columns as the CV results, so `retest_vs_cv` plots both.
- **Partial build.** A full build takes about an hour, almost all of it in the per-run git calls of `git_features`. `build(t, select=...)` still sweeps the history features over every run (cheap), but computes git features and returns rows only where `select` is true: runs after `C` for `retest`, one run for `predict_run`. The processed table and the manifest are never overwritten, so the CV stays reproducible. `fetch_windows` tops up the cache; a window saved before its last day ended is fetched again.
- **`predict_run`.** Fetch the run (by id, or the latest master run for a SHA), top up the cache since `C`, build its row, impute with the sidecar medians, and return the label if known, `P(fail)`, the prediction at the saved threshold, `in_target`, and `explain(x)` (None for a forest).
- **skops loading.** `TRUSTED` lists `CART`, `Tree` and `RandomForest`; NumPy arrays and builtins are trusted by skops already. Save and load both check `get_untrusted_types` against it, so an unexpected type fails loudly.
- **Final fit.** Same protocol as one CV fold on all runs up to `C`: fit on the head, alpha and threshold on the last 20%, no refit. `--params` fixes the hyperparameters; without it the grid search runs on that split. `C` (`cutoff`) is the last training run's `created_at`.
- A run still in progress gets a prediction with label "unknown"; this is the live use case.

## Verification

- Round-trip: `train_final` on a toy dataset, then `load_model` gives identical predictions; a modified `.skops` byte fails the hash check.
- `retest` on a toy table split at `C` scores only rows after `C`, and the model's arrays are unchanged afterwards.
- `predict_run` on a cached fixture run returns an explanation whose first line names the root split feature.
- No network in tests: `collect` is patched.
