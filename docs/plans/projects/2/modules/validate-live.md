# `validate/live.py`

Status: todo (step 8). Spec: [implementation-backbone.md](../implementation-backbone.md). Tests: `projects/2/tests/test_live.py`.

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

`repo`, `target`, `features`, `params`, `alpha`, `threshold`, `collected_at` (C), `n_train`, `trained_at`, code version (`git rev-parse HEAD` of this repository), and the SHA-256 of the `.skops` file. `load_model` checks the hash before loading.

## Design decisions

- **Training data for the final model.** `load(repo, target, until=C)`. Imputation medians are computed on these rows and stored in the sidecar, so new rows are imputed with training statistics, not their own.
- **Retest flow.** `collect(t, since=C)` adds new windows to the same cache; `build(t)` rebuilds the whole table (old + new), so the new rows' history features see the old runs and earlier new runs, never later ones (the build's normal rule). Rows with `created_at > C` are scored by the loaded model **as saved**. Parrot and majority are scored on the same rows. The output frame has the same columns as the CV results, so `retest_vs_cv` plots both.
- **`predict_run`.** Fetch the run (by id, or the latest master run for a SHA) into the cache, rebuild, take its row, impute with the sidecar medians, and return the label if known, `P(fail)`, the prediction at the saved threshold, and `explain(x)`. Rebuilding the whole table is simpler than an incremental path and takes seconds at this size. `ponytail:` full rebuild per call; add incremental feature computation only if the CLI is used in a loop.
- **skops loading** passes `trusted=["acise_p2_cidt.tree.cart.CART", "acise_p2_cidt.tree.cart.Tree"]` plus any NumPy types that `skops.io.get_untrusted_types` reports at save time. That list is stored in the sidecar and compared on load, so an unexpected type fails loudly.
- A run still in progress gets a prediction with label "unknown"; this is the live use case.

## Verification

- Round-trip: `train_final` on a toy dataset, then `load_model` gives identical predictions; a modified `.skops` byte fails the hash check.
- `retest` on a toy table split at `C` scores only rows after `C`, and the model's arrays are unchanged afterwards.
- `predict_run` on a cached fixture run returns an explanation whose first line names the root split feature.
- No network in tests: `collect` is patched.
