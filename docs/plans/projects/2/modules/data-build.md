# `data/build.py`

Status: todo (step 3). Spec: [implementation-backbone.md](../implementation-backbone.md). Tests: `projects/2/tests/test_build.py`.

## Purpose

Turn the raw API cache and the git log into one table with one row per CI run, all pre-run features, and the label. This is where leakage is prevented, so it is the most tested module.

## API

```python
build(t: Target) -> tuple[pd.DataFrame, dict]   # table, schema
write(df: pd.DataFrame, schema: dict, t: Target) -> pathlib.Path
    # data/processed/<owner>__<name>.csv and results/collection/schema_<owner>__<name>.json
main()                                           # CLI `build-dataset [--repo ...]`

# stages, each a pure function of its inputs (tested on toy frames)
runs_frame(raw_dir) -> pd.DataFrame          # one row per run, first-attempt label and times
history_features(runs) -> pd.DataFrame       # CI history, load, author-run features
git_features(runs, log, repo_dir) -> pd.DataFrame
pr_features(runs, pulls, pr_runs) -> pd.DataFrame
author_type(runs, pulls) -> pd.Series
hash_id(value: str, salt: bytes) -> str
```

## Output columns

- `run_id`, `head_sha`: identifiers (dropped from `X`).
- `created_at`: order column (dropped from `X`).
- `failed`: label, 1 = first attempt failed.
- `prev_failed`: the Parrot input, also a feature; T2 filters on it.
- All features in the plan's features table. `author_type` is one-hot encoded (`author_type_bot`, `author_type_member`, `author_type_contributor`, `author_type_first_time`).
- Diagnostics, never features: `was_rerun`, `flaky`, `conclusion_latest`, `failed_job_names`.

`schema.json`: `repo`, `workflow_id`, `branch`, `collected_at` (C), `n_runs`, `n_dropped` per reason, `order_col`, `id_cols`, `label_col`, `prev_label_col`, `nominal` (empty after one-hot), `diagnostic_cols`, `feature_cols`.

## Design decisions

- **Rows.** First-attempt `status == completed` and `conclusion ∈ {success, failure}`; other conclusions are dropped and counted in the schema. The label and the completion time come from the **first attempt** (`attempts/1` when the run was re-run).
- **Completed-before sweep.** Rows are sorted by `(created_at, run_id)`. A min-heap holds pending runs keyed by first-attempt completion time. Before row *i* is processed, every pending run with completion `< created_at_i` is popped into the "known" history (its outcome, duration, failed jobs, files, author). Row *i*'s features read only the known history; then row *i* is pushed as pending. This one pass gives `prev_*`, streaks, all windowed rates, `concurrent_runs` (pending count at `created_at_i`), and author and file failure rates.
- **Load windows** (`ci_load_*`) count runs **created** in the window before `created_at_i`. Creation is known at that time even if the outcome is not, so this does not leak.
- **Windows.** Time windows use two-pointer scans over the sorted known history (`collections.deque` per window); count windows use the last *k* known runs. A window with no known runs yields NaN, not 0, so the tree can tell "no data" from "no failures" after imputation.
- **Git features.** `head_sha` gives this run's change from `commit_log`. The reference SHAs are the last known run's SHA (since-previous-run) and the last known **passing** run's SHA (since-green), through `diff_numstat` / `rev_count`. File history (`file_age_mean`, `file_prior_changes`, `file_distinct_authors`, `secs_since_files_last_changed`) uses only commits with `commit_time < created_at_i` that are ancestors of `head_sha`; on a linear master that is the log prefix up to `head_sha`.
- **File failure probability.** For each touched file: known runs whose change touched it, failed / total; `max` and `mean` over the touched files. Files never seen give NaN.
- **Revert link.** `is_revert` when the subject starts with `Revert "` and the body has `This reverts commit <sha>`; `reverted_sha_failed` is the known outcome of the run whose `head_sha` is that SHA (NaN if unknown).
- **Author identity.** Git author e-mail and API login are hashed with a salted SHA-256 (`hash_id`). The salt is generated once into `data/raw/<repo>/salt.bin` and never leaves `data/raw/`. Raw logins and e-mails are never written to `data/processed/`.
- **`author_type`.** `bot` if the API `author.type == "Bot"` or the login ends in `[bot]`; else from the PR's `author_association`: OWNER/MEMBER/COLLABORATOR → `member`, CONTRIBUTOR → `contributor`, FIRST_TIME_CONTRIBUTOR/FIRST_TIMER/NONE → `first_time`; no PR → `member` (a direct push to master needs write access).
- **PR features.** From the PR returned for `head_sha` (the merged one if several). `pr_ci_failed_runs` counts PR CI runs whose first attempt failed. No PR → NaN for every `pr_*`.
- **Workflow and release.** `workflow_changed`: the run's change touches the workflow file (`ci_files` filtered to this workflow's path, from the run JSON's `path`). `days_since_workflow_change` and `days_since_last_tag` from `commit_log` and `tags` restricted to `commit_time < created_at_i`.
- `ponytail:` the sweep is pure Python over ~4k rows (well under a second per feature group). Vectorise only if a repository with >100k runs is ported.

## Verification

- **No look-ahead (main test).** Toy table of 5 runs where run 2 is created while run 1 is still running: run 2's `prev_failed` comes from run 0, not run 1. Changing run 1's label does not change any feature of run 2. Changing any *later* run does not change any earlier row (checked over all columns).
- Hand-computed `fail_streak`, `fail_rate_last_5`, `fail_rate_1h`, `concurrent_runs`, `ci_load_1h` on the toy table.
- First-attempt label: a run re-run to success keeps `failed = 1`, `flaky = 1`.
- Dropped conclusions are counted in the schema.
- No raw login or e-mail appears anywhere in the output frame (string search over all cells).
- `author_type` mapping table, parametrised.
