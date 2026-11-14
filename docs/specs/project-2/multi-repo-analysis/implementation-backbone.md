# Multiple Repository Analysis - Implementation backbone

Code-level companion to [plan.md](plan.md), which implements [SPEC.md](SPEC.md). Package root: `projects/2/src/acise_p2_cidt/`. The [project backbone](../../../plans/projects/2/implementation-backbone.md) still governs coding style, tests, leakage boundary, secrets, and "one module at a time, on request". Each section below is one unit of work. It names the files it touches, its API, and its tests.

## 1. Modules and order

| # | Section | Files | Plan phase |
|---|---|---|---|
| 1 | Sec. 2 Registry | `collect/targets.py` (new), `config/targets.json` (new) | 1 |
| 2 | Sec. 3 Identity and layout | `collect/gh_runs.py`, `data/load.py`, `data/build.py`, `analysis/features.py`, `validate/live.py` | 1 |
| 3 | Sec. 4 Collector | `collect/gh_runs.py` | 2 |
| 4 | Sec. 5 Git | `collect/git_changes.py` | 2 |
| 5 | Sec. 8 Label check | `analysis/labels.py` (new) | 3 |
| 6 | Sec. 6 Build | `data/build.py`, `data/history.py` | 4 |
| 7 | Sec. 7 Pooled load and CV | `data/load.py`, `training/cv.py` | 6 |
| 8 | Sec. 9 Live and sandbox | `validate/live.py`, `validate/sandbox.py` (new) | 8, 9 |
| 9 | Sec. 11 Mock data, notebook harness, notebooks | `tests/mock_pool.py`, `tests/nb_harness.py`, `tests/test_notebooks.py` (new), `notebook/*.ipynb` | plan Sec. 5 |
| 10 | Sec. 12 ASCII cleanup | every file in the scope of Sec. 12, `tests/test_ascii.py` (new) | 10 |

Order: 1 -> 2 -> (3, 4) -> 5 -> 6 -> 7 -> 8, then 10 once the project is complete. New and edited text in every unit is ASCII-only (plan, ASCII rule); unit 10 cleans what was written before. `tests/mock_pool.py` is written with unit 1 and extended by each later unit, because every unit tests on it. The notebooks are edited in steps N1-N6, each right after the unit it calls (plan Sec. 5.2); `tests/nb_harness.py` and `tests/test_notebooks.py` come with N1, after unit 4. Sections 3 and 4 are independent. Every file stays under 500 lines. `build.py` and `history.py` are near 300 lines, so the lineage helpers go to a new `data/lineage.py` if either file would pass 450.

New CLIs, registered in `pyproject.toml` `[project.scripts]`: `label-check = "acise_p2_cidt.analysis.labels:main"`, `sandbox-report = "acise_p2_cidt.validate.sandbox:main"`.

## 2. Registry (`collect/targets.py`, `config/targets.json`)

`projects/2/config/targets.json` is a JSON list:

```json
[
  {"repo": "n8n-io/n8n", "workflow_id": 33322401, "branch": null, "enabled": false, "exclude_jobs": []},
  {"repo": "n8n-io/n8n", "workflow_id": 33323319, "branch": "master", "enabled": false, "exclude_jobs": []},
  {"repo": "servo/servo", "workflow_id": 3876879, "branch": null, "enabled": false, "exclude_jobs": ["^Result$", "Bencher"]}
]
```

The initial file holds every candidate of SPEC Sec. 9, plus n8n `ci-master` (P-D3). The workflow ids are in plan Sec. 1.

```python
# projects/2/src/acise_p2_cidt/collect/targets.py -> projects/2/config/targets.json
REGISTRY: pathlib.Path

@dataclass(frozen=True)
class TargetConfig:
    target: Target
    enabled: bool
    exclude_jobs: tuple[str, ...]  # regexes on "<job>" or "<job> :: <failed step>"

    excludes(job: str | None, step: str | None) -> bool  # job None (no failed job) is infra

load_targets(path: pathlib.Path = REGISTRY, enabled_only: bool = True) -> list[TargetConfig]
find_target(slug: str, path: pathlib.Path = REGISTRY) -> TargetConfig  # enabled or not
```

Validation, done once in `load_targets`. Each failure raises `ValueError` naming the entry index and `repo`:
- The top level is a list. Each entry is an object.
- Required keys: `repo`, `workflow_id`, `enabled`. Optional: `branch` (default `null`), `exclude_jobs` (default `[]`). Unknown keys are rejected, which catches typos such as `enable`.
- `repo` matches `^[\w.-]+/[\w.-]+$`. `workflow_id` is an `int > 0` and not a `bool`. `branch` is `null` or a non-empty string. `enabled` is a `bool`.
- Every `exclude_jobs` item is a string that compiles with `re.compile`.
- No two entries share a slug (Sec. 3).

## 3. Identity and layout

### 3.1 `Target`

```python
class Target(NamedTuple):
    repo: str                 # "owner/name"
    workflow_id: int
    branch: str | None = None # None = all branches

slug(t: Target) -> str        # "n8n-io__n8n__33323319@master", "astral-sh__uv__82881044"
```

`slug` = `owner__name__<workflow_id>`, plus `@<branch>` with `/` replaced by `~`. Every per-target file is keyed by `slug`, not by repository.

### 3.2 Layout

| Path | Keyed by | Content |
|---|---|---|
| `data/raw/<owner>__<name>/repo.git` | repository | one clone shared by all targets of the repository |
| `data/raw/<owner>__<name>/salt.bin` | repository | shared, so one author gets one id across the repository's targets |
| `data/raw/<owner>__<name>/<slug>/{runs,attempts,jobs}/` | target | API cache |
| `data/processed/<slug>.csv` | target | built table |
| `results/collection/{manifest,schema}_<slug>.json` | target | collection manifest, build schema |
| `results/features/selected_features_<T>_<dataset>.json` | dataset | `dataset` = a slug, or `pool-<hash>` for a pool |
| `results/model/<model>_<dataset>_<T>.skops` | dataset | final model + `.json` sidecar |

Pool name (P-D2): `pool_name(slugs: list[str]) -> str` = `"pool-" + sha256("\n".join(sorted(slugs)))[:8]`, in `data/load.py`. A changed target list gives a new name, so the artefacts of an earlier pool are never overwritten. The full slug list is stored next to every pool artefact (`meta["sources"]`, the feature-list JSON, the model sidecar).

Functions:
- `gh_runs`: `repo_root(repo)` (clone, salt) and `raw_dir(t)` (API cache).
- `load`: `table_path(slug)`, `schema_path(slug)`, `available_targets() -> list[str]` (slugs), replacing `available_repos`.
- `features`: `load_features` / `_file_name` take `dataset: str` instead of `repo`.
- `live`: `model_path(dataset, target, model)`.
- CLIs (`collect-runs`, `build-dataset`): exactly one of `--target <slug>` (looked up with `find_target`), `--targets <path>` (all enabled), or `--repo` with `--workflow` (`--branch` optional; omitted = all branches). There is no default target. `train-final` requires `--dataset`, `predict-run` requires `--model`.

## 4. Collector (`collect/gh_runs.py`)

### 4.1 Query

```python
EVENTS: frozenset[str] = frozenset({"push", "pull_request", "merge_group"})  # carry a code change
```

- Runs query params: `created=<window>`, plus `branch=t.branch` only when `t.branch` is set. No `event` param: one query per event would triple the calls, and the few other runs (schedule, dispatch) are cheap to list.
- `fetch_details` skips runs whose `event` is not in `EVENTS` (no `attempts/1`, no jobs).
- The manifest `queries` list drops `&event=push` and shows `&branch=` only for restricted targets. The manifest adds `runs_per_event`.

### 4.2 Sub-day windows

- Top level: calendar months. File names are `YYYY-MM-DD_YYYY-MM-DD.json`.
- Halving works on `datetime` (UTC) bounds. A window of whole days halves on day boundaries while it spans more than one day. A one-day window halves into hour-aligned datetime windows.
- Sub-day files are named `YYYY-MM-DDTHH-MM-SS_YYYY-MM-DDTHH-MM-SS.json`. The query uses `created=YYYY-MM-DDTHH:MM:SSZ..YYYY-MM-DDTHH:MM:SSZ` (both ends inclusive, so the halves are `[a, mid]` and `[mid + 1 s, b]`).
- The `RuntimeError` is raised only when a **one-hour** window holds >= `MAX_WINDOW_RUNS`.
- `MIN_WINDOW: timedelta = timedelta(hours=1)  # openclaw peaks near 2,000 runs a day; an hour stays far below the 1,000 cap`.
- The re-fetch rule (a file saved before its window ended is fetched again) compares datetimes, not dates.

## 5. Git (`collect/git_changes.py`)

- `clone(repo, branch: str | None)`:
  - With a branch: as today.
  - Without one: a full bare clone of all heads, then `fetch origin +refs/heads/*:refs/heads/* +refs/pull/*/head:refs/pull/*/head --prune`. On an existing single-branch clone, the same fetch adds the missing refs.
  - The destination is `repo_root(repo) / "repo.git"`.
- `default_branch(repo_dir) -> str`: `git symbolic-ref --short HEAD` of the bare clone.
- `merge_base(repo_dir, a, b) -> str | None`: `git merge-base a b`, memoised. None when a SHA is missing.
- `commit_log(repo_dir, branch)` is unchanged and walks the **default** branch for targets without a branch.
- `fetch_heads(repo_dir, shas) -> int`: the SHAs missing from the clone (`git cat-file --batch-check`) are fetched by SHA in batches of 200, each into `refs/ci/<sha>` so `gc` never prunes them. Returns the count still missing. Called by `build` before `git_features`, with every head and `base_sha`.
- Without it, only each branch's and PR's current tip is in the clone: heads replaced by a later push, deleted branches and merge-queue heads are missing (24-50% on PR-heavy targets). GitHub serves them by SHA. Heads still missing after the fetch give NaN git features and are counted (existing `missing` warning).

## 6. Build (`data/build.py`, `data/history.py`)

### 6.1 Run context (`runs_frame`)

New columns per run:

| Column | Kind | Rule |
|---|---|---|
| `event` | diagnostic | `run["event"]` |
| `event_push`, `event_pull_request`, `event_merge_group` | feature | one-hot of `event` |
| `from_fork` | feature | `head_repository.full_name != repo`, compared case-insensitively |
| `is_default_branch` | feature | `head_branch == default_branch` and `event == "push"` |
| `lineage` | id | see below |
| `base_sha` | internal | the base commit as of the run: `pull_requests[0].base.sha`; else the SHA ending a merge-queue branch `gh-readonly-queue/<base>/pr-<n>-<sha>`; else the default branch's last first-parent commit created before the run (`rev-list -1 --first-parent --before=<created_at>`). Never today's branch tip: a PR merged later would have its head on it, giving an empty diff. `pull_requests` is empty for PRs no longer open (88% of n8n PR runs, all servo runs) |

`lineage(run, repo) -> str`, a pure function with its own tests:
- `pull_request`: `head:<head_repository.full_name>:<head_branch>`. Not the PR number: `pull_requests` is filled only while the PR is open, so it is empty for most runs.
- `merge_group`: `pr:<n>` parsed from `head_branch` `gh-readonly-queue/<base>/pr-<n>-<sha>`. Queue runs form their own lineage per PR and do not join the PR's runs; a PR's first queue run is the first of its lineage. Joining them would cost one API call per queued PR.
- `push`: `branch:<head_branch>`.

### 6.2 Drops (`runs_frame`)

- An unsupported event goes to `dropped[f"event_{event}"]`, before the label is read.
- `exclude_jobs`: a first-attempt failure whose failed jobs **all** satisfy `cfg.excludes(name, failed_step)` goes to `dropped["infra_failure"]`. A pattern matches (`re.search`) the job name or `"<job> :: <failed step>"`, so a job that fails both for infra and for code reasons is split by step. A failure whose jobs file has no failed job (the run failed before it tested the code, mostly fork PRs) is also `infra_failure`. A failure without job data (`failed_jobs is None`) is kept and counted in `dropped["no_jobs_kept"]`, a counter that drops nothing.
- `build(cfg: TargetConfig | Target, ...)`: a bare `Target` means `exclude_jobs = ()`, which keeps the old callers working.

### 6.3 Lineage-scoped history (`history_features`)

The completed-before sweep is unchanged. The pointer still moves runs into the known set by completion time. The changes:
- Per lineage, a list of known creation positions, kept sorted with `bisect.insort`: runs complete out of creation order, and "previous" means the latest-created known run.
- From the lineage list: `prev_failed`, `prev_duration`, `prev_queue_delay`, `prev_failed_jobs`, `fail_streak`, `runs_since_last_fail`, `secs_since_last_fail`, `fail_rate_last_{5,10,20,50}`, `overlap_with_last_failed_run`, `_prev_idx`, `_green_idx`.
- New: `lineage_prior_runs` (count of known runs in the lineage), `lineage_first_run` (1 when that count is 0), `secs_since_lineage_prev_run`.
- Unchanged, target-wide: `secs_since_prev_run`, `concurrent_runs`, `ci_load_*`, `fail_rate_{1h..7d}`, `flaky_rate_*`, author and file statistics, `reverted_sha_failed`.
- New, target-wide: `sha_prior_failed`, the label of the latest known run with the same `head_sha` (NaN if none).
- Without a `lineage` column, every run shares one lineage. Single-branch output is then identical to today, which keeps old tests valid.

### 6.4 Change features of non-default heads (`git_features`)

- `anchor`: the head itself if it is on the first-parent log, else `merge_base(base_sha, head)` if that is on the log, else NaN (counted as `missing`).
- File history, author history and `days_since_workflow_change` read the log up to and including `anchor`. This is the existing `searchsorted` on the anchor position, with `"right"` when the anchor is the merge-base, because the merge-base's own change precedes the PR.
- `_change` input:
  - Push run: the head commit's rows of the log, as today. A push to a non-default branch has no log rows, so it uses `diff_numstat(head^, head)`.
  - PR or merge-queue run: `diff_numstat(anchor, head)` (the whole proposed change).
  - `n_parents` and `is_revert` read `git show -s --format=%P%x1f%s%x1f%b head` for heads off the log (memoised).
- `*_since_prev_run` and `*_since_green` use the lineage-scoped `_prev_idx` / `_green_idx`. For the first run of a lineage they are NaN.

### 6.5 T2 and schema

- `data/load.py`: `in_t2(df) -> pd.Series` = `df[PREV_LABEL_COL] != 1` (P-D4: passed, or first of lineage). It is used by `load` and by `validate/live._in_target`. On T2, `prev_failed` stays dropped from `X` (it is 0 or NaN, the same information as `lineage_first_run`).
- The schema adds `target_slug`, `exclude_jobs`, `events`, `id_cols += ["lineage"]`, and `diagnostic_cols += ["event"]`.

## 7. Pooled load and CV (`data/load.py`, `training/cv.py`)

```python
load(targets: str | list[str], target: Label, until: datetime | None = None) -> Dataset
```

- A single slug behaves as today.
- A list: read each table, raise `ValueError` if the `feature_cols` of the schemas differ (name the first mismatching slug), add `source = slug`, concatenate, sort by `created_at` (stable, ties by `source`, `run_id`), then apply `until`, T2, and the dev cut, as today.
- `Dataset` gains `source: pd.Series`, the slug per row (for a single slug, the same constant).
- `meta["dataset"]` (the slug since phase 1) is `pool_name(slugs)` for a list. `meta` adds `sources: list[str]` and `per_source: {slug: {"n_rows", "n_failed"}}`. `meta["repo"]` is dropped for pooled datasets; `training/cv.py` then checks `meta["dataset"]` instead of `meta["repo"]`.
- `run_cv`: when `ds.source.nunique() > 1`, each fold also emits one row per source with `source = slug`. The pooled row has `source = "all"`, and single-source runs keep that column too. Baselines are scored on the same per-source slices.
- E3 needs no new API: `load([s for s in slugs if repo_of(s) != X], ...)` for training, `load(slugs_of(X), ...)` for testing. Notebook helpers stay in the notebook.

## 8. Label check (`analysis/labels.py`)

```python
failed_jobs(cfg: TargetConfig) -> pd.DataFrame
    # one row per (run, failed job): run_id, event, job, failed_step (None if none), excluded (bool)
summary(cfg: TargetConfig) -> dict
    # n_runs, runs_per_event, first-attempt failure rate, infra share, T2 failures,
    # flaky rate, top jobs covering >= 80% of the failures left after exclusion
main()  # CLI `label-check --target <slug> | --targets <path> [--all]`
```

- It reads the raw cache only (`cached_runs`, `first_attempt`, `jobs/`), with no API calls.
- The ranking lists only jobs not yet excluded, so the re-run after `exclude_jobs` shows what is left. A trailing ` [<uuid>]` (servo) is stripped from job names before grouping.
- T2 failures come from the lineage rule of Sec. 6.1/Sec. 6.3 without git: group the supported-event runs by lineage, order by completion, and count failures whose lineage-previous run passed or does not exist.
- The CLI prints the top-jobs table and the summary with the rich console. `--all` includes disabled targets (needed in phase 3, when all are disabled).

## 9. Live and sandbox

### 9.1 `validate/live.py`

- `train_final(dataset: str | list[str], target, model, params)`: the sidecar replaces `repo/workflow_id/branch` with `targets: [{"repo", "workflow_id", "branch", "slug"}]`.
- `retest(path)`: refresh and build each sidecar target, concatenate the post-C rows, and score pooled plus per source. `_retest_runs.csv` gains `source`.
- `predict_run(path, ref, slug=None)`: `slug` is required when the sidecar has more than one target, and must be one of them (`ValueError` otherwise). `_fetch_run`'s SHA lookup drops the `branch` filter when the target has none.
- `predict_run_main` gains `--target <slug>` and `--record <path.jsonl>`. `--record` appends `{slug, run_id, created_at, predicted_at, p_fail, threshold, pred, imputed: [features], scenario: <str|null>}`. The scenario comes from `--scenario`.

### 9.2 `validate/sandbox.py`

```python
report(path: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame]
    # fetch each recorded run's first-attempt conclusion; drop rows predicted after completion
    # (counted); return (scores of model / parrot / majority, per-scenario table)
main()  # CLI `sandbox-report <predictions.jsonl>`
```

Parrot for a sandbox run is read from the run's built row (`prev_failed`), which is recorded at prediction time as an extra field.

## 10. Tests

All tests run offline. API data comes from recorded JSON fixtures, git from tiny repositories in `tmp_path`. Run one file at a time under the memory cap.

| File | Cases |
|---|---|
| `test_targets.py` (new) | valid file; each validation error of Sec. 2 (missing key, unknown key, bad repo, `workflow_id` bool/0/str, empty branch, bad regex, duplicate slug); `enabled_only`; `find_target` of a disabled target |
| `test_collect.py` | `slug` with and without a branch, `/` in a branch; no `branch`/`event` params for an all-branch target; a day with 2,400 runs splits into hour windows and only leaves are written; resume makes zero calls; an hour above the cap raises; unsupported events get no details |
| `test_git_changes.py` | `clone` without a branch fetches `refs/pull/*/head` (local bare "remote" in `tmp_path`); `merge_base` of a branch; `merge_base` with a missing SHA is None; `fetch_heads` fetches an unreferenced commit of the remote into `refs/ci/` and skips present SHAs |
| `test_history.py` (new) | two interleaved lineages: `prev_failed` and `fail_streak` follow their own lineage; the first run of a lineage gets NaN and `lineage_first_run = 1`; a lineage run completed after the current run's creation is not "previous"; `sha_prior_failed` reads only completed runs; no `lineage` column means the same output as today (golden frame) |
| `test_build.py` | `lineage()` for push, PR, fork PR (empty `pull_requests`), merge queue; all-excluded failure dropped as `infra_failure`; mixed failure kept; unsupported event dropped as `event_schedule`; event one-hot and `from_fork` columns |
| `test_load.py` | two toy tables interleave by `created_at`; `source` set; mismatched `feature_cols` raise; `until`, T2 (P-D4: NaN kept) and the dev cut act after pooling; one slug behaves as before |
| `test_cv.py` | per-source rows sum to the pooled row's `n_test` and `n_fail`; single source gives `source = "all"` only |
| `test_labels.py` (new) | top-job coverage stops at >= 80%; infra share after `exclude_jobs`; T2 failure count on a toy lineage set |
| `test_live.py` | multi-target sidecar round-trip; `predict_run` without `slug` on a multi-target model raises; `--record` writes one JSON line |
| `test_sandbox.py` (new) | a row predicted after completion is dropped and counted; per-scenario table counts |

`test_load.py::test_t2` changes on purpose (P-D4): row 0 (NaN `prev_failed`) is now kept.

## 11. Mock data, notebook harness, notebooks

### 11.1 `tests/mock_pool.py`

Synthetic data with the shapes of the real pipeline, small enough for seconds-long tests. The layout functions are monkeypatched to `tmp_path`: `gh_runs.RAW_ROOT`, `RESULTS_DIR`, `load.PROCESSED_DIR`, `SCHEMA_DIR`, `features.OUT_DIR`, `live.MODEL_DIR`, and the figure directory of `plots`.

```python
SOURCES: list[str]  # 3 mock slugs: two workflows of one repo (one restricted to a branch) + one other repo

mock_runs(slug, n=200, seed=0) -> list[dict]
    # raw run objects: push / pull_request / merge_group / schedule, several lineages, a fork PR,
    # re-runs (run_attempt 2), cancelled runs, overlapping completion times
write_raw_cache(tmp, slug, runs) -> None
    # runs/<window>.json, attempts/, jobs/ (with steps, incl. an infra-only and a mixed failure)
write_table(tmp, slug, n=800, fail_rate=0.2, seed=0) -> None
    # data/processed/<slug>.csv + schema with the real feature_cols list, time-overlapping
    # across slugs, failure probability tied to 3 features so models can learn something
mock_pool(tmp, monkeypatch) -> list[str]
    # patches all paths, writes registry + tables + manifests for SOURCES, returns SOURCES
```

- `write_table` takes the feature names from one source of truth: a frozen copy of the real `schema["feature_cols"]` in `tests/fixtures/feature_cols.json`, refreshed whenever phase 4 changes the features. Notebooks then see real column names.
- The tables are sized so the pooled data has >= 5 folds x `MIN_TEST_FAILURES` failures. Per source it can be fewer, which exercises the "fewer than the minimum" paths.

### 11.2 `tests/nb_harness.py`

```python
run_notebook(path: pathlib.Path, stubs: dict[str, Callable], skip_tags=("live",)) -> dict
    # exec every code cell, in order, in one namespace; returns the namespace
```

- It reads the notebook with `nbformat` (already in the venv through `jupyter`), with no kernel, so it is fast and debuggable.
- The namespace gets a checking `display` stub instead of the no-op. It renders nothing and enforces the display rule (plan): a call that shows a non-text object (`DataFrame`, `Series`, `Styler`, a figure, a `dict` or `list`) must pass a caption first, a non-empty `str` or `IPython.display.Markdown`, in the same call: `display(Markdown("**Runs per source (pooled T2)**"), df)`. Otherwise it raises `AssertionError` naming the object type.
- `plots.save` is patched to return the figure without writing a PNG. The patch asserts a non-empty `fig.layout.title.text` (set with `px.*(title=...)` or `fig.update_layout(title=...)`). A figure passed to `display` is checked the same way.
- Before a cell runs, its source is parsed with `ast`. A last statement that is a bare expression (other than a call) would render an object with no caption in Jupyter, so it fails the test.
- `live` cells run against stubs of the drivers they call (`collect`, `build`, `live.retest`, `live.predict_run`, `fetch_windows`, `git.clone`). The stubs return mock-pool outputs. `http.get` is patched to raise, so a cell that reaches the network without a stub fails the test instead of calling GitHub.
- Slow knobs are patched for the test only: `tune.GRID` and `forest.GRID` reduced to one point, forest `n_estimators = 5`. The notebook code is unchanged.
- An error reports the notebook, the cell index and the cell's first line.

### 11.3 `tests/test_notebooks.py`

- One test per notebook, parametrized: `run_notebook` on `mock_pool`. After it, assertions on key names in the namespace (e.g. `results["T2"]` has a `source` column with every mock slug plus `"all"`; `sel.features` is non-empty; the pooled `t1.meta["sources"] == SOURCES`).
- A grep test: no notebook contains `N8N_MASTER`, `REGIME` or `REPO =`.
- Marked `@pytest.mark.notebook`, run separately: `uv run pytest -m notebook tests/test_notebooks.py`.

### 11.4 Editing notebooks

- Cells are developed as plain code, run through `run_notebook` on a scratch copy, then written to the real notebook with `nbformat` (or the NotebookEdit tool), keeping the cell order of plan Sec. 5.3.
- Edited code cells are saved with `outputs = []` and `execution_count = None`. Unchanged cells keep their outputs until the user re-runs.
- Every cell keeps the notebook's style: imports in the setup cell, display through `display(...)`, figures through `plots.save(fig, "<notebook>_<name>")`, comments only for decisions.
- Titles and captions name the data and the view, not the code: "Monthly failure rate per source (T1, all runs)", not "df2". Per-source variants of a pooled view repeat the pooled title with the source slug.
- `Markdown` is imported in each notebook's setup cell (`from IPython.display import Markdown, display`).
- CLI tables (`rich.table.Table`) pass `title=`; `labels.main` gets its titles in N4.

## 12. ASCII cleanup (plan phase 10)

Runs once, after N6. Refactoring only: no logic, data path or output value changes.

Scope (files we write; data from GitHub is out of scope):
- `projects/2/src/`, `projects/2/tests/`, `projects/2/scripts/`, `projects/2/config/`, `projects/2/pyproject.toml`, `projects/2/**/README*`.
- `projects/2/notebook/*.ipynb`: cell sources only. Outputs are cleared or regenerated by the user, since they can hold data text.
- `docs/specs/project-2/` and `docs/plans/projects/2/`.
- Excluded: `data/`, `results/`, `.venv/`, lockfiles, recorded API fixtures under `tests/fixtures/` (they mirror GitHub payloads).

Replacements (one table, applied by hand or by a throwaway script in the scratchpad, then reviewed in the diff):

| Character | Replacement |
|---|---|
| em/en dash | `-` |
| approximately | `~` |
| greater/less or equal | `>=`, `<=` |
| multiplication sign | `x` |
| arrows | `->`, `<-` |
| plus-minus | `+/-` |
| section sign | `Sec.` |
| subscript digits (A12) | plain digits |
| curly quotes, ellipsis | `"`, `'`, `...` |
| non-breaking space | space |

Anything not in the table is rewritten case by case. A column name or registry key with non-ASCII would change the data contract; there is none today, and if one appears it is renamed together with a rebuild and recorded in the plan.

`tests/test_ascii.py` (new, permanent): walks the scope above and fails on any byte above `0x7F`, naming file, line and character. For `.ipynb` it checks the cell sources only.
