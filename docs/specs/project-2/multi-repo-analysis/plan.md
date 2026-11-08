# Multiple Repository Analysis — Plan

Implements [SPEC.md](SPEC.md). Code-level details (modules, signatures, layout, tests) are in [implementation-backbone.md](implementation-backbone.md). The project-wide rules of the [main plan](../../../plans/projects/2/dt-ci-failure-prediction.md) and the [project backbone](../../../plans/projects/2/implementation-backbone.md) still apply: leakage boundary, portability rule, coding style, one module at a time on request. One exception for this feature: Claude edits the notebooks itself (§5).

**Documentation rule.** Plan and backbone references (decision ids such as `P-D4`, phase or step numbers, `§` sections) and other planning remarks stay in the plan. They do not go into code, comments, docstrings, tests, or notebooks. The code and its short comments must document themselves: a comment states a non-obvious reason or constraint in plain terms, in one or two lines, never the history of a decision.

**Testing rule.** Every driver change and every changed notebook cell is first tested offline on **mock data**: small synthetic run caches and per-target tables, generated in `tmp_path`. This keeps tests fast and independent of the API and the multi-hour builds. Real data only enters in phase 5 (operations) and when the user runs the notebooks.

## 1. Probe data (2026-11-07)

Counts come from the Actions API `total_count` with no branch filter. 2,500 is the API's display cap, so `≥` marks a capped value. Failures are counted on the latest attempt, so first-attempt failures (the label) are slightly higher.

| Target | Id | Triggers (`on:`) | Oct 2026 runs | Failures | push / PR / merge_group | Notes |
|---|---|---|---|---|---|---|
| n8n-io/n8n `ci-pull-requests.yml` | 33322401 | pull_request, merge_group | ≈ 12,500 (≥ 2,900 a week) | ≈ 20% | 0 / 2,286 / 614 a week | no push runs; the master pushes are in `ci-master.yml` (33323319) |
| astral-sh/ruff `ci.yaml` | 32130301 | push [main], pull_request | ≈ 2,800 | ≈ 16% | 67 / 590 / 0 a week | |
| astral-sh/uv `ci.yml` | 82881044 | push [main], pull_request | ≈ 5,700 | ≈ 15% | 163 / 1,162 / 0 a week | one sampled main failure was `test uv publish` (infra) |
| windmill-labs/windmill `backend-check.yml` | 133951533 | push (all branches) | ≈ 3,200 | ≈ 9% | 579 / 0 / 0 a week | feature branches arrive as pushes |
| windmill-labs/windmill `frontend-check.yml` | 45664283 | push, merge_group | ≈ 3,100 | ≈ 10% | 556 / 0 / 0 a week | |
| sharkdp/fd `CICD.yml` | 11202521 | push [master], pull_request | 117 | 12 | 27 / 90 / 0 | small |
| github/gh-aw `ci.yml` | 180858647 | push [main], schedule | 1,803 | 739 | 1,076 / 0 / 0 | no PR runs; scheduled runs are dropped; sampled failure was a pre-flight check (infra) |
| Homebrew/brew `tests.yml` | 9591 | push [main, master], pull_request, merge_group | 910 | 78 | 190 / 497 / 223 | |
| openclaw/openclaw `ci.yml` | 209874334 | push [main], pull_request, schedule | ≈ 57,000 (1,915 on 2026-10-08) | ≈ 20% | 406 / 1,456 / 0 a day | above 1,000 runs a day: needs sub-day windows |
| tokio-rs/tokio `ci.yml` | 1972898 | push [master, tokio-*.x], pull_request | 617 | 176 | 122 / 495 / 0 | sampled main failure was a kernel build (infra) |
| sveltejs/svelte `ci.yml` | 14783 | push [main], pull_request | 511 | 167 | 40 / 471 / 0 | |
| nushell/nushell `ci.yml` | 7519865 | push [main, patch-release-*], pull_request | 668 | 57 | 182 / 486 / 0 | |
| servo/servo `main.yml` | 3876879 | push [main], pull_request, merge_group | ≈ 3,000 | ≈ 28% | 119 / 373 / 210 a week | 100+ jobs a run; aggregate `Result` job; job names carry run ids; many sampled failures were tool installs (infra) |

Findings:
- For most workflows, push runs only cover the default branch. "All branches" mostly means PR and merge-queue runs. These multiply the failures: svelte has 167 failures a month instead of 2, tokio 176 instead of 8, nushell 57 instead of 3.
- openclaw alone has about 1.6 times the volume of all the other targets together.

## 2. Budget

Estimated per **month of collection window**, from the probe:

| | All targets except openclaw | openclaw |
|---|---|---|
| Runs | ≈ 34,000 | ≈ 57,000 |
| API calls (pages, `attempts/1` for re-runs, jobs of failed first attempts) | ≈ 8,000 (≈ 1.7 h at 5,000/h) | ≈ 14,000 (≈ 2.9 h) |
| Clone (one per repository, measured 2026-11-08) | ≈ 20 min, 14.6 GB for 11 repositories | not measured |
| Build: git work per run (measured) | ≈ 1.7 h for ≈ 210,000 runs, fetches included (2–27 ms a run against the base as of the run; commit-log walks 0.1–177 s per target) | not measured |
| Fetching superseded heads by SHA | ≈ 1,400 heads a minute (ruff: 1,386 in 58 s); 24–50% of PR-heavy targets' heads | not measured |

The git side is small next to the API side: the build's git work and the head fetches together stay under 2 h for the 6-month window (`scripts/build_time.py`, `results/collection/build_time.json`). A 6-month window without openclaw costs about 10 h of API time. The full ~13 months would take about 22 h. Collection resumes after interruption, so it can run over several sessions.

## 3. Decisions (confirmed 2026-11-07)

| Id | Decision | Why |
|---|---|---|
| P-D1 | Collection window: the last **6 months** before C, the same for all targets | fits the budget (≈ 10 h of API time); the same span for all targets keeps the dev segment mixed |
| P-D2 | openclaw is **disabled** in the first pool. A changed target list later is a **new pool**, run as a separate experiment | 1.6 times all the other targets together; it would dominate the pool and the budget. Pool artefacts are named by their source list (backbone §3.2), so a second pool never overwrites the first |
| P-D3 | add n8n `ci-master.yml` (33323319) as a second n8n target | adds the post-merge runs that `ci-pull-requests.yml` lacks |
| P-D4 | T2 = runs whose previous run **in the same lineage** passed, **or** that are the first run of their lineage | see below |
| P-D5 | Small targets: pool-eligible with ≥ 20 T2 failures in the window; E1/E3-eligible with ≥ 50 | small targets with clean labels still add failures to the pool |
| P-D6 | Failures that do not come from the code (infra-only) are dropped, not relabelled (FR5). They are also absent from the history of later runs | they are noise: no pre-run feature can predict them, and their code outcome is unknown |

**P-D4 and the single-workflow view.** The pool is analysed as if it were one CI workflow, and the lineage rule is consistent with that:
- History features, `prev_failed` included, are computed per source **before** pooling (FR6). Pooling only decides the row order for the dev segment and the folds.
- Membership in T2 is a property of each run, computed inside its own workflow. A real single workflow with pull requests computes it the same way.
- The alternatives are worse. "Previous run of the whole workflow" is, in a PR-heavy workflow, the outcome of an unrelated PR, so T2 and Parrot become noise. "Previous run of the pool" mixes repositories and has no meaning at all.
- First runs of a lineage are kept. T2 removes failures that **continue** a known failure, which Parrot already predicts. A PR's first run continues nothing. Dropping first runs would remove a large share of PR runs and the first failure of every PR. `lineage_first_run` lets the model tell them apart.
- On a single-branch target, the rule differs from today's T2 in one row (the very first run).

## 4. Phases

Each phase ends with its checks. A phase is not started before the previous one's checks pass, except where noted.

Each driver phase is followed by its notebook step (N1–N6, §5.2), so the changed code can be run from the notebooks before the next phase starts. Order:

1 → 2 → **N1** → 3 → **N2** → 4 → **N3** → 6 → **N4** → 5 → 7 (**N5**) → 8 → 9 → **N6**

Phase 6 is code only (tested on mock data), so it runs **before** phase 5. Phase 5 then runs from the finished `collection.ipynb`, and its outputs feed the already-edited analysis notebooks.

### Phase 1 — Target registry and storage layout

Tasks:
1. Add `projects/2/config/targets.json` with every candidate, all `enabled: false` (backbone §2).
2. Key the raw cache, datasets, manifests, schemas and models by **target slug** (repository + workflow + optional branch), not by repository (backbone §3).

Checks:
- Registry validation tests pass (backbone §10, `test_targets.py`).
- The full test suite passes.

### Phase 2 — Collector: all branches, all supported events

Tasks:
1. Query without branch and event filters when the target has no branch. Keep a `branch` filter for restricted targets (backbone §4.1).
2. Split windows below one day when needed, down to one hour (backbone §4.2).
3. Fetch details (`attempts/1`, jobs) only for runs of supported events.
4. Make the clone hold every branch head and every `refs/pull/*/head`, so PR and fork heads are reachable (backbone §5).

Checks:
- Collector tests pass, including sub-day splitting and the resume test (backbone §10).
- Live smoke test: collect 2 days of `sveltejs/svelte` `ci.yml`. The cache holds `push` and `pull_request` runs from more than one branch, and a second run makes zero API calls.

### Phase 3 — Label check (gate)

Run for **every** candidate on a recent **28-day sample** (2026-10-08 to 2026-11-04), before any build change.

Procedure per candidate:
1. `collect-runs --target <slug> --since 2026-10-08 --until 2026-11-04`.
2. Read the workflow file at the default branch. Record the triggers and what the jobs do. A workflow that only publishes or deploys fails.
3. `label-check --target <slug>` (backbone §8) lists failed jobs and failed steps over first-attempt failures, ranked by count until they cover ≥ 80%.
4. Classify each listed job as `code` or `infra` by its **failing step**:
   - `infra`: publish, release, deploy, docker push, tool or SDK installs, pre-flight or dependency checks, "prepare"/"setup" steps, benchmark uploads, device or emulator provisioning.
   - An aggregate job that fails whenever another job fails (servo's `Result`) is `infra`, so it never keeps a run labelled on its own.
   - A failed job with no failed step (timeout, cancelled reusable workflow) is classified by its name.
5. Write the `infra` jobs into the target's `exclude_jobs`. Use anchored regexes on the stable part of the name, because names can carry run ids (servo: `[bc91d74b-...]`).
6. Re-run `label-check`. It reports, after the exclusion: first-attempt failure rate, infra share, T2 failures in the sample, flaky rate, and the share of runs per event.
7. Estimate the T2 failures in the P-D1 window: monthly failure count from the API × (1 − infra share) × the T2 share of failures in the sample.
8. Measure the build time per run on the sample (phase 4 code is not needed: time `git diff --numstat` and `git merge-base` on 200 sampled PR heads).

Pass criteria:
- The workflow runs the project's build or tests on a supported event.
- Infra share ≤ 30% of first-attempt failures.
- Estimated T2 failures in the window: ≥ 20 for the pool, ≥ 50 for E1/E3 (P-D5).
- The flaky rate is reported. Above 50% of failures it is recorded as a risk, not a failure.

Results (2026-11-08; sample 2026-10-08 to 2026-11-04). Est. T2 failures = T2 failures in the sample × 182.6 / 28 days. Infra = failures whose failed jobs all match `exclude_jobs`, plus failures without any failed job (the run failed before it tested the code). Patterns match `<job>` or `<job> :: <failed step>`:

| Target | Workflow role | Top failed jobs (≥ 80%) | `exclude_jobs` | Infra share | Est. T2 failures | Flaky rate | Events | Verdict |
|---|---|---|---|---|---|---|---|---|
| n8n-io/n8n ci-pull-requests | PR + merge-queue build, typecheck, unit/DB tests | Unit tests (fail-if-tests-failed), Typecheck (`Setup Node.js` runs `turbo typecheck`), DB integration, Install & Build, Format check | `^Required Checks$` | 5% | 9,254 | 9% | PR 82%, merge_group 17% | pass |
| n8n-io/n8n ci-master | master push build + tests | Verify npm-install closure of changed packages (code: caused by the change), Unit tests (24.x) | — | 0% | 913 | 0% | push 100% | pass |
| astral-sh/ruff ci | PR + main build, tests, lint | cargo test (macOS, linux, windows), cargo hawk, clippy | `^all required checks passed$`, `^security-review / security review$` | 26% | 1,793 | 12% | PR 51%, dispatch 42% (dropped), push 6% | pass |
| astral-sh/uv ci | PR + main tests, lint, system tests | cargo test (windows ×3, linux, macOS), system Python tests, hawk, clippy | `^all required jobs passed$`, `^test uv publish$`, `^review / security review$` | 8% | 2,348 | 27% | PR 86%, push 13% | pass |
| windmill-labs/windmill backend-check | push (all branches) cargo check | check_ee_full / check_ee `cargo check` | `^check_ee :: Run actions/checkout` (EE repo checkout; always fails with check_ee_full, so 0 runs dropped) | 0% | 672 | 0% | push 79%, workflow_run 21% (dropped) | pass |
| windmill-labs/windmill frontend-check | push (all branches) npm check | npm_check | — | 1% | 293 | 0% | push 80%, workflow_run 20% (dropped) | pass |
| sharkdp/fd CICD | PR + master tests, clippy, fmt per target | Run tests (per target), clippy, fmt | `^Publish to Winget$` | 8% | 59 | 0% | PR 76%, push 24% | pass (E1 borderline) |
| github/gh-aw ci | main push tests and checks | guidance tests, integration tests, workflow lock drift | `^Validate latest release alignment$` | 14% | 333 | 0% | push 60%, schedule 40% (dropped) | pass; risk: 62% failure rate, T2 share 13% (main stays broken) |
| Homebrew/brew tests | PR + merge-queue + main brew tests, audits | `brew tests` shards, formula audit | ` :: Install Subversion$`, ` :: Run brew test-bot --only-setup$` | 11% | 391 | 8% | PR 59%, merge_group 23%, push 19% | pass |
| openclaw/openclaw ci | — | — | — | — | — | — | — | deferred (P-D2, not sampled) |
| tokio-rs/tokio ci | PR + master tests, clippy, docs, FreeBSD, io_uring | FreeBSD tests, clippy, spellcheck, fmt, docs, minrust, io_uring QEMU tests | `^Test io_uring on Linux [\d.]+ / build :: Download & build Linux kernel$` | 21% | 411 | 10% | PR 81%, push 19% | pass |
| sveltejs/svelte ci | PR + main lint, typecheck, tests | Lint (lint, type check), TestNoAsync, TSGo, Tests | — | 13% | 378 | 2% | PR 94%, push 6% | pass |
| nushell/nushell ci | PR + main fmt, check, tests, fuzz | Linux/Windows Test, Check, fuzz Fmt | — | 29% (12 of 41: fork PRs without jobs) | 157 | 15% | PR 72%, push 28% | pass (infra near the limit) |
| servo/servo main | PR + merge-queue + main build, lint, unit tests, WPT | Lint (Tidy, Clippy), Linux build, WPT, unit tests | `^Result$`, ` / Perf analysis`, ` / Mossel Scenario test`, ` / Benching HarmonyOS` | 17% | 3,469 | 0% | PR 55%, merge_group 29%, push 16% | pass |

Checks:
- Every row has a verdict, and `targets.json` has `enabled` and `exclude_jobs` set to match.
- The budget (§2) is re-estimated with the measured build time. If it exceeds the P-D1 budget, the user re-decides the window before phase 5.

Build timing (2026-11-08, 200 sampled runs per target, bases as of the run, missing commits fetched by SHA): ≈ 1.7 h for all 13 targets, servo and n8n PRs most of it. After the fetch, no sampled run lacks its head or base, and every diff succeeds. Phase 3 is done (2026-11-08). Missing heads in the clone: n8n PR 37%, ruff 50%, uv 48%, servo 41%, windmill 24–31%, brew 29%, tokio 26%, svelte, nushell and fd 11–15%, n8n master and gh-aw 0%. Most are same-repository PR heads that a later push replaced. All tried were fetchable by SHA, so phase 4 adds the fetch, and the window stays at 6 months.

### Phase 4 — Build: lineages, events and PR changes

Tasks (backbone §6):
1. Run context per row: `event_*` one-hot columns, `from_fork`, `is_default_branch`, `lineage`.
2. Drop runs of unsupported events and runs failed only in `exclude_jobs` jobs, each counted in `n_dropped`.
3. Lineage-scoped previous-run features: `prev_*`, `fail_streak`, `runs_since_last_fail`, `secs_since_last_fail`, the reference runs of the `*_since_prev_run` / `*_since_green` diffs, plus the new `lineage_prior_runs` and `lineage_first_run`. Window rates, load, flaky, author and file statistics stay target-wide.
4. `sha_prior_failed`: the outcome of the latest run completed earlier on the same `head_sha` (any lineage).
5. Before the change features: fetch, by SHA, every run head missing from the clone (backbone §5, `fetch_heads`). Change features of PR and merge-queue runs over `merge-base(base, head)..head`. File and author history are anchored at the merge-base when the head is not on the base branch's first-parent line.
6. T2 per P-D4.

Checks:
- New `test_history.py` cases: two interleaved lineages, where `prev_failed` follows its own lineage; the first run of a lineage gets NaN and `lineage_first_run = 1`; `sha_prior_failed` only reads completed runs.
- `test_build.py`: a failed run whose failed jobs all match `exclude_jobs` is dropped and counted; one with a non-matching failed job is kept.
- `test_git_changes.py`: `merge_base` on a tiny repository with a branch; change features of a PR head equal the branch's total diff.
- The full test suite passes.

### Phase 5 — Full collection and build (operations)

Runs after phase 6 and N4 (see the order above).

Tasks:
1. `collect-runs --targets config/targets.json --since <C − window> --until <C>` for the enabled targets, on one collection day C. It can be resumed over several sessions; C is the `--until` value. Equivalent: run sections 1–2 of `collection.ipynb`.
2. `build-dataset --targets config/targets.json`.

Checks:
- Each manifest records the window, queries, counts per window, and API calls.
- Each schema records `n_dropped` per reason, including `infra_failure` and `event_<name>`.
- Per target, the share of rows with NaN git features (unreachable heads) is ≤ 10%, or it is recorded as a risk.

### Phase 6 — Pooled loading and per-source reporting

Runs before phase 5 (see the order above).

Tasks (backbone §7):
1. `load` accepts several target slugs, concatenates them, sorts by `created_at`, and records `source` per row.
2. `run_cv` adds one row per (fold, source) next to the pooled row when the dataset has sources.

Checks:
- `test_load.py`: two toy tables interleave by time; `source` is set; mismatched feature columns raise `ValueError`; T2 and `until` act after pooling.
- `test_cv.py`: per-source rows add up to the pooled row's `n_test` and `n_fail`.

### Phase 7 — Experiments E1–E3

Runs in `feature_selection.ipynb` and `tree.ipynb`. Their E1/E3 sections are notebook step N5 (§5.2), edited at the start of this phase. The user runs them on the real pool.

Tasks:
1. Feature selection on the pooled dev segment (same procedure as the main plan, D6/D7).
2. E1: CART and forest per E1-eligible source, current protocol.
3. E2: CART and forest on the pool, reported pooled and per source, with Parrot (lineage-scoped `prev_failed`) and majority per source.
4. E3: for each E1-eligible repository X, select features and tune on the pool without X, then test on all rows of X.
5. Statistics: paired Wilcoxon and A₁₂, E2 against E1 per source.

Checks:
- Every CV test fold has ≥ `MIN_TEST_FAILURES` failures pooled. Per-source fold failure counts are shown.
- Repository-proxy check: the top splits of the pooled CART are listed, and per-source results are compared with per-source majority.

### Phase 8 — Live validation on real runs

Tasks (backbone §9.1):
1. `train-final` on the pool. The sidecar lists the target slugs.
2. `retest` refreshes every source in the sidecar and scores runs created after C, pooled and per source.
3. `predict-run --target <slug> <run id | url>`.

Checks:
- `test_live.py`: the sidecar round-trips several targets; `predict-run` refuses a target that is not in the sidecar.

### Phase 9 — Sandbox (E4)

Tasks:
1. The user creates the sandbox repository: a small codebase with real tests and a CI workflow on `push` and `pull_request`. It is added to the registry as a disabled target, so it is never trained on.
2. Warm-up: at least 30 normal changes (PRs and pushes) before the first scored run.
3. Scenario matrix:
   - Change shape: small or large diff; source only or source plus tests; one or many directories; workflow file touched or not.
   - Broken or not: every shape appears both ways. A breakage is a real defect that a test catches.
4. Scenario order is shuffled with a fixed seed and recorded.
5. For each scored run: `predict-run --record results/sandbox/predictions.jsonl` after the run is created and before it ends. The conclusion is added once the run ends (`sandbox-report`, backbone §9.2).
6. At least 40 scored runs, with the breakage rate stated.

Checks:
- The report shows the confusion matrix, MCC, Parrot and majority on the same runs, the per-scenario table, and the imputed features per prediction.
- No scored run is predicted after its completion time (checked from the recorded timestamps).

## 5. Notebook changes

The notebooks in `projects/2/notebook/` analyse the pooled dataset, the list of enabled targets in `config/targets.json`, as if it came from one CI workflow. Next to every pooled result they report the same statistic per source (repo:workflow pair).

### 5.1 Rules

- **Ownership.** Claude edits the `.ipynb` files. This overrides the project backbone's "Claude does not edit notebooks" for this feature. Edited cells are saved without outputs; the user re-runs the notebooks on real data.
- **Test before edit.** Each new or changed code cell is first run by the notebook harness (backbone §11) on the mock pool. Only cells that pass go into the notebook.
- **Network cells.** Cells that call the API or clone repositories (collect, build, retest, predict-run, the in-progress run) get the cell tag `live`. The harness replaces their driver calls with mock-data stubs, so the cells after them still run. A call that would reach the network without a stub fails the test.
- **Shared setup.** Every notebook opens with the same constants cell: `SOURCES = [slug(c.target) for c in load_targets()]` (enabled targets) and `DATASET = pool_name(SOURCES)`. The data for that list is loaded once.
- **n8n-specific content goes.** The Node 25.x regime (`REGIME`), n8n job categories, and observations written for n8n master no longer describe the data. Regime analysis becomes a per-source failure-rate view. `## Observations` cells become placeholders, filled in after the real run.
- **When.** In steps N1–N6 (§5.2), each right after the driver phase whose code it calls. A step edits only the sections listed for it; the other sections stay as they are until their step.
- **Harness first.** `tests/nb_harness.py` and `tests/test_notebooks.py` (backbone §11) are written in N1. Each step extends `tests/mock_pool.py` with what its sections read, and adds its notebook's sections to `test_notebooks.py`. Until a notebook's last step, its test runs only the sections edited so far (the cells up to the last edited one).

### 5.2 Order

Phases 1–2 removed `N8N_MASTER`, so every setup cell that imports it fails until its notebook's first step. Before phase 5 there are no built tables, so the analysis notebooks run on the user's machine only after phase 5. Until then they are checked on the mock pool.

| Step | After | Notebook sections (§5.3) | What the user can run then |
|---|---|---|---|
| N1 | phase 2 | `collection.ipynb`: setup, 1. Collect (`live`), manifest table | collect the 28-day phase 3 sample from the notebook; manifest table per source |
| N2 | phase 3 | `collection.ipynb`: new "Label check" section after 1: `labels.summary` per candidate (top jobs, infra share, T2 failures, flaky rate, events) | the phase 3 classification and its re-run after `exclude_jobs` |
| N3 | phase 4 | `collection.ipynb`: 2. Build per source (`live`, schema table, `n_dropped`), and sections 3–6 **per source** (coverage, events and lineages, label balance, lineage streaks, flaky), each source read with `load(slug, ...)` | build the sample tables and inspect lineages, events, drops, T2 per target |
| N4 | phase 6 | `collection.ipynb`: pooled `load(SOURCES, ...)` and the pooled side of sections 2–6. `eda.ipynb`: all. `feature_selection.ipynb`: setup, 1–8 except the per-source check. `tree.ipynb`: setup, 1–7, 9 (pooled and per-source rows) | after phase 5: the whole collection, EDA, pooled feature selection and CV |
| N5 | start of phase 7 | `feature_selection.ipynb`: per-source check (E1 lists). `tree.ipynb`: 5. E2 vs E1 tests, 6. repository-proxy check, 8. E1 and E3 | E1–E3 on the real pool |
| N6 | phase 9 | `validation.ipynb`: all, including 6. Sandbox | freeze, re-test, case studies, in-progress run, sandbox report |

Phase 8 has no step of its own: `validation.ipynb` 1–4 and 6 are edited together in N6, once `sandbox.report` exists. If the user wants to freeze and re-test before the sandbox is ready, sections 1–4 move to a step after phase 8.

### 5.3 Per notebook

| Notebook | Section | Change |
|---|---|---|
| `collection.ipynb` | setup | `TARGET_REPO = N8N_MASTER` becomes `CONFIGS = load_targets()`, `SOURCES`, `COLLECTION_UNTIL` (C), `SINCE` = C − P-D1 window |
| | 1. Collect | loop over `CONFIGS` (`live`). Manifest table, one row per source: runs, API calls, windows, largest window (assert < 1000), runs per event |
| | new: label check | `labels.summary(cfg)` for every registry entry (`load_targets(enabled_only=False)`): one summary row per target, plus its top-jobs table |
| | 2. Build | loop over `CONFIGS`, skipping built tables (`live`). Schema table per source: runs, failed, features, `n_dropped` per reason (`infra_failure`, `event_*`). Assert equal `feature_cols` across sources. Pooled `t1, t2 = load(SOURCES, ...)` |
| | 3. Coverage | monthly runs and failure rate, pooled and per source (pivot table, plus `px.line` with one line per source). The pooled `runs_per_week` keeps the dev-end line. The n8n regime cells are replaced by "per-source failure-rate shifts" (the months where a source's rate moves more than 2× its median) |
| | new: events and lineages | runs and failure rate per event, per source; runs per lineage (distribution); share of first-of-lineage runs; fork share |
| | 4. Label balance | T1 vs T2, pooled and per source |
| | 5. Failure streaks | streaks computed **within a lineage**: a pooled, interleaved order mixes unrelated runs |
| | 6. Re-runs and flaky | pooled and per source |
| `eda.ipynb` | setup | pooled `load(SOURCES, T)`. `frame()` adds `source`. `period` becomes `dev` and `fold 1..5` (the CV test segments) instead of the n8n regime periods |
| | 1. Profile | pooled profile, plus a NaN% table per source × feature (shows unreachable heads and missing history) |
| | 2. Failure rate over time | pooled weekly plot, plus per-source lines; the period table per source |
| | 3. Failed-job categories | job names read from each source's table (`table_path(slug)`). Generic categories only (`test`, `lint`, `build`, `other`); crosstab category × source |
| | 4. Separating features | Spearman with the label on the pooled dev segment, plus a source × feature heatmap for the top features (does a feature mean the same in every source?) |
| | 4.2 Binary features | includes `event_*`, `from_fork`, `lineage_first_run`; failure rate if 1 / if 0, pooled and per source |
| | 5. Correlations | pooled dev segment, unchanged otherwise |
| | 6. Drift | medians per period (dev, folds), plus medians per source for the top features |
| | new: repository proxy | a depth-3 CART predicting `source` from the features on the dev segment. Its accuracy and splits show how easily the model can tell sources apart (risk §6) |
| `feature_selection.ipynb` | setup | `REPO` becomes `SOURCES` and `DATASET`. `EXCLUDE` (D6, chosen on n8n) is re-decided from the pooled EDA |
| | 1–6 | unchanged procedure on the pooled dev segment and the pooled folds |
| | new: per-source check | `select_features` on each E1-eligible source's own dev segment. Jaccard of each list with the pooled list. The per-source lists are frozen under their slug, for E1 |
| | 7. T1 and Parrot | the Parrot family is now lineage-scoped (`prev_failed` of the same PR or branch); the text says so |
| | 8. Freeze | writes `selected_features_<T>_<DATASET>.json` plus the per-source lists |
| `tree.ipynb` | setup | `data = {t: load(SOURCES, t)}`, `feats = load_features(t, DATASET)` |
| | 1–3 | unchanged (pooled dev split) |
| | 4. CV | `run_cv` rows carry `source`. Pooled tables and plots filter `source == "all"`. New: mean ± std per source × method, and the failure count per fold and source |
| | 5. Statistical tests | on the pooled rows; new: E2 vs E1 per source (Wilcoxon, A₁₂) |
| | 6. What the tree uses | unchanged, plus the repository-proxy check: top splits of the last-fold tree and per-source results against per-source majority |
| | 7. Errors | confusion pooled and per source |
| | 8. Sensitivity (regime) | replaced by **E1** (CART per E1-eligible source, its own list) and **E3** (leave one repository out) |
| | 9. Forest | same changes as sections 4–7 |
| `validation.ipynb` | setup | `REPO` becomes `SOURCES`, `DATASET` |
| | 1. Freeze | `train_final(SOURCES, t, m)`, `model_path(DATASET, t, m)`. The model table lists the sources |
| | 2. Re-test | pooled plus per-source table (`live`). `retest_vs_cv` reads the pooled CV rows |
| | 3. Case studies | cases picked per source; `predict_run(path, run_id, slug)` (`live`) |
| | 4. In-progress run | first source with a run in progress; no `branch`/`event` filter for all-branch targets (`live`) |
| | new: 6. Sandbox (E4) | `sandbox.report(predictions.jsonl)`: confusion matrix, MCC, Parrot and majority, per-scenario table, imputed features |

### 5.4 Checks

- `test_notebooks.py` (backbone §11) runs every notebook top to bottom on the mock pool, with `live` cells stubbed, and passes.
- No notebook still references `N8N_MASTER`, `REPO`, or `REGIME` (grep check).
- Every pooled table or figure has its per-source counterpart, listed in §5.3.

## 6. Risks

- **Repository proxy.** Scale differs per source (durations, load, churn). The tree may first identify the source, then predict its base rate. Mitigation: the phase 7 checks.
- **Dominant source.** n8n (and openclaw, if enabled) can set most splits. Report per-source sizes. Down-weighting only if E2 shows the problem.
- **Unreachable heads.** The clone holds only the current tip of each branch and PR. Heads replaced by a later push, deleted branches and merge-queue refs are missing: 24–50% of sampled heads on PR-heavy targets (2026-11-08). GitHub still serves them by SHA, so the build fetches them first (phase 4 task 5). What is still missing gives NaN git features and is counted per target (phase 5 check).
- **Cancelled runs.** New pushes often cancel running PR runs. These runs are dropped (not a label) and are missing from history.
- **Young repositories.** openclaw (created 2025-11-24) and gh-aw (2025-08-12) have less history, and their early months may differ from later ones.
- **Training time.** The pool can be 10 times larger than n8n master. The from-scratch CART grid and the 100-tree forest grow with it. Measure the time of one fold on the real pool before running all sections. If it is too slow, reduce the forest grid; do not subsample rows.
- **Label drift.** `exclude_jobs` comes from a 28-day sample. Jobs renamed earlier in the window can escape it. `label-check` on the full window after phase 5 confirms the infra share.
