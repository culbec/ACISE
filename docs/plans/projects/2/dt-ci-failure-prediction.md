# Project 2 - From-scratch decision tree for CI failure prediction on self-collected GitHub Actions data

Status: **ready for implementation (2026-11-02)**. D1-D7 decided (see end).
Theme: *Prediction of CI failures using ML-based algorithms.*
Companion: [implementation-backbone.md](implementation-backbone.md).

## Assignment recap (Lecture 1, p.5)

- Pick a real SE activity and build an ML solution that automates, supports, or improves it.
- **Data collection:** collect our own data with a strategy that fits the activity. The data must be representative and large enough to train on. An existing dataset is not allowed.
- **ML approach:** choose a technique that fits the data and the problem.
- **Implementation:** preprocessing, training, evaluation. Document the process and the reasons for each choice.
- **Validation:** test on real-world examples, ideally in a live development setting.
- **Report:** problem, data collection, approach and implementation, validation results and feedback, future work. Written after the project is finished.
- Deadline: week 8. Instructor allows reusing the project 1 theme (CI failure prediction).
- Own constraint: the ML algorithm is implemented **from scratch** (no `sklearn` estimator in the final model).

## Core idea

Collect the GitHub Actions history of one large OSS repository. Build one row per CI run, using only information known **before** the run starts. Train a from-scratch CART decision tree to predict whether the run will fail.

Why a decision tree:

- It is on the course list (Lecture 1, p.4 and p.12).
- It is explainable: each prediction is a readable path of threshold tests. This links back to project 1, where the flow-control programs were also IF-THEN-ELSE trees.
- It is small enough to implement and test fully in the time available.
- It needs no feature scaling, handles mixed feature types, and supports class weights for the imbalanced label.

### Alternatives considered

| Algorithm | Verdict |
|---|---|
| **CART decision tree** | **Chosen.** Explainable, cheap to implement, suits tabular data. |
| Random forest (bagging over our own CART) | **Optional extension.** About 40 extra lines on top of the tree. Usually more accurate on CI data, but less explainable. Add only if time allows; it gives a stronger comparison point. |
| Logistic regression (gradient descent) | Simple, but a linear model on skewed count features needs heavy preprocessing. Weaker story. |
| Naive Bayes | Trivial to implement. The independence assumption breaks on churn features, which are strongly correlated. |
| Neural network | Too much effort for ~5k tabular rows, and not explainable. |

Research question: *Can a from-scratch, depth-limited decision tree trained on pre-run GitHub Actions data predict CI failures of a large OSS repository better than the history baselines (previous-outcome, majority class)? Which pre-run signals drive its decisions?*

## Data collection strategy

### Source repository

**n8n-io/n8n** (TypeScript monorepo, ~209k stars, default branch `master`). Probed on 2026-11-02 with an authenticated `gh` session:

| Workflow | id | Trigger | Runs visible | Recent mix (last 100) |
|---|---|---|---|---|
| `CI: Master (Build, Test, Lint)` | 33323319 | `push` to `master` | ~1.5k in 2025, >=2.5k in 2026 | 82 success / 16-18 failure |
| `CI: Pull Requests (Build, Test, Lint)` | 33322401 | `pull_request`, `merge_group` | >=2.5k per year (count capped) | 62 success / 11 failure / 9 cancelled |

Observations from the probe:

- The runs endpoint returns at most **1000 results per query**, and `total_count` caps at 2500. Collection must page through **date windows** (`created=YYYY-MM-DD..YYYY-MM-DD`), small enough to stay under 1000 runs each.
- A master run has ~29 jobs: build and format, lint, unit-test shards on two Node versions, integration tests, and a Slack notification. The job list shows which part failed.
- The PR workflow has `cancel-in-progress` concurrency, so it has many `cancelled` runs. It also runs from merge refs and merge-queue SHAs. That makes PR data noisier and harder to align with diffs.
- Master CI ignores `packages/@n8n/task-runner-python/**` (`paths-ignore`).
- n8n squash-merges PRs, so one master push is normally one commit. The commit title follows conventional commits with the PR number (`fix(core): ... (#12345)`).
- Run **logs** expire after 90 days. Run and job **metadata** stay available. Logs are not needed: they describe the current run, so they would leak the label.

### Scope (D1, decided)

Dataset: **`CI: Master` only**, all `push` runs available at collection time. PR runs are out of scope. Master holds changes that were already reviewed and approved, and they are not mixed with work-in-progress PR runs.

- One linear, chronological history, like project 1. The expanding-window CV can be reused unchanged.
- Each run maps to one commit, so diff features come from a **local git clone** at no API cost.
- Failures on master happen *after* PR CI passed (flaky tests, integration effects, merge interactions). They are harder to predict and closer to what a developer cannot see in advance. This is a known characteristic, not a defect. The report must discuss it.
- The collector takes the repository, workflow id, and branch as parameters, so it can be pointed at another repository later (see "Porting candidates").

### Collection pipeline

1. **Runs.** `GET /repos/n8n-io/n8n/actions/workflows/33323319/runs?created=<window>&per_page=100`, paginated, one window per month. Save raw JSON to `data/raw/runs/<YYYY-MM>.json`. A window already on disk is skipped (resumable). Fields kept: `id`, `run_number`, `run_attempt`, `event`, `head_sha`, `head_branch`, `created_at`, `run_started_at`, `updated_at`, `status`, `conclusion`, `actor.login`, `triggering_actor.login`, `head_commit.{message, author}`.
2. **First-attempt label.** The runs endpoint returns the latest attempt. For runs with `run_attempt > 1`, fetch `/actions/runs/{id}/attempts/1`. The label is the **first-attempt** conclusion, which is what the developer saw first. Record `was_rerun` and `flaky = first failed and a later attempt passed` as diagnostics, never as features.
3. **Jobs of failed runs.** `/actions/runs/{id}/jobs` only for failed runs (~15%). Store the failed job names and the total job count. These feed the *previous run's* failed-job features. A mapping from job names to categories (`build`, `lint`, `unit`, `integration`, `other`) is used **only in the EDA**, never as a feature, because job names are specific to each repository.
4. **Diffs.** `git clone --single-branch --branch master https://github.com/n8n-io/n8n data/raw/n8n.git` once (a full clone: `--numstat` needs file contents, and a blobless clone would fetch them one by one). One `git log --numstat` pass gives files, additions, deletions, parents, and message for every commit. `git rev-list` and `git diff` between the current SHA and the previous (and last green) run's SHA give the since-previous-run features; `git log -- <files>` gives file history. No API calls.
5. **Build table.** One chronological pass over the runs computes history features from **earlier rows only**. Writes `data/processed/n8n_master.csv` plus `schema.json` (order column, label column, positive label, previous-label column, nominal columns, columns to drop).

API budget (re-measured 2026-11-02: ~10.2k master runs since 2025-09, ~40% failed first attempts): ~150 calls for runs, a few hundred for attempts, ~4k for jobs. The limit is 5000 per hour, so collection takes about 1 hour and must survive a restart (every response is cached on disk). The token comes from `gh auth token` at runtime and is never written to disk.

### Labels and filtering

- Keep `status == completed` and `conclusion in {success, failure}`. Drop `cancelled`, `skipped`, `startup_failure`, `action_required`, and log the counts.
- `failed = 1` for `failure`, `0` for `success`.
- **T1**: all runs. **T2**: runs whose previous run passed (first failures). Same targets as project 1, so the two projects can be compared.

### Features (pre-run only)

Windows: **time** windows `{1,3,6,12,24}h` plus `7d`, and **count** windows `{5,10,20,50}` runs. They overlap on purpose. Selection on the dev segment (mRMR plus correlation pruning at rho = 0.8, reused from project 1) keeps the windows that carry signal and removes the redundant ones.

| Group | Source | Features |
|---|---|---|
| Run | runs API | `queue_delay` (`run_started_at - created_at`), `merge_to_ci_delay` (`created_at - head_commit.timestamp`), `ci_load_{1,3,6,12,24}h` (master runs created in the window), `concurrent_runs` (earlier runs still in progress at `created_at`), `secs_since_prev_run` |
| CI history | earlier runs | `prev_failed` (Parrot input), `fail_streak`, `runs_since_last_fail`, `secs_since_last_fail`, `fail_rate_{1,3,6,12,24h,7d}`, `fail_rate_last_{5,10,20,50}`, `flaky_rate_{24h,7d}`, `prev_failed_jobs`, `prev_duration`, `prev_queue_delay` |
| Author | earlier runs, git | `author_is_bot` (`[bot]` in the head commit's author name or e-mail), `author_prior_runs`, `author_prior_fail_rate`, `author_prior_commits`, `author_file_familiarity` (share of touched files the author changed before). Logins are **hashed** and never used as features. |
| Change since previous run | git | `commits_since_prev_run` (`git rev-list --count <prev_sha>..<head_sha>`), churn and file counts since the previous run and since the last **green** run (`git diff --numstat`) |
| Change size and type | git | `files_changed`, `additions`, `deletions`, `churn`, `src_churn`, `test_churn`, `test_file_ratio`, `src_files`, `test_files`, `config_files` (includes lockfiles and manifests), `doc_files`, `ci_files` (`.github/workflows/**`), `n_dirs_touched`, `dir_entropy` |
| Commit | git | `n_parents` (merge commit), `is_revert` (message written by `git revert`), `reverted_sha_failed` (the reverted commit's run failed) |
| File history | git, earlier runs | `file_age_mean`, `file_prior_changes`, `file_distinct_authors`, `secs_since_files_last_changed`, `file_fail_prob_{max,mean}` (failure rate of earlier runs that touched the same files), `overlap_with_last_failed_run` |
| Workflow and release | git | `workflow_changed`, `days_since_workflow_change`, `days_since_last_tag` |

**Not features.** `created_at` is the order column: it sorts the runs for the time-split CV and is dropped from `X`, like `build_nums` in project 1. `hour` and `weekday` are left out, and so is the absolute date, because a split on the date only learns which period a run belongs to. Commit-message conventions (`commit_type`, `is_breaking`), `msg_len`, PR label names, and job-name categories are left out too, because they depend on each repository's habits.

**"Previous" means completed before.** Master runs overlap (one run takes ~10 min). Every history feature reads only runs whose `updated_at` (completion time) is before the current run's `created_at`. `prev_*` refers to the last such run, not to the run just before in creation order. A unit test checks this on a toy table with overlapping runs.

**Portability rule.** Every feature must be computable from any GitHub repository's Actions API and git history, without knowing its directory layout, package names, or job names. Change *location* (which package or folder was touched) is deliberately left out: it would tie the model to n8n's monorepo layout. Files are classified as src, test, config, doc, or CI with one generic rule set based on extensions and path patterns (`test`/`spec`/`__tests__`, lockfiles and manifests, `*.md`, `.github/workflows/`). The rule set lives in the code once and is the same for every repository.

**Deferred.** Sizes of the touched files (`git cat-file -s`); add only if selection asks for more change signal. Run logs are never used: they describe the current run and leak the label.

**Leakage drop:** current run duration, current job results, the current run's `updated_at`, `run_attempt`, `flaky`. History features follow the completed-before rule above.

### Ethics and data hygiene

- Public data only, read through the official API under GitHub's terms, with the authenticated rate limit respected.
- Author identities are pseudonymised (salted hash) before anything is saved under `data/processed/`.
- Raw and processed data stay out of git (`data/raw` and `data/processed` in `.gitignore`). The report lists the collection date range and the exact API queries, so the collection can be repeated.

## Modelling

### Algorithm: CART from scratch

Reference: Breiman, Friedman, Olshen & Stone, *Classification and Regression Trees*, Wadsworth, 1984. The implementation follows the book: binary splits, Gini impurity, class priors and costs (which give the class weights), and minimal cost-complexity pruning (Ch. 3). The only change is the split search: sorting once and sweeping class counts gives the same splits as the book's search, only faster. Other code (e.g. `Valdecy/CART` on GitHub) may be consulted for how to print rules, but no code is copied from it.

- Binary splits `x_j <= t` on numeric and ordinal features. The nominal feature `commit_type` is one-hot encoded when the table is built.
- Impurity: Gini (default) or entropy, as a hyperparameter.
- Split search: per feature, sort once and sweep cumulative class counts. That is `O(n log n)` per feature per node, vectorised in NumPy.
- Stopping: `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease`.
- Imbalance: `class_weight` (`None` or `"balanced"`), applied as sample weights inside the impurity.
- Pruning: minimal cost-complexity pruning (`ccp_alpha` path). Alpha is chosen on the inner validation split.
- Output: `predict_proba`, a decision threshold tuned on validation, `feature_importances_` (weighted impurity decrease), `export_text()`, and `explain(x)`, which returns the path of tests for one prediction.
- Missing values: median imputation per fold, as in project 1. No in-tree surrogate splits.
- Scaling: **none**. Trees do not change under monotone transforms, so the project 1 `QuantileTransformer` step is skipped. Thresholds then stay in raw units, which keeps them readable.

### Correctness check

`sklearn.tree.DecisionTreeClassifier` is used **only in tests**, as a reference. On toy and real data with the same hyperparameters, both must give the same training-set predictions and impurity per node. Where ties are broken differently, they must give the same accuracy. sklearn is never part of the reported model.

### Evaluation protocol (reused from project 1)

- Dev segment: the first ~20% of runs, used for EDA and feature selection only.
- Expanding-window time-series CV with 5 folds after the dev segment. Each train set includes the dev rows. An inner validation tail is used for pruning, the threshold, and early hyperparameter choices.
- Hyperparameter search: a small grid (`max_depth in {3,4,5,6,8}`, `min_samples_leaf in {5,10,20}`, criterion, class weight), scored on validation MCC.
- Metrics: precision, recall, F1, MCC, accuracy, confusion matrix, plus domain metrics (`% builds saved`, `% failures caught`).
- Baselines: Parrot (previous outcome), majority class, the sklearn tree (reference only), and the random forest if the extension is built.
- Statistics: paired Wilcoxon over folds, Vargha-Delaney A12.
- Sensitivity check (D5): the same CV repeated without the 2026-02-10 -> 2026-06-10 regime rows.

### Validation on real-world examples (D4, decided)

1. **Develop on everything collected now.** Collect all master runs available at collection date `C`. CV, tuning, and analysis all use this data.
2. **Freeze the model.** Train the final tree, with the hyperparameters chosen by CV, on all runs up to `C`. Save it with **skops** (`skops.io.dump`) to `results/model/cart_n8n_master.skops`, next to `schema.json` and the feature list. The tree is our own class, so loading needs `skops.io.load(path, trusted=[...])` listing our types. Keep the model a plain object (NumPy arrays and scalars) so skops can serialise it without custom hooks.
3. **Re-test on new data before hand-off.** Collect the runs created after `C`, build their features using the stored history plus the new runs (history features may use earlier new runs, never later ones), load the saved model **unchanged**, and report the same metrics next to the CV metrics. This is the "live environment" check: the model has never seen these runs.
4. **Live prediction CLI.** `predict-run <run_id|sha>` loads the saved model, fetches one master run, builds its features, and prints the prediction, the probability, and the decision path in plain language. It is used for case studies in the report: a few correct and a few wrong predictions, with the path explaining each one.
5. **User feedback (light).** Show 5-10 explained predictions to 2-3 fellow students. Ask two questions: is the explanation understandable, and would they trust it to skip or prioritise a run? Record the answers for the report's "validation and user feedback" section.

## Analysis and plots

Reuse the project 1 plot helpers: label balance, fail streaks, distribution by label, correlation heatmap, relevance scores, stability, drift, metrics box plot, confusion matrix. New plots:

| Figure | Purpose |
|---|---|
| Runs and failure rate per week | Shows that the data covers the period and that the rate drifts over time |
| Failed-job category share (EDA only) | Shows what kind of failures master CI has |
| Tree diagram (top levels) | Main explainability figure |
| Feature importance (impurity decrease, averaged over folds) | Which signals the tree uses |
| Validation curve over `max_depth` and `ccp_alpha` | Overfitting control |
| New-data re-test vs CV metrics | Generalisation to unseen future runs |

## Reuse from project 1

Project 1 is finished and stays frozen. Its utilities are **copied into the project 2 package and generalised**, not imported (D2). The two copies stay separate, even where they are identical, until a later decision merges them into a shared library. Details are in the backbone.

| Project 1 module | Reuse | Change needed |
|---|---|---|
| `viz/plots.py` | almost as is | output dir from package root; label names as parameters |
| `analysis/features.py` | as is | `NOMINAL` taken from `schema.json`; `Dataset` import path |
| `metrics/metrics.py` | as is | `parrot()` reads the previous-label column from the schema, not from project 1 constants |
| `training/cv.py` | mostly | median imputation only (no scaling for trees), no GP seeds; keep folds and the dev split |
| `stats/tests.py` | as is | none |
| `utils/logging/log.py` | as is | logger root name |
| `data/load.py` | rewrite | schema-driven loader; same `Dataset` dataclass and T1/T2 logic |
| `gp/*`, `solutions/*` | not reused | - |

## Porting candidates (future work, optional)

The same pipeline can be pointed at another repository to test whether the approach transfers. Probed on 2026-11-02 (push runs on the default branch, recent sample):

| Repository | Default branch | Push test workflow | Recent mix | Verdict |
|---|---|---|---|---|
| `anomalyco/opencode` | `dev` | `check-test` (id 396447938); older history under `test` (id 193014382) | 119 success / 21 failure | **Best candidate.** Real test runs on push with a useful failure rate. The workflow was renamed recently, so the history of the two workflow ids must be joined. |
| `microsoft/vscode` | `main` | `Code OSS (node_modules)` (id 172517798) | 310 success / 12 failure / 10 cancelled | **Weak.** The push runs only build `node_modules`; the main tests run on Azure Pipelines, outside GitHub Actions. Failures are rare (~4%). |
| `anthropics/claude-code` | `main` | none (only a workflow-security check) | 60 success | **Not usable.** The repository has no CI on source code. |

Porting means re-running collection and build with new parameters (repository, workflow ids, branch). No feature mapping is needed (see the portability rule). Two experiments are possible: (a) apply the saved n8n model as it is (transfer), or (b) retrain on the new repository (does the method generalise).

## Literature to read and cite (verify each entry before citing)

- Breiman, Friedman, Olshen & Stone, *Classification and Regression Trees*, Wadsworth, 1984. The algorithm reference for `tree/cart.py`.
- Saidani et al., *Predicting CI build failures using evolutionary search*, IST 2020 (already in `papers/`).
- Hassan & Wang, *Change-aware build prediction model for stall avoidance in continuous integration*, ESEM 2017. Uses decision trees on Travis data; the closest prior work to this approach.
- Ni & Li, *Cost-effective build outcome prediction using cascaded classifiers*, MSR 2017.
- Jin & Servant, *A cost-efficient approach to building in continuous integration* (SmartBuildSkip), ICSE 2020.
- Chen et al., *BuildFast: History-aware build outcome prediction for fast feedback and reduced cost in CI*, ASE 2020.
- Sun, Habchi & McIntosh, *RavenBuild*, FSE 2024 (project 1 dataset).
- Decan et al., *On the use of GitHub Actions in software development repositories*, ICSME 2022. Background on GitHub Actions.

## Risks

| Risk | Mitigation |
|---|---|
| Master failures are mostly flaky, so the signal is weak | Report the flaky share (from reruns). Compare against Parrot honestly. A negative result with an explanation is still a valid result. |
| Workflow definition changed over time (jobs added or renamed) | Use the workflow file's commit history to mark major changes. Add `days_since_workflow_change` if the drift plot shows breaks. |
| Environment-caused failure regime (Feb-May 2026, Node 25.x job) inflates history-based results | D5: keep it, use T2 as the main target, repeat the CV without the regime rows. |
| Too few failures in T2 | Check counts after collection. Fall back to T1 as the main target if T2 has fewer than ~300 failures. |
| API caps hide older runs | Date windows of one month or less. Log the per-window count and assert it is below 1000. |
| History starts only in 2025-09 for this workflow id | Collection starts at the workflow's `created_at`; ~10k runs are enough for a tree. |
| skops refuses to load the custom tree class | Pass our types in `trusted=`. Keep the model free of lambdas and open handles. A test round-trips the model and checks that the predictions are identical. |

## Decisions

- **D1 (decided).** `CI: Master` runs only. No PR runs, and no pull-request data (PR metadata or PR CI runs) as features: one workflow keeps collection simple and cheap.
- **D2 (decided).** Copy the project 1 utilities into `projects/2` and generalise them there. Keep both copies separate until a shared library is decided on.
- **D3 (decided, 2026-11-03).** Random forest extension built (`tree/forest.py`): the single tree is weak on T2 (CV MCC 0.063) and does not beat Parrot on T1, so the forest is the stronger comparison point. It is evaluated in `tree.ipynb` with the same folds, features and validation protocol as CART. CART stays the explainable model; the forest answers how much accuracy the explanation costs.
- **D4 (decided).** Develop on all data available at collection date `C`. Save the final model with skops. Before hand-off, re-test the saved model on runs created after `C`. Porting to another repository is optional future work.
- **D5 (decided, 2026-11-02).** Keep the failure regime of 2026-02-10 -> 2026-06-10 in the data. In that period a Node 25.x unit-test job failed on most runs, so the regime holds 54% of all failures (58% failure rate vs. 16% outside). The cause is the CI environment, not the pushed change. The regime is real CI behaviour, so it stays. Two consequences follow: T2 is the main target, because it is less affected (42% vs. 12% failure rate); and the tree notebook repeats the CV without the regime rows as a sensitivity check, with the same code and hyperparameter grid. Dropping the regime (cleaner signal, but it redefines the data) and ignoring Node-version jobs in the label (breaks the portability rule) were rejected.
- **D6 (decided, 2026-11-03).** Feature selection inputs, from the EDA. (a) `merge_to_ci_delay` is excluded: it jumps from ~4 s to ~1,100 s on 2026-01-18 (a change in how n8n merges), right after the T2 dev segment ends, so any threshold would split time periods, not changes. (b) `flaky_rate_24h` and `flaky_rate_7d` are excluded: re-runs happened mostly early (118 of 149 non-zero T2 values are in the dev segment, from ~9 flaky runs), so they look relevant on the dev rows and are ~0 afterwards. Both exclusions go through `select_features(exclude=...)` and are recorded in the selection meta. (c) `clean()` checks near-constant columns on percentile ranks instead of min-max values, so heavy-tailed counts (`deletions`, `commits_since_*`) are no longer dropped by one outlier. (d) `author_is_bot` stays dropped as a rare binary on the dev rows (0.5%), although it reaches 10% later; forcing it in would use later rows for a choice. (e) The rare-flag rule in `clean()` (minority value < 1% of dev rows) applies to every two-valued column, not only 0/1: `n_parents` is 2 only for the weekly release-tag merges (13 of 1,525 T2 dev runs) and is dropped like any rare flag.
- **D7 (decided, 2026-11-03).** As in project 1, a frozen T2 feature that drops out of the top 20 in 3 or more of the 5 CV fold trains (`analysis.features.drift`) is removed from the list. This uses labels after the dev segment to shrink the list; it can never add a feature. The report states this next to the CV results. T1 is not drift-filtered.
