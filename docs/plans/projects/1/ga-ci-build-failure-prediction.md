# Project 1 — GP-evolved flow-control programs for CI build-failure prediction

Status: **ready for implementation** once the reading on Saidani et al. (IST 2020) is finished (2026-10-17).
Main inspiration: Saidani, Ouni, Chouchen, Mkaouer, *Predicting continuous integration build failures using evolutionary search*, IST 128 (2020) — `papers/IST2020-saidani.pdf`.

## Assignment recap (Lecture 1, p.4)

- **(a)** Install an open-source ML framework and implement one technique from the list on an existing dataset. Chosen technique: **genetic programming (GP)**, a type of genetic algorithm.
- **(b)** Solve the same problem with at most `N` if/case statements, where `N` is the number of attributes. It must match or beat (a).
- Evaluate both with several metrics, at least precision, recall, and accuracy.
- Deadline: week 4. Deliverables: Python scripts and Jupyter notebooks.

## Core idea

GP evolves **programs**, not weight vectors. Each individual is an IF-THEN-ELSE tree over CI features, which is the same representation Saidani et al. use. Parts (a) and (b) therefore produce the same kind of artefact:

- (a) a flow-control program found by evolution;
- (b) a hand-written flow-control program with at most `N` statements, derived from the best GP programs and from domain knowledge.

How this differs from Saidani et al.:

- different CI system and data: CircleCI, not Travis;
- different feature set: includes code-level (AST) change features;
- explicit feature selection before evolution;
- a hard budget of `N` statements on the evolved programs;
- the first-failure target (see below).

Research question: *Can a short, human-readable program distilled from the GP output keep the GP's predictive performance under the same `N`-statement budget?*

## Dataset

**RavenBuild replication package** (Sun, Habchi, McIntosh, *RavenBuild: Context, Relevance, and Dependency Aware Build Outcome Prediction*, FSE 2024).

- Source: Zenodo, DOI `10.5281/zenodo.10719029`, file `ravenbuild_data.zip` (1.1 GB, CC BY 4.0).
- CI system: CircleCI. It is a subset of Gallaba et al.'s 23M-build CircleCI dataset.
- Covers 22 open-source Java projects from 2018–2023, about 11.2k builds. Builds triggered on the same commit are already merged into one row.
- Files used:
  - `data/circleci_repos/features/<project>.csv`: one row per build, with about 90 precomputed features.
  - `data/circleci_java_builds.csv`: raw build list. Use it only to check the label direction.
  - `data/general_java_repo_information.csv`: project metadata.
- **Do not load** the `.txt` files under `circleci_repos/{circleci,github,raw_files,tests,...}`. They are Python pickles, and unpickling untrusted data can execute code. The feature CSVs are enough.
- The paper's own baselines (BuildFast, RavenBuild-CR, and Parrot, which predicts the last outcome) are reference points only. They are not reproduced here.

### Columns

| Role | Columns |
|---|---|
| Order / id | `build_nums` (chronological per project) |
| Label | `now_label` (outcome of this build) |
| Previous outcome | `last_label` |
| Change size | `src_churn`, `test_churn`, `line_added`, `line_deleted`, `files_added`, `files_deleted`, `num_commits` |
| Change type | `src_files`, `test_files`, `config_files`, `doc_files` |
| Code-level (AST) | `src_ast_diff`, `test_ast_diff`, `class_*`, `met_*`, `field_*`, `import_*` |
| Context | `is_main`, `time_interval`, `day_of_week`, `time_of_day`, `merge_commits`, `fix_commits` |
| People | `committers`, `by_core_member`, `gh_team_size`, `user_changed` |
| History | `fail_ratio_*`, `last_fail_gap_avg`, `consec_fail_builds_*`, `commits_on_files`, `file_fail_prob_*` |
| Previous build | `pr_*` (previous build's errors, tests, duration, files), `prev_srcchurn`, `prev_testchurn`, `log_*` |
| Change intent | `change_type`, `type_changed`, `action_changed`, `relevance`, `is_corrective`, `is_addition`, `is_non_func`, `is_perfective`, `is_preventive`, `is_same*`, `has_old_change` |
| **Drop: leakage** | `build_duration` (only known after the build runs) |

### Known facts and checks before modelling

- [ ] **Label direction.** `1` probably means pass, since testcontainers-java is 98% `1`. Confirm by joining with `outcome` in `circleci_java_builds.csv`. fresco and cbioportal would be 80–88% failing under that reading, so check them as well.
- [ ] Make sure the `pr_*` and `log_*` columns describe the **previous** build, never the current one.
- [ ] Per-project sizes vary a lot (65–3,156 builds). Candidate projects for the main experiments:

| Project | Builds | `now_label=0` | First failures |
|---|---|---|---|
| atlasdb | 3156 | 1021 | 597 |
| rskj | 1042 | 198 | 125 |
| cassandra | 493 | 180 | 81 |
| spring-cloud-aws | 684 | 265 | 72 |
| gradle-baseline | 489 | 111 | 64 |

testcontainers-java has 1,034 builds but only 16 failures, which is too few. Use it at most for cross-project tests.

## Problem framing

- **Unit:** one build (merged per commit).
- **Label:** `failed = 1`, `passed = 0`. Remap from `now_label` after the direction check.
- **Use case:** predict "will pass" so a build can be skipped or deprioritised. Each correct skip saves CI time; each missed failure is costly.
- **Targets:**
  - **T1, all failures.** Expect `last_label` to dominate, which is just Parrot. Report T1 for completeness.
  - **T2, first failures** (headline): restrict to builds with `last_label = pass` and predict whether the build fails. `last_label` is constant on T2, so it is dropped automatically.
- **Leakage rule:** never use `build_duration`, the current build's logs, or anything else known only after the build runs.

## Evaluation protocol (shared by (a) and (b))

### Time-series cross-validation (instructor requirement)

Builds are ordered in time, so shuffled k-fold CV would train on future builds and leak project state. Use **expanding-window time-series CV** instead: `sklearn.model_selection.TimeSeriesSplit(n_splits=5)` on rows ordered by `build_nums`, per project. Saidani et al.'s "online validation" (exp1–5) follows the same idea.

```
rows ordered by build_nums ──────────────────────────────────────▶
dev | fold 1: train ▓▓▓▓▓▓▓▓      test ░░
dev | fold 2: train ▓▓▓▓▓▓▓▓▓▓    test ░░
... | fold 5: train ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ test ░░
```

- **Development segment** (`dev`): the first ~20% of each project, before every test fold. Use it for:
  - feature selection, which freezes the feature list and `N`;
  - GP hyperparameter tuning;
  - building the structure of program (b).

  The dev segment comes before all test folds, so none of these choices sees test data.
- **CV folds** (`TimeSeriesSplit(n_splits=5)` over the rows after `dev`, with `dev` also prepended to each fold's train):
  - In each fold, the last 20% of the fold's train is the **inner validation** split. It is used for early stopping and for picking the final program from the hall of fame or the Pareto front.
  - Each test fold is used once.
- Report each metric as **mean ± std across the 5 folds**, with the per-fold values in a table.
- **T2 caveat:** first failures are sparse. If a test fold of a small project has fewer than about 10 failures, use `n_splits=3`, or report only the larger projects.
- Optional sensitivity check: a single 2/3–1/3 temporal holdout, as in RavenBuild, to compare with the CV numbers.
- **Metrics** (fail class): precision, recall, F1, MCC, accuracy, and the confusion matrix. Also explain why accuracy misleads under class imbalance.
- **Domain metrics:** `% builds saved` and `% failures caught`.
- **Reference points:** Parrot (`prediction = last_label`; trivial on T2), a majority-class predictor, and a depth-limited `DecisionTreeClassifier` using the same `N` features.
- **Stochasticity:** run 10 independent GP runs per fold with different seeds, giving 5 × 10 = 50 runs per project and target. Report mean ± std over all runs and per fold.
- **Statistics:** compare (a) and (b) with a Wilcoxon signed-rank test on the paired per-fold scores. Use the GP's per-fold median over its seeds. With 5 folds the test has low power, so also report effect size (Vargha–Delaney A₁₂) and where (b) falls within the GP's distribution in each fold.

## Feature selection

Following IBM's taxonomy (https://www.ibm.com/think/topics/feature-selection): supervised selection, starting with fast **filter** methods, then **hybrid** checks against an **embedded** method. Fit everything on the **development segment only**, which comes before all CV test folds. The selected feature list and `N` are then frozen for every fold, and are shared by (a) and (b). Re-running selection inside each fold would give a different `N` per fold and break the fixed statement budget. Scaling (min-max) and imputation are re-fitted on each fold's train.

### Step 1 — Clean (filter: missing-value ratio, variance threshold)

1. Drop leakage columns (`build_duration`) and identifiers (`build_nums`).
2. Drop columns with a missing-value ratio above 30%. Impute the rest with the train median.
3. Drop constant and near-constant columns with `VarianceThreshold`, on min-max-scaled data, threshold about `1e-3`. Also drop binary columns whose minority value occurs in less than 1% of rows. Many `pr_*` and AST columns are mostly 0.

### Step 2 — Relevance (filter: pick the statistic by variable type, as IBM recommends)

The label is binary (categorical). Features are mixed.

| Feature type | Statistic | Library |
|---|---|---|
| numeric → categorical label | ANOVA F (`f_classif`) **and** Kendall's τ | `sklearn.feature_selection`, `scipy.stats.kendalltau` |
| binary → categorical label | chi-square (`chi2`) **and** information gain | `sklearn.feature_selection` |
| all | mutual information (`mutual_info_classif`) | `sklearn.feature_selection` |

- Use rank-based or information-based measures rather than Pearson. Churn and count features are heavily skewed, and mutual information also captures threshold effects, which is what an if-rule expresses.
- Final relevance score: the mean rank across the statistics that apply to each feature.

### Step 3 — Additional steps from the dataset analysis

These address what the dataset analysis showed.

1. **Redundancy pruning (mRMR-style).** The dataset contains near-duplicate feature families:
   - `src_churn` / `line_added` / `src_ast_diff`;
   - the 5 `fail_ratio_*` columns;
   - `file_fail_prob_{max,avg,sum}`;
   - `consec_fail_builds_{max,avg,sum}`;
   - `pr_*` against `log_*`.

   Walk the relevance ranking and keep a feature only if its |Spearman ρ| with every feature already kept is ≤ 0.8. Keep **one representative per family**.
2. **Target-specific selection.** Run Steps 1–3 separately for T1 and T2. On T2, `last_label` is constant and is removed. On T1, keep it but report results with and without it, to show how much is pure Parrot.
3. **Budget cut.** Keep the top **N = 15–20** features, so `N` statements is still a size a human can write by hand. Report the chosen `N`.
4. **Hybrid sanity check (embedded).** Compare the subset with `RandomForestClassifier` impurity or permutation importance, or with an L1-regularised `LogisticRegression`, fitted on train. Large disagreements, such as a feature in the forest's top 5 that the filters dropped, go in the notebook with a decision.
5. **Stability check.** Repeat the selection on 5 bootstrap resamples of train and report how often each feature is selected. Prefer features that are selected consistently.
6. **Per-project vs pooled.** Start per project with atlasdb (largest). If the subsets differ a lot between projects, document that. A pooled subset is an option for cross-project experiments.

Output: `selected_features_<target>_<project>.json`, holding the feature list, scores, and `N`.

## (a) Genetic programming

### Prerequisites

- Python ≥ 3.10, a virtual env (`python -m venv .venv`), and `requirements.txt` with pinned versions.
- Extract only the CSVs from `ravenbuild_data.zip` into `data/raw/` (never the pickles), and keep `data/` out of git.
- Complete the label-direction check and feature selection (above) first.

### Libraries

| Library | Use |
|---|---|
| **DEAP** (`deap.gp`, `deap.tools`, `deap.algorithms`) | GP framework: strongly typed primitive sets, tree generation, crossover/mutation, bloat limits, NSGA-II selection (`tools.selNSGA2`) |
| numpy, pandas | data handling, vectorised evaluation of programs |
| scikit-learn | feature selection, scaling, metrics, decision-tree / random-forest references |
| scipy | Kendall/Spearman, Wilcoxon test |
| matplotlib | convergence plots, Pareto fronts, trade-off curves |
| jupyter | notebooks |

`gplearn` is not used, because its `SymbolicClassifier` evolves numeric expressions, not typed boolean flow-control programs.

### Program representation (strongly typed GP, `gp.PrimitiveSetTyped`)

- **Types:** `float` (scaled feature values and thresholds) and `bool` (conditions and outputs).
- **Terminals:**
  - one `float` argument per selected feature, min-max scaled to `[0, 1]` on train;
  - an ephemeral random constant `thr ~ U(0, 1)` of type `float`;
  - the `bool` constants `True` and `False`.
- **Primitives (flow control):**
  - `if_then_else(bool, bool, bool) -> bool`
  - `and_(bool, bool) -> bool`, `or_(bool, bool) -> bool`, `not_(bool) -> bool`
  - `lt(float, float) -> bool`, `ge(float, float) -> bool`
- Output `True` means "predict failure".
- **Statement budget:** a *statement* is one `if_then_else` or comparison node. Programs with more than `N` statements are rejected using `gp.staticLimit` with a custom key that counts those nodes. A max tree height of about 8 is a second bloat guard.
- Compile programs with `gp.compile`. Evaluate them vectorised with numpy: use `np.where` for `if_then_else` and `&`, `|`, `~` for the boolean operators.

### Fitness — compare two variants

1. **Single-objective** (headline): `MCC(train) − λ · statements / N`. Use `tools.selTournament`.
2. **Multi-objective**, as in Saidani: maximise TPR and minimise FPR with NSGA-II (`tools.selNSGA2`). From the Pareto front, pick the point closest to `(TPR=1, FPR=0)`, deciding on the validation split.

Optional cost-aware variant: `saved_builds − α · missed_failures`. Sweep `α` to plot the trade-off curve.

### Operators and hyperparameters

- Initialisation: `gp.genHalfAndHalf` with depth 1–4.
- Crossover: `gp.cxOnePoint`.
- Mutation: `gp.mutUniform` (sub-tree), `gp.mutNodeReplacement`, and `gp.mutEphemeral` (threshold tweaks).
- Elitism with `tools.HallOfFame`.
- Hyperparameters to explore on validation: population (200–1000), generations (50–200), crossover/mutation rates, tournament size, `λ`, max height.

### Outputs of (a)

- Best program per run, as a DEAP tree plus pretty-printed if/else code.
- Hall of fame per fold, with the feature frequency across those programs.
- Convergence plot (best and mean fitness per generation) and the Pareto front for the multi-objective variant.
- Test metrics: mean ± std over 5 folds × 10 seeds, plus a per-fold table.

## (b) Hand-written flow-control program (≤ N statements)

(b) is built **from the GP's output**, distilled by a human. Under CV, (b) must be **one fixed program structure**, evaluated on every fold. It is not hand-written again for each fold. The structure comes from GP runs on the **development segment** (dev train plus a dev-internal validation tail), so it never sees a test fold. Only the numeric thresholds are re-tuned per fold, automatically, on that fold's train.

1. **Collect candidates.** Run the GP 10 times on the dev segment. Take the top programs from the hall of fame, ranked by dev-validation score.
2. **Find recurring structure.** Count how often each feature, comparison direction, and threshold range appears across the candidates. Recurring sub-trees are the building blocks.
3. **Simplify the best program:**
   - remove dead branches (e.g. `if_then_else(c, x, x)` becomes `x`) and constant conditions;
   - merge repeated comparisons on the same feature;
   - remove double negations;
   - rewrite nested `and_`/`or_` as readable `if/elif/else` chains.
4. **Make thresholds interpretable.** Map the scaled thresholds back to raw units and round them to meaningful values. For example, `src_churn ≥ 0.07` (scaled) becomes `src_churn >= 200 lines`. Write the program with named threshold parameters, `predict(build, thr) -> bool`, so that step 8 can re-tune them per fold with a small grid search over raw-unit quantiles on that fold's train.
5. **Add domain knowledge.** Where a GP rule has no plausible CI explanation, consider replacing it with a domain rule. Examples: config files touched, a high recent fail ratio for the committer, a large AST change in source files. Write down the reasoning for each rule in the notebook.
6. **Write the program** as a plain Python function `predict(build) -> bool` with **at most `N` if/elif/case statements**. Count and report its statements.
7. **Validate:** compare (b) with the best dev-segment GP program on the dev-validation tail. If (b) loses, iterate steps 3–6. Freeze the structure once you are done; it is not changed after this step.
8. **Cross-validate:** for each of the 5 folds, tune `thr` on the fold's train (the grid search from step 4), then evaluate once on the fold's test with the same metrics as (a). Report mean ± std across folds, and run the paired Wilcoxon test and A₁₂ against (a).

Caveat to report: (b) is not independent of (a), because it is derived from (a)'s programs. The comparison measures how much performance survives when a human simplifies the program, not human intuition against evolution.

## Suggested layout

```
data/raw/       extracted RavenBuild CSVs (gitignored)
src/
  data.py       load CSVs, label remap, temporal split, T1/T2 targets
  features.py   cleaning + filter/hybrid feature selection
  gp.py         DEAP primitive set, fitness variants, run loop
  rules.py      hand-written program for (b)
  evaluate.py   shared metrics, domain metrics, statistical tests
notebooks/
  01_eda.ipynb
  02_feature_selection.ipynb
  03_gp.ipynb
  04_handwritten.ipynb
  05_results.ipynb
```

## Open questions

- [ ] Label direction in `now_label`. Check this before anything else.
- [ ] Headline fitness: single-objective MCC or NSGA-II (TPR/FPR)?
- [ ] Final `N`: within 15–20, decided by feature selection.
- [ ] Per-project only, or also pooled / cross-project?
- [ ] Is T2 (first failures) large enough per test fold? atlasdb gives roughly 60–100 per fold, while smaller projects may need `n_splits=3`.
- [ ] Compute budget: 5 folds × 10 seeds × 2 targets × projects. Time one GP run on atlasdb first, then scale the population and generations to fit.

## Reading list

- Saidani et al., IST 128 (2020) — GP representation, operators, NSGA-II setup (main inspiration).
- Sun, Habchi, McIntosh, *RavenBuild*, FSE 2024 — dataset, features, Parrot baseline.
- Chen et al., *BuildFast*, ASE 2020 — origin of many of the features.
- IBM, *What is feature selection?* — https://www.ibm.com/think/topics/feature-selection
- DEAP documentation — strongly typed GP (`PrimitiveSetTyped`), bloat control (`staticLimit`), `selNSGA2`.
- Koza, *Genetic Programming*, MIT Press, 1992.
- Harman & Jones, *Search-based software engineering*, IST 43(14), 2001 (lecture bibliography).
- Jin & Servant, *A Cost-efficient Approach to Building in Continuous Integration*, ICSE 2020 — build skipping.
