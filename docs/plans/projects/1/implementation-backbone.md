# Project 1 — Implementation backbone

Companion to [ga-ci-build-failure-prediction.md](ga-ci-build-failure-prediction.md). The main plan defines the experiment; this spec defines the code modules, their order, and their boundaries.

Package root: `projects/1/src/acise_p1_cigp/`.

## Modules and order

| # | Module | Spec | Status | Responsibility |
|---|---|---|---|---|
| 0 | `viz/plots.py` | [viz-plots](modules/viz-plots.md) | done | Plotly theme, `save()` helper, one helper per figure in the plan's plotting table. |
| 1 | `data/load.py` | [data-load](modules/data-load.md) | todo | Load one project's feature CSV and apply stateless preprocessing (see below). |
| 2 | `analysis/features.py` | [analysis-features](modules/analysis-features.md) | done | Feature selection on the dev segment: clean, relevance ranks, mRMR pruning, block-subsample stability with a `min_freq` threshold (cap N), embedded check, per-fold drift diagnostic. Writes `selected_features_<target>_<project>.json`. |
| 3a | `metrics/metrics.py` | [metrics](modules/metrics.md) | todo | Precision, recall, F1, MCC, accuracy, confusion matrix, TPR/FPR, `% builds saved`, `% failures caught`; baselines (Parrot, majority class, depth-limited decision tree). |
| 3b | `training/cv.py` | [training-cv](modules/training-cv.md) | todo | Expanding-window `TimeSeriesSplit` after the dev segment, dev prepended to each train, inner validation split, per-fold scaling and imputation, seed loop. Takes a `fit_predict` callable, so it does not depend on GP. |
| 3c | `stats/tests.py` | [stats-tests](modules/stats-tests.md) | todo | Wilcoxon signed-rank, Vargha–Delaney A₁₂, mean ± std tables. |
| 4a | `gp/engine.py` | [gp-engine](modules/gp-engine.md) | todo | DEAP typed primitive set, statement limit, single-objective and NSGA-II fitness, run loop, hall of fame. |
| 4b | `gp/simplify.py` | [gp-simplify](modules/gp-simplify.md) | todo | Program simplification, statement counting, pretty-printing, recurring-structure counts. |
| 5 | `solutions/flow_control/model.py` | [solutions-flow-control-model](modules/solutions-flow-control-model.md) | owner: user | Program (b), written by the user. Claude supplies only supporting helpers if asked (simplification, per-fold threshold tuning). |

Steps 3a–3c are independent; write `metrics` first because `training` and `analysis` checks use it.

## Rules

- **Leakage boundary.** `data/load.py` does only stateless work. Anything fitted on data (imputation, scaling, variance filter, feature selection) happens later: on the dev segment for selection, per fold in `training/cv.py` for imputation and scaling.
- **Unit of loading.** `load_project(name, target)` loads one project. Pooled loading is added only if cross-project runs are wanted.
- **Stateless load steps.** Sort by `build_nums`, drop `build_duration` and `build_nums`, remap `now_label` to `failed = 1`, build T1 (all builds) or T2 (`last_label = pass`), and cut the dev segment (first ~20%).
- **Label direction** must be confirmed against `outcome` in `circleci_java_builds.csv` before `load.py` is finished. Needs the data in `projects/1/data/raw/`.
- **One module at a time**, implemented on the user's request.
