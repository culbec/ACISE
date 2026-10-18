# `viz/plots.py`

Status: done. Spec: [implementation-backbone.md](../implementation-backbone.md).

## Purpose

One place for every figure in the plan's plotting table, so notebooks call a single function per chart and all figures share a theme and an export path.

## What it does

- `viz/__init__.py` sets `pio.templates.default = "plotly_white"` and `pio.renderers.default = "notebook+png"` on package import. The PNG renderer is there because GitHub's notebook viewer and PDF exports drop interactive output.
- `save(fig, name, png=True)` writes `results/figures/<name>.html` and `<name>.png` (kaleido).
- One helper per figure, grouped by notebook:
  - 01_eda: `label_balance`, `streak_hist`, `dist_by_label`
  - 02_feature_selection: `corr_heatmap`, `scores_bar`, `stability_bar`
  - 03_gp: `convergence`, `pareto`, `size_vs_fitness`, `feature_freq_performers`
  - 05_results: `metrics_box`, `tradeoff`, `confusion`

## The plots: what each shows and when to use it

Notebook numbers refer to the layout in the main plan. "Input" is what the caller must prepare.

### 01_eda — understand the data before modelling

**`label_balance(df, project, label)`** — stacked bar, one bar per project, split by outcome.
- Use it first, to pick projects. Failure counts decide which projects are usable (testcontainers-java has 16 failures; atlasdb has ~1,000).
- Also the quickest way to see a wrong label direction: a project that is 98% one class, or fresco showing mostly failures, is a prompt to re-check the remap.
- Input: one row per build, with a project column and a label column. Needs several projects in one frame, so concatenate the per-project loads.

**`streak_hist(streaks)`** — histogram of consecutive-failure run lengths.
- Use it to see how failures cluster. Long streaks mean most failures follow a failure, which is why `last_label` dominates T1 (Parrot) and why T2 (first failures only) is the headline target.
- Input: a 1-D series or array of streak lengths, one per streak. Computing the streaks is the caller's job.

**`dist_by_label(df, features, label, kind, log_y)`** — one box or violin per feature, split by label, in a grid of facets (4 per row, independent y-axes).
- Use it to eyeball which features separate pass from fail before any statistic is computed, and to spot skew and outliers.
- `log_y=True` for churn and count features (`src_churn`, `line_added`); they are heavily skewed and a linear axis flattens the boxes. Log axes drop zeros, so features with many zeros need a look at the raw values too.
- `kind="violin"` when you care about the shape (bimodal, many zeros), `"box"` for quartiles and outliers.
- Pass a manageable `features` list (about 8–16). Over about 20 the grid is unreadable.

### 02_feature_selection — justify the cuts

**`corr_heatmap(corr)`** — matrix of correlations, fixed colour range −1 to 1, blue-white-red.
- Use it for the redundancy step: it shows the near-duplicate families (`src_churn` / `line_added` / `src_ast_diff`, the `fail_ratio_*` columns) that mRMR pruning collapses to one representative.
- Input: a square correlation DataFrame. Use Spearman, to match the pruning rule (|ρ| ≤ 0.8). Plot it before pruning to show the problem and after to show it is solved.

**`scores_bar(scores, top)`** — grouped bars, one group per feature, one bar per statistic (F, τ, χ², info gain, MI).
- Use it to show how the relevance statistics agree or disagree on a feature, and to explain the final mean rank.
- Input: features × statistics DataFrame. Scores of different statistics are on different scales, so pass ranks or normalised scores. Raw values make one statistic dwarf the others.
- `top` keeps the N features with the highest mean score, which is what you want when there are 60+ columns.

**`stability_bar(freq)`** — bar per feature, y from 0 to 1, sorted descending.
- Use it after the bootstrap check. Bars near 1 are features selected in nearly every resample. Features near 0.4 are fragile and the plan says to prefer the stable ones.
- Input: a Series indexed by feature, with the selection frequency in [0, 1].

### 03_gp — look inside the evolution

**`convergence(stats)`** — best and mean fitness per generation, each with a ±1 std band over seeds.
- Use it to decide generations and population: a best line that is flat for the last half of the run means you can shorten it, and a mean line far below best means the population is diverse (or stuck on bad individuals).
- A wide band shows the seeds disagree, which is the stochasticity the 10 runs are meant to expose.
- Input: columns `gen`, `seed`, `best`, `mean`. One figure per fold, so filter by fold first.

**`pareto(front, chosen)`** — TPR against FPR, both axes 0 to 1, with an optional star for the picked point.
- Use it only for the NSGA-II variant. Points toward the top-left (high TPR, low FPR) are better. The star marks the point selected on the validation split.
- Input: columns `tpr`, `fpr`; `chosen` is one row with the same columns. Plot train and validation fronts as separate figures, or pass validation values only, to avoid hiding overfitting.

**`size_vs_fitness(df)`** — scatter of statement count against fitness.
- Use it to check the `λ · statements / N` penalty is doing something: you want to see whether bigger programs buy fitness, and where the gain stops. It is the justification for choosing `N`.
- Input: columns `statements`, `fitness`, one row per evaluated program (for example, per hall-of-fame member).

**`feature_freq_performers(freq)`** — bar per feature of how often it appears in hall-of-fame programs.
- Use it before writing program (b). Features that recur across programs are the building blocks of the hand-written structure.
- Input: a Series indexed by feature with counts, from `gp/simplify.motifs`.

### 05_results — compare and report

**`metrics_box(df)`** — box plot per method, one facet per metric, every per-fold point overlaid.
- Use it as the main comparison figure: GP (a), program (b) and the baselines on precision, recall, F1, MCC. Showing every point matters because there are only 5 folds, and a box alone would hide that.
- Input: long format with columns `method`, `metric`, `value`, one row per fold (and per seed, if you plot the GP seeds).
- Metric scales differ (MCC runs from −1), so the facets do not share a y-axis scale unless you set it yourself.

**`tradeoff(df, point)`** — line of `% failures caught` against `% builds saved`, markers at each α, optional star for another method.
- Use it for the cost-aware variant: each α trades missed failures for saved builds. Moving right (more builds saved) usually moves down (fewer failures caught).
- The star puts program (b) on the same plane, so you can see whether it sits on, above or below the GP curve.
- Input: columns `saved`, `caught`, `alpha`; `point` as `(saved, caught)`. Both columns are percentages, 0–100.

**`confusion(cm, labels)`** — 2×2 heatmap with counts written in the cells.
- Use it per method for the report, to make the false-negative cost visible: the bottom-left cell (failed but predicted pass) is the costly one.
- Input: `[[tn, fp], [fn, tp]]` with rows as actual and columns as predicted, in the order of `labels`. This is `sklearn.metrics.confusion_matrix` order when labels are `(pass, fail)`.

## Design decisions

- Helpers take tidy DataFrames or Series and return a `go.Figure`. They do not call `show()` or `save()`, so the caller can still tweak the figure.
- No data loading and no metric computation here. Plots only draw what they are given.
- `convergence` draws one figure per call. For one panel per fold, filter by fold and call it once for each.

## Open points

- Helpers are thin on purpose. Add new ones only when a notebook needs a chart that no existing helper draws.
