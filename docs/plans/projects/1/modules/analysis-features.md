# `analysis/features.py`

Status: todo (step 2). Spec: [implementation-backbone.md](../implementation-backbone.md). Method: the main plan's *Feature selection* section.

## Purpose

Choose the feature list and the statement budget `N` once, from the development segment only. Both are then frozen for every CV fold and shared by (a) and (b).

## Proposed API

```python
select_features(ds: Dataset, n_max: int = 20, n_boot: int = 5, seed: int = 0) -> Selection
```

`Selection` holds the feature list, per-statistic scores, bootstrap frequencies, the embedded-method comparison, and `N`. `Selection.save(path)` writes `selected_features_<target>_<project>.json`.

Each stage is its own function so notebooks can plot the intermediate results:

1. `clean(X)`: drop columns with more than 30% missing, impute with the median, drop near-constant columns (`VarianceThreshold` ~1e-3 on min-max scaled data) and binary columns whose minority value is under 1%.
2. `relevance(X, y)`: a DataFrame of features by statistic. Numeric features get ANOVA F and Kendall's τ. Binary features get chi-square and information gain. All get mutual information. The score is the mean rank over the statistics that apply.
3. `prune_redundant(X, ranking, rho=0.8)`: walk the ranking and keep a feature only if its |Spearman ρ| with every kept feature is at most 0.8.
4. `embedded_check(X, y, selected)`: random-forest permutation importance or L1 logistic regression. Returns the disagreements (top-5 features the filters dropped) so the notebook can record a decision.
5. `stability(X, y, ...)`: repeat 1–3 on bootstrap resamples and return how often each feature is chosen.

## Design decisions

- Input is `X[:dev_end]` only. The function takes the dev slice and has no access to later rows.
- Time series: bootstrap resamples contiguous blocks, not single rows, so the dependence between neighbouring builds is kept. This is the one place I deviate from the plan's wording ("bootstrap resamples of train"), and I'll flag it in the code.
- Run separately for T1 and T2. On T1, `last_label` is kept, and the notebook reports results with and without it.
- Scaling and imputation used inside the selection are local to it and are not returned. `training/cv.py` re-fits its own.
- `N` is the number of selected features after the budget cut (15–20).

## Verification

- Synthetic data with known informative, redundant and noise columns: informative ones are kept, one representative per redundant group, noise dropped.
- Determinism with a fixed seed.

## Open points

- Per-project selection first (atlasdb). A pooled subset is a later option.
- Plots for this module live in `viz/plots.py` and are already written.
