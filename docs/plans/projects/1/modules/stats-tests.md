# `stats/tests.py`

Status: todo (step 3c). Spec: [implementation-backbone.md](../implementation-backbone.md).

## Purpose

Compare (a) and (b), and the baselines, on the per-fold scores that `training/cv.py` produces. Kept apart from `metrics` because it works on arrays of scores, not on predictions.

## Proposed API

```python
summary(results: pd.DataFrame) -> pd.DataFrame        # mean ± std per method, metric; per-fold table
paired_wilcoxon(a, b) -> TestResult                    # statistic, p-value, n
vargha_delaney_a12(a, b) -> float
position_in_distribution(value, samples) -> float      # where (b) falls within GP's seeds, per fold
```

## Design decisions

- **Paired by fold.** For GP, each fold's score is the median over its 10 seeds. Program (b) has one score per fold. That gives 5 pairs.
- With 5 folds the Wilcoxon test cannot reach p below 0.0625 (two-sided, exact). The module reports this limit in the result, and reports A₁₂ and the per-fold position of (b) alongside, as the plan requires.
- `scipy.stats.wilcoxon` for the test. A₁₂ is a short pure-numpy rank implementation (0.5 means no difference).
- Ties and all-zero differences are handled explicitly and flagged, not hidden as NaN.

## Verification

- A₁₂ on hand-built arrays: identical gives 0.5, fully separated gives 0 or 1.
- Wilcoxon wrapper matches `scipy` on random pairs and handles the all-ties case.
