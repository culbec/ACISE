# `stats/tests.py`

Status: done (step 3c). Spec: [implementation-backbone.md](../implementation-backbone.md). Tests: `projects/1/tests/test_stats.py`.

## Purpose

Compare (a), (b) and the baselines on the per-fold scores that `training/cv.py` produces. Kept apart from `metrics` because it works on arrays of scores, not on predictions.

## API

```python
summary(results) -> DataFrame                          # method, metric, mean, std over all runs
per_fold(results, metric) -> DataFrame                 # fold x method, median over seeds
paired_wilcoxon(a, b) -> WilcoxonResult                # statistic, p, n, min_p, flag
vargha_delaney_a12(a, b) -> float                      # 0.5 = no difference
position_per_fold(results, metric, point, dist) -> Series   # A12 of `point` vs `dist`'s seeds, per fold
```

`results` is the concatenation of `run_cv` frames (one per method).

## Design decisions

- **Paired by fold.** `per_fold` takes the median over seeds, so GP has one score per fold, like program (b). That gives 5 pairs.
- **`min_p` is reported with every test.** With `n` non-zero pairs the exact two-sided Wilcoxon cannot go below 2^(1−n): 0.0625 for 5 folds. A p above 0.05 is therefore not evidence of no difference; A₁₂ and the per-fold position are reported alongside, as the plan requires.
- **Ties:** zero differences are dropped (scipy's default) and flagged `zero_diffs_dropped`, so `n` can be below 5. All-zero differences make scipy raise; they return `p = 1.0` with flag `all_ties`.
- **A₁₂ from Mann–Whitney U:** `A12 = U / (m·n)` with `scipy.stats.mannwhitneyu`; no hand-written rank code.
- **Position of (b) within GP:** the plan's "where (b) falls within the GP's distribution" is A₁₂ of (b)'s single score against that fold's GP seeds. 1.0 means (b) beats every seed. `position_per_fold` computes it for every fold, so there is no separate formula.
- The class is named `WilcoxonResult`, not `TestResult`: pytest tries to collect any imported class named `Test*`.

## Check with stand-in models (atlasdb, T2, in memory only)

GP and program (b) do not exist yet. Stand-ins, not saved anywhere:

- "GP": `DecisionTreeClassifier(max_depth=3, max_features=3, class_weight="balanced")`, 10 seeds;
- "(b)": `time_interval >= median of fit`.

| | MCC mean ± std | F1 | % builds saved |
|---|---|---|---|
| tree stand-in | 0.123 ± 0.073 | 0.375 | 58.8 |
| rule stand-in | 0.100 ± 0.066 | 0.382 | 53.3 |

Wilcoxon on per-fold MCC: statistic 2, p = 0.1875, n = 5, min_p = 0.0625. A₁₂ (tree vs rule) = 0.6. The rule's per-fold position within the tree seeds: 0.6, 0.2, 0.4, 0.2, 0.2. These numbers only show that the pipeline runs end to end; they are not results.

## Verification

- A₁₂: identical samples give 0.5; fully separated give 1.0 and 0.0.
- Wilcoxon wrapper matches `scipy.stats.wilcoxon` on random pairs; all-ties and dropped-zero cases are flagged, and `min_p` follows `n`.
- Pipeline test on synthetic data: seeded trees and a fixed rule (in-memory stand-ins) through `run_cv`, then every function in this module.
