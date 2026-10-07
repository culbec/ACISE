# Project 1 — GA-evolved rules for CI build-failure prediction

Status: **draft** (2026-10-07). Dataset not chosen yet — pending reading on GAs in ML.

## Assignment recap (Lecture 1, p.4)

- **(a)** Install an open-source ML framework; implement one technique from the list (chosen: **genetic algorithms**) on an existing dataset.
- **(b)** Solve the same problem with at most `nnn` if/case statements (`nnn` = number of attributes), matching or exceeding (a).
- Evaluate both with multiple metrics (at least precision, recall, accuracy).
- Deadline: week 4. Deliverable: Python scripts + Jupyter notebooks.

## Core idea

A GA is a search method, not a classifier. It beats standard ML when:

1. The objective is non-differentiable (F1, MCC, misclassification cost, rule-count penalty).
2. The solution space mixes discrete and continuous choices (which feature, which operator, which threshold).
3. The output must be interpretable.

So: **evolve a classifier that is itself an at-most-`n`-if program.** Part (a) and part (b) then produce the same kind of artefact:

- (a) the if-program is found by evolution;
- (b) the if-program is written by a human from domain knowledge.

Research question: *can an engineer's intuition match evolution under the same `n`-if budget?*

## Problem framing

- **Unit:** one CI build (push or PR trigger).
- **Label:** `failed = 1`, `passed = 0`. Drop `cancelled` / infra-`errored` builds.
- **Use case:** predict "will pass" to skip or deprioritise builds. Each correct skip saves CI time; each missed failure is costly.
- **Target choice:** predict **first failures** (a failure following a pass), not every failure.
  - Reason: failures come in streaks, so a single `if prev_failed` scores well on the all-failures target and leaves the GA nothing to show.
  - First failures are harder, more useful in practice, and where non-obvious feature combinations matter.
  - Prior work: Jin & Servant on CI build skipping (look up exact title).

## Features (pre-build only)

Target ~15–20 features, so `nnn` ≈ 15–20 ifs. Final list depends on what the chosen dataset exposes.

| Group | Features |
|---|---|
| Change size | `files_changed`, `lines_added`, `lines_deleted`, `commits_in_push` |
| Change type | `src_files`, `test_files`, `build_files_touched` (CI yaml, Dockerfile, pom/gradle, package.json, lockfile), `deps_changed` (bool), `dirs_touched` |
| Context | `is_pr`, `is_merge`, `branch_is_main`, `hours_since_last_build`, `builds_since_last_failure` |
| People | `author_prior_commits`, `author_recent_fail_rate` |
| Commit message | `msg_has_fix`, `msg_has_wip` (bools) |

**Leakage rule:** never use build duration, tests run, log content, or anything else only known after the build runs.

## GA design (part a)

Library: **DEAP** (flexible custom genomes). PyGAD is a simpler fallback.

### Genome

One gene per feature:

- numeric feature: `[active ∈ {0,1}, op ∈ {<, ≥}, threshold ∈ [min_i, max_i]]`
- boolean feature: `[active ∈ {0,1}, expected_value ∈ {0,1}]`

Plus one global gene for combination: `AND` | `OR` | `k-of-n` vote.

Decoding yields at most `n` if-statements — the same budget as part (b).

### Fitness (pick one, or compare)

- Cost-weighted: `saved_builds − α·missed_failures − λ·active_rules`
- Constrained: maximise `saved_builds` subject to `recall_fail ≥ 0.9` (penalty on violation)
- Simple alternative: `MCC(train) − λ·active_rules`

### Operators

- Tournament selection
- Uniform crossover
- Bit-flip mutation on `active` / `op` / bools; Gaussian mutation on thresholds
- Elitism

### Hyperparameters to explore

Population size, generations, crossover/mutation rates, tournament size, `λ`, `α`.

## Hand-written baseline (part b)

- At most `n` if/case statements, written from CI domain knowledge (e.g. build files touched, dependency changes, large diffs, inexperienced author).
- Thresholds tuned on the **training split only**.
- Keep the reasoning for each rule in the notebook — it is part of the comparison.

## Evaluation

- **Split by time:** train on older builds, test on newer ones. A random split leaks future project state.
- **Same split** for GA and baseline.
- **Metrics** (fail class): precision, recall, F1, MCC, accuracy, confusion matrix. Note why accuracy misleads under class imbalance.
- **Domain metrics:** `% builds saved`, `% failures caught`. Baseline = one point; GA = a trade-off curve from sweeping `α`.
- **Stochasticity:** 30 independent GA runs, report mean ± std; Wilcoxon test (or check where the baseline falls in the GA's spread).
- **Optional:** convergence plot (best fitness per generation); sklearn decision tree as a reference point.

## Data

Decision: **public dataset** (to be chosen).

- TravisTorrent (Beller et al., MSR 2017) — large, well known, but Travis-era (old).
- Newer GitHub Actions datasets from MSR mining challenges — verify names and contents.
- Selection criteria: build status per build, commit/diff metadata to derive the features above, enough history per project for a temporal split, a fail rate that is imbalanced but not extreme.

## Suggested layout

```
src/        data loading, feature extraction, GA, baseline rules, shared evaluation
notebooks/  EDA, GA experiments, baseline, results comparison
```

## Open questions

- [ ] Which public dataset? (after reading on GAs in ML)
- [ ] All failures vs first failures — confirm once fail-streak statistics are visible in EDA.
- [ ] Which fitness variant becomes the headline result?
- [ ] Single project vs multiple projects (cross-project generalisation is a possible extra).

## Reading list

- Harman & Jones, *Search-based software engineering*, IST 43(14), 2001 (lecture bibliography).
- Pittsburgh vs Michigan approaches to GA-based rule learning (genetics-based machine learning).
- DEAP documentation — custom individuals and mixed-type mutation.
- Jin & Servant — CI build skipping / build-outcome prediction.
- Beller et al., *TravisTorrent*, MSR 2017.
