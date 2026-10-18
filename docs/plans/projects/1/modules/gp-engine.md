# `gp/engine.py`

Status: todo (step 4). Spec: [implementation-backbone.md](../implementation-backbone.md). Design: the main plan's *(a) Genetic programming*.

## Purpose

Evolve boolean flow-control programs over the selected features, under a hard limit of `N` statements, using DEAP. It plugs into `training/cv.py` as the `fit_predict` callable.

## Proposed API

```python
make_pset(features: list[str]) -> gp.PrimitiveSetTyped
evolve(train, val, pset, cfg: GPConfig, seed: int) -> GPResult
```

`GPConfig` holds population, generations, crossover/mutation rates, tournament size, `lambda`, max height, `N`, and the fitness variant. `GPResult` holds the best program, hall of fame, per-generation stats (feeds `viz.convergence`), and the Pareto front for the multi-objective variant.

## Representation

- Types `float` and `bool`. Terminals: one `float` argument per feature (min-max scaled), an ephemeral constant `U(0, 1)`, and `True`/`False`.
- Primitives: `if_then_else`, `and_`, `or_`, `not_`, `lt`, `ge`. `True` means predict failure.
- Evaluation is vectorised: `np.where` for `if_then_else`, `&`, `|`, `~` for the booleans, and one `gp.compile` per individual.

## Statement budget

A statement is one `if_then_else` or comparison node. `gp.staticLimit` with a custom key counts those nodes and rejects offspring above `N`. Max tree height of about 8 is a second guard.

## Fitness variants

1. Single objective (headline): `MCC(train) - lambda * statements / N`, tournament selection.
2. Multi-objective: maximise TPR and minimise FPR, `selNSGA2`. The final point comes from the Pareto front, chosen by the distance to `(TPR=1, FPR=0)` on the validation split.

Both use `metrics` for the numbers, so there is no second implementation of MCC or TPR.

## Operators

`genHalfAndHalf` depth 1–4, `cxOnePoint`, `mutUniform`, `mutNodeReplacement`, `mutEphemeral`, `HallOfFame` for elitism.

## Design decisions

- No global state: the RNG comes from `seed`, so each of the 50 runs per project and target is reproducible.
- The best program is picked on validation, never on test.
- Time one run on atlasdb before fixing population and generations, as the plan says.

## Verification

- A planted target (`fail = churn > 0.5 and fail_ratio > 0.3`) on synthetic data is recovered with high MCC.
- Every returned program has at most `N` statements.
- Same seed gives the same program.
