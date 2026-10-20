# `gp/engine.py`

Status: done (step 4a). Tests: `projects/1/tests/test_engine.py`. Spec: [implementation-backbone.md](../implementation-backbone.md). Design: the main plan's *(a) Genetic programming*. Training runs in `notebook/gp.ipynb`, not in a script.

## Purpose

Evolve boolean flow-control programs over the frozen T2 feature list, under a hard limit of `N` statements, using DEAP. It plugs into `training/cv.py` as the seeded `fit_predict`.

The objective (`score`) and the vectorised evaluation (`predict`) do not depend on DEAP individuals, so program (b) uses the same definitions for its per-fold threshold tuning. There is one fitness definition for (a) and (b).

## API

```python
make_pset(features: list[str]) -> gp.PrimitiveSetTyped
statements(tree) -> int                                            # primitive nodes; the one counting rule
predict(tree, pset, X: pd.DataFrame) -> np.ndarray                  # bool per row
score(y, pred, size: int, cfg: GPConfig) -> tuple[float, ...]       # SO: (mcc - lambda_ * size / N,), MO: (tpr, fpr, size)
evolve(X_fit, y_fit, X_val, y_val, cfg: GPConfig, seed: int) -> GPResult
fit_predict(split, seed: int, cfg: GPConfig) -> Result             # run_cv adapter: partial(fit_predict, cfg=cfg)
```

```python
@dataclass(frozen=True)
class GPConfig:
    variant: Literal["so", "mo"] = "so"
    pop: int = 300
    ngen: int = 100
    cxpb: float = 0.5
    mutpb: float = 0.5
    tournsize: int = 3
    lam: float = 0.05
    max_height: int = 8
    init_depth: tuple[int, int] = (1, 4)
    hof_size: int = 10
    elite: int = 1                 # SO only
    patience: int | None = None    # early stopping on val; None = off
    n_max: int | None = None       # statement budget N; None = len(features)

@dataclass
class GPResult:
    best: gp.PrimitiveTree         # picked on val
    pset: gp.PrimitiveSetTyped
    hof: list                      # SO: HallOfFame; MO: Pareto front individuals
    stats: pd.DataFrame            # gen, seed, best, mean, std (feeds viz.convergence)
    table: pd.DataFrame            # per hof member: program, statements; SO fitness, val_fitness; MO fit_tpr, fit_fpr, tpr, fpr (val)
    val: dict                      # metrics.scores of `best` on val
    seed: int
```

Defaults are placeholders for the notebook to tune. On a planted target, tournament 5 with cx/mut 0.8/0.2 never left the one-comparison local optimum; tournament 3 with 0.5/0.5 recovered the full rule in 4 of 6 seeds. One run on atlasdb T2 takes 8-20 s (SO faster than MO).

## Representation

- `PrimitiveSetTyped("MAIN", [float] * N, bool)`, arguments renamed to the feature names, so `str(tree)` is readable.
- Terminals: `True`, `False` (bool); one ephemeral `thr ~ U(0, 1)` (float). The ephemeral generator is a module-level function, so several psets in one session reuse it without a DEAP name clash (checked on DEAP 1.4).
- Primitives: `if_then_else` = `np.where`, `and_` = `np.logical_and`, `or_` = `np.logical_or`, `not_` = `np.logical_not`, `lt` = `np.less`, `le` = `np.less_equal`, `gt` = `np.greater`, `ge` = `np.greater_equal`. The `np.logical_*` ufuncs are used instead of `&`, `|`, `~` because `~True == -2` on a Python bool constant.
- `predict` compiles once (`gp.compile`), calls it with the feature columns as arrays, and broadcasts the output to `len(X)` (a constant program returns a scalar).

## Statement budget

- `statements` lives here and `gp/simplify.py` imports it (one counting rule): a statement is any primitive node (`if_then_else`, `and_`, `or_`, `not_`, `lt`, `le`, `gt`, `ge`); terminals are free. With N = 10 this is tight: `if_then_else(ge(f, t), a, b)` already costs 2.
- `mate` and `mutate` are decorated with `gp.staticLimit(key=statements, max_value=N)` and `gp.staticLimit(key=attrgetter("height"), max_value=max_height)`.
- `gp.generate` raises when a float slot must hold a primitive (no primitive returns float), so a local `_generate` (full or grow) places a terminal there instead. Mutation uses it too.
- The initial population is drawn ramped half-and-half and redrawn until every individual is within budget (`staticLimit` does not cover initialisation).

## Fitness variants

1. **SO** (headline): `score` = `mcc(confusion(y_fit, pred)) - lam * statements / N`, weights `(1.0,)`, `selTournament`, plus `elite` hall-of-fame members carried to the next generation.
2. **MO**: `score` = `(tpr, fpr, statements)`, weights `(1.0, -1.0, -1.0)`, mu + lambda with `selNSGA2`, `tools.ParetoFront` as the archive. Size is the third objective because without it NSGA-II kept bigger programs with the same rates (first atlasdb run: 7.8 statements on average vs 3.7 for SO at similar MCC).

Duplicates: both archives treat two programs as one when they predict the same on every fit row (`_same_predictions`, on a packed-bits key set during evaluation). Before this, `le(c, f)` and `not_(gt(c, f))` filled the hall of fame with copies of one rule.

`confusion`, `mcc`, `rates` come from `metrics`, so there is no second MCC or TPR.

## Run loop (`evolve`)

1. `random.seed(seed)`: DEAP draws from the `random` module, so the seed fixes the whole run. Not thread-safe; fine while `run_cv` is sequential.
2. Per generation: `algorithms.varAnd` (`cxOnePoint`; mutation picks one of `mutUniform` with `genFull(0, 2)`, `mutNodeReplacement`, `mutEphemeral("one")` uniformly), evaluate invalid individuals, select (variant above), update the archive, record stats.
3. Stats scalar per generation: SO fitness; MO Youden's J (`tpr - fpr`), so both feed `viz.convergence`.
4. Early stopping (`patience`): the best val score over the archive (SO fitness on val; MO minus the distance to `(1, 0)` on val) has not improved for `patience` generations.
5. Final pick, on val only:
   - SO: hall-of-fame member with the highest val fitness; ties go to fewer statements.
   - MO: front member closest to `(TPR=1, FPR=0)` on val.
6. Logging: one `INFO` line per run (seed, generations run, best fit/val score, statements, seconds), `DEBUG` per generation.

## Classes and global state

`creator.create` is global in DEAP. Four classes (`FitnessSO`, `FitnessMO`, `IndSO`, `IndMO`) are created once at import, guarded with `hasattr(creator, ...)` so notebook re-imports do not warn. No other module state; the toolbox is built inside `evolve`.

## Supporting changes (outside this file)

- `analysis/features.py`: `load_features(target, project, out_dir=OUT_DIR) -> list[str]` reads `features` from the JSON written by `Selection.save`. N = its length.
- `training/cv.py`: `dev_split(ds, features, val_frac=0.2) -> Split`: fit on the head of dev, val on its tail, `X_test = X_val`. Used for GP hyperparameter tuning and for the dev-segment runs that feed program (b), so no test fold is touched.
- `utils/logging/log.py`: see below.

### Logging (`utils/logging/log.py`)

```python
log = Console()                              # kept: download_data uses log.status
get_logger(name: str) -> logging.Logger      # child of the "acise_p1_cigp" logger
set_level(level: int | str) -> None          # e.g. set_level("DEBUG") in the notebook
```

- One `RichHandler(console=log, show_path=False, rich_tracebacks=True, markup=False)` on the `acise_p1_cigp` logger, added once at import; `propagate = False` so Jupyter's root handler does not print twice. Default level `INFO`.
- `markup=False`: messages carrying lists or DEAP strings must not be parsed as rich tags.
- New modules use `logger = get_logger(__name__)`. `viz/plots.save` and `utils/data/download_data` move from `log.print` to `logger.info` / `logger.error` / `logger.exception`; the `log.status` spinner stays.

## Notebook (`notebook/gp.ipynb`)

1. Load `ds = preprocess(load_project("atlasdb"), "T2")`, `features = load_features("T2", "atlasdb")`.
2. Time one SO run on `dev_split`; scale `pop` and `ngen` to the budget (5 folds x 10 seeds x 2 variants).
3. Small grid on `dev_split` (pop, ngen, `lam`, tournsize, max height), scored on dev val.
4. `run_cv(ds, features, partial(fit_predict, cfg=cfg), "gp_so", seeds=range(10))`, same for `gp_mo`. Save the frames under `results/gp/`.
5. Figures: `convergence`, `pareto`, `size_vs_fitness`, `feature_freq_performers` (from `simplify.motifs`), saved with `viz.save`.
6. Dev-segment runs (10 seeds) whose hall of fame feeds program (b).

## Verification (`tests/test_engine.py`)

- Planted target (`fail = churn > 0.5 and fail_ratio > 0.3`) on synthetic data is recovered with MCC > 0.9 (small pop and ngen, so the test stays fast).
- Every individual in the hall of fame / front has at most `N` statements and height at most `max_height`.
- Same seed gives the same `str(best)`; a different seed may not.
- `score` (SO) equals `metrics.mcc` minus the penalty; `predict` on a constant program returns a full-length array.
