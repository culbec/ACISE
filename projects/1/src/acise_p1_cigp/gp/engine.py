import math
import random
import time
from dataclasses import dataclass, replace
from functools import partial
from operator import attrgetter
from typing import Literal

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, gp, tools

from acise_p1_cigp.metrics.metrics import confusion, mcc, rates, scores
from acise_p1_cigp.training.cv import Result
from acise_p1_cigp.utils.logging.log import get_logger

logger = get_logger(__name__)

PRIMITIVES = (
    (np.where, [bool, bool, bool], "if_then_else"),
    (np.logical_and, [bool, bool], "and_"),
    (np.logical_or, [bool, bool], "or_"),
    (np.logical_not, [bool], "not_"),
    (np.less, [float, float], "lt"),
    (np.less_equal, [float, float], "le"),
    (np.greater, [float, float], "gt"),
    (np.greater_equal, [float, float], "ge"),
)

if not hasattr(creator, "FitnessSO"):
    creator.create("FitnessSO", base.Fitness, weights=(1.0,))
    creator.create("FitnessMO", base.Fitness, weights=(1.0, -1.0))
    creator.create("IndSO", gp.PrimitiveTree, fitness=creator.FitnessSO)  # pyright: ignore[reportAttributeAccessIssue]
    creator.create("IndMO", gp.PrimitiveTree, fitness=creator.FitnessMO)  # pyright: ignore[reportAttributeAccessIssue]


@dataclass(frozen=True)
class GPConfig:
    variant: Literal["so", "mo"] = "so"
    pop: int = 300
    n_gen: int = 100
    cx_prob: float = 0.5
    mut_prob: float = 0.5
    tournament_n: int = 3
    lambda_: float = 0.05
    max_height: int = 8
    init_depth: tuple[int, int] = (1, 4)
    hof_size: int = 10 # hall of fame size; MO: Pareto front size
    elite: int = 1  # SO only
    patience: int | None = None  # generations without val improvement; None = off
    n_max: int | None = None  # statement budget N; None = number of features


@dataclass
class GPResult:
    best: gp.PrimitiveTree  # picked on val
    pset: gp.PrimitiveSetTyped
    hof: list  # SO: hall of fame; MO: Pareto front
    table: pd.DataFrame  # one row per hof member, see `_table`
    stats: pd.DataFrame  # gen, seed, best, mean, std
    val: dict  # metrics.scores of `best` on val
    seed: int


def _thr() -> float:
    """
    Ephemeral threshold generator; module level, so psets in one session can share it.

    Returns:
        float: U(0, 1) draw from the `random` module.
    """
    return random.uniform(0, 1)


def make_pset(features: list[str]) -> gp.PrimitiveSetTyped:
    """
    Strongly typed primitive set: one float argument per feature, named after it,
    the flow-control primitives, `True`/`False` and a U(0, 1) threshold.

    Args:
        features (list[str]): feature names, in column order.

    Returns:
        gp.PrimitiveSetTyped: primitive set returning bool (True = predict failure).
    """
    pset = gp.PrimitiveSetTyped("MAIN", [float] * len(features), bool)
    pset.renameArguments(**{f"ARG{i}": f for i, f in enumerate(features)})
    for fn, args, name in PRIMITIVES:
        pset.addPrimitive(fn, args, bool, name=name)
    pset.addTerminal(True, bool)
    pset.addTerminal(False, bool)
    pset.addEphemeralConstant("thr", _thr, float)
    return pset


def statements(tree) -> int:
    """
    Statement count: every primitive node; terminals are free. The one counting
    rule for the budget, the penalty and the report.

    Args:
        tree: DEAP tree (or any sequence of nodes).

    Returns:
        int: number of primitive nodes.
    """
    return sum(isinstance(node, gp.Primitive) for node in tree)


def _generate(pset, min_: int, max_: int, full: bool, type_=None) -> list:
    """
    `gp.genFull` / `gp.genGrow` that places a terminal wherever the slot type has no
    primitive (float here), instead of raising like `gp.generate`.

    Args:
        pset (gp.PrimitiveSetTyped): primitive set.
        min_ (int): minimum height.
        max_ (int): maximum height.
        full (bool): True grows every branch to the drawn height (genFull).
        type_: return type of the tree; None = `pset.ret`.

    Returns:
        list: nodes in prefix order.
    """
    height = random.randint(min_, max_)
    expr, stack = [], [(0, type_ or pset.ret)]
    while stack:
        depth, t = stack.pop()
        leaf = (
            depth == height
            or not pset.primitives[t]
            or (not full and depth >= min_ and random.random() < pset.terminalRatio)
        )
        if leaf:
            term = random.choice(pset.terminals[t])
            expr.append(term() if isinstance(term, gp.MetaEphemeral) else term)
        else:
            prim = random.choice(pset.primitives[t])
            expr.append(prim)
            stack.extend((depth + 1, a) for a in reversed(prim.args))
    return expr


def _mutate(ind, pset) -> tuple:
    """
    One of sub-tree, node-replacement or threshold mutation, picked uniformly.

    Args:
        ind: individual, changed in place.
        pset (gp.PrimitiveSetTyped): primitive set.

    Returns:
        tuple: the mutated individual, as DEAP mutations return.
    """
    op = random.randrange(3)
    if op == 0:
        return gp.mutUniform(ind, partial(_generate, min_=0, max_=2, full=True), pset)
    if op == 1:
        return gp.mutNodeReplacement(ind, pset)
    return gp.mutEphemeral(ind, "one")


def predict(tree, pset: gp.PrimitiveSetTyped, X: pd.DataFrame) -> np.ndarray:
    """
    Vectorised evaluation of a program on every row of `X`.

    Args:
        tree: DEAP tree over `pset`.
        pset (gp.PrimitiveSetTyped): primitive set the tree was built with.
        X (pd.DataFrame): scaled features; columns are looked up by name.

    Returns:
        np.ndarray: bool per row, True = will fail.
    """
    out = gp.compile(tree, pset)(*X[pset.arguments].to_numpy(float).T)
    return np.broadcast_to(out, (len(X),)).astype(bool)


def score(y, pred, size: int, cfg: GPConfig) -> tuple[float, ...]:
    """
    The objective, shared by the GP and the threshold tuning of program (b).

    Args:
        y: labels, 1/True = failed.
        pred: predictions, True = will fail.
        size (int): statement count of the program.
        cfg (GPConfig): variant, `lam` and `n_max` (must be set).

    Returns:
        tuple[float, ...]: SO `(mcc - lam * size / n_max,)`; MO `(tpr, fpr)`.
    """
    c = confusion(y, pred)
    if cfg.variant == "mo":
        return rates(c)
    if not cfg.n_max:
        raise ValueError("cfg.n_max must be set to score the SO variant")
    return (mcc(c) - cfg.lambda_ * size / cfg.n_max,)


def _val_key(ind, pset, X, y, cfg: GPConfig) -> float:
    """
    Validation score, higher is better: SO fitness, or MO minus the distance of
    (tpr, fpr) to (1, 0).

    Args:
        ind: individual.
        pset (gp.PrimitiveSetTyped): primitive set.
        X (pd.DataFrame): validation features.
        y: validation labels.
        cfg (GPConfig): configuration.

    Returns:
        float: score used for early stopping and the final pick.
    """
    s = score(y, predict(ind, pset, X), statements(ind), cfg)
    return -math.hypot(1 - s[0], s[1]) if cfg.variant == "mo" else s[0]


def _init(cls, pset, cfg: GPConfig) -> list:
    """
    Ramped half-and-half population, redrawn until every program is within the
    statement and height budget (`staticLimit` does not cover initialisation).

    Args:
        cls: individual class.
        pset (gp.PrimitiveSetTyped): primitive set.
        cfg (GPConfig): configuration with `n_max` set.

    Returns:
        list: `cfg.pop` individuals.
    """
    assert cfg.n_max, "cfg.n_max must be set to initialise the population"
    
    pop = []
    while len(pop) < cfg.pop:
        ind = cls(_generate(pset, *cfg.init_depth, full=random.random() < 0.5))
        if statements(ind) <= cfg.n_max and ind.height <= cfg.max_height:
            pop.append(ind)
    return pop


def _table(hof, pset, X_val, y_val, cfg: GPConfig) -> pd.DataFrame:
    """
    One row per archive member, for `viz.size_vs_fitness` (SO) and `viz.pareto` (MO).

    Args:
        hof: hall of fame or Pareto front.
        pset (gp.PrimitiveSetTyped): primitive set.
        X_val (pd.DataFrame): validation features.
        y_val: validation labels.
        cfg (GPConfig): configuration.

    Returns:
        pd.DataFrame: program, statements, then SO `fitness`, `val_fitness` or
            MO `fit_tpr`, `fit_fpr`, `tpr`, `fpr` (the last two on val).
    """
    rows = []
    for ind in hof:
        size = statements(ind)
        val = score(y_val, predict(ind, pset, X_val), size, cfg)
        row = {"program": str(ind), "statements": size}
        if cfg.variant == "mo":
            row |= dict(zip(("fit_tpr", "fit_fpr", "tpr", "fpr"), (*ind.fitness.values, *val)))
        else:
            row |= {"fitness": ind.fitness.values[0], "val_fitness": val[0]}
        rows.append(row)
    return pd.DataFrame(rows)


def evolve(
    X_fit: pd.DataFrame,
    y_fit,
    X_val: pd.DataFrame,
    y_val,
    cfg: GPConfig,
    seed: int,
) -> GPResult:
    """
    One GP run: fitness on the fit rows, early stopping and the final pick on val.

    Args:
        X_fit (pd.DataFrame): scaled fit features; columns define the pset arguments.
        y_fit: fit labels, 1 = failed.
        X_val (pd.DataFrame): scaled validation features.
        y_val: validation labels.
        cfg (GPConfig): configuration.
        seed (int): seeds the `random` module, which drives all of DEAP.

    Returns:
        GPResult: best program, archive, per-generation stats and val scores.
    """
    t0 = time.perf_counter()
    random.seed(seed)
    pset = make_pset([str(c) for c in X_fit.columns])
    cfg = replace(cfg, n_max=cfg.n_max or len(pset.arguments))
    mo = cfg.variant == "mo"
    y_fit, y_val = np.asarray(y_fit, dtype=bool), np.asarray(y_val, dtype=bool)

    tb = base.Toolbox()
    tb.register("mate", gp.cxOnePoint)
    tb.register("mutate", _mutate, pset=pset)
    for op in ("mate", "mutate"):
        tb.decorate(op, gp.staticLimit(statements, cfg.n_max))
        tb.decorate(op, gp.staticLimit(attrgetter("height"), cfg.max_height))

    def evaluate(inds):
        for ind in inds:
            if not ind.fitness.valid:
                ind.fitness.values = score(y_fit, predict(ind, pset, X_fit), statements(ind), cfg)

    def val_key(ind):
        return _val_key(ind, pset, X_val, y_val, cfg)

    def record(gen):
        v = np.array([i.fitness.values[0] - (i.fitness.values[1] if mo else 0) for i in pop])
        rows.append({"gen": gen, "seed": seed, "best": v.max(), "mean": v.mean(), "std": v.std()})
        logger.debug(f"gen {gen}: best {v.max():.4f} mean {v.mean():.4f}")

    pop = _init(creator.IndMO if mo else creator.IndSO, pset, cfg)  # pyright: ignore[reportAttributeAccessIssue]
    evaluate(pop)
    hof = tools.ParetoFront() if mo else tools.HallOfFame(cfg.hof_size)
    hof.update(pop)
    rows: list[dict] = []
    record(0)

    best_val, stale, gen = -math.inf, 0, 0
    for gen in range(1, cfg.n_gen + 1):
        off = algorithms.varAnd(pop, tb, cfg.cx_prob, cfg.mut_prob)
        evaluate(off)
        if mo:
            pop = tools.selNSGA2(pop + off, cfg.pop)
        else:
            elite = [tb.clone(i) for i in hof[: cfg.elite]]
            pop = tools.selTournament(off, cfg.pop - len(elite), cfg.tournament_n) + elite
        hof.update(pop)
        record(gen)
        if cfg.patience:
            v = max(val_key(i) for i in hof)
            best_val, stale = (v, 0) if v > best_val else (best_val, stale + 1)
            if stale >= cfg.patience:
                break

    best = max(hof, key=lambda i: (val_key(i), -statements(i)))
    val = scores(y_val, predict(best, pset, X_val))
    logger.info(
        f"{cfg.variant} seed {seed}: {gen} gens, fit {best.fitness.values}, "
        f"val mcc {val['mcc']:.3f}, {statements(best)} statements, "
        f"{time.perf_counter() - t0:.1f}s"
    )
    return GPResult(
        best=best,
        pset=pset,
        hof=list(hof),
        table=_table(hof, pset, X_val, y_val, cfg),
        stats=pd.DataFrame(rows),
        val=val,
        seed=seed,
    )


def fit_predict(split, seed: int, cfg: GPConfig) -> Result:
    """
    `run_cv` adapter: `run_cv(..., partial(fit_predict, cfg=cfg), seeds=range(10))`.

    Args:
        split: `training.cv.Split` of one fold.
        seed (int): run seed.
        cfg (GPConfig): configuration.

    Returns:
        Result: test predictions of the val-picked program, `GPResult` as payload.
    """
    res = evolve(split.X_fit, split.y_fit, split.X_val, split.y_val, cfg, seed)
    return Result(predict(res.best, res.pset, split.X_test), res)
