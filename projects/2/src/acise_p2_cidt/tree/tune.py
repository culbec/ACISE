import itertools
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from acise_p2_cidt.metrics.metrics import mcc as matthews_corrcoef
from acise_p2_cidt.training.cv import MIN_TEST_FAILURES, Result, Split
from acise_p2_cidt.tree.cart import CART
from acise_p2_cidt.utils.logging.log import get_logger

# 7 x 3 x 2 x 2 = 84 fits per fold; pruning cuts deep trees back.
GRID: dict[str, list] = {
    "max_depth": [3, 4, 5, 6, 8, 10, 12],
    "min_samples_leaf": [5, 10, 20],
    "criterion": ["gini", "entropy"],
    "class_weight": [None, "balanced"],
}

logger = get_logger(__name__)


@dataclass(frozen=True)
class Choice:
    params: dict  # grid params passed to CART
    alpha: float  # pruning alpha
    threshold: float  # predict "fail" when P(fail) > threshold
    val_mcc: float
    # per source; a source missing here uses `threshold`
    thresholds: dict[str, float] | None = None


def choose_alpha(tree: CART, split: Split) -> float:
    """
    Pruning alpha with the best validation MCC at threshold 0.5.

    Args:
        tree (CART): tree fitted on the fit rows, unpruned.
        split (Split): fold; only `X_val` and `y_val` are read.

    Returns:
        float: chosen alpha; ties go to the larger alpha (smaller tree).
    """
    best_a, best = 0.0, -np.inf
    for a in np.unique(tree.cost_complexity_path()[0]):  # increasing
        mcc = matthews_corrcoef(split.y_val, tree.pruned(a).predict(split.X_val))
        if mcc >= best:
            best_a, best = float(a), mcc
    return best_a


def choose_threshold(proba: np.ndarray, y: np.ndarray) -> float:
    """
    Decision threshold on P(fail) with the best MCC.

    A tree has few distinct outputs, so the candidates are the distinct
    probabilities plus 0.5; with the strict `>` of `CART.predict`, these cover
    every distinct prediction except "all fail".

    Args:
        proba (np.ndarray): P(fail) per validation row.
        y (np.ndarray): validation labels.

    Returns:
        float: chosen threshold; ties go to the one nearer 0.5.
    """
    cands = np.union1d(proba, [0.5])
    cands = cands[np.argsort(np.abs(cands - 0.5), kind="stable")]
    mccs = [matthews_corrcoef(y, proba > t) for t in cands]
    return float(cands[int(np.argmax(mccs))])  # first max = nearest 0.5


def choose_thresholds(
    proba: np.ndarray, y: np.ndarray, src: np.ndarray, pooled: float
) -> dict[str, float]:
    """
    One threshold per source, each tuned on that source's validation rows. A source
    with fewer than `MIN_TEST_FAILURES` validation failures keeps `pooled`: a
    threshold tuned on a handful of failures is noise.
    """
    out = {}
    for s in np.unique(src):
        m = src == s
        out[str(s)] = (
            choose_threshold(proba[m], y[m]) if y[m].sum() >= MIN_TEST_FAILURES else pooled
        )
    return out


def apply_thresholds(proba: np.ndarray, src: np.ndarray | None, c: Choice) -> np.ndarray:
    """True where P(fail) is above the row's source threshold (pooled if none)."""
    if src is None or not c.thresholds:
        return proba > c.threshold
    return proba > np.array([c.thresholds.get(str(s), c.threshold) for s in src])


def predict_per_source(
    proba_val: np.ndarray, proba_test: np.ndarray, split: Split, c: Choice
) -> tuple[Choice, np.ndarray, dict[str, np.ndarray] | None]:
    """
    Test predictions with per-source thresholds, when the validation rows hold more
    than one source. The pooled-threshold predictions come back as the variant
    `pooled_thr`, for comparison.
    """
    if split.src_val is None or len(np.unique(split.src_val)) < 2:
        return c, proba_test > c.threshold, None
    c = replace(
        c,
        thresholds=choose_thresholds(
            proba_val, split.y_val.to_numpy(), split.src_val, c.threshold
        ),
    )
    pred = apply_thresholds(proba_test, split.src_test, c)
    return c, pred, {"pooled_thr": proba_test > c.threshold}


def _tune_point(split: Split, params: dict) -> tuple[Choice, CART]:
    """Fit on the fit rows, prune on validation, tune the threshold."""
    tree = CART(**params).fit(split.X_fit, split.y_fit)
    alpha = choose_alpha(tree, split)
    tree = tree.pruned(alpha)
    proba = tree.predict_proba(split.X_val)[:, 1]
    threshold = choose_threshold(proba, split.y_val.to_numpy())
    mcc = float(matthews_corrcoef(split.y_val, proba > threshold))
    return Choice(params, alpha, threshold, mcc), tree


def grid_search(split: Split, grid: dict | None = None) -> tuple[Choice, pd.DataFrame]:
    """
    Every grid point, scored on validation MCC after pruning and threshold tuning.

    Args:
        split (Split): fold; `X_test` is not read.
        grid (dict | None): CART parameter name -> values; None = `GRID`, read at
            call time.

    Returns:
        tuple[Choice, pd.DataFrame]: best choice (first in grid order on ties) and
            one row per grid point.
    """
    grid = GRID if grid is None else grid
    rows, best = [], None
    for values in itertools.product(*grid.values()):
        choice, tree = _tune_point(split, dict(zip(grid, values)))
        rows.append(
            choice.params
            | {
                "alpha": choice.alpha,
                "threshold": choice.threshold,
                "val_mcc": choice.val_mcc,
                "depth": tree.depth_,
                "n_leaves": tree.n_leaves_,
            }
        )
        if best is None or choice.val_mcc > best.val_mcc:
            best = choice
    if best is None:
        raise ValueError(f"grid has no points, got {grid!r}")
    logger.debug(f"best {best.params}, alpha {best.alpha:.2e}, mcc {best.val_mcc:.3f}")
    return best, pd.DataFrame(rows)


def fit_predict(split: Split, params: dict | None = None) -> Result:
    """
    `run_cv` callable: tune on validation, then predict the test rows.

    The tree is not refitted on fit + val: alpha and threshold were chosen for this
    tree, and a refit would change it.

    Args:
        split (Split): prepared fold.
        params (dict | None): fixed CART parameters; None runs `grid_search`.

    Returns:
        Result: test predictions (per-source thresholds on a pool); payload
            {"choice", "tree", "grid"} (grid None when `params` is given); variant
            `pooled_thr` on a pool.
    """
    grid = None
    if params is None:
        best, grid = grid_search(split)
        params = best.params
    choice, tree = _tune_point(split, params)
    choice, pred, variants = predict_per_source(
        tree.predict_proba(split.X_val)[:, 1],
        tree.predict_proba(split.X_test)[:, 1],
        split,
        choice,
    )
    return Result(pred, {"choice": choice, "tree": tree, "grid": grid}, variants)


def source_rate(split: Split) -> np.ndarray:
    """
    Baseline that knows only the source: P(fail) is the source's failure rate on the
    fit rows (the overall rate for an unseen source), with one threshold tuned on
    validation. A model close to it mostly tells the sources apart.
    """
    if split.src_fit is None:
        raise ValueError("source_rate needs a dataset with sources")
    rate = split.y_fit.groupby(split.src_fit).mean()

    def proba(src: np.ndarray) -> np.ndarray:
        return pd.Series(src).map(rate).fillna(split.y_fit.mean()).to_numpy()

    threshold = choose_threshold(proba(split.src_val), split.y_val.to_numpy())
    return proba(split.src_test) > threshold
