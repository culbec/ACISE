import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from acise_p2_cidt.metrics.metrics import mcc as matthews_corrcoef
from acise_p2_cidt.training.cv import Result, Split
from acise_p2_cidt.tree.cart import CART
from acise_p2_cidt.utils.logging.log import get_logger

# 5 x 3 x 2 x 2 = 60 fits per fold. Depths stop at 8: pruning cuts deeper trees back.
GRID: dict[str, list] = {
    "max_depth": [3, 4, 5, 6, 8],
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
        Result: test predictions; payload {"choice", "tree", "grid"} (grid None
            when `params` is given).
    """
    grid = None
    if params is None:
        best, grid = grid_search(split)
        params = best.params
    choice, tree = _tune_point(split, params)
    pred = tree.predict(split.X_test, threshold=choice.threshold)
    return Result(pred, {"choice": choice, "tree": tree, "grid": grid})
