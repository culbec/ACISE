import itertools
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from acise_p2_cidt.training.cv import Result, Split
from acise_p2_cidt.tree.cart import CART, Criterion
from acise_p2_cidt.tree.tune import Choice, choose_threshold
from acise_p2_cidt.utils.logging.log import get_logger

# 2 x 2 x 2 = 8 forests per fold. Depth and leaf size are the only tree controls
# that matter once the trees are averaged; trees are not pruned (Breiman, 2001).
GRID: dict[str, list] = {
    "max_depth": [8, None],
    "min_samples_leaf": [5, 20],
    "class_weight": [None, "balanced"],
}

logger = get_logger(__name__)


class RandomForest:
    """
    Random forest over our own CART (Breiman, 2001).

    Each tree is fitted on a bootstrap sample and draws `max_features` columns at
    every node. The bootstrap is passed as sample weights (the count of each drawn
    row), so every tree sees each distinct row once, as sklearn does. P(fail) is the
    mean of the trees' leaf shares (soft voting). Holds only CARTs, NumPy arrays and
    scalars, so skops can save it like the tree.

    Args:
        n_estimators (int): number of trees.
        max_features (Literal["sqrt"] | float | int | None): columns drawn per node:
            "sqrt" of the column count, a share in (0, 1], a count, or None for all
            (plain bagging).
        bootstrap (bool): False fits every tree on all rows.
        random_state (int): seed of the bootstrap and of the column draws.
        criterion, max_depth, min_samples_split, min_samples_leaf: as in `CART`.
        class_weight (Literal["balanced"] | None): "balanced" weights class k by
            n / (2 * n_k) on the full training set, before the bootstrap.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_features: Literal["sqrt"] | float | None = "sqrt",
        bootstrap: bool = True,
        random_state: int = 0,
        criterion: Criterion = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        class_weight: Literal["balanced"] | None = None,
    ):
        if n_estimators < 1:
            raise ValueError(f"invalid n_estimators, got {n_estimators!r}")
        if class_weight not in (None, "balanced"):
            raise ValueError(f"invalid class_weight, got {class_weight!r}")
        if not (
            max_features is None
            or max_features == "sqrt"
            or (isinstance(max_features, float) and 0 < max_features <= 1)
            or (isinstance(max_features, int) and max_features >= 1)
        ):
            raise ValueError(f"invalid max_features, got {max_features!r}")
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight

    def _n_drawn(self, p: int) -> int:
        """Columns drawn per node for `p` columns."""
        m = self.max_features
        if m is None:
            return p
        if m == "sqrt":
            return max(1, int(np.sqrt(p)))
        if isinstance(m, float):
            return max(1, int(m * p))
        return min(int(m), p)

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> RandomForest:
        """
        Fits `n_estimators` trees.

        Args:
            X (pd.DataFrame | np.ndarray): (n, p) features, no NaN.
            y (pd.Series | np.ndarray): 0/1 or bool label, 1 = failed.

        Returns:
            RandomForest: self.
        """
        # A DataFrame keeps the column names, so each tree can still `explain`.
        Xf = pd.DataFrame(X).reset_index(drop=True)
        yv = np.asarray(y)
        if yv.shape != (len(Xf),) or not np.isin(yv, (0, 1)).all():
            raise ValueError(f"y must be 0/1 of shape ({len(Xf)},), got {yv.shape}")
        yv = yv.astype(int)
        n = len(yv)
        sw = np.ones(n)
        if self.class_weight == "balanced":
            # maximum(..., 1): an absent class has no rows to weight.
            sw = (n / (2.0 * np.maximum(np.bincount(yv, minlength=2), 1)))[yv]

        rng = np.random.default_rng(self.random_state)
        k = self._n_drawn(Xf.shape[1])
        self.estimators_: list[CART] = []
        for _ in range(self.n_estimators):
            counts = (
                np.bincount(rng.integers(0, n, n), minlength=n)
                if self.bootstrap
                else np.ones(n, dtype=int)
            )
            rows = np.flatnonzero(counts)
            tree = CART(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=k,
                random_state=int(rng.integers(2**32)),
            )
            tree.fit(Xf.iloc[rows], yv[rows], sample_weight=sw[rows] * counts[rows])
            self.estimators_.append(tree)

        self.n_features_ = Xf.shape[1]
        self.feature_names_ = self.estimators_[0].feature_names_
        self.feature_importances_ = np.mean(
            [t.feature_importances_ for t in self.estimators_], axis=0
        )
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """(n, 2) mean of the trees' P(pass), P(fail)."""
        Xf = pd.DataFrame(X)
        return np.mean([t.predict_proba(Xf) for t in self.estimators_], axis=0)

    def predict(
        self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        """True where P(fail) > threshold, as in `CART.predict`."""
        return self.predict_proba(X)[:, 1] > threshold


# =============================================================================
# Tuning on validation
# =============================================================================


def _tune_point(split: Split, params: dict) -> tuple[Choice, RandomForest]:
    """Fit on the fit rows, tune the threshold on validation."""
    forest = RandomForest(**params).fit(split.X_fit, split.y_fit)
    proba = forest.predict_proba(split.X_val)[:, 1]
    threshold = choose_threshold(proba, split.y_val.to_numpy())
    mcc = float(matthews_corrcoef(split.y_val, proba > threshold))
    # alpha 0: forest trees are not pruned.
    return Choice(params, 0.0, threshold, mcc), forest


def grid_search(split: Split, grid: dict = GRID) -> tuple[Choice, pd.DataFrame]:
    """
    Every grid point, scored on validation MCC after threshold tuning.

    Args:
        split (Split): fold; `X_test` is not read.
        grid (dict): RandomForest parameter name -> values.

    Returns:
        tuple[Choice, pd.DataFrame]: best choice (first in grid order on ties) and
            one row per grid point.
    """
    rows, best = [], None
    for values in itertools.product(*grid.values()):
        choice, forest = _tune_point(split, dict(zip(grid, values)))
        depths = [t.depth_ for t in forest.estimators_]
        rows.append(
            choice.params
            | {
                "threshold": choice.threshold,
                "val_mcc": choice.val_mcc,
                "mean_depth": float(np.mean(depths)),
                "mean_leaves": float(
                    np.mean([t.n_leaves_ for t in forest.estimators_])
                ),
            }
        )
        if best is None or choice.val_mcc > best.val_mcc:
            best = choice
    if best is None:
        raise ValueError(f"grid has no points, got {grid!r}")
    logger.debug(f"best {best.params}, mcc {best.val_mcc:.3f}")
    return best, pd.DataFrame(rows)


def fit_predict(split: Split, params: dict | None = None) -> Result:
    """
    `run_cv` callable: tune on validation, then predict the test rows.

    As in `tune.fit_predict`, the forest is not refitted on fit + val, so the
    threshold stays matched to the forest it was chosen for.

    Args:
        split (Split): prepared fold.
        params (dict | None): fixed RandomForest parameters; None runs `grid_search`.

    Returns:
        Result: test predictions; payload {"choice", "forest", "grid"} (grid None
            when `params` is given).
    """
    grid = None
    if params is None:
        best, grid = grid_search(split)
        params = best.params
    choice, forest = _tune_point(split, params)
    pred = forest.predict(split.X_test, threshold=choice.threshold)
    return Result(pred, {"choice": choice, "forest": forest, "grid": grid})
