import copy
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from acise_p2_cidt.tree.prune import collapse, pruning_steps

Criterion = Literal["gini", "entropy"]

EPSILON: float = float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class Tree:
    """Flat node arrays of a fitted tree, in depth-first (pre-order) node order."""

    feature: np.ndarray  # int, -1 at leaves
    threshold: np.ndarray  # float, go left when x <= threshold; nan at leaves
    left: np.ndarray  # int, -1 at leaves
    right: np.ndarray  # int, -1 at leaves
    value: np.ndarray  # (n_nodes, 2) weighted class totals: pass, fail
    n_samples: np.ndarray  # int, unweighted rows in the node
    impurity: np.ndarray  # float


# =============================================================================
# Impurity and split search
# =============================================================================


def _impurity(w0: np.ndarray, w1: np.ndarray, criterion: Criterion) -> np.ndarray:
    """Gini or entropy (log2) of the weighted class totals, elementwise."""
    w = w0 + w1
    with np.errstate(divide="ignore", invalid="ignore"):
        p0, p1 = w0 / w, w1 / w
        if criterion == "gini":
            out = 1.0 - p0 * p0 - p1 * p1
        else:
            out = -sum(np.where(p > 0, p * np.log2(p), 0.0) for p in (p0, p1))
    return np.where(w > 0, out, 0.0)


def _best_split(
    Xn: np.ndarray,
    w0: np.ndarray,
    w1: np.ndarray,
    node_impurity: float,
    w_root: float,
    min_samples_leaf: int,
    criterion: Criterion,
    cols: np.ndarray,
) -> tuple[int, float, float]:
    """
    Best split of one node over all features (Breiman Ch. 2).

    Each column is sorted once and the cumulative class totals are swept, so every
    position where the value changes is scored in one vectorised pass. Ties keep the
    first feature (column order) and the first position (smallest threshold).

    Args:
        Xn (np.ndarray): (n, p) node rows.
        w0 (np.ndarray): weight of each row towards class 0 (0 for class 1 rows).
        w1 (np.ndarray): weight of each row towards class 1.
        node_impurity (float): impurity of the node.
        w_root (float): total weight at the root, normalises the gain.
        min_samples_leaf (int): minimum rows on each side.
        criterion (Criterion): impurity measure.
        cols (np.ndarray): sorted columns to search (all, or a random-forest draw).

    Returns:
        tuple[int, float, float]: feature, threshold, gain; feature -1 if none.
    """
    Xn = Xn[:, cols]
    n, p = Xn.shape
    if n < 2 * min_samples_leaf:
        return -1, np.nan, -np.inf
    order = np.argsort(Xn, axis=0, kind="stable")
    Xs = np.take_along_axis(Xn, order, axis=0)
    l0 = np.cumsum(w0[order], axis=0)[:-1]  # left totals after row i: (n-1, p)
    l1 = np.cumsum(w1[order], axis=0)[:-1]
    t0, t1 = w0.sum(), w1.sum()
    r0, r1 = t0 - l0, t1 - l1
    gain = (
        (t0 + t1) * node_impurity
        - (l0 + l1) * _impurity(l0, l1, criterion)
        - (r0 + r1) * _impurity(r0, r1, criterion)
    ) / w_root

    n_left = np.arange(1, n)[:, None]
    valid = (
        (Xs[1:] != Xs[:-1])
        & (n_left >= min_samples_leaf)
        & (n - n_left >= min_samples_leaf)
    )
    gain = np.where(valid, gain, -np.inf)
    pos = gain.argmax(axis=0)  # first best position per feature
    best = gain[pos, np.arange(p)]
    j = int(best.argmax())  # first best feature
    if not np.isfinite(best[j]):
        return -1, np.nan, -np.inf
    a, b = Xs[pos[j], j], Xs[pos[j] + 1, j]
    t = a / 2.0 + b / 2.0
    if t == b or not np.isfinite(t):  # midpoint rounded up to b: keep a on the left
        t = a
    return int(cols[j]), float(t), float(best[j])


# =============================================================================
# Tree helpers
# =============================================================================


def _check_X(X: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, list[str] | None]:
    """Float (n, p) array and column names; NaN or inf raises."""
    names = [str(c) for c in X.columns] if isinstance(X, pd.DataFrame) else None
    A = np.asarray(X, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {A.shape}")
    if not np.isfinite(A).all():
        raise ValueError("X contains NaN or inf; impute before fitting the tree")
    return A, names


def _depth(tree: Tree) -> int:
    depth = np.zeros(len(tree.feature), dtype=int)
    for i in range(len(tree.feature)):  # pre-order: a parent comes before its children
        if tree.left[i] >= 0:
            depth[tree.left[i]] = depth[tree.right[i]] = depth[i] + 1
    return int(depth.max())


def _importances(tree: Tree, n_features: int) -> np.ndarray:
    """Total weighted impurity decrease per feature, normalised to sum to 1."""
    out = np.zeros(n_features)
    inner = np.flatnonzero(tree.feature >= 0)
    if len(inner) == 0:
        return out
    wi = tree.value.sum(axis=1) * tree.impurity
    dec = wi[inner] - wi[tree.left[inner]] - wi[tree.right[inner]]
    np.add.at(out, tree.feature[inner], dec)
    return out / out.sum() if out.sum() > 0 else out


def _fmt(v: float) -> str:
    return f"{v:.6g}"


# =============================================================================
# CART
# =============================================================================


class CART:
    """
    Binary classification tree (Breiman, Friedman, Olshen & Stone, 1984).

    Class 1 is "will fail". Splits `x_j <= t` with Gini or entropy, class weights as
    sample weights inside the impurity, and minimal cost-complexity pruning. The
    fitted model holds only NumPy arrays, scalars, strings and `Tree`, so skops can
    save it without custom hooks.

    Args:
        criterion (Criterion): "gini" or "entropy".
        max_depth (int | None): maximum depth; None grows until another rule stops.
        min_samples_split (int): a node with fewer rows becomes a leaf.
        min_samples_leaf (int): minimum rows in each child.
        min_impurity_decrease (float): minimum gain (normalised by root weight).
        class_weight (Literal["balanced"] | None): "balanced" weights class k by
            n / (2 * n_k).
        ccp_alpha (float): cost-complexity pruning alpha applied after growing.
        max_features (int | None): columns drawn at random at each node, as in a
            random forest (Breiman, 2001); None searches all columns.
        random_state (int | None): seed of that draw.
    """

    def __init__(
        self,
        criterion: Criterion = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        class_weight: Literal["balanced"] | None = None,
        ccp_alpha: float = 0.0,
        max_features: int | None = None,
        random_state: int | None = None,
    ):
        valid = {
            "criterion": criterion in ("gini", "entropy"),
            "max_depth": max_depth is None or max_depth >= 1,
            "min_samples_split": min_samples_split >= 2,
            "min_samples_leaf": min_samples_leaf >= 1,
            "min_impurity_decrease": min_impurity_decrease >= 0,
            "class_weight": class_weight in (None, "balanced"),
            "ccp_alpha": ccp_alpha >= 0,
            "max_features": max_features is None or max_features >= 1,
        }
        for name, ok in valid.items():
            if not ok:
                raise ValueError(f"invalid {name}, got {locals()[name]!r}")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_features = max_features
        self.random_state = random_state

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> CART:
        """
        Grows the tree depth-first, then prunes it if `ccp_alpha > 0`.

        Args:
            X (pd.DataFrame | np.ndarray): (n, p) features, no NaN.
            y (pd.Series | np.ndarray): 0/1 or bool label, 1 = failed.
            sample_weight (np.ndarray | None): per-row weights; None = 1.

        Returns:
            CART: self.
        """
        A, names = _check_X(X)
        yv = np.asarray(y)
        if yv.shape != (len(A),):
            raise ValueError(f"y must have shape ({len(A)},), got {yv.shape}")
        if not np.isin(yv, (0, 1)).all():
            raise ValueError(f"y must be 0/1, got values {np.unique(yv)[:5]}")
        yv = yv.astype(bool)
        sw = (
            np.ones(len(A))
            if sample_weight is None
            else np.asarray(sample_weight, float)
        )
        if sw.shape != (len(A),) or (sw < 0).any():
            raise ValueError("sample_weight must be (n,) and non-negative")
        if self.class_weight == "balanced":
            # maximum(..., 1): an absent class has no rows to weight.
            cw = len(yv) / (2.0 * np.maximum(np.bincount(yv, minlength=2), 1))
            sw = sw * cw[yv.astype(int)]

        self.n_features_ = A.shape[1]
        self.feature_names_ = names or [f"x{j}" for j in range(self.n_features_)]
        w0, w1 = np.where(yv, 0.0, sw), np.where(yv, sw, 0.0)
        w_root = float(sw.sum())
        max_depth = self.max_depth if self.max_depth is not None else np.inf
        rng, p = np.random.default_rng(self.random_state), self.n_features_
        k = min(self.max_features or p, p)

        # One row per node: feature, threshold, left, right, v0, v1, n_samples, imp.
        rows: list[list[float]] = []
        # (rows, depth, parent, is_left); left pushed last so it gets the next id
        stack = [(np.arange(len(A)), 0, -1, True)]
        while stack:
            idx, depth, par, is_left = stack.pop()
            node = len(rows)
            if par >= 0:
                rows[par][2 if is_left else 3] = node
            v0, v1 = float(w0[idx].sum()), float(w1[idx].sum())
            node_imp = float(_impurity(np.array(v0), np.array(v1), self.criterion))
            rows.append([-1, np.nan, -1, -1, v0, v1, len(idx), node_imp])
            if (
                depth >= max_depth
                or len(idx) < self.min_samples_split
                or node_imp <= EPSILON
            ):
                continue
            # Sorted draw: equal-gain ties still go to the first column. A draw with
            # no valid split makes a leaf (sklearn would draw more columns).
            cols = np.arange(p) if k == p else np.sort(rng.choice(p, k, replace=False))
            j, t, gain = _best_split(
                A[idx], w0[idx], w1[idx], node_imp, w_root,
                self.min_samples_leaf, self.criterion, cols,
            )  # fmt: skip
            # As sklearn: a zero-gain split is kept at the default 0.0, so a parity
            # or XOR pattern can still be split below the root.
            if j < 0 or gain + EPSILON < self.min_impurity_decrease:
                continue
            rows[node][:2] = [j, t]
            go_left = A[idx, j] <= t
            stack.append((idx[~go_left], depth + 1, node, False))
            stack.append((idx[go_left], depth + 1, node, True))

        c = np.array(rows, dtype=float).T
        self._set_tree(
            Tree(
                feature=c[0].astype(int),
                threshold=c[1],
                left=c[2].astype(int),
                right=c[3].astype(int),
                value=c[4:6].T.copy(),
                n_samples=c[6].astype(int),
                impurity=c[7],
            )
        )
        if self.ccp_alpha > 0:
            self._set_tree(self.pruned(self.ccp_alpha).tree_)
        return self

    def _set_tree(self, tree: Tree) -> None:
        self.tree_ = tree
        self.depth_ = _depth(tree)
        self.n_leaves_ = int((tree.left < 0).sum())
        self.feature_importances_ = _importances(tree, self.n_features_)

    def cost_complexity_path(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Pruning path of the fitted tree.

        Returns:
            tuple[np.ndarray, np.ndarray]: alphas (increasing, first 0) and the total
                leaf impurity of the subtree at each alpha.
        """
        alphas, totals, _ = pruning_steps(self.tree_)
        return np.array(alphas), np.array(totals)

    def pruned(self, alpha: float) -> CART:
        """
        Smallest minimal-cost subtree for `alpha`; `self` is not changed.

        Args:
            alpha (float): complexity cost per leaf, >= 0.

        Returns:
            CART: a copy holding the pruned tree.
        """
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha!r}")
        alphas, _, nodes = pruning_steps(self.tree_)
        cut = {t for a, t in zip(alphas[1:], nodes[1:]) if a <= alpha}
        out = copy.copy(self)
        out._set_tree(collapse(self.tree_, cut))
        return out

    def _rows(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        A, names = _check_X(X)
        if A.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {A.shape[1]} features, the tree {self.n_features_}"
            )
        if names is not None and names != self.feature_names_:
            raise ValueError("X columns differ from the fitted feature names or order")
        return A

    def apply(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Leaf id of every row."""
        A, tr = self._rows(X), self.tree_
        node = np.zeros(len(A), dtype=int)
        active = tr.left[node] >= 0
        while active.any():
            i = node[active]
            go_left = A[active, tr.feature[i]] <= tr.threshold[i]
            node[active] = np.where(go_left, tr.left[i], tr.right[i])
            active = tr.left[node] >= 0
        return node

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """(n, 2) weighted class shares of each row's leaf: P(pass), P(fail)."""
        v = self.tree_.value[self.apply(X)]
        return v / v.sum(axis=1, keepdims=True)

    def predict(
        self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        """
        True where P(fail) > threshold. Strict, so a 50/50 leaf predicts "pass" at
        the default, as sklearn's argmax does.
        """
        return self.predict_proba(X)[:, 1] > threshold

    def _leaf_text(self, i: int) -> str:
        v = self.tree_.value[i]
        p = v[1] / v.sum() if v.sum() > 0 else 0.0
        return f"leaf: P(fail) = {p:.2f} from {self.tree_.n_samples[i]} runs"

    def export_text(self, max_depth: int | None = None) -> str:
        """
        The tree as indented rules, in the wording of `explain`.

        Args:
            max_depth (int | None): deeper subtrees are shown as "...".

        Returns:
            str: one line per test or leaf.
        """
        tr, lines = self.tree_, []
        # Entries are a node id or a ready line; the "> t" line waits on the stack
        # until the whole left subtree is printed.
        stack: list[tuple[int | str, int]] = [(0, 0)]
        while stack:
            item, d = stack.pop()
            pad = "|   " * d + "|--- "
            if isinstance(item, str):
                lines.append(pad + item)
            elif tr.left[item] < 0:
                lines.append(pad + self._leaf_text(item))
            elif max_depth is not None and d >= max_depth:
                lines.append(pad + "...")
            else:
                name = self.feature_names_[tr.feature[item]]
                t = _fmt(tr.threshold[item])
                lines.append(pad + f"{name} <= {t}")
                stack += [
                    (int(tr.right[item]), d + 1),
                    (f"{name} > {t}", d),
                    (int(tr.left[item]), d + 1),
                ]
        return "\n".join(lines)

    def explain(self, x: pd.Series | pd.DataFrame | np.ndarray) -> list[str]:
        """
        Decision path of one row, in raw feature units.

        Args:
            x (pd.Series | pd.DataFrame | np.ndarray): one row.

        Returns:
            list[str]: one line per test (`"name = value <= t"` or `> t`), then the
                leaf line with P(fail) and the number of training runs.
        """
        if isinstance(x, pd.Series):
            x = x.to_frame().T
        row = self._rows(x if isinstance(x, pd.DataFrame) else np.atleast_2d(x))
        if len(row) != 1:
            raise ValueError(f"explain takes one row, got {len(row)}")
        tr, out, i = self.tree_, [], 0
        while tr.left[i] >= 0:
            j, t = tr.feature[i], tr.threshold[i]
            xv = row[0, j]
            op, nxt = ("<=", tr.left[i]) if xv <= t else (">", tr.right[i])
            out.append(f"{self.feature_names_[j]} = {_fmt(xv)} {op} {_fmt(t)}")
            i = nxt
        return out + [self._leaf_text(i)]
