import json

import numpy as np
import pandas as pd
import pytest
import skops.io as sio
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from acise_p2_cidt.data import load
from acise_p2_cidt.tree.cart import CART, Tree, _impurity

# (criterion, class_weight, max_depth, min_samples_leaf) compared with sklearn.
CASES = [
    (c, w, d, m)
    for c in ("gini", "entropy")
    for w in (None, "balanced")
    for d, m in ((3, 1), (5, 10), (None, 1), (None, 5))
]
# Configurations whose sklearn tree has no gain ties at any node (found by the
# divergence walk below), so structure and pruning path must match exactly.
TIE_FREE = [
    (0, "gini", None, 5, 10),
    (1, "entropy", "balanced", 5, 10),
    (2, "entropy", "balanced", 3, 1),
    (4, "entropy", None, None, 5),
]
TRUSTED = ["acise_p2_cidt.tree.cart.CART", "acise_p2_cidt.tree.cart.Tree"]


def synthetic(seed: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(500, 8, random_state=seed, weights=[0.8])
    # sklearn fits on float32; rounding first gives both trees the same values.
    return X.astype(np.float32).astype(np.float64), y


def weights(y: np.ndarray, class_weight: str | None) -> np.ndarray:
    if class_weight is None:
        return np.ones(len(y))
    return (len(y) / (2 * np.bincount(y)))[y]


def fit_both(X, y, criterion, class_weight, max_depth, min_samples_leaf, **kw):
    params = dict(
        criterion=criterion,
        class_weight=class_weight,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        **kw,
    )
    ours = CART(**params).fit(X, y)
    sk = DecisionTreeClassifier(**params, random_state=0).fit(X, y)
    return ours, sk


def divergences(ours, sk, X, y, sw, criterion) -> list[float]:
    """
    Walk both trees in step and return the gain gap at every node where they pick
    different splits. sklearn scans features in a random order, so an equal-gain
    tie can go to another feature; a gap of ~0 means the choice was a tie.
    """
    o, s = ours.tree_, sk.tree_
    w0, w1 = np.where(y == 1, 0.0, sw), np.where(y == 1, sw, 0.0)

    def gain(idx, j, t):
        if j < 0:
            return 0.0
        go = X[idx, j] <= t
        wi = [
            (w0[m].sum() + w1[m].sum())
            * _impurity(np.array(w0[m].sum()), np.array(w1[m].sum()), criterion)
            for m in (idx, idx[go], idx[~go])
        ]
        return (wi[0] - wi[1] - wi[2]) / sw.sum()

    gaps, stack = [], [(0, 0, np.arange(len(X)))]
    while stack:
        i, k, idx = stack.pop()
        fo, fs = o.feature[i], s.feature[k]
        if fo < 0 and fs < 0:
            continue
        if fo == fs and o.threshold[i] == pytest.approx(s.threshold[k], abs=1e-6):
            go = X[idx, fo] <= o.threshold[i]
            stack += [
                (o.left[i], s.children_left[k], idx[go]),
                (o.right[i], s.children_right[k], idx[~go]),
            ]
            continue
        gaps.append(abs(gain(idx, fo, o.threshold[i]) - gain(idx, fs, s.threshold[k])))
    return gaps


# =============================================================================
# Agreement with sklearn
# =============================================================================


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("case", CASES)
def test_matches_sklearn_up_to_ties(seed, case):
    X, y = synthetic(seed)
    ours, sk = fit_both(X, y, *case)
    assert ours.tree_.impurity[0] == pytest.approx(sk.tree_.impurity[0], abs=1e-12)
    gaps = divergences(ours, sk, X, y, weights(y, case[1]), case[0])
    assert max(gaps, default=0.0) < 1e-12  # every structural difference is a tie
    if not gaps:
        assert np.array_equal(ours.predict(X), sk.predict(X).astype(bool))
        assert ours.n_leaves_ == sk.get_n_leaves()
        assert ours.feature_importances_ == pytest.approx(sk.feature_importances_)


@pytest.mark.parametrize("seed, criterion, class_weight, max_depth, msl", TIE_FREE)
def test_pruning_path_matches_sklearn(seed, criterion, class_weight, max_depth, msl):
    X, y = synthetic(seed)
    ours, sk = fit_both(X, y, criterion, class_weight, max_depth, msl)
    assert not divergences(ours, sk, X, y, weights(y, class_weight), criterion)
    alphas, totals = ours.cost_complexity_path()
    path = sk.cost_complexity_pruning_path(X, y)
    np.testing.assert_allclose(alphas, path.ccp_alphas, rtol=0, atol=1e-10)
    np.testing.assert_allclose(totals, path.impurities, rtol=0, atol=1e-10)

    # Midway between two path alphas, so no step sits on the boundary.
    a = (alphas[len(alphas) // 2] + alphas[len(alphas) // 2 + 1]) / 2
    sk_pruned = DecisionTreeClassifier(
        criterion=criterion,
        class_weight=class_weight,
        max_depth=max_depth,
        min_samples_leaf=msl,
        ccp_alpha=a,
        random_state=0,
    ).fit(X, y)
    pruned = ours.pruned(a)
    assert pruned.n_leaves_ == sk_pruned.get_n_leaves()
    assert np.array_equal(pruned.predict(X), sk_pruned.predict(X).astype(bool))
    assert ours.n_leaves_ == sk.get_n_leaves()  # self unchanged
    refit = CART(
        criterion=criterion,
        class_weight=class_weight,
        max_depth=max_depth,
        min_samples_leaf=msl,
        ccp_alpha=a,
    ).fit(X, y)
    assert np.array_equal(refit.tree_.feature, pruned.tree_.feature)


@pytest.mark.skipif(
    not load.table_path("n8n-io/n8n").is_file(), reason="built n8n table not present"
)
@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_real_slice_matches_sklearn_up_to_ties(criterion):
    ds = load.load("n8n-io/n8n", "T2")
    sel = load.SCHEMA_DIR.parent / "features" / "selected_features_T2_n8n-io__n8n.json"
    X = ds.X[json.loads(sel.read_text())["features"]].iloc[:2000]
    X = X.fillna(X.median()).to_numpy().astype(np.float32).astype(np.float64)
    y = ds.y.to_numpy()[:2000]
    ours, sk = fit_both(X, y, criterion, "balanced", 4, 10)
    assert ours.tree_.impurity[0] == pytest.approx(sk.tree_.impurity[0], abs=1e-12)
    assert (
        max(divergences(ours, sk, X, y, weights(y, "balanced"), criterion), default=0)
        < 1e-12
    )


# =============================================================================
# Hand-checked behaviour
# =============================================================================


def test_toy_split_and_tie_rule():
    # y = 0 0 1 0 1 1. Root Gini = 0.5. Split after row 2: L [0,0] pure,
    # R [1,0,1,1] Gini 1 - (1 + 9)/16 = 0.375 -> gain (6*0.5 - 4*0.375)/6 = 0.25.
    # Split after row 4 (L [0,0,1,0], R [1,1]) gives the same 0.25; the first wins.
    X = np.arange(1.0, 7.0)[:, None]
    tr = CART(max_depth=1).fit(X, [0, 0, 1, 0, 1, 1]).tree_
    assert tr.threshold[0] == 2.5
    assert tr.n_samples.tolist() == [6, 2, 4]
    assert tr.impurity == pytest.approx([0.5, 0.0, 0.375])
    w = tr.value.sum(axis=1)
    assert (w[0] * tr.impurity[0] - w[1:] @ tr.impurity[1:]) / w[0] == pytest.approx(
        0.25
    )


def test_zero_gain_root_still_splits():
    # XOR: every root split has gain 0; as in sklearn the tree splits anyway.
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    tree = CART().fit(X, y)
    assert tree.depth_ == 2 and np.array_equal(tree.predict(X), y.astype(bool))


def test_min_samples_leaf_and_max_depth_respected():
    X, y = synthetic(3)
    tree = CART(max_depth=4, min_samples_leaf=7).fit(X, y)
    leaves = tree.tree_.left < 0
    assert (tree.tree_.n_samples[leaves] >= 7).all()
    assert tree.depth_ <= 4
    assert tree.n_leaves_ == leaves.sum() == len(np.unique(tree.apply(X)))


def test_balanced_moves_threshold_into_majority():
    # 9:1, minority shifted right. Unweighted, the stump isolates only the far tail;
    # balanced weights make each failure count 9x and lower the threshold.
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(0, 1, 90), rng.normal(1.5, 1, 10)].round(3)[:, None]
    y = np.r_[np.zeros(90, int), np.ones(10, int)]
    plain = CART(max_depth=1).fit(X, y)
    bal = CART(max_depth=1, class_weight="balanced").fit(X, y)
    assert bal.tree_.threshold[0] < plain.tree_.threshold[0]
    assert y[bal.predict(X)].sum() > y[plain.predict(X)].sum()


def test_explain_and_export_text():
    X = pd.DataFrame({"a": [0, 0, 1, 1, 2, 2.0], "b": [0, 1, 0, 1, 0, 1.0]})
    # Root: b <= 0.5 leaves a pure left [0,0,0] (gain 2/9 vs 1/9 for a <= 0.5);
    # right [0,1,1] by a: a <= 0.5 isolates the 0.
    y = np.array([0, 0, 0, 1, 0, 1])
    tree = CART(max_depth=2).fit(X, y)
    lines = tree.explain(X.iloc[3])
    assert lines == ["b = 1 > 0.5", "a = 1 > 0.5", "leaf: P(fail) = 1.00 from 2 runs"]
    text = tree.export_text().splitlines()
    inner = int((tree.tree_.left >= 0).sum())
    assert len(text) == 2 * inner + tree.n_leaves_  # "<=" and ">" per test, 1 per leaf
    assert text[0] == "|--- b <= 0.5"
    assert tree.export_text(max_depth=1).splitlines()[-1] == "|   |--- ..."


def test_skops_round_trip(tmp_path):
    X, y = synthetic(0)
    tree = CART(max_depth=4, class_weight="balanced").fit(X, y)
    path = tmp_path / "cart.skops"
    sio.dump(tree, path)
    loaded = sio.load(path, trusted=TRUSTED)
    assert isinstance(loaded, CART) and isinstance(loaded.tree_, Tree)
    assert np.array_equal(loaded.predict_proba(X), tree.predict_proba(X))


def test_bad_input_raises():
    X = np.array([[0.0], [1.0], [np.nan]])
    with pytest.raises(ValueError, match="NaN"):
        CART().fit(X, [0, 1, 0])
    tree = CART().fit(pd.DataFrame({"a": [0.0, 1.0]}), [0, 1])
    with pytest.raises(ValueError, match="NaN"):
        tree.predict(pd.DataFrame({"a": [np.nan]}))
    with pytest.raises(ValueError, match="feature names"):
        tree.predict(pd.DataFrame({"b": [0.0]}))
    with pytest.raises(ValueError, match="criterion"):
        CART(criterion="mse")  # pyright: ignore[reportArgumentType]
