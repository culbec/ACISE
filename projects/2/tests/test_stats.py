import numpy as np
import pandas as pd
import pytest
from scipy.stats import wilcoxon
from sklearn.tree import DecisionTreeClassifier

from acise_p2_cidt.data.load import Dataset
from acise_p2_cidt.stats.tests import (
    paired_wilcoxon,
    per_fold,
    summary,
    vargha_delaney_a12,
)
from acise_p2_cidt.training.cv import run_cv


def test_a12():
    assert vargha_delaney_a12([1, 2, 3], [1, 2, 3]) == 0.5
    assert vargha_delaney_a12([4, 5], [1, 2, 3]) == 1.0
    assert vargha_delaney_a12([1, 2], [4, 5, 6]) == 0.0


def test_wilcoxon_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.random((2, 8))
    r = paired_wilcoxon(a, b)
    stat, p = wilcoxon(a, b)
    assert (r.statistic, r.p, r.n, r.flag) == (stat, p, 8, None)


def test_wilcoxon_ties():
    assert paired_wilcoxon([0.3] * 5, [0.3] * 5).flag == "all_ties"
    r = paired_wilcoxon([0.1, 0.2, 0.3, 0.4, 0.5], [0.0, 0.2, 0.0, 0.0, 0.0])
    assert (r.n, r.flag, r.min_p) == (4, "zero_diffs_dropped", 0.125)


def test_pipeline_with_stand_in_models():
    # In-memory stand-ins until tree/cart.py exists: an sklearn tree plays CART,
    # a fixed one-threshold rule plays a baseline.
    rng = np.random.default_rng(0)
    n = 400
    a = rng.normal(size=n)
    y = pd.Series((a + rng.normal(0, 0.8, n) > 0.5).astype(int))
    X = pd.DataFrame({"a": a, "b": rng.normal(size=n)})
    ds = Dataset(
        X,
        y,
        np.arange(n) < n // 5,
        {"repo": "owner/name"},
        pd.Series(pd.date_range("2026-01-01", periods=n, freq="h")),
    )

    def tree_stand_in(split):
        clf = DecisionTreeClassifier(max_depth=2, random_state=0)
        return clf.fit(split.X_fit, split.y_fit).predict(split.X_test).astype(bool)

    def rule_stand_in(split):
        return (split.X_test["a"] > 0.6).to_numpy()  # threshold in raw units

    tree, _ = run_cv(ds, ["a", "b"], tree_stand_in, "tree")
    rule, _ = run_cv(ds, ["a", "b"], rule_stand_in, "rule")
    results = pd.concat([tree, rule], ignore_index=True)

    s = summary(results)
    assert set(s["method"]) == {"tree", "rule"} and "mcc" in set(s["metric"])
    assert not s["metric"].isin(["fold", "n_test", "n_fail"]).any()

    t = per_fold(results, "mcc")
    assert t.shape == (5, 2)
    r = paired_wilcoxon(t["tree"], t["rule"])
    assert r.min_p == pytest.approx(0.0625) or r.flag  # 2^(1-5) with 5 folds
    assert 0 <= vargha_delaney_a12(t["tree"], t["rule"]) <= 1
