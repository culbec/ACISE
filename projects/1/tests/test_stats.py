import numpy as np
import pandas as pd
import pytest
from scipy.stats import wilcoxon
from sklearn.tree import DecisionTreeClassifier

from acise_p1_cigp.data.load import Dataset
from acise_p1_cigp.stats.tests import (
    paired_wilcoxon,
    per_fold,
    position_per_fold,
    summary,
    vargha_delaney_a12,
)
from acise_p1_cigp.training.cv import run_cv


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
    # In-memory stand-ins until GP and program (b) exist: seeded trees play GP,
    # a fixed one-threshold rule plays program (b).
    rng = np.random.default_rng(0)
    n = 400
    a = rng.normal(size=n)
    y = pd.Series((a + rng.normal(0, 0.8, n) > 0.5).astype(int))
    X = pd.DataFrame({"a": a, "b": rng.normal(size=n)})
    ds = Dataset(
        X, y, np.arange(n) < n // 5, {"projects": ["p"]}, pd.Series("p", index=X.index)
    )

    def gp_stand_in(split, seed):
        clf = DecisionTreeClassifier(max_depth=2, max_features=1, random_state=seed)
        return clf.fit(split.X_fit, split.y_fit).predict(split.X_test).astype(bool)

    def b_stand_in(split):
        return (split.X_test["a"] > 0.6).to_numpy()  # threshold in scaled units

    gp, _ = run_cv(ds, ["a", "b"], gp_stand_in, "gp", seeds=range(4))
    hb, _ = run_cv(ds, ["a", "b"], b_stand_in, "b")
    results = pd.concat([gp, hb], ignore_index=True)

    s = summary(results)
    assert set(s["method"]) == {"gp", "b"} and "mcc" in set(s["metric"])
    assert not s.query("metric == 'seed'").shape[0]

    t = per_fold(results, "mcc")
    assert t.shape == (5, 2)
    r = paired_wilcoxon(t["gp"], t["b"])
    assert r.min_p == pytest.approx(0.0625) or r.flag
    assert 0 <= vargha_delaney_a12(t["gp"], t["b"]) <= 1

    pos = position_per_fold(results, "mcc", point="b", dist="gp")
    assert len(pos) == 5 and pos.between(0, 1).all()
