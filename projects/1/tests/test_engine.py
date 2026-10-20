import random

import numpy as np
import pandas as pd
import pytest
from deap import gp
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

from acise_p1_cigp.gp.engine import (
    GPConfig,
    _generate,
    evolve,
    make_pset,
    predict,
    score,
    statements,
)
from acise_p1_cigp.gp.simplify import motifs, simplify, to_python
from acise_p1_cigp.metrics.metrics import confusion, mcc

FAST = GPConfig(pop=200, n_gen=50, lambda_=0.02)


def planted(n=800, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.random((n, 4)), columns=["churn", "fail_ratio", "a", "b"])
    y = ((X.churn > 0.5) & (X.fail_ratio > 0.3)).to_numpy()
    cut = int(n * 0.75)
    return X[:cut], y[:cut], X[cut:], y[cut:]


@pytest.fixture(scope="module")
def runs():
    data = planted()
    return {v: [evolve(*data, GPConfig(**{**FAST.__dict__, "variant": v}), s) for s in range(3)] for v in ("so", "mo")}


def test_planted_target_recovered(runs):
    assert max(r.val["mcc"] for r in runs["so"]) > 0.9


@pytest.mark.parametrize("variant", ["so", "mo"])
def test_budget_respected(runs, variant):
    for r in runs[variant]:
        assert all(statements(i) <= 4 and i.height <= FAST.max_height for i in r.hof)
        assert (r.table["statements"] <= 4).all()


@pytest.mark.parametrize("variant", ["so", "mo"])
def test_archive_has_no_prediction_duplicates(runs, variant):
    X_fit = planted()[0]
    for r in runs[variant]:
        preds = {predict(i, r.pset, X_fit).tobytes() for i in r.hof}
        assert len(preds) == len(r.hof)


def test_mo_minimises_size(runs):
    for r in runs["mo"]:
        assert all(i.fitness.values[2] == statements(i) for i in r.hof)


def test_same_seed_same_program():
    data = planted()
    cfg = GPConfig(pop=50, n_gen=5)
    assert str(evolve(*data, cfg, 7).best) == str(evolve(*data, cfg, 7).best)


def test_score_and_constant_predict():
    pset = make_pset(["a", "b"])
    X = pd.DataFrame({"a": [0.1, 0.9, 0.5], "b": [0.2, 0.3, 0.4]})
    const = gp.PrimitiveTree.from_string("or_(True, False)", pset)
    pred = predict(const, pset, X)
    assert pred.shape == (3,) and pred.all()
    y = np.array([1, 0, 1])
    cfg = GPConfig(n_max=4, lambda_=0.1)
    assert score(y, pred, 2, cfg) == (mcc(confusion(y, pred)) - 0.1 * 2 / 4,)
    with pytest.raises(ValueError, match="n_max"):
        score(y, pred, 2, GPConfig())


def raw_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame(rng.integers(0, 50, (n, 3)).astype(float), columns=list("abc"))
    prep = make_pipeline(SimpleImputer(strategy="median"), QuantileTransformer(n_quantiles=200))
    prep.set_output(transform="pandas").fit(raw)
    return raw, prep, prep.transform(raw)


def test_simplify_preserves_predictions_and_to_python_matches():
    pset = make_pset(["a", "b", "c"])
    raw, prep, X = raw_frame()
    random.seed(0)
    for _ in range(200):
        t = gp.PrimitiveTree(_generate(pset, 1, 5, full=random.random() < 0.5))
        s = simplify(t, pset)
        expected = predict(t, pset, X)
        assert np.array_equal(predict(s, pset, X), expected), str(t)
        assert statements(s) <= statements(t)
        ns: dict = {}
        exec(to_python(t, pset, prep), ns)  # noqa: S102 - our own generated source
        got = np.array([ns["predict"](r) for _, r in raw.iterrows()], dtype=bool)
        assert np.array_equal(got, expected), str(t)


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("if_then_else(ge(a, 0.5), lt(b, 0.2), lt(b, 0.2))", "lt(b, 0.2)"),
        ("if_then_else(True, ge(a, 0.5), False)", "ge(a, 0.5)"),
        ("if_then_else(ge(a, 0.5), False, True)", "lt(a, 0.5)"),
        ("not_(not_(ge(a, 0.5)))", "ge(a, 0.5)"),
        ("not_(le(a, 0.5))", "gt(a, 0.5)"),
        ("lt(0.3, a)", "gt(a, 0.3)"),
        ("and_(ge(a, 0.2), ge(a, 0.6))", "ge(a, 0.6)"),
        ("or_(lt(b, 0.2), lt(b, 0.6))", "lt(b, 0.6)"),
        ("and_(ge(a, 0.5), True)", "ge(a, 0.5)"),
        ("or_(ge(a, 0.5), True)", "True"),
        ("lt(0.2, 0.7)", "True"),
        ("ge(a, a)", "True"),
    ],
)
def test_rewrites(expr, expected):
    pset = make_pset(["a", "b"])
    assert str(simplify(gp.PrimitiveTree.from_string(expr, pset), pset)) == expected


def test_to_python_elif_chain():
    pset = make_pset(["a", "b", "c"])
    _, prep, _ = raw_frame()
    t = gp.PrimitiveTree.from_string(
        "if_then_else(ge(a, 0.5), True, if_then_else(lt(b, 0.2), False, gt(c, 0.1)))",
        pset,
    )
    src = to_python(t, pset, prep)
    assert "    if " in src and "    elif " in src and "    else:" in src
    assert src.isascii()


def test_motifs_counts():
    pset = make_pset(["a", "b"])
    trees = [
        gp.PrimitiveTree.from_string("and_(ge(a, 0.2), lt(0.4, b))", pset),
        gp.PrimitiveTree.from_string("ge(a, 0.6)", pset),
    ]
    m = motifs(trees).set_index(["feature", "op"])
    assert m.loc[("a", "ge"), "count"] == 2
    assert m.loc[("a", "ge"), "thr_min"] == 0.2 and m.loc[("a", "ge"), "thr_max"] == 0.6
    assert m.loc[("b", "gt"), "count"] == 1
