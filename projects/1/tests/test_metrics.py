import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from acise_p1_cigp.metrics.metrics import (
    Counts,
    confusion,
    domain,
    majority,
    mcc,
    parrot,
    scores,
)

Y = [1, 1, 0, 0, 1]
P = [1, 0, 0, 1, 1]


def test_hand_computed():
    assert confusion(Y, P) == Counts(tp=2, fp=1, tn=1, fn=1)
    s = scores(Y, P)
    assert s["precision"] == pytest.approx(2 / 3)
    assert s["recall"] == s["tpr"] == pytest.approx(2 / 3)
    assert s["f1"] == pytest.approx(2 / 3)
    assert s["accuracy"] == pytest.approx(3 / 5)
    assert s["fpr"] == pytest.approx(1 / 2)
    assert s["mcc"] == pytest.approx(1 / 6)  # (2*1 - 1*1) / sqrt(3*3*2*2)
    d = domain(Y, P)
    assert d["pct_builds_saved"] == pytest.approx(40.0)
    assert d["pct_failures_caught"] == pytest.approx(200 / 3)


@pytest.mark.parametrize(
    "y, p",
    [
        ([1, 0, 1, 0], [0, 0, 0, 0]),  # all-pass prediction
        ([1, 0, 1, 0], [1, 1, 1, 1]),  # all-fail prediction
        ([0, 0, 0, 0], [0, 1, 0, 0]),  # no failures in the fold
    ],
)
def test_edge_cases_never_nan(y, p):
    s = {**scores(y, p), **domain(y, p)}
    assert all(math.isfinite(v) for v in s.values())
    assert mcc(confusion(y, p)) == s["mcc"] == 0.0


def test_fast_mcc_matches_sklearn():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, (2, 40))
    assert mcc(confusion(y, p)) == pytest.approx(scores(y, p)["mcc"])


def test_all_fail_prediction_saves_nothing():
    d = domain([1, 0, 1, 0], [1, 1, 1, 1])
    assert d == {"pct_builds_saved": 0.0, "pct_failures_caught": 100.0}


def test_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        confusion([1, 0], [1])


def test_parrot():
    t1 = SimpleNamespace(raw_test=pd.DataFrame({"last_label": [1, 0, 1]}))
    assert parrot(t1).tolist() == [False, True, False]
    t2 = SimpleNamespace(raw_test=pd.DataFrame({"src_churn": [5, 6]}))
    assert parrot(t2).tolist() == [False, False]


def test_majority():
    split = SimpleNamespace(y_fit=np.array([1, 1, 0]), X_test=np.zeros((4, 2)))
    assert majority(split).tolist() == [True] * 4
    split.y_fit = np.array([1, 0])  # tie predicts pass
    assert majority(split).tolist() == [False] * 4
