import numpy as np
import pandas as pd

from acise_p1_cigp.analysis.features import select_features
from acise_p1_cigp.data.load import Dataset


def synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = pd.Series(rng.integers(0, 2, n))
    signal = y * 3 + rng.normal(0, 1, n)
    X = pd.DataFrame(
        {
            "signal": signal,
            "signal_dup": signal * 2 + rng.normal(0, 0.01, n),  # redundant
            "flag": np.where(rng.random(n) < 0.15, 1 - y, y),  # informative binary
            "constant": 1.0,
            "rare_flag": (np.arange(n) == 0).astype(int),  # minority < 1%
            "day_of_week": y * 3,  # nominal: excluded despite being predictive
            **{f"noise{i}": rng.normal(0, 1, n) for i in range(5)},
        }
    )
    return Dataset(X=X, y=y, dev_mask=np.ones(n, bool), meta={"target": "T2", "projects": ["p"]}, groups=pd.Series("p", index=X.index))


def test_select_features():
    s = select_features(synthetic(), n_max=4, n_boot=10)
    assert s.features[:2] == ["signal", "flag"] or s.features[:2] == ["flag", "signal"]
    assert "signal_dup" not in s.features
    assert {"constant", "rare_flag", "day_of_week"} <= set(s.meta["dropped_in_clean"])
    # picks of the duplicate are credited to the family representative
    assert s.stability["flag"] == 1 and s.stability["signal"] == 1
    assert s.stability["signal_dup"] == 0
    assert all(s.stability[f] >= 0.6 for f in s.features)


def test_deterministic():
    a, b = select_features(synthetic(), n_boot=2), select_features(synthetic(), n_boot=2)
    assert a.features == b.features and a.stability.equals(b.stability)


def test_drift():
    from acise_p1_cigp.analysis.features import drift

    t = drift(synthetic(), [np.arange(300), np.arange(600)], ["signal", "flag"], n_max=4)
    assert t.shape == (2, 2) and t.all().all()
