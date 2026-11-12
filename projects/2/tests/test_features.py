import numpy as np
import pandas as pd

from acise_p2_cidt.analysis.features import load_features, select_features
from acise_p2_cidt.data.load import Dataset


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
            "rare_one_two": 1 + (np.arange(n) == 1).astype(int),  # same, coded 1/2
            # many distinct values, one huge outlier: min-max variance ~1e-6
            "heavy_tail": np.r_[rng.integers(0, 50, n - 1), 1e6],
            "day_of_week": y * 3,  # nominal: excluded despite being predictive
            **{f"noise{i}": rng.normal(0, 1, n) for i in range(5)},
        }
    )
    return Dataset(
        X=X,
        y=y,
        dev_mask=np.ones(n, bool),
        meta={"target": "T2", "dataset": "owner__name__1", "nominal": ["day_of_week"]},
        order=pd.Series(pd.date_range("2026-01-01", periods=n, freq="h")),
    )


def test_select_features():
    s = select_features(synthetic(), n_max=4, n_boot=10)
    assert s.features[:2] == ["signal", "flag"] or s.features[:2] == ["flag", "signal"]
    assert "signal_dup" not in s.features
    dropped = {"constant", "rare_flag", "rare_one_two", "day_of_week"}
    assert dropped <= set(s.meta["dropped_in_clean"])
    assert "heavy_tail" not in s.meta["dropped_in_clean"]
    # picks of the duplicate are credited to the family representative
    assert s.stability["flag"] == 1 and s.stability["signal"] == 1
    assert s.stability["signal_dup"] == 0
    assert all(s.stability[f] >= 0.6 for f in s.features)


def test_save_and_load(tmp_path):
    s = select_features(synthetic(), n_max=4, n_boot=2)
    path = s.save(tmp_path)
    assert path.name == "selected_features_T2_owner__name__1.json"
    assert load_features("T2", "owner__name__1", tmp_path) == s.features


def test_deterministic():
    a, b = (
        select_features(synthetic(), n_boot=2),
        select_features(synthetic(), n_boot=2),
    )
    assert a.features == b.features and a.stability.equals(b.stability)


def test_drift():
    from acise_p2_cidt.analysis.features import drift

    t = drift(
        synthetic(), [np.arange(300), np.arange(600)], ["signal", "flag"], n_max=4
    )
    assert t.shape == (2, 2) and t.all().all()


def test_exclude():
    s = select_features(synthetic(), n_max=4, n_boot=2, exclude=("signal",))
    assert "signal" not in s.features and "signal" not in s.scores.index
    assert s.meta["excluded"] == ["signal"]
    assert "signal" not in s.meta["dropped_in_clean"]


def test_clean_keeps_history_empty_only_on_first_runs():
    from acise_p2_cidt.analysis.features import clean

    n = 100
    first = (np.arange(n) % 2 == 0).astype(int)  # half the runs open a lineage
    X = pd.DataFrame(
        {
            "lineage_first_run": first,
            "prev_failed": np.where(first == 1, np.nan, np.arange(n) % 3 == 0),
            "gappy": np.where(np.arange(n) % 3 == 0, 1.0, np.nan),  # 67% missing anywhere
        }
    )
    kept = clean(X)
    assert "prev_failed" in kept and "gappy" not in kept
    assert not kept.isna().any().any()
