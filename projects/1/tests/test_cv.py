import dataclasses

import numpy as np
import pandas as pd
import pytest

from acise_p1_cigp.data.load import Dataset
from acise_p1_cigp.metrics.metrics import majority
from acise_p1_cigp.training.cv import Result, folds, prepare, run_cv


def dataset(n=400, fail_rate=0.5, seed=0):
    rng = np.random.default_rng(seed)
    y = pd.Series((rng.random(n) < fail_rate).astype(int), name="failed")
    X = pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n), "last_label": 1}
    )
    X.loc[::7, "a"] = np.nan
    dev = np.arange(n) < n // 5
    meta = {"projects": ["p"], "target": "T1"}
    return Dataset(
        X=X, y=y, dev_mask=dev, meta=meta, groups=pd.Series("p", index=X.index)
    )


def test_fold_properties():
    ds = dataset()
    dev = np.flatnonzero(ds.dev_mask)
    prev_train = 0
    for f in folds(ds):
        assert f.fit.max() < f.val.min() and f.val.max() < f.test.min()  # time order
        assert np.isin(dev, f.fit).all()
        assert np.array_equal(
            f.train, np.arange(len(f.train))
        )  # contiguous, val = tail
        assert len(f.val) == int(len(f.train) * 0.2)
        assert len(f.train) > prev_train  # expanding window
        prev_train = len(f.train)


def test_pooled_rejected():
    ds = dataclasses.replace(dataset(), meta={"projects": ["p", "q"]})
    with pytest.raises(ValueError, match="one project"):
        folds(ds)


def test_few_failures_warns():
    with pytest.warns(UserWarning, match="failures in test"):
        folds(dataset(fail_rate=0.02))


def test_scaler_fitted_on_fit_rows_only():
    ds = dataset()
    f = folds(ds)[0]
    X = ds.X.copy()
    X.loc[f.test[0], "b"] = 1e6  # spike in test
    spiked = prepare(dataclasses.replace(ds, X=X), ["a", "b"], f)
    clean = prepare(ds, ["a", "b"], f)
    pd.testing.assert_frame_equal(spiked.X_fit, clean.X_fit)
    assert spiked.X_test["b"].max() == 1  # quantile scaling clips to the fit range
    assert not spiked.X_fit.isna().any().any()  # imputed


def test_run_cv_deterministic_and_seeded():
    ds = dataset()
    df, payloads = run_cv(ds, ["a", "b"], majority, "majority")
    assert len(df) == 5 and not payloads
    assert df["seed"].isna().to_numpy().all()

    def seeded(split, seed):
        return Result(np.zeros(len(split.X_test), dtype=bool), payload=seed)

    df, payloads = run_cv(ds, ["a", "b"], seeded, "gp", seeds=[0, 1])
    assert len(df) == 10 and payloads[(4, 1)] == 1
    assert (df["pct_builds_saved"] == 100).all()
