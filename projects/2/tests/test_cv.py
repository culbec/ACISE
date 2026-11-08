import dataclasses

import numpy as np
import pandas as pd
import pytest

from acise_p2_cidt.data.load import Dataset
from acise_p2_cidt.metrics.metrics import majority
from acise_p2_cidt.training.cv import (
    Result,
    folds,
    prepare,
    run_cv,
)


def dataset(n=400, fail_rate=0.5, seed=0):
    rng = np.random.default_rng(seed)
    y = pd.Series((rng.random(n) < fail_rate).astype(int), name="failed")
    X = pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n), "prev_failed": 0}
    )
    X.loc[::7, "a"] = np.nan
    dev = np.arange(n) < n // 5
    meta = {"dataset": "owner__name__1", "target": "T1"}
    order = pd.Series(pd.date_range("2026-01-01", periods=n, freq="h"))
    return Dataset(X=X, y=y, dev_mask=dev, meta=meta, order=order)


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


def test_no_dataset_rejected():
    ds = dataclasses.replace(dataset(), meta={"target": "T1"})
    with pytest.raises(ValueError, match="dataset"):
        folds(ds)


def test_few_failures_warns():
    with pytest.warns(UserWarning, match="failures in test"):
        folds(dataset(fail_rate=0.02))


def test_imputer_fitted_on_fit_rows_only():
    ds = dataset()
    f = folds(ds)[0]
    X = ds.X.copy()
    X.loc[f.test[0], "b"] = 1e6  # spike in test
    spiked = prepare(dataclasses.replace(ds, X=X), ["a", "b"], f)
    clean = prepare(ds, ["a", "b"], f)
    pd.testing.assert_frame_equal(spiked.X_fit, clean.X_fit)
    assert spiked.X_test["b"].max() == 1e6  # no scaling: raw units, no clipping
    assert not spiked.X_fit.isna().any().any()  # imputed


def test_imputer_keeps_raw_values():
    ds = dataset()
    f = folds(ds)[0]
    split = prepare(ds, ["a", "b"], f)
    raw = ds.X.iloc[f.test][["a", "b"]]
    present = raw.notna()
    assert (split.X_test[present] == raw[present]).sum().sum() == present.sum().sum()
    assert not split.X_test.isna().any().any()
    # missing values get the median of the fit rows, not of the test rows
    fit_median = ds.X.iloc[f.fit]["a"].median()
    assert (split.X_test.loc[~present["a"], "a"] == fit_median).all()


def test_run_cv_one_row_per_fold():
    ds = dataset()
    df, payloads = run_cv(ds, ["a", "b"], majority, "majority")
    assert len(df) == 5 and not payloads
    assert "seed" not in df

    def all_pass(split):
        return Result(np.zeros(len(split.X_test), dtype=bool), payload=len(split.X_fit))

    df, payloads = run_cv(ds, ["a", "b"], all_pass, "all_pass")
    assert len(df) == 5 and sorted(payloads) == [0, 1, 2, 3, 4]
    assert payloads[0] < payloads[4]  # expanding window
    assert (df["pct_builds_saved"] == 100).all()


def test_run_cv_single_source_is_all():
    df, _ = run_cv(dataset(), ["a", "b"], majority, "majority")
    assert (df["source"] == "all").all()


def test_run_cv_per_source_rows():
    ds = dataset()
    source = pd.Series(np.where(np.arange(len(ds.y)) % 3 == 0, "s1", "s2"))
    ds = dataclasses.replace(ds, source=source)
    df, _ = run_cv(ds, ["a", "b"], majority, "majority")
    assert len(df) == 5 * 3 and set(df["source"]) == {"all", "s1", "s2"}
    pooled = df[df["source"] == "all"].set_index("fold")
    parts = df[df["source"] != "all"].groupby("fold")[["n_test", "n_fail"]].sum()
    pd.testing.assert_frame_equal(parts, pooled[["n_test", "n_fail"]])
    # each slice is scored on its own rows only
    assert (df.loc[df["source"] == "s1", "n_test"] < pooled["n_test"].to_numpy()).all()


def test_per_source_without_failures_is_nan():
    ds = dataset()
    source = pd.Series(np.where(np.arange(len(ds.y)) % 3 == 0, "s1", "s2"))
    y = ds.y.where(source != "s1", 0)  # s1 never fails
    ds = dataclasses.replace(ds, y=y, source=source)
    df, _ = run_cv(ds, ["a", "b"], majority, "majority")
    s1 = df[df["source"] == "s1"]
    assert s1[["mcc", "recall", "f1", "precision"]].isna().all().all()
    assert s1["accuracy"].notna().all() and (s1["n_test"] > 0).all()
    assert df.loc[df["source"] != "s1", "mcc"].notna().all()
