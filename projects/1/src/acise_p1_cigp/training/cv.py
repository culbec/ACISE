import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler

from acise_p1_cigp.data.load import Dataset
from acise_p1_cigp.metrics.metrics import domain, scores

# Below this many failures a test fold gives unstable recall/MCC.
MIN_TEST_FAILURES = 10


@dataclass(frozen=True)
class Fold:
    k: int
    fit: np.ndarray  # row positions in ds.X; dev rows first
    val: np.ndarray  # tail of train: early stopping, picking the final program
    test: np.ndarray

    @property
    def train(self) -> np.ndarray:
        return np.r_[self.fit, self.val]


@dataclass(frozen=True)
class Split:
    """
    What a model's ``fit_predict`` receives for one fold. ``X_*`` hold the selected
    features, median-imputed and min-max scaled by ``prep``, which is fitted on the
    fit rows only (test values may fall outside [0, 1]).
    """

    X_fit: pd.DataFrame
    y_fit: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    raw_test: (
        pd.DataFrame
    )  # unscaled ds.X test rows, all columns (Parrot reads last_label)
    prep: Pipeline  # maps raw units to scaled space


@dataclass(frozen=True)
class Result:
    pred: np.ndarray  # bool per test row, True = will fail
    payload: Any = None  # e.g. hall of fame, convergence stats


def folds(ds: Dataset, n_splits: int = 5, val_frac: float = 0.2) -> list[Fold]:
    """
    Expanding-window time-series folds over the rows after dev; dev is prepended to
    every train, and the last ``val_frac`` of train is the inner validation split.

    Args:
        ds (Dataset): preprocessed dataset of one project.
        n_splits (int): number of test folds.
        val_frac (float): share of each train used as validation.

    Returns:
        list[Fold]: chronological folds.
    """
    if len(ds.meta["projects"]) != 1:
        raise ValueError(f"one project per CV run, got {ds.meta['projects']}")
    dev = np.flatnonzero(ds.dev_mask)
    rest = np.flatnonzero(~ds.dev_mask)
    if dev.size and rest.size and dev.max() > rest.min():
        raise ValueError("dev rows must come before all other rows")

    out = []
    for k, (tr, te) in enumerate(TimeSeriesSplit(n_splits).split(rest)):
        train = np.r_[dev, rest[tr]]
        cut = len(train) - int(len(train) * val_frac)
        test = rest[te]
        n_fail = int(ds.y.iloc[test].sum())
        if n_fail < MIN_TEST_FAILURES:
            warnings.warn(
                f"fold {k}: {n_fail} failures in test (< {MIN_TEST_FAILURES}); "
                "consider n_splits=3",
                stacklevel=2,
            )
        out.append(Fold(k, train[:cut], train[cut:], test))
    return out


def prepare(ds: Dataset, features: list[str], fold: Fold) -> Split:
    """
    Fit imputation and scaling on the fold's fit rows; apply them to val and test.
    """
    # keep_empty_features: an all-NaN column on fit must not shift the columns
    prep = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True), MinMaxScaler()
    ).set_output(transform="pandas")
    X = ds.X[features]
    return Split(
        X_fit=prep.fit_transform(X.iloc[fold.fit]),
        y_fit=ds.y.iloc[fold.fit],
        X_val=prep.transform(X.iloc[fold.val]),
        y_val=ds.y.iloc[fold.val],
        X_test=prep.transform(X.iloc[fold.test]),
        raw_test=ds.X.iloc[fold.test],
        prep=prep,
    )


def dev_split(ds: Dataset, features: list[str], val_frac: float = 0.2) -> Split:
    """
    Fit on the head of the dev segment, validate on its tail. Used for tuning and
    for the dev-segment runs that feed program (b), so no CV test fold is touched.

    Args:
        ds (Dataset): preprocessed dataset of one project.
        features (list[str]): frozen feature list.
        val_frac (float): share of dev used as validation.

    Returns:
        Split: `X_test` and `raw_test` are the validation rows.
    """
    dev = np.flatnonzero(ds.dev_mask)
    cut = len(dev) - int(len(dev) * val_frac)
    return prepare(ds, features, Fold(-1, dev[:cut], dev[cut:], dev[cut:]))


def run_cv(
    ds: Dataset,
    features: list[str],
    fit_predict: Callable[..., np.ndarray | Result],
    method: str,
    n_splits: int = 5,
    seeds: Iterable[int] | None = None,
) -> tuple[pd.DataFrame, dict[tuple[int, int | None], Any]]:
    """
    Score ``fit_predict`` on every fold (and seed). Without ``seeds`` it is called as
    ``fit_predict(split)`` once per fold; with seeds as ``fit_predict(split, seed)``.

    Returns:
        tuple: one row per fold x seed (method, fold, seed, n_test, n_fail, every
            ``scores`` and ``domain`` key), and the payloads keyed by (fold, seed).
    """
    seed_list = list(seeds) if seeds is not None else [None]
    rows, payloads = [], {}
    for fold in folds(ds, n_splits):
        split = prepare(ds, features, fold)
        y_test = ds.y.iloc[fold.test]
        for seed in seed_list:
            out = fit_predict(split) if seed is None else fit_predict(split, seed)
            res = out if isinstance(out, Result) else Result(out)
            rows.append(
                {
                    "method": method,
                    "fold": fold.k,
                    "seed": seed,
                    "n_test": len(fold.test),
                    "n_fail": int(y_test.sum()),
                    **scores(y_test, res.pred),
                    **domain(y_test, res.pred),
                }
            )
            if res.payload is not None:
                payloads[(fold.k, seed)] = res.payload
    return pd.DataFrame(rows), payloads
