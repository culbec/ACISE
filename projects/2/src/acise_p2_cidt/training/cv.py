import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit

from acise_p2_cidt.data.load import Dataset
from acise_p2_cidt.metrics.metrics import domain, scores
from acise_p2_cidt.utils.logging.log import get_logger, log

logger = get_logger(__name__)

# Below this many failures a test fold gives unstable recall/MCC.
MIN_TEST_FAILURES = 10
# Scores of the fail class. A source's slice of a fold can hold no failure; there
# they are undefined, so its row gets NaN and per-source means skip it.
FAIL_CLASS_SCORES = ("precision", "recall", "tpr", "f1", "mcc", "pct_failures_caught")


def _imputer() -> SimpleImputer:
    """
    Median imputation, the only preprocessing: a tree does not change under monotone
    transforms, so nothing is scaled and thresholds stay in raw units.
    """
    # keep_empty_features: an all-NaN column on fit must not shift the columns.
    return SimpleImputer(strategy="median", keep_empty_features=True).set_output(
        transform="pandas"
    )


@dataclass(frozen=True)
class Fold:
    k: int
    fit: np.ndarray  # row positions in ds.X; dev rows first
    val: np.ndarray  # tail of train: pruning alpha, threshold, hyperparameters
    test: np.ndarray

    @property
    def train(self) -> np.ndarray:
        return np.r_[self.fit, self.val]


@dataclass(frozen=True)
class Split:
    """
    What a model's ``fit_predict`` receives for one fold. ``X_*`` hold the selected
    features, median-imputed by ``prep``, which is fitted on the fit rows only.
    """

    X_fit: pd.DataFrame
    y_fit: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    raw_test: (
        pd.DataFrame
    )  # unprepared ds.X test rows, all columns (Parrot reads prev_failed)
    prep: SimpleImputer  # fitted on the fit rows; reused on new runs


@dataclass(frozen=True)
class Result:
    pred: np.ndarray  # bool per test row, True = will fail
    payload: Any = None  # e.g. the fitted tree, chosen hyperparameters


def folds(ds: Dataset, n_splits: int = 5, val_frac: float = 0.2) -> list[Fold]:
    """
    Expanding-window time-series folds over the rows after dev; dev is prepended to
    every train, and the last ``val_frac`` of train is the inner validation split.

    Args:
        ds (Dataset): loaded dataset, one source or a pool.
        n_splits (int): number of test folds.
        val_frac (float): share of each train used as validation.

    Returns:
        list[Fold]: chronological folds.
    """
    if not ds.meta.get("dataset"):
        raise ValueError(
            "the dataset has no name (meta['dataset']); load it with `load`"
        )
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
    Fit the imputer on the fold's fit rows; apply it to val and test.

    Args:
        ds (Dataset): loaded dataset of one repository.
        features (list[str]): frozen feature list.
        fold (Fold): row positions of fit, val and test.

    Returns:
        Split: prepared fold.
    """
    pipe = _imputer()
    X = ds.X[features]
    return Split(
        X_fit=pipe.fit_transform(X.iloc[fold.fit]),
        y_fit=ds.y.iloc[fold.fit],
        X_val=pipe.transform(X.iloc[fold.val]),
        y_val=ds.y.iloc[fold.val],
        X_test=pipe.transform(X.iloc[fold.test]),
        raw_test=ds.X.iloc[fold.test],
        prep=pipe,
    )


def dev_split(ds: Dataset, features: list[str], val_frac: float = 0.2) -> Split:
    """
    Fit on the head of the dev segment, validate on its tail. Used for tuning, so no
    CV test fold is touched.

    Args:
        ds (Dataset): loaded dataset of one repository.
        features (list[str]): frozen feature list.
        val_frac (float): share of dev used as validation.

    Returns:
        Split: `X_test` and `raw_test` are the validation rows.
    """
    dev = np.flatnonzero(ds.dev_mask)
    cut = len(dev) - int(len(dev) * val_frac)
    return prepare(ds, features, Fold(-1, dev[:cut], dev[cut:], dev[cut:]))


def _row(method: str, k: int, source: str, y: pd.Series, pred: np.ndarray) -> dict:
    """One result row: the scores of ``pred`` on the test rows ``y``."""
    return {
        "method": method,
        "fold": k,
        "source": source,
        "n_test": len(y),
        "n_fail": int(y.sum()),
        **scores(y, pred),
        **domain(y, pred),
    }


def run_cv(
    ds: Dataset,
    features: list[str],
    fit_predict: Callable[[Split], np.ndarray | Result],
    method: str,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, dict[int, Any]]:
    """
    Score ``fit_predict(split)`` on every fold. The tree is deterministic, so each
    fold is run once.

    Verbosity follows the package log level: INFO logs each fold transition and
    shows a progress bar over the folds; DEBUG adds the scores of each fold.

    Returns:
        tuple: one row per fold with ``source = "all"`` (method, fold, source, n_test,
            n_fail, every ``scores`` and ``domain`` key), plus, when the dataset has
            several sources, one row per fold and source scored on that source's test
            rows (fail-class scores NaN when the slice has no failure); and the
            payloads keyed by fold.
    """
    fold_list = folds(ds, n_splits)
    per_source = ds.source is not None and ds.source.nunique() > 1
    rows, payloads = [], {}
    bar = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=log,
        disable=not logger.isEnabledFor(logging.INFO),
    )
    with bar:
        task = bar.add_task(method, total=len(fold_list))
        for fold in fold_list:
            split = prepare(ds, features, fold)
            y_test = ds.y.iloc[fold.test]
            n_fail = int(y_test.sum())
            name = f"{method} fold {fold.k + 1}/{len(fold_list)}"
            logger.info(
                f"{name}: fit {len(fold.fit)}, val {len(fold.val)}, "
                f"test {len(fold.test)} ({n_fail} failures)"
            )
            bar.update(task, description=name)
            out = fit_predict(split)
            res = out if isinstance(out, Result) else Result(out)
            row = _row(method, fold.k, "all", y_test, res.pred)
            rows.append(row)
            if per_source:
                src = ds.source.iloc[fold.test].to_numpy()  # pyright: ignore[reportOptionalMemberAccess]
                for s in sorted(set(src)):
                    m = src == s
                    part = _row(method, fold.k, s, y_test[m], np.asarray(res.pred)[m])
                    if part["n_fail"] == 0:
                        part |= dict.fromkeys(FAIL_CLASS_SCORES, np.nan)
                    rows.append(part)
            logger.debug(
                f"{name}: mcc {row['mcc']:.3f}, "
                f"recall {row['recall']:.3f}, precision {row['precision']:.3f}"
            )
            if res.payload is not None:
                payloads[fold.k] = res.payload
            bar.advance(task)
    return pd.DataFrame(rows), payloads
