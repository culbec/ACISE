from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

# run_cv columns that are not metrics
ID_COLS = ["method", "fold", "source", "n_test", "n_fail"]


class WilcoxonResult(NamedTuple):
    statistic: float
    p: float
    n: int  # pairs with a non-zero difference
    min_p: float  # smallest two-sided p reachable with n pairs
    flag: str | None  # "all_ties" or "zero_diffs_dropped"


def summary(results: pd.DataFrame) -> pd.DataFrame:
    """
    Mean and std per method and metric over the folds, and per source when the
    results have a ``source`` column.

    Args:
        results (pd.DataFrame): concatenated ``run_cv`` frames.

    Returns:
        pd.DataFrame: columns method, (source,) metric, mean, std.
    """
    ids = [c for c in ID_COLS if c in results]
    keys = (
        ["method", "source", "metric"] if "source" in results else ["method", "metric"]
    )
    long = results.melt(id_vars=ids, var_name="metric")
    return long.groupby(keys)["value"].agg(["mean", "std"]).reset_index()


def per_fold(results: pd.DataFrame, metric: str, source: str = "all") -> pd.DataFrame:
    """
    Fold x method table of ``metric`` on one source (default: the pooled rows): the
    paired input of the tests.
    """
    if "source" in results:
        results = results[results["source"] == source]
    return results.pivot(index="fold", columns="method", values=metric)


def paired_wilcoxon(a, b) -> WilcoxonResult:
    """
    Two-sided Wilcoxon signed-rank test on paired per-fold scores. Zero differences
    are dropped (scipy's default), so ``n`` can be below the number of folds.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = int(np.count_nonzero(d))
    if n == 0:  # scipy raises on all-zero differences
        return WilcoxonResult(0.0, 1.0, 0, 1.0, "all_ties")
    stat, p = wilcoxon(d)  # scipy result fields are untyped: unpack, then .item()
    flag = "zero_diffs_dropped" if n < d.size else None
    return WilcoxonResult(
        np.asarray(stat).item(), np.asarray(p).item(), n, 2.0 ** (1 - n), flag
    )


def vargha_delaney_a12(a, b) -> float:
    """
    P(a > b) + 0.5 P(a = b) for a random draw from each sample; 0.5 means no
    difference. Computed as U / (m n) from the Mann-Whitney U of ``a``.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    u, _ = mannwhitneyu(a, b)
    return np.asarray(u).item() / (a.size * b.size)
