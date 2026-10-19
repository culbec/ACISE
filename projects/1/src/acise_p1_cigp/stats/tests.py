from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

# run_cv columns that are not metrics
ID_COLS = ["method", "fold", "seed", "n_test", "n_fail"]


class WilcoxonResult(NamedTuple):
    statistic: float
    p: float
    n: int  # pairs with a non-zero difference
    min_p: float  # smallest two-sided p reachable with n pairs
    flag: str | None  # "all_ties" or "zero_diffs_dropped"


def summary(results: pd.DataFrame) -> pd.DataFrame:
    """
    Mean and std per method and metric over all runs (folds x seeds).

    Args:
        results (pd.DataFrame): concatenated ``run_cv`` frames.

    Returns:
        pd.DataFrame: columns method, metric, mean, std.
    """
    ids = [c for c in ID_COLS if c in results]
    long = results.melt(id_vars=ids, var_name="metric")
    return (
        long.groupby(["method", "metric"])["value"].agg(["mean", "std"]).reset_index()
    )


def per_fold(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Fold x method table of ``metric``, median over seeds: the paired input of the tests.
    """
    return results.pivot_table(
        index="fold", columns="method", values=metric, aggfunc="median"
    )


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


def position_per_fold(
    results: pd.DataFrame, metric: str, point: str, dist: str
) -> pd.Series:
    """
    Per fold, where method ``point`` (e.g. program (b), one score) falls within the
    seeds of method ``dist`` (e.g. GP): A12 of ``point`` against them. 1.0 means
    ``point`` beats every seed, 0.0 that every seed beats it.
    """
    return pd.Series(
        {
            k: vargha_delaney_a12(
                g.loc[g["method"] == point, metric], g.loc[g["method"] == dist, metric]
            )
            for k, g in results.groupby("fold")
        },
        name=f"A12({point} vs {dist})",
    )
