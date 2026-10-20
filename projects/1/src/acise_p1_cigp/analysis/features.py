import json
import pathlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    VarianceThreshold,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler

from acise_p1_cigp.data.load import Dataset

# projects/1/src/acise_p1_cigp/analysis/features.py -> projects/1/results/features
OUT_DIR = pathlib.Path(__file__).resolve().parents[3] / "results" / "features"

# Unordered integer codes: a GP `lt`/`ge` threshold on them has no meaning.
NOMINAL = ("change_type", "action_changed", "day_of_week", "time_of_day")


@dataclass
class Selection:
    features: list[str]  # frozen list, ordered by relevance; N = len(features)
    scores: pd.DataFrame  # feature x statistic, plus `score` (mean percentile rank)
    stability: pd.Series  # selection frequency over block subsamples, per family
    importance: pd.Series  # random-forest permutation importance (embedded check)
    disagreements: list[str]  # embedded top-k features the filters dropped
    meta: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        """
        Statement budget N, i.e. the number of selected features.

        Returns:
            int: length of `features`.
        """
        return len(self.features)

    def save(self, out_dir: pathlib.Path = OUT_DIR) -> pathlib.Path:
        """
        Writes the selected features, according to target class, per project,
        in the directory specified by `out_dir` as a JSON file.

        Args:
            out_dir (pathlib.Path): out directory.

        Returns:
            pathlib.Path: JSON file path.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        path = (
            out_dir
            / f"selected_features_{self.meta['target']}_{self.meta['project']}.json"
        )
        path.write_text(
            json.dumps(
                {
                    "N": self.size,
                    "features": self.features,
                    "disagreements": self.disagreements,
                    "meta": self.meta,
                    # to_json maps NaN to null; it returns None only when given a path
                    "scores": json.loads(self.scores.to_json(orient="index") or "{}"),
                    "stability": self.stability.to_dict(),
                    "importance": self.importance.to_dict(),
                },
                indent=2,
            )
        )
        return path


def load_features(target: str, project: str, out_dir: pathlib.Path = OUT_DIR) -> list[str]:
    """
    Reads the frozen feature list written by `Selection.save`.

    Args:
        target (str): "T1" or "T2".
        project (str): project name, or "pooled".
        out_dir (pathlib.Path): directory holding the JSON files.

    Returns:
        list[str]: selected features, ordered by relevance; N = its length.
    """
    path = out_dir / f"selected_features_{target}_{project}.json"
    return json.loads(path.read_text())["features"]


def _binary(X: pd.DataFrame) -> list[str]:
    """
    Binary columns, decided by value set ({0, 1}) and not by `nunique == 2`, since
    a count column can take only two values on a small slice.

    Args:
        X (pd.DataFrame): feature frame.

    Returns:
        list[str]: names of the binary columns.
    """
    return [str(c) for c in X.columns if bool(X[c].dropna().isin([0, 1]).all())]


def clean(
    X: pd.DataFrame,
    max_nan: float = 0.3,
    var_thr: float = 1e-3,
    min_minority: float = 0.01,
) -> pd.DataFrame:
    """
    Drops nominal codes, mostly-missing, near-constant and rare-binary columns,
    then median-imputes the rest.

    Args:
        X (pd.DataFrame): raw feature frame (dev rows).
        max_nan (float): maximum missing-value ratio a column may have.
        var_thr (float): `VarianceThreshold` on min-max scaled data.
        min_minority (float): minimum share of the minority value in a binary column.

    Returns:
        pd.DataFrame: cleaned, imputed frame.
    """
    X = X.drop(columns=[c for c in NOMINAL if c in X])
    X = X.loc[:, X.isna().mean() <= max_nan]
    X = X.fillna(X.median())
    keep = VarianceThreshold(var_thr).fit(MinMaxScaler().fit_transform(X)).get_support()
    X = X.loc[:, keep]
    # a 1%-minority binary still has variance ~0.01, so the threshold above misses it
    rare = [c for c in _binary(X) if min(X[c].mean(), 1 - X[c].mean()) < min_minority]
    return X.drop(columns=rare)


def relevance(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> pd.DataFrame:
    """
    Per-statistic relevance of each feature to the label.

    Numeric: ANOVA F, |Kendall tau|. Binary: chi-square. All: mutual information
    (on binary columns this is the information gain). `score` is the mean percentile
    rank over the statistics that apply, so numeric and binary features are comparable.

    Args:
        X (pd.DataFrame): cleaned feature frame.
        y (pd.Series): label, failed = 1.
        seed (int): random state for mutual information.

    Returns:
        pd.DataFrame: feature x statistic, plus `score`, sorted by `score` descending;
            NaN where a statistic does not apply.
    """
    binary = _binary(X)
    numeric = [c for c in X if c not in binary]
    s = pd.DataFrame(
        index=X.columns, columns=["anova_f", "kendall_tau", "chi2"], dtype=float
    )
    if numeric:
        s.loc[numeric, "anova_f"] = f_classif(X[numeric], y)[0]
        s.loc[numeric, "kendall_tau"] = [
            abs(kendalltau(X[c], y).statistic)  # pyright: ignore[reportAttributeAccessIssue]
            for c in numeric
        ]
    if binary:
        s.loc[binary, "chi2"] = chi2(X[binary], y)[0]
    s["mutual_info"] = mutual_info_classif(
        X,
        y,
        discrete_features=X.columns.isin(binary),  # pyright: ignore[reportArgumentType]
        random_state=seed,
    )
    s["score"] = s.rank(pct=True).mean(axis=1)  # NaN (not applicable) is skipped
    return s.sort_values("score", ascending=False)


def prune_redundant(X: pd.DataFrame, ranking: list[str], rho: float = 0.8) -> list[str]:
    """
    mRMR-style redundancy pruning: walks `ranking` and keeps a feature only if its
    |Spearman rho| with every kept feature is at most `rho`.

    Args:
        X (pd.DataFrame): cleaned feature frame.
        ranking (list[str]): features, most relevant first.
        rho (float): maximum |Spearman rho| allowed between kept features.

    Returns:
        list[str]: kept features, in ranking order.
    """
    corr = X.loc[:, ranking].corr(method="spearman").abs().fillna(0)
    kept: list[str] = []
    for f in ranking:
        if (corr.loc[f, kept] <= rho).all():
            kept.append(f)
    return kept


def _select(
    X: pd.DataFrame, y: pd.Series, rho: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Runs `clean`, `relevance` and `prune_redundant` in sequence.

    Args:
        X (pd.DataFrame): raw feature frame.
        y (pd.Series): label, failed = 1.
        rho (float): pruning threshold.
        seed (int): random state for mutual information.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, list[str]]: cleaned frame, relevance scores,
            full pruned ranking (no `n_max` cut).
    """
    Xc = clean(X)
    scores = relevance(Xc, y, seed)
    return Xc, scores, prune_redundant(Xc, scores.index.tolist(), rho)


def _family_rep(Xc: pd.DataFrame, kept: list[str], rho: float) -> dict[str, str]:
    """
    Maps each column to its family representative: the kept feature it
    near-duplicates (|Spearman| > `rho`), or itself.

    Args:
        Xc (pd.DataFrame): cleaned feature frame.
        kept (list[str]): pruned features (the representatives).
        rho (float): near-duplicate threshold.

    Returns:
        dict[str, str]: column -> representative.
    """
    corr = Xc.corr(method="spearman").abs().fillna(0).loc[:, kept]
    best, top = corr.idxmax(axis=1), corr.max(axis=1)
    return {str(f): str(best[f]) if top[f] > rho else str(f) for f in Xc.columns}


def embedded_check(
    X: pd.DataFrame,
    y: pd.Series,
    selected: list[str],
    top: int = 5,
    val_frac: float = 0.3,
    seed: int = 0,
) -> tuple[pd.Series, list[str]]:
    """
    Hybrid sanity check against an embedded method: random-forest permutation
    importance, fit on the first part of `X` and scored (ROC AUC) on its
    chronological tail. Does not change the selection.

    Args:
        X (pd.DataFrame): cleaned feature frame (dev rows).
        y (pd.Series): label, failed = 1.
        selected (list[str]): features chosen by the filters.
        top (int): how many of the forest's top features to compare.
        val_frac (float): share of the chronological tail used for scoring.
        seed (int): random state for the forest and the permutations.

    Returns:
        tuple[pd.Series, list[str]]: importances sorted descending, and the forest's
            top-`top` features missing from `selected`.
    """
    cut = int(len(X) * (1 - val_frac))
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
    ).fit(X.iloc[:cut], y.iloc[:cut])
    imp = permutation_importance(
        rf,
        X.iloc[cut:],
        y.iloc[cut:],
        scoring="roc_auc",
        n_repeats=10,
        random_state=seed,
    )
    importance = pd.Series(imp["importances_mean"], index=X.columns).sort_values(
        ascending=False
    )
    return importance, [str(f) for f in importance.head(top).index if f not in selected]


def stability(
    X: pd.DataFrame,
    y: pd.Series,
    rep: dict[str, str] | None = None,
    n_boot: int = 20,
    n_max: int = 20,
    rho: float = 0.8,
    block: int | None = None,
    frac: float = 0.8,
    seed: int = 0,
) -> pd.Series:
    """
    How often each feature is selected (top `n_max` after pruning) over `n_boot`
    block subsamples.

    `rep` credits a pick to its family representative (see `_family_rep`): near-duplicates
    swap the top slot between resamples, which would otherwise split the family's
    frequency and drop it under any threshold.

    Each resample keeps `frac` of the contiguous blocks, drawn without replacement and left in time order.
    - blocks keep the dependence between neighbouring builds;
    - no replacement, because duplicated rows inflate the k-NN mutual-information
      estimate for continuous columns, which then win every resample.

    Args:
        X (pd.DataFrame): raw feature frame (dev rows), chronological.
        y (pd.Series): label, failed = 1.
        rep (dict[str, str] | None): column -> family representative; None counts
            each column on its own.
        n_boot (int): number of subsamples.
        n_max (int): top-k after pruning that counts as selected.
        rho (float): pruning threshold.
        block (int | None): block length; None uses n^(1/3).
        frac (float): share of blocks kept per subsample.
        seed (int): random state.

    Returns:
        pd.Series: selection frequency in [0, 1], indexed by feature, descending.
    """
    rep = rep or {}
    rng = np.random.default_rng(seed)

    block = block or max(1, round(len(X) ** (1 / 3)))
    blocks = np.array_split(np.arange(len(X)), -(-len(X) // block))

    counts = pd.Series(0.0, index=X.columns)
    for _ in range(n_boot):
        pick = np.sort(
            rng.choice(len(blocks), round(frac * len(blocks)), replace=False)
        )
        idx = np.concatenate([blocks[i] for i in pick])
        *_, feats = _select(
            X.iloc[idx].reset_index(drop=True),
            y.iloc[idx].reset_index(drop=True),
            rho,
            seed,
        )
        counts[list({rep.get(f, f) for f in feats[:n_max]})] += 1
    return (counts / n_boot).rename_axis("feature").sort_values(ascending=False)


def select_features(
    ds: Dataset,
    n_max: int = 20,
    min_freq: float = 0.6,
    n_boot: int = 20,
    rho: float = 0.8,
    seed: int = 0,
) -> Selection:
    """
    Full selection on `ds`'s dev rows only. Imputation and scaling here are local;
    `training/cv.py` re-fits its own per fold.

    Final list: pruned features selected in at least `min_freq` of the block
    subsamples, in relevance order, capped at `n_max`. N = its length.

    Args:
        ds (Dataset): preprocessed dataset; only `ds.dev_mask` rows are used.
        n_max (int): cap on N, and top-k per subsample.
        min_freq (float): minimum selection frequency over subsamples.
        n_boot (int): number of subsamples.
        rho (float): pruning threshold.
        seed (int): random state.

    Returns:
        Selection: frozen feature list with scores, stability, embedded check and meta.
    """
    X = ds.X.loc[ds.dev_mask].reset_index(drop=True)
    y = ds.y.loc[ds.dev_mask].reset_index(drop=True)

    Xc, scores, kept = _select(X, y, rho, seed)
    freq = stability(X, y, _family_rep(Xc, kept, rho), n_boot, n_max, rho, seed=seed)

    # kept is a subset of X's columns, so every kept feature has a frequency
    features = [f for f in kept if freq[f] >= min_freq][:n_max]

    importance, disagreements = embedded_check(Xc, y, features, seed=seed)
    projects = ds.meta["projects"]

    return Selection(
        features=features,
        scores=scores,
        stability=freq,
        importance=importance,
        disagreements=disagreements,
        meta={
            "target": ds.meta["target"],
            "project": projects[0] if len(projects) == 1 else "pooled",
            "projects": projects,
            "dev_rows": len(X),
            "dev_failed": int(y.sum()),
            "dropped_in_clean": sorted(set(X.columns) - set(Xc.columns)),
            "params": {
                "n_max": n_max,
                "min_freq": min_freq,
                "n_boot": n_boot,
                "rho": rho,
                "seed": seed,
            },
        },
    )


def drift(
    ds: Dataset,
    trains: list[np.ndarray],
    selected: list[str],
    n_max: int = 20,
    rho: float = 0.8,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Diagnostic only; the frozen list never changes. Re-runs selection on each fold's
    train rows to show how the feature choice drifts over time.

    Args:
        ds (Dataset): preprocessed dataset.
        trains (list[np.ndarray]): per fold, train row positions in `ds.X`.
        selected (list[str]): frozen feature list.
        n_max (int): top-k after pruning that counts as selected.
        rho (float): pruning and near-duplicate threshold.
        seed (int): random state.

    Returns:
        pd.DataFrame: feature x fold, True when the frozen feature's family (itself,
            or a column with |Spearman| > `rho` on that train) is in the fold's top `n_max`.
    """
    out = {}
    for k, idx in enumerate(trains):
        X = ds.X.iloc[idx].reset_index(drop=True)
        Xc, _, kept = _select(X, ds.y.iloc[idx].reset_index(drop=True), rho, seed)
        top = kept[:n_max]
        corr = Xc.corr(method="spearman").abs()
        out[f"fold{k}"] = {
            f: f in top or (f in Xc and bool((corr.loc[f, top] > rho).any()))
            for f in selected
        }
    return pd.DataFrame(out)
