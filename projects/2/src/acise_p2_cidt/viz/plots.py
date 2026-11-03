import pathlib
from collections.abc import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from acise_p2_cidt.utils.logging.log import get_logger

# projects/2/src/acise_p2_cidt/viz/plots.py -> projects/2/results/figures
FIG_DIR: pathlib.Path = (
    pathlib.Path(__file__).resolve().parents[3] / "results" / "figures"
)

DIST_PLOT_F: dict[str, Callable] = {
    "box": px.box,
    "violin": px.violin,
}
COLOR_SCALE: str = "Spectral"

logger = get_logger(__name__)


def save(fig: go.Figure, name: str, *, png: bool = True) -> go.Figure:
    """Write `<name>.html` (and `<name>.png` via kaleido) to results/figures/."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(FIG_DIR / f"{name}.html")
    if png:
        fig.write_image(FIG_DIR / f"{name}.png", scale=2)
    logger.info(f"Saved figure: {FIG_DIR / name}.*")
    return fig


# =============================
# Exploratory Data Analysis
# ==============================
def label_balance(
    df: pd.DataFrame, label: str = "failed", by: str | None = None
) -> go.Figure:
    """
    Run counts per label, optionally per group (e.g. `by="target"` to put T1 next
    to T2). Shows how imbalanced the label is, which drives the class weights.

    Args:
        df (pd.DataFrame): one row per run.
        label (str): label column, failed = 1.
        by (str | None): grouping column; None gives one bar per label.

    Returns:
        go.Figure: stacked bar plot of run counts.
    """
    keys = [label] if by is None else [by, label]
    counts = df.groupby(keys).size().reset_index(name="runs")  # pyright: ignore[reportCallIssue]
    return px.bar(
        counts, x=by or label, y="runs", color=counts[label].astype(str)
    ).update_layout(legend_title_text=label, barmode="stack")


def streak_hist(streaks: pd.Series | np.ndarray, **kw) -> go.Figure:
    """
    Histogram of consecutive-failure/consecutive-success streak lengths.

    Used (mainly) to identify how failures cluster together (failures follow a failure),
    providing a way to decide if we shall use all failures in a streak or only
    the first one that appears.

    Using the first one indicates a regression, compared to a failure following a failure,
    which can indicate the same regression not being resolved.

    Args:
        streaks (pd.Series): Series describing the consecutive failure / success streak lengths.
        kw (dict): additional parameters for the histogram plot.

    Returns:
        go.Figure: the histogram plot.
    """
    return px.histogram(x=streaks, labels={"x": "streak length"}, **kw)


def dist_by_label(
    df: pd.DataFrame,
    features: list[str],
    label: str = "failed",
    kind: str = "box",
    log_y: bool = False,
) -> go.Figure:
    """
    One distribution plot (box | violin) per feature, split by label.

    The passed `df` is melted with respect to `label`; this produces
    a DF that identifies items by `label`, one facet per feature.

    Args:
        df (pd.DataFrame): DF to plot.
        features (list[str]): list of features, grouped by `label`.
        label (str): label to group features by.
        kind (str): distribution plot type.
        log_y (bool): whether to represent feature values in `log` or not;
            can be used if you notice that features are not evenly distributed.

    Returns:
        go.Figure: the distribution plot.
    """
    long = df.melt(
        id_vars=label, value_vars=features, var_name="feature", value_name="value"
    )
    plot = DIST_PLOT_F[kind]
    return plot(
        long,
        x=label,
        y="value",
        color=label,
        facet_col="feature",
        facet_col_wrap=4,
        log_y=log_y,
    ).update_yaxes(matches=None)


def runs_per_week(
    df: pd.DataFrame, order: str = "created_at", label: str = "failed"
) -> go.Figure:
    """
    Runs per week (bars) and the weekly failure rate (line, right axis).

    Shows that the collection covers the period without gaps, and how much the
    failure rate drifts over time.

    Args:
        df (pd.DataFrame): one row per run.
        order (str): run creation time column.
        label (str): label column, failed = 1.

    Returns:
        go.Figure: bar and line plot over weeks.
    """
    weekly = (
        df.set_index(pd.to_datetime(df[order]))[label]
        .resample("W")
        .agg(["size", "mean"])
    )
    fig = go.Figure()
    fig.add_bar(x=weekly.index, y=weekly["size"], name="runs")
    fig.add_scatter(
        x=weekly.index,
        y=weekly["mean"],
        name="failure rate",
        mode="lines+markers",
        yaxis="y2",
    )
    return fig.update_layout(
        yaxis_title="runs",
        yaxis2={
            "title": "failure rate",
            "overlaying": "y",
            "side": "right",
            "range": (0, 1),
        },
    )


def failed_job_share(df: pd.DataFrame, category: str = "category") -> go.Figure:
    """
    Share of failed jobs per job category (build, lint, unit, integration, other).

    EDA only: job categories are specific to each repository and never features.

    Args:
        df (pd.DataFrame): one row per failed job, with its category.
        category (str): category column.

    Returns:
        go.Figure: bar plot of shares, largest first.
    """
    share = df[category].value_counts(normalize=True).rename("share").reset_index()
    return px.bar(share, x=category, y="share", range_y=(0, 1))


# ==============================
# Feature Selection
# ==============================
def corr_heatmap(corr: pd.DataFrame) -> go.Figure:
    """
    Plots the correlation heatmap, passed as a DataFrame.

    Args:
        corr (pd.DataFrame): correlation heatmap.

    Returns:
        go.Figure: image showing the correlation heatmap.
    """
    return px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale=COLOR_SCALE)


def scores_bar(scores: pd.DataFrame, top: int | None = None) -> go.Figure:
    """
    Feature relevance, grouped by `statistic` column in `scores` DF.

    Used to show how the relevance statistics agree or disagree on a feature, and
    to explain the final mean rank.

    Args:
        scores (pd.DataFrame): DF containing feature relevance scores;
            if top is not None, only the n-top features are taken according to mean
            value of statistic.
        top (int | None): the top n largest features to get.

    Returns:
        go.Figure: barplot showing the top n features (or all) according to presented statistic
            in `scores`.
    """
    s = scores if top is None else scores.loc[scores.mean(axis=1).nlargest(top).index]
    long = s.reset_index(names="feature").melt(
        "feature", var_name="statistic", value_name="score"
    )
    return px.bar(long, x="feature", y="score", color="statistic", barmode="group")


def stability_bar(freq: pd.Series) -> go.Figure:
    """
    Plots the selection frequency over bootstraps; indexed by feature, more frequent features
    are selected in more resamples; `freq` values in [0, 1].

    Called once, on the frequencies aggregated over all bootstrap resamples, to check
    which features are consistently selected, preferring stability.

    Args:
        freq (pd.Series): indexed by feature, describes frequency of each feature in a set.

    Returns:
        go.Figure: barplot showing the feature frequency in a set.
    """
    return px.bar(
        freq.sort_values(ascending=False).rename("frequency").reset_index(),
        x="feature",
        y="frequency",
        range_y=(0, 1),
    )


def drift_heatmap(kept: pd.DataFrame, gap: int = 2) -> go.Figure:
    """
    Feature x fold grid: whether each frozen feature (or a near-duplicate) is still
    selected on that fold's train. Cells are separated by `gap` pixels, so every
    feature row and fold column has a visible border.

    Args:
        kept (pd.DataFrame): boolean feature x fold table (`analysis.features.drift`).
        gap (int): border width in pixels between cells.

    Returns:
        go.Figure: two-color grid, one row per feature, one column per fold.
    """
    fig = px.imshow(
        kept.astype(int),
        zmin=0,
        zmax=1,
        aspect="auto",
        color_continuous_scale=[
            (0, "#d7301f"),
            (0.5, "#d7301f"),
            (0.5, "#1a9850"),
            (1, "#1a9850"),
        ],
    )
    return (
        fig.update_traces(
            xgap=gap,
            ygap=gap,
            text=np.where(kept, "kept", "lost"),
            texttemplate="%{text}",
            hovertemplate="%{y} | %{x}: %{text}<extra></extra>",
        )
        .update_coloraxes(showscale=False)
        .update_layout(
            plot_bgcolor="black",
            height=max(300, 32 * len(kept) + 120),
            xaxis={"side": "top", "showgrid": False},
            yaxis={"showgrid": False},
        )
    )


# ==============================
# Tree plots
# ==============================
def validation_curve(
    df: pd.DataFrame, param: str, metric: str = "mcc", log_x: bool = False
) -> go.Figure:
    """
    `metric` against one hyperparameter (e.g. `max_depth`, `ccp_alpha`), median over
    folds with a min-max band as error bars. A train line far above the val line
    means the tree overfits at that value.

    Args:
        df (pd.DataFrame): one row per fold x value; cols `param`, `split`
            (train / val), `metric`.
        param (str): hyperparameter column.
        metric (str): score column.
        log_x (bool): log x axis, for `ccp_alpha`, whose values span decades.

    Returns:
        go.Figure: line plot, one line per split.
    """
    agg = df.groupby(["split", param])[metric].agg(["median", "min", "max"])
    agg = agg.reset_index()
    return px.line(
        agg,
        x=param,
        y="median",
        color="split",
        error_y=agg["max"] - agg["median"],
        error_y_minus=agg["median"] - agg["min"],
        markers=True,
        log_x=log_x,
        labels={"median": f"{metric} (median over folds)"},
    )


def tree_icicle(tree, feature_names: list[str], max_depth: int = 3) -> go.Figure:
    """
    Icicle of the top levels of a fitted tree: the main explainability figure.

    Each box is a node, labelled with the test that leads to it; its width is the
    number of runs that reach it, its colour P(fail) (the weighted class share,
    as `predict_proba` gives it). Hover shows the node's own split.

    Args:
        tree: fitted `CART.tree_` (flat arrays `feature`, `threshold`, `left`,
            `right`, `value`, `n_samples`; children -1 at leaves).
        feature_names (list[str]): names indexed by `tree.feature`.
        max_depth (int): deepest level drawn; deeper nodes are folded into it.

    Returns:
        go.Figure: icicle plot, root on top.
    """
    ids, labels, parents, runs, p_fail, splits = [], [], [], [], [], []
    stack = [(0, "", "all runs", 0)]  # node, parent id, label, depth
    while stack:
        i, parent, label, depth = stack.pop()
        leaf = tree.left[i] == -1
        name, t = feature_names[tree.feature[i]], tree.threshold[i]
        v = tree.value[i]
        ids.append(str(i))
        labels.append(label)
        parents.append(parent)
        runs.append(int(tree.n_samples[i]))
        p_fail.append(float(v[1] / v.sum()) if v.sum() else 0.0)
        splits.append("leaf" if leaf else f"{name} <= {t:.4g}")
        if not leaf and depth < max_depth:
            stack.append((int(tree.right[i]), str(i), f"{name} > {t:.4g}", depth + 1))
            stack.append((int(tree.left[i]), str(i), f"{name} <= {t:.4g}", depth + 1))
    return go.Figure(
        go.Icicle(
            ids=ids,
            labels=labels,
            parents=parents,
            values=runs,
            branchvalues="total",  # left + right runs = parent runs
            customdata=list(zip(p_fail, splits, strict=True)),
            marker={
                "colors": p_fail,
                "colorscale": "RdYlGn_r",
                "cmin": 0,
                "cmax": 1,
                "showscale": True,
                "colorbar": {"title": "P(fail)"},
            },
            hovertemplate="%{label}<br>runs: %{value}<br>P(fail): "
            "%{customdata[0]:.2f}<br>split: %{customdata[1]}<extra></extra>",
            tiling={"orientation": "v"},
        )
    )


# ==============================
# Results
# ==============================
def metrics_box(df: pd.DataFrame) -> go.Figure:
    """
    Box plot per method, used as a main comparison between methods.

    Args:
        df (pd.DataFrame): long format DF with cols `method`, `metric`, `value`.
    Returns:
        go.Figure: box plot, with one facet per metric, every per-fold point overlaid;
            the dashed line is the mean and the diamond spans mean +- 1 std.
    """
    return px.box(
        df, x="method", y="value", color="method", facet_col="metric", points="all"
    ).update_traces(boxmean="sd")


def retest_vs_cv(
    cv: pd.DataFrame,
    retest: pd.DataFrame,
    metrics: tuple[str, ...] = ("mcc", "f1", "precision", "recall"),
) -> go.Figure:
    """
    CV scores (one point per fold) next to the re-test on runs after `C` (one point
    per method), per method and metric. A re-test point inside the CV box means the
    frozen model generalises to unseen future runs.

    Args:
        cv (pd.DataFrame): `run_cv` results.
        retest (pd.DataFrame): re-test results, same columns.
        metrics (tuple[str, ...]): metric columns to show.

    Returns:
        go.Figure: box plot, one facet per metric, coloured by source.
    """
    long = pd.concat(
        [cv.assign(source="cv"), retest.assign(source="retest")], ignore_index=True
    ).melt(id_vars=["method", "source"], value_vars=list(metrics), var_name="metric")
    return px.box(
        long, x="method", y="value", color="source", facet_col="metric", points="all"
    )


def tradeoff(
    df: pd.DataFrame,
    point: tuple[float, float] | None = None,
    point_name: str = "chosen",
) -> go.Figure:
    """
    Line plot of `% failures caught` vs. `% builds saved`.

    One marker per decision threshold on P(fail): a higher threshold saves more
    builds and catches fewer failures. `point` places another method (e.g. Parrot)
    on the same axes.

    Args:
        df (pd.DataFrame): DF `saved, caught, threshold`.
        point (tuple[float, float] | None): as `(saved, caught)`.
        point_name (str): legend name of `point`.

    Returns:
        go.Figure: line plot to show trade-off between failures caught and builds saved.
    """
    fig = px.line(df, x="saved", y="caught", hover_data=["threshold"], markers=True)
    if point is not None:
        fig.add_scatter(
            x=[point[0]],
            y=[point[1]],
            mode="markers",
            name=point_name,
            marker={"symbol": "star", "size": 14},
        )
    return fig.update_layout(
        xaxis_title="% builds saved", yaxis_title="% failures caught"
    )


def confusion(cm: np.ndarray, labels: tuple[str, str] = ("pass", "fail")) -> go.Figure:
    """
    Confusion matrix for showing builds failed/passed per method.

    Args:
        cm (np.ndarray): 2x2 counts `[[tn, fp], [fn, tp]]`, rows actual and columns
            predicted, in the order of `labels` (sklearn `confusion_matrix` order).
        labels (tuple[str, str]): labels for classes.

    Returns:
        go.Figure: image showing the confusion matrix.
    """
    return px.imshow(
        cm,
        x=list(labels),
        y=list(labels),
        text_auto=True,
        labels={"x": "predicted", "y": "actual"},
    )
