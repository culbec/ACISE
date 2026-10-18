import pathlib
from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from acise_p1_cigp.utils.logging.log import log

# projects/1/src/acise_p1_cigp/viz/plots.py -> projects/1/results/figures
FIG_DIR: pathlib.Path = (
    pathlib.Path(__file__).resolve().parents[3] / "results" / "figures"
)

DIST_PLOT_F: dict[str, Callable] = {
    "box": px.box,
    "violin": px.violin,
}
COLOR_SCALE: str = "Spectral"


def save(fig: go.Figure, name: str, *, png: bool = True) -> go.Figure:
    """Write `<name>.html` (and `<name>.png` via kaleido) to results/figures/."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(FIG_DIR / f"{name}.html")
    if png:
        fig.write_image(FIG_DIR / f"{name}.png", scale=2)
    log.print(f"[green]Saved figure:[/green] {FIG_DIR / name}.*")
    return fig


# =============================
# Exploratory Data Analysis
# ==============================
def label_balance(
    df: pd.DataFrame, project: str = "project", label: str = "label"
) -> go.Figure:
    """
    Build counts per label, per project. Mainly used for
    identifying usable projects regarding the build status.

    Projects that mainly fail or mainly succeed are not that useful
    due to them not being various enough in CI run statuses.

    Args:
        df (pd.DataFrame): DF containing historical CI data.
        project (str): column determining the analyzed project.
        label (str): column determining the status of a CI run (e.g., failed | success).

    Returns:
        go.Figure: stacked bar plot describing the CI runs characterized by `label`.
    """
    counts = df.groupby([project, label]).size().reset_index(name="builds")  # pyright: ignore[reportCallIssue]
    return px.bar(counts, x=project, y="builds", color=label, barmode="stack")


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
    label: str = "label",
    kind: str = "box",
    log_y: bool = False,
) -> go.Figure:
    """
    One distribution plot (box | violin) per feature, split by label.

    The passed `df` is melted with respect to `label`; this produces
    a DF that identifies items by `label`, with features being grouped
    from multiple projects together.

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


# ==============================
# Genetic Programming plots
# ==============================
def convergence(stats: pd.DataFrame) -> go.Figure:
    """
    Plots the best and mean fitness per generation with a +-std band over seeds.

    Used to decide the generations and population:
        - best: if the line is flat for the last half => we can prune the population.
        - mean: if far below the best line => population diverse or stuck on bad individuals.

    Args:
        stats (pd.DataFrame): DF describing the generations;
            gen, seed, best, mean.
    """
    agg = stats.groupby("gen")[["best", "mean"]].agg(["mean", "std"]).fillna(0)
    fig = go.Figure()

    col_rgb = (("best", "31,119,180"), ("mean", "255,127,14"))

    for col, rgb in col_rgb:
        m, s = agg[(col, "mean")], agg[(col, "std")]

        fig.add_scatter(
            x=m.index, y=m + s, line_width=0, showlegend=False, hoverinfo="skip"
        )
        fig.add_scatter(
            x=m.index,
            y=m - s,
            line_width=0,
            fill="tonexty",
            showlegend=False,
            fillcolor=f"rgba({rgb},0.2)",
            hoverinfo="skip",
        )
        fig.add_scatter(x=m.index, y=m, name=col, line_color=f"rgb({rgb})")
    return fig.update_layout(xaxis_title="generation", yaxis_title="fitness")


def pareto(front: pd.DataFrame, chosen: pd.Series | None = None) -> go.Figure:
    """
    TPR against FPR for the non-dominated individuals, one per trade-off between
    the two contrasting fitness measures.

    Points towards top left (high TPR, low FPR) are better. Optional star
    shows the point selected on the validation split.

    Args:
        front (pd.DataFrame): TPR and FPR on the Pareto front.
        chosen (pd.Series | None): chosen individual after validation split.

    Returns:
        go.Figure: scatter plot of the Pareto front.
    """
    fig = px.scatter(front, x="fpr", y="tpr")
    if chosen is not None:
        fig.add_scatter(
            x=[chosen["fpr"]],
            y=[chosen["tpr"]],
            mode="markers",
            name="chosen",
            marker=dict(symbol="star", size=14),
        )
    return fig.update_xaxes(range=(0, 1)).update_yaxes(range=(0, 1))


def size_vs_fitness(df: pd.DataFrame) -> go.Figure:
    """
    Used to assess if the size of the program (number of statements, not
    tree depth) actually improves solutions or not.

    Penalty `lambda * statements / N` is applied to check whether we improve
    with bigger programs or not and where the gain stops.

    Args:
        df (pd.DataFrame): `statements` and `fitness` columns.

    Returns:
        go.Figure: scatter plot.
    """
    return px.scatter(df, x="statements", y="fitness")


def feature_freq_performers(freq: pd.Series) -> go.Figure:
    """
    Feature frequency across best performer programs.

    Args:
        freq (pd.Series): Series indexed by feature with counts.

    Returns:
        go.Figure: bar plot that shows feature frequency across best performers
            resulted from GP.
    """
    return px.bar(
        freq.sort_values(ascending=False).rename("count").reset_index(),
        x="feature",
        y="count",
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
        go.Figure: box plot, with one facet per metric, every per-fold point overlaid.
    """
    return px.box(
        df, x="method", y="value", color="method", facet_col="metric", points="all"
    )


def tradeoff(df: pd.DataFrame, point: tuple[float, float] | None = None) -> go.Figure:
    """
    Line plot of `% failures caught` vs. `% builds saved`.

    Moving right along the plot usually moves down across failures caught.

    Markers at each `alpha`, using `point` to compare methods to see if they
    are worse/better than the method plotted.

    Args:
        df (pd.DataFrame): DF `saved, caught, alpha`.
        point (tuple[float, float] | None): as `(saved, caught)`.

    Returns:
        go.Figure: line plot to show trade-off between failures caught and builds saved.
    """
    fig = px.line(df, x="saved", y="caught", hover_data=["alpha"], markers=True)
    if point is not None:
        fig.add_scatter(
            x=[point[0]],
            y=[point[1]],
            mode="markers",
            name="(b)",
            marker=dict(symbol="star", size=14),
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
        labels=dict(x="predicted", y="actual"),
    )
