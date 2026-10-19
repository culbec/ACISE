import pathlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from acise_p1_cigp.utils.data.download_data import RAW_DIR

# T1: keep all builds, and ``last_label` is kept for completeness.
# T2: keeps only builds that fail after the last one succeeded. Prediction is done
# so that we check if the current build fails considering the set of failed builds.
Target = Literal["T1", "T2"]

ORDER_COLS = ["project", "build_nums"]
LABEL_COL = "now_label"
PREV_LABEL_COL = "last_label"
LEAKAGE_COLS = ("build_duration",)  # only known after the build runs

# Assumed by looking at each features, and 1 seems like the build succeeds.
# We assume that most builds succeeds in CI runs, so 1 seems fair.
PASS_LABEL = 1

# ``line_added`` is -1 when a build has a commit missing from the extractor's commit
# dump (RavenBuild ``get_line_added``); it is a missing-value marker, not a count.
MISSING_SENTINELS = {"line_added": -1}

# The fraction of features used in development mode.
DEV_FRACTION = 0.2


@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame  # features only, chronological, RangeIndex
    y: pd.Series  # failed = 1, passed = 0
    dev_mask: np.ndarray  # True on dev rows (first dev_fraction of each project)
    meta: dict
    groups: pd.Series  # project per row


def _features_dir() -> pathlib.Path:
    return next(
        RAW_DIR.rglob("circleci_repos/features"), RAW_DIR / "circleci_repos/features"
    )


def available_projects() -> list[str]:
    return sorted(p.stem for p in _features_dir().glob("*.csv"))


def load_project(name: str | None = None, all: bool = False) -> pd.DataFrame:
    """
    Raw feature frame with a leading ``project`` column, sorted by
    ``ORDER_COLS`` (order columns are only comparable within a project).

    With ``all=True`` ``name`` is ignored and every project is stacked. Columns are
    the union; a column absent from a project is NaN there.
    """
    if not all and not name:
        raise ValueError("name is required unless all=True")

    frames = []
    for p in available_projects() if all else [name]:
        path = _features_dir() / f"{p}.csv"
        if not path.is_file():
            raise ValueError(
                f"unknown project {p!r}; available: {available_projects()}"
            )
        frames.append(pd.read_csv(path).assign(project=p))

    df = pd.concat(frames, ignore_index=True).sort_values(ORDER_COLS, kind="stable")
    return df[["project", *df.columns.drop("project")]].reset_index(drop=True)


def preprocess(
    df: pd.DataFrame, target: Target, dev_fraction: float | None = None
) -> Dataset:
    """
    Stateless preprocessing of a frame from ``load_project`` (one or many projects).

    The T2 filter and the dev cut are applied per project; ``project`` goes to
    ``Dataset.groups``, never into ``X``.
    """
    if target not in ("T1", "T2"):
        raise ValueError(f"target must be 'T1' or 'T2', got {target!r}")
    if dev_fraction is None or not 0.0 <= dev_fraction <= 1.0:
        dev_fraction = DEV_FRACTION

    if target == "T2":  # first failures: previous build passed
        df = df[df[PREV_LABEL_COL] == PASS_LABEL]
    df = df.reset_index(drop=True)

    y = (df[LABEL_COL] != PASS_LABEL).astype(int).rename("failed")
    groups = df["project"]
    drop = [*ORDER_COLS, LABEL_COL, *LEAKAGE_COLS]
    if target == "T2":
        drop.append(PREV_LABEL_COL)  # constant on T2
    X = df.drop(columns=drop)
    for col, sentinel in MISSING_SENTINELS.items():
        X[col] = X[col].replace(sentinel, np.nan)

    # first dev_fraction of each project; rows are already chronological per project
    by_project = df.groupby("project")["project"]
    dev_mask = (by_project.cumcount() < (by_project.transform("size") * dev_fraction).astype(int)).to_numpy()

    meta = {
        "projects": sorted(groups.unique()),
        "target": target,
        "n_rows": len(X),
        "n_failed": int(y.sum()),
    }
    return Dataset(X=X, y=y, dev_mask=dev_mask, meta=meta, groups=groups)
