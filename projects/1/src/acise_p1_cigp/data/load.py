import pathlib
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from acise_p1_cigp.utils.data.download_data import RAW_DIR

# T1: keep all builds, and ``last_label` is kept for completeness.
# T2: keeps only builds that fail after the last one succeeded. Prediction is done
# so that we check if the current build fails considering the set of failed builds.
Target = Literal["T1", "T2"]

ORDER_COL = "build_nums"
LABEL_COL = "now_label"
PREV_LABEL_COL = "last_label"
LEAKAGE_COLS = ("build_duration",)  # only known after the build runs

# Assumed by looking at each features, and 1 seems like the build succeeds.
# We assume that most builds succeeds in CI runs, so 1 seems fair.
PASS_LABEL = 1

# The fraction of features used in development mode.
DEV_FRACTION = 0.2


@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame  # features only, chronological, RangeIndex
    y: pd.Series  # failed = 1, passed = 0
    dev_end: int  # rows [0, dev_end) form the dev segment
    meta: dict


def _features_dir() -> pathlib.Path:
    return next(
        RAW_DIR.rglob("circleci_repos/features"), RAW_DIR / "circleci_repos/features"
    )


def available_projects() -> list[str]:
    return sorted(p.stem for p in _features_dir().glob("*.csv"))


def load_project(name: str) -> pd.DataFrame:
    """Raw feature frame for one project, sorted by ``build_nums``."""
    path = _features_dir() / f"{name}.csv"
    if not path.is_file():
        raise ValueError(f"unknown project {name!r}; available: {available_projects()}")
    return (
        pd.read_csv(path).sort_values(ORDER_COL, kind="stable").reset_index(drop=True)
    )


def preprocess(df: pd.DataFrame, target: Target, name: str = "") -> Dataset:
    """Stateless preprocessing of a (sorted) frame from ``load_project``."""
    if target not in ("T1", "T2"):
        raise ValueError(f"target must be 'T1' or 'T2', got {target!r}")

    if target == "T2":  # first failures: previous build passed
        df = df[df[PREV_LABEL_COL] == PASS_LABEL]
    df = df.reset_index(drop=True)

    y = (df[LABEL_COL] != PASS_LABEL).astype(int).rename("failed")
    drop = [ORDER_COL, LABEL_COL, *LEAKAGE_COLS]
    if target == "T2":
        drop.append(PREV_LABEL_COL)  # constant on T2
    X = df.drop(columns=drop)

    meta = {
        "project": name,
        "target": target,
        "n_rows": len(X),
        "n_failed": int(y.sum()),
    }
    return Dataset(X=X, y=y, dev_end=int(len(X) * DEV_FRACTION), meta=meta)
