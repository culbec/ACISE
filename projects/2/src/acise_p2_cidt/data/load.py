import json
import pathlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import numpy as np
import pandas as pd

# T1: all runs. T2: runs whose previous completed run of the same PR or branch
# passed, or that have none (first failures).
Target = Literal["T1", "T2"]

# Fixed by data/build.py for every repository, so they are constants, not schema.
ORDER_COL: str = "created_at"
LABEL_COL: str = "failed"
PREV_LABEL_COL: str = "prev_failed"  # 1 = the lineage's previous completed run failed

# Share of the earliest rows used for EDA and feature selection only.
DEV_FRACTION: float = 0.2

# projects/2/src/acise_p2_cidt/data/load.py -> projects/2/data/processed
PROCESSED_DIR = pathlib.Path(__file__).resolve().parents[3] / "data" / "processed"
# projects/2/src/acise_p2_cidt/data/load.py -> projects/2/results/collection
SCHEMA_DIR = pathlib.Path(__file__).resolve().parents[3] / "results" / "collection"


@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame  # feature columns only, chronological, RangeIndex
    y: pd.Series  # failed = 1, passed = 0
    dev_mask: np.ndarray  # True on the first DEV_FRACTION of rows
    meta: dict  # schema.json content + dataset, target, n_rows, n_failed, n_dev
    order: pd.Series  # created_at per row, for plots and the retest cut


def table_path(slug: str) -> pathlib.Path:
    """`data/processed/<slug>.csv`, written by `build-dataset`."""
    return PROCESSED_DIR / f"{slug}.csv"


def schema_path(slug: str) -> pathlib.Path:
    """`results/collection/schema_<slug>.json`, written by `build-dataset`."""
    return SCHEMA_DIR / f"schema_{slug}.json"


def in_t2(df: pd.DataFrame) -> pd.Series:
    """
    T2 membership: the previous run passed, or there is none. A PR's first run
    continues no known failure, so it stays.
    """
    return df[PREV_LABEL_COL] != 1


def available_targets() -> list[str]:
    """
    Targets with a built table.

    Returns:
        list[str]: sorted target slugs.
    """
    return sorted(p.stem for p in PROCESSED_DIR.glob("*.csv"))


def load(slug: str, target: Target, until: datetime | None = None) -> Dataset:
    """
    Loads a built table as a `Dataset`. Stateless: nothing is fitted here.

    Args:
        slug (str): target slug, e.g. "n8n-io__n8n__33323319@master".
        target (Target): T1 = all runs; T2 = `in_t2` runs.
        until (datetime | None): keep runs created at or before it (naive = UTC);
            None keeps all.

    Returns:
        Dataset: chronological features, label, dev mask, schema and order.
    """
    if target not in ("T1", "T2"):
        raise ValueError(f"target must be 'T1' or 'T2', got {target!r}")
    path, sp = table_path(slug), schema_path(slug)
    if not path.is_file() or not sp.is_file():
        raise FileNotFoundError(
            f"no built table for {slug!r} ({path}); run `build-dataset --target {slug}`"
        )
    schema = json.loads(sp.read_text())
    df = pd.read_csv(path)
    df[ORDER_COL] = pd.to_datetime(df[ORDER_COL], utc=True, format="ISO8601")
    df = df.sort_values(ORDER_COL, kind="stable")

    if until is not None:
        cut = pd.Timestamp(until)
        df = df[df[ORDER_COL] <= (cut if cut.tzinfo else cut.tz_localize(UTC))]
    if target == "T2":
        df = df[in_t2(df)]
    df = df.reset_index(drop=True)

    features = list(schema["feature_cols"])
    if target == "T2":
        features.remove(PREV_LABEL_COL)  # 0 or NaN on T2, as `lineage_first_run`
    n_dev = int(len(df) * DEV_FRACTION)  # cut after the T2 filter, as in project 1
    y = df[LABEL_COL].astype(int).rename("failed")
    meta = schema | {
        "dataset": slug,  # file key of artefacts derived from this data
        "target": target,
        "n_rows": len(df),
        "n_failed": int(y.sum()),
        "n_dev": n_dev,
    }
    return Dataset(
        X=df[features].astype(float),
        y=y,
        dev_mask=np.arange(len(df)) < n_dev,
        meta=meta,
        order=df[ORDER_COL],
    )
