import hashlib
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
    source: pd.Series | None = None  # target slug per row


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


def pool_name(slugs: list[str]) -> str:
    """
    File key of a pool. It depends only on the set of slugs, so a changed target
    list gets new artefacts instead of overwriting the old ones.
    """
    return "pool-" + hashlib.sha256("\n".join(sorted(slugs)).encode()).hexdigest()[:8]


def available_targets() -> list[str]:
    """
    Targets with a built table.

    Returns:
        list[str]: sorted target slugs.
    """
    return sorted(p.stem for p in PROCESSED_DIR.glob("*.csv"))


def _read(slug: str) -> tuple[pd.DataFrame, dict]:
    """One built table, with `created_at` parsed and a `source` column, and its schema."""
    path, sp = table_path(slug), schema_path(slug)
    if not path.is_file() or not sp.is_file():
        raise FileNotFoundError(
            f"no built table for {slug!r} ({path}); run `build-dataset --target {slug}`"
        )
    df = pd.read_csv(path)
    df[ORDER_COL] = pd.to_datetime(df[ORDER_COL], utc=True, format="ISO8601")
    return df.assign(source=slug), json.loads(sp.read_text())


def load(
    targets: str | list[str], target: Target, until: datetime | None = None
) -> Dataset:
    """
    Loads one built table, or several pooled into one chronological table, as a
    `Dataset`. Stateless: nothing is fitted here.

    Args:
        targets (str | list[str]): a target slug, e.g. "n8n-io__n8n__33323319@master",
            or a list of slugs to pool.
        target (Target): T1 = all runs; T2 = `in_t2` runs.
        until (datetime | None): keep runs created at or before it (naive = UTC);
            None keeps all.

    Returns:
        Dataset: chronological features, label, dev mask, schema, order and source.
            A pool's `meta` keeps only the schema keys equal in every source.

    Raises:
        ValueError: the sources' feature columns differ.
    """
    if target not in ("T1", "T2"):
        raise ValueError(f"target must be 'T1' or 'T2', got {target!r}")
    slugs = [targets] if isinstance(targets, str) else list(targets)
    if not slugs:
        raise ValueError("no target slug given")
    parts = [_read(s) for s in slugs]
    schemas = [sc for _, sc in parts]
    for s, sc in zip(slugs, schemas):
        if sc["feature_cols"] != schemas[0]["feature_cols"]:
            raise ValueError(f"feature columns of {s!r} differ from {slugs[0]!r}")
    # ties broken by source and run id, so the pooled order is reproducible
    df = pd.concat([d for d, _ in parts], ignore_index=True).sort_values(
        [ORDER_COL, "source", "run_id"], kind="stable"
    )

    if until is not None:
        cut = pd.Timestamp(until)
        df = df[df[ORDER_COL] <= (cut if cut.tzinfo else cut.tz_localize(UTC))]
    if target == "T2":
        df = df[in_t2(df)]
    df = df.reset_index(drop=True)

    features = list(schemas[0]["feature_cols"])
    if target == "T2":
        features.remove(PREV_LABEL_COL)  # 0 or NaN on T2, as `lineage_first_run`
    n_dev = int(len(df) * DEV_FRACTION)  # cut after the T2 filter, as in project 1
    y = df[LABEL_COL].astype(int).rename("failed")
    if isinstance(targets, str):
        schema, dataset = schemas[0], targets
    else:
        schema = {
            k: v for k, v in schemas[0].items() if all(sc.get(k) == v for sc in schemas)
        }
        dataset = pool_name(slugs)
    by_source = y.groupby(df["source"])
    meta = schema | {
        "dataset": dataset,  # file key of artefacts derived from this data
        "target": target,
        "n_rows": len(df),
        "n_failed": int(y.sum()),
        "n_dev": n_dev,
        "sources": slugs,
        "per_source": {
            s: {
                "n_rows": int(by_source.size().get(s, 0)),
                "n_failed": int(by_source.sum().get(s, 0)),
            }
            for s in slugs
        },
    }
    return Dataset(
        X=df[features].astype(float),
        y=y,
        dev_mask=np.arange(len(df)) < n_dev,
        meta=meta,
        order=df[ORDER_COL],
        source=df["source"],
    )
