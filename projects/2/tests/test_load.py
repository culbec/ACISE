import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from acise_p2_cidt.data import load

SLUG = "owner__name__1"
# prev_failed: NaN (first run), then alternating; 10 rows, one per day.
TABLE = pd.DataFrame(
    {
        "run_id": range(10),
        "head_sha": [f"{i:040x}" for i in range(10)],
        "author_id": "x",
        "created_at": pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC"),
        "failed": [0, 1, 0, 0, 1, 1, 0, 0, 0, 1],
        "prev_failed": [np.nan, 0, 1, 0, 0, 1, 1, 0, 0, 0],
        "churn": np.arange(10.0),
        "flaky": 0,
        "failed_job_names": "",
    }
)
SCHEMA = {
    "repo": "owner/name",
    "id_cols": ["run_id", "head_sha", "author_id"],
    "diagnostic_cols": ["flaky", "failed_job_names"],
    "nominal": [],
    "feature_cols": ["prev_failed", "churn"],
}


@pytest.fixture(autouse=True)
def table(tmp_path, monkeypatch):
    monkeypatch.setattr(load, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(load, "SCHEMA_DIR", tmp_path / "results")
    load.PROCESSED_DIR.mkdir()
    load.SCHEMA_DIR.mkdir()
    TABLE.sample(frac=1, random_state=0).to_csv(
        load.table_path(SLUG), index=False
    )  # unsorted
    load.schema_path(SLUG).write_text(json.dumps(SCHEMA))


def test_t1():
    ds = load.load(SLUG, "T1")
    assert len(ds.X) == 10 and ds.order.is_monotonic_increasing
    assert list(ds.X.columns) == ["prev_failed", "churn"]  # no id, label or diagnostic
    assert ds.y.tolist() == TABLE["failed"].tolist()
    assert ds.dev_mask.tolist() == [True] * 2 + [False] * 8  # int(10 * 0.2)
    assert (ds.meta["repo"], ds.meta["dataset"]) == ("owner/name", SLUG)
    assert ds.meta["n_failed"] == 4


def test_t2():
    ds = load.load(SLUG, "T2")
    # prev_failed == 0 on rows 1, 3, 4, 7, 8, 9; NaN (row 0, no previous run) is kept
    assert ds.X["churn"].tolist() == [0, 1, 3, 4, 7, 8, 9]
    assert "prev_failed" not in ds.X  # 0 or NaN on T2
    assert ds.dev_mask.tolist() == [True] + [False] * 6  # cut after the filter
    assert ds.y.tolist() == [0, 1, 0, 1, 0, 0, 1]


@pytest.mark.parametrize(
    "until, n",
    [
        (datetime(2026, 1, 3), 3),  # naive = UTC, inclusive
        (pd.Timestamp("2026-01-03T23:00", tz="UTC").to_pydatetime(), 3),
        (None, 10),
    ],
)
def test_until(until, n):
    assert len(load.load(SLUG, "T1", until=until).X) == n


def test_bad_input():
    with pytest.raises(ValueError, match="'T3'"):
        load.load(SLUG, "T3")
    with pytest.raises(FileNotFoundError, match="build-dataset"):
        load.load("other__repo__2", "T1")
    assert load.available_targets() == [SLUG]


OTHER = "owner__other__2"


def write_other(feature_cols=("prev_failed", "churn")):
    """A second source, 12 hours after each SLUG row, with 3 rows sharing SLUG times."""
    other = TABLE.assign(
        run_id=TABLE["run_id"] + 100,
        created_at=TABLE["created_at"] + pd.Timedelta(hours=12),
        churn=TABLE["churn"] + 0.5,
    )
    other.loc[:2, "created_at"] = TABLE.loc[:2, "created_at"]  # ties with SLUG
    other.to_csv(load.table_path(OTHER), index=False)
    load.schema_path(OTHER).write_text(
        json.dumps(SCHEMA | {"repo": "owner/other", "feature_cols": list(feature_cols)})
    )


def test_pool_interleaves_by_time():
    write_other()
    ds = load.load([SLUG, OTHER], "T1")
    assert len(ds.X) == 20 and ds.order.is_monotonic_increasing
    # rows 0-2 tie on created_at: SLUG before OTHER (by source name)
    assert ds.source.tolist()[:8] == [SLUG, OTHER] * 4
    assert ds.X["churn"].tolist()[:8] == [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
    assert ds.meta["dataset"] == load.pool_name([OTHER, SLUG])
    assert ds.meta["sources"] == [SLUG, OTHER] and "repo" not in ds.meta
    assert ds.meta["per_source"] == {
        SLUG: {"n_rows": 10, "n_failed": 4},
        OTHER: {"n_rows": 10, "n_failed": 4},
    }
    assert ds.meta["feature_cols"] == SCHEMA["feature_cols"]


def test_pool_filters_after_pooling():
    write_other()
    t2 = load.load([SLUG, OTHER], "T2")
    assert len(t2.X) == 14 and t2.dev_mask.sum() == int(14 * 0.2)  # dev cut on the pool
    assert (t2.source.value_counts() == 7).all()
    cut = load.load([SLUG, OTHER], "T1", until=datetime(2026, 1, 2))
    assert cut.source.tolist() == [
        SLUG,
        OTHER,
        SLUG,
        OTHER,
    ]  # 01-01, 01-01 12h, 01-02, ...
    assert cut.meta["per_source"][OTHER]["n_rows"] == 2


def test_pool_feature_mismatch():
    write_other(feature_cols=("churn",))
    with pytest.raises(ValueError, match=OTHER):
        load.load([SLUG, OTHER], "T1")


def test_single_slug_source():
    ds = load.load(SLUG, "T1")
    assert (ds.source == SLUG).all() and ds.meta["sources"] == [SLUG]


def test_pool_name():
    assert load.pool_name([SLUG, OTHER]) == load.pool_name([OTHER, SLUG])
    assert load.pool_name([SLUG]) != load.pool_name([SLUG, OTHER])
    assert (
        load.pool_name([SLUG]).startswith("pool-") and len(load.pool_name([SLUG])) == 13
    )


def test_subset_and_repo_of():
    write_other()
    ds = load.load([SLUG, OTHER], "T1")
    part = load.subset(ds, (ds.source == OTHER).to_numpy(), dataset="part")
    assert (part.source == OTHER).all() and len(part.X) == 10
    assert part.order.is_monotonic_increasing and part.X.index.equals(pd.RangeIndex(10))
    assert part.dev_mask.tolist() == ds.dev_mask[(ds.source == OTHER).to_numpy()].tolist()
    assert part.meta["dataset"] == "part" and part.meta["sources"] == [OTHER]
    assert load.repo_of("acme__app__12@main") == "acme__app"
