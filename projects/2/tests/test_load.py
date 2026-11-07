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
    # prev_failed == 0 on rows 1, 3, 4, 7, 8, 9; NaN (row 0) is dropped
    assert ds.X["churn"].tolist() == [1, 3, 4, 7, 8, 9]
    assert "prev_failed" not in ds.X  # constant on T2
    assert ds.dev_mask.tolist() == [True] + [False] * 5  # cut after the filter
    assert ds.y.tolist() == [1, 0, 1, 0, 0, 1]


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
