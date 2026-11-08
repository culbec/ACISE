import pathlib

import numpy as np
import pandas as pd
import pytest

from acise_p2_cidt.data.history import history_features

GOLDEN = pathlib.Path(__file__).parent / "fixtures" / "history_golden.csv"


def frame(rows):
    """rows: (lineage, created, completed, failed, head_sha)."""
    df = pd.DataFrame(
        rows, columns=["lineage", "created_at", "completed_at", "failed", "head_sha"]
    )
    df[["created_at", "completed_at"]] = df[["created_at", "completed_at"]].astype(
        float
    )
    return df.assign(
        run_id=range(len(df)),
        author_id="a",
        latest_completed_at=df["completed_at"],
        flaky=0,
        duration=df["completed_at"] - df["created_at"],
        queue_delay=0.0,
        n_failed_jobs=df["failed"].astype(float),
    )


# Two PRs whose runs interleave in time; every run completes before the next starts.
INTERLEAVED = [
    ("A", 0, 10, 1, "a0"),
    ("B", 20, 30, 0, "b0"),
    ("A", 40, 50, 1, "a1"),
    ("B", 60, 70, 1, "b1"),
    ("A", 80, 90, 0, "a2"),
]


def test_previous_run_follows_its_own_lineage():
    h = history_features(frame(INTERLEAVED))
    # the workflow-wide previous run of B's first run is A's failure: not used
    assert np.isnan(h.loc[1, "prev_failed"])
    assert h.loc[2, "prev_failed"] == 1 and h.loc[2, "fail_streak"] == 1  # a0
    assert h.loc[3, "prev_failed"] == 0 and h.loc[3, "fail_streak"] == 0  # b0
    assert h.loc[4, "prev_failed"] == 1 and h.loc[4, "fail_streak"] == 2  # a0, a1
    assert h.loc[4, "fail_rate_last_5"] == 1.0  # A only
    assert h.loc[4, "fail_rate_7d"] == pytest.approx(3 / 4)  # time windows: all runs
    assert h.loc[3, "secs_since_lineage_prev_run"] == 40
    assert h.loc[3, "secs_since_prev_run"] == 20
    assert h["_prev_idx"].tolist()[2:] == [0, 1, 2]


def test_first_run_of_a_lineage():
    h = history_features(frame(INTERLEAVED))
    assert h["lineage_first_run"].tolist() == [1, 1, 0, 0, 0]
    assert h["lineage_prior_runs"].tolist() == [0, 0, 1, 1, 2]
    assert h.loc[1, "fail_streak"] == 0 and np.isnan(h.loc[1, "runs_since_last_fail"])
    assert np.isnan(h.loc[1, "_green_idx"]) and np.isnan(
        h.loc[1, "secs_since_last_fail"]
    )


def test_running_lineage_run_is_not_previous():
    # A's first run is still running when its second run is created
    h = history_features(frame([("A", 0, 100, 1, "x"), ("A", 50, 60, 0, "y"),
                                ("A", 200, 210, 0, "z")]))  # fmt: skip
    assert np.isnan(h.loc[1, "prev_failed"]) and h.loc[1, "lineage_first_run"] == 1
    assert h.loc[1, "secs_since_lineage_prev_run"] == 50  # its creation is known
    assert h.loc[2, "prev_failed"] == 0  # latest-created known run: the second


def test_sha_prior_failed_reads_only_completed_runs():
    h = history_features(frame([("A", 0, 100, 1, "s"), ("B", 50, 60, 0, "s"),
                                ("C", 200, 210, 0, "s")]))  # fmt: skip
    assert np.isnan(h.loc[0, "sha_prior_failed"])
    assert np.isnan(h.loc[1, "sha_prior_failed"])  # run 0 not completed at t=50
    assert h.loc[2, "sha_prior_failed"] == 0  # latest-created known: run 1


def test_without_lineage_matches_the_workflow_wide_history():
    rows = [(0, 100, 0), (200, 1000, 1), (300, 400, 0), (500, 600, 1), (4200, 4300, 0)]
    df = frame([("", c, e, f, f"{i:040x}") for i, (c, e, f) in enumerate(rows)])
    df["author_id"] = ["a", "b", "a", "a", "b"]
    golden = pd.read_csv(GOLDEN)
    h = history_features(df.drop(columns="lineage")).reset_index(drop=True)
    pd.testing.assert_frame_equal(h[golden.columns], golden, check_dtype=False)
