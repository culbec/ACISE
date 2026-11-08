"""
Notebooks on the mock pool. Each notebook runs only the cells already edited for
several targets; `UNTIL` holds the last such cell per notebook.
"""

import pathlib

import mock_pool
import pytest
from nb_harness import run_notebook

from acise_p2_cidt.collect import http
from acise_p2_cidt.collect.targets import slug

NB = pathlib.Path(__file__).resolve().parents[1] / "notebook"
# last cell edited for multi-repo, per notebook
UNTIL = {"collection.ipynb": None}


@pytest.fixture
def pool(tmp_path, monkeypatch):
    def no_network(*a, **k):
        raise AssertionError("network call without a stub")

    monkeypatch.setattr(http, "get", no_network)
    monkeypatch.setattr(http, "get_all", no_network)
    return mock_pool.mock_pool(tmp_path, monkeypatch)


@pytest.mark.notebook
def test_collection(pool):
    ns = run_notebook(
        NB / "collection.ipynb",
        {
            "collect": mock_pool.write_manifest,
            "build": lambda cfg: mock_pool.mock_table(slug(cfg.target)),
        },
        until=UNTIL["collection.ipynb"],
    )
    assert ns["SOURCES"] == pool
    assert list(ns["manifest_table"].index) == pool
    assert {"runs", "api_calls", "event: push"} <= set(ns["manifest_table"].columns)
    lt = ns["label_table"]
    assert list(lt.index) == pool and (lt["failed"] == 3).all()
    # infra: the `Result`-only failure and the one without failed jobs
    assert (lt["infra_share"] == 2 / 3).all() and (lt["t2_failures"] == 1).all()
    # build and per-source sections
    st = ns["schema_table"]
    assert (
        list(st.index) == pool and (st["features"] == len(mock_pool.FEATURE_COLS)).all()
    )
    assert (st["dropped: infra_failure"] == 3).all() and (st["no_git"] < 0.1).all()
    assert list(ns["shifts"].index) == [(pool[0], "2026-07")]
    assert list(ns["lineage_table"].index) == pool
    bt = ns["balance_table"]
    assert (bt[("runs", "T2")] < bt[("runs", "T1")]).all() and bt[("eligible", "e1_e3")].all()
    assert (
        list(ns["streak_table"].index) == pool
        and (ns["streak_table"]["max"] >= 1).all()
    )
    assert set(ns["flaky_table"].index) == set(pool)
