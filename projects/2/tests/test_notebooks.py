"""
Notebooks on the mock pool. Each notebook runs only the cells already edited for
several targets; `UNTIL` holds the last such cell per notebook.
"""

import pathlib

import mock_pool
import pytest
from nb_harness import run_notebook

from acise_p2_cidt.collect import http

NB = pathlib.Path(__file__).resolve().parents[1] / "notebook"
# last cell edited for multi-repo, per notebook
UNTIL = {"collection.ipynb": 9}


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
        {"collect": mock_pool.write_manifest},
        until=UNTIL["collection.ipynb"],
    )
    assert ns["SOURCES"] == pool
    assert list(ns["manifest_table"].index) == pool
    assert {"runs", "api_calls", "event: push"} <= set(ns["manifest_table"].columns)
    lt = ns["label_table"]
    assert list(lt.index) == pool and (lt["failed"] == 3).all()
    # infra: the `Result`-only failure and the one without failed jobs
    assert (lt["infra_share"] == 2 / 3).all() and (lt["t2_failures"] == 1).all()
