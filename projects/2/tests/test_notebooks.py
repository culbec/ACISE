"""
Notebooks on the mock pool. Each notebook runs only the cells already edited for
several targets; `UNTIL` holds the last such cell per notebook.
"""

import json
import pathlib
import re
from types import SimpleNamespace

import mock_pool
import nbformat
import numpy as np
import pandas as pd
import pytest
from nb_harness import check_display, run_notebook

from acise_p2_cidt.analysis import features
from acise_p2_cidt.collect import http
from acise_p2_cidt.collect.targets import slug
from acise_p2_cidt.data.load import pool_name
from acise_p2_cidt.metrics.metrics import domain, scores
from acise_p2_cidt.tree import forest, tune
from acise_p2_cidt.validate import live

NB = pathlib.Path(__file__).resolve().parents[1] / "notebook"
# last cell edited for multi-repo, per notebook
UNTIL = {"collection.ipynb": None, "validation.ipynb": 14}


def test_display_rule():
    """Captions and titles are checked; a bare table or untitled figure fails."""
    import pandas as pd
    import plotly.graph_objects as go
    from IPython.display import Markdown

    check_display("text", Markdown("**t**"), pd.DataFrame())
    check_display(go.Figure().update_layout(title="t"))
    for bad in [(pd.DataFrame(),), (go.Figure(),), ("", pd.Series(dtype=int))]:
        with pytest.raises(AssertionError):
            check_display(*bad)


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
    assert ns["SOURCES"] == pool and ns["t1"].meta["sources"] == pool
    assert list(ns["pool_table"].columns) == ["T1", "T2"]
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
    with_pool = [*pool, "all"]
    assert list(ns["lineage_table"].index) == with_pool
    bt = ns["balance_table"]
    assert (
        list(bt.index) == with_pool and (bt[("runs", "T2")] < bt[("runs", "T1")]).all()
    )
    assert bt.loc[pool, ("eligible", "e1_e3")].all()
    assert bt.loc["all", ("runs", "T1")] == ns["t1"].meta["n_rows"]
    st = ns["streak_table"]
    assert list(st.index) == with_pool and (st["max"] >= 1).all()
    assert st.loc["all", "streaks"] == st.loc[pool, "streaks"].sum()
    assert list(ns["flaky_table"].index) == with_pool


@pytest.fixture
def built(pool):
    """The mock pool with every source's table already built."""
    mock_pool.write_tables()
    return pool


@pytest.mark.notebook
def test_eda(built):
    ns = run_notebook(NB / "eda.ipynb", {})
    assert ns["t2"].meta["sources"] == built
    assert list(ns["sizes"].index) == [*built, "all"]
    assert set(ns["nan_by_source"].columns) == set(built)
    assert set(ns["rates"].columns.get_level_values(0)) == {"all", *built}
    assert {"event_push", "from_fork", "lineage_first_run"} <= set(ns["binary"])
    assert set(ns["job_share"].columns) == set(built) and 0 <= ns["acc"] <= 1


@pytest.mark.notebook
def test_feature_selection(built):
    ns = run_notebook(NB / "feature_selection.ipynb", {})
    assert ns["t2"].meta["sources"] == built and ns["sel"].features
    assert set(ns["sel_src"]) == set(built)  # every mock source has enough failures
    for s in built:
        assert features.load_features("T2", s) == ns["sel_src"][s].features
    for s in (ns["sel"], ns["sel1"]):
        path = (
            features.OUT_DIR
            / f"selected_features_{s.meta['target']}_{pool_name(built)}.json"
        )
        assert json.loads(path.read_text())["meta"]["sources"] == built


def small_grids(monkeypatch) -> None:
    """One grid point and a 5-tree forest keep the tests fast; the notebooks are unchanged."""
    monkeypatch.setattr(tune, "GRID", {k: v[:1] for k, v in tune.GRID.items()})
    monkeypatch.setattr(
        forest,
        "GRID",
        {k: v[:1] for k, v in forest.GRID.items()} | {"n_estimators": [5]},
    )


def write_feature_lists(lists: set[tuple[str, str]]) -> None:
    """Frozen feature lists, as `feature_selection.ipynb` writes them, per (target, name)."""
    extra = [
        c
        for c in ("files_changed", "lineage_first_run", "from_fork")
        if c in mock_pool.FEATURE_COLS
    ]
    meta = {"params": {"n_max": 5, "min_freq": 0.6, "n_boot": 2, "rho": 0.8, "seed": 0},
            "excluded": []}  # fmt: skip
    for t, name in lists:
        path = features.OUT_DIR / features._file_name(t, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"features": [*mock_pool.SIGNAL, *extra], "meta": meta})
        )


@pytest.mark.notebook
def test_tree(built, monkeypatch):
    small_grids(monkeypatch)
    write_feature_lists(
        {(t, pool_name(built)) for t in ("T1", "T2")}
        | {("T2", built[0]), ("T2", built[2])}
    )
    ns = run_notebook(NB / "tree.ipynb", {})
    for t in ("T1", "T2"):
        r = ns["results"][t]
        assert set(r["source"]) == {"all", *built}
        assert set(r["method"]) == {
            "cart", "cart_source_thr", "sklearn", "parrot", "majority", "source_rate",
            "forest", "forest_source_thr",
        }  # fmt: skip
        pooled = r[r["source"] == "all"].groupby(["method", "fold"])["n_test"].sum()
        parts = r[r["source"] != "all"].groupby(["method", "fold"])["n_test"].sum()
        assert pooled.equals(parts)
        assert (live.MODEL_DIR / f"cv_results_{t}_{pool_name(built)}.csv").exists()
    # E1 for the two sources with a list; E3 leaves each of their repositories out
    e = ns["e_results"]
    assert set(e.query("method == 'cart_e1'")["source"]) == {built[0], built[2]}
    assert set(e.query("method == 'cart_e3'")["source"]) == set(built)
    cv = (
        ns["results"]["T2"]
        .query("method == 'cart'")
        .set_index(["fold", "source"])["n_test"]
    )
    for m in ("cart_e1", "cart_e3"):
        cells = e.query("method == @m").set_index(["fold", "source"])["n_test"]
        assert cells.equals(cv.loc[cells.index])  # same test rows as the pooled model


def fake_retest(paths: list[pathlib.Path]) -> pd.DataFrame:
    """`live.retest` without the network: 20 new runs per source, metrics as `retest`."""
    return pd.concat([_fake_retest_one(p) for p in paths], ignore_index=True)


def _fake_retest_one(path: pathlib.Path) -> pd.DataFrame:
    _, side = live.load_model(path)
    rng = np.random.default_rng(0)
    runs = pd.concat(
        pd.DataFrame({"run_id": np.arange(20) + 1000 * i, "source": e["slug"],
                      "head_sha": "0" * 40, "created_at": "2026-11-03T00:00:00Z",
                      "failed": rng.random(20) < 0.3, "p_fail": rng.random(20)})
        for i, e in enumerate(side["sources"])
    ).assign(pred=lambda d: d["p_fail"] > side["threshold"])  # fmt: skip
    runs.to_csv(f"{path.with_suffix('')}_retest_runs.csv", index=False)
    y, pred = runs["failed"].to_numpy(int), runs["pred"].to_numpy()
    rows = []
    for s in ["all", *runs["source"].unique()]:
        m = np.ones(len(y), bool) if s == "all" else (runs["source"] == s).to_numpy()
        for k in (side["model"], "parrot", "majority"):
            p = pred[m] if k == side["model"] else np.zeros(m.sum(), bool)
            rows.append({"method": k, "fold": -1, "source": s, "n_test": int(m.sum()),
                         "n_fail": int(y[m].sum()), **scores(y[m], p), **domain(y[m], p)})  # fmt: skip
    return pd.DataFrame(rows).assign(target=side["target"])


def fake_predict_run(path: pathlib.Path, ref: str, source: str) -> dict:
    """`live.predict_run` without the network; refuses an unknown source as it does."""
    _, side = live.load_model(path)
    live._source(side, source)
    return {"source": source, "run_id": int(ref), "sha": "0" * 40, "url": "u",
            "created_at": "2026-11-03T00:00:00Z", "label": None, "p_fail": 0.4,
            "predicted": False, "threshold": side["threshold"], "in_target": True,
            "path": ["leaf: P(fail) = 0.40"]}  # fmt: skip


@pytest.mark.notebook
def test_validation(built, monkeypatch):
    small_grids(monkeypatch)
    write_feature_lists({(t, pool_name(built)) for t in ("T1", "T2")})
    for t in mock_pool.TARGETS:
        mock_pool.write_manifest(t, None, "2026-11-04")
    # CV rows as tree.ipynb saves them, for the re-test vs CV view
    rng = np.random.default_rng(1)
    for t in ("T1", "T2"):
        live.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"method": m, "fold": k, "source": s, "mcc": rng.random(), "f1": rng.random(),
             "precision": rng.random(), "recall": rng.random()}
            for m in ("cart", "forest", "parrot", "majority", "sklearn")
            for k in range(5)
            for s in ("all", *built)
        ).to_csv(live.MODEL_DIR / f"cv_results_{t}_{pool_name(built)}.csv", index=False)  # fmt: skip
    # only the branch-scoped source (workflow 12) has a run in progress
    calls = []

    def get(url, params=None):
        calls.append((url, params))
        return {"workflow_runs": [{"id": 5}] if "/workflows/12/" in url else []}

    ns = run_notebook(
        NB / "validation.ipynb",
        {"retest": fake_retest, "predict_run": fake_predict_run,
         "http": SimpleNamespace(get=get)},
        until=UNTIL["validation.ipynb"],
    )  # fmt: skip
    assert ns["DATASET"] == pool_name(built) and len(ns["paths"]) == 4
    for p in ns["paths"].values():
        assert p.exists() and pool_name(built) in p.name
        assert [e["slug"] for e in live.load_model(p)[1]["sources"]] == built
    for t in ("T1", "T2"):
        assert set(ns["retests"][t]["source"]) == {"all", *built}
    assert ns["case_results"]
    assert ns["live_run"]["source"] == built[1]
    # the all-branch source is asked without a branch or event filter
    assert calls[0][1] == {"status": "in_progress", "per_page": 1}
    assert calls[1][1]["branch"] == "main"


def test_no_single_repo_names():
    """Notebooks edited for the pool no longer name n8n constants."""
    for name in ("collection", "eda", "feature_selection", "tree", "validation"):
        text = "".join(
            c.source for c in nbformat.read(NB / f"{name}.ipynb", as_version=4).cells
        )
        assert not re.search(r"N8N_MASTER|REGIME|\bREPO\s*=", text), name
