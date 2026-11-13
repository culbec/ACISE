import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from acise_p2_cidt.data.load import Dataset
from acise_p2_cidt.tree.cart import CART
from acise_p2_cidt.tree.forest import RandomForest
from acise_p2_cidt.validate import live

N = 400
CUTOFF_ROW = 299  # rows 0..299 are training (up to C), 300..399 are new
UNTIL = date(2025, 12, 31)  # last collection day; C = 2026-01-01T00:00Z
C = 1767225600.0
T0 = C - 3600.0 * (CUTOFF_ROW + 1) + 60  # one run per hour; row 300 is just after C
# two sources: a branch-scoped one (even rows) and an all-branch one (odd rows)
SOURCES = {
    "o__r__1@main": {"repo": "o/r", "workflow_id": 1, "branch": "main"},
    "o__s__2": {"repo": "o/s", "workflow_id": 2, "branch": None},
}


def table() -> pd.DataFrame:
    """Built-table stand-in: f0 drives the label, f1 is noise with some NaN."""
    rng = np.random.default_rng(0)
    f0, f1 = rng.normal(size=N), rng.normal(size=N)
    failed = (f0 + 0.3 * rng.normal(size=N) > 0.8).astype(int)
    f1[::17] = np.nan
    return pd.DataFrame(
        {
            "run_id": np.arange(N) + 1000,
            "head_sha": [f"{i:040x}" for i in range(N)],
            "created_at": T0 + 3600.0 * np.arange(N),
            "failed": failed,
            "prev_failed": np.r_[np.nan, failed[:-1]],
            "f0": f0,
            "f1": f1,
            "workflow_id": np.where(np.arange(N) % 2 == 0, 1, 2),
        }
    )


def dataset(df: pd.DataFrame) -> Dataset:
    """`load(..., "T1")` of the training rows."""
    df = df.iloc[: CUTOFF_ROW + 1]
    return Dataset(
        X=df[["prev_failed", "f0", "f1"]],
        y=df["failed"],
        dev_mask=np.arange(len(df)) < 60,
        meta={},
        order=pd.to_datetime(df["created_at"], unit="s", utc=True),
        source=pd.Series(np.where(df["workflow_id"] == 1, *SOURCES)),
    )


@pytest.fixture
def trained(tmp_path, monkeypatch):
    df = table()
    monkeypatch.setattr(live, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(live, "load", lambda sources, target: dataset(df))
    monkeypatch.setattr(live, "_refresh", lambda t, cutoff: None)
    monkeypatch.setattr(live, "schema_path", lambda s: tmp_path / f"schema_{s}.json")
    for s, t in SOURCES.items():
        (tmp_path / f"schema_{s}.json").write_text(
            json.dumps(t | {"exclude_jobs": ["^x$"]})
        )

    def fake_build(cfg, select=None, extra=None):
        assert cfg.exclude_jobs == ("^x$",)  # the source's own label rules
        own = df[df["workflow_id"] == cfg.target.workflow_id]
        return (own if select is None else own[select(own)]), {}

    monkeypatch.setattr(live, "build", fake_build)
    path = live.train_final(
        list(SOURCES), "T1", "cart", {"max_depth": 3}, ["f0", "f1"], until=UNTIL
    )
    return path, df


def fitted(kind: str) -> CART | RandomForest:
    X = table()[["f0"]]
    y = table()["failed"]
    if kind == "cart":
        return CART(max_depth=4, ccp_alpha=1e-3).fit(X, y)
    return RandomForest(n_estimators=5, max_depth=3).fit(X, y)


@pytest.mark.parametrize("kind", ["cart", "forest"])
def test_round_trip_gives_identical_predictions(tmp_path, kind):
    est = fitted(kind)
    path = live.save_model(est, tmp_path / "m.skops", {"a": 1})
    loaded, side = live.load_model(path)
    X = table()[["f0"]]
    assert type(loaded) is type(est)
    assert side["a"] == 1
    np.testing.assert_array_equal(loaded.predict_proba(X), est.predict_proba(X))


def test_modified_byte_fails_the_hash_check(tmp_path):
    path = live.save_model(fitted("cart"), tmp_path / "m.skops", {})
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="SHA-256"):
        live.load_model(path)


class Foreign:
    pass


def test_type_outside_trusted_is_not_saved(tmp_path):
    est = fitted("cart")
    est.extra_ = Foreign()
    with pytest.raises(ValueError, match="outside TRUSTED"):
        live.save_model(est, tmp_path / "m.skops", {})
    assert not (tmp_path / "m.skops").exists()


def test_train_final_sidecar(trained):
    path, df = trained
    _, side = live.load_model(path)
    # 300 training rows: 300 - int(300 * 0.2) = 240 fit, 60 validation
    assert (side["n_train"], side["n_val"]) == (240, 60)
    assert side["cutoff"] == pd.Timestamp(C, unit="s", tz="UTC").isoformat()
    assert side["collection_until"] == "2025-12-31"
    assert side["medians"]["f1"] == pytest.approx(df["f1"].iloc[:240].median())
    assert side["features"] == ["f0", "f1"]
    assert side["dataset"] == live.pool_name(list(SOURCES))
    assert path.name == f"cart_{side['dataset']}_T1.skops"
    # every source round-trips, the all-branch one with branch None
    assert side["sources"] == [
        {"slug": s, **t, "exclude_jobs": ["^x$"]} for s, t in SOURCES.items()
    ]


def test_train_final_refuses_runs_after_c(trained, monkeypatch):
    with pytest.raises(ValueError, match="after C"):
        live.train_final(list(SOURCES), "T1", "cart", {"max_depth": 3}, ["f0"],
                         until=date(2025, 12, 30))  # fmt: skip


def test_collection_until_needs_one_day(tmp_path, monkeypatch):
    monkeypatch.setattr(live.gh_runs, "RESULTS_DIR", tmp_path)
    for s, d in (("a", "2026-11-01"), ("b", "2026-11-01")):
        (tmp_path / f"manifest_{s}.json").write_text(json.dumps({"until": d}))
    assert live.collection_until(["a", "b"]) == date(2026, 11, 1)
    (tmp_path / "manifest_b.json").write_text(json.dumps({"until": "2026-10-31"}))
    with pytest.raises(ValueError, match="different days"):
        live.collection_until(["a", "b"])


def test_retest_scores_only_rows_after_cutoff(trained):
    path, df = trained
    est, _ = live.load_model(path)
    before = est.tree_.value.copy()
    out = live.retest(path)
    new = df.iloc[CUTOFF_ROW + 1 :]
    pooled = out[out["source"] == "all"]
    assert list(pooled["method"]) == ["cart", "parrot", "majority"]
    assert (pooled["n_test"] == len(new)).all()
    assert (pooled["n_fail"] == new["failed"].sum()).all()
    # per source: the pooled rows split into the two sources
    per = out[out["source"] != "all"].groupby("method")[["n_test", "n_fail"]].sum()
    assert (
        per.values
        == pooled.set_index("method").loc[per.index, ["n_test", "n_fail"]].values
    ).all()
    assert set(out["source"]) == {"all", *SOURCES}
    runs = pd.read_csv(f"{path.with_suffix('')}_retest_runs.csv")
    assert runs["run_id"].min() == new["run_id"].min()
    assert set(runs["source"]) == set(SOURCES)
    np.testing.assert_array_equal(live.load_model(path)[0].tree_.value, before)


def test_retest_builds_each_source_once_for_all_models(trained, monkeypatch):
    path, _ = trained
    forest_path = live.train_final(
        list(SOURCES), "T1", "forest", {"n_estimators": 3}, ["f0", "f1"], until=UNTIL
    )
    calls, build = [], live.build
    monkeypatch.setattr(
        live, "build", lambda cfg, **k: calls.append(cfg) or build(cfg, **k)
    )
    out = live.retest([path, forest_path])
    assert len(calls) == len(SOURCES)
    assert {"cart", "forest"} <= set(out["method"]) and set(out["target"]) == {"T1"}
    assert (
        pd.read_csv(f"{forest_path.with_suffix('')}_retest.csv")["method"][0]
        == "forest"
    )


def test_predict_run_of_a_running_run(trained, monkeypatch):
    path, df = trained
    row = df.iloc[350]
    run = {
        "id": int(row["run_id"]),
        "head_sha": row["head_sha"],
        "html_url": "https://example.invalid/run",
        "created_at": "2026-01-01T00:00:00Z",
        "status": "in_progress",
        "conclusion": None,
    }
    monkeypatch.setattr(live, "_fetch_run", lambda t, ref: run)
    r = live.predict_run(path, str(run["id"]), "o__r__1@main")
    est, side = live.load_model(path)
    root = side["features"][est.tree_.feature[0]]
    assert r["label"] is None
    assert r["path"][0].startswith(f"{root} = ")
    assert r["path"][-1].startswith("leaf: P(fail)")
    assert r["predicted"] == (r["p_fail"] > side["threshold"])


def test_predict_run_refuses_unknown_source(trained):
    path, _ = trained
    with pytest.raises(ValueError, match="not a source"):
        live.predict_run(path, "1", "x__y__3")


def test_fetch_run_checks_branch_only_when_scoped(monkeypatch):
    run = {"id": 7, "workflow_id": 2, "head_branch": "feature/x"}
    monkeypatch.setattr(live.http, "get", lambda url, params=None: run)
    assert live._fetch_run(live.Target("o/s", 2, None), "7") is run
    with pytest.raises(ValueError, match="on 'feature/x'"):
        live._fetch_run(live.Target("o/s", 2, "main"), "7")
    with pytest.raises(ValueError, match="workflow 2"):
        live._fetch_run(live.Target("o/s", 3, None), "7")


def test_predict_run_reads_a_retested_run_without_fetching(trained, monkeypatch):
    path, df = trained
    live.retest(path)
    monkeypatch.setattr(live, "_fetch_run", lambda t, ref: pytest.fail("fetched"))
    monkeypatch.setattr(live, "build", lambda *a, **k: pytest.fail("rebuilt"))
    row = df.iloc[350]  # even row: source o__r__1@main
    r = live.predict_run(path, str(row["run_id"]), "o__r__1@main")
    assert (r["run_id"], r["label"]) == (row["run_id"], row["failed"])
    assert r["url"].endswith(f"o/r/actions/runs/{row['run_id']}")
    assert r["path"][-1].startswith("leaf: P(fail)")
