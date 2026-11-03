import numpy as np
import pandas as pd
import pytest

from acise_p2_cidt.data.load import Dataset
from acise_p2_cidt.tree.cart import CART
from acise_p2_cidt.tree.forest import RandomForest
from acise_p2_cidt.validate import live

N = 400
CUTOFF_ROW = 299  # rows 0..299 are training (up to C), 300..399 are new
T0 = 1.7e9  # unix seconds of row 0; one run per hour


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
        }
    )


def dataset(df: pd.DataFrame) -> Dataset:
    """`load(..., "T1")` of the training rows."""
    df = df.iloc[: CUTOFF_ROW + 1]
    return Dataset(
        X=df[["prev_failed", "f0", "f1"]],
        y=df["failed"],
        dev_mask=np.arange(len(df)) < 60,
        meta={"repo": "o/r", "workflow_id": 1, "branch": "main", "collected_at": "x"},
        order=pd.to_datetime(df["created_at"], unit="s", utc=True),
    )


@pytest.fixture
def trained(tmp_path, monkeypatch):
    df = table()
    monkeypatch.setattr(live, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(live, "load", lambda repo, target: dataset(df))
    monkeypatch.setattr(live, "_refresh", lambda t, cutoff: None)

    def fake_build(t, select=None, extra=None):
        return (df if select is None else df[select(df)]), {}

    monkeypatch.setattr(live, "build", fake_build)
    path = live.train_final("o/r", "T1", "cart", {"max_depth": 3}, ["f0", "f1"])
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
    assert side["cutoff"] == pd.Timestamp(T0 + 3600 * CUTOFF_ROW, unit="s",
                                          tz="UTC").isoformat()  # fmt: skip
    assert side["medians"]["f1"] == pytest.approx(df["f1"].iloc[:240].median())
    assert side["features"] == ["f0", "f1"]


def test_retest_scores_only_rows_after_cutoff(trained):
    path, df = trained
    est, _ = live.load_model(path)
    before = est.tree_.value.copy()
    out = live.retest(path)
    new = df.iloc[CUTOFF_ROW + 1 :]
    assert list(out["method"]) == ["cart", "parrot", "majority"]
    assert (out["n_test"] == len(new)).all()
    assert (out["n_fail"] == new["failed"].sum()).all()
    runs = pd.read_csv(f"{path.with_suffix('')}_retest_runs.csv")
    assert runs["run_id"].min() == new["run_id"].min()
    np.testing.assert_array_equal(live.load_model(path)[0].tree_.value, before)


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
    r = live.predict_run(path, str(run["id"]))
    est, side = live.load_model(path)
    root = side["features"][est.tree_.feature[0]]
    assert r["label"] is None
    assert r["path"][0].startswith(f"{root} = ")
    assert r["path"][-1].startswith("leaf: P(fail)")
    assert r["predicted"] == (r["p_fail"] > side["threshold"])
