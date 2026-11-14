import importlib.util
import json
import pathlib

import mock_pool
import numpy as np
import pandas as pd
import pytest
from test_notebooks import built, pool, write_feature_lists, write_sandbox  # noqa: F401

from acise_p2_cidt.data.load import pool_name
from acise_p2_cidt.validate import live, sandbox
from acise_p2_cidt.viz import plots

spec = importlib.util.spec_from_file_location(
    "report_assets", pathlib.Path(__file__).parents[1] / "scripts" / "report_assets.py"
)
ra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ra)


def test_tex_table_escapes_and_names_are_letters():
    t = ra.tex_table(pd.DataFrame({"a_b": ["50%"]}, index=["x&y"]))
    assert r"a\_b" in t and r"50\%" in t and r"x\&y" in t and r"\toprule" in t
    n = ra.Numbers()
    n["Rows"], n["Share"], n["Name"] = 12345, 0.25, "pool_1"
    assert n.text().splitlines() == [r"\newcommand{\Rows}{12\,345}",
                                     r"\newcommand{\Share}{0.250}",
                                     r"\newcommand{\Name}{pool\_1}"]  # fmt: skip
    with pytest.raises(ValueError):
        n["T2Rows"] = 1


def _results(sources: list[str]) -> None:
    """CV, E1/E3 and re-test rows and one sidecar, as the notebooks save them."""
    ds = pool_name(sources)
    rng = np.random.default_rng(0)
    d = live.MODEL_DIR
    d.mkdir(parents=True, exist_ok=True)

    def rows(methods, srcs, fold=range(5)):
        return pd.DataFrame(
            {"method": m, "fold": k, "source": s, "n_test": 100, "n_fail": 20,
             "mcc": rng.random(), "precision": rng.random(), "recall": rng.random()}
            for m in methods for k in fold for s in srcs
        )  # fmt: skip

    methods = ["cart", "cart_source_thr", "forest", "forest_source_thr", "source_rate",
               "parrot", "majority"]  # fmt: skip
    for t in ("T1", "T2"):
        rows(methods, ["all", *sources]).to_csv(
            d / f"cv_results_{t}_{ds}.csv", index=False
        )
        for m in ("cart", "forest"):
            stem = live.model_path(ds, t, m).with_suffix("")
            r = rows([m, "parrot", "majority"], ["all", *sources], fold=[-1])
            r.to_csv(f"{stem}_retest.csv", index=False)
    rows(["cart_e1", "cart_e3"], sources).to_csv(
        d / f"e_results_T2_{ds}.csv", index=False
    )
    side = {"cutoff": "2026-11-05T00:00:00+00:00", "threshold": 0.4, "n_train": 80,
            "n_val": 20, "params": {"max_depth": 4}}  # fmt: skip
    live.model_path(ds, "T2", "cart").with_suffix(".json").write_text(json.dumps(side))


def test_assets_from_mock_results(built, tmp_path, monkeypatch):  # noqa: F811
    write_feature_lists({(t, pool_name(built)) for t in ("T1", "T2")})
    for t in mock_pool.TARGETS:
        mock_pool.write_manifest(t, None, "2026-11-04")
    _results(built)
    monkeypatch.setattr(plots, "FIG_DIR", tmp_path / "fig")
    monkeypatch.setattr(sandbox, "SANDBOX_DIR", tmp_path / "sandbox")
    (tmp_path / "fig").mkdir()
    (tmp_path / "fig" / "tree_cv_box_T2.png").write_bytes(b"png")
    out = ra.main(tmp_path / "assets")
    numbers = (out / "numbers.tex").read_text()
    assert r"\newcommand{\NSources}{3}" in numbers
    assert r"\newcommand{\Cutoff}{2026-11-05}" in numbers
    assert r"\newcommand{\SandboxDone}{no}" in numbers
    names = {line.split("}")[0].split("\\")[-1] for line in numbers.splitlines()}
    assert all(n.isalpha() for n in names)
    for k in (
        "sources",
        "cv",
        "cells",
        "transfer",
        "retest",
        "retest_sources",
        "features",
    ):
        assert (out / "tables" / f"{k}.tex").read_text().count(r"\midrule") == 1, k
    sources = (out / "tables" / "sources.tex").read_text()
    assert sources.count(r"\\") == 1 + len(built)  # header + one row per source
    assert (out / "figures" / "tree_cv_box_T2.png").exists()


def test_sandbox_numbers_from_saved_report(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "SANDBOX_DIR", tmp_path)
    outcome = write_sandbox(tmp_path)
    done = "2026-11-20T10:03:00Z"
    rep = sandbox.report(tmp_path / "predictions.jsonl",
                         lambda repo, rid: {"status": "completed", "conclusion": outcome[rid],
                                            "updated_at": done})  # fmt: skip
    # as validation.ipynb saves them
    for k in ("scores", "confusion", "by_breakage", "runs", "imputed"):
        rep[k].to_csv(tmp_path / f"report_{k}.csv")
    (tmp_path / "report_dropped.json").write_text(json.dumps(rep["dropped"]))
    n, tables = ra.Numbers(), {}
    ra.sandbox_section(n, tables)
    text = n.text()
    assert (
        r"\newcommand{\SandboxRuns}{32}" in text
        and r"\newcommand{\SandboxWarmup}{2}" in text
    )
    assert set(tables) == {"sandbox_scores", "sandbox_breakage", "sandbox_confusion",
                           "sandbox_imputed"}  # fmt: skip
