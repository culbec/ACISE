import json
import random

import numpy as np
import pandas as pd
import pytest

from acise_p2_cidt.collect.targets import Target, TargetConfig
from acise_p2_cidt.validate import live, sandbox, scenarios
from acise_p2_cidt.validate.scenarios import SHAPES, plan

# =============================================================================
# Plan
# =============================================================================


def test_plan_is_seeded():
    assert plan(3) == plan(3)
    assert plan(3) != plan(4)


def test_every_shape_both_ways_after_the_warmup():
    p = plan(0, warmup=30, scored=40)
    warm, scored = p[:30], p[30:]
    assert all(s.phase == "warmup" and s.breakage == "none" for s in warm)
    assert len(scored) == 40 and all(s.phase == "scored" for s in scored)
    for shape in SHAPES:
        kinds = {s.breakage for s in scored if s.shape == shape}
        assert "none" in kinds and kinds - {"none"}, shape.name
    # a CI break needs a workflow change
    assert all(s.shape.workflow == "wf" for s in scored if s.breakage == "ci")
    assert {s.event for s in scored} == {"push", "pr"}
    assert len({s.branch for s in p}) == len(p)
    with pytest.raises(ValueError, match="both ways"):
        plan(0, scored=2 * len(SHAPES) - 1)


# =============================================================================
# Changes, on a tiny sandbox package
# =============================================================================

CORE = '''\
def inc(x: int) -> int:
    """Next integer."""
    y = x + 1
    return y


def untested(x: int) -> int:
    """Nothing covers this."""
    z = x * 7
    return z
'''
TEXT = '''\
def shout(s: str) -> str:
    """Upper case with a mark."""
    out = s.upper()
    if s:
        out = out + "!"
    return out
'''
WORKFLOW = "name: sandbox\n\npermissions:\n  contents: read\n\njobs:\n  test:\n" \
    "    steps:\n      - uses: actions/checkout@v4\n      - run: uv sync\n" \
    "      - run: uv run ruff check\n"  # fmt: skip


@pytest.fixture
def pkg(tmp_path):
    files = {
        "src/ledger/__init__.py": "",
        "src/ledger/core/__init__.py": "",
        "src/ledger/core/m.py": CORE,
        "src/ledger/text/__init__.py": "",
        "src/ledger/text/t.py": TEXT,
        "tests/test_m.py": "from ledger.core.m import inc\n\n\ndef test_inc():\n"
        "    assert inc(1) == 2\n",
        "tests/test_t.py": "from ledger.text.t import shout\n\n\ndef test_shout():\n"
        "    assert shout('a') == 'A!'\n    assert shout('') == ''\n",
        "pyproject.toml": '[tool.ruff.lint]\nselect = ["E4", "E7", "E9", "F"]\n',
        scenarios.WORKFLOW_FILE: WORKFLOW,
    }
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    return tmp_path


def test_mutation_is_kept_only_if_its_test_fails(pkg):
    m = pkg / "src/ledger/core/m.py"
    for seed in range(5):
        d = scenarios._mutate(m, pkg, random.Random(seed))
        assert int(d["mutation"].rsplit("line ", 1)[1]) <= 4  # inside `inc`
        assert not scenarios.tests_pass(pkg, [d["test"]])
        m.write_text(CORE)


def test_no_caught_mutation_raises_and_restores(pkg):
    m = pkg / "src/ledger/core/m.py"
    m.write_text(CORE.replace("x + 1", "x + 1 if x != 99 else 0"))
    (pkg / "tests/test_m.py").write_text("def test_nothing():\n    pass\n")
    before = m.read_text()
    with pytest.raises(RuntimeError, match="no mutation"):
        scenarios._mutate(m, pkg, random.Random(0))
    assert m.read_text() == before


def test_harmless_edits_change_no_behaviour():
    rng = random.Random(0)
    for name, edit in scenarios.HARMLESS.items():
        new = edit(CORE, rng)
        assert new != CORE, name
        ns: dict = {}
        exec(new, ns)  # noqa: S102
        assert ns["inc"](4) == 5 and ns["untested"](1) == 7, name


@pytest.mark.parametrize("breakage", ["none", "code", "ci"])
def test_apply(pkg, breakage):
    shape = scenarios.Shape("small", "src+tests", "many", "wf")
    sc = scenarios.Scenario(40, "scored", shape, breakage, "push")
    info = scenarios.apply(sc, pkg, seed=1)
    assert scenarios.WORKFLOW_FILE in info["files"]
    assert {f.split("/")[2] for f in info["files"] if f.startswith("src/")} == {
        "core",
        "text",
    }
    assert any(f.startswith("tests/") for f in info["files"])
    assert scenarios.lint_pass(pkg)
    assert scenarios.tests_pass(pkg) == (breakage != "code")
    wf = (pkg / scenarios.WORKFLOW_FILE).read_text()
    assert 'SCENARIO: "40"' in wf
    assert (info["ci_break"] is not None) == (breakage == "ci")
    if breakage == "ci":
        assert wf != scenarios.workflow_harmless(WORKFLOW, 40)


# =============================================================================
# Prediction outside the pool
# =============================================================================


def test_predict_marks_imputed_features_and_out_of_pool(monkeypatch):
    side = {"features": ["a", "b"], "medians": {"a": 1.0, "b": 2.0}, "threshold": 0.5,
            "train_fail_rate": 0.2, "target": "T2", "sources": [{"slug": "o__p__1"}]}  # fmt: skip

    class Est:
        def predict_proba(self, X):
            assert X.to_numpy().tolist() == [[3.0, 2.0]]  # b imputed
            return np.array([[0.3, 0.7]])

    monkeypatch.setattr(live, "load_model", lambda p: (Est(), side))
    df = pd.DataFrame({"a": [3.0], "b": [np.nan], "failed": [0], "prev_failed": [np.nan],
                       "lineage_first_run": [1]})  # fmt: skip
    monkeypatch.setattr(sandbox, "build", lambda cfg, select, extra: (df, {}))
    cfg = TargetConfig(Target("culbec/ACISE", 9, None), False, (), "ci-sandbox")
    run = {"id": 5, "head_sha": "s", "html_url": "u", "created_at": "2026-11-20T00:00:00Z",
           "status": "in_progress", "event": "push"}  # fmt: skip
    r = sandbox.predict(live.MODEL_DIR / "m_T2.skops", run, cfg)
    assert r["predicted"] and r["label"] is None and r["imputed"] == ["b"]
    assert r["in_pool"] is False and r["prev_failed"] is None and not r["majority"]


# =============================================================================
# Report
# =============================================================================


def _row(i, p_fail, prev, predicted_at, breakage="none"):
    return {"repo": "o/r", "run_id": i, "event": "push", "p_fail": p_fail,
            "predicted": p_fail > 0.5, "prev_failed": prev, "majority": False,
            "imputed": ["x"] if i % 2 else [], "model": "cart_pool-1_T2.skops",
            "predicted_at": predicted_at,
            "scenario": {"n": i, "shape": "small-src-one-nowf", "breakage": breakage}}  # fmt: skip


def test_report(tmp_path):
    early, end = "2026-11-20T10:00:00+00:00", "2026-11-20T10:05:00Z"
    rows = [
        _row(1, 0.9, None, early, "code"),  # TP
        _row(2, 0.8, 0, early),  # FP
        _row(3, 0.1, 1, early, "ci"),  # FN, parrot says fail
        _row(4, 0.2, 0, early),  # TN
        _row(5, 0.9, 0, "2026-11-20T10:06:00+00:00", "code"),  # after completion
        _row(6, 0.9, 0, early, "ci"),  # startup failure
        _row(7, 0.1, 0, early),  # still running
    ]
    path = tmp_path / "p.jsonl"
    for r in rows:
        sandbox.record(path, r)
    outcome = {1: "failure", 2: "success", 3: "failure", 4: "success", 5: "failure",
               6: "startup_failure"}  # fmt: skip

    def attempt(repo, rid):
        if rid == 7:
            return {"status": "in_progress", "conclusion": None, "updated_at": end}
        return {"status": "completed", "conclusion": outcome[rid], "updated_at": end}

    rep = sandbox.report(path, attempt)
    assert rep["dropped"] == {"predicted_after_completion": 1,
                              "conclusion_startup_failure": 1, "not_completed": 1}  # fmt: skip
    s = rep["scores"]
    assert s.loc["cart", "n_test"] == 4 and s.loc["cart", "n_fail"] == 2
    assert s.loc["cart", "mcc"] == pytest.approx(0.0)  # 1 TP, 1 FP, 1 FN, 1 TN
    assert s.loc["parrot", "recall"] == 0.5 and s.loc["majority", "recall"] == 0
    assert rep["confusion"].to_numpy().tolist() == [[1, 1], [1, 1]]
    assert rep["by_breakage"].loc["none", "runs"] == 2
    assert (rep["runs"]["lead_secs"] == 300).all()
    assert rep["imputed"].loc["x", "share imputed"] == pytest.approx(4 / 7)
    assert json.loads(path.read_text().splitlines()[0])["run_id"] == 1
