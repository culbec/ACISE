import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from acise_p2_cidt.collect import gh_runs
from acise_p2_cidt.collect.gh_runs import Target
from acise_p2_cidt.data import build, load
from acise_p2_cidt.data.build import history_features

H = 3600.0

# (created, completed, failed). Run 1 is long: still running when runs 2 and 3
# are created, and it completes before run 4.
TOY = [
    (0, 100, 0),
    (200, 1000, 1),
    (300, 400, 0),
    (500, 600, 1),
    (4200, 4300, 0),
]


def toy(rows=TOY):
    df = pd.DataFrame(rows, columns=["created_at", "completed_at", "failed"]).astype(
        float
    )
    df["failed"] = df["failed"].astype(int)
    n = len(df)
    return df.assign(
        run_id=range(n),
        head_sha=[f"{i:040x}" for i in range(n)],
        author_id=["a", "b", "a", "a", "b"][:n],
        latest_completed_at=df["completed_at"],
        flaky=0,
        duration=df["completed_at"] - df["created_at"],
        queue_delay=0.0,
        n_failed_jobs=df["failed"].astype(float),
    )


def test_previous_means_completed_before():
    h = history_features(toy())
    assert np.isnan(h.loc[0, "prev_failed"])
    assert h.loc[2, "prev_failed"] == 0  # from run 0: run 1 is still running
    assert h.loc[3, "prev_failed"] == 0  # from run 2
    assert h.loc[4, "prev_failed"] == 1  # from run 3


def test_hand_computed():
    h = history_features(toy())
    assert h.loc[2, "concurrent_runs"] == 1  # run 1 pending at t=300
    assert h.loc[3, "concurrent_runs"] == 1
    assert h.loc[3, "ci_load_1h"] == 3  # runs 0, 1, 2 created in [500-3600, 500)
    assert h.loc[4, "ci_load_1h"] == 0  # nothing created in [600, 4200)
    # run 4 knows runs 0..3 with outcomes 0,1,0,1 (creation order)
    assert h.loc[4, "fail_streak"] == 1
    assert h.loc[4, "fail_rate_last_5"] == pytest.approx(0.5)  # 2 of 4
    # completed in [600, 4200): run 3 (600) and run 1 (1000), both failed
    assert h.loc[4, "fail_rate_1h"] == pytest.approx(1.0)
    assert h.loc[4, "runs_since_last_fail"] == 0
    assert h.loc[4, "secs_since_last_fail"] == pytest.approx(4200 - 600)
    assert h.loc[1, "fail_rate_last_5"] == 0  # knows run 0 (passed)
    assert np.isnan(h.loc[0, "fail_rate_last_5"])  # no known run: NaN, not 0
    # author "a" at run 3 knows runs 0 and 2 (both passed)
    assert (h.loc[3, "author_prior_runs"], h.loc[3, "author_prior_fail_rate"]) == (2, 0)


def same(a, b):
    return a.fillna(-999).equals(b.fillna(-999))


def test_no_look_ahead():
    base = history_features(toy())
    # Run 1 is still running when run 2 starts: its label must not reach run 2.
    flipped = [list(r) for r in TOY]
    flipped[1][2] = 0
    assert same(history_features(toy(flipped)).loc[2], base.loc[2])
    # Changing any later run never changes an earlier row.
    for i in range(1, len(TOY)):
        later = [list(r) for r in TOY]
        later[i][2] = 1 - later[i][2]
        later[i][1] += 5  # and its completion time
        assert same(history_features(toy(later)).iloc[:i], base.iloc[:i]), i


# =============================================================================
# End to end on a tiny repository and a fake raw cache
# =============================================================================

T = Target("owner/name", 1, "main")
EMAIL, LOGIN = "alice@example.com", "alice-login"


def _git(repo, *args, env=None):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()


def _iso(s):
    return pd.Timestamp(s, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture
def raw(tmp_path, monkeypatch):
    monkeypatch.setattr(gh_runs, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(build, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(load, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(load, "SCHEMA_DIR", tmp_path / "results")
    d = gh_runs.raw_dir(T)
    repo = d / "repo.git"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", EMAIL)
    _git(repo, "config", "user.name", "Alice")
    shas, t0 = [], 1_767_225_600  # 2026-01-01
    for i in range(4):
        (repo / "src").mkdir(exist_ok=True)
        (repo / "src" / "a.py").write_text(f"{i}\n" * (i + 1))
        if i == 2:
            (repo / "tests").mkdir()
            (repo / "tests" / "test_a.py").write_text("x\n")
        _git(repo, "add", "-A")
        date = f"@{t0 + i * 1000}"
        env = {
            "GIT_AUTHOR_DATE": date,
            "GIT_COMMITTER_DATE": date,
            "PATH": "/usr/bin:/bin",
        }
        _git(repo, "commit", "-q", "-m", f"c{i}", env=env)
        shas.append(_git(repo, "rev-parse", "HEAD"))

    def run(i, attempt=1, conclusion="success", status="completed", end=300):
        c = t0 + i * 1000 + 10
        return {
            "id": 100 + i, "run_attempt": attempt, "status": status, "conclusion": conclusion,
            "head_sha": shas[i], "created_at": _iso(c), "run_started_at": _iso(c + 5),
            "updated_at": _iso(c + end), "path": ".github/workflows/ci.yml",
            "actor": {"login": LOGIN, "type": "User"},
            "head_commit": {"timestamp": _iso(c - 10), "author": {"email": EMAIL}},
        }  # fmt: skip

    runs = [
        run(0, end=400 * 86400),  # updated_at bumped by GitHub's clean-up
        run(1, attempt=2),
        run(2, conclusion="failure", end=400 * 86400),
        run(3, conclusion="cancelled"),
    ]
    gh_runs._write_json(
        d / "runs" / "2026-01-01_2026-01-31.json",
        {"total_count": 4, "workflow_runs": runs},
    )
    # run 1 failed first and passed on the re-run
    gh_runs._write_json(d / "attempts" / "101.json", run(1, conclusion="failure"))
    gh_runs._write_json(
        d / "jobs" / "101.json", {"jobs": [{"name": "unit", "conclusion": "failure"}]}
    )
    job = {"name": "unit", "conclusion": "failure", "completed_at": _iso(t0 + 2410)}
    gh_runs._write_json(d / "jobs" / "102.json", {"jobs": [job]})
    return d


def test_cleanup_updated_at_is_not_a_completion(raw):
    runs, _ = build.runs_frame(T, b"salt")
    c0 = 1_767_225_600 + 10
    # run 2 ends with its last job: 2410 - 2015 (started) = 395 s
    assert runs.loc[2, "duration"] == pytest.approx(395)
    assert runs.loc[2, "completed_at"] == c0 + 2400
    # run 0 has no jobs: duration unknown, completion = created + median run length
    # (runs 1 and 2 complete 300 s and 400 s after creation -> median 350 s)
    assert np.isnan(runs.loc[0, "duration"])
    assert runs.loc[0, "completed_at"] == c0 + 350
    assert runs.loc[0, "latest_completed_at"] == runs.loc[0, "completed_at"]


def test_build_end_to_end(raw):
    df, schema = build.build(T)
    assert df["failed"].tolist() == [0, 1, 1]
    assert df["flaky"].tolist() == [0, 1, 0]  # first attempt kept as the label
    assert schema["n_dropped"] == {"cancelled": 1}
    assert df.loc[1, "failed_job_names"] == "unit"
    assert df["author_is_bot"].eq(0).all()
    assert df.loc[2, "prev_failed_jobs"] == 1  # run 1 is known by then
    assert df.loc[2, "test_files"] == 1 and df.loc[2, "src_files"] == 1
    assert df.loc[1, "file_prior_changes"] == 1  # src/a.py changed once before
    assert df.loc[2, "commits_since_prev_run"] == 1
    assert df.loc[2, "file_fail_prob_max"] == pytest.approx(0.5)  # runs 0 and 1 on a.py
    cells = [str(c) for c in df.to_numpy().ravel()]
    assert not any(EMAIL in c or LOGIN in c for c in cells)  # pseudonymised
    feats = set(schema["feature_cols"])
    assert not feats & {
        *schema["id_cols"],
        *schema["diagnostic_cols"],
        "created_at",
        "failed",
    }

    out = build.write(df, schema, T)
    assert json.loads(load.schema_path(T.repo).read_text())["n_runs"] == 3
    assert load.available_repos() == [T.repo]
    ds = load.load(T.repo, "T1")
    assert list(ds.X.columns) == schema["feature_cols"] and out.exists()
