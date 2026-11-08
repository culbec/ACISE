import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from acise_p2_cidt.collect import gh_runs
from acise_p2_cidt.collect.targets import Target, TargetConfig, slug
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
    repo = gh_runs.repo_root(T.repo) / "repo.git"
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
            "event": "push", "head_branch": "main", "head_sha": shas[i], "created_at": _iso(c), "run_started_at": _iso(c + 5),
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
    assert json.loads(load.schema_path(slug(T)).read_text())["n_runs"] == 3
    assert load.available_targets() == [slug(T)]
    ds = load.load(slug(T), "T1")
    assert list(ds.X.columns) == schema["feature_cols"] and out.exists()


@pytest.mark.parametrize(
    ("run", "expected"),
    [
        ({"event": "push", "head_branch": "main"}, "branch:main"),
        ({"event": "pull_request", "head_branch": "fix", "pull_requests": [{"number": 7}],
          "head_repository": {"full_name": "acme/app"}}, "head:acme/app:fix"),
        ({"event": "pull_request", "head_branch": "fix", "pull_requests": [],
          "head_repository": {"full_name": "someone/app"}}, "head:someone/app:fix"),
        ({"event": "merge_group",
          "head_branch": "gh-readonly-queue/main/pr-7-0123abcd"}, "pr:7"),
    ],
)  # fmt: skip
def test_lineage(run, expected):
    assert build.lineage(run) == expected


def _extra(i, event="push", conclusion="failure", **kw):
    """A raw run added through `extra`, based on the first cached run."""
    r = gh_runs.cached_runs(T)[0]
    r |= {"id": 200 + i, "event": event, "conclusion": conclusion,
          "updated_at": r["run_started_at"]}  # fmt: skip
    return r | kw


def _jobs(d, rid, *jobs):
    gh_runs._write_json(d / "jobs" / f"{rid}.json", {"jobs": [
        {"name": n, "conclusion": "failure", "steps": [{"name": s, "conclusion": "failure"}]}
        for n, s in jobs
    ]})  # fmt: skip


def test_drops_infra_and_unsupported_events(raw):
    cfg = TargetConfig(T, True, ("^Result$", r"^unit :: Install"))
    _jobs(raw, 201, ("Result", "Failure"))  # infra only
    _jobs(raw, 202, ("Result", "Failure"), ("unit", "Run tests"))  # mixed: kept
    _jobs(raw, 203, ("unit", "Install deps"))  # infra by its failed step
    _jobs(raw, 204)  # no failed job: failed before testing the code
    # 206 has no jobs file: kept, as its jobs were not collected
    extra = [
        _extra(1),
        _extra(2),
        _extra(3),
        _extra(4),
        _extra(5, "schedule"),
        _extra(6),
    ]
    runs, dropped = build.runs_frame(cfg, b"salt", extra)
    assert set(runs["run_id"]) & {201, 202, 203, 204, 205, 206} == {202, 206}
    assert dropped == {"cancelled": 1, "infra_failure": 3, "event_schedule": 1,
                       "no_jobs_kept": 1}  # fmt: skip
    assert runs.set_index("run_id").loc[202, "failed_job_names"] == "Result|unit"


def test_event_columns_and_fork(raw):
    fork = _extra(1, "pull_request", "success", head_branch="fix",
                  head_repository={"full_name": "Someone/Name"})  # fmt: skip
    same = _extra(2, "pull_request", "success", head_branch="fix2",
                  head_repository={"full_name": "OWNER/name"})  # fmt: skip
    runs, _ = build.runs_frame(T, b"salt", [fork, same])
    r = runs.set_index("run_id")
    assert r.loc[
        201, ["event_pull_request", "event_push", "event_merge_group"]
    ].tolist() == [1, 0, 0]
    assert (
        r.loc[201, "from_fork"],
        r.loc[202, "from_fork"],
        r.loc[100, "from_fork"],
    ) == (1, 0, 0)
    assert r.loc[201, "lineage"] == "head:Someone/Name:fix"


def test_pr_change_is_the_whole_branch(raw):
    repo = gh_runs.repo_root(T.repo) / "repo.git"
    shas = _git(repo, "rev-list", "--reverse", "main").split()
    _git(repo, "checkout", "-q", "-b", "feat", shas[1])
    for name, text in (("b.py", "1\n2\n3\n"), ("c.py", "x\n")):
        (repo / name).write_text(text)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", name)
    head = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "main")
    created = 1_767_225_600 + 3500  # after main's c3: the base as of the run is c3
    pr = _extra(10, "pull_request", "success", head_sha=head, head_branch="feat",
                pull_requests=[], head_repository={"full_name": "owner/name"},
                created_at=_iso(created), run_started_at=_iso(created + 5),
                updated_at=_iso(created + 100))  # fmt: skip
    # a second, passing run of the same PR once the first (failed) one is known
    again = pr | {
        "id": 211,
        "conclusion": "success",
        "created_at": _iso(created + 1000),
        "run_started_at": _iso(created + 1005),
        "updated_at": _iso(created + 1100),
    }
    df, schema = build.build(T, extra=[pr | {"conclusion": "failure"}, again])
    row = df.set_index("run_id").loc[210]
    # merge-base(c3, head) = c1: the diff covers both branch commits, nothing of main
    assert (row["files_changed"], row["additions"], row["deletions"]) == (2, 4, 0)
    assert row["file_prior_changes"] == 0  # both files are new
    assert row["lineage_first_run"] == 1 and np.isnan(row["prev_failed"])
    # the PR's own files, not the default branch's, feed the file history
    nxt = df.set_index("run_id").loc[211]
    assert nxt["prev_failed"] == 1 and nxt["file_fail_prob_max"] == 1.0
    assert nxt["overlap_with_last_failed_run"] == 1.0
    assert (
        row["is_default_branch"] == 0
        and df.set_index("run_id").loc[100, "is_default_branch"] == 1
    )
    assert {"event_pull_request", "from_fork", "lineage_first_run"} <= set(
        schema["feature_cols"]
    )
    assert "lineage" in schema["id_cols"] and "event" in schema["diagnostic_cols"]
    assert schema["target_slug"] == slug(T) and schema["events"] == sorted(
        gh_runs.EVENTS
    )
