import json

import pytest

from acise_p2_cidt.analysis import labels
from acise_p2_cidt.collect import gh_runs
from acise_p2_cidt.collect.targets import Target, TargetConfig

T = Target("acme/app", 1)


def run(i, created, completed, conclusion, branch="main", event="push", attempt=1):
    ts = lambda s: f"2026-10-01T00:{s // 60:02d}:{s % 60:02d}Z"
    return {"id": i, "event": event, "head_branch": branch, "status": "completed",
            "conclusion": conclusion, "run_attempt": attempt,
            "created_at": ts(created), "updated_at": ts(completed)}  # fmt: skip


def job(name, step=None):
    steps = [{"name": "Set up job", "conclusion": "success"}]
    if step:
        steps.append({"name": step, "conclusion": "failure"})
    return {"name": name, "conclusion": "failure", "steps": steps}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(gh_runs, "RAW_ROOT", tmp_path)
    d = gh_runs.raw_dir(T)

    def write(runs, jobs, attempts=()):
        (d / "runs").mkdir(parents=True)
        (d / "runs" / "w.json").write_text(json.dumps({"workflow_runs": runs}))
        for sub, items in [("jobs", jobs.items()), ("attempts", attempts)]:
            (d / sub).mkdir()
            for k, v in items:
                obj = {"jobs": v} if sub == "jobs" else v
                (d / sub / f"{k}.json").write_text(json.dumps(obj))

    return write


def test_t2_follows_lineage_and_drops_infra(cache):
    # main: pass, fail(code), fail(code) -> only run 2 is T2.
    # feat: run 4 is infra-only (dropped), so run 5 has no previous: T2. Run 6 is
    # created before run 5 completes, so it has no previous either: T2.
    cache(
        [run(1, 0, 10, "success"), run(2, 20, 30, "failure"),
         run(3, 40, 50, "failure"), run(4, 0, 5, "failure", "feat"),
         run(5, 10, 70, "failure", "feat"), run(6, 65, 80, "failure", "feat"),
         run(7, 0, 5, "failure", event="schedule")],
        {2: [job("test", "Run tests")], 3: [job("test", "Run tests")],
         4: [job("Result")], 5: [job("lint", "Run lint"), job("Result")],
         6: [job("lint", "Run lint")]},
    )  # fmt: skip
    s = labels.summary(TargetConfig(T, True, ("^Result$",)))
    assert s["n_runs"] == 7 and s["runs_per_event"] == {"push": 6, "schedule": 1}
    assert s["n_failed"] == 5 and s["infra_share"] == pytest.approx(1 / 5)
    assert s["t2_failures"] == 3 and s["t2_share"] == pytest.approx(3 / 4)


def test_first_attempt_label_and_flaky(cache):
    # run 1 re-run to success: first attempt failed, so it is a flaky failure
    cache(
        [run(1, 0, 10, "success", attempt=2), run(2, 20, 30, "success")],
        {1: [job("test", "Run tests")]},
        [(1, run(1, 0, 5, "failure"))],
    )
    s = labels.summary(TargetConfig(T, True, ()))
    assert s["n_failed"] == 1 and s["flaky_rate"] == 1.0 and s["infra_share"] == 0


def test_top_jobs_stop_at_coverage(cache):
    # 10 failures: 6 in "a", 2 in "b", 1 in "c", 1 without a failed job (infra,
    # so 9 are left and "a" + "b" cover 8 of them)
    runs = [run(i, i * 10, i * 10 + 5, "failure") for i in range(10)]
    jobs = {i: [job("a", "x")] for i in range(6)}
    jobs |= {6: [job("b", "y")], 7: [job("b", "y")], 8: [job("c")], 9: []}
    cache(runs, jobs)
    s = labels.summary(TargetConfig(T, True, ()))
    top = s["top_jobs"]
    assert list(top["job"]) == ["a", "b"] and list(top["n_runs"]) == [6, 2]
    assert top["coverage"].iloc[-1] == pytest.approx(8 / 9)
    assert s["n_no_failed_job"] == 1 and s["infra_share"] == pytest.approx(0.1)
    assert set(labels.failed_jobs(TargetConfig(T, True, ()))["failed_step"].dropna()) == {"x", "y"}


def test_exclude_by_job_or_step():
    cfg = TargetConfig(T, True, ("^Result$", "^check_ee :: Run actions/checkout"))
    assert cfg.excludes("Result", "Failure")
    assert cfg.excludes("check_ee", "Run actions/checkout@v4")
    assert not cfg.excludes("check_ee", "cargo check")
    assert not cfg.excludes("check_ee", None)
    assert cfg.excludes(None, None)  # no failed job: infra


def test_step_exclusion_keeps_code_failures(cache):
    # check_ee fails on checkout (infra) in run 1 and on cargo check in run 2
    cache(
        [run(1, 0, 5, "failure"), run(2, 10, 15, "failure")],
        {1: [job("check_ee", "Run actions/checkout@v4")],
         2: [job("check_ee", "cargo check")]},
    )  # fmt: skip
    s = labels.summary(TargetConfig(T, True, ("^check_ee :: Run actions/checkout",)))
    assert s["infra_share"] == 0.5
    assert list(s["top_jobs"]["failed_step"]) == ["cargo check"]


def test_empty_cache(cache):
    cache([], {})
    s = labels.summary(TargetConfig(T, True, ()))
    assert s["n_failed"] == 0 and s["top_jobs"].empty


def test_top_jobs_group_run_id_suffix():
    uid = "7cb18c8e-5ea6-4666-860c-5274a605df5a"
    jobs = labels.pd.DataFrame(
        {"run_id": [1, 2], "event": "push", "job": [f"Lint [{uid}]", f"Lint [{uid[::-1]}]"],
         "failed_step": "Tidy", "excluded": False}
    )  # fmt: skip
    top = labels.top_jobs(jobs)
    assert list(top["job"]) == ["Lint"] and list(top["n_runs"]) == [2]
