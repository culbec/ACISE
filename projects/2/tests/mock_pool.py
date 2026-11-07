"""
Synthetic data with the shapes of the real pipeline. Each unit
extends it; so far it holds the mock targets, their registry, manifests and
raw API caches.
"""

import json
import pathlib
from datetime import date

from acise_p2_cidt.collect import gh_runs, targets
from acise_p2_cidt.collect.targets import Target, slug

# two workflows of one repo (one restricted to a branch) + one other repo
TARGETS: list[Target] = [
    Target("acme/app", 11, None),
    Target("acme/app", 12, "main"),
    Target("other/lib", 21, None),
]
SOURCES: list[str] = [slug(t) for t in TARGETS]


def write_registry(path: pathlib.Path, enabled: bool = True) -> pathlib.Path:
    """`targets.json` holding the mock targets."""
    entries = [
        {"repo": t.repo, "workflow_id": t.workflow_id, "branch": t.branch,
         "enabled": enabled, "exclude_jobs": ["^Result$"]}
        for t in TARGETS
    ]  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries))
    return path


def write_manifest(t: Target, since: date | None, until: date | None) -> pathlib.Path:
    """A collection manifest as `gh_runs.collect` writes it; a stub for `collect`."""
    n = 50 + 10 * t.workflow_id
    manifest = {
        "target": t._asdict(), "since": str(since), "until": str(until),
        "collected_at": "2026-11-05T00:00:00+00:00",
        "windows": {"2026-10-01_2026-10-31": n - 5, "2026-11-01_2026-11-04": 5},
        "n_runs": n, "runs_per_event": {"pull_request": n - 12, "push": 10, "schedule": 2},
        "api_calls": n // 4, "queries": [],
    }  # fmt: skip
    out = gh_runs.RESULTS_DIR / f"manifest_{slug(t)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest))
    return out


def write_raw_cache(t: Target) -> None:
    """
    A tiny API cache: a pass, a code failure, an infra-only failure (`Result`, the
    registry's exclude pattern), a failure without failed jobs, and a schedule run.
    """
    def run(i, event, conclusion, minute):
        ts = f"2026-10-01T00:{minute:02d}:00Z"
        return {"id": t.workflow_id * 100 + i, "event": event, "head_branch": "main",
                "status": "completed", "conclusion": conclusion, "run_attempt": 1,
                "created_at": ts, "updated_at": ts, "pull_requests": []}  # fmt: skip

    def job(name, step):
        return {"name": name, "conclusion": "failure",
                "steps": [{"name": step, "conclusion": "failure"}]}  # fmt: skip

    runs = [run(1, "push", "success", 0), run(2, "push", "failure", 10),
            run(3, "push", "failure", 20), run(4, "pull_request", "failure", 30),
            run(5, "schedule", "failure", 40)]  # fmt: skip
    jobs = {runs[1]["id"]: [job("test", "Run tests")],
            runs[2]["id"]: [job("Result", "Failure")], runs[3]["id"]: []}  # fmt: skip
    d = gh_runs.raw_dir(t)
    (d / "runs").mkdir(parents=True, exist_ok=True)
    (d / "runs" / "2026-10-01_2026-10-31.json").write_text(
        json.dumps({"total_count": len(runs), "workflow_runs": runs})
    )
    (d / "jobs").mkdir(exist_ok=True)
    for rid, js in jobs.items():
        (d / "jobs" / f"{rid}.json").write_text(json.dumps({"jobs": js}))


def mock_pool(tmp: pathlib.Path, monkeypatch) -> list[str]:
    """Patches every data path into `tmp`, writes the registry; returns SOURCES."""
    monkeypatch.setattr(targets, "REGISTRY", tmp / "config" / "targets.json")
    monkeypatch.setattr(gh_runs, "RAW_ROOT", tmp / "raw")
    monkeypatch.setattr(gh_runs, "RESULTS_DIR", tmp / "results" / "collection")
    write_registry(targets.REGISTRY)
    for t in TARGETS:
        write_raw_cache(t)
    return SOURCES
