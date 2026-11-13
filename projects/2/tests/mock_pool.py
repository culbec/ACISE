"""
Synthetic data with the shapes of the real pipeline: the mock targets, their
registry, manifests, raw API caches and built tables.
"""

import json
import pathlib
from datetime import date

import numpy as np
import pandas as pd

from acise_p2_cidt.analysis import features
from acise_p2_cidt.collect import gh_runs, targets
from acise_p2_cidt.collect.targets import Target, slug
from acise_p2_cidt.data import build, load
from acise_p2_cidt.validate import live
from acise_p2_cidt.viz import plots

# the real build's feature list; refresh when the build's features change
FEATURE_COLS: list[str] = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "feature_cols.json").read_text()
)
# the label depends on these, so models can learn something
SIGNAL: tuple[str, ...] = ("churn", "fail_rate_24h", "file_fail_prob_max")

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


def mock_table(s: str, n: int = 800) -> tuple[pd.DataFrame, dict]:
    """
    A built table and its schema, as `build.build` returns them: runs of three
    events over six months (overlapping across sources), several PR lineages, a
    fork share, re-runs and flaky failures. A stub for `build`.
    """
    k = SOURCES.index(s) if s in SOURCES else 0
    rng = np.random.default_rng(k)
    n = n + 200 * k
    start = pd.Timestamp("2026-05-01", tz="UTC")
    created = start + pd.to_timedelta(np.sort(rng.uniform(0, 182 * 86400, n)), unit="s")
    event = rng.choice(["push", "pull_request", "merge_group"], n, p=[0.3, 0.6, 0.1])
    pr = rng.integers(0, n // 6, n)
    fork = rng.random(n) < 0.1
    lineage = np.where(
        event == "push", "branch:main",
        np.where(event == "pull_request",
                 [f"head:{'someone' if f else 'acme'}/app:f{p}" for f, p in zip(fork, pr)],
                 [f"pr:{p}" for p in pr]),
    )  # fmt: skip
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLS))), columns=FEATURE_COLS)
    logit = -1.6 + 0.3 * k + df[list(SIGNAL)].sum(axis=1).to_numpy()
    if k == 0:  # a month of a failing CI leg: a failure-rate shift
        logit += 3.0 * (created.month == 7)
    failed = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    for e in ("push", "pull_request", "merge_group"):
        df[f"event_{e}"] = (event == e).astype(int)
    df["from_fork"] = (fork & (event == "pull_request")).astype(int)
    df["is_default_branch"] = (event == "push").astype(int)
    lin = pd.Series(lineage)
    df["lineage_prior_runs"] = lin.groupby(lin).cumcount()
    df["lineage_first_run"] = (df["lineage_prior_runs"] == 0).astype(int)
    df[load.PREV_LABEL_COL] = pd.Series(failed).groupby(lin).shift()
    df.loc[rng.random(n) < 0.05, "files_changed"] = np.nan  # unreachable heads
    rerun = rng.random(n) < 0.05
    flaky = rerun & (failed == 1) & (rng.random(n) < 0.5)
    ids = pd.DataFrame({
        "run_id": np.arange(n) + 10_000 * k, "head_sha": [f"{i:040x}" for i in range(n)],
        "author_id": rng.choice(["a1", "a2", "a3"], n), "lineage": lineage,
        "created_at": created, "failed": failed,
        "event": event, "was_rerun": rerun.astype(int), "flaky": flaky.astype(int),
        "conclusion_latest": np.where(failed & ~flaky, "failure", "success"),
        "failed_job_names": np.where(failed == 1, "test (linux)", ""),
    })  # fmt: skip
    df = pd.concat([ids, df], axis=1)
    schema = {
        "target_slug": s, "collected_at": "2026-11-05T00:00:00+00:00",
        **TARGETS[k]._asdict(),  # repo, workflow_id, branch
        "n_runs": n, "n_failed": int(failed.sum()),
        "n_dropped": {"cancelled": 7 + k, "infra_failure": 3, "event_schedule": 2},
        "exclude_jobs": ["^Result$"], "events": ["merge_group", "pull_request", "push"],
        "order_col": load.ORDER_COL, "id_cols": build.ID_COLS, "label_col": load.LABEL_COL,
        "prev_label_col": load.PREV_LABEL_COL, "nominal": [],
        "diagnostic_cols": build.DIAGNOSTIC_COLS, "feature_cols": FEATURE_COLS,
    }  # fmt: skip
    return df, schema


def mock_pool(tmp: pathlib.Path, monkeypatch) -> list[str]:
    """
    Patches every data path into `tmp` and `plots.save` to write nothing, writes
    the registry and raw caches; returns SOURCES.
    """
    monkeypatch.setattr(targets, "REGISTRY", tmp / "config" / "targets.json")
    monkeypatch.setattr(gh_runs, "RAW_ROOT", tmp / "raw")
    monkeypatch.setattr(gh_runs, "RESULTS_DIR", tmp / "results" / "collection")
    monkeypatch.setattr(build, "RESULTS_DIR", tmp / "results" / "collection")
    monkeypatch.setattr(load, "PROCESSED_DIR", tmp / "processed")
    monkeypatch.setattr(load, "SCHEMA_DIR", tmp / "results" / "collection")
    monkeypatch.setattr(features, "OUT_DIR", tmp / "results" / "features")
    monkeypatch.setattr(live, "MODEL_DIR", tmp / "results" / "model")
    monkeypatch.setattr(plots, "save", lambda fig, name, **k: fig)
    write_registry(targets.REGISTRY)
    for t in TARGETS:
        write_raw_cache(t)
    return SOURCES


def write_tables() -> None:
    """The built table and schema of every mock target, as `build-dataset` writes them."""
    for t in TARGETS:
        build.write(*mock_table(slug(t)), t)
