import argparse
from collections import Counter
from datetime import datetime

import pandas as pd
from rich.table import Table

from acise_p2_cidt.collect.gh_runs import (
    EVENTS,
    cached_runs,
    first_attempt,
    raw_dir,
    read_json,
)
from acise_p2_cidt.collect.targets import (
    TargetConfig,
    add_target_args,
    load_targets,
    slug,
    targets_from_args,
)
from acise_p2_cidt.data.build import lineage
from acise_p2_cidt.utils.logging.log import log

# the listed jobs cover at least this share of first-attempt failures
COVERAGE: float = 0.8
# " [<uuid>]" suffix of job names that carry a run id
_RUN_ID = r" \[[0-9a-f-]{36}\]$"


def _ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def _labelled(cfg: TargetConfig) -> list[tuple[dict, dict]]:
    """(latest, first attempt) of supported-event runs whose first attempt passed or failed."""
    out = []
    for r in cached_runs(cfg.target):
        if r["event"] not in EVENTS:
            continue
        first = first_attempt(cfg.target, r)
        if first and first["status"] == "completed" and first["conclusion"] in {
            "success",
            "failure",
        }:
            out.append((r, first))
    return out


def failed_jobs(cfg: TargetConfig) -> pd.DataFrame:
    """
    The failed jobs of every first-attempt failure with collected jobs.

    Args:
        cfg (TargetConfig): target and its `exclude_jobs`.

    Returns:
        pd.DataFrame: one row per (run, failed job): run_id, event, job, failed_step
            (None if no step failed), excluded (`cfg.excludes`). A failure
            without any failed job gets one row with job None, always excluded.
    """
    rows = []
    for r, first in _labelled(cfg):
        p = raw_dir(cfg.target) / "jobs" / f"{r['id']}.json"
        if first["conclusion"] != "failure" or not p.exists():
            continue
        failed = [j for j in read_json(p)["jobs"] if j["conclusion"] == "failure"]
        for j in failed or [{"name": None}]:
            step = next(
                (s["name"] for s in j.get("steps") or [] if s["conclusion"] == "failure"),
                None,
            )
            name = j["name"]
            rows.append(
                {
                    "run_id": r["id"],
                    "event": r["event"],
                    "job": name,
                    "failed_step": step,
                    "excluded": cfg.excludes(name, step),
                }
            )
    return pd.DataFrame(
        rows, columns=["run_id", "event", "job", "failed_step", "excluded"]
    )


def top_jobs(jobs: pd.DataFrame) -> pd.DataFrame:
    """
    (job, failed step) pairs not excluded yet, ranked by failed runs, until the
    union of their runs covers `COVERAGE` of the failures left after exclusion.

    Returns:
        pd.DataFrame: job, failed_step, n_runs, coverage (cumulative).
    """
    left = jobs[~jobs["excluded"].astype(bool)]
    n_failed = left["run_id"].nunique()
    g = (
        left.fillna({"failed_step": "<no failed step>"})
        # servo appends a per-run id to its job names: group on the stable part
        .assign(job=left["job"].str.replace(_RUN_ID, "", regex=True))
        .groupby(["job", "failed_step"])
        .agg(runs=("run_id", set))
    )
    g["n_runs"] = g["runs"].map(len)
    g = g.sort_values("n_runs", ascending=False, kind="stable")
    seen: set[int] = set()
    cov = []
    for runs in g["runs"]:
        if n_failed and len(seen) >= COVERAGE * n_failed:
            break
        seen |= runs
        cov.append(len(seen) / n_failed)
    g = g.iloc[: len(cov)].assign(coverage=cov)
    return g.drop(columns="runs").reset_index()


def t2_failures(runs: list[tuple[dict, dict]], infra: set[int]) -> int:
    """
    Failures whose lineage-previous run passed or does not exist. The
    previous run is the latest-created run of the lineage completed before this
    run was created. Infra-only failures are not labels and not history.
    """
    by_lineage: dict[str, list[tuple[float, float, int]]] = {}
    for r, first in runs:
        if r["id"] in infra:
            continue
        by_lineage.setdefault(lineage(r), []).append(
            (_ts(r["created_at"]), _ts(first["updated_at"]),
             int(first["conclusion"] == "failure"))
        )  # fmt: skip
    n = 0
    for lin in by_lineage.values():
        lin.sort()
        for i, (created, _, failed) in enumerate(lin):
            if not failed:
                continue
            prev = next((f for c, e, f in reversed(lin[:i]) if e < created), 0)
            n += prev == 0
    return n


def summary(cfg: TargetConfig) -> dict:
    """
    Label statistics of a target's cached runs.

    Args:
        cfg (TargetConfig): target and its `exclude_jobs`.

    Returns:
        dict: n_runs, runs_per_event (every cached run), n_labelled, failure_rate
            (first attempt, before exclusion), n_failed, n_no_jobs (jobs not
            collected), n_no_failed_job (infra), infra_share,
            t2_failures, t2_share (of the failures left after exclusion),
            flaky_rate, top_jobs (DataFrame).
    """
    all_runs = cached_runs(cfg.target)
    runs = _labelled(cfg)
    failed = [r["id"] for r, f in runs if f["conclusion"] == "failure"]
    jobs = failed_jobs(cfg)
    excl = jobs.groupby("run_id")["excluded"].all()
    infra = set(excl[excl].index)
    n_flaky = sum(
        f["conclusion"] == "failure" and r["conclusion"] == "success" for r, f in runs
    )
    t2 = t2_failures(runs, infra)
    kept = len(failed) - len(infra)
    return {
        "n_runs": len(all_runs),
        "runs_per_event": dict(Counter(r["event"] for r in all_runs).most_common()),
        "n_labelled": len(runs),
        "n_failed": len(failed),
        "failure_rate": len(failed) / len(runs) if runs else float("nan"),
        "n_no_jobs": len(set(failed) - set(jobs["run_id"])),
        "n_no_failed_job": int(jobs["job"].isna().sum()),
        "infra_share": len(infra) / len(failed) if failed else float("nan"),
        "t2_failures": t2,
        "t2_share": t2 / kept if kept else float("nan"),
        "flaky_rate": n_flaky / len(failed) if failed else float("nan"),
        "top_jobs": top_jobs(jobs),
    }


def _print(name: str, s: dict) -> None:
    t = Table(title=f"{name}: top failed jobs left after exclusion (first attempt)")
    for c in ["job", "failed step", "runs", "coverage"]:
        t.add_column(c)
    for row in s["top_jobs"].itertuples():
        t.add_row(row.job, row.failed_step, str(row.n_runs), f"{row.coverage:.0%}")
    log.print(t)
    log.print({k: v for k, v in s.items() if k != "top_jobs"})


def main() -> None:
    p = argparse.ArgumentParser(description="Failed jobs and label statistics.")
    add_target_args(p)
    p.add_argument("--all", action="store_true", help="include disabled targets")
    a = p.parse_args()
    if a.all:
        cfgs = load_targets(a.targets, enabled_only=False)
    else:
        cfgs = targets_from_args(a)
    for cfg in cfgs:
        _print(slug(cfg.target), summary(cfg))


if __name__ == "__main__":
    main()
