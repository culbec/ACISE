import argparse
import hashlib
import json
import os
import pathlib
import re
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime

import numpy as np
import pandas as pd

from acise_p2_cidt.collect import git_changes as git
from acise_p2_cidt.collect.gh_runs import (
    EVENTS,
    RESULTS_DIR,
    cached_runs,
    first_attempt,
    raw_dir,
    read_json,
    repo_root,
)
from acise_p2_cidt.collect.targets import (
    Target,
    TargetConfig,
    add_target_args,
    slug,
    targets_from_args,
)
from acise_p2_cidt.data.history import git_features, history_features, run_changes
from acise_p2_cidt.data.load import (
    LABEL_COL,
    ORDER_COL,
    PREV_LABEL_COL,
    schema_path,
    table_path,
)
from acise_p2_cidt.utils.logging.log import get_logger, log

# Every other conclusion (cancelled, skipped, ...) is dropped and counted.
KEEP: frozenset[str] = frozenset({"success", "failure"})
# GitHub bumps `updated_at` of a run when it cleans it up (~400 days after it ran),
# so a run "lasting" longer than this was touched later, not completed then.
MAX_RUN_SECS: float = 24 * 3600

ID_COLS: list[str] = ["run_id", "head_sha", "author_id", "lineage"]
DIAGNOSTIC_COLS: list[str] = [
    "event",
    "was_rerun",
    "flaky",
    "conclusion_latest",
    "failed_job_names",
]

logger = get_logger(__name__)


# =============================================================================
# Identity
# =============================================================================


def hash_id(value: str, salt: bytes) -> str:
    """
    Salted SHA-256 pseudonym of a login or e-mail; the raw value is never saved.

    Args:
        value (str): login or e-mail.
        salt (bytes): per-repository secret salt.

    Returns:
        str: first 16 hex digits of the digest.
    """
    return hashlib.sha256(salt + value.lower().encode()).hexdigest()[:16]


def _salt(t: Target) -> bytes:
    # Lives in data/raw (git-ignored), never next to the results. One per
    # repository, so an author gets the same id in every target of it.
    p = repo_root(t.repo) / "salt.bin"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(os.urandom(16))
    return p.read_bytes()


def _ts(s: str | None) -> float:
    return datetime.fromisoformat(s).timestamp() if s else np.nan


# merge-queue head branch: gh-readonly-queue/<base>/pr-<n>-<base sha>
_QUEUE = re.compile(r"^gh-readonly-queue/.+/pr-(\d+)-([0-9a-f]+)$")


def lineage(run: dict) -> str:
    """
    The run's lineage: the PR or branch whose previous run it follows.

    PR runs are keyed by their head branch, not the PR number: the API fills
    `pull_requests` only while a PR is open, so the number is missing for most runs.

    Args:
        run (dict): raw run object.

    Returns:
        str: `head:<head repo>:<branch>` for PR runs, `pr:<n>` for merge-queue runs
            (one lineage per queued PR), `branch:<name>` for pushes and the rest.
    """
    event, branch = run["event"], run.get("head_branch") or ""
    if event == "pull_request":
        head = (run.get("head_repository") or {}).get("full_name", "")
        return f"head:{head}:{branch}"
    if event == "merge_group" and (m := _QUEUE.match(branch)):
        return f"pr:{m[1]}"
    return f"branch:{branch}"


def _pr_base(run: dict) -> str | None:
    """The base commit the API names for a PR or merge-queue run, if any."""
    if prs := run.get("pull_requests"):
        return prs[0]["base"]["sha"]
    if run["event"] == "merge_group" and (
        m := _QUEUE.match(run.get("head_branch") or "")
    ):
        return m[2]
    return None


def _failed_step(job: dict) -> str | None:
    return next(
        (s["name"] for s in job.get("steps") or [] if s["conclusion"] == "failure"),
        None,
    )


# =============================================================================
# Stage 1: runs
# =============================================================================


def runs_frame(
    cfg: TargetConfig | Target, salt: bytes, extra: list[dict] | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    One row per run with its first-attempt label and times, oldest first.

    Dropped and counted: runs of events that carry no code change, runs not passed
    or failed on the first attempt, and failures whose failed jobs all match
    `exclude_jobs` (or that have no failed job): they did not come from the code.
    A failure without collected jobs is kept and counted as `no_jobs_kept`.

    Args:
        cfg (TargetConfig | Target): collection target (reads its raw cache); a bare
            `Target` excludes no job.
        salt (bytes): salt for `hash_id`.
        extra (list[dict] | None): raw runs not in the cache; they replace a cached
            run with the same id.

    Returns:
        tuple[pd.DataFrame, dict[str, int]]: runs kept, and dropped counts per reason.
    """
    if isinstance(cfg, Target):
        cfg = TargetConfig(cfg, True, ())
    t = cfg.target
    rows, dropped = [], defaultdict(int)
    raw = {r["id"]: r for r in [*cached_runs(t), *(extra or [])]}
    for r in raw.values():
        event = r["event"]
        if event not in EVENTS:
            dropped[f"event_{event}"] += 1
            continue
        first = first_attempt(t, r)
        if first is None:
            dropped["first_attempt_not_collected"] += 1
            continue
        if first["status"] != "completed" or first["conclusion"] not in KEEP:
            dropped[first["conclusion"] or first["status"]] += 1
            continue
        failed = int(first["conclusion"] == "failure")
        jobs_p = raw_dir(t) / "jobs" / f"{r['id']}.json"
        jobs = read_json(jobs_p)["jobs"] if jobs_p.exists() else None
        if not failed:
            failed_jobs: list[str] | None = []
        elif jobs is not None:
            bad = [j for j in jobs if j["conclusion"] == "failure"]
            # also true when no job failed: the run failed before testing the code
            if all(cfg.excludes(j["name"], _failed_step(j)) for j in bad):
                dropped["infra_failure"] += 1
                continue
            failed_jobs = [j["name"] for j in bad]
        else:
            failed_jobs = None
            dropped["no_jobs_kept"] += 1
        head = r.get("head_commit") or {}
        author = head.get("author") or {}
        created, started = _ts(r["created_at"]), _ts(first["run_started_at"])
        completed, latest = _ts(first["updated_at"]), _ts(r["updated_at"])
        if completed - started > MAX_RUN_SECS:
            # the last job's end is the true completion; NaN = estimated below
            ends = [_ts(j.get("completed_at")) for j in jobs or []]
            completed = np.nanmax(ends) if ends else np.nan
        if latest - started > MAX_RUN_SECS:
            latest = completed
        head_repo = (r.get("head_repository") or {}).get("full_name", t.repo)
        rows.append(
            {
                "run_id": r["id"],
                "head_sha": r["head_sha"],
                "author_id": hash_id(author.get("email", ""), salt),
                "lineage": lineage(r),
                "event": event,
                **{f"event_{e}": int(event == e) for e in sorted(EVENTS)},
                "from_fork": int(head_repo.lower() != t.repo.lower()),
                "head_branch": r.get("head_branch") or "",
                "base_sha": _pr_base(r),
                "created_at": created,
                "completed_at": completed,
                # a re-run's outcome is known only once its last attempt ends
                "latest_completed_at": latest,
                "failed": failed,
                "duration": completed - started,
                "queue_delay": started - created,
                "merge_to_ci_delay": created - _ts(head.get("timestamp")),
                "n_failed_jobs": np.nan if failed_jobs is None else len(failed_jobs),
                "workflow_path": r.get("path", ""),
                # bot accounts commit as `<name>[bot]` / `<id>+<name>[bot]@users...`
                "author_is_bot": int(
                    "[bot]" in author.get("name", "") + author.get("email", "")
                ),
                "was_rerun": int(r["run_attempt"] > 1),
                "flaky": int(failed and r["conclusion"] == "success"),
                "conclusion_latest": r["conclusion"],
                "failed_job_names": "|".join(failed_jobs or []),
            }
        )
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["created_at", "run_id"], kind="stable")
        # Runs without a usable completion time are placed in time with the median
        # run length, so the history still counts them; `duration` stays NaN.
        lost = df["completed_at"].isna()
        if lost.any():
            est = df["created_at"] + (df["completed_at"] - df["created_at"]).median()
            df.loc[lost, "completed_at"] = est[lost]
            df["latest_completed_at"] = df["latest_completed_at"].fillna(
                df["completed_at"]
            )
            logger.warning(
                "%d runs have a clean-up `updated_at`: completion estimated",
                lost.sum(),
            )
    return df.reset_index(drop=True), dict(dropped)


# =============================================================================
# Driver
# =============================================================================


def build(
    cfg: TargetConfig | Target,
    select: Callable[[pd.DataFrame], pd.Series] | None = None,
    extra: list[dict] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Joins the API cache and the git log into one row per run with every pre-run
    feature and the first-attempt label.

    History features always sweep every run, and need each run's changed files, so
    every run's change is read (diffs for heads off the branch). The remaining git
    features cost a few more git calls per run, so `select` limits them, and the
    returned rows, to the runs that are needed.

    Args:
        cfg (TargetConfig | Target): collection target; its raw cache must exist
            (`collect-runs`). A bare `Target` excludes no job.
        select (Callable | None): `runs_frame` frame -> bool mask of the rows to
            return; None returns every run.
        extra (list[dict] | None): raw runs not in the cache, as in `runs_frame`.

    Returns:
        tuple[pd.DataFrame, dict]: the table and its schema.
    """
    if isinstance(cfg, Target):
        cfg = TargetConfig(cfg, True, ())
    t = cfg.target
    salt = _salt(t)
    runs, dropped = runs_frame(cfg, salt, extra)
    if runs.empty:
        raise FileNotFoundError(
            f"no runs cached for {t.repo}; run `collect-runs` first"
        )
    repo_dir = repo_root(t.repo) / "repo.git"
    if not repo_dir.exists():
        repo_dir = git.clone(t.repo, t.branch)
    default = cfg.mainline or git.default_branch(repo_dir)
    log_df = git.commit_log(repo_dir, t.branch or default)
    log_df["author_id"] = [hash_id(e, salt) for e in log_df.pop("author_email")]
    runs["is_default_branch"] = (
        (runs["head_branch"] == default) & (runs["event"] == "push")
    ).astype(int)
    # Without a base from the API: the branch as of the run, never today's tip (a
    # PR merged later has its head on today's tip, which would give an empty diff).
    commits = log_df.drop_duplicates("sha")
    ct = commits["commit_time"].to_numpy(float)
    # same commit as `rev-list -1 --first-parent --before=<created_at>`
    later_min = np.minimum.accumulate(ct[::-1])[::-1]
    at = np.searchsorted(later_min, runs["created_at"].to_numpy(float), "right") - 1
    as_of = np.where(at >= 0, commits["sha"].to_numpy()[np.maximum(at, 0)], None)
    runs["base_sha"] = runs["base_sha"].where(runs["base_sha"].notna(), as_of)
    lost = git.fetch_heads(repo_dir, [*runs["head_sha"], *runs["base_sha"].dropna()])
    if lost:
        logger.warning("%d head or base commits are not fetchable", lost)
    # Each run's own change: a PR's files are on its branch, not on the default one.
    changes = run_changes(runs, log_df, repo_dir)
    runs["files"] = [
        frozenset(rc.change["path"]) if rc else frozenset() for rc in changes
    ]
    runs["reverts"] = [rc.reverts if rc else None for rc in changes]
    tag_times = np.sort(git.tags(repo_dir)["commit_time"].to_numpy(float))

    hist = history_features(runs)
    rows = None if select is None else np.flatnonzero(select(runs))
    gitf = git_features(runs, hist, log_df, changes, repo_dir, tag_times, rows)

    base = runs[["run_id", "head_sha", "author_id", "lineage", "created_at", "failed",
                 "queue_delay", "merge_to_ci_delay", "author_is_bot",
                 *(f"event_{e}" for e in sorted(EVENTS)), "from_fork",
                 "is_default_branch"]]  # fmt: skip
    df = pd.concat([base, hist.drop(columns=["_prev_idx", "_green_idx"]), gitf], axis=1)
    df = pd.concat([df, runs[DIAGNOSTIC_COLS]], axis=1)
    if rows is not None:
        df = df.iloc[rows]
    df[ORDER_COL] = pd.to_datetime(df[ORDER_COL], unit="s", utc=True)

    skip = {*ID_COLS, ORDER_COL, LABEL_COL, *DIAGNOSTIC_COLS}
    meta = read_json(m) if (m := _manifest(t)).exists() else {}
    schema = {
        "target_slug": slug(t),
        "repo": t.repo,
        "workflow_id": t.workflow_id,
        "branch": t.branch,
        "collected_at": meta.get("collected_at"),
        "n_runs": len(df),
        "n_failed": int(df[LABEL_COL].sum()),
        "n_dropped": dropped,
        "exclude_jobs": list(cfg.exclude_jobs),
        "events": sorted(EVENTS),
        "order_col": ORDER_COL,
        "id_cols": ID_COLS,
        "label_col": LABEL_COL,
        "prev_label_col": PREV_LABEL_COL,
        "nominal": [],
        "diagnostic_cols": DIAGNOSTIC_COLS,
        "feature_cols": [c for c in df.columns if c not in skip],
    }
    return df, schema


def _manifest(t: Target) -> pathlib.Path:
    return RESULTS_DIR / f"manifest_{slug(t)}.json"


def write(df: pd.DataFrame, schema: dict, t: Target) -> pathlib.Path:
    """
    Writes the table to `data/processed/<slug>.csv` and the schema to
    `results/collection/`.

    Returns:
        pathlib.Path: the CSV.
    """
    out = table_path(slug(t))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    sp = schema_path(slug(t))
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(schema, indent=2))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build the per-run feature table.")
    add_target_args(p)
    for cfg in targets_from_args(p.parse_args()):
        df, schema = build(cfg)
        out = write(df, schema, cfg.target)
        log.print(
            f"{out}: {schema['n_runs']} runs, {schema['n_failed']} failed, "
            f"{len(schema['feature_cols'])} features, dropped {schema['n_dropped']}"
        )


if __name__ == "__main__":
    main()
