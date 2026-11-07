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
    RESULTS_DIR,
    cached_runs,
    first_attempt,
    raw_dir,
    read_json,
    repo_root,
)
from acise_p2_cidt.collect.targets import (
    Target,
    add_target_args,
    slug,
    targets_from_args,
)
from acise_p2_cidt.data.history import git_features, history_features
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

ID_COLS: list[str] = ["run_id", "head_sha", "author_id"]
DIAGNOSTIC_COLS: list[str] = [
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


# merge-queue head branch: gh-readonly-queue/<base>/pr-<n>-<sha>
_QUEUE = re.compile(r"^gh-readonly-queue/.+/pr-(\d+)-[0-9a-f]+$")


def lineage(run: dict) -> str:
    """
    The run's lineage: the PR or branch whose previous run it follows.

    Args:
        run (dict): raw run object.

    Returns:
        str: `pr:<n>` for PR and merge-queue runs, `head:<repo>:<branch>` for a PR
            without a PR number (fork PRs), `branch:<name>` for pushes and the rest.
    """
    event, branch = run["event"], run.get("head_branch") or ""
    if event == "pull_request":
        if prs := run.get("pull_requests"):
            return f"pr:{prs[0]['number']}"
        head = (run.get("head_repository") or {}).get("full_name", "")
        return f"head:{head}:{branch}"
    if event == "merge_group" and (m := _QUEUE.match(branch)):
        return f"pr:{m[1]}"
    return f"branch:{branch}"


# =============================================================================
# Stage 1: runs
# =============================================================================


def runs_frame(
    t: Target, salt: bytes, extra: list[dict] | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    One row per run with its first-attempt label and times, oldest first.

    Args:
        t (Target): collection target (reads its raw cache).
        salt (bytes): salt for `hash_id`.
        extra (list[dict] | None): raw runs not in the cache; they replace a cached
            run with the same id.

    Returns:
        tuple[pd.DataFrame, dict[str, int]]: runs kept, and dropped counts per reason.
    """
    rows, dropped = [], defaultdict(int)
    raw = {r["id"]: r for r in [*cached_runs(t), *(extra or [])]}
    for r in raw.values():
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
            failed_jobs = [j["name"] for j in jobs if j["conclusion"] == "failure"]
        else:
            failed_jobs = None
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
        rows.append(
            {
                "run_id": r["id"],
                "head_sha": r["head_sha"],
                "author_id": hash_id(author.get("email", ""), salt),
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
    t: Target,
    select: Callable[[pd.DataFrame], pd.Series] | None = None,
    extra: list[dict] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Joins the API cache and the git log into one row per run with every pre-run
    feature and the first-attempt label.

    History features always sweep every run. The git features cost a few git calls
    per run (about an hour for all n8n runs), so `select` limits them, and the
    returned rows, to the runs that are needed.

    Args:
        t (Target): collection target; its raw cache must exist (`collect-runs`).
        select (Callable | None): `runs_frame` frame -> bool mask of the rows to
            return; None returns every run.
        extra (list[dict] | None): raw runs not in the cache, as in `runs_frame`.

    Returns:
        tuple[pd.DataFrame, dict]: the table and its schema.
    """
    salt = _salt(t)
    runs, dropped = runs_frame(t, salt, extra)
    if runs.empty:
        raise FileNotFoundError(
            f"no runs cached for {t.repo}; run `collect-runs` first"
        )
    repo_dir = repo_root(t.repo) / "repo.git"
    if not repo_dir.exists():
        repo_dir = git.clone(t.repo, t.branch)
    log_df = git.commit_log(repo_dir, t.branch)
    log_df["author_id"] = [hash_id(e, salt) for e in log_df.pop("author_email")]
    by_sha = log_df[log_df["path"].notna()].groupby("sha")["path"].agg(frozenset)
    runs["files"] = (
        runs["head_sha"]
        .map(by_sha)
        .apply(lambda s: s if isinstance(s, frozenset) else frozenset())
    )
    runs["reverts"] = runs["head_sha"].map(
        log_df.drop_duplicates("sha").set_index("sha")["reverts"]
    )
    tag_times = np.sort(git.tags(repo_dir)["commit_time"].to_numpy(float))

    hist = history_features(runs)
    rows = None if select is None else np.flatnonzero(select(runs))
    gitf = git_features(runs, hist, log_df, repo_dir, tag_times, rows)

    base = runs[["run_id", "head_sha", "author_id", "created_at", "failed",
                 "queue_delay", "merge_to_ci_delay", "author_is_bot"]]  # fmt: skip
    df = pd.concat([base, hist.drop(columns=["_prev_idx", "_green_idx"]), gitf], axis=1)
    df = pd.concat([df, runs[DIAGNOSTIC_COLS]], axis=1)
    if rows is not None:
        df = df.iloc[rows]
    df[ORDER_COL] = pd.to_datetime(df[ORDER_COL], unit="s", utc=True)

    skip = {*ID_COLS, ORDER_COL, LABEL_COL, *DIAGNOSTIC_COLS}
    meta = read_json(m) if (m := _manifest(t)).exists() else {}
    schema = {
        "repo": t.repo,
        "workflow_id": t.workflow_id,
        "branch": t.branch,
        "collected_at": meta.get("collected_at"),
        "n_runs": len(df),
        "n_failed": int(df[LABEL_COL].sum()),
        "n_dropped": dropped,
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
        df, schema = build(cfg.target)
        out = write(df, schema, cfg.target)
        log.print(
            f"{out}: {schema['n_runs']} runs, {schema['n_failed']} failed, "
            f"{len(schema['feature_cols'])} features, dropped {schema['n_dropped']}"
        )


if __name__ == "__main__":
    main()
