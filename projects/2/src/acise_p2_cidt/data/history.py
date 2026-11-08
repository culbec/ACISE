import pathlib
from bisect import insort
from collections import defaultdict
from typing import NamedTuple

import numpy as np
import pandas as pd
from rich.progress import track

from acise_p2_cidt.collect import git_changes as git
from acise_p2_cidt.data.load import PREV_LABEL_COL
from acise_p2_cidt.utils.logging.log import get_logger

# Renamed: git_features takes a `log` (commit log) argument.
from acise_p2_cidt.utils.logging.log import log as console

HOUR, DAY = 3600.0, 86400.0
# Overlapping on purpose: selection on the dev segment keeps the windows with signal.
RATE_WINDOWS: dict[str, float] = {
    "1h": HOUR, "3h": 3 * HOUR, "6h": 6 * HOUR, "12h": 12 * HOUR, "24h": DAY, "7d": 7 * DAY,
}  # fmt: skip
LOAD_WINDOWS: dict[str, float] = {k: v for k, v in RATE_WINDOWS.items() if k != "7d"}
FLAKY_WINDOWS: dict[str, float] = {"24h": DAY, "7d": 7 * DAY}
COUNT_WINDOWS: tuple[int, ...] = (5, 10, 20, 50)

logger = get_logger(__name__)


# =============================================================================
# Stage 2: CI history (completed-before sweep)
# =============================================================================


def _mean_or_nan(x: np.ndarray) -> float:
    return float(x.mean()) if len(x) else np.nan


def history_features(runs: pd.DataFrame) -> pd.DataFrame:
    """
    CI history, load and author features, reading only runs completed before each
    run's `created_at`.

    Runs are in creation order. A pointer over the completion order moves every run
    with `completed_at < created_at_i` into the known set before row i is processed.
    A run completed before row i was also created before it, so the known set is a
    subset of the earlier rows. "Previous" is the latest-created known run of the
    same lineage (PR or branch): the previous run of the whole workflow is often an
    unrelated PR. Rates over time windows, load, flaky, author and file statistics
    stay workflow-wide. Without a `lineage` column all runs share one lineage.
    Optional `files` (set of paths) and `reverts` (SHA) columns feed the file
    failure and revert features.

    Args:
        runs (pd.DataFrame): `runs_frame` output, optionally with `lineage`,
            `files`, `reverts`.

    Returns:
        pd.DataFrame: one row per run, same index; `_prev_idx` and `_green_idx` are
            the positions of the reference runs for the git diff features.
    """
    n = len(runs)
    created = runs["created_at"].to_numpy(float)
    completed = runs["completed_at"].to_numpy(float)
    latest = runs["latest_completed_at"].to_numpy(float)
    failed = runs["failed"].to_numpy(int)
    flaky = runs["flaky"].to_numpy(float)
    duration = runs["duration"].to_numpy(float)
    queue = runs["queue_delay"].to_numpy(float)
    n_failed_jobs = runs["n_failed_jobs"].to_numpy(float)
    authors = runs["author_id"].to_numpy()
    files = runs["files"].tolist() if "files" in runs else [frozenset()] * n
    reverts = runs["reverts"].tolist() if "reverts" in runs else [None] * n
    shas = runs["head_sha"].tolist()
    lineages = runs["lineage"].tolist() if "lineage" in runs else [""] * n

    known = np.zeros(n, bool)
    # known positions per lineage and per head SHA, in creation order
    by_lineage: dict[str, list[int]] = defaultdict(list)
    by_sha: dict[str, list[int]] = defaultdict(list)
    last_created: dict[str, int] = {}  # latest-created run per lineage, known or not
    by_completion = np.argsort(completed, kind="stable")
    k = 0
    file_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # runs, failed
    author_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    out = []
    for i in range(n):
        t = created[i]
        while k < n and completed[by_completion[k]] < t:
            j = by_completion[k]
            known[j] = True
            insort(by_lineage[lineages[j]], j)
            insort(by_sha[shas[j]], j)
            for f in files[j]:
                file_stats[f][0] += 1
                file_stats[f][1] += failed[j]
            author_stats[authors[j]][0] += 1
            author_stats[authors[j]][1] += failed[j]
            k += 1

        idx = np.flatnonzero(known[:i])  # creation order
        f, comp = failed[idx], completed[idx]
        lin_name = lineages[i]
        lin = np.array(by_lineage[lin_name], dtype=int)
        lf = failed[lin]
        prev_created = last_created.get(lin_name)
        last_created[lin_name] = i
        row: dict[str, float] = {
            "secs_since_prev_run": t - created[i - 1] if i else np.nan,
            "concurrent_runs": i - len(idx),  # created before, not completed yet
            "lineage_prior_runs": len(lin),
            "lineage_first_run": int(len(lin) == 0),
            # creation is known before the outcome, so a running run counts
            "secs_since_lineage_prev_run": (
                t - created[prev_created] if prev_created is not None else np.nan
            ),
        }
        # Creation is known before the outcome, so load counts every earlier run.
        for name, w in LOAD_WINDOWS.items():
            row[f"ci_load_{name}"] = i - np.searchsorted(created, t - w, "left")

        last = lin[-1] if len(lin) else None
        row[PREV_LABEL_COL] = failed[last] if last is not None else np.nan
        row["prev_duration"] = duration[last] if last is not None else np.nan
        row["prev_queue_delay"] = queue[last] if last is not None else np.nan
        row["prev_failed_jobs"] = n_failed_jobs[last] if last is not None else np.nan
        rev = lf[::-1]
        row["fail_streak"] = len(rev) if rev.all() else int(np.argmin(rev))
        fails = lin[lf == 1]
        passes = lin[lf == 0]
        last_fail = fails[-1] if len(fails) else None
        row["runs_since_last_fail"] = (
            int((lin > last_fail).sum()) if last_fail is not None else np.nan
        )
        row["secs_since_last_fail"] = (
            t - completed[last_fail] if last_fail is not None else np.nan
        )
        for name, w in RATE_WINDOWS.items():
            row[f"fail_rate_{name}"] = _mean_or_nan(f[comp >= t - w])
        for c in COUNT_WINDOWS:
            row[f"fail_rate_last_{c}"] = _mean_or_nan(lf[-c:])
        for name, w in FLAKY_WINDOWS.items():
            sel = idx[(comp >= t - w) & (latest[idx] < t)]  # final outcome known
            row[f"flaky_rate_{name}"] = _mean_or_nan(flaky[sel])

        a_runs, a_fails = author_stats.get(authors[i], (0, 0))
        row["author_prior_runs"] = a_runs
        row["author_prior_fail_rate"] = a_fails / a_runs if a_runs else np.nan

        probs = [s[1] / s[0] for p in files[i] if (s := file_stats.get(p))]
        row["file_fail_prob_max"] = max(probs) if probs else np.nan
        row["file_fail_prob_mean"] = float(np.mean(probs)) if probs else np.nan
        row["overlap_with_last_failed_run"] = (
            len(files[i] & files[last_fail]) / len(files[i])
            if last_fail is not None and files[i]
            else np.nan
        )
        row["reverted_sha_failed"] = np.nan
        if isinstance(reverts[i], str):  # NaN when the commit reverts nothing
            hits = [j for j in idx if shas[j].startswith(reverts[i])]
            if hits:
                row["reverted_sha_failed"] = failed[hits[-1]]
        same = by_sha.get(shas[i])  # e.g. the push and the PR run of one commit
        row["sha_prior_failed"] = failed[same[-1]] if same else np.nan
        row["_prev_idx"] = last if last is not None else np.nan
        row["_green_idx"] = passes[-1] if len(passes) else np.nan
        out.append(row)
    return pd.DataFrame(out, index=runs.index)


# =============================================================================
# Stage 3: git change and file history
# =============================================================================


def _entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _change(ch: pd.DataFrame) -> dict[str, float]:
    """Size and type of one commit's change (rows of `commit_log` with a path)."""
    churn = ch["added"].fillna(0) + ch["deleted"].fillna(0)
    kinds = ch["kind"].value_counts()
    n = len(ch)
    dirs = ch["path"].map(lambda p: p.rsplit("/", 1)[0] if "/" in p else "")
    out = {
        "files_changed": n,
        "additions": ch["added"].sum(),
        "deletions": ch["deleted"].sum(),
        "churn": churn.sum(),
        "src_churn": churn[ch["kind"] == "src"].sum(),
        "test_churn": churn[ch["kind"] == "test"].sum(),
        "n_dirs_touched": dirs.nunique(),
        "dir_entropy": _entropy(dirs.value_counts().to_numpy(float)) if n else np.nan,
    }
    for kind in ("src", "test", "config", "doc", "ci"):
        out[f"{kind}_files"] = int(kinds.get(kind, 0))
    out["test_file_ratio"] = out["test_files"] / n if n else np.nan
    return out


def _since(repo_dir: pathlib.Path, ref: str | None, sha: str, tag: str) -> dict:
    d = git.diff_numstat(repo_dir, ref, sha) if ref else None
    c = git.rev_count(repo_dir, ref, sha) if ref else None
    return {
        f"commits_since_{tag}": np.nan if c is None else c,
        f"files_since_{tag}": np.nan if d is None else len(d),
        f"churn_since_{tag}": (
            np.nan
            if d is None
            else float(d["added"].fillna(0).sum() + d["deleted"].fillna(0).sum())
        ),
    }


class RunChange(NamedTuple):
    pos: int  # anchor position in the deduplicated commit log
    before: int  # history reads log positions strictly below this
    change: pd.DataFrame  # path, kind, added, deleted
    parents: list[str]
    commit_time: float
    subject: str
    reverts: str | None
    author_id: str


def _ch_log(log: pd.DataFrame, commits: pd.DataFrame) -> pd.DataFrame:
    return log[log["path"].notna()].merge(
        commits[["sha"]].reset_index(names="pos"), on="sha"
    )


def run_changes(
    runs: pd.DataFrame, log: pd.DataFrame, repo_dir: pathlib.Path
) -> list[RunChange | None]:
    """
    Anchors each run on the walked branch's first-parent line and reads its change.

    The anchor is the head if it is on the line, else `merge-base(base_sha, head)`,
    else (a merge-base off the line: stacked PR, earlier merge-queue entry, release
    branch) the line commit its branch forked from. History is read before the
    head, or up to and including the anchor. The
    change of a head on the line is its log rows; off the line it is the whole
    proposed change, `merge-base..head` (PR and merge-queue runs), or `head^..head`
    (pushes to other branches).

    Args:
        runs (pd.DataFrame): `runs_frame` output, with `event`, `base_sha` and
            `author_id`; without `event` every run is a push.
        log (pd.DataFrame): `commit_log` output with an `author_id` column.
        repo_dir (pathlib.Path): the clone, for merge-bases and diffs.

    Returns:
        list[RunChange | None]: one per run; None without an anchor (head or
            base missing, or no fork point on the line).
    """
    commits = log.drop_duplicates("sha").reset_index(drop=True)
    pos_of = dict(zip(commits["sha"], commits.index))
    ch_log = _ch_log(log, commits)
    on_log = dict(
        tuple(ch_log[ch_log["sha"].isin(set(runs["head_sha"]))].groupby("sha"))
    )
    empty = ch_log.iloc[:0][["path", "kind", "added", "deleted"]]
    n = len(runs)
    events = runs["event"].tolist() if "event" in runs else ["push"] * n
    bases = runs["base_sha"].tolist() if "base_sha" in runs else [None] * n
    authors = runs["author_id"].tolist()
    out: list[RunChange | None] = []
    for i, sha in enumerate(
        track(runs["head_sha"], description="run changes", console=console)
    ):
        if sha in pos_of:
            p = pos_of[sha]
            c = commits.loc[p]
            ch = on_log.get(sha, empty)[["path", "kind", "added", "deleted"]]
            out.append(RunChange(p, p, ch, c["parents"], float(c["commit_time"]),
                                 c["subject"], c["reverts"], c["author_id"]))  # fmt: skip
            continue
        base = bases[i]
        mb = git.merge_base(repo_dir, base, sha) if isinstance(base, str) else None
        # a merge-base off the line (stacked PR, earlier queue entry, release
        # branch): anchor where its branch forked from the line
        anchor = mb if mb in pos_of or mb is None else git.first_parent_on(repo_dir, mb, pos_of)  # fmt: skip
        info = git.commit_info(repo_dir, sha) if anchor is not None else None
        ref = f"{sha}^" if events[i] == "push" else mb
        ch = git.diff_numstat(repo_dir, ref, sha) if info is not None else None
        if ch is None:
            out.append(None)
            continue
        parents, ct, subject, reverts = info
        # the anchor's own change precedes the head
        out.append(RunChange(pos_of[anchor], pos_of[anchor] + 1, ch, parents,
                             float(ct), subject, reverts, authors[i]))  # fmt: skip
    return out


def git_features(
    runs: pd.DataFrame,
    hist: pd.DataFrame,
    log: pd.DataFrame,
    changes: list[RunChange | None],
    repo_dir: pathlib.Path,
    tag_times: np.ndarray,
    rows: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Change size and type, commit, file history, since-previous/green and workflow
    features, read at each run's anchor (`run_changes`).

    Args:
        runs (pd.DataFrame): `runs_frame` output.
        hist (pd.DataFrame): `history_features` output (reference run positions).
        log (pd.DataFrame): `commit_log` output with an `author_id` column.
        changes (list[RunChange | None]): `run_changes` output.
        repo_dir (pathlib.Path): the clone, for the diffs between run SHAs.
        tag_times (np.ndarray): sorted tag commit times (unix seconds).
        rows (np.ndarray | None): positions of the runs to compute; None = all.

    Returns:
        pd.DataFrame: one row per computed run, same index; NaN where the run has
            no anchor.
    """
    commits = log.drop_duplicates("sha").reset_index(drop=True)
    ch_log = _ch_log(log, commits)
    file_hist = {}  # path -> (positions, commit times, cumulative distinct authors)
    for path, g in ch_log.groupby("path", sort=False):
        first_seen = ~g["author_id"].duplicated()
        file_hist[path] = (
            g["pos"].to_numpy(), g["commit_time"].to_numpy(float), first_seen.cumsum().to_numpy()
        )  # fmt: skip
    author_pos = {
        a: g.to_numpy() for a, g in commits.reset_index().groupby("author_id")["index"]
    }
    first_touch = ch_log.groupby(["author_id", "path"])["pos"].min().to_dict()
    shas = runs["head_sha"].tolist()

    rows = np.arange(len(runs)) if rows is None else rows
    out, missing = [], 0
    for i in track(rows, description="git features", console=console):
        sha, created = shas[i], runs["created_at"].iat[i]
        rc = changes[i]
        if rc is None:
            missing += 1
            out.append({})
            continue
        p, q, ch, a = rc.pos, rc.before, rc.change, rc.author_id
        parents, head_t, subject, reverts = (
            rc.parents,
            rc.commit_time,
            rc.subject,
            rc.reverts,
        )
        F = set(ch["path"])
        row = _change(ch)
        row["n_parents"] = len(parents)
        row["is_revert"] = int(
            subject.startswith('Revert "') and isinstance(reverts, str)
        )

        ages, prior, n_authors, lasts = [], [], [], []
        for f in F:
            if f not in file_hist:  # a file new in this change
                prior.append(0)
                n_authors.append(0)
                continue
            fp, ft, fa = file_hist[f]
            k = int(np.searchsorted(fp, q))
            prior.append(k)
            n_authors.append(int(fa[k - 1]) if k else 0)
            if k:
                ages.append((head_t - ft[0]) / DAY)
                lasts.append(head_t - ft[k - 1])
        row["file_age_mean"] = float(np.mean(ages)) if ages else np.nan
        row["file_prior_changes"] = float(np.mean(prior)) if prior else np.nan
        row["file_distinct_authors"] = (
            float(np.mean(n_authors)) if n_authors else np.nan
        )
        row["secs_since_files_last_changed"] = min(lasts) if lasts else np.nan

        row["author_prior_commits"] = (
            int(np.searchsorted(author_pos[a], q)) if a in author_pos else 0
        )
        row["author_file_familiarity"] = (
            float(np.mean([first_touch.get((a, f), q) < q for f in F])) if F else np.nan
        )

        wf = runs["workflow_path"].iat[i]
        row["workflow_changed"] = int(wf in F)
        row["days_since_workflow_change"] = np.nan
        if wf in file_hist:
            fp, ft, _ = file_hist[wf]
            if k := int(np.searchsorted(fp, p, "right")):
                row["days_since_workflow_change"] = (head_t - ft[k - 1]) / DAY
        k = int(np.searchsorted(tag_times, created, "left"))
        row["days_since_last_tag"] = (created - tag_times[k - 1]) / DAY if k else np.nan

        for tag, col in (("prev_run", "_prev_idx"), ("green", "_green_idx")):
            ref = hist[col].iat[i]
            row |= _since(repo_dir, None if np.isnan(ref) else shas[int(ref)], sha, tag)
        out.append(row)
    if missing:
        logger.warning(
            "%d runs have no anchor on the branch: git features NaN", missing
        )
    return pd.DataFrame(out, index=runs.index[rows])
