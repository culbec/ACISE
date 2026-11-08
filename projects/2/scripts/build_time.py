"""
Estimates how long building each target's dataset takes, before the full build.

For every target it clones the repository (once; later calls only fetch), times
one walk of the default branch's commit log, then on a random sample of cached runs:
- fetches by SHA the heads and bases missing from the clone (a clone holds only
  the current tip of each branch and PR),
- times the per-run git work: for pull_request / merge_group runs
  `merge-base(base, head)` then the diff base..head, for push runs head^..head.

The base is a commit as of the run, never today's branch tip: the PR's `base.sha`,
the base SHA in a merge-queue branch name, or else the default branch's last
commit created before the run.

The per-run mean is scaled to the runs expected in the collection window.

Usage (from projects/2):
    uv run python scripts/build_time.py                 # every enabled target
    uv run python scripts/build_time.py <slug> [...]    # chosen targets
    uv run python scripts/build_time.py --sample 50 --window-days 182.6

Writes results/collection/build_time.json and prints a table.
"""

import argparse
import json
import random
import re
import subprocess
import time
from datetime import date

from rich.table import Table

from acise_p2_cidt.collect import git_changes as git
from acise_p2_cidt.collect.gh_runs import (
    EVENTS,
    RESULTS_DIR,
    cached_runs,
    read_json,
    repo_root,
)
from acise_p2_cidt.collect.targets import find_target, load_targets, slug
from acise_p2_cidt.utils.logging.log import log


def _sample_days(s: str) -> float:
    m = RESULTS_DIR / f"manifest_{s}.json"
    if not m.exists():
        raise SystemExit(f"no manifest for {s}: run collect-runs first")
    meta = read_json(m)
    return (date.fromisoformat(meta["until"]) - date.fromisoformat(meta["since"])).days + 1


# gh-readonly-queue/<base branch>/pr-<n>-<base sha>
_QUEUE = re.compile(r"/pr-\d+-([0-9a-f]{40})$")
FETCH_BATCH = 200


def _git(repo, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], input=stdin, check=False,
                          capture_output=True, text=True)  # fmt: skip


def _base(repo, r: dict, default: str) -> str | None:
    """The commit the run's change is measured against, as known when it ran."""
    if r["event"] == "push":
        return f"{r['head_sha']}^"
    if prs := r.get("pull_requests"):
        return prs[0]["base"]["sha"]
    if m := _QUEUE.search(r.get("head_branch") or ""):
        return m[1]
    out = _git(repo, "rev-list", "-1", "--first-parent",
               f"--before={r['created_at']}", default).stdout.strip()  # fmt: skip
    return out or None


def _missing(repo, shas: list[str]) -> list[str]:
    out = _git(repo, "cat-file", "--batch-check", stdin="\n".join(shas)).stdout
    return [line.split()[0] for line in out.splitlines() if line.endswith("missing")]


def _fetch(repo, shas: list[str]) -> None:
    """Fetches commits by SHA; refs/ci/<sha> keeps them from being garbage-collected."""
    for i in range(0, len(shas), FETCH_BATCH):
        specs = [f"{h}:refs/ci/{h}" for h in shas[i : i + FETCH_BATCH]]
        _git(repo, "fetch", "-q", "--no-tags", "origin", *specs)


def _time_target(cfg, n: int, window_days: float, seed: int, cloned: set[str]) -> dict:
    t, s = cfg.target, slug(cfg.target)
    clone_s = 0.0
    if t.repo not in cloned:  # one clone per repository, shared by its targets
        t0 = time.perf_counter()
        git.clone(t.repo, None)
        clone_s = time.perf_counter() - t0
        cloned.add(t.repo)
    repo = repo_root(t.repo) / "repo.git"
    default = git.default_branch(repo)

    t0 = time.perf_counter()
    git.commit_log(repo, t.branch or default)
    log_s = time.perf_counter() - t0

    runs = [r for r in cached_runs(t) if r["event"] in EVENTS]
    sample = random.Random(seed).sample(runs, min(n, len(runs)))
    bases = [_base(repo, r, default) for r in sample]
    shas = sorted({r["head_sha"] for r in sample} | {b for b in bases if b and "^" not in b})
    before = _missing(repo, shas)
    t0 = time.perf_counter()
    _fetch(repo, before)
    fetch_s = time.perf_counter() - t0
    after = _missing(repo, before)

    failed = 0
    t0 = time.perf_counter()
    for r, base in zip(sample, bases):
        head = r["head_sha"]
        if base is not None and not base.endswith("^"):
            base = git.merge_base(repo, base, head)
        if base is None or git.diff_numstat(repo, base, head) is None:
            failed += 1
    per_run = (time.perf_counter() - t0) / max(len(sample), 1)
    k = max(len(sample), 1)

    est_runs = len(runs) * window_days / _sample_days(s)
    return {
        "target": s,
        "clone_s": round(clone_s, 1),
        "commit_log_s": round(log_s, 1),
        "sampled": len(sample),
        "missing_before": len(before),
        "missing_after": len(after),
        "fetch_s": round(fetch_s, 1),
        "failed_share": round(failed / k, 3),
        "s_per_run": round(per_run, 4),
        "est_runs": round(est_runs),
        # fetch time per sampled run, scaled like the git work
        "est_build_h": round((log_s + (per_run + fetch_s / k) * est_runs) / 3600, 2),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Time the git part of the dataset build.")
    p.add_argument("slugs", nargs="*", help="target slugs; none = every enabled one")
    p.add_argument("--sample", type=int, default=200, help="runs sampled per target")
    p.add_argument("--window-days", type=float, default=182.6, help="collection window")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    cfgs = [find_target(s) for s in a.slugs] if a.slugs else load_targets()
    rows, cloned = [], set()
    for cfg in cfgs:
        log.print(f"timing {slug(cfg.target)}")
        rows.append(_time_target(cfg, a.sample, a.window_days, a.seed, cloned))

    out = RESULTS_DIR / "build_time.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))

    table = Table(title=f"build time (window {a.window_days} days)")
    for c in rows[0]:
        table.add_column(c)
    for r in rows:
        table.add_row(*map(str, r.values()))
    log.print(table)
    log.print(f"total est. build: {sum(r['est_build_h'] for r in rows):.1f} h, saved to {out}")


if __name__ == "__main__":
    main()
