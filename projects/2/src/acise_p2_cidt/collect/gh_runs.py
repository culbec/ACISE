import argparse
import json
import os
import pathlib
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta
from typing import Any

from rich.progress import MofNCompleteColumn, Progress

from acise_p2_cidt.collect import http
from acise_p2_cidt.collect.targets import (
    Target,
    add_target_args,
    slug,
    targets_from_args,
)
from acise_p2_cidt.utils.logging.log import get_logger, log

# The runs endpoint returns at most 1000 results per query; a window at or above
# this count is split, otherwise its tail would be cut.
MAX_WINDOW_RUNS: int = 1000
# openclaw peaks near 2,000 runs a day; an hour stays far below the 1,000 cap.
MIN_WINDOW: timedelta = timedelta(hours=1)
# Events that carry a code change; other runs (schedule, dispatch) get no details.
EVENTS: frozenset[str] = frozenset({"push", "pull_request", "merge_group"})

# projects/2/src/acise_p2_cidt/collect/gh_runs.py -> projects/2/data/raw
RAW_ROOT = pathlib.Path(__file__).resolve().parents[3] / "data" / "raw"
# projects/2/src/acise_p2_cidt/collect/gh_runs.py -> projects/2/results/collection
RESULTS_DIR = pathlib.Path(__file__).resolve().parents[3] / "results" / "collection"

logger = get_logger(__name__)


# =============================================================================
# Cache
# =============================================================================


def repo_root(repo: str) -> pathlib.Path:
    """`data/raw/<owner>__<name>`: the clone and salt, shared by the repo's targets."""
    return RAW_ROOT / repo.replace("/", "__")


def raw_dir(t: Target) -> pathlib.Path:
    """`data/raw/<owner>__<name>/<slug>`: the API cache of one target."""
    return repo_root(t.repo) / slug(t)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: pathlib.Path, obj: Any) -> None:
    # Write then rename: an interrupted run never leaves a half file.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def cached_runs(t: Target) -> list[dict]:
    """
    Every cached run of the target, deduplicated by id, oldest first.

    Args:
        t (Target): collection target.

    Returns:
        list[dict]: raw run objects as the API returned them (latest attempt).
    """
    runs = {
        r["id"]: r
        for p in sorted((raw_dir(t) / "runs").glob("*.json"))
        for r in read_json(p)["workflow_runs"]
    }
    return sorted(runs.values(), key=lambda r: (r["created_at"], r["id"]))


def first_attempt(t: Target, run: dict) -> dict | None:
    """
    The run's first attempt: the run itself if it was never re-run, else the cached
    `attempts/1`. None when that file is not collected yet.
    """
    if run.get("run_attempt", 1) <= 1:
        return run
    p = raw_dir(t) / "attempts" / f"{run['id']}.json"
    return read_json(p) if p.exists() else None


# =============================================================================
# Fetching
# =============================================================================


def _months(since: date, until: date) -> list[tuple[date, date]]:
    out, a = [], since
    while a <= until:
        nxt = date(a.year + a.month // 12, a.month % 12 + 1, 1)
        out.append((a, min(nxt - timedelta(days=1), until)))
        a = nxt
    return out


def _bounds(a: date, b: date) -> tuple[datetime, datetime]:
    """Days `[a, b]` as UTC datetimes `[a 00:00, b + 1 day 00:00)`."""
    start = datetime(a.year, a.month, a.day, tzinfo=UTC)
    return start, start + timedelta(days=(b - a).days + 1)


def _span(a: datetime, e: datetime) -> tuple[str, str]:
    """
    File stem and `created` query of the window `[a, e)`. Whole-day windows keep
    the date form, so caches written before sub-day windows stay valid.
    """
    if a.time() == e.time() == datetime.min.time():
        last = (e - timedelta(days=1)).date()
        return f"{a.date()}_{last}", f"{a.date()}..{last}"
    b = e - timedelta(seconds=1)  # the query is inclusive at both ends
    f, q = "%Y-%m-%dT%H-%M-%S", "%Y-%m-%dT%H:%M:%SZ"
    return f"{a:{f}}_{b:{f}}", f"{a:{q}}..{b:{q}}"


def _fetch_window(t: Target, a: datetime, e: datetime) -> list[pathlib.Path]:
    """
    Runs created in `[a, e)`, halved until each half is under `MAX_WINDOW_RUNS`:
    on day boundaries while the window spans several days, then on hours. A
    `.split` marker records a halving, so a resumed collection makes no calls for
    windows already on disk.
    """
    stem, created = _span(a, e)
    out = raw_dir(t) / "runs" / f"{stem}.json"
    # A window saved before it ended misses later runs and holds runs still in
    # progress, so it is fetched again.
    if out.exists() and datetime.fromtimestamp(out.stat().st_mtime, UTC) >= e:
        return [out]
    split = out.with_suffix(".split")
    if not split.exists():
        path = f"/repos/{t.repo}/actions/workflows/{t.workflow_id}/runs"
        params = {"created": created}
        if t.branch is not None:
            params["branch"] = t.branch
        total = http.get(path, {**params, "per_page": 1})["total_count"]
        if total < MAX_WINDOW_RUNS:
            runs = http.get_all(path, params, key="workflow_runs")
            _write_json(out, {"total_count": total, "workflow_runs": runs})
            logger.info("window %s: %d runs", created, len(runs))
            return [out]
        if e - a <= MIN_WINDOW:
            raise RuntimeError(f"{total} runs in {created}: above the API cap")
        split.parent.mkdir(parents=True, exist_ok=True)
        split.touch()
    days = e - a > timedelta(days=1) and a.time() == e.time() == datetime.min.time()
    unit = timedelta(days=1) if days else MIN_WINDOW
    mid = a + unit * (((e - a) // unit + 1) // 2)
    return _fetch_window(t, a, mid) + _fetch_window(t, mid, e)


def fetch_details(t: Target, runs: Iterable[dict]) -> int:
    """
    First attempts of re-run runs, then the first-attempt jobs of runs whose first
    attempt failed. The label is the first-attempt conclusion; successful runs have
    no failed jobs, so they cost no call.

    Args:
        t (Target): collection target.
        runs (Iterable[dict]): raw runs, e.g. one window (or a progress track of it).

    Returns:
        int: number of files fetched.
    """
    d, n = raw_dir(t), 0
    for r in runs:
        if r["status"] != "completed" or r["event"] not in EVENTS:
            continue
        run_url = f"/repos/{t.repo}/actions/runs/{r['id']}/attempts/1"
        attempt = d / "attempts" / f"{r['id']}.json"
        if r["run_attempt"] > 1 and not attempt.exists():
            _write_json(attempt, http.get(run_url))
            n += 1
        first = first_attempt(t, r)
        jobs = d / "jobs" / f"{r['id']}.json"
        if first and first["conclusion"] == "failure" and not jobs.exists():
            _write_json(jobs, {"jobs": http.get_all(f"{run_url}/jobs", key="jobs")})
            n += 1
    return n


def fetch_windows(t: Target, since: date, until: date) -> list[pathlib.Path]:
    """
    Runs of every month window in `[since, until]`, each followed by its details.
    Windows on disk cost no call unless they were saved before their last day ended.

    Args:
        t (Target): collection target.
        since (date): first day.
        until (date): last day.

    Returns:
        list[pathlib.Path]: the window files.
    """
    # One live display for the whole collection: in Jupyter every rich Live is its
    # own output widget, so a bar per window leaves a trail of blank widgets.
    windows, months = [], _months(since, until)
    with Progress(
        *Progress.get_default_columns(), MofNCompleteColumn(), console=log
    ) as bar:
        w_task = bar.add_task("windows", total=len(months))
        r_task = bar.add_task("runs")
        for a, b in months:
            for p in _fetch_window(t, *_bounds(a, b)):
                runs = read_json(p)["workflow_runs"]
                bar.reset(r_task, total=len(runs), description=p.stem)
                fetch_details(t, bar.track(runs, task_id=r_task))
                windows.append(p)
            bar.advance(w_task)
        bar.refresh()
    return windows


def collect(
    t: Target, since: date | None = None, until: date | None = None
) -> pathlib.Path:
    """
    Fetches runs month by month; each window's details are fetched as soon as the
    window is on disk. Everything cached is skipped, so an interrupted collection
    resumes where it stopped.

    Args:
        t (Target): collection target.
        since (date | None): first day; None = the workflow's creation day.
        until (date | None): last day; None = today (UTC).

    Returns:
        pathlib.Path: the manifest, which records the collection date C.
    """
    calls0 = http.calls
    if since is None:
        workflow = http.get(f"/repos/{t.repo}/actions/workflows/{t.workflow_id}")
        since = datetime.fromisoformat(workflow["created_at"]).date()
    until = until or datetime.now(UTC).date()
    windows = fetch_windows(t, since, until)
    runs = cached_runs(t)
    branch = "" if t.branch is None else f"branch={t.branch}&"

    manifest = {
        "target": t._asdict(),
        "since": str(since),
        "until": str(until),
        "collected_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "windows": {p.stem: read_json(p)["total_count"] for p in windows},
        "n_runs": len(runs),
        "runs_per_event": dict(Counter(r["event"] for r in runs).most_common()),
        "api_calls": http.calls - calls0,
        "queries": [
            (
                f"GET /repos/{t.repo}/actions/workflows/{t.workflow_id}/runs"
                f"?{branch}created=<window>"
            ),
            f"GET /repos/{t.repo}/actions/runs/<id>/attempts/1",
            f"GET /repos/{t.repo}/actions/runs/<id>/attempts/1/jobs",
        ],
    }
    out = RESULTS_DIR / f"manifest_{slug(t)}.json"
    _write_json(out, manifest)
    log.print(f"{manifest['n_runs']} runs cached, {manifest['api_calls']} API calls")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Collect GitHub Actions runs.")
    add_target_args(p)
    p.add_argument("--since", type=date.fromisoformat)
    p.add_argument("--until", type=date.fromisoformat)
    a = p.parse_args()
    for cfg in targets_from_args(a):
        log.print(f"manifest: {collect(cfg.target, a.since, a.until)}")


if __name__ == "__main__":
    main()
