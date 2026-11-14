# `collect/gh_runs.py`, `collect/http.py`

Status: done (step 1). Spec: [implementation-backbone.md](../implementation-backbone.md). Tests: `projects/2/tests/test_collect.py`.

> **Superseded in part (2026-11-07)** by the [multi-repo backbone](../../../../specs/project-2/multi-repo-analysis/implementation-backbone.md): Sec. 2 registry, Sec. 3 identity and layout (done, phase 1: `Target.branch` optional, `slug`, `repo_root`, cache at `data/raw/<owner>__<name>/<slug>/`, manifest by slug, no `N8N_MASTER`, `--target`/`--targets` CLI), Sec. 4 collector (phase 2). Where the two differ, the multi-repo backbone wins.

## Purpose

Fetch everything the dataset needs from the GitHub REST API and cache it on disk, so collection can stop and resume, and so the build step never touches the network. One workflow only: pull requests and the PR workflow are out of scope.

## API

```python
# collect/http.py
token() -> str                                         # $GITHUB_TOKEN, else `gh auth token`
get(path: str, params: dict | None = None) -> Any      # one call: rate limit, retry, no caching
get_all(path: str, params: dict | None = None, key: str | None = None) -> list  # every page

# collect/gh_runs.py
class Target(NamedTuple):
    repo: str          # "n8n-io/n8n"
    workflow_id: int   # 33323319, CI: Master
    branch: str        # "master"

N8N_MASTER: Target

raw_dir(t) -> pathlib.Path
cached_runs(t) -> list[dict]                            # every cached run, oldest first
first_attempt(t, run) -> dict | None                   # the run, or its cached attempts/1
fetch_details(t, runs) -> int                          # first attempts + failed jobs of `runs`
collect(t, since: date | None, until: date | None) -> pathlib.Path  # all of it, writes the manifest
main()                                                 # CLI `collect-runs`
```

CLI: `collect-runs [--repo n8n-io/n8n --workflow 33323319 --branch master] [--since YYYY-MM-DD] [--until YYYY-MM-DD]`.

## Cache layout

`projects/2/data/raw/<owner>__<name>/`:

| Dir | One file per | Content |
|---|---|---|
| `runs/` | date window, `<since>_<until>.json` | runs list, all pages merged; `<since>_<until>.split` marks a window that was halved |
| `attempts/` | run id | `/actions/runs/{id}/attempts/1` |
| `jobs/` | run id | `/actions/runs/{id}/attempts/1/jobs`, all pages |

A file that exists is never fetched again. Writes go to `*.tmp` and are renamed, so an interrupted run never leaves a half file. The manifest goes to `projects/2/results/collection/manifest_<owner>__<name>.json`: target, windows with their run counts, `collected_at` (date `C`), and the API call count.

## Design decisions

- **Date windows.** Calendar months. A one-result probe reads `total_count`; at >= 1000 the window is halved (recursively) and a `.split` marker is written, so a resumed collection re-enters the halves without probing again.
- **Processing per window.** As soon as a window is on disk, its re-run runs get `attempts/1` and its failed first attempts get their jobs. Details follow the runs, with no separate pass over the cache.
- **Default range.** `since` defaults to the workflow's `created_at` (one call to `/actions/workflows/{id}`); `until` defaults to today (UTC). `--since C` is how `retest` gets new runs.
- **Rate limits.** `X-RateLimit-Remaining: 0` sleeps until `X-RateLimit-Reset`. 5xx and secondary limits (403/429) are retried with exponential backoff (max 5 tries, honouring `Retry-After`). Any other error raises.
- **Token.** Read once per process, kept in memory, never logged or written. Sent as `Authorization: Bearer`, with `X-GitHub-Api-Version: API_VERSION`.
- **First attempt.** Only runs with `run_attempt > 1` need `attempts/1`; the others are their own first attempt.
- **Jobs only for failed first attempts.** Successful runs have 0 failed jobs by definition.
- **No fields are dropped at fetch time.** Raw JSON is kept whole; `data/build.py` picks fields.

## Verification

- Window split: a fake `get` returns `total_count` 2400 for a month and 600 per half; only the halves are written, and a second call makes no request.
- Resume: a second `collect` with the same arguments makes zero calls.
- First-attempt label: a run re-run to success keeps its failed `attempts/1` and gets its jobs.
- Atomic write: a fake `get` raising mid-window leaves no `.json`.
- Month windows across a year boundary.
- Rate limit: `Remaining: 0` with `Reset` 2 s ahead makes `get` sleep about 3 s (2 s + 1 s margin).
- No live network in tests.
