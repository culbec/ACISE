# `collect/gh_runs.py`

Status: todo (step 1). Spec: [implementation-backbone.md](../implementation-backbone.md). Tests: `projects/2/tests/test_collect.py`.

## Purpose

Fetch everything the dataset needs from the GitHub REST API and cache it on disk, so collection can stop and resume, and so the build step never touches the network.

## API

```python
class Target(NamedTuple):
    repo: str                 # "n8n-io/n8n"
    workflow_id: int          # 33323319, CI: Master
    branch: str               # "master"
    pr_workflow_id: int | None  # 33322401, for pre-merge PR CI counts; None skips them

N8N_MASTER: Target

token() -> str                                   # $GITHUB_TOKEN, else `gh auth token`
get(path: str, params: dict | None = None) -> dict  # one call: rate limit, retry, no caching

fetch_runs(t: Target, since: date, until: date) -> list[pathlib.Path]   # one JSON per window
fetch_first_attempts(t: Target) -> int          # for cached runs with run_attempt > 1
fetch_failed_jobs(t: Target) -> int             # jobs of runs whose first attempt failed
fetch_pulls(t: Target) -> int                   # /commits/{sha}/pulls per head_sha
fetch_pr_runs(t: Target) -> int                 # PR CI runs of each merged PR
collect(t: Target, since: date | None, until: date | None) -> pathlib.Path  # all of the above, writes the manifest
main()                                          # CLI `collect-runs`
```

CLI: `collect-runs [--repo n8n-io/n8n --workflow 33323319 --branch master --pr-workflow 33322401] [--since YYYY-MM-DD] [--until YYYY-MM-DD]`.

## Cache layout

`projects/2/data/raw/<owner>__<name>/`:

| Dir | One file per | Content |
|---|---|---|
| `runs/` | date window, `<since>_<until>.json` | runs list, all pages merged |
| `attempts/` | run id | `/actions/runs/{id}/attempts/1` |
| `jobs/` | run id | `/actions/runs/{id}/jobs`, all pages |
| `pulls/` | head SHA | `/commits/{sha}/pulls` (may be empty: direct pushes) |
| `pr_runs/` | PR number | PR workflow runs filtered as below |

A file that exists is never fetched again. Writes go to `*.tmp` and are renamed, so an interrupted run never leaves a half file. The manifest goes to `projects/2/results/collection/manifest_<owner>__<name>.json`: target, windows with their run counts, `collected_at` (this is date `C`), and the API call count.

## Design decisions

- **Date windows.** Start with calendar months. The runs endpoint returns at most 1000 results per query, so if a window's `total_count` is ≥ 1000 it is split in half and both halves are fetched (recursively). The test checks the split on a fake `get`.
- **Default range.** `since` defaults to the workflow's earliest run (found by an ascending probe on yearly windows); `until` defaults to now (UTC). `--since C` is how `retest` gets new runs.
- **Rate limits.** Read `X-RateLimit-Remaining` / `X-RateLimit-Reset` on every response; at 0, sleep until reset with a rich countdown. Retry 5xx and secondary-limit 403/429 with exponential backoff (max 5 tries, honouring `Retry-After`). Anything else raises.
- **Token.** Read once per process, kept in memory, never logged or written. Sent as `Authorization: Bearer`.
- **First attempt.** Only runs with `run_attempt > 1` need `attempts/1`; others are their own first attempt.
- **Jobs only for failed runs.** The build needs the failed-job count of earlier failed runs, and the EDA the job names. Successful runs have 0 failed jobs by definition.
- **PR CI runs.** For each merged PR: PR workflow runs with `branch=<pr.head.ref>`, kept only if `head_repository.full_name == pr.head.repo.full_name` (forks reuse branch names like `master`) and `created_at < pr.merged_at`. The window `[pr.created_at, pr.merged_at]` goes in `created=` so the 1000 cap is never hit for one PR.
- **No fields are dropped at fetch time.** Raw JSON is kept whole; `data/build.py` picks fields. A later feature then needs no re-collection.
- `requests.Session` reused for keep-alive; timeout 30 s per call.

## Verification

- Window split: a fake `get` returns `total_count` 2400 for a month and 600 per half; `fetch_runs` writes the two halves, not the month.
- Resume: a second `collect` with the same arguments makes zero calls (fake `get` counts them).
- Atomic write: a fake `get` raising mid-window leaves no `.json` for that window.
- PR filter: recorded fixture with a fork PR on branch `master` and a run after `merged_at`; both are excluded.
- Rate limit: a response with `Remaining: 0` and `Reset` 2 s ahead makes `get` sleep about 2 s (sleep is patched and its argument checked).
- No live network in tests.
