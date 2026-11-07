from datetime import UTC, date, datetime, timedelta

import pytest

from acise_p2_cidt.collect import gh_runs, http
from acise_p2_cidt.collect.gh_runs import Target

T = Target("owner/name", 1, "master")
JAN = (date(2026, 1, 1), date(2026, 1, 31))
DAY = datetime(2026, 1, 1, tzinfo=UTC)


def run(i, attempt=1, conclusion="success", event="push"):
    return {
        "id": i,
        "event": event,
        "run_attempt": attempt,
        "status": "completed",
        "conclusion": conclusion,
        "head_sha": f"{i:040x}",
        "created_at": f"2026-01-01T00:00:{i % 60:02d}Z",
    }


class FakeGet:
    """
    Runs endpoint with a total per `created` window (a dict, or a function of the
    `created` value); every 10th run failed and every 25th was re-run after a
    failed first attempt. Counts calls and records the runs-query params.
    """

    def __init__(self, totals, fail_on_page=None):
        self.total = totals.__getitem__ if isinstance(totals, dict) else totals
        self.fail_on_page, self.calls, self.params, self.windows = fail_on_page, 0, [], []

    def __call__(self, path, params=None):
        self.calls += 1
        params = params or {}
        if path.endswith("/jobs"):
            return {"jobs": [{"name": "unit", "conclusion": "failure"}]}
        if path.endswith("/attempts/1"):
            return run(int(path.split("/")[-3]), conclusion="failure")
        if self.fail_on_page and params.get("page") == self.fail_on_page:
            raise RuntimeError("network down")
        self.params.append(params)
        if params["created"] not in self.windows:
            self.windows.append(params["created"])
        total = self.total(params["created"])
        page, per = params.get("page", 1), params["per_page"]
        n = max(0, min(per, total - (page - 1) * per))
        base = self.windows.index(params["created"]) * 10**5 + (page - 1) * per
        runs = [
            run(
                i,
                attempt=2 if i % 25 == 0 else 1,
                conclusion="failure" if i % 10 == 0 else "success",
            )
            for i in range(base, base + n)
        ]
        return {"total_count": total, "workflow_runs": runs}


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(gh_runs, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(gh_runs, "RESULTS_DIR", tmp_path / "results")


def test_window_split(monkeypatch):
    # Jan has 2400 runs (above the cap); the halves split on Jan 16 have 600 each.
    fake = FakeGet(
        {
            "2026-01-01..2026-01-31": 2400,
            "2026-01-01..2026-01-16": 600,
            "2026-01-17..2026-01-31": 600,
        }
    )
    monkeypatch.setattr(http, "get", fake)
    paths = gh_runs._fetch_window(T, *gh_runs._bounds(*JAN))
    assert [p.name for p in paths] == [
        "2026-01-01_2026-01-16.json",
        "2026-01-17_2026-01-31.json",
    ]
    assert len(gh_runs.cached_runs(T)) == 1200
    # three count probes, then per half 6 full pages and an empty 7th
    assert fake.calls == 3 + 7 + 7
    assert gh_runs._fetch_window(T, *gh_runs._bounds(*JAN)) == paths
    assert fake.calls == 17  # the split marker spares the month probe


def test_resume_makes_no_calls(monkeypatch):
    fake = FakeGet({"2026-01-01..2026-01-31": 150})
    monkeypatch.setattr(http, "get", fake)
    gh_runs.collect(T, *JAN)
    # probe + 2 pages; re-runs 0, 25, ..., 125 -> 6 attempts, all failed first;
    # failed first attempts: 15 ids % 10 == 0, plus re-runs 25, 75, 125 -> 18 jobs
    assert fake.calls == 3 + 6 + 18
    gh_runs.collect(T, *JAN)
    assert fake.calls == 27


def test_first_attempt_label(monkeypatch):
    monkeypatch.setattr(http, "get", FakeGet({"2026-01-01..2026-01-31": 30}))
    gh_runs.collect(T, *JAN)
    r25 = next(r for r in gh_runs.cached_runs(T) if r["id"] == 25)
    assert r25["conclusion"] == "success"  # latest attempt passed
    assert gh_runs.first_attempt(T, r25)["conclusion"] == "failure"
    assert (gh_runs.raw_dir(T) / "jobs" / "25.json").exists()


def test_interrupted_window_leaves_no_file(monkeypatch):
    fake = FakeGet({"2026-01-01..2026-01-31": 150}, fail_on_page=2)
    monkeypatch.setattr(http, "get", fake)
    with pytest.raises(RuntimeError):
        gh_runs._fetch_window(T, *gh_runs._bounds(*JAN))
    assert not list(gh_runs.RAW_ROOT.rglob("*.json"))


def test_query_params_per_target(monkeypatch):
    fake = FakeGet({"2026-01-01..2026-01-31": 5})
    monkeypatch.setattr(http, "get", fake)
    gh_runs.collect(Target("o/r", 2), *JAN)  # all branches: no filter at all
    assert all(set(p) <= {"created", "per_page", "page"} for p in fake.params)
    fake.params.clear()
    gh_runs.collect(T, *JAN)  # restricted: branch filter, still no event filter
    assert all(p["branch"] == "master" and "event" not in p for p in fake.params)
    m = gh_runs.read_json(gh_runs.RESULTS_DIR / f"manifest_{gh_runs.slug(T)}.json")
    assert m["runs_per_event"] == {"push": 5}
    assert "event=" not in m["queries"][0] and "branch=master&" in m["queries"][0]


def per_hour_total(per_hour):
    """`created` -> runs on 2026-01-01, from a count per hour of that day."""

    def total(created):
        a, b = created.split("..")
        if "T" not in a:
            a, b = f"{a}T00:00:00Z", f"{b}T23:59:59Z"
        lo, hi = datetime.fromisoformat(a), datetime.fromisoformat(b)
        return sum(n for h, n in per_hour.items() if lo <= DAY + timedelta(hours=h) <= hi)

    return total


def test_day_above_cap_splits_into_hours(monkeypatch):
    # 1,500 runs in one day, all in hours 10 and 11: halves 24 h -> 12 h -> 6 h
    # -> 3 h -> 2 h + 1 h until each leaf is under the cap
    fake = FakeGet(per_hour_total({10: 750, 11: 750}))
    monkeypatch.setattr(http, "get", fake)
    paths = gh_runs._fetch_window(T, DAY, DAY + timedelta(days=1))
    assert [p.stem for p in paths] == [
        "2026-01-01T00-00-00_2026-01-01T05-59-59",
        "2026-01-01T06-00-00_2026-01-01T08-59-59",
        "2026-01-01T09-00-00_2026-01-01T10-59-59",  # 3 h halves into 2 h + 1 h
        "2026-01-01T11-00-00_2026-01-01T11-59-59",
        "2026-01-01T12-00-00_2026-01-01T23-59-59",
    ]
    assert "2026-01-01T11:00:00Z..2026-01-01T11:59:59Z" in fake.windows
    # only the leaves are written; split windows leave a marker only
    runs_dir = gh_runs.raw_dir(T) / "runs"
    assert sorted(p.stem for p in runs_dir.glob("*.json")) == [p.stem for p in paths]
    assert len(gh_runs.cached_runs(T)) == 1500
    calls = fake.calls
    assert gh_runs._fetch_window(T, DAY, DAY + timedelta(days=1)) == paths
    assert fake.calls == calls  # resume: zero calls


def test_hour_above_cap_raises(monkeypatch):
    monkeypatch.setattr(http, "get", FakeGet(per_hour_total({10: 1000})))
    with pytest.raises(RuntimeError, match="above the API cap"):
        gh_runs._fetch_window(T, DAY, DAY + timedelta(days=1))


def test_unsupported_events_get_no_details(monkeypatch):
    fake = FakeGet({})
    monkeypatch.setattr(http, "get", fake)
    runs = [
        run(1, conclusion="failure", event="schedule"),
        run(2, attempt=2, event="workflow_dispatch"),
        run(3, conclusion="failure", event="pull_request"),
        run(4, conclusion="failure", event="merge_group"),
    ]
    assert gh_runs.fetch_details(T, runs) == 2
    assert sorted(p.stem for p in gh_runs.raw_dir(T).rglob("*.json")) == ["3", "4"]


@pytest.mark.parametrize(
    "since, until, expected",
    [
        (date(2026, 1, 15), date(2026, 1, 20), [("2026-01-15", "2026-01-20")]),
        (
            date(2025, 12, 5),
            date(2026, 2, 3),
            [
                ("2025-12-05", "2025-12-31"),  # year boundary
                ("2026-01-01", "2026-01-31"),
                ("2026-02-01", "2026-02-03"),
            ],
        ),
    ],
)
def test_months(since, until, expected):
    got = [(str(a), str(b)) for a, b in gh_runs._months(since, until)]
    assert got == expected


class FakeResp:
    ok, status_code = True, 200

    def __init__(self, headers):
        self.headers = headers

    def json(self):
        return {"x": 1}


def test_rate_limit_sleeps_until_reset(monkeypatch):
    now = 1_000_000.0
    resp = FakeResp({"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(now + 2)})
    slept = []
    monkeypatch.setattr(http.time, "time", lambda: now)
    monkeypatch.setattr(http.time, "sleep", slept.append)
    session = type("S", (), {"get": lambda *a, **k: resp})()
    monkeypatch.setattr(http, "_session", lambda: session)
    assert http.get("/x") == {"x": 1}
    assert slept == [pytest.approx(3.0)]  # 2 s to reset + 1 s margin


def test_layout_two_targets_of_one_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(gh_runs, "RAW_ROOT", tmp_path)
    a, b = Target("o/r", 1, "release/1.x"), Target("o/r", 2)
    assert gh_runs.raw_dir(a) == tmp_path / "o__r" / "o__r__1@release~1.x"
    assert gh_runs.raw_dir(b) == tmp_path / "o__r" / "o__r__2"  # all branches
    # one clone and salt per repository, shared by its targets
    assert gh_runs.repo_root(a.repo) == gh_runs.raw_dir(b).parent == tmp_path / "o__r"
