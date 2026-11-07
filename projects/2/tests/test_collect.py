from datetime import date

import pytest

from acise_p2_cidt.collect import gh_runs, http
from acise_p2_cidt.collect.gh_runs import Target

T = Target("owner/name", 1, "master")
JAN = (date(2026, 1, 1), date(2026, 1, 31))


def run(i, attempt=1, conclusion="success"):
    return {
        "id": i,
        "run_attempt": attempt,
        "status": "completed",
        "conclusion": conclusion,
        "head_sha": f"{i:040x}",
        "created_at": f"2026-01-01T00:00:{i % 60:02d}Z",
    }


class FakeGet:
    """
    Runs endpoint with a fixed total per `created` window; every 10th run failed
    and every 25th was re-run after a failed first attempt. Counts calls.
    """

    def __init__(self, totals, fail_on_page=None):
        self.totals, self.fail_on_page, self.calls = totals, fail_on_page, 0

    def __call__(self, path, params=None):
        self.calls += 1
        params = params or {}
        if path.endswith("/jobs"):
            return {"jobs": [{"name": "unit", "conclusion": "failure"}]}
        if path.endswith("/attempts/1"):
            return run(int(path.split("/")[-3]), conclusion="failure")
        if self.fail_on_page and params.get("page") == self.fail_on_page:
            raise RuntimeError("network down")
        total = self.totals[params["created"]]
        page, per = params.get("page", 1), params["per_page"]
        n = max(0, min(per, total - (page - 1) * per))
        base = sorted(self.totals).index(params["created"]) * 10**5 + (page - 1) * per
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
    paths = gh_runs._fetch_window(T, *JAN)
    assert [p.name for p in paths] == [
        "2026-01-01_2026-01-16.json",
        "2026-01-17_2026-01-31.json",
    ]
    assert len(gh_runs.cached_runs(T)) == 1200
    # three count probes, then per half 6 full pages and an empty 7th
    assert fake.calls == 3 + 7 + 7
    assert gh_runs._fetch_window(T, *JAN) == paths
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
        gh_runs._fetch_window(T, *JAN)
    assert not list(gh_runs.RAW_ROOT.rglob("*.json"))


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
