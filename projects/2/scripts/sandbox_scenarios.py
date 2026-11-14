"""
Runs the sandbox scenarios (E4) on the `ci-sandbox` branch of culbec/ACISE.

Setup (done 2026-11-14):
  1. The orphan branch `ci-sandbox` holds the sandbox package and
     `.github/workflows/sandbox.yml` (workflow id 418201936).
  2. `config/targets.json` lists it, disabled, so it is never trained on, with
     `"mainline": "ci-sandbox"`.
  3. `git worktree add ../ACISE-ci-sandbox ci-sandbox` (from the repository root):
     the scenarios are written there, never in the project checkout.

Run from `projects/2`:
  uv run python scripts/sandbox_scenarios.py --worktree <dir>          # dry run
  uv run python scripts/sandbox_scenarios.py --worktree <dir> --push   # live

The dry run writes every scenario as a local commit (unbroken ones move the local
`ci-sandbox`) and pushes nothing. `--push` pushes each scenario branch, opens a
pull request for the `pr` scenarios, predicts every run of a scored scenario as
soon as it appears (`results/sandbox/predictions.jsonl`), waits for the runs to
end, then fast-forwards `ci-sandbox` to the unbroken scenarios and closes the pull
requests of the broken ones. The state is saved after every scenario in
`results/sandbox/scenarios.json`; a second start resumes after the last one done.
"""

import argparse
import json
import pathlib
import subprocess
import time

import pandas as pd

from acise_p2_cidt.collect import http
from acise_p2_cidt.utils.logging.log import log
from acise_p2_cidt.validate import live, sandbox, scenarios

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLAN = sandbox.SANDBOX_DIR / "scenarios.json"
PREDICTIONS = sandbox.SANDBOX_DIR / "predictions.jsonl"
BASE = "ci-sandbox"
POLL_SECS = 3
TIMEOUT_SECS = 45 * 60


def git(wt: pathlib.Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(wt), *args], check=True,
                          capture_output=True, text=True).stdout.strip()  # fmt: skip


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True,
                          text=True).stdout.strip()  # fmt: skip


def _check_worktree(wt: pathlib.Path) -> None:
    project = pathlib.Path(git(ROOT, "rev-parse", "--show-toplevel"))
    if pathlib.Path(git(wt, "rev-parse", "--show-toplevel")) == project:
        raise SystemExit(f"{wt} is the project checkout; give a worktree of {BASE}")
    if not (wt / scenarios.WORKFLOW_FILE).exists():
        raise SystemExit(f"{wt} holds no {scenarios.WORKFLOW_FILE}: not the sandbox")
    if git(wt, "status", "--porcelain"):
        raise SystemExit(f"{wt} has uncommitted changes")


def _state(seed: int, plan: list[scenarios.Scenario]) -> dict:
    """The saved plan with its progress; a plan for another seed is refused."""
    if PLAN.exists():
        st = json.loads(PLAN.read_text())
        if st["seed"] != seed or [s["branch"] for s in st["scenarios"]] != [
            s.branch for s in plan
        ]:
            raise SystemExit(f"{PLAN} holds another plan; move it away to start anew")
        return st
    scenarios.save_plan(PLAN, seed, plan)
    return json.loads(PLAN.read_text())


def _runs(cfg, sha: str, branch: str) -> list[dict]:
    t = cfg.target
    q = {"head_sha": sha, "per_page": 20}
    runs = http.get(f"/repos/{t.repo}/actions/workflows/{t.workflow_id}/runs", q)
    # the fast-forward of ci-sandbox runs the same commit again: not the scenario's
    return [r for r in runs["workflow_runs"] if r["head_branch"] == branch]


def _watch(sc, sha, cfg, model, expected) -> list[dict]:
    """Predicts each run of the scenario as it appears, then waits for all to end."""
    seen: dict[int, dict] = {}
    start = time.monotonic()
    while True:
        for r in _runs(cfg, sha, sc.branch):
            if r["id"] not in seen and sc.phase == "scored":
                row = sandbox.predict(model, r, cfg) | {"scenario": sc.to_dict()}
                sandbox.record(PREDICTIONS, row)
                state = "running" if r["status"] != "completed" else "ALREADY DONE"
                log.print(f"  run {r['id']} ({r['event']}, {state}): "
                          f"P(fail) {row['p_fail']:.2f}")  # fmt: skip
            seen[r["id"]] = r
        done = [r for r in seen.values() if r["status"] == "completed"]
        if len(seen) >= expected and len(done) == len(seen):
            return [{"id": r["id"], "event": r["event"], "conclusion": r["conclusion"]}
                    for r in seen.values()]  # fmt: skip
        if time.monotonic() - start > TIMEOUT_SECS:
            raise RuntimeError(f"scenario {sc.n}: runs not over after {TIMEOUT_SECS} s")
        time.sleep(POLL_SECS)


def main() -> None:
    p = argparse.ArgumentParser(description="Sandbox scenarios on ci-sandbox.")
    p.add_argument("--worktree", type=pathlib.Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--scored", type=int, default=48)
    p.add_argument("--model", type=pathlib.Path, help=".skops file; default T2 CART")
    p.add_argument("--push", action="store_true", help="push, open PRs, predict")
    p.add_argument("--limit", type=int, help="stop after this many scenarios")
    a = p.parse_args()
    wt = a.worktree.resolve()
    _check_worktree(wt)
    plan = scenarios.plan(a.seed, a.warmup, a.scored)
    st = _state(a.seed, plan) if a.push else {"scenarios": [s.to_dict() for s in plan]}
    cfg = model = cutoff = None
    if a.push:
        cfg = sandbox.target()
        model = a.model or next(live.MODEL_DIR.glob("cart_pool-*_T2.skops"))
        cutoff = pd.Timestamp(live.load_model(model)[1]["cutoff"]).to_pydatetime()
        log.print(f"sandbox {cfg.target}, model {model.name}")
    base = f"origin/{BASE}" if a.push else BASE
    todo = [s for s, d in zip(plan, st["scenarios"]) if "sha" not in d]
    for sc in todo[: a.limit]:
        entry = st["scenarios"][sc.n]
        if a.push:
            git(wt, "fetch", "-q", "origin", BASE)
        git(wt, "checkout", "-q", "-B", sc.branch, base)
        info = scenarios.apply(sc, wt, a.seed)
        git(wt, "add", "-A")
        git(
            wt,
            "commit",
            "-q",
            "-m",
            f"scenario {sc.n}: {sc.shape.name} ({sc.breakage})",
        )
        sha = git(wt, "rev-parse", "HEAD")
        log.print(f"{sc.n} {sc.phase} {sc.branch} {sc.breakage} {sc.event} {sha[:10]}")
        entry |= info | {"sha": sha}
        if not a.push:
            if sc.breakage == "none":
                git(wt, "branch", "-f", BASE, sha)
            continue
        if sc.phase == "scored":
            sandbox.refresh(cfg, cutoff)  # history up to date before the run exists
        git(wt, "push", "-q", "origin", sc.branch)
        pr = None
        if sc.event == "pr":
            pr = gh("pr", "create", "--repo", cfg.target.repo, "--base", BASE,
                    "--head", sc.branch, "--title", f"sandbox scenario {sc.n}",
                    "--body", f"Generated scenario {sc.n} ({sc.shape.name}).")  # fmt: skip
        entry["pr"] = pr
        entry["runs"] = _watch(sc, sha, cfg, model, 2 if pr else 1)
        if sc.breakage == "none":
            git(wt, "push", "-q", "origin", f"{sha}:refs/heads/{BASE}")  # fast-forward
        elif pr:
            gh("pr", "close", pr, "--comment", "Broken on purpose; not merged.")
        entry["merged"] = sc.breakage == "none"
        PLAN.write_text(json.dumps(st, indent=1))
    if a.push:
        git(wt, "fetch", "-q", "origin", BASE)
        git(wt, "checkout", "-q", "-B", BASE, f"origin/{BASE}")
    else:
        git(wt, "checkout", "-q", BASE)


if __name__ == "__main__":
    main()
