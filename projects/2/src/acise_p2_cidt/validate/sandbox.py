"""
Sandbox validation (E4): predictions for the runs of a registry target outside the
pool, recorded before each run ends, and their report once the runs are over.
"""

import argparse
import json
import pathlib
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from acise_p2_cidt.collect import http
from acise_p2_cidt.collect.gh_runs import fetch_windows
from acise_p2_cidt.collect.git_changes import clone
from acise_p2_cidt.collect.targets import TargetConfig, load_targets, slug
from acise_p2_cidt.data.build import build
from acise_p2_cidt.data.load import PREV_LABEL_COL
from acise_p2_cidt.metrics.metrics import confusion, domain, scores
from acise_p2_cidt.utils.logging.log import log
from acise_p2_cidt.validate import live

# projects/2/results/sandbox
SANDBOX_DIR = live.MODEL_DIR.parent / "sandbox"
SANDBOX_SLUG = (
    "culbec__ACISE"  # prefix; the workflow id is added once the workflow exists
)


def target(prefix: str = SANDBOX_SLUG) -> TargetConfig:
    """The registry entry of the sandbox: the one whose slug starts with `prefix`."""
    found = [
        c for c in load_targets(enabled_only=False) if slug(c.target).startswith(prefix)
    ]
    if len(found) != 1:
        raise ValueError(
            f"expected one registry entry {prefix}__<workflow id>, got {len(found)}"
        )
    if found[0].enabled:
        raise ValueError(
            "the sandbox target must stay disabled: it is never trained on"
        )
    return found[0]


def refresh(cfg: TargetConfig, since: datetime) -> None:
    """Runs since `since` into the cache, and the new commits into the clone."""
    fetch_windows(cfg.target, since.date(), datetime.now(UTC).date())
    clone(cfg.target.repo, cfg.target.branch)


def predict(path: pathlib.Path, run: dict, cfg: TargetConfig) -> dict:
    """
    Prediction for one run of a target outside the pool, built from that target's
    own cached history with its own `TargetConfig`. The cache should be refreshed
    (`refresh`) before the run was created; the run itself is passed in, so a run
    still in progress is predicted as `live.predict_run` does it.

    Returns:
        dict: `live.predict_run`'s fields plus imputed (features filled with the
            training medians), prev_failed (Parrot's input), majority, in_pool,
            and predicted_at.
    """
    est, side = live.load_model(path)
    s = slug(cfg.target)
    pending = run["status"] != "completed"
    if pending:
        run = run | {"status": "completed", "conclusion": "success",
                     "run_attempt": 1, "updated_at": run["created_at"]}  # fmt: skip
    df, _ = build(cfg, select=lambda r: r["run_id"] == run["id"], extra=[run])
    if df.empty:
        raise ValueError(
            f"run {run['id']} is not kept by the build: {run['conclusion']}"
        )
    out = live._explain(est, side, s, run, df, pending)
    raw = df[side["features"]].astype(float).iloc[0]
    prev = df[PREV_LABEL_COL].iat[0]
    return out | {
        "repo": cfg.target.repo,
        "event": run["event"],
        "imputed": [f for f in side["features"] if np.isnan(raw[f])],
        "prev_failed": None if pd.isna(prev) else int(prev),
        "majority": bool(side["train_fail_rate"] > 0.5),
        "in_pool": s in [e["slug"] for e in side["sources"]],
        "model": path.name,
        "predicted_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def record(path: pathlib.Path, row: dict) -> None:
    """Appends one prediction as a JSON line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def first_attempt(repo: str, run_id: int) -> dict:
    """The first attempt of a run (status, conclusion, updated_at)."""
    return http.get(f"/repos/{repo}/actions/runs/{run_id}/attempts/1")


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s).tz_convert(UTC)  # API "...Z" and isoformat "+00:00" alike


def report(
    path: pathlib.Path, attempt: Callable[[str, int], dict] | None = None
) -> dict:
    """
    Scores the recorded predictions against each run's first-attempt outcome.

    A row predicted after its run completed is dropped and counted, as is a run
    that is not over or ended other than passed or failed (`startup_failure`,
    `cancelled`, ...).

    Args:
        path (pathlib.Path): `predictions.jsonl`.
        attempt (Callable | None): (repo, run id) -> first attempt; None = the API.

    Returns:
        dict: scores (model, parrot, majority on the same runs), confusion (model),
            runs (one row per scored run), by_breakage, imputed (share of the
            predictions that imputed each feature), dropped (counts per reason).
    """
    attempt = attempt or first_attempt
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    kept, dropped = [], Counter()
    for r in rows:
        a = attempt(r["repo"], r["run_id"])
        if a["status"] != "completed":
            dropped["not_completed"] += 1
        elif _ts(r["predicted_at"]) > _ts(a["updated_at"]):
            dropped["predicted_after_completion"] += 1
        elif a["conclusion"] not in ("success", "failure"):
            dropped[f"conclusion_{a['conclusion']}"] += 1
        else:
            sc = r.get("scenario") or {}
            kept.append({
                "scenario": sc.get("n"), "shape": sc.get("shape"), "breakage": sc.get("breakage"),
                "event": r.get("event"), "run_id": r["run_id"],
                "failed": int(a["conclusion"] == "failure"), "p_fail": r["p_fail"],
                "pred": int(r["predicted"]), "parrot": int(r["prev_failed"] == 1),
                "majority": int(r["majority"]), "n_imputed": len(r["imputed"]),
                "lead_secs": (_ts(a["updated_at"]) - _ts(r["predicted_at"])).total_seconds(),
            })  # fmt: skip
    runs = pd.DataFrame(kept)
    if runs.empty:
        raise ValueError(f"no scorable run in {path}: dropped {dict(dropped)}")
    y = runs["failed"].to_numpy()
    model = rows[0]["model"].split("_")[0]
    scored = pd.DataFrame(
        {"method": m, "n_test": len(y), "n_fail": int(y.sum()),
         **scores(y, runs[c].to_numpy()), **domain(y, runs[c].to_numpy())}
        for m, c in ((model, "pred"), ("parrot", "parrot"), ("majority", "majority"))
    ).set_index("method")  # fmt: skip
    c = confusion(y, runs["pred"].to_numpy())
    conf = pd.DataFrame([[c.tn, c.fp], [c.fn, c.tp]],
                        index=["passed", "failed"], columns=["predicted pass", "predicted fail"])  # fmt: skip
    conf.index.name = "actual"
    by = runs.groupby("breakage").agg(
        runs=("run_id", "size"), failed=("failed", "sum"), predicted_fail=("pred", "sum"),
        mean_p_fail=("p_fail", "mean"),
    )  # fmt: skip
    imputed = (
        pd.Series([f for r in rows for f in r["imputed"]], dtype=str)
        .value_counts().div(len(rows)).rename("share imputed").to_frame()
    )  # fmt: skip
    return {"scores": scored, "confusion": conf, "runs": runs, "by_breakage": by,
            "imputed": imputed, "dropped": dict(dropped)}  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(description="Score recorded sandbox predictions.")
    p.add_argument("predictions", type=pathlib.Path, nargs="?",
                   default=SANDBOX_DIR / "predictions.jsonl")  # fmt: skip
    rep = report(p.parse_args().predictions)
    for k in ("scores", "confusion", "by_breakage", "imputed"):
        log.print(f"{k}\n{rep[k].round(3).to_string()}\n")
    log.print(f"dropped: {rep['dropped']}")
