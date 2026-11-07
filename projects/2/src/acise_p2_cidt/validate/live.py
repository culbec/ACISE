import argparse
import hashlib
import json
import pathlib
import subprocess
from datetime import UTC, datetime
from typing import Literal

import numpy as np
import pandas as pd
import skops.io as sio

from acise_p2_cidt.analysis.features import load_features
from acise_p2_cidt.collect import git_changes as git
from acise_p2_cidt.collect import http
from acise_p2_cidt.collect.gh_runs import fetch_details, fetch_windows
from acise_p2_cidt.collect.targets import Target
from acise_p2_cidt.data.build import build
from acise_p2_cidt.data.load import LABEL_COL, PREV_LABEL_COL, load
from acise_p2_cidt.data.load import Target as Label
from acise_p2_cidt.metrics.metrics import domain, scores
from acise_p2_cidt.training.cv import Fold, prepare
from acise_p2_cidt.tree import forest, tune
from acise_p2_cidt.tree.cart import CART
from acise_p2_cidt.tree.forest import RandomForest
from acise_p2_cidt.utils.logging.log import get_logger, log

Model = Literal["cart", "forest"]

# Share of the final training rows held out to choose alpha and the threshold,
# as the inner validation tail of every CV fold.
VAL_FRAC: float = 0.2
# The only types skops may rebuild beyond its defaults; anything else in a file
# means it was not written by `train_final`, and loading stops.
TRUSTED: frozenset[str] = frozenset(
    {
        "acise_p2_cidt.tree.cart.CART",
        "acise_p2_cidt.tree.cart.Tree",
        "acise_p2_cidt.tree.forest.RandomForest",
    }
)

# projects/2/src/acise_p2_cidt/validate/live.py -> projects/2/results/model
MODEL_DIR = pathlib.Path(__file__).resolve().parents[3] / "results" / "model"

logger = get_logger(__name__)


# =============================================================================
# Saving and loading
# =============================================================================


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_version() -> str:
    # "-dirty" marks a model trained from uncommitted code.
    cmd = ["describe", "--always", "--dirty", "--abbrev=40"]
    return subprocess.run(
        ["git", "-C", str(pathlib.Path(__file__).parent), *cmd],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def model_path(dataset: str, target: Label, model: Model) -> pathlib.Path:
    """`results/model/<model>_<dataset>_<target>.skops`."""
    return MODEL_DIR / f"{model}_{dataset}_{target}.skops"


def save_model(
    est: CART | RandomForest, path: pathlib.Path, sidecar: dict
) -> pathlib.Path:
    """
    Writes the model with skops and a `.json` sidecar holding its SHA-256.

    Args:
        est (CART | RandomForest): fitted model.
        path (pathlib.Path): `.skops` file.
        sidecar (dict): metadata saved next to it.

    Returns:
        pathlib.Path: the `.skops` file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(est, path)
    untrusted = set(sio.get_untrusted_types(file=path))
    if not untrusted <= TRUSTED:
        path.unlink()
        raise ValueError(f"model holds types outside TRUSTED: {untrusted - TRUSTED}")
    sidecar = sidecar | {"sha256": _sha256(path)}
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    return path


def load_model(path: pathlib.Path) -> tuple[CART | RandomForest, dict]:
    """
    Loads a model written by `save_model`, after checking its hash and types.

    Args:
        path (pathlib.Path): `.skops` file.

    Returns:
        tuple[CART | RandomForest, dict]: the model and its sidecar.
    """
    sidecar = json.loads(path.with_suffix(".json").read_text())
    if _sha256(path) != sidecar["sha256"]:
        raise ValueError(f"{path} does not match the SHA-256 in its sidecar")
    untrusted = set(sio.get_untrusted_types(file=path))
    if not untrusted <= TRUSTED:
        raise ValueError(f"{path} holds untrusted types {untrusted - TRUSTED}")
    return sio.load(path, trusted=sorted(untrusted)), sidecar


# =============================================================================
# Final model
# =============================================================================


def train_final(
    dataset: str,
    target: Label,
    model: Model = "cart",
    params: dict | None = None,
    features: list[str] | None = None,
) -> pathlib.Path:
    """
    Fits the frozen model on every run collected up to C and saves it.

    Same protocol as one CV fold: fit on the head, choose alpha and the threshold
    on the last `VAL_FRAC` rows, no refit, so both stay matched to the saved model.

    Args:
        dataset (str): target slug.
        target (Label): "T1" or "T2".
        model (Model): "cart" or "forest".
        params (dict | None): fixed model parameters; None runs the grid search.
        features (list[str] | None): None reads the frozen selection.

    Returns:
        pathlib.Path: the `.skops` file.
    """
    if model not in ("cart", "forest"):
        raise ValueError(f"model must be 'cart' or 'forest', got {model!r}")
    features = features or load_features(target, dataset)
    ds = load(dataset, target)
    n = len(ds.y)
    cut = n - int(n * VAL_FRAC)
    split = prepare(ds, features, Fold(-1, np.arange(cut), np.arange(cut, n),
                                       np.arange(cut, n)))  # fmt: skip
    fit_predict = tune.fit_predict if model == "cart" else forest.fit_predict
    payload = fit_predict(split, params).payload
    est, choice = payload["tree" if model == "cart" else "forest"], payload["choice"]
    sidecar = {
        "dataset": dataset,
        "repo": ds.meta.get("repo"),
        "workflow_id": ds.meta.get("workflow_id"),
        "branch": ds.meta.get("branch"),
        "target": target,
        "model": model,
        "features": features,
        "params": choice.params,
        "alpha": choice.alpha,
        "threshold": choice.threshold,
        "val_mcc": choice.val_mcc,
        "collected_at": ds.meta.get("collected_at"),
        # C: runs created after the last training run are new to the model
        "cutoff": ds.order.iloc[-1].isoformat(),
        "n_train": cut,
        "n_val": n - cut,
        "train_fail_rate": float(split.y_fit.mean()),
        # training medians: a new run is imputed with these, never its own
        "medians": dict(zip(features, split.prep.statistics_.tolist())),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_version": _code_version(),
    }
    logger.info(f"{model} {target}: {choice.params}, val mcc {choice.val_mcc:.3f}")
    return save_model(est, model_path(dataset, target, model), sidecar)


# =============================================================================
# Live data
# =============================================================================


def _target(sidecar: dict) -> Target:
    return Target(sidecar["repo"], sidecar["workflow_id"], sidecar["branch"])


def _refresh(t: Target, cutoff: pd.Timestamp) -> None:
    """Runs created since C into the cache, and the new commits into the clone."""
    fetch_windows(t, cutoff.date(), datetime.now(UTC).date())
    git.clone(t.repo, t.branch)


def _X(df: pd.DataFrame, sidecar: dict) -> pd.DataFrame:
    """Saved features in saved order, imputed with the training medians."""
    return df[sidecar["features"]].astype(float).fillna(sidecar["medians"])


def _in_target(df: pd.DataFrame, target: str) -> pd.Series:
    # T2 holds only runs whose previous completed run passed, as in `load`.
    return df[PREV_LABEL_COL] == 0 if target == "T2" else pd.Series(True, df.index)


def retest(path: pathlib.Path) -> pd.DataFrame:
    """
    Scores the saved model, unchanged, on the runs created after C.

    New runs are fetched, and their features are built from the whole history
    (old runs and earlier new runs, never later ones). Parrot and majority are
    scored on the same rows. Writes `<model>_retest.csv` (metrics) and
    `<model>_retest_runs.csv` (one row per run, for picking case studies).

    Args:
        path (pathlib.Path): `.skops` file from `train_final`.

    Returns:
        pd.DataFrame: one row per method, with the `run_cv` columns.
    """
    est, side = load_model(path)
    t, cutoff = _target(side), pd.Timestamp(side["cutoff"])
    _refresh(t, cutoff)
    df, _ = build(t, select=lambda r: r["created_at"] > cutoff.timestamp())
    df = df[_in_target(df, side["target"])]
    if df.empty:
        raise ValueError(f"no {side['target']} runs created after {cutoff}")

    y = df[LABEL_COL].astype(int).to_numpy()
    proba = est.predict_proba(_X(df, side))[:, 1]
    preds = {
        side["model"]: proba > side["threshold"],
        "parrot": (df[PREV_LABEL_COL] == 1).to_numpy(),
        "majority": np.full(len(df), side["train_fail_rate"] > 0.5),
    }
    out = pd.DataFrame(
        {"method": m, "fold": -1, "n_test": len(y), "n_fail": int(y.sum()),
         **scores(y, p), **domain(y, p)}
        for m, p in preds.items()
    )  # fmt: skip
    stem = path.with_suffix("")
    out.to_csv(f"{stem}_retest.csv", index=False)
    df[["run_id", "head_sha", "created_at", LABEL_COL]].assign(
        p_fail=proba, pred=preds[side["model"]]
    ).to_csv(f"{stem}_retest_runs.csv", index=False)
    return out


def _fetch_run(t: Target, ref: str) -> dict:
    """One run of the workflow by id, or the latest master run of a commit SHA."""
    if ref.isdigit():
        run = http.get(f"/repos/{t.repo}/actions/runs/{ref}")
    else:
        runs = http.get(
            f"/repos/{t.repo}/actions/workflows/{t.workflow_id}/runs",
            {"head_sha": ref, "branch": t.branch, "event": "push"},
        )["workflow_runs"]
        if not runs:
            raise ValueError(f"no {t.branch} run of workflow {t.workflow_id} for {ref}")
        run = runs[0]  # newest first
    if run["workflow_id"] != t.workflow_id or run["head_branch"] != t.branch:
        raise ValueError(
            f"run {run['id']} is workflow {run['workflow_id']} on "
            f"{run['head_branch']!r}, the model {t.workflow_id} on {t.branch!r}"
        )
    return run


def predict_run(path: pathlib.Path, ref: str) -> dict:
    """
    Prediction and decision path for one run, finished or still running.

    A run still in progress is built as if it completed when it was created: its
    features read only runs completed before its creation, so the stand-in outcome
    never reaches them, and its label is reported as unknown.

    Args:
        path (pathlib.Path): `.skops` file from `train_final`.
        ref (str): run id, or commit SHA (its latest run).

    Returns:
        dict: run_id, sha, url, created_at, label (None while running), p_fail,
            predicted (True = will fail), threshold, in_target (False: outside
            the model's training population, e.g. after a failure on T2), and
            path (`CART.explain` lines; None for a forest).
    """
    est, side = load_model(path)
    t = _target(side)
    run = _fetch_run(t, ref)
    pending = run["status"] != "completed"
    if pending:
        run = run | {"status": "completed", "conclusion": "success",
                     "run_attempt": 1, "updated_at": run["created_at"]}  # fmt: skip
    else:
        fetch_details(t, [run])
    _refresh(t, pd.Timestamp(side["cutoff"]))
    df, _ = build(t, select=lambda r: r["run_id"] == run["id"], extra=[run])
    if df.empty:
        raise ValueError(
            f"run {run['id']} is not kept by the build: {run['conclusion']}"
        )
    X = _X(df, side)
    p = float(est.predict_proba(X)[0, 1])
    return {
        "run_id": run["id"],
        "sha": run["head_sha"],
        "url": run["html_url"],
        "created_at": run["created_at"],
        "label": None if pending else int(df[LABEL_COL].iat[0]),
        "p_fail": p,
        "predicted": p > side["threshold"],
        "threshold": side["threshold"],
        "in_target": bool(_in_target(df, side["target"]).iat[0]),
        "path": est.explain(X) if isinstance(est, CART) else None,
    }


# =============================================================================
# CLIs
# =============================================================================


def train_final_main() -> None:
    p = argparse.ArgumentParser(description="Fit and save the frozen model.")
    p.add_argument("--dataset", required=True, help="target slug")
    p.add_argument("--target", choices=["T1", "T2"], default="T2")
    p.add_argument("--model", choices=["cart", "forest"], default="cart")
    p.add_argument("--params", type=json.loads, help="JSON; omit to grid-search")
    a = p.parse_args()
    log.print(f"saved {train_final(a.dataset, a.target, a.model, a.params)}")


def retest_main() -> None:
    p = argparse.ArgumentParser(description="Re-test a saved model on new runs.")
    p.add_argument("model", type=pathlib.Path, help=".skops file")
    log.print(retest(p.parse_args().model).to_string(index=False))


def predict_run_main() -> None:
    p = argparse.ArgumentParser(description="Explain the prediction for one run.")
    p.add_argument("ref", help="run id or commit SHA")
    p.add_argument("--model", type=pathlib.Path, required=True, help=".skops file")
    a = p.parse_args()
    r = predict_run(a.model, a.ref)
    label = {None: "running", 0: "passed", 1: "failed"}[r["label"]]
    log.print(f"run {r['run_id']} ({r['sha'][:10]}, {label}) {r['url']}")
    for line in r["path"] or []:
        log.print(f"  {line}")
    verdict = "FAIL" if r["predicted"] else "pass"
    log.print(f"P(fail) = {r['p_fail']:.2f} -> {verdict} at {r['threshold']:.2f}")
    if not r["in_target"]:
        log.print("note: run is outside the model's target population")
