import argparse
import hashlib
import json
import pathlib
import subprocess
from datetime import UTC, date, datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
import skops.io as sio

from acise_p2_cidt.analysis.features import load_features
from acise_p2_cidt.collect import gh_runs, http
from acise_p2_cidt.collect import git_changes as git
from acise_p2_cidt.collect.gh_runs import fetch_details, fetch_windows
from acise_p2_cidt.collect.targets import Target, TargetConfig, load_targets, slug
from acise_p2_cidt.data.build import build
from acise_p2_cidt.data.load import (
    LABEL_COL,
    PREV_LABEL_COL,
    in_t2,
    load,
    pool_name,
    schema_path,
)
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


def collection_until(sources: list[str]) -> date:
    """
    The collection's last day, read from each source's manifest; the sources of
    one pool must share it.
    """
    days = {
        s: json.loads((gh_runs.RESULTS_DIR / f"manifest_{s}.json").read_text())["until"]
        for s in sources
    }
    if len(set(days.values())) != 1:
        raise ValueError(f"sources were collected up to different days: {days}")
    return date.fromisoformat(next(iter(days.values())))


def _source_entry(s: str) -> dict:
    """What `retest` and `predict_run` need to rebuild a source's runs."""
    sc = json.loads(schema_path(s).read_text())
    return {
        "slug": s,
        "repo": sc["repo"],
        "workflow_id": sc["workflow_id"],
        "branch": sc["branch"],
        "exclude_jobs": list(sc.get("exclude_jobs", [])),
    }


def train_final(
    sources: str | list[str],
    target: Label,
    model: Model = "cart",
    params: dict | None = None,
    features: list[str] | None = None,
    until: date | None = None,
) -> pathlib.Path:
    """
    Fits the frozen model on every run collected up to C and saves it.

    Same protocol as one CV fold: fit on the head, choose alpha and the threshold
    on the last `VAL_FRAC` rows, no refit, so both stay matched to the saved model.
    One pooled threshold, as in the CV.

    Args:
        sources (str | list[str]): target slug, or the slugs of a pool.
        target (Label): "T1" or "T2".
        model (Model): "cart" or "forest".
        params (dict | None): fixed model parameters; None runs the grid search.
        features (list[str] | None): None reads the frozen selection.
        until (date | None): the collection's last day; C is the end of it.
            None reads the manifests.

    Returns:
        pathlib.Path: the `.skops` file.
    """
    if model not in ("cart", "forest"):
        raise ValueError(f"model must be 'cart' or 'forest', got {model!r}")
    slugs = [sources] if isinstance(sources, str) else list(sources)
    dataset = slugs[0] if isinstance(sources, str) else pool_name(slugs)
    features = features or load_features(target, dataset)
    until = until or collection_until(slugs)
    cutoff = pd.Timestamp(until, tz=UTC) + timedelta(days=1)  # C
    ds = load(sources, target)
    if ds.order.iloc[-1] >= cutoff:
        raise ValueError(f"training runs created after C ({cutoff})")
    n = len(ds.y)
    cut = n - int(n * VAL_FRAC)
    split = prepare(ds, features, Fold(-1, np.arange(cut), np.arange(cut, n),
                                       np.arange(cut, n)))  # fmt: skip
    fit_predict = tune.fit_predict if model == "cart" else forest.fit_predict
    payload = fit_predict(split, params).payload
    est, choice = payload["tree" if model == "cart" else "forest"], payload["choice"]
    sidecar = {
        "dataset": dataset,
        "sources": [_source_entry(s) for s in slugs],
        "target": target,
        "model": model,
        "features": features,
        "params": choice.params,
        "alpha": choice.alpha,
        "threshold": choice.threshold,
        "val_mcc": choice.val_mcc,
        "collection_until": until.isoformat(),
        # C, the end of the collection's last day: runs created at or after it
        # are new to the model
        "cutoff": cutoff.isoformat(),
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


def _config(entry: dict) -> TargetConfig:
    t = Target(entry["repo"], entry["workflow_id"], entry["branch"])
    return TargetConfig(t, True, tuple(entry["exclude_jobs"]))


def _source(sidecar: dict, s: str) -> TargetConfig:
    """The sidecar's source `s`; a source the model was not trained on is refused."""
    for e in sidecar["sources"]:
        if e["slug"] == s:
            return _config(e)
    known = [e["slug"] for e in sidecar["sources"]]
    raise ValueError(f"{s!r} is not a source of this model: {known}")


def _refresh(t: Target, cutoff: pd.Timestamp) -> None:
    """Runs created since C into the cache, and the new commits into the clone."""
    fetch_windows(t, cutoff.date(), datetime.now(UTC).date())
    git.clone(t.repo, t.branch)


def _X(df: pd.DataFrame, sidecar: dict) -> pd.DataFrame:
    """Saved features in saved order, imputed with the training medians."""
    return df[sidecar["features"]].astype(float).fillna(sidecar["medians"])


def _in_target(df: pd.DataFrame, target: str) -> pd.Series:
    # same T2 rule as `load`
    return in_t2(df) if target == "T2" else pd.Series(True, df.index)


def _new_runs_path(path: pathlib.Path, side: dict) -> pathlib.Path:
    """Built rows of every run after C, written by `retest` next to the models."""
    return path.parent / f"new_runs_{side['dataset']}.csv"


def _new_runs(side: dict) -> pd.DataFrame:
    """
    Every source's runs created after C, all targets, with a `source` column. Each
    source is refreshed and built once; its features read the whole history of
    that source (old runs and earlier new runs, never later ones).
    """
    cutoff = pd.Timestamp(side["cutoff"])
    parts = []
    for e in side["sources"]:
        cfg = _config(e)
        _refresh(cfg.target, cutoff)
        df, _ = build(cfg, select=lambda r: r["created_at"] >= cutoff.timestamp())
        parts.append(df.assign(source=e["slug"]))
    df = pd.concat(parts, ignore_index=True)
    return df.sort_values(["created_at", "source"]).reset_index(drop=True)


def _score(path: pathlib.Path, est, side: dict, df: pd.DataFrame) -> pd.DataFrame:
    """One model's re-test rows (pooled and per source); writes its two CSVs."""
    df = df[_in_target(df, side["target"])]
    if df.empty:
        raise ValueError(f"no {side['target']} runs created after {side['cutoff']}")
    proba = est.predict_proba(_X(df, side))[:, 1]
    preds = {
        side["model"]: proba > side["threshold"],
        "parrot": (df[PREV_LABEL_COL] == 1).to_numpy(),
        "majority": np.full(len(df), side["train_fail_rate"] > 0.5),
    }
    y = df[LABEL_COL].astype(int).to_numpy()
    src = df["source"].to_numpy()
    rows = []
    for s in ["all", *sorted(set(src))]:
        m = np.ones(len(y), bool) if s == "all" else src == s
        rows += [
            {"method": k, "fold": -1, "source": s, "n_test": int(m.sum()),
             "n_fail": int(y[m].sum()), **scores(y[m], p[m]), **domain(y[m], p[m])}
            for k, p in preds.items()
        ]  # fmt: skip
    out = pd.DataFrame(rows)
    stem = path.with_suffix("")
    out.to_csv(f"{stem}_retest.csv", index=False)
    df[["run_id", "source", "head_sha", "created_at", LABEL_COL]].assign(
        p_fail=proba, pred=preds[side["model"]]
    ).to_csv(f"{stem}_retest_runs.csv", index=False)
    return out.assign(target=side["target"])


def retest(paths: pathlib.Path | list[pathlib.Path]) -> pd.DataFrame:
    """
    Scores saved models, unchanged, on the runs created after C.

    The models must share their sources and C: the new runs are fetched and built
    once, then every model is scored on them. Parrot and majority are scored on the
    same rows as each model. As in `run_cv`, the scores are given pooled
    (`source = "all"`) and per source; a source without new runs has no row.
    Writes `<model>_retest.csv` (metrics) and `<model>_retest_runs.csv` (one row
    per run, for picking case studies) next to each model.

    Args:
        paths (pathlib.Path | list[pathlib.Path]): `.skops` files from
            `train_final`.

    Returns:
        pd.DataFrame: one row per model's method, source and target, with the
            `run_cv` columns and `target`; Parrot and majority repeat per model.
    """
    paths = [paths] if isinstance(paths, pathlib.Path) else list(paths)
    models = [load_model(p) for p in paths]
    side = models[0][1]
    for p, (_, s) in zip(paths, models):
        if (s["sources"], s["cutoff"]) != (side["sources"], side["cutoff"]):
            raise ValueError(f"{p.name}: other sources or C than {paths[0].name}")
    df = _new_runs(side)
    # `predict_run` reads a re-tested run's row from here instead of rebuilding
    df.to_csv(_new_runs_path(paths[0], side), index=False)
    return pd.concat(
        [_score(p, est, s, df) for p, (est, s) in zip(paths, models)],
        ignore_index=True,
    )


def _fetch_run(t: Target, ref: str) -> dict:
    """
    One run of the workflow by id, or the latest run of a commit SHA (on a
    branch-scoped target: its latest push run on that branch).
    """
    if ref.isdigit():
        run = http.get(f"/repos/{t.repo}/actions/runs/{ref}")
    else:
        q = {"head_sha": ref}
        if t.branch is not None:
            q |= {"branch": t.branch, "event": "push"}
        runs = http.get(f"/repos/{t.repo}/actions/workflows/{t.workflow_id}/runs", q)[
            "workflow_runs"
        ]
        if not runs:
            raise ValueError(f"no run of workflow {t.workflow_id} for {ref}")
        run = runs[0]  # newest first
    if run["workflow_id"] != t.workflow_id:
        raise ValueError(
            f"run {run['id']} is workflow {run['workflow_id']}, "
            f"the source {t.workflow_id}"
        )
    if t.branch is not None and run["head_branch"] != t.branch:
        raise ValueError(
            f"run {run['id']} is on {run['head_branch']!r}, the source on {t.branch!r}"
        )
    return run


def predict_run(path: pathlib.Path, ref: str, source: str) -> dict:
    """
    Prediction and decision path for one run, finished or still running.

    A run already re-tested by `retest` is read from its saved row, with no fetch
    or build. Any other run is fetched and its source rebuilt. A run still in
    progress is built as if it completed when it was created: its features read
    only runs completed before its creation, so the stand-in outcome never reaches
    them, and its label is reported as unknown.

    Args:
        path (pathlib.Path): `.skops` file from `train_final`.
        ref (str): run id, or commit SHA (its latest run).
        source (str): slug of the run's source; it must be one of the sidecar's.

    Returns:
        dict: source, run_id, sha, url, created_at, label (None while running), p_fail,
            predicted (True = will fail), threshold, in_target (False: outside
            the model's training population, e.g. after a failure on T2), and
            path (`CART.explain` lines; None for a forest).
    """
    est, side = load_model(path)
    cfg = _source(side, source)
    t = cfg.target
    cached = _new_runs_path(path, side)
    if ref.isdigit() and cached.exists():
        rows = pd.read_csv(cached)
        df = rows[(rows["run_id"] == int(ref)) & (rows["source"] == source)]
        if len(df):
            meta = {"id": int(ref), "head_sha": df["head_sha"].iat[0],
                    "html_url": f"https://github.com/{t.repo}/actions/runs/{ref}",
                    "created_at": str(df["created_at"].iat[0])}  # fmt: skip
            return _explain(est, side, source, meta, df, pending=False)
    run = _fetch_run(t, ref)
    pending = run["status"] != "completed"
    if pending:
        run = run | {"status": "completed", "conclusion": "success",
                     "run_attempt": 1, "updated_at": run["created_at"]}  # fmt: skip
    else:
        fetch_details(t, [run])
    _refresh(t, pd.Timestamp(side["cutoff"]))
    df, _ = build(cfg, select=lambda r: r["run_id"] == run["id"], extra=[run])
    if df.empty:
        raise ValueError(
            f"run {run['id']} is not kept by the build: {run['conclusion']}"
        )
    return _explain(est, side, source, run, df, pending)


def _explain(
    est, side: dict, source: str, run: dict, df: pd.DataFrame, pending: bool
) -> dict:
    """`predict_run`'s result for the run's one built row `df`."""
    X = _X(df, side)
    p = float(est.predict_proba(X)[0, 1])
    return {
        "source": source,
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
    p.add_argument(
        "--sources",
        nargs="+",
        help="target slugs to pool; omit for the enabled targets",
    )
    p.add_argument("--target", choices=["T1", "T2"], default="T2")
    p.add_argument("--model", choices=["cart", "forest"], default="cart")
    p.add_argument("--params", type=json.loads, help="JSON; omit to grid-search")
    a = p.parse_args()
    sources = a.sources or [slug(c.target) for c in load_targets()]
    log.print(f"saved {train_final(sources, a.target, a.model, a.params)}")


def retest_main() -> None:
    p = argparse.ArgumentParser(description="Re-test saved models on new runs.")
    p.add_argument("models", type=pathlib.Path, nargs="+", help=".skops files")
    log.print(retest(p.parse_args().models).to_string(index=False))


def predict_run_main() -> None:
    p = argparse.ArgumentParser(description="Explain the prediction for one run.")
    p.add_argument("ref", help="run id or commit SHA")
    p.add_argument("--model", type=pathlib.Path, required=True, help=".skops file")
    p.add_argument("--target", required=True, help="slug of the run's source")
    a = p.parse_args()
    r = predict_run(a.model, a.ref, a.target)
    label = {None: "running", 0: "passed", 1: "failed"}[r["label"]]
    log.print(f"run {r['run_id']} ({r['sha'][:10]}, {label}) {r['url']}")
    for line in r["path"] or []:
        log.print(f"  {line}")
    verdict = "FAIL" if r["predicted"] else "pass"
    log.print(f"P(fail) = {r['p_fail']:.2f} -> {verdict} at {r['threshold']:.2f}")
    if not r["in_target"]:
        log.print("note: run is outside the model's target population")
