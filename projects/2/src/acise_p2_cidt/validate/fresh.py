"""
Fresh-repository test: the frozen pooled models, unchanged, on repositories that
are not in their pool. Only a short window is collected per repository: a warm-up
that feeds the history features, then the scored test days.
"""

import json
import pathlib
from dataclasses import dataclass
from datetime import UTC, date

import pandas as pd

from acise_p2_cidt.collect import gh_runs, http
from acise_p2_cidt.collect import git_changes as git
from acise_p2_cidt.collect.targets import TargetConfig, _parse, slug
from acise_p2_cidt.data.build import build
from acise_p2_cidt.utils.logging.log import get_logger
from acise_p2_cidt.validate import live

# projects/2/src/acise_p2_cidt/validate/fresh.py -> projects/2/config/fresh.json
CONFIG = pathlib.Path(__file__).resolve().parents[3] / "config" / "fresh.json"
# The whole collection, warm-up included, stays within two weeks.
MAX_DAYS: int = 14

logger = get_logger(__name__)


@dataclass(frozen=True)
class FreshConfig:
    since: date  # first warm-up day
    test_since: date  # first scored day
    until: date  # last scored day
    targets: list[TargetConfig]


def load_config(path: pathlib.Path | None = None) -> FreshConfig:
    """
    Reads `config/fresh.json`: the dates, and targets in the `targets.json` format.

    Args:
        path (pathlib.Path | None): config file; None = `CONFIG`.

    Returns:
        FreshConfig: the dates and the enabled targets.
    """
    c = json.loads((path or CONFIG).read_text())
    since, test_since, until = (
        date.fromisoformat(c[k]) for k in ("since", "test_since", "until")
    )
    if not since < test_since <= until:
        raise ValueError("need since < test_since <= until")
    if (until - since).days + 1 > MAX_DAYS:
        raise ValueError(f"collection spans more than {MAX_DAYS} days")
    cfgs = [_parse(i, e) for i, e in enumerate(c["targets"])]
    return FreshConfig(since, test_since, until, [c for c in cfgs if c.enabled])


def _clone(cfg: TargetConfig) -> None:
    """
    The default branch only. The pool's clones also mirror `refs/pull/*`, which is
    gigabytes on large repositories; `build` fetches the PR heads it needs by SHA.
    """
    t = cfg.target
    if not (gh_runs.repo_root(t.repo) / "repo.git").exists():
        git.clone(t.repo, t.branch or http.get(f"/repos/{t.repo}")["default_branch"])


def fresh_runs(fc: FreshConfig) -> pd.DataFrame:
    """
    Built rows of every target's runs created in `[test_since, until]`, with a
    `source` column. Their history features read the warm-up runs and earlier test
    runs of the same target, never later ones.
    """
    start = pd.Timestamp(fc.test_since, tz=UTC).timestamp()
    parts = []
    for cfg in fc.targets:
        gh_runs.collect(cfg.target, fc.since, fc.until)
        _clone(cfg)
        df, _ = build(cfg, select=lambda r: r["created_at"] >= start)
        parts.append(df.assign(source=slug(cfg.target)))
        logger.info("%s: %d test runs", slug(cfg.target), len(df))
    df = pd.concat(parts, ignore_index=True)
    return df.sort_values(["created_at", "source"]).reset_index(drop=True)


def fresh_test(
    paths: list[pathlib.Path], fc: FreshConfig | None = None
) -> pd.DataFrame:
    """
    Scores saved models, unchanged, on repositories outside their pool.

    Writes `<model>_fresh.csv` and `<model>_fresh_runs.csv` next to each model, and
    the built rows to `fresh_runs_<dataset>.csv`.

    Args:
        paths (list[pathlib.Path]): `.skops` files from `train_final`, one pool.
        fc (FreshConfig | None): None reads `CONFIG`.

    Returns:
        pd.DataFrame: as `live.retest`: one row per model's method, source and
            target.
    """
    fc = fc or load_config()
    models = [live.load_model(p) for p in paths]
    side = models[0][1]
    seen = {e["repo"].lower() for _, s in models for e in s["sources"]}
    if clash := sorted(
        c.target.repo for c in fc.targets if c.target.repo.lower() in seen
    ):
        raise ValueError(f"repositories in the training pool: {clash}")
    df = fresh_runs(fc)
    df.to_csv(live.MODEL_DIR / f"fresh_runs_{side['dataset']}.csv", index=False)
    return pd.concat(
        [live._score(p, est, s, df, "fresh") for p, (est, s) in zip(paths, models)],
        ignore_index=True,
    )
