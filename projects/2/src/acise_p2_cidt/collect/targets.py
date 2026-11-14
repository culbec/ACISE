import argparse
import json
import pathlib
import re
from dataclasses import dataclass
from typing import NamedTuple

# projects/2/src/acise_p2_cidt/collect/targets.py -> projects/2/config/targets.json
REGISTRY = pathlib.Path(__file__).resolve().parents[3] / "config" / "targets.json"

_REPO = re.compile(r"^[\w.-]+/[\w.-]+$")
_REQUIRED: frozenset[str] = frozenset({"repo", "workflow_id", "enabled"})
_KEYS: frozenset[str] = _REQUIRED | {"branch", "exclude_jobs", "mainline"}


class Target(NamedTuple):
    repo: str  # "owner/name"
    workflow_id: int
    branch: str | None = None  # None = all branches


def slug(t: Target) -> str:
    """
    File key of a target: `owner__name__<workflow_id>`, plus `@<branch>` with `/`
    replaced by `~`.

    Args:
        t (Target): collection target.

    Returns:
        str: e.g. "n8n-io__n8n__33323319@master", "astral-sh__uv__82881044".
    """
    s = f"{t.repo.replace('/', '__')}__{t.workflow_id}"
    return s if t.branch is None else f"{s}@{t.branch.replace('/', '~')}"


@dataclass(frozen=True)
class TargetConfig:
    target: Target
    enabled: bool
    # regexes on "<job>" or "<job> :: <failed step>"; a failed run whose failed
    # jobs all match is infra
    exclude_jobs: tuple[str, ...]
    # branch whose first-parent line the build walks for git features; None = the
    # remote's default branch (an orphan branch shares no history with it)
    mainline: str | None = None

    def excludes(self, job: str | None, step: str | None) -> bool:
        """
        Whether a failed job is infrastructure. `job` None stands for a failed run
        without any failed job: it failed before testing the code, so it is infra.
        """
        if job is None:
            return True
        key = f"{job} :: {step}"
        return any(re.search(p, job) or re.search(p, key) for p in self.exclude_jobs)


def _parse(i: int, e: object) -> TargetConfig:
    if not isinstance(e, dict):
        raise ValueError(f"entry {i}: expected an object, got {type(e).__name__}")  # noqa: TRY004

    where = f"entry {i} ({e.get('repo')!r})"

    if missing := _REQUIRED - e.keys():
        raise ValueError(f"{where}: missing keys {sorted(missing)}")
    if unknown := e.keys() - _KEYS:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")

    repo, wid, branch = e["repo"], e["workflow_id"], e.get("branch")

    if not isinstance(repo, str) or not _REPO.match(repo):
        raise ValueError(f"{where}: repo must be 'owner/name'")
    # bool is an int subclass: `true` would pass as workflow 1
    if isinstance(wid, bool) or not isinstance(wid, int) or wid <= 0:
        raise ValueError(f"{where}: workflow_id must be an int > 0, got {wid!r}")
    if branch is not None and (not isinstance(branch, str) or not branch):
        raise ValueError(f"{where}: branch must be null or a non-empty string")
    mainline = e.get("mainline")
    if mainline is not None and (not isinstance(mainline, str) or not mainline):
        raise ValueError(f"{where}: mainline must be null or a non-empty string")
    if not isinstance(e["enabled"], bool):
        raise ValueError(f"{where}: enabled must be true or false")  # noqa: TRY004

    jobs = e.get("exclude_jobs", [])
    if not isinstance(jobs, list) or not all(isinstance(j, str) for j in jobs):
        raise ValueError(f"{where}: exclude_jobs must be a list of strings")

    for j in jobs:
        try:
            re.compile(j)
        except re.error as err:
            raise ValueError(f"{where}: bad exclude_jobs regex {j!r}: {err}") from None

    return TargetConfig(Target(repo, wid, branch), e["enabled"], tuple(jobs), mainline)


def load_targets(
    path: pathlib.Path | None = None, enabled_only: bool = True
) -> list[TargetConfig]:
    """
    Reads and validates the target registry.

    Args:
        path (pathlib.Path | None): JSON list of targets; None = `REGISTRY`, read
            at call time so tests can patch it.
        enabled_only (bool): drop entries with `enabled: false`.

    Returns:
        list[TargetConfig]: entries in file order.
    """
    path = path or REGISTRY
    entries = json.loads(path.read_text())
    if not isinstance(entries, list):
        raise ValueError(f"{path}: the top level must be a list")  # noqa: TRY004

    cfgs = [_parse(i, e) for i, e in enumerate(entries)]
    seen: dict[str, int] = {}
    for i, c in enumerate(cfgs):
        s = slug(c.target)
        if s in seen:
            raise ValueError(
                f"entry {i} ({c.target.repo!r}): duplicate slug {s} (entry {seen[s]})"
            )
        seen[s] = i

    return [c for c in cfgs if c.enabled or not enabled_only]


def find_target(slug_: str, path: pathlib.Path | None = None) -> TargetConfig:
    """
    The registry entry with this slug, enabled or not.

    Args:
        slug_ (str): target slug.
        path (pathlib.Path | None): registry file; None = `REGISTRY`.

    Returns:
        TargetConfig: the entry.
    """
    for c in load_targets(path, enabled_only=False):
        if slug(c.target) == slug_:
            return c
    raise ValueError(f"no target {slug_!r} in {path or REGISTRY}")


# =============================================================================
# CLI
# =============================================================================


def add_target_args(p: argparse.ArgumentParser) -> None:
    """`--target <slug>`, `--targets <path>`, or `--repo/--workflow [--branch]`."""
    g = p.add_mutually_exclusive_group()
    g.add_argument("--target", help="registry slug, enabled or not")
    g.add_argument("--targets", type=pathlib.Path, help="registry; all enabled")
    g.add_argument("--repo", help="owner/name, with --workflow")
    p.add_argument("--workflow", type=int)
    p.add_argument("--branch", help="omit for all branches")


def targets_from_args(a: argparse.Namespace) -> list[TargetConfig]:
    """The targets named by `add_target_args` options."""
    if a.target:
        return [find_target(a.target)]
    if a.targets:
        return load_targets(a.targets)
    if a.repo and a.workflow:
        return [TargetConfig(Target(a.repo, a.workflow, a.branch), True, ())]
    raise SystemExit("give --target, --targets, or --repo with --workflow")
