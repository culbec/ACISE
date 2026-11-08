import functools
import os
import pathlib
import re
import subprocess
import time
from collections.abc import Container
from typing import Literal

import numpy as np
import pandas as pd

from acise_p2_cidt.collect.gh_runs import repo_root
from acise_p2_cidt.utils.logging.log import get_logger

FileKind = Literal["src", "test", "config", "doc", "ci"]

_TEST_DIRS: frozenset[str] = frozenset({"test", "tests", "__tests__", "spec", "e2e"})
_TEST_NAME = re.compile(r"\.test\.|\.spec\.|^test_.*\.py$|_test\.go$")
_DOC_EXT: tuple[str, ...] = (".md", ".rst", ".txt", ".adoc")
# Lockfiles and manifests of the common ecosystems.
_MANIFESTS: frozenset[str] = frozenset(
    [
        "package.json",
        "pnpm-lock.yaml",
        "package-lock.json",
        "yarn.lock",
        "pyproject.toml",
        "uv.lock",
        "go.mod",
        "go.sum",
        "cargo.toml",
        "cargo.lock",
        "pom.xml",
    ]
)
_CONFIG_EXT: tuple[str, ...] = (".json", ".yml", ".yaml", ".toml", ".ini", ".cfg")
_CONFIG_PREFIX: tuple[str, ...] = (".env", "dockerfile", "tsconfig", "build.gradle")

_RENAME = re.compile(r"\{(.*?) => (.*?)\}")  # `dir/{old => new}/file` in numstat
_REVERTS = re.compile(r"This reverts commit ([0-9a-f]{7,40})")
_REC, _FLD = "\x1e", "\x1f"  # record/field separators: never in paths or messages

LOG_COLS: list[str] = [
    "sha",
    "parents",
    "author_email",
    "author_time",
    "commit_time",
    "subject",
    "reverts",  # SHA named by "This reverts commit <sha>", else None
    "path",  # None for a commit without file changes
    "kind",
    "added",  # NaN for binary files
    "deleted",
]

logger = get_logger(__name__)


# Never wait on a credential prompt: fail instead.
_ENV: dict[str, str] = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}


def _git(repo_dir: pathlib.Path, *args: str, stdin: str | None = None) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_dir), "-c", "core.quotePath=false", *args],
        input=stdin,
        check=True,
        capture_output=True,
        text=True,
        env=_ENV,
    ).stdout


NET_TRIES: int = 5
# stderr of a fetch the remote cannot serve: retrying will not help.
_GONE: tuple[str, ...] = ("not our ref", "couldn't find remote ref", "no such remote ref")


def _git_net(repo_dir: pathlib.Path, *args: str) -> str:
    """
    `_git` for commands that talk to the remote: retries dropped connections
    (early EOF, RPC failed, hung up) and stalled transfers (under 1 KB/s for 60s)
    with exponential backoff. A ref or SHA the remote does not have fails at once.
    A lock left by an interrupted git raises `RuntimeError`, so callers that skip
    missing SHAs do not swallow it.
    """
    for attempt in range(NET_TRIES):
        try:
            return _git(repo_dir, "-c", "http.lowSpeedLimit=1000",
                        "-c", "http.lowSpeedTime=60", *args)  # fmt: skip
        except subprocess.CalledProcessError as e:
            err = (e.stderr or "").strip()
            last = err.splitlines()[-1] if err else f"exit {e.returncode}"
            if lock := next(
                (ln for ln in err.splitlines() if "file exists" in ln.lower()), None
            ):
                raise RuntimeError(
                    f"{repo_dir}: stale lock from an interrupted git, delete the "
                    f"*.lock it names if no other git runs there: {lock}"
                ) from e
            if attempt == NET_TRIES - 1 or any(g in err.lower() for g in _GONE):
                raise
            logger.warning("git %s: %s: retrying in %ds", args[0], last, 2**attempt)
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


REMOTE_ROOT: str = "https://github.com"
# Every branch head and every PR head (fork PRs included), so PR runs' heads are
# reachable.
_ALL_REFS: tuple[str, ...] = (
    "+refs/heads/*:refs/heads/*",
    "+refs/pull/*/head:refs/pull/*/head",
)


def clone(repo: str, branch: str | None) -> pathlib.Path:
    """
    Full bare clone into `data/raw/<owner>__<name>/repo.git`, or a fetch if it
    exists. Not blobless: `--numstat` reads every file's content.

    Args:
        repo (str): "owner/name".
        branch (str | None): branch to clone; None = every branch head and
            `refs/pull/*/head`. On an existing single-branch clone, None adds the
            missing refs.

    Returns:
        pathlib.Path: the bare repository.
    """
    dest = repo_root(repo) / "repo.git"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        one = ["--single-branch", "-b", branch] if branch else []
        _git_net(dest.parent, "clone", "--bare", *one, f"{REMOTE_ROOT}/{repo}", dest.name)
        if branch:
            return dest
    if branch:
        _git_net(dest, "fetch", "origin", f"+{branch}:{branch}")
    else:
        _git_net(dest, "fetch", "origin", *_ALL_REFS, "--prune")
    return dest


def default_branch(repo_dir: pathlib.Path) -> str:
    """The clone's HEAD branch, i.e. the remote's default branch."""
    return _git(repo_dir, "symbolic-ref", "--short", "HEAD").strip()


@functools.cache
def merge_base(repo_dir: pathlib.Path, a: str, b: str) -> str | None:
    """Best common ancestor of two revisions; memoised. None if a SHA is missing."""
    try:
        return _git(repo_dir, "merge-base", a, b).strip() or None
    except subprocess.CalledProcessError:
        return None


def first_parent_on(
    repo_dir: pathlib.Path, sha: str, line: Container[str]
) -> str | None:
    """
    The first commit of `line` on `sha`'s first-parent chain: where the branch that
    holds `sha` (a stacked PR's base, an earlier merge-queue entry, a release
    branch merged back) forked from the walked line. Always an ancestor of `sha`.
    None if the chain never reaches the line or `sha` is missing.
    """
    with subprocess.Popen(
        ["git", "-C", str(repo_dir), "rev-list", "--first-parent", sha],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ) as p:
        hit = next((c for c in map(str.strip, p.stdout or ()) if c in line), None)
        p.kill()  # stop the walk at the hit
    return hit


FETCH_BATCH: int = 200
_SHA = re.compile(r"^[0-9a-f]{40}$")


def _missing(repo_dir: pathlib.Path, shas: list[str]) -> list[str]:
    out = _git(repo_dir, "cat-file", "--batch-check", stdin="\n".join(shas) + "\n")
    return [line.split()[0] for line in out.splitlines() if line.endswith(" missing")]


def fetch_heads(repo_dir: pathlib.Path, shas: list[str]) -> int:
    """
    Fetches by SHA the commits missing from the clone. A clone holds only each
    branch's and PR's current tip, but GitHub still serves replaced heads by SHA.
    Each lands in `refs/ci/<sha>`, so `gc` never prunes it.

    Args:
        repo_dir (pathlib.Path): the clone.
        shas (list[str]): full SHAs; anything else is ignored.

    Returns:
        int: how many are still missing after the fetch.
    """
    shas = sorted({s for s in shas if isinstance(s, str) and _SHA.match(s)})
    missing = _missing(repo_dir, shas) if shas else []
    for i in range(0, len(missing), FETCH_BATCH):
        batch = missing[i : i + FETCH_BATCH]
        try:
            _git_net(repo_dir, "fetch", "-q", "--no-tags", "origin",
                 *(f"{h}:refs/ci/{h}" for h in batch))  # fmt: skip
        except subprocess.CalledProcessError:
            # one unknown SHA fails the whole batch: retry the rest one by one
            for h in batch:
                try:
                    _git_net(
                        repo_dir,
                        "fetch",
                        "-q",
                        "--no-tags",
                        "origin",
                        f"{h}:refs/ci/{h}",
                    )
                except subprocess.CalledProcessError:
                    pass
    left = _missing(repo_dir, missing) if missing else []
    if missing:
        logger.info(
            "fetched %d of %d missing heads", len(missing) - len(left), len(missing)
        )
    return len(left)


@functools.cache
def commit_info(
    repo_dir: pathlib.Path, sha: str
) -> tuple[list[str], int, str, str | None] | None:
    """
    Parents, commit time, subject and reverted SHA of one commit; memoised. For
    heads off the walked branch, which have no `commit_log` row. None if missing.
    """
    try:
        out = _git(repo_dir, "show", "-s", f"--format=%P{_FLD}%ct{_FLD}%s{_FLD}%b", sha)
    except subprocess.CalledProcessError:
        return None
    parents, ct, subject, body = out.split(_FLD, 3)
    m = _REVERTS.search(body)
    return parents.split(), int(ct), subject, m and m[1]


@functools.cache
def classify(path: str) -> FileKind:
    """
    Kind of a changed file by generic rules, the same for every repository.

    Args:
        path (str): repository-relative path.

    Returns:
        FileKind: `ci`, `test`, `doc`, `config`, or `src`, checked in that order.
    """
    p = path.lower()
    *dirs, name = p.split("/")
    is_requirements = name.startswith("requirements") and name.endswith(".txt")
    if p.startswith((".github/workflows/", ".github/actions/")):
        return "ci"
    if _TEST_DIRS.intersection(dirs) or _TEST_NAME.search(name):
        return "test"
    if (name.endswith(_DOC_EXT) and not is_requirements) or "docs" in dirs:
        return "doc"
    if (
        name in _MANIFESTS
        or is_requirements
        or name.endswith(_CONFIG_EXT)
        or name.startswith(_CONFIG_PREFIX)
        or any(part.startswith(".") for part in (*dirs, name))
    ):
        return "config"
    return "src"


def _new_path(path: str) -> str:
    """The new path of a numstat rename (`old => new`, `dir/{old => new}/f`)."""
    if "{" in path:
        return re.sub(r"//+", "/", _RENAME.sub(r"\2", path)).lstrip("/")
    return path.split(" => ")[-1]


def _count(s: str) -> float:
    return np.nan if s == "-" else float(s)  # "-": binary file


def _numstat(block: str) -> list[tuple[str, float, float]]:
    """`(path, added, deleted)` per numstat line."""
    rows = [line.split("\t", 2) for line in block.splitlines() if line.strip()]
    return [(_new_path(path), _count(a), _count(d)) for a, d, path in rows]


def commit_log(repo_dir: pathlib.Path, branch: str) -> pd.DataFrame:
    """
    One `git log --numstat` pass over the branch's first-parent line, so earlier on
    the branch is a prefix of the log and a merge commit carries its whole change.

    Args:
        repo_dir (pathlib.Path): the clone.
        branch (str): branch to walk.

    Returns:
        pd.DataFrame: one row per (commit, file), oldest commit first (`LOG_COLS`).
    """
    fmt = _REC + _FLD.join(["%H", "%P", "%ae", "%at", "%ct", "%s", "%b", ""])
    out = _git(
        repo_dir, "log", "--first-parent", "--numstat", f"--format={fmt}", branch
    )
    rows = []
    for rec in out.split(_REC)[1:]:
        sha, parents, email, at, ct, subject, body, numstat = rec.split(_FLD)
        m = _REVERTS.search(body)
        head = (sha, parents.split(), email, int(at), int(ct), subject, m and m[1])
        files = _numstat(numstat) or [(None, np.nan, np.nan)]
        rows += [(*head, p, p and classify(p), a, d) for p, a, d in files]
    return pd.DataFrame(rows[::-1], columns=LOG_COLS)  # git prints newest first


@functools.cache
def diff_numstat(repo_dir: pathlib.Path, a: str, b: str) -> pd.DataFrame | None:
    """
    Files changed between two SHAs; memoised, as many runs share a reference SHA.

    Args:
        repo_dir (pathlib.Path): the clone.
        a (str): older SHA.
        b (str): newer SHA.

    Returns:
        pd.DataFrame | None: `path`, `kind`, `added`, `deleted`; None if a SHA is
            missing (e.g. force-pushed away).
    """
    try:
        rows = _numstat(_git(repo_dir, "diff", "--numstat", a, b))
    except subprocess.CalledProcessError:
        return None
    return pd.DataFrame(
        [(p, classify(p), x, y) for p, x, y in rows],
        columns=["path", "kind", "added", "deleted"],
    )


@functools.cache
def rev_count(repo_dir: pathlib.Path, a: str, b: str) -> int | None:
    """Number of commits in `a..b`, None if a SHA is missing."""
    try:
        return int(_git(repo_dir, "rev-list", "--count", f"{a}..{b}"))
    except subprocess.CalledProcessError:
        return None


def tags(repo_dir: pathlib.Path) -> pd.DataFrame:
    """
    Tags with the commit they point to; annotated tags are peeled to their commit.

    Returns:
        pd.DataFrame: `tag`, `sha`, `commit_time` (unix seconds).
    """
    fields = ["refname:short", "*objectname", "objectname", "*committerdate:unix",
              "committerdate:unix"]  # fmt: skip
    fmt = _FLD.join(f"%({f})" for f in fields)
    rows = []
    for line in _git(
        repo_dir, "for-each-ref", "refs/tags", f"--format={fmt}"
    ).splitlines():
        tag, peeled, obj, peeled_t, obj_t = line.split(_FLD)
        rows.append((tag, peeled or obj, int(peeled_t or obj_t)))
    return pd.DataFrame(rows, columns=["tag", "sha", "commit_time"])
