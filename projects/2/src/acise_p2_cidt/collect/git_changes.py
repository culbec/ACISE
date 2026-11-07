import functools
import pathlib
import re
import subprocess
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


def _git(repo_dir: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_dir), "-c", "core.quotePath=false", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


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
        _git(dest.parent, "clone", "--bare", *one, f"{REMOTE_ROOT}/{repo}", dest.name)
        if branch:
            return dest
    if branch:
        _git(dest, "fetch", "origin", f"+{branch}:{branch}")
    else:
        _git(dest, "fetch", "origin", *_ALL_REFS, "--prune")
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
