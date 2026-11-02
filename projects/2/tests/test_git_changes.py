import math
import subprocess

import pytest

from acise_p2_cidt.collect import git_changes as gc


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def commit(repo, msg, files):
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    git(r, "config", "user.email", "a@example.com")
    git(r, "config", "user.name", "A")
    shas = [commit(r, "c1", {"a.py": "1\n2\n", "tests/test_a.py": "x\n"})]
    shas.append(commit(r, "c2", {"a.py": "1\n3\n", "bin.dat": b"\x00\x01\x02"}))
    git(r, "checkout", "-q", "-b", "feature")
    commit(r, "f1", {"b.py": "1\n2\n3\n"})
    git(r, "checkout", "-q", "main")
    shas.append(commit(r, "c3", {"README.md": "hi\n"}))
    git(r, "merge", "-q", "--no-ff", "feature", "-m", "merge feature")
    shas.append(git(r, "rev-parse", "HEAD"))
    return r, shas


def test_commit_log(repo):
    r, (c1, c2, c3, merge) = repo
    log = gc.commit_log(r, "main")
    assert log.drop_duplicates("sha")["sha"].tolist() == [
        c1,
        c2,
        c3,
        merge,
    ]  # oldest first
    row = log[(log.sha == c1) & (log.path == "a.py")].iloc[0]
    assert (row.added, row.deleted) == (2, 0)
    assert log[(log.sha == c1) & (log.path == "tests/test_a.py")].kind.item() == "test"
    a = log[(log.sha == c2) & (log.path == "a.py")].iloc[0]
    assert (a.added, a.deleted) == (1, 1)  # "2" -> "3"
    assert math.isnan(log[(log.sha == c2) & (log.path == "bin.dat")].added.item())
    m = log[log.sha == merge]
    assert len(m.parents.iloc[0]) == 2
    assert m.path.tolist() == ["b.py"]  # full change against the first parent
    assert m.added.item() == 3


def test_diff_and_rev_count(repo):
    r, (c1, c2, c3, merge) = repo
    d = gc.diff_numstat(r, c1, c2).set_index("path")
    assert (d.loc["a.py", "added"], d.loc["a.py", "deleted"]) == (1, 1)
    assert math.isnan(d.loc["bin.dat", "added"])
    assert gc.rev_count(r, c1, merge) == 4  # c2, c3, f1, merge
    assert gc.rev_count(r, c2, c3) == 1


def test_missing_sha_returns_none(repo):
    r, (c1, *_) = repo
    gone = "f" * 40
    assert gc.diff_numstat(r, c1, gone) is None
    assert gc.rev_count(r, gone, c1) is None


def test_revert_link(repo):
    r, (*_, merge) = repo
    git(r, "revert", "--no-edit", "HEAD~1")  # reverts c3 (README)
    log = gc.commit_log(r, "main")
    last = log.iloc[-1]
    assert last.subject.startswith('Revert "c3"')
    assert merge != last.sha and len(last.reverts) == 40


@pytest.mark.parametrize(
    "path, kind",
    [
        (".github/workflows/ci.yml", "ci"),  # rule 1 before config's .yml
        (".github/actions/setup/action.yml", "ci"),
        ("tests/utils.py", "test"),  # test directory
        ("src/test_utils.py", "test"),  # test_*.py name
        ("pkg/__tests__/a.ts", "test"),
        ("pkg/a.spec.ts", "test"),
        ("cmd/x_test.go", "test"),
        ("README.md", "doc"),
        ("README.txt", "doc"),
        ("docs/intro.html", "doc"),  # under docs/
        ("requirements-dev.txt", "config"),  # not the generic .txt doc rule
        ("packages/cli/package.json", "config"),
        ("pnpm-lock.yaml", "config"),
        ("Dockerfile.ci", "config"),
        ("tsconfig.build.json", "config"),
        (".eslintrc.js", "config"),  # dotfile
        ("packages/cli/src/index.ts", "src"),
        ("lib/attest.py", "src"),  # "test" inside a word is not a test file
    ],
)
def test_classify(path, kind):
    assert gc.classify(path) == kind


@pytest.mark.parametrize(
    "raw, new",
    [
        ("a.py", "a.py"),
        ("old.py => new.py", "new.py"),
        ("src/{old => new}/f.ts", "src/new/f.ts"),
        ("src/{ => sub}/f.ts", "src/sub/f.ts"),  # moved into a new directory
    ],
)
def test_rename_keeps_new_path(raw, new):
    assert gc._new_path(raw) == new
