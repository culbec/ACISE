# `collect/git_changes.py`

Status: todo (step 2). Spec: [implementation-backbone.md](../implementation-backbone.md). Tests: `projects/2/tests/test_git_changes.py`.

## Purpose

Link each CI run's `head_sha` to the local clone and extract change and file-history data, with no API calls.

## API

```python
FileKind = Literal["src", "test", "config", "doc", "ci"]

clone(repo: str, branch: str) -> pathlib.Path      # data/raw/<owner>__<name>/repo.git; fetch if it exists
classify(path: str) -> FileKind                    # generic rules, same for every repository
commit_log(repo_dir: pathlib.Path, branch: str) -> pd.DataFrame
    # one row per (commit, file): sha, parents, author_email, author_time, commit_time,
    # subject, path, kind, added, deleted (binary files: NaN)
diff_numstat(repo_dir, a: str, b: str) -> pd.DataFrame   # path, kind, added, deleted between two SHAs
rev_count(repo_dir, a: str, b: str) -> int               # commits in a..b
tags(repo_dir) -> pd.DataFrame                           # tag, sha, commit_time
```

## Design decisions

- **Full single-branch clone** (`git clone --single-branch --branch <branch>`). `--numstat` needs file contents; a blobless clone would fetch them one by one, thousands of times. The size is logged after cloning.
- **One `git log` pass, not one call per run.** `git log --numstat --format=%x00%H%x00%P%x00%ae%x00%at%x00%ct%x00%s <branch>` is parsed once into `commit_log`. File history features (age, prior changes, distinct authors, time since last change) are then computed in memory by `data/build.py`. Per-file `git log` calls would cost one process per file per run.
- **Diffs between runs** (`diff_numstat`, `rev_count`) are per-run calls, because they depend on which earlier run is the reference. About 2 calls × ~4k runs, each a few ms. Results are memoised on `(a, b)`.
- **Missing SHA.** A run's `head_sha` may be gone from the branch (force-push). `diff_numstat` / `rev_count` return `None`, and the build marks those features NaN; the count is logged.
- **Renames** (`old => new` in numstat) count under the new path.
- **`classify` rules**, checked in this order, on the lower-cased path:
  1. `ci`: under `.github/workflows/` or `.github/actions/`.
  2. `test`: a path part is `test`, `tests`, `__tests__`, `spec`, `e2e`, or the file name matches `*.test.*`, `*.spec.*`, `test_*.py`, `*_test.go`.
  3. `doc`: `.md`, `.rst`, `.txt`, `.adoc`, or under `docs/`.
  4. `config`: lockfiles and manifests (`package.json`, `pnpm-lock.yaml`, `package-lock.json`, `yarn.lock`, `pyproject.toml`, `uv.lock`, `requirements*.txt`, `go.mod`, `go.sum`, `Cargo.toml`, `Cargo.lock`, `pom.xml`, `build.gradle*`), plus `.json`, `.yml`, `.yaml`, `.toml`, `.ini`, `.cfg`, `.env*`, `Dockerfile*`, `tsconfig*`, dotfiles at any level.
  5. `src`: everything else.
  The rules are ecosystem-generic, not repository-specific: they hold for any repository (portability rule). `requirements*.txt` is matched by rule 4 before the generic `.txt` doc rule, so rule 3 skips it explicitly.
- All git calls go through one `_git(repo_dir, *args) -> str` helper (`subprocess.run`, `check=True`, `text=True`, no shell). Arguments are passed as a list, never a formatted string.

## Verification

Tests build a tiny repository in `tmp_path` with `git init` and scripted commits:
- `commit_log`: hand-counted adds/dels per file, a merge commit with two parents, a binary file with NaN.
- `diff_numstat` and `rev_count` across three commits; a missing SHA returns `None`.
- `classify`: parametrised table, one row per rule, including the `requirements-dev.txt` vs `README.txt` order case and `src/test_utils.py` vs `tests/utils.py`.
