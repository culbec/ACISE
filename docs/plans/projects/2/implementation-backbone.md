# Project 2 — Implementation backbone

Companion to [dt-ci-failure-prediction.md](dt-ci-failure-prediction.md). The main plan defines the experiment; this spec defines the code modules, their order, and their boundaries. Each module has its own spec under [modules/](modules/).

Package root: `projects/2/src/acise_p2_cidt/`. Same tooling as project 1: `uv`, Python ≥ 3.14, `pytest`, Plotly with Kaleido.
Dependencies: `numpy`, `pandas`, `scipy`, `plotly`, `kaleido`, `requests`, `rich`, `jupyter`, `skops` (model persistence). `scikit-learn` is used only for the reference check in tests and for `SimpleImputer` / `TimeSeriesSplit` / `sklearn.metrics` in the ported utilities, never as the model.

## Modules and order

| # | Module | Spec | Status | Responsibility |
|---|---|---|---|---|
| 0 | `utils/logging/log.py`, `viz/plots.py`, `stats/tests.py`, `metrics/metrics.py`, `analysis/features.py`, `training/cv.py` | [p1-ports](modules/p1-ports.md) | done | Copied from project 1 and generalised (D2: separate copies, no imports from `acise_p1_cigp`). |
| 1 | `collect/gh_runs.py`, `collect/http.py` | [collect-gh-runs](modules/collect-gh-runs.md) | done | GitHub API collection of one workflow: runs per date window, first attempts, jobs of failed first attempts. Resumable on-disk cache. CLI `collect-runs`. |
| 2 | `collect/git_changes.py` | [collect-git-changes](modules/collect-git-changes.md) | done | Local clone, one-pass `git log` table, diffs between run SHAs, generic file classifier. No API calls. |
| 3 | `data/build.py`, `data/history.py` | [data-build](modules/data-build.md) | done | Joins API and git data. One chronological pass computes all features under the completed-before rule. Writes `<repo>.csv` and `schema.json`. CLI `build-dataset`. |
| 4 | `data/load.py` | [data-load](modules/data-load.md) | done | Loads a built table into the project 1 `Dataset` shape: T1/T2, dev cut, drop non-features. |
| 5 | `tree/cart.py`, `tree/prune.py` | [tree-cart](modules/tree-cart.md) | done | From-scratch CART following Breiman et al. (1984). `prune.py` holds the cost-complexity helpers (split out to keep `cart.py` under 500 lines). |
| 6 | `tree/tune.py` | [tree-tune](modules/tree-tune.md) | done | Grid search, pruning-alpha and threshold choice on validation; `fit_predict` for `training/cv.py`. |
| 7 | `tree/forest.py` | — | done (D3) | Random forest (Breiman, 2001) over `tree/cart.py`: bootstrap as sample weights, `max_features` columns drawn per node (`CART(max_features=, random_state=)`), soft voting, unpruned trees. Grid (depth × leaf size × class weight, 100 trees, `sqrt` columns) and threshold on validation; `fit_predict` for `run_cv`. Tests: `tests/test_forest.py`. |
| 8 | `validate/live.py` | [validate-live](modules/validate-live.md) | done | `train-final` (skops dump of CART or forest), `retest` on runs after `C`, `predict-run` with an explained decision path. |

Order: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8. Module 0 can start any time; modules 1 and 2 are independent of each other.

## Notebooks

Jupyter notebooks present the task and the implementation: what problem is solved, how the data is collected and built, how the tree is trained and evaluated, and what the results are. The package modules hold the logic. Notebooks only call them, show results, and explain them in markdown.

| Notebook | Content |
|---|---|
| `collection.ipynb` | Problem statement, collection strategy, coverage over time, run counts, label balance, flaky share |
| `eda.ipynb` | Feature distributions by label, correlations, failure rate over time, failed-job categories |
| `feature_selection.ipynb` | Selection on the dev segment, stability, drift |
| `tree.ipynb` | CART walkthrough, CV, tuning, baselines, statistical tests, tree and importance figures |
| `validation.ipynb` | Saved model, new-data re-test, `predict-run` case studies, user feedback |

**Who does what.** Claude does **not** create or edit `.ipynb` files. For each notebook, Claude provides the cell contents in order, each one marked as code or markdown. The user creates the notebook from them in `projects/2/notebook/`.

## Coding style

Taken from the project 1 modules; new code must read as if written by the same author.

**Layout and formatting**
- `ruff format` defaults: 88 columns, double quotes, trailing commas in wrapped calls. Imports in three groups (stdlib, third-party, local) with absolute local imports (`from acise_p2_cidt.metrics.metrics import scores`).
- Modern typing everywhere: `list[str]`, `dict[str, float]`, `X | None`, `Literal[...]`, `collections.abc.Callable`. Every public function has a typed signature.
- Module order: imports → module constants → `logger = get_logger(__name__)` → data classes → private helpers → public functions → `main()` for CLIs.
- Long modules are divided by comment banners (`# =============================`) between groups of related functions.
- Files stay under 500 lines.
- **Formatting and lint are required.** `uv run ruff format src tests` and `uv run ruff check src tests` must both pass with no errors before any module is handed over. Do not leave `noqa` suppressions where a small rewrite would fix the finding.
- Inline a helper used in only one place unless it has its own test or a clear name that the call site needs.

**Naming and constants**
- Constants are `UPPER_CASE` with a type annotation, and a comment saying *why* the value is what it is (`MIN_TEST_FAILURES = 10  # below this a test fold gives unstable recall/MCC`).
- Paths are built from the file location, with a comment showing the mapping: `# projects/2/src/acise_p2_cidt/viz/plots.py -> projects/2/results/figures`, then `pathlib.Path(__file__).resolve().parents[3] / "results" / "figures"`.
- Private helpers start with `_`. Short local names are fine in numeric code (`X`, `y`, `c`, `tp`), as in `metrics.py`.

**Data structures**
- Records are `@dataclass(frozen=True)` or `NamedTuple`, with field comments for anything not obvious (`dev_mask: np.ndarray  # True on dev rows`).
- Cross-module coupling is by duck typing where project 1 did it (baselines read `split.X_test`, `split.raw_test`), so `metrics` does not import `training`.

**Docstrings and comments**
- Google style, opened on the line after `"""`: one summary sentence, an optional paragraph on the reasoning, then `Args:` with `name (type): meaning` and `Returns:` with `type: meaning`. Trivial helpers get a one-line docstring or none.
- Comments explain decisions and pitfalls, not mechanics (`# keep_empty_features: an all-NaN column on fit must not shift the columns.`).
- Type-checker suppressions carry the reason: `# pyright: ignore[reportArgumentType]` after a comment saying why.
- No `ponytail:` or other tool-specific markers, and no comments that restate the code. Keep a comment only when it records a decision or a pitfall.

**Errors, logging, warnings**
- Invalid input raises `ValueError` with the received value: `raise ValueError(f"target must be 'T1' or 'T2', got {target!r}")`.
- Conditions that are suspicious but allowed use `warnings.warn(..., stacklevel=2)`.
- Logging goes through `get_logger(__name__)` (rich handler); progress bars use `rich.progress`. No `print` in library code; CLIs print through the rich console `log`.
- CLIs use `argparse` with a `main()` registered under `[project.scripts]` in `pyproject.toml`.

**Tests**
- Plain `pytest` functions in `projects/2/tests/test_<module>.py`. Module-level constants for small fixtures, `pytest.mark.parametrize` with an inline comment per case, `pytest.approx` for floats, hand-computed expected values with the arithmetic in a comment.
- No live network in tests: API tests use recorded JSON fixtures; git tests build a tiny repository in `tmp_path`.
- Run one file at a time under the memory cap: `systemd-run --user --scope -p MemoryMax=... uv run pytest tests/test_<module>.py`.

## Rules

- **Leakage boundary.** `data/build.py` computes history features only from runs **completed before** the current run's `created_at` (completed = the first attempt's `updated_at`), and git features only from commits reachable from the current `head_sha`. `data/load.py` is stateless. Anything fitted on data (imputation, selection, pruning alpha, threshold) is fitted in the dev segment or per fold.
- **Raw data never in git.** `projects/**/data/raw` is already in the root `.gitignore`; add `projects/2/data/processed` when module 3 lands. Store `schema.json` and the collection manifest (date range, query list, counts per window) under `results/` so the run can be repeated.
- **Secrets.** Never write the token to disk or log it. The login-hashing salt lives in `data/raw/` (ignored), never in `results/`.
- **Tests.** One focused test file per module, listed in each module spec.
- **One module at a time**, implemented on the user's request.
