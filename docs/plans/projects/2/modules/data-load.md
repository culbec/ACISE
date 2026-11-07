# `data/load.py`

Status: done (step 4). Spec: [implementation-backbone.md](../implementation-backbone.md). Tests: `projects/2/tests/test_load.py`.

> **Superseded in part (2026-11-07)** by the [multi-repo backbone](../../../../specs/project-2/multi-repo-analysis/implementation-backbone.md): §3 layout (done, phase 1: `load(slug)`, `table_path(slug)`, `schema_path(slug)`, `available_targets()`, `meta["dataset"]`), §6.5 T2 rule P-D4 (phase 4), §7 pooled load (phase 6). Where the two differ, the multi-repo backbone wins.

## Purpose

Load one built table into the `Dataset` shape the ported modules expect. Stateless only: nothing here is fitted on data.

## API

```python
Target = Literal["T1", "T2"]

ORDER_COL = "created_at"
LABEL_COL = "failed"
PREV_LABEL_COL = "prev_failed"
DEV_FRACTION = 0.2

@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame        # feature columns only, chronological, RangeIndex
    y: pd.Series           # failed = 1, passed = 0
    dev_mask: np.ndarray   # True on the first DEV_FRACTION of rows
    meta: dict             # schema.json content + target, n_rows, n_failed, n_dev
    order: pd.Series       # created_at per row, for plots and the retest cut

available_repos() -> list[str]
load(repo: str, target: Target, until: datetime | None = None) -> Dataset
```

## Design decisions

- Same `Dataset` fields as project 1, except `groups` (one repository per dataset) is replaced by `order`, which the plots and `validate/live.py` need.
- The column names are fixed by `data/build.py` for every repository, so they are module constants here. `schema.json` gives `feature_cols`, `nominal`, and provenance.
- **T1**: all rows. **T2**: rows with `prev_failed == 0`; then `prev_failed` is constant and dropped from `X`. Rows with NaN `prev_failed` (the first run) stay in T1 and are dropped from T2.
- `until` keeps rows with `created_at <= until`. `train-final` uses it to pin the data to `C`; `None` keeps all.
- The dev cut is taken **after** the T2 filter, as in project 1.
- `X` holds `schema["feature_cols"]` only. Identifiers, the order column, the label, and diagnostics never reach `X`; a test checks this.
- Missing table → `FileNotFoundError` naming the `build-dataset` command to run.

## Verification

- A toy CSV and schema in `tmp_path`: T1 and T2 row counts, `prev_failed` dropped on T2, dev mask is a prefix of the right length, no diagnostic or id column in `X`.
- `until` cuts at the right row.
- Unknown target → `ValueError` with the value.
